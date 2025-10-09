// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Main module for univariate KZG commitment scheme
use crate::{
    pcs::{
        poly::GeneralDensePolynomial, univariate_kzg::ptau_digests::expected_sha256_for_label,
        PCSError, PolynomialCommitmentScheme, StructuredReferenceString, UnivariatePCS,
    },
    toeplitz::ToeplitzMatrix,
};
use ark_bn254::Bn254;
use ark_ec::{
    pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, AffineRepr, CurveGroup,
};
use ark_ff::{FftField, Field, PrimeField};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, Polynomial, Radix2EvaluationDomain,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use ark_std::{
    borrow::Borrow,
    end_timer, format, fs,
    hash::Hash,
    io,
    marker::PhantomData,
    ops::Mul,
    path::Path,
    path::PathBuf,
    rand::{CryptoRng, RngCore},
    start_timer,
    string::String,
    string::ToString,
    vec,
    vec::Vec,
    One, UniformRand, Zero,
};
use jf_utils::par_utils::parallelizable_slice_iter;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use srs::{UnivariateProverParam, UnivariateUniversalParams, UnivariateVerifierParam};

use super::Accumulation;
use ark_serialize::{Read, Write};
use log::{error, warn};

pub mod ptau;
pub mod ptau_digests;
pub(crate) mod srs;
/// KZG Polynomial Commitment Scheme on univariate polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnivariateKzgPCS<E: Pairing> {
    #[doc(hidden)]
    phantom: PhantomData<E>,
}

impl<E: Pairing> Default for UnivariateKzgPCS<E> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
/// proof of opening
pub struct UnivariateKzgProof<E: Pairing> {
    /// Evaluation of quotients
    pub proof: E::G1Affine,
}

impl<E: Pairing> Default for UnivariateKzgProof<E> {
    fn default() -> Self {
        Self {
            proof: E::G1Affine::default(),
        }
    }
}

impl<E: Pairing> Hash for UnivariateKzgProof<E> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.proof.hash(state);
    }
}
/// batch proof
pub type UnivariateKzgBatchProof<E> = Vec<UnivariateKzgProof<E>>;

impl<E: Pairing> PolynomialCommitmentScheme for UnivariateKzgPCS<E> {
    /// This is is a fixed, trusted SRS generated during the [Powers-of-Tau ceremony](https://zfnd.org/conclusion-of-the-powers-of-tau-ceremony).
    /// Since the verifier is hard-wired to the same SRS, we exclude it from the Fiat-Shamir transcript.
    /// In Nightfall, the SRS is instantiated with:
    /// const MAX_KZG_DEGREE: usize = 26;
    /// let ptau_file = path.join(format!("bin/ppot_{}.ptau", MAX_KZG_DEGREE));
    /// UnivariateKzgPCS::download_ptau_file_if_needed(MAX_KZG_DEGREE, &ptau_file).unwrap();
    /// UnivariateKzgPCS::universal_setup_bn254(&ptau_file, 1 << MAX_KZG_DEGREE).unwrap();
    type SRS = UnivariateUniversalParams<E>;
    // Polynomial and its associated types
    type Polynomial = DensePolynomial<E::ScalarField>;
    type Point = E::ScalarField;
    type Evaluation = E::ScalarField;
    // Polynomial and its associated types
    type Commitment = E::G1Affine;
    type Proof = UnivariateKzgProof<E>;
    type BatchProof = UnivariateKzgBatchProof<E>;

    /// Trim the universal parameters to specialize the public parameters.
    /// Input `max_degree` for univariate.
    /// `supported_num_vars` must be None or an error is returned.
    fn trim(
        srs: impl Borrow<Self::SRS>,
        supported_degree: usize,
        supported_num_vars: Option<usize>,
    ) -> Result<(UnivariateProverParam<E>, UnivariateVerifierParam<E>), PCSError> {
        if supported_num_vars.is_some() {
            return Err(PCSError::InvalidParameters(
                "univariate should not receive a num_var param".to_string(),
            ));
        }
        srs.borrow().trim(supported_degree)
    }

    /// Generate a commitment for a polynomial
    /// Note that the scheme is not hidding
    fn commit(
        prover_param: impl Borrow<UnivariateProverParam<E>>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let prover_param = prover_param.borrow();
        let commit_time =
            start_timer!(|| format!("Committing to polynomial of degree {} ", poly.degree()));

        if poly.degree() > prover_param.powers_of_g.len() {
            return Err(PCSError::InvalidParameters(format!(
                "poly degree {} is larger than allowed {}",
                poly.degree(),
                prover_param.powers_of_g.len()
            )));
        }

        let (num_leading_zeros, plain_coeffs) = skip_leading_zeros_and_convert_to_bigints(poly);

        let msm_time = start_timer!(|| "MSM to compute commitment to plaintext poly");
        let commitment = E::G1::msm_bigint(
            &prover_param.powers_of_g[num_leading_zeros..],
            plain_coeffs.as_slice(),
        )
        .into_affine();
        end_timer!(msm_time);

        end_timer!(commit_time);
        Ok(commitment)
    }

    /// Generate a commitment for a list of polynomials
    fn batch_commit(
        prover_param: impl Borrow<UnivariateProverParam<E>>,
        polys: &[Self::Polynomial],
    ) -> Result<Vec<Self::Commitment>, PCSError> {
        let prover_param = prover_param.borrow();
        let commit_time = start_timer!(|| format!("batch commit {} polynomials", polys.len()));
        let res = parallelizable_slice_iter(polys)
            .map(|poly| Self::commit(prover_param, poly))
            .collect::<Result<Vec<Self::Commitment>, PCSError>>()?;

        end_timer!(commit_time);
        Ok(res)
    }

    /// On input a polynomial `p` and a point `point`, outputs a proof for the
    /// same.
    fn open(
        prover_param: impl Borrow<UnivariateProverParam<E>>,
        polynomial: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let open_time =
            start_timer!(|| format!("Opening polynomial of degree {}", polynomial.degree()));
        let divisor = Self::Polynomial::from_coefficients_vec(vec![-*point, E::ScalarField::one()]);

        let witness_time = start_timer!(|| "Computing witness polynomial");
        let witness_polynomial = polynomial / &divisor;
        end_timer!(witness_time);

        let (num_leading_zeros, witness_coeffs) =
            skip_leading_zeros_and_convert_to_bigints(&witness_polynomial);

        let proof: E::G1Affine = E::G1::msm_bigint(
            &prover_param.borrow().powers_of_g[num_leading_zeros..],
            &witness_coeffs,
        )
        .into_affine();

        let eval = polynomial.evaluate(point);

        end_timer!(open_time);
        Ok((Self::Proof { proof }, eval))
    }

    /// Input a list of polynomials, and a same number of points,
    /// compute a multi-opening for all the polynomials.
    // This is a naive approach
    // TODO: to implement the more efficient batch opening algorithm
    // (e.g., the appendix C.4 in https://eprint.iacr.org/2020/1536.pdf)
    fn batch_open(
        prover_param: impl Borrow<UnivariateProverParam<E>>,
        _multi_commitment: &[Self::Commitment],
        polynomials: &[Self::Polynomial],
        points: &[Self::Point],
    ) -> Result<(Self::BatchProof, Vec<Self::Evaluation>), PCSError> {
        let open_time = start_timer!(|| format!("batch opening {} polynomials", polynomials.len()));
        if polynomials.len() != points.len() {
            return Err(PCSError::InvalidParameters(format!(
                "poly length {} is different from points length {}",
                polynomials.len(),
                points.len()
            )));
        }
        let mut batch_proof = vec![];
        let mut evals = vec![];
        for (poly, point) in polynomials.iter().zip(points.iter()) {
            let (proof, eval) = Self::open(prover_param.borrow(), poly, point)?;
            batch_proof.push(proof);
            evals.push(eval);
        }

        end_timer!(open_time);
        Ok((batch_proof, evals))
    }
    /// Verifies that `value` is the evaluation at `x` of the polynomial
    /// committed inside `comm`.
    fn verify(
        verifier_param: &UnivariateVerifierParam<E>,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        let check_time = start_timer!(|| "Checking evaluation");
        let pairing_inputs_l: Vec<E::G1Prepared> = vec![
            (verifier_param.g * value - proof.proof * point - commitment.into_group())
                .into_affine()
                .into(),
            proof.proof.into(),
        ];
        let pairing_inputs_r: Vec<E::G2Prepared> =
            vec![verifier_param.h.into(), verifier_param.beta_h.into()];

        let res = E::multi_pairing(pairing_inputs_l, pairing_inputs_r)
            .0
            .is_one();

        end_timer!(check_time, || format!("Result: {res}"));
        Ok(res)
    }

    /// Verifies that `value_i` is the evaluation at `x_i` of the polynomial
    /// `poly_i` committed inside `comm`.
    // This is a naive approach
    // TODO: to implement the more efficient batch verification algorithm
    // (e.g., the appendix C.4 in https://eprint.iacr.org/2020/1536.pdf)
    fn batch_verify<R: RngCore + CryptoRng>(
        verifier_param: &UnivariateVerifierParam<E>,
        multi_commitment: &[Self::Commitment],
        points: &[Self::Point],
        values: &[E::ScalarField],
        batch_proof: &Self::BatchProof,
        rng: &mut R,
    ) -> Result<bool, PCSError> {
        let check_time =
            start_timer!(|| format!("Checking {} evaluation proofs", multi_commitment.len()));

        let mut total_c = <E::G1>::zero();
        let mut total_w = <E::G1>::zero();

        let combination_time = start_timer!(|| "Combining commitments and proofs");
        let mut randomizer = E::ScalarField::one();
        // Instead of multiplying g and gamma_g in each turn, we simply accumulate
        // their coefficients and perform a final multiplication at the end.
        let mut g_multiplier = E::ScalarField::zero();
        for (((c, z), v), proof) in multi_commitment
            .iter()
            .zip(points)
            .zip(values)
            .zip(batch_proof)
        {
            let w = proof.proof;
            let mut temp = w.mul(*z);
            temp += c;
            let c = temp;
            g_multiplier += &(randomizer * v);
            total_c += c * randomizer;
            total_w += w * randomizer;
            // We don't need to sample randomizers from the full field,
            // only from 128-bit strings.
            randomizer = u128::rand(rng).into();
        }
        total_c -= &verifier_param.g.mul(g_multiplier);
        end_timer!(combination_time);

        let to_affine_time = start_timer!(|| "Converting results to affine for pairing");
        let affine_points = E::G1::normalize_batch(&[-total_w, total_c]);
        let (total_w, total_c) = (affine_points[0], affine_points[1]);
        end_timer!(to_affine_time);

        let pairing_time = start_timer!(|| "Performing product of pairings");
        let result = E::multi_pairing(
            [total_w, total_c],
            [verifier_param.beta_h, verifier_param.h],
        )
        .0
        .is_one();
        end_timer!(pairing_time);
        end_timer!(check_time, || format!("Result: {result}"));
        Ok(result)
    }

    /// Fast computation of batch opening for a single polynomial at multiple
    /// arbitrary points.
    /// Details see Sec 2.1~2.3 of [FK23](https://eprint.iacr.org/2023/033.pdf).
    ///
    /// Only accept `polynomial` with power-of-two degree, no constraint on the
    /// size of `points`
    fn multi_open(
        prover_param: impl Borrow<UnivariateProverParam<E>>,
        polynomial: &Self::Polynomial,
        points: &[Self::Point],
    ) -> Result<(Vec<Self::Proof>, Vec<Self::Evaluation>), PCSError> {
        let h_poly = Self::compute_h_poly_in_fk23(prover_param, &polynomial.coeffs)?;
        let proofs: Vec<_> = h_poly
            .batch_evaluate(points)
            .into_iter()
            .map(|g| UnivariateKzgProof {
                proof: g.into_affine(),
            })
            .collect();

        // Evaluate at all points
        let evals =
            GeneralDensePolynomial::from_coeff_slice(&polynomial.coeffs).batch_evaluate(points);
        Ok((proofs, evals))
    }
}

impl<E: Pairing> Accumulation for UnivariateKzgPCS<E> {}

impl<E: Pairing> UnivariatePCS for UnivariateKzgPCS<E> {
    fn multi_open_rou_proofs(
        prover_param: impl Borrow<<Self::SRS as StructuredReferenceString>::ProverParam>,
        polynomial: &Self::Polynomial,
        num_points: usize,
        domain: &Radix2EvaluationDomain<Self::Evaluation>,
    ) -> Result<Vec<Self::Proof>, PCSError> {
        let mut h_poly = Self::compute_h_poly_in_fk23(prover_param, &polynomial.coeffs)?;
        let proofs: Vec<_> = h_poly
            .batch_evaluate_rou(domain)?
            .into_iter()
            .take(num_points)
            .map(|g| UnivariateKzgProof {
                proof: g.into_affine(),
            })
            .collect();
        Ok(proofs)
    }

    /// Compute the evaluations in [`Self::multi_open_rou()`].
    fn multi_open_rou_evals(
        polynomial: &Self::Polynomial,
        num_points: usize,
        domain: &Radix2EvaluationDomain<Self::Evaluation>,
    ) -> Result<Vec<Self::Evaluation>, PCSError> {
        let evals = GeneralDensePolynomial::from_coeff_slice(&polynomial.coeffs)
            .batch_evaluate_rou(domain)?
            .into_iter()
            .take(num_points)
            .collect();
        Ok(evals)
    }
}

impl<E, F> UnivariateKzgPCS<E>
where
    E: Pairing<ScalarField = F>,
    F: FftField,
{
    // Sec 2.2. of <https://eprint.iacr.org/2023/033>
    fn compute_h_poly_in_fk23(
        prover_param: impl Borrow<UnivariateProverParam<E>>,
        poly_coeffs: &[E::ScalarField],
    ) -> Result<GeneralDensePolynomial<E::G1, F>, PCSError> {
        // First, pad to power_of_two, since Toeplitz mul only works for 2^k
        let mut padded_coeffs: Vec<F> = poly_coeffs.to_vec();
        let padded_degree = (padded_coeffs.len() - 1)
            .checked_next_power_of_two()
            .ok_or_else(|| {
                PCSError::InvalidParameters(ark_std::format!(
                    "Next power of two overflows! Got: {}",
                    (padded_coeffs.len() - 1)
                ))
            })?;
        let padded_len = padded_degree + 1;
        padded_coeffs.resize(padded_len, F::zero());

        // Step 1. compute \vec{h} using fast Toeplitz matrix multiplication
        // 1.1 Toeplitz matrix A (named `poly_coeff_matrix` here)
        let mut toep_col = vec![*padded_coeffs
            .last()
            .ok_or_else(|| PCSError::InvalidParameters("poly degree should >= 1".to_string()))?];
        toep_col.resize(padded_degree, <<E as Pairing>::ScalarField as Field>::ZERO);
        let toep_row = padded_coeffs.iter().skip(1).rev().cloned().collect();
        let poly_coeff_matrix = ToeplitzMatrix::new(toep_col, toep_row)?;

        // 1.2 vector s (named `srs_vec` here)
        let srs_vec: Vec<E::G1> = prover_param
            .borrow()
            .powers_of_g
            .iter()
            .take(padded_degree)
            .rev()
            .cloned()
            .map(|g| g.into_group())
            .collect();

        // 1.3 compute \vec{h}
        let h_vec = poly_coeff_matrix.fast_vec_mul(&srs_vec)?;

        Ok(GeneralDensePolynomial::from_coeff_vec(h_vec))
    }
}

fn skip_leading_zeros_and_convert_to_bigints<F: PrimeField, P: DenseUVPolynomial<F>>(
    p: &P,
) -> (usize, Vec<F::BigInt>) {
    let mut num_leading_zeros = 0;
    while num_leading_zeros < p.coeffs().len() && p.coeffs()[num_leading_zeros].is_zero() {
        num_leading_zeros += 1;
    }
    let coeffs = convert_to_bigints(&p.coeffs()[num_leading_zeros..]);
    (num_leading_zeros, coeffs)
}

fn convert_to_bigints<F: PrimeField>(p: &[F]) -> Vec<F::BigInt> {
    let to_bigint_time = start_timer!(|| "Converting polynomial coeffs to bigints");
    let coeffs = p.iter().map(|s| s.into_bigint()).collect::<Vec<_>>();
    end_timer!(to_bigint_time);
    coeffs
}
use crate::pcs::univariate_kzg::ptau::parse_ptau_file;
use ark_bn254::{g1::Config as Bn254ConfigOne, g2::Config as Bn254ConfigTwo, Fq};

const KZG_CACHE_FORMAT_VERSION: u32 = 1;

#[derive(Debug)]
struct KzgCacheHeader {
    magic: [u8; 8], // b"KZGSRS\0\0"
    version: u32,   // cache format version
    max_degree: u32,
    curve_id: [u8; 8],     // e.g. b"bn254\0\0\0"
    ptau_sha256: [u8; 32], // exact PTAU hash used to derive this SRS
}

impl KzgCacheHeader {
    fn write_to<W: io::Write>(&self, mut w: W) -> io::Result<()> {
        w.write_all(&self.magic)?;
        w.write_all(&self.version.to_le_bytes())?;
        w.write_all(&self.max_degree.to_le_bytes())?;
        w.write_all(&self.curve_id)?;
        w.write_all(&self.ptau_sha256)?;
        Ok(())
    }

    fn read_from<R: io::Read>(mut r: R) -> io::Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        let mut v = [0u8; 4];
        r.read_exact(&mut v)?;
        let version = u32::from_le_bytes(v);
        let mut d = [0u8; 4];
        r.read_exact(&mut d)?;
        let max_degree = u32::from_le_bytes(d);
        let mut curve_id = [0u8; 8];
        r.read_exact(&mut curve_id)?;
        let mut ptau_sha256 = [0u8; 32];
        r.read_exact(&mut ptau_sha256)?;

        if &magic != b"KZGSRS\0\0" {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Bad KZG SRS cache magic",
            ));
        }
        Ok(Self {
            magic,
            version,
            max_degree,
            curve_id,
            ptau_sha256,
        })
    }
}

/// Compute SHA-256 (raw bytes + hex) of a file; we bind cache to the exact PTAU.
fn file_sha256(path: &Path) -> Result<([u8; 32], String), io::Error> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    io::copy(&mut f, &mut h)?;
    let raw = h.finalize();
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&raw);
    Ok((arr, hex::encode(raw)))
}

impl UnivariateKzgPCS<Bn254> {
    /// Cached version that persists/loads the exact `UnivariateUniversalParams<Bn254>`
    /// (i.e., {powers_of_g, h, beta_h}) keyed by:
    ///   - curve = "bn254"
    ///   - max_degree
    ///   - SHA-256 of the PTAU used
    ///   - cache format version
    ///
    /// Flow:
    ///   1) Verify PTAU (you should have done this already before calling).
    ///   2) Try to load cache; validate header; deserialize params.
    ///   3) On miss or validation failure, derive via `universal_setup_bn254`,
    ///      then write cache atomically and return.
    // pub fn universal_setup_bn254_cached(
    //     ptau_file: &Path,
    //     max_degree: usize,
    //     cache_dir: &Path,
    // ) -> Result<UnivariateUniversalParams<Bn254>, PCSError> {
    //     // Ensure cache dir
    //     fs::create_dir_all(cache_dir)
    //         .map_err(|e| PCSError::InvalidParameters(format!("cache dir error: {e}")))?;

    //     // Bind cache identity to the exact PTAU content
    //     let (ptau_hash_bytes, ptau_hash_hex) = file_sha256(ptau_file)
    //         .map_err(|e| PCSError::InvalidParameters(format!("ptau sha256: {e}")))?;

    //     let cache_path = cache_dir.join(format!(
    //         "kzg_srs_bn254_deg{}_ptau_{}.bin",
    //         max_degree,
    //         &ptau_hash_hex[..16] // short prefix for readability
    //     ));

    //     // Try loading from cache
    //     if let Ok(mut f) = fs::File::open(&cache_path) {
    //         if let Ok(header) = KzgCacheHeader::read_from(&mut f) {
    //             if header.version == KZG_CACHE_FORMAT_VERSION
    //                 && header.max_degree as usize == max_degree
    //                 && &header.curve_id == b"bn254\0\0\0"
    //                 && header.ptau_sha256 == ptau_hash_bytes
    //             {
    //                 let mut payload = Vec::new();
    //                 if let Err(e) = f.read_to_end(&mut payload) {
    //                     error!("KZG cache read error (fallback to rebuild): {:?}", e);
    //                 } else {
    //                     match UnivariateUniversalParams::<Bn254>::deserialize_with_mode(
    //                         &*payload,
    //                         Compress::Yes,
    //                         Validate::Yes,
    //                     ) {
    //                         Ok(params) => return Ok(params),
    //                         Err(e) => {
    //                             error!("KZG cache decode error (fallback): {:?}", e);
    //                         },
    //                     }
    //                 }
    //             }
    //         } else {
    //             error!("KZG cache header parse error (fallback).");
    //         }
    //     }

    //     // Miss or invalid cache → derive once from PTAU
    //     let params = Self::universal_setup_bn254(&ptau_file.to_path_buf(), max_degree)?;

    //     // Serialize and write atomically
    //     let tmp = cache_path.with_extension("tmp");
    //     {
    //         let mut w = fs::File::create(&tmp)
    //             .map_err(|e| PCSError::InvalidParameters(format!("cache create: {e}")))?;
    //         let header = KzgCacheHeader {
    //             magic: *b"KZGSRS\0\0",
    //             version: KZG_CACHE_FORMAT_VERSION,
    //             max_degree: max_degree as u32,
    //             curve_id: *b"bn254\0\0\0",
    //             ptau_sha256: ptau_hash_bytes,
    //         };
    //         header
    //             .write_to(&mut w)
    //             .map_err(|e| PCSError::InvalidParameters(format!("cache header write: {e}")))?;

    //         let mut buf = Vec::new();
    //         params
    //             .serialize_with_mode(&mut buf, Compress::Yes)
    //             .map_err(|e| PCSError::InvalidParameters(format!("params serialize: {e}")))?;
    //         w.write_all(&buf)
    //             .map_err(|e| PCSError::InvalidParameters(format!("cache payload write: {e}")))?;
    //         w.flush()
    //             .map_err(|e| PCSError::InvalidParameters(format!("cache flush: {e}")))?;
    //     }
    //     fs::rename(&tmp, &cache_path)
    //         .map_err(|e| PCSError::InvalidParameters(format!("cache rename: {e}")))?;

    //     Ok(params)
    // }

    /// Specialized implementation of universal_setup for BN254
    pub fn universal_setup_bn254(
        ptau_file: &PathBuf,
        max_degree: usize,
    ) -> Result<UnivariateUniversalParams<Bn254>, PCSError> {
        let (powers_of_g, h) =
            parse_ptau_file::<Fq, Bn254ConfigOne, Bn254ConfigTwo>(ptau_file, max_degree, 2)
                .map_err(|e| {
                    error!("Error parsing PTAU file: {:?}", e);
                    PCSError::InvalidSRS
                })?;

        let beta_h = h[1];
        Ok(UnivariateUniversalParams::<Bn254> {
            powers_of_g,
            h: h[0],
            beta_h,
        })
    }

    /// Download a PPoT (pot28_0080) PTAU file for BN254 and verify integrity
    /// against the canonical digest table in `ptau_digests.rs`.
    /// Nightfall supports only max_degree <= 26 (2^26).
    pub fn download_ptau_file_if_needed(
        max_degree: usize,
        ptau_file: &PathBuf,
    ) -> Result<(), io::Error> {
        // Map degree -> server label
        let degree_label = match max_degree {
            0 => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "max_degree must be >= 1; got 0",
                ))
            },
            1..=26 => format!("{:02}", max_degree),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "max_degree {} exceeds supported range (Nightfall supports only up to 26)",
                        max_degree
                    ),
                ))
            },
        };

        // Lookup canonical expected hash from the embedded table
        let expected_hex = expected_sha256_for_label(&degree_label).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("No embedded SHA-256 for label {}", degree_label),
            )
        })?;

        // If a file already exists but the checksum is wrong, delete it and continue as if missing.
        if fs::metadata(ptau_file).is_ok() {
            match Self::verify_ptau_checksum_against_label(ptau_file.as_path(), expected_hex) {
                Ok(()) => return Ok(()), // already correct, short-circuit
                Err(e) => {
                    warn!(
                        "PTAU at {} failed checksum ({}). Deleting and re-downloading...",
                        ptau_file.display(),
                        e
                    );
                    let _ = fs::remove_file(ptau_file);
                    // fall through to download
                },
            }
        }

        // Remote URL (PSE Trusted Setup bucket)
        let url = format!(
        "https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_{}.ptau",
        degree_label
    );

        // Prepare temp path for atomic move
        let parent = ptau_file.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "Invalid ptau_file path (no parent)")
        })?;
        fs::create_dir_all(parent)?;
        let tmp_path = parent.join(format!(
            ".{}.download",
            ptau_file.file_name().unwrap().to_string_lossy()
        ));

        // Download to temp
        let mut response = reqwest::blocking::get(&url)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("HTTP error: {}", e)))?;
        {
            let mut tmp_file = fs::File::create(&tmp_path)?;
            io::copy(&mut response, &mut tmp_file)?;
        }

        // Verify temp file against embedded digest; delete on mismatch
        if let Err(e) = Self::verify_ptau_checksum_against_label(tmp_path.as_path(), expected_hex) {
            let _ = fs::remove_file(&tmp_path);
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "PTAU checksum mismatch for freshly downloaded file (label {}): {}",
                    degree_label, e
                ),
            ));
        }

        // Atomic move into place (verified)
        fs::rename(tmp_path, ptau_file)?;
        Ok(())
    }

    /// Compute SHA-256 of `path` and compare with the expected hex digest.
    fn verify_ptau_checksum_against_label(
        path: &Path,
        expected_hex: &str,
    ) -> Result<(), io::Error> {
        let mut file = fs::File::open(path)?;
        let mut hasher = Sha256::new();
        io::copy(&mut file, &mut hasher)?;
        let actual = hex::encode(hasher.finalize());

        if actual != expected_hex {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "checksum mismatch (expected {}, got {}) at {}",
                    expected_hex,
                    actual,
                    path.display()
                ),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::StructuredReferenceString;
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    use ark_poly::{univariate::DensePolynomial, EvaluationDomain};
    use ark_std::{
        fs::{self, File},
        io::{Read as IoRead, Write as IoWrite},
        rand::Rng,
        UniformRand,
    };
    use jf_utils::test_rng;
    use std::time::Duration;

    fn end_to_end_test_template<E>() -> Result<(), PCSError>
    where
        E: Pairing,
    {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let mut degree = 0;
            while degree <= 1 {
                degree = usize::rand(rng) % 20;
            }
            let pp = UnivariateKzgPCS::<E>::gen_srs_for_testing(rng, degree)?;
            let (ck, vk) = pp.trim(degree)?;
            let p = <DensePolynomial<E::ScalarField> as DenseUVPolynomial<E::ScalarField>>::rand(
                degree, rng,
            );
            let comm = UnivariateKzgPCS::<E>::commit(&ck, &p)?;
            let point = E::ScalarField::rand(rng);
            let (proof, value) = UnivariateKzgPCS::<E>::open(&ck, &p, &point)?;
            assert!(
                UnivariateKzgPCS::<E>::verify(&vk, &comm, &point, &value, &proof)?,
                "proof was incorrect for max_degree = {}, polynomial_degree = {}",
                degree,
                p.degree(),
            );
        }
        Ok(())
    }

    fn linear_polynomial_test_template<E>() -> Result<(), PCSError>
    where
        E: Pairing,
    {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let degree = 50;

            let pp = UnivariateKzgPCS::<E>::gen_srs_for_testing(rng, degree)?;
            let (ck, vk) = pp.trim(degree)?;
            let p = <DensePolynomial<E::ScalarField> as DenseUVPolynomial<E::ScalarField>>::rand(
                degree, rng,
            );
            let comm = UnivariateKzgPCS::<E>::commit(&ck, &p)?;
            let point = E::ScalarField::rand(rng);
            let (proof, value) = UnivariateKzgPCS::<E>::open(&ck, &p, &point)?;
            assert!(
                UnivariateKzgPCS::<E>::verify(&vk, &comm, &point, &value, &proof)?,
                "proof was incorrect for max_degree = {}, polynomial_degree = {}",
                degree,
                p.degree(),
            );
        }
        Ok(())
    }

    fn batch_check_test_template<E>() -> Result<(), PCSError>
    where
        E: Pairing,
    {
        let rng = &mut test_rng();
        for _ in 0..10 {
            let mut degree = 0;
            while degree <= 1 {
                degree = usize::rand(rng) % 20;
            }
            let pp = UnivariateKzgPCS::<E>::gen_srs_for_testing(rng, degree)?;
            let (ck, vk) = UnivariateKzgPCS::<E>::trim(&pp, degree, None)?;
            let mut comms = Vec::new();
            let mut values = Vec::new();
            let mut points = Vec::new();
            let mut proofs = Vec::new();
            for _ in 0..10 {
                let p =
                    <DensePolynomial<E::ScalarField> as DenseUVPolynomial<E::ScalarField>>::rand(
                        degree, rng,
                    );
                let comm = UnivariateKzgPCS::<E>::commit(&ck, &p)?;
                let point = E::ScalarField::rand(rng);
                let (proof, value) = UnivariateKzgPCS::<E>::open(&ck, &p, &point)?;

                assert!(UnivariateKzgPCS::<E>::verify(
                    &vk, &comm, &point, &value, &proof
                )?);
                comms.push(comm);
                values.push(value);
                points.push(point);
                proofs.push(proof);
            }
            assert!(UnivariateKzgPCS::<E>::batch_verify(
                &vk, &comms, &points, &values, &proofs, rng
            )?);
        }
        Ok(())
    }

    #[test]
    fn end_to_end_test() {
        end_to_end_test_template::<Bls12_381>().expect("test failed for bls12-381");
    }

    #[test]
    fn linear_polynomial_test() {
        linear_polynomial_test_template::<Bls12_381>().expect("test failed for bls12-381");
    }
    #[test]
    fn batch_check_test() {
        batch_check_test_template::<Bls12_381>().expect("test failed for bls12-381");
    }

    #[test]
    fn test_multi_open() -> Result<(), PCSError> {
        type E = Bls12_381;
        type Fr = ark_bls12_381::Fr;

        let mut rng = test_rng();
        let max_degree = 33;
        let pp = UnivariateKzgPCS::<E>::gen_srs_for_testing(&mut rng, max_degree)?;
        let degrees = [14, 15, 16, 17, 18];

        for degree in degrees {
            let num_points = rng.gen_range(5..30); // should allow more points than degree
            ark_std::println!(
                "Multi-opening: poly deg: {}, num of points: {}",
                degree,
                num_points
            );

            // NOTE: THIS IS IMPORTANT FOR USER OF `multi_open()`!
            // since we will pad your polynomial degree to the next_power_of_two, you will
            // need to trim to the correct padded degree as follows:
            let (ck, _) = UnivariateKzgPCS::<E>::trim_fft_size(&pp, degree)?;
            let poly = <DensePolynomial<Fr> as DenseUVPolynomial<Fr>>::rand(degree, &mut rng);
            let points: Vec<Fr> = (0..num_points).map(|_| Fr::rand(&mut rng)).collect();

            // First, test general points
            let (proofs, evals) = UnivariateKzgPCS::<E>::multi_open(&ck, &poly, &points)?;
            assert!(
                proofs.len() == evals.len() && proofs.len() == num_points,
                "fn multi_open() should return the correct number of proofs and evals"
            );
            points
                .iter()
                .zip(proofs.into_iter())
                .zip(evals.into_iter())
                .for_each(|((point, proof), eval)| {
                    assert_eq!(
                        UnivariateKzgPCS::<E>::open(&ck, &poly, point).unwrap(),
                        (proof, eval)
                    );
                });
            // Second, test roots-of-unity points
            let domain: Radix2EvaluationDomain<Fr> =
                UnivariateKzgPCS::<E>::multi_open_rou_eval_domain(degree, num_points)?;
            let (proofs, evals) =
                UnivariateKzgPCS::<E>::multi_open_rou(&ck, &poly, num_points, &domain)?;
            assert!(
                proofs.len() == evals.len() && proofs.len() == num_points,
                "fn multi_open_rou() should return the correct number of proofs and evals"
            );

            domain
                .elements()
                .take(num_points)
                .zip(proofs.into_iter())
                .zip(evals.into_iter())
                .for_each(|((point, proof), eval)| {
                    assert_eq!(
                        UnivariateKzgPCS::<E>::open(&ck, &poly, &point).unwrap(),
                        (proof, eval)
                    );
                });
        }

        Ok(())
    }

    /// Real download: first call downloads PTAU(7) and writes sidecar; second call should
    /// verify and return without rewriting (mtime unchanged).
    #[test]
    #[ignore]
    fn ptau_real_download_then_skip_on_second_run() {
        // Arrange: pick a unique temp directory to avoid polluting the repo
        let bin_dir = new_tmpdir();
        let ptau_path = bin_dir.join("ppot_7.ptau");

        // Sanity: ensure clean slate
        let _ = fs::remove_file(&ptau_path);

        // --- First run: should download and, with TOFU, write sidecar
        UnivariateKzgPCS::<Bn254>::download_ptau_file_if_needed(7, &ptau_path)
            .expect("first download should succeed");

        // Verify file exists and has some content
        let meta1 = fs::metadata(&ptau_path).expect("ptau exists after first run");
        assert!(meta1.len() > 0, "downloaded PTAU must be non-empty");

        // Capture modification time & contents for later comparison
        let mtime1 = meta1.modified().expect("mtime supported");
        let mut bytes1 = Vec::new();
        File::open(&ptau_path)
            .unwrap()
            .read_to_end(&mut bytes1)
            .unwrap();

        //  Sleep 1s to avoid coarse FS timestamp resolutions
        std::thread::sleep(Duration::from_secs(1));

        // --- Second run: should verify via sidecar and SKIP download/rewrites
        UnivariateKzgPCS::<Bn254>::download_ptau_file_if_needed(7, &ptau_path)
            .expect("second run should verify and return without rewriting");

        let meta2 = fs::metadata(&ptau_path).expect("ptau exists after second run");
        let mtime2 = meta2.modified().expect("mtime supported");

        // Assert: content unchanged
        let mut bytes2 = Vec::new();
        File::open(&ptau_path)
            .unwrap()
            .read_to_end(&mut bytes2)
            .unwrap();
        assert_eq!(bytes1, bytes2, "PTAU content changed unexpectedly");

        // Assert: mtime unchanged -> no rewrite happened
        assert_eq!(
            mtime1, mtime2,
            "PTAU mtime changed; second run should NOT rewrite the file"
        );

        // tidy up, remove downloaded files
        let _ = fs::remove_file(&ptau_path);
    }

    #[test]
    #[ignore]
    fn ptau_is_broken_before_second_run_recovers_by_redownloading() {
        // Arrange: pick a unique temp directory to avoid polluting the repo
        let bin_dir = new_tmpdir();
        let ptau_path = bin_dir.join("ppot_7.ptau");

        // Clean slate
        let _ = fs::remove_file(&ptau_path);

        // --- First run: create a broken/local bogus PTAU file
        let bad_bytes = b"CORRUPTED_PTAU_BYTES";
        fs::create_dir_all(ptau_path.parent().unwrap()).unwrap();
        File::create(&ptau_path)
            .unwrap()
            .write_all(bad_bytes)
            .unwrap();

        // Capture pre-call metadata (mtime/len)
        let meta1 = fs::metadata(&ptau_path).expect("broken ptau should exist");
        let len1 = meta1.len();
        let mtime1 = meta1.modified().expect("mtime supported");

        // Sleep 1s to avoid coarse FS timestamp resolution issues
        std::thread::sleep(Duration::from_secs(1));

        // --- Call: should detect mismatch, delete, re-download, and verify
        UnivariateKzgPCS::<Bn254>::download_ptau_file_if_needed(7, &ptau_path)
            .expect("should auto-heal by re-downloading a verified PTAU");

        // --- After: file should exist, be different/larger, and match embedded digest
        let meta2 = fs::metadata(&ptau_path).expect("ptau exists after recovery");
        let len2 = meta2.len();
        let mtime2 = meta2.modified().expect("mtime supported");

        // Should not be the tiny corrupted file anymore
        assert!(
            len2 > len1,
            "expected re-downloaded PTAU to be larger than the corrupted stub ({} <= {})",
            len2,
            len1
        );
        assert!(
            mtime2 > mtime1,
            "expected mtime to increase after re-download"
        );

        // Verify actual sha256 equals embedded canonical digest
        let (_raw, actual_hex) = super::file_sha256(&ptau_path).expect("sha256 of recovered PTAU");
        let expected_hex =
            crate::pcs::univariate_kzg::ptau_digests::expected_sha256_for_label("07")
                .expect("embedded digest for label 07");
        assert_eq!(
            actual_hex, expected_hex,
            "re-downloaded PTAU checksum must match embedded canonical digest"
        );

        // cleanup
        let _ = fs::remove_file(&ptau_path);
    }

    /// Helper: create a unique temp directory under the OS temp folder.
    fn new_tmpdir() -> PathBuf {
        let mut dir = std::env::temp_dir();
        // Avoid collisions across concurrent test runs
        let mut rnd = [0u8; 8];
        test_rng().fill_bytes(&mut rnd);
        dir.push(format!("nf4_kzg_tests_{:x}", u64::from_le_bytes(rnd)));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    // #[test]
    // #[ignore] // real network + large parse; unignore when running locally
    // fn cached_params_roundtrip_equals_original() {
    //     // Use a real PTAU (label 07) and a real universal setup (max_degree = 1<<7)
    //     let bin_dir = new_tmpdir();
    //     let ptau_path = bin_dir.join("ppot_7.ptau");

    //     // Clean slate
    //     let _ = fs::remove_file(&ptau_path);

    //     // 1) Download a verified PTAU(07). Function auto-heals corrupt files.
    //     UnivariateKzgPCS::<Bn254>::download_ptau_file_if_needed(7, &ptau_path)
    //         .expect("PTAU download/verify should succeed");

    //     // Universal setup expects the actual 'max_degree' (number of powers), not the label.
    //     let max_degree = 1usize << 7;

    //     // 2) First load: derives the params from the real PTAU and writes the cache
    //     let params1 = UnivariateKzgPCS::<Bn254>::universal_setup_bn254_cached(
    //         &ptau_path, max_degree, &bin_dir,
    //     )
    //     .expect("first cached load should succeed and write cache");

    //     // Locate the cache file to check it won't be rewritten on a cache hit
    //     let (_sha_bytes, sha_hex) = super::file_sha256(&ptau_path).expect("sha256 of PTAU");
    //     let cache_path = bin_dir.join(format!(
    //         "kzg_srs_bn254_deg{}_ptau_{}.bin",
    //         max_degree,
    //         &sha_hex[..16]
    //     ));
    //     let meta1 = fs::metadata(&cache_path).expect("cache file exists after first load");
    //     let mtime1 = meta1.modified().expect("mtime supported");

    //     // Avoid coarse FS timestamp issues
    //     std::thread::sleep(std::time::Duration::from_secs(1));

    //     // 3) Second load: must hit the cache (no rebuild / no rewrite)
    //     let params2 = UnivariateKzgPCS::<Bn254>::universal_setup_bn254_cached(
    //         &ptau_path, max_degree, &bin_dir,
    //     )
    //     .expect("second cached load should succeed from cache");

    //     // 4) Assert exact equality of all fields
    //     assert_eq!(
    //         params1.powers_of_g, params2.powers_of_g,
    //         "powers_of_g differ"
    //     );
    //     assert_eq!(params1.h, params2.h, "h differs");
    //     assert_eq!(params1.beta_h, params2.beta_h, "beta_h differs");

    //     // 5) Cache file should not have been rewritten on the second call
    //     let mtime2 = fs::metadata(&cache_path)
    //         .expect("cache file still exists")
    //         .modified()
    //         .expect("mtime supported");
    //     assert_eq!(
    //         mtime1, mtime2,
    //         "cache mtime changed; second load should not rewrite"
    //     );
    //     // tidy up
    //     let _ = fs::remove_file(&ptau_path);
    //     let _ = fs::remove_file(&cache_path);
    // }
}
