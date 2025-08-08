// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Main module for univariate KZG commitment scheme
use ark_std::{fs, io, path::PathBuf};

use crate::{
    pcs::{
        poly::GeneralDensePolynomial, PCSError, PolynomialCommitmentScheme,
        StructuredReferenceString, UnivariatePCS,
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
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow,
    end_timer, format,
    hash::Hash,
    marker::PhantomData,
    ops::Mul,
    rand::{CryptoRng, RngCore},
    start_timer,
    string::ToString,
    vec,
    vec::Vec,
    One, UniformRand, Zero,
};
use jf_utils::par_utils::parallelizable_slice_iter;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use srs::{UnivariateProverParam, UnivariateUniversalParams, UnivariateVerifierParam};

use super::Accumulation;

pub mod ptau;
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
    // Config
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
        ark_std::println!("JJ: am here in verify");
        ark_std::println!("commitment = {:?}", commitment);
        ark_std::println!("point = {:?}", point);
        ark_std::println!("value = {:?}", value);
        ark_std::println!("proof = {:?}", proof.proof);
        let check_time = start_timer!(|| "Checking evaluation");
        let pairing_inputs_l: Vec<E::G1Prepared> = vec![
            (verifier_param.g * value - proof.proof * point - commitment.into_group())
                .into_affine()
                .into(),
            proof.proof.into(),
        ];
        ark_std::println!("JJ: pairing_inputs_l = {:?}", pairing_inputs_l);
        let pairing_inputs_r: Vec<E::G2Prepared> =
            vec![verifier_param.h.into(), verifier_param.beta_h.into()];
        // ark_std::println!("JJ: pairing_inputs_r = {:?}", pairing_inputs_r);
        let g1_points: Vec<E::G1Affine> = vec![
            (verifier_param.g * value - proof.proof * point - commitment.into_group())
                .into_affine(),
            proof.proof.into(),
        ];
        for (i, affine) in g1_points.iter().enumerate() {
            ark_std::println!(
                "pairing_inputs_l[{}]: x = {:?}, y = {:?}",
                i,
                affine.x(),
                affine.y()
            );
        }
        let pairing_inputs_l: Vec<E::G1Prepared> = g1_points.iter().map(|p| p.into()).collect();

        let res = E::multi_pairing(pairing_inputs_l, pairing_inputs_r)
            .0
            .is_one();

        end_timer!(check_time, || format!("Result: {res}"));
        let pairing_inputs_l: Vec<E::G1Prepared> = vec![
            (commitment.into_group()).into_affine().into(),
            proof.proof.into(),
        ];
        let pairing_inputs_r: Vec<E::G2Prepared> =
            vec![verifier_param.h.into(), verifier_param.beta_h.into()];

        let res_new = E::multi_pairing(pairing_inputs_l, pairing_inputs_r)
            .0
            .is_one();
        ark_std::println!("JJ: res_new = {}", res_new);

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
        ark_std::println!("JJ: do this onchain");
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
            // randomizer = u128::rand(rng).into();
            randomizer = 2u128.into();
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

impl UnivariateKzgPCS<Bn254> {
    /// Specialized implementation of universal_setup for BN254
    pub fn universal_setup_bn254(
        ptau_file: &PathBuf,
        max_degree: usize,
    ) -> Result<UnivariateUniversalParams<Bn254>, PCSError> {
        let (powers_of_g, h) =
            parse_ptau_file::<Fq, Bn254ConfigOne, Bn254ConfigTwo>(ptau_file, max_degree, 2)
                .map_err(|e| {
                    ark_std::println!("Error parsing PTAU file: {:?}", e);
                    PCSError::InvalidSRS
                })?;

        let beta_h = h[1];
        Ok(UnivariateUniversalParams::<Bn254> {
            powers_of_g,
            h: h[0],
            beta_h,
        })
    }

    /// download a ptau file for BN254
    pub fn download_ptau_file_if_needed(
        max_degree: usize,
        ptau_file: &PathBuf,
    ) -> Result<(), io::Error> {
        // if the file already exists, we don't need to download it again
        if fs::metadata(ptau_file).is_ok() {
            return Ok(());
        }
        let url = format!(
        "https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_{}.ptau",
        max_degree,
        );
        let mut response = reqwest::blocking::get(url)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let mut file = fs::File::create(ptau_file)?;
        io::copy(&mut response, &mut file)?;
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
    use ark_std::{rand::Rng, UniformRand};
    use jf_utils::test_rng;

    use num_bigint::BigUint;

    /// Converts a decimal string or BigUint to 4 little-endian u64 limbs.
    pub fn bigint_to_limbs(n: &BigUint) -> [u64; 4] {
        let mut bytes = n.to_bytes_le();
        bytes.resize(32, 0); // pad to 32 bytes
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            limbs[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        limbs
    }

    use ark_std::One;
    use ark_std::Zero;
    use num_bigint::BigInt;

    fn limbs_to_bigint(limbs: &[u64]) -> BigInt {
        let base: BigInt = BigInt::one() << 64; // 2^64
        let mut result = BigInt::zero();

        for (i, &limb) in limbs.iter().enumerate() {
            // Convert each limb to BigInt and multiply by base^i, then add to the result
            result += BigInt::from(limb) * &base.pow(i as u32);
        }

        result
    }
    // Converts an Arkworks `[u64; 4]` (BigInt) to a `BigUint`
    fn ark_bigint_to_biguint(limbs: &[u64; 4]) -> BigUint {
        let mut bytes = vec![];
        for limb in limbs {
            bytes.extend_from_slice(&limb.to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes)
    }
    #[test]
    fn test_bn254_pairing_generator() {
        use ark_bn254::G1Affine;
        use ark_bn254::G2Affine;
        // G1 generator
        let g1 = G1Affine::generator();
        // G2 generator
        let g2 = G2Affine::generator();
        ark_std::println!("G1 generator: {:?}", g1);
        ark_std::println!("G2 generator: {:?}", g2);

        // e(g1, g2) == e(g1, g2)
        let res = Bn254::multi_pairing([g1, -g1], [g2, g2]).0.is_one();
        ark_std::println!("Pairing test: {}", res); // Should print true
    }
    #[test]
    fn test_accumulator() {
        let limbs = [
            3577443717552838115,
            11705489611516651137,
            16255439218571906434,
            3025449904741847268,
        ];

        let number = limbs_to_bigint(&limbs);
        ark_std::println!("Public key 1 x:{}", number);
        let limbs = [
            14665545369933317881,
            9933958648616565356,
            14589912518833666847,
            2177993963312036411,
        ];

        let number = limbs_to_bigint(&limbs);
        ark_std::println!("Public key 1 x:{}", number);

        // let limbs = bigint_to_limbs(&number.to_biguint().unwrap());
        // ark_std::println!("Public key 1 limbs: {:?}", limbs);
        use ark_ff::MontFp;
        let x: Fr254 = MontFp!(
            "11559732032986387107991004021392285783925812861821192530917403151452391805634"
        );
        let limb = x.into_bigint(); // [u64; 4]
        let biguint = ark_bigint_to_biguint(limb.as_ref().try_into().unwrap());
        let limbs_back = bigint_to_limbs(&biguint);
        ark_std::println!("limbsx0: {:?}", limbs_back);
        let x: Fr254 =
            MontFp!("4082367875863433681332203403145435568316851327593401208105741076214120093531");
        let limb = x.into_bigint(); // [u64; 4]
        let biguint = ark_bigint_to_biguint(limb.as_ref().try_into().unwrap());
        let limbs_back = bigint_to_limbs(&biguint);
        ark_std::println!("limbsx0: {:?}", limbs_back);
        let x: Fr254 = MontFp!(
            "10857046999023057135944570762232829481370756359578518086990519993285655852781"
        );
        let limb = x.into_bigint(); // [u64; 4]
        let biguint = ark_bigint_to_biguint(limb.as_ref().try_into().unwrap());
        let limbs_back = bigint_to_limbs(&biguint);
        ark_std::println!("limbsx0: {:?}", limbs_back);
        let x: Fr254 =
            MontFp!("8495653923123431417604973247489272438418190587263600148770280649306958101930");
        let limb = x.into_bigint(); // [u64; 4]
        let biguint = ark_bigint_to_biguint(limb.as_ref().try_into().unwrap());
        let limbs_back = bigint_to_limbs(&biguint);
        ark_std::println!("limbsx0: {:?}", limbs_back);
        // let x_again = limbs_to_bigint(&limbs_back);
        // ark_std::println!("x_again: {:?}", x_again);
        use ark_bn254::Fr as Fr254;
        use ark_poly::univariate::DensePolynomial;
        let rng = &mut test_rng();
        let mut degree = 0;
        while degree <= 1 {
            degree = usize::rand(rng) % 20;
        }
        let pp = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, degree).unwrap();
        let (ck, vk) = pp.trim(degree).unwrap();
        let p = <DensePolynomial<Fr254> as DenseUVPolynomial<Fr254>>::rand(degree, rng);
        let comm = UnivariateKzgPCS::<Bn254>::commit(&ck, &p).unwrap();
        let point = Fr254::rand(rng);
        let (proof, value) = UnivariateKzgPCS::<Bn254>::open(&ck, &p, &point).unwrap();
        UnivariateKzgPCS::<Bn254>::verify(&vk, &comm, &point, &value, &proof).unwrap();

        let mut vk_jj = vk.clone();
        use ark_bn254::Fq;
        vk_jj.g = <Bn254 as Pairing>::G1Affine::new(Fq::from(1), Fq::from(2));
        use ark_bn254::Fq2;
        use ark_ff::BigInteger256;
        let x = Fq2::new(
            Fq::new(BigInteger256::new([
                5106727233969649389,
                7440829307424791261,
                4785637993704342649,
                1729627375292849782,
            ])),
            Fq::new(BigInteger256::new([
                10945020018377822914,
                17413811393473931026,
                8241798111626485029,
                1841571559660931130,
            ])),
        );
        let y = Fq2::new(
            Fq::new(BigInteger256::new([
                5541340697920699818,
                16416156555105522555,
                5380518976772849807,
                1353435754470862315,
            ])),
            Fq::new(BigInteger256::new([
                6173549831154472795,
                13567992399387660019,
                17050234209342075797,
                650358724130500725,
            ])),
        );

        ark_std::println!("x = {}", x);
        ark_std::println!("y = {}", y);
        vk_jj.h = <Bn254 as Pairing>::G2Affine::new(x, y);
        ark_std::println!("JJ: vk_jj.g = {:?}", vk_jj.g);
        ark_std::println!("JJ: vk_jj.h = {:?}", vk_jj.h);
        // vk_jj.beta_h = Bn254::G2Affine::zero();

        let x: Fr254 = MontFp!(
            "10764647077472957448033591885865458661573660819003350325268673957890498500987"
        );
        let limb = x.into_bigint(); // [u64; 4]
        let biguint = ark_bigint_to_biguint(limb.as_ref().try_into().unwrap());
        let limbs_back = bigint_to_limbs(&biguint);
        ark_std::println!("limbsx0: {:?}", limbs_back);
        let x: Fr254 = MontFp!(
            "15207030507740967976352749097256929091435606784526748170016829002013506957017"
        );
        let limb = x.into_bigint(); // [u64; 4]
        let biguint = ark_bigint_to_biguint(limb.as_ref().try_into().unwrap());
        let limbs_back = bigint_to_limbs(&biguint);
        ark_std::println!("limbsx1: {:?}", limbs_back);
        let x: Fr254 = MontFp!(
            "18253511544609001572866960948873128266198935669250718031100637619547827597184"
        );
        let limb = x.into_bigint(); // [u64; 4]
        let biguint = ark_bigint_to_biguint(limb.as_ref().try_into().unwrap());
        let limbs_back = bigint_to_limbs(&biguint);
        ark_std::println!("limbsy0: {:?}", limbs_back);
        let x: Fr254 = MontFp!(
            "19756181390911900613508142947142748782977087973617411469215564659012323409872"
        );
        let limb = x.into_bigint(); // [u64; 4]
        let biguint = ark_bigint_to_biguint(limb.as_ref().try_into().unwrap());
        let limbs_back = bigint_to_limbs(&biguint);
        ark_std::println!("limbsy1: {:?}", limbs_back);

        let x = Fq2::new(
            Fq::new(BigInteger256::new([
                3059198416762171264,
                17826071752375934067,
                2540209951312773215,
                2907952159147943523,
            ])),
            Fq::new(BigInteger256::new([
                9632549258536950139,
                4162086999619294322,
                15740780115627737347,
                1714907218531776084,
            ])),
        );
        let y = Fq2::new(
            Fq::new(BigInteger256::new([
                14329149837203636176,
                662368139879402519,
                3020902600790832773,
                3147341276872722899,
            ])),
            Fq::new(BigInteger256::new([
                11975304264280600281,
                16369974670302398806,
                7444268968364960217,
                2422619729422656478,
            ])),
        );
        ark_std::println!("x = {}", x);
        ark_std::println!("y = {}", y);

        vk_jj.beta_h = <Bn254 as Pairing>::G2Affine::new(x, y);
        ark_std::println!("JJ: vk_jj.beta_h = {:?}", vk_jj.beta_h);

        let x: Fq =
            MontFp!("3887810704895428322962904948451372935129289338514763739831754287772972287096");
        let y: Fq = MontFp!(
            "17425095276760945095381060928122902375593783607513862431869872042018379379257"
        );

        let commitment = <Bn254 as Pairing>::G1Affine::new(x, y);
        let point = Fr254::from(0u64);
        let value = Fr254::from(0u64);
        let x: Fq = MontFp!(
            "18991056847380517498711743163082681994472759809034084754240255183913686701539"
        );
        let y: Fq = MontFp!(
            "13671489686767698476199199614913759207054156363727130982459349913990206588665"
        );
        let proof = UnivariateKzgProof::<Bn254> {
            proof: <Bn254 as Pairing>::G1Affine::new(x, y),
        };
        UnivariateKzgPCS::<Bn254>::verify(&vk, &comm, &point, &value, &proof).unwrap();
    }

    fn end_to_end_test_template<E>() -> Result<(), PCSError>
    where
        E: Pairing,
    {
        let rng = &mut test_rng();
        for _ in 0..1 {
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
        for _ in 0..1 {
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
}
