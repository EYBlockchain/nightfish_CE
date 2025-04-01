// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Main module for multilinear KZG commitment scheme

mod batching;
pub(crate) mod srs;
pub(crate) mod util;

use crate::pcs::{
    prelude::Commitment, PCSError, PolynomialCommitmentScheme, StructuredReferenceString,
};
#[cfg(target_has_atomic = "ptr")]
use ark_ec::{
    pairing::Pairing,
    scalar_mul::{fixed_base::FixedBase, variable_base::VariableBaseMSM},
    AffineRepr, CurveGroup,
};
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow,
    cfg_iter, end_timer, format,
    hash::Hash,
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    start_timer,
    string::ToString,
    sync::Arc,
    vec,
    vec::Vec,
    One, Zero,
};
use batching::{batch_verify_internal, multi_open_internal};
use rayon::prelude::*;
use srs::{MultilinearProverParam, MultilinearUniversalParams, MultilinearVerifierParam};

pub use self::batching::MultilinearKzgBatchProof;

use super::Accumulation;

/// KZG Polynomial Commitment Scheme on multilinear polynomials.
#[derive(Clone)]
pub struct MultilinearKzgPCS<E: Pairing> {
    #[doc(hidden)]
    phantom: PhantomData<E>,
}

impl<E: Pairing> Default for MultilinearKzgPCS<E> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
/// proof of opening
pub struct MultilinearKzgProof<E: Pairing> {
    /// Evaluation of quotients
    pub proofs: Vec<E::G1Affine>,
}

impl<E: Pairing> Default for MultilinearKzgProof<E> {
    fn default() -> Self {
        Self {
            proofs: Vec::<E::G1Affine>::default(),
        }
    }
}

impl<E: Pairing> Hash for MultilinearKzgProof<E> {
    fn hash<H: ark_std::hash::Hasher>(&self, state: &mut H) {
        self.proofs.hash(state);
    }
}

// #[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
// /// proof of batch opening
// pub struct MultilinearKzgBatchProof<E: Pairing> {
//     /// The actual proof
//     pub proof: MultilinearKzgProof<E>,
//     /// Commitment to q(x):= w(l(x)) where
//     /// - `w` is the merged MLE
//     /// - `l` is the list of univariate polys that goes through all points
//     pub q_x_commit: Commitment<E>,
//     /// openings of q(x) at 1, omega, ..., and r
//     pub q_x_opens: Vec<UnivariateKzgProof<E>>,
// }

/// Multi-linear Extension (MLE) polynomial, this type alias is set to owned
/// `DenseMultilinearExtension` on wasm platforms since only message-passing
/// concurrency is supported. And set to `Arc<DenseMultilinearExtension>` for
/// platforms that supports atomic operations (e.g. mostly non-wasm, MIPS, x86
/// etc.)
pub type MLE<F> = Arc<DenseMultilinearExtension<F>>;

impl<E: Pairing> PolynomialCommitmentScheme for MultilinearKzgPCS<E> {
    // Config
    type SRS = MultilinearUniversalParams<E>;
    // Polynomial and its associated types
    type Polynomial = MLE<E::ScalarField>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    // Commitments and proofs
    type Commitment = Commitment<E>;
    type Proof = MultilinearKzgProof<E>;
    type BatchProof = MultilinearKzgBatchProof<E>;

    /// Trim the universal parameters to specialize the public parameters.
    /// Input both `supported_log_degree` for univariate and
    /// `supported_num_vars` for multilinear.
    fn trim(
        srs: impl Borrow<Self::SRS>,
        _supported_log_degree: usize,
        supported_num_vars: Option<usize>,
    ) -> Result<(MultilinearProverParam<E>, MultilinearVerifierParam<E>), PCSError> {
        let supported_num_vars = match supported_num_vars {
            Some(p) => p,
            None => {
                return Err(PCSError::InvalidParameters(
                    "multilinear should receive a num_var param".to_string(),
                ))
            },
        };

        let (ml_ck, ml_vk) = srs.borrow().trim(supported_num_vars)?;

        Ok((ml_ck, ml_vk))
    }

    /// Generate a commitment for a polynomial.
    ///
    /// This function takes `2^num_vars` number of scalar multiplications over
    /// G1.
    fn commit(
        prover_param: impl Borrow<MultilinearProverParam<E>>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let prover_param = prover_param.borrow();
        let commit_timer = start_timer!(|| "commit");
        if prover_param.num_vars < poly.num_vars {
            return Err(PCSError::InvalidParameters(format!(
                "Poly length ({}) exceeds param limit ({})",
                poly.num_vars, prover_param.num_vars
            )));
        }
        let ignored = prover_param.num_vars - poly.num_vars;
        let scalars: Vec<_> = poly
            .to_evaluations()
            .into_iter()
            .map(|x| x.into_bigint())
            .collect();
        let commitment =
            E::G1::msm_bigint(&prover_param.powers_of_g[ignored].evals, scalars.as_slice())
                .into_affine();

        end_timer!(commit_timer);
        Ok(commitment)
    }

    /// Batch commit a list of polynomials.
    ///
    /// This function takes `2^(num_vars + log(polys.len())` number of scalar
    /// multiplications over G1.
    fn batch_commit(
        prover_param: impl Borrow<MultilinearProverParam<E>>,
        polys: &[Self::Polynomial],
    ) -> Result<Vec<Self::Commitment>, PCSError> {
        let prover_param = prover_param.borrow();

        let commitments = cfg_iter!(polys)
            .map(|poly| Self::commit(prover_param, poly))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(commitments)
    }

    /// On input a polynomial `p` and a point `point`, outputs a proof for the
    /// same. This function does not need to take the evaluation value as an
    /// input.
    ///
    /// This function takes 2^{num_var +1} number of scalar multiplications over
    /// G1:
    /// - it prodceeds with `num_var` number of rounds,
    /// - at round i, we compute an MSM for `2^{num_var - i + 1}` number of G2
    ///   elements.
    fn open(
        prover_param: impl Borrow<MultilinearProverParam<E>>,
        polynomial: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        open_internal(prover_param.borrow(), polynomial, point)
    }

    /// Input
    /// - the prover parameters for univariate KZG,
    /// - the prover parameters for multilinear KZG,
    /// - a list of polynomials,
    /// - a (batch) commitment to all polynomials,
    /// - and a same number of points,
    ///   compute a batch opening for all the polynomials.
    ///
    /// For simplicity, this API requires each MLE to have only one point. If
    /// the caller wish to use more than one points per MLE, it should be
    /// handled at the caller layer.
    ///
    /// Returns an error if the lengths do not match.
    ///
    /// Returns the proof, consists of
    /// - the multilinear KZG opening
    /// - the univariate KZG commitment to q(x)
    /// - the openings and evaluations of q(x) at omega^i and r
    ///
    /// Steps:
    /// 1. build `l(points)` which is a list of univariate polynomials that goes
    ///    through the points
    /// 2. build MLE `w` which is the merge of all MLEs.
    /// 3. build `q(x)` which is a univariate polynomial `W circ l`
    /// 4. commit to q(x) and sample r from transcript
    ///    transcript contains: w commitment, points, q(x)'s commitment
    /// 5. build q(omega^i) and their openings
    /// 6. build q(r) and its opening
    /// 7. get a point `p := l(r)`
    /// 8. output an opening of `w` over point `p`
    /// 9. output `w(p)`
    fn batch_open(
        prover_param: impl Borrow<MultilinearProverParam<E>>,
        _batch_commitment: &[Self::Commitment],
        polynomials: &[Self::Polynomial],
        points: &[Self::Point],
    ) -> Result<(MultilinearKzgBatchProof<E>, Self::Point), PCSError> {
        let evals = polynomials
            .iter()
            .zip(points.iter())
            .map(|(poly, point)| {
                poly.evaluate(point).ok_or_else(|| {
                    PCSError::InvalidParameters("fail to eval poly at the point".to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut transcript = transcript::IOPTranscript::new(b"batch");
        for point in points.iter() {
            for p in point.iter() {
                transcript.append_serializable_element(b"point", p)?;
            }
        }
        let proof = multi_open_internal::<E>(
            prover_param.borrow(),
            polynomials,
            points,
            &evals,
            &mut transcript,
        )?;

        Ok((proof, vec![]))
    }

    /// Verifies that `value` is the evaluation at `x` of the polynomial
    /// committed inside `comm`.
    ///
    /// This function takes
    /// - num_var number of pairing product.
    /// - num_var number of MSM
    fn verify(
        verifier_param: &MultilinearVerifierParam<E>,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        verify_internal(verifier_param, commitment, point, value, proof)
    }

    /// Verifies that `value` is the evaluation at `x_i` of the polynomial
    /// `poly_i` committed inside `commitment`.
    /// steps:
    ///
    /// 1. put `q(x)`'s evaluations over `(1, omega,...)` into transcript
    /// 2. sample `r` from transcript
    /// 3. check `q(r) == value`
    /// 4. build `l(points)` which is a list of univariate polynomials that goes
    ///    through the points
    /// 5. get a point `p := l(r)`
    /// 6. verifies `p` is verifies against proof
    fn batch_verify<R: RngCore + CryptoRng>(
        verifier_param: &MultilinearVerifierParam<E>,
        batch_commitment: &[Self::Commitment],
        points: &[Self::Point],
        _values: &[E::ScalarField],
        batch_proof: &Self::BatchProof,
        _rng: &mut R,
    ) -> Result<bool, PCSError> {
        let mut transcript = transcript::IOPTranscript::new(b"batch");
        for point in points.iter() {
            for p in point.iter() {
                transcript.append_serializable_element(b"point", p)?;
            }
        }
        batch_verify_internal(
            verifier_param,
            batch_commitment,
            points,
            batch_proof,
            &mut transcript,
        )
    }
}

/// On input a polynomial `p` and a point `point`, outputs a proof for the
/// same. This function does not need to take the evaluation value as an
/// input.
///
/// This function takes 2^{num_var} number of scalar multiplications over
/// G1:
/// - it proceeds with `num_var` number of rounds,
/// - at round i, we compute an MSM for `2^{num_var - i}` number of G1 elements.
fn open_internal<E: Pairing>(
    prover_param: &MultilinearProverParam<E>,
    polynomial: &DenseMultilinearExtension<E::ScalarField>,
    point: &[E::ScalarField],
) -> Result<(MultilinearKzgProof<E>, E::ScalarField), PCSError> {
    let open_timer = start_timer!(|| format!("open mle with {} variable", polynomial.num_vars));

    if polynomial.num_vars() > prover_param.num_vars {
        return Err(PCSError::InvalidParameters(format!(
            "Polynomial num_vars {} exceed the limit {}",
            polynomial.num_vars, prover_param.num_vars
        )));
    }

    if polynomial.num_vars() != point.len() {
        return Err(PCSError::InvalidParameters(format!(
            "Polynomial num_vars {} does not match point len {}",
            polynomial.num_vars,
            point.len()
        )));
    }

    let nv = polynomial.num_vars();
    // the first `ignored` SRS vectors are unused
    let ignored = prover_param.num_vars - nv + 1;

    let mut f = polynomial.to_evaluations();

    let mut proofs = Vec::new();

    for (i, (&point_at_k, gi)) in point
        .iter()
        .zip(prover_param.powers_of_g[ignored..ignored + nv].iter())
        .enumerate()
    {
        let ith_round = start_timer!(|| format!("{}-th round", i));

        let k = nv - 1 - i;
        let cur_dim = 1 << k;
        let mut q = vec![E::ScalarField::zero(); cur_dim];
        let mut r = vec![E::ScalarField::zero(); cur_dim];

        let ith_round_eval = start_timer!(|| format!("{}-th round eval", i));
        for b in 0..(1 << k) {
            // q[b] = f[1, b] - f[0, b]
            q[b] = f[(b << 1) + 1] - f[b << 1];

            // r[b] = f[0, b] + q[b] * p
            r[b] = f[b << 1] + (q[b] * point_at_k);
        }
        f = r;
        end_timer!(ith_round_eval);
        let scalars: Vec<_> = q.iter().map(|x| x.into_bigint()).collect();

        // this is a MSM over G1 and is likely to be the bottleneck
        let msm_timer = start_timer!(|| format!("msm of size {} at round {}", gi.evals.len(), i));

        proofs.push(E::G1::msm_bigint(&gi.evals, &scalars).into_affine());
        end_timer!(msm_timer);

        end_timer!(ith_round);
    }
    let eval = polynomial
        .evaluate(point)
        .ok_or_else(|| PCSError::InvalidParameters("fail to eval poly at the point".to_string()))?;
    end_timer!(open_timer);
    Ok((MultilinearKzgProof { proofs }, eval))
}

/// Verifies that `value` is the evaluation at `x` of the polynomial
/// committed inside `comm`.
///
/// This function takes
/// - num_var number of pairing product.
/// - num_var number of MSM
fn verify_internal<E: Pairing>(
    verifier_param: &MultilinearVerifierParam<E>,
    commitment: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &MultilinearKzgProof<E>,
) -> Result<bool, PCSError> {
    let verify_timer = start_timer!(|| "verify");
    let num_var = point.len();

    if num_var > verifier_param.num_vars {
        return Err(PCSError::InvalidParameters(format!(
            "point length ({}) exceeds param limit ({})",
            num_var, verifier_param.num_vars
        )));
    }

    let prepare_inputs_timer = start_timer!(|| "prepare pairing inputs");

    let scalar_size = E::ScalarField::MODULUS_BIT_SIZE as usize;
    let window_size = FixedBase::get_mul_window_size(num_var);

    let h_table =
        FixedBase::get_window_table(scalar_size, window_size, verifier_param.h.into_group());
    let h_mul: Vec<E::G2> = FixedBase::msm(scalar_size, window_size, &h_table, point);

    // the first `ignored` G2 parameters are unused
    let ignored = verifier_param.num_vars - num_var;
    let h_vec: Vec<_> = (0..num_var)
        .map(|i| verifier_param.h_mask[ignored + i].into_group() - h_mul[i])
        .collect();
    let h_vec: Vec<E::G2Affine> = E::G2::normalize_batch(&h_vec);
    end_timer!(prepare_inputs_timer);

    let pairing_product_timer = start_timer!(|| "pairing product");

    let mut pairings_l: Vec<E::G1Prepared> = proof
        .proofs
        .iter()
        .map(|&x| E::G1Prepared::from(x))
        .collect();

    let mut pairings_r: Vec<E::G2Prepared> = h_vec
        .into_iter()
        .take(num_var)
        .map(E::G2Prepared::from)
        .collect();
    pairings_l.push(E::G1Prepared::from(
        (verifier_param.g * (*value) - commitment.into_group()).into_affine(),
    ));
    pairings_r.push(E::G2Prepared::from(verifier_param.h));

    if pairings_l.len() != pairings_r.len() {
        return Err(PCSError::InvalidParameters(
            "pairings_l and pairings_r have different lengths".to_string(),
        ));
    }

    let res = E::multi_pairing(pairings_l, pairings_r).0 == E::TargetField::one();

    end_timer!(pairing_product_timer);
    end_timer!(verify_timer);
    Ok(res)
}

impl<E: Pairing> Accumulation for MultilinearKzgPCS<E> {}
#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{rand::RngCore, vec::Vec, UniformRand};
    use jf_utils::test_rng;
    type E = Bls12_381;
    type Fr = <E as Pairing>::ScalarField;

    fn test_single_helper<R: RngCore + CryptoRng>(
        params: &MultilinearUniversalParams<E>,
        poly: &MLE<Fr>,
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let nv = poly.num_vars();
        assert_ne!(nv, 0);
        let uni_degree = 1;
        let (ck, vk) = MultilinearKzgPCS::trim(params, uni_degree, Some(nv))?;
        let point: Vec<_> = (0..nv).map(|_| Fr::rand(rng)).collect();
        let com = MultilinearKzgPCS::commit(&ck, poly)?;
        let (proof, value) = MultilinearKzgPCS::open(&ck, poly, &point)?;

        assert!(MultilinearKzgPCS::verify(
            &vk, &com, &point, &value, &proof
        )?);

        let value = Fr::rand(rng);
        assert!(!MultilinearKzgPCS::verify(
            &vk, &com, &point, &value, &proof
        )?);

        Ok(())
    }

    #[test]
    fn test_single_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();

        let params = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;

        // normal polynomials
        let poly1 = Arc::new(DenseMultilinearExtension::rand(8, &mut rng));
        test_single_helper(&params, &poly1, &mut rng)?;

        // single-variate polynomials
        let poly2 = Arc::new(DenseMultilinearExtension::rand(1, &mut rng));
        test_single_helper(&params, &poly2, &mut rng)?;

        Ok(())
    }

    #[test]
    fn setup_commit_verify_constant_polynomial() {
        let mut rng = test_rng();

        // normal polynomials
        assert!(MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 0).is_err());
    }
}
