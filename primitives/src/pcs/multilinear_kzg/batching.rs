// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

use super::MLE;
use super::{
    srs::{MultilinearProverParam, MultilinearVerifierParam},
    MultilinearKzgPCS, MultilinearKzgProof,
};
use crate::pcs::{
    multilinear_kzg::util::eq_eval, prelude::Commitment, PCSError, PolynomialCommitmentScheme,
};
use ark_ec::pairing::Pairing;
use ark_poly::DenseMultilinearExtension;

use ark_std::{end_timer, start_timer, string::ToString, vec, vec::Vec};

// /// Input
// /// - the prover parameters for univariate KZG,
// /// - the prover parameters for multilinear KZG,
// /// - a list of MLEs,
// /// - a batch commitment to all MLEs
// /// - and a same number of points,
// /// compute a batch opening for all the polynomials.
// ///
// /// For simplicity, this API requires each MLE to have only one point. If
// /// the caller wish to use more than one points per MLE, it should be
// /// handled at the caller layer.
// ///
// /// Returns an error if the lengths do not match.
// ///
// /// Returns the proof, consists of
// /// - the multilinear KZG opening
// /// - the univariate KZG commitment to q(x)
// /// - the openings and evaluations of q(x) at omega^i and r
// ///
// /// Steps:
// /// 1. build `l(points)` which is a list of univariate polynomials that goes
// /// through the points
// /// 2. build MLE `w` which is the merge of all MLEs.
// /// 3. build `q(x)` which is a univariate polynomial `W circ l`
// /// 4. commit to q(x) and sample r from transcript
// /// transcript contains: w commitment, points, q(x)'s commitment
// /// 5. build q(omega^i) and their openings
// /// 6. build q(r) and its opening
// /// 7. get a point `p := l(r)`
// /// 8. output an opening of `w` over point `p`
// /// 9. output `w(p)`
// ///
// /// TODO: Migrate the batching algorithm in HyperPlonk repo
// pub(super) fn batch_open_internal<E: Pairing>(
//     uni_prover_param: &UnivariateProverParam<E>,
//     ml_prover_param: &MultilinearProverParam<E>,
//     polynomials: &[MLE<E::ScalarField>],
//     batch_commitment: &Commitment<E>,
//     points: &[Vec<E::ScalarField>],
// ) -> Result<(MultilinearKzgBatchProof<E>, Vec<E::ScalarField>), PCSError> {
//     let open_timer = start_timer!(|| "batch open");

//     // ===================================
//     // Sanity checks on inputs
//     // ===================================
//     let points_len = points.len();
//     if points_len == 0 {
//         return Err(PCSError::InvalidParameters("points is empty".to_string()));
//     }

//     if points_len != polynomials.len() {
//         return Err(PCSError::InvalidParameters(
//             "polynomial length does not match point length".to_string(),
//         ));
//     }

//     let num_var = polynomials[0].num_vars();
//     for poly in polynomials.iter().skip(1) {
//         if poly.num_vars() != num_var {
//             return Err(PCSError::InvalidParameters(
//                 "polynomials do not have same num_vars".to_string(),
//             ));
//         }
//     }
//     for point in points.iter() {
//         if point.len() != num_var {
//             return Err(PCSError::InvalidParameters(
//                 "points do not have same num_vars".to_string(),
//             ));
//         }
//     }

//     let domain = get_uni_domain::<E::ScalarField>(points_len)?;

//     // 1. build `l(points)` which is a list of univariate polynomials that goes
//     // through the points
//     let uni_polys = build_l(num_var, points, &domain)?;

//     // 2. build MLE `w` which is the merge of all MLEs.
//     let merge_poly = merge_polynomials(polynomials)?;

//     // 3. build `q(x)` which is a univariate polynomial `W circ l`
//     let q_x = compute_w_circ_l(&merge_poly, &uni_polys)?;

//     // 4. commit to q(x) and sample r from transcript
//     // transcript contains: w commitment, points, q(x)'s commitment
//     let mut transcript = IOPTranscript::new(b"ml kzg");
//     transcript.append_serializable_element(b"w", batch_commitment)?;
//     for point in points {
//         transcript.append_serializable_element(b"w", point)?;
//     }

//     let q_x_commit = UnivariateKzgPCS::<E>::commit(uni_prover_param, &q_x)?;
//     transcript.append_serializable_element(b"q(x)", &q_x_commit)?;
//     let r = transcript.get_and_append_challenge(b"r")?;

//     // 5. build q(omega^i) and their openings
//     let mut q_x_opens = vec![];
//     let mut q_x_evals = vec![];
//     for i in 0..points_len {
//         let (q_x_open, q_x_eval) =
//             UnivariateKzgPCS::<E>::open(uni_prover_param, &q_x, &domain.element(i))?;
//         q_x_opens.push(q_x_open);
//         q_x_evals.push(q_x_eval);

//         // sanity check
//         let point: Vec<E::ScalarField> = uni_polys
//             .iter()
//             .rev()
//             .map(|poly| poly.evaluate(&domain.element(i)))
//             .collect();
//         let mle_eval = merge_poly.evaluate(&point).unwrap();
//         if mle_eval != q_x_eval {
//             return Err(PCSError::InvalidProver(
//                 "Q(omega) does not match W(l(omega))".to_string(),
//             ));
//         }
//     }

//     // 6. build q(r) and its opening
//     let (q_x_open, q_r_value) = UnivariateKzgPCS::<E>::open(uni_prover_param, &q_x, &r)?;
//     q_x_opens.push(q_x_open);
//     q_x_evals.push(q_r_value);

//     // 7. get a point `p := l(r)`
//     let point: Vec<E::ScalarField> = uni_polys
//         .iter()
//         .rev()
//         .map(|poly| poly.evaluate(&r))
//         .collect();

//     // 8. output an opening of `w` over point `p`
//     let (mle_opening, mle_eval) = open_internal(ml_prover_param, &merge_poly, &point)?;

//     // 9. output value that is `w` evaluated at `p` (which should match `q(r)`)
//     if mle_eval != q_r_value {
//         return Err(PCSError::InvalidProver(
//             "Q(r) does not match W(l(r))".to_string(),
//         ));
//     }
//     end_timer!(open_timer);

//     Ok((
//         MultilinearKzgBatchProof {
//             proof: mle_opening,
//             q_x_commit,
//             q_x_opens,
//         },
//         q_x_evals,
//     ))
// }

// /// Verifies that the `batch_commitment` is a valid commitment
// /// to a list of MLEs for the given openings and evaluations in
// /// the batch_proof.
// ///
// /// steps:
// ///
// /// 1. push w, points and q_com into transcript
// /// 2. sample `r` from transcript
// /// 3. check `q(r) == batch_proof.q_x_value.last` and
// /// `q(omega^i) == batch_proof.q_x_value[i]`
// /// 4. build `l(points)` which is a list of univariate
// /// polynomials that goes through the points
// /// 5. get a point `p := l(r)`
// /// 6. verifies `p` is valid against multilinear KZG proof
// pub(super) fn batch_verify_internal<E: Pairing>(
//     uni_verifier_param: &UnivariateVerifierParam<E>,
//     ml_verifier_param: &MultilinearVerifierParam<E>,
//     batch_commitment: &Commitment<E>,
//     points: &[Vec<E::ScalarField>],
//     values: &[E::ScalarField],
//     batch_proof: &MultilinearKzgBatchProof<E>,
// ) -> Result<bool, PCSError> {
//     let verify_timer = start_timer!(|| "batch verify");

//     // ===================================
//     // Sanity checks on inputs
//     // ===================================
//     let points_len = points.len();
//     if points_len == 0 {
//         return Err(PCSError::InvalidParameters("points is empty".to_string()));
//     }

//     // add one here because we also have q(r) and its opening
//     if points_len + 1 != batch_proof.q_x_opens.len() {
//         return Err(PCSError::InvalidParameters(
//             "openings length does not match point length".to_string(),
//         ));
//     }

//     if points_len + 1 != values.len() {
//         return Err(PCSError::InvalidParameters(
//             "values length does not match point length".to_string(),
//         ));
//     }

//     let num_var = points[0].len();
//     for point in points.iter().skip(1) {
//         if point.len() != num_var {
//             return Err(PCSError::InvalidParameters(format!(
//                 "points do not have same num_vars ({} vs {})",
//                 point.len(),
//                 num_var,
//             )));
//         }
//     }

//     let domain = get_uni_domain::<E::ScalarField>(points_len)?;

//     // 1. push w, points and q_com into transcript
//     let mut transcript = IOPTranscript::new(b"ml kzg");
//     transcript.append_serializable_element(b"w", batch_commitment)?;
//     for point in points {
//         transcript.append_serializable_element(b"w", point)?;
//     }

//     transcript.append_serializable_element(b"q(x)", &batch_proof.q_x_commit)?;

//     // 2. sample `r` from transcript
//     let r = transcript.get_and_append_challenge(b"r")?;

//     // 3. check `q(r) == batch_proof.q_x_value.last` and `q(omega^i) =
//     // batch_proof.q_x_value[i]`
//     for (i, value) in values.iter().enumerate().take(points_len) {
//         if !UnivariateKzgPCS::verify(
//             uni_verifier_param,
//             &batch_proof.q_x_commit,
//             &domain.element(i),
//             value,
//             &batch_proof.q_x_opens[i],
//         )? {
//             #[cfg(debug_assertion)]
//             println!("q(omega^{}) verification failed", i);
//             return Ok(false);
//         }
//     }

//     if !UnivariateKzgPCS::verify(
//         uni_verifier_param,
//         &batch_proof.q_x_commit,
//         &r,
//         &values[points_len],
//         &batch_proof.q_x_opens[points_len],
//     )? {
//         #[cfg(debug_assertion)]
//         println!("q(r) verification failed");
//         return Ok(false);
//     }

//     // 4. build `l(points)` which is a list of univariate polynomials that goes
//     // through the points
//     let uni_polys = build_l(num_var, points, &domain)?;

//     // 5. get a point `p := l(r)`
//     let point: Vec<E::ScalarField> = uni_polys.iter().rev().map(|x| x.evaluate(&r)).collect();

//     // 6. verifies `p` is valid against multilinear KZG proof
//     let res = verify_internal(
//         ml_verifier_param,
//         batch_commitment,
//         &point,
//         &values[points_len],
//         &batch_proof.proof,
//     )?;

//     #[cfg(debug_assertion)]
//     if !res {
//         println!("multilinear KZG verification failed");
//     }

//     end_timer!(verify_timer);

//     Ok(res)
// }

use arithmetic::{build_eq_x_r_vec, VPAuxInfo, VirtualPolynomial};
use ark_ec::{scalar_mul::variable_base::VariableBaseMSM, CurveGroup};
use subroutines::{
    poly_iop::{prelude::SumCheck, PolyIOP},
    IOPProof,
};

use ark_std::{
    collections::BTreeMap, iter, log2, marker::PhantomData, ops::Deref, sync::Arc, One, Zero,
};
use transcript::IOPTranscript as HyperTranscript;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchProof<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme,
{
    /// A sum check proof proving tilde g's sum
    pub(crate) sum_check_proof: IOPProof<E::ScalarField>,
    /// f_i(point_i)
    pub f_i_eval_at_point_i: Vec<E::ScalarField>,
    /// proof for g'(a_2)
    pub(crate) g_prime_proof: PCS::Proof,
}

/// Batch proof for multilinear kzg using hyperplonk method for batching
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultilinearKzgBatchProof<E: Pairing> {
    /// A sum check proof proving tilde g's sum
    pub(crate) sum_check_proof: IOPProof<E::ScalarField>,
    /// f_i(point_i)
    pub f_i_eval_at_point_i: Vec<E::ScalarField>,
    /// proof for g'(a_2)
    pub(crate) g_prime_proof: MultilinearKzgProof<E>,
}

impl<E: Pairing> Default for MultilinearKzgBatchProof<E> {
    fn default() -> Self {
        Self {
            sum_check_proof: IOPProof::<E::ScalarField>::default(),
            f_i_eval_at_point_i: Vec::<E::ScalarField>::default(),
            g_prime_proof: MultilinearKzgProof::<E>::default(),
        }
    }
}

/// Steps:
/// 1. get challenge point t from transcript
/// 2. build eq(t,i) for i in [0..k]
/// 3. build \tilde g_i(b) = eq(t, i) * f_i(b)
/// 4. compute \tilde eq_i(b) = eq(b, point_i)
/// 5. run sumcheck on \sum_i=1..k \tilde eq_i * \tilde g_i
/// 6. build g'(X) = \sum_i=1..k \tilde eq_i(a2) * \tilde g_i(X) where (a2) is
///    the sumcheck's point 7. open g'(X) at point (a2)
pub(crate) fn multi_open_internal<E>(
    prover_param: &MultilinearProverParam<E>,
    polynomials: &[MLE<E::ScalarField>],
    points: &[Vec<E::ScalarField>],
    evals: &[E::ScalarField],
    transcript: &mut HyperTranscript<E::ScalarField>,
) -> Result<MultilinearKzgBatchProof<E>, PCSError>
where
    E: Pairing,
{
    let open_timer = start_timer!(|| format!("multi open {} points", points.len()));

    // TODO: sanity checks
    let num_var = polynomials[0].num_vars;
    let k = polynomials.len();
    let ell = log2(k) as usize;

    // challenge point t
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

    // eq(t, i) for i in [0..k]
    let eq_t_i_list = build_eq_x_r_vec(t.as_ref())?;

    // \tilde g_i(b) = eq(t, i) * f_i(b)
    let timer = start_timer!(|| format!("compute tilde g for {} points", points.len()));
    // combine the polynomials that have same opening point first to reduce the
    // cost of sum check later.
    let point_indices = points
        .iter()
        .fold(BTreeMap::<_, _>::new(), |mut indices, point| {
            let idx = indices.len();
            indices.entry(point).or_insert(idx);
            indices
        });
    let deduped_points =
        BTreeMap::from_iter(point_indices.iter().map(|(point, idx)| (*idx, *point)))
            .into_values()
            .collect::<Vec<_>>();
    let merged_tilde_gs = polynomials
        .iter()
        .zip(points.iter())
        .zip(eq_t_i_list.iter())
        .fold(
            iter::repeat_with(DenseMultilinearExtension::zero)
                .map(Arc::new)
                .take(point_indices.len())
                .collect::<Vec<_>>(),
            |mut merged_tilde_gs, ((poly, point), coeff)| {
                *Arc::make_mut(&mut merged_tilde_gs[point_indices[point]]) +=
                    (*coeff, poly.deref());
                merged_tilde_gs
            },
        );
    end_timer!(timer);

    let timer = start_timer!(|| format!("compute tilde eq for {} points", points.len()));
    let tilde_eqs: Vec<_> = deduped_points
        .iter()
        .map(|point| {
            let eq_b_zi = build_eq_x_r_vec(point).unwrap();
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_var, eq_b_zi,
            ))
        })
        .collect();
    end_timer!(timer);

    // built the virtual polynomial for SumCheck
    let timer = start_timer!(|| format!("sum check prove of {} variables", num_var));

    let step = start_timer!(|| "add mle");
    let mut sum_check_vp = VirtualPolynomial::new(num_var);
    for (merged_tilde_g, tilde_eq) in merged_tilde_gs.iter().zip(tilde_eqs.into_iter()) {
        sum_check_vp.add_mle_list([merged_tilde_g.clone(), tilde_eq], E::ScalarField::one())?;
    }
    end_timer!(step);

    let proof = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(
        &sum_check_vp,
        transcript,
    ) {
        Ok(p) => p,
        Err(_e) => {
            // cannot wrap IOPError with PCSError due to cyclic dependency
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch proving Failed".to_string(),
            ));
        },
    };

    end_timer!(timer);

    // a2 := sumcheck's point
    let a2 = &proof.point[..num_var];

    // build g'(X) = \sum_i=1..k \tilde eq_i(a2) * \tilde g_i(X) where (a2) is the
    // sumcheck's point \tilde eq_i(a2) = eq(a2, point_i)
    let step = start_timer!(|| "evaluate at a2");
    let mut g_prime = Arc::new(DenseMultilinearExtension::zero());
    for (merged_tilde_g, point) in merged_tilde_gs.iter().zip(deduped_points.iter()) {
        let eq_i_a2 = eq_eval(a2, point)?;
        *Arc::make_mut(&mut g_prime) += (eq_i_a2, merged_tilde_g.deref());
    }
    end_timer!(step);

    let step = start_timer!(|| "pcs open");
    let (g_prime_proof, _g_prime_eval) =
        MultilinearKzgPCS::<E>::open(prover_param, &g_prime, a2.to_vec().as_ref())?;
    // assert_eq!(g_prime_eval, tilde_g_eval);
    end_timer!(step);

    let step = start_timer!(|| "evaluate fi(pi)");
    end_timer!(step);
    end_timer!(open_timer);

    Ok(MultilinearKzgBatchProof {
        sum_check_proof: proof,
        f_i_eval_at_point_i: evals.to_vec(),
        g_prime_proof,
    })
}

/// Steps:
/// 1. get challenge point t from transcript
/// 2. build g' commitment
/// 3. ensure \sum_i eq(a2, point_i) * eq(t, <i>) * f_i_evals matches the sum
///    via SumCheck verification 4. verify commitment
pub(crate) fn batch_verify_internal<E>(
    verifier_param: &MultilinearVerifierParam<E>,
    f_i_commitments: &[Commitment<E>],
    points: &[Vec<E::ScalarField>],
    proof: &MultilinearKzgBatchProof<E>,
    transcript: &mut HyperTranscript<E::ScalarField>,
) -> Result<bool, PCSError>
where
    E: Pairing,
{
    let open_timer = start_timer!(|| "batch verification");

    // TODO: sanity checks

    let k = f_i_commitments.len();
    let ell = log2(k) as usize;
    let num_var = proof.sum_check_proof.point.len();

    // challenge point t
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

    // sum check point (a2)
    let a2 = &proof.sum_check_proof.point[..num_var];

    // build g' commitment
    let step = start_timer!(|| "build homomorphic commitment");
    let eq_t_list = build_eq_x_r_vec(t.as_ref())?;

    let mut scalars = vec![];
    let mut bases = vec![];

    for (i, point) in points.iter().enumerate() {
        let eq_i_a2 = eq_eval(a2, point)?;
        scalars.push(eq_i_a2 * eq_t_list[i]);
        bases.push(f_i_commitments[i]);
    }
    let g_prime_commit = E::G1::msm_unchecked(&bases, &scalars);
    end_timer!(step);

    // ensure \sum_i eq(t, <i>) * f_i_evals matches the sum via SumCheck
    let mut sum = E::ScalarField::zero();
    for (i, &e) in eq_t_list.iter().enumerate().take(k) {
        sum += e * proof.f_i_eval_at_point_i[i];
    }
    let aux_info = VPAuxInfo {
        max_degree: 2,
        num_variables: num_var,
        phantom: PhantomData,
    };
    let subclaim = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
        sum,
        &proof.sum_check_proof,
        &aux_info,
        transcript,
    ) {
        Ok(p) => p,
        Err(_e) => {
            // cannot wrap IOPError with PCSError due to cyclic dependency
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch verification failed".to_string(),
            ));
        },
    };
    let tilde_g_eval = subclaim.expected_evaluation;

    // verify commitment
    let res = MultilinearKzgPCS::<E>::verify(
        verifier_param,
        &g_prime_commit.into_affine(),
        a2.to_vec().as_ref(),
        &tilde_g_eval,
        &proof.g_prime_proof,
    )?;

    end_timer!(open_timer);
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::{
        super::{util::get_batched_nv, *},
        *,
    };
    use crate::pcs::StructuredReferenceString;
    use ark_bls12_381::Bls12_381 as E;
    use ark_ec::pairing::Pairing;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{rand::Rng, vec::Vec, UniformRand};
    use jf_utils::test_rng;
    type Fr = <E as Pairing>::ScalarField;

    // fn test_batch_commit_helper<R: RngCore + CryptoRng>(
    //     uni_params: &UnivariateUniversalParams<E>,
    //     ml_params: &MultilinearUniversalParams<E>,
    //     polys: &[MLE<Fr>],
    //     rng: &mut R,
    // ) -> Result<(), PCSError> {
    //     let merged_nv = get_batched_nv(polys[0].num_vars(), polys.len());
    //     let qx_degree = compute_qx_degree(merged_nv, polys.len());
    //     let padded_qx_degree = 1usize << log2(qx_degree);

    //     let (uni_ck, uni_vk) = uni_params.trim(padded_qx_degree)?;
    //     let (ml_ck, ml_vk) = ml_params.trim(merged_nv)?;

    //     let mut points = Vec::new();
    //     for poly in polys.iter() {
    //         let point = (0..poly.num_vars())
    //             .map(|_| Fr::rand(rng))
    //             .collect::<Vec<Fr>>();
    //         points.push(point);
    //     }

    //     let evals = generate_evaluations(polys, &points)?;

    //     let com = MultilinearKzgPCS::batch_commit(&(ml_ck.clone(), uni_ck.clone()), polys)?;
    //     let (batch_proof, evaluations) =
    //         batch_open_internal(&uni_ck, &ml_ck, polys, &com, &points)?;

    //     for (a, b) in evals.iter().zip(evaluations.iter()) {
    //         assert_eq!(a, b)
    //     }

    //     // good path
    //     assert!(batch_verify_internal(
    //         &uni_vk,
    //         &ml_vk,
    //         &com,
    //         &points,
    //         &evaluations,
    //         &batch_proof,
    //     )?);

    //     // bad commitment
    //     assert!(!batch_verify_internal(
    //         &uni_vk,
    //         &ml_vk,
    //         &Commitment(<E as Pairing>::G1Affine::default()),
    //         &points,
    //         &evaluations,
    //         &batch_proof,
    //     )?);

    //     // bad points
    //     assert!(
    //         batch_verify_internal(&uni_vk, &ml_vk, &com, &points[1..], &[], &batch_proof,).is_err()
    //     );

    //     // bad proof
    //     assert!(batch_verify_internal(
    //         &uni_vk,
    //         &ml_vk,
    //         &com,
    //         &points,
    //         &evaluations,
    //         &MultilinearKzgBatchProof {
    //             proof: MultilinearKzgProof { proofs: Vec::new() },
    //             q_x_commit: Commitment(<E as Pairing>::G1Affine::default()),
    //             q_x_opens: vec![],
    //         },
    //     )
    //     .is_err());

    //     // bad value
    //     let mut wrong_evals = evaluations.clone();
    //     wrong_evals[0] = Fr::default();
    //     assert!(!batch_verify_internal(
    //         &uni_vk,
    //         &ml_vk,
    //         &com,
    //         &points,
    //         &wrong_evals,
    //         &batch_proof
    //     )?);

    //     // bad q(x) commit
    //     let mut wrong_proof = batch_proof;
    //     wrong_proof.q_x_commit = Commitment(<E as Pairing>::G1Affine::default());
    //     assert!(!batch_verify_internal(
    //         &uni_vk,
    //         &ml_vk,
    //         &com,
    //         &points,
    //         &evaluations,
    //         &wrong_proof,
    //     )?);
    //     Ok(())
    // }

    // #[test]
    // fn test_batch_commit_internal() -> Result<(), PCSError> {
    //     let mut rng = test_rng();

    //     let uni_params =
    //         UnivariateUniversalParams::<E>::gen_srs_for_testing(&mut rng, 1usize << 15)?;
    //     let ml_params = MultilinearUniversalParams::<E>::gen_srs_for_testing(&mut rng, 15)?;

    //     // normal polynomials
    //     let polys1: Vec<_> = (0..5)
    //         .map(|_| DenseMultilinearExtension::rand(4, &mut rng))
    //         .collect();
    //     test_batch_commit_helper(&uni_params, &ml_params, &polys1, &mut rng)?;

    //     // single-variate polynomials
    //     let polys1: Vec<_> = (0..5)
    //         .map(|_| DenseMultilinearExtension::rand(1, &mut rng))
    //         .collect();
    //     test_batch_commit_helper(&uni_params, &ml_params, &polys1, &mut rng)?;

    //     Ok(())
    // }

    fn test_multi_open_helper<R: Rng>(
        ml_params: &MultilinearUniversalParams<E>,
        polys: &[Arc<DenseMultilinearExtension<Fr>>],
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let merged_nv = get_batched_nv(polys[0].num_vars(), polys.len());
        let (ml_ck, ml_vk) = ml_params.trim(merged_nv)?;

        let mut points = Vec::new();
        for poly in polys.iter() {
            let point = (0..poly.num_vars())
                .map(|_| Fr::rand(rng))
                .collect::<Vec<Fr>>();
            points.push(point);
        }

        let evals = polys
            .iter()
            .zip(points.iter())
            .map(|(f, p)| f.evaluate(p).unwrap())
            .collect::<Vec<_>>();

        let commitments = polys
            .iter()
            .map(|poly| MultilinearKzgPCS::commit(&ml_ck, poly).unwrap())
            .collect::<Vec<_>>();

        let mut transcript = HyperTranscript::new("test transcript".as_ref());
        transcript.append_field_element("init".as_ref(), &Fr::zero())?;

        let batch_proof =
            multi_open_internal::<E>(&ml_ck, polys, &points, &evals, &mut transcript)?;

        // good path
        let mut transcript = HyperTranscript::new("test transcript".as_ref());
        transcript.append_field_element("init".as_ref(), &Fr::zero())?;
        assert!(batch_verify_internal::<E>(
            &ml_vk,
            &commitments,
            &points,
            &batch_proof,
            &mut transcript
        )?);

        Ok(())
    }

    #[test]
    fn test_multi_open_internal() -> Result<(), PCSError> {
        let mut rng = test_rng();

        let ml_params = MultilinearUniversalParams::<E>::gen_srs_for_testing(&mut rng, 20)?;
        for num_poly in 5..6 {
            for nv in 15..16 {
                let polys1: Vec<_> = (0..num_poly)
                    .map(|_| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
                    .collect();
                test_multi_open_helper(&ml_params, &polys1, &mut rng)?;
            }
        }

        Ok(())
    }
}
