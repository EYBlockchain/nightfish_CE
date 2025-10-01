// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

use super::ipa_structs::{
    eval_merged_lookup_witness, eval_merged_table, Challenges, Proof, ScalarsAndBases, VerifyingKey,
};
use crate::{
    constants::EXTRA_TRANSCRIPT_MSG_LABEL,
    errors::{PlonkError, SnarkError::ParameterError},
    nightfall::ipa_structs::{MapKey, VerificationKeyId},
    transcript::*,
};

use ark_ec::{short_weierstrass::Affine, AffineRepr, CurveGroup};
use ark_ff::{Field, One, Zero};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Polynomial,
    Radix2EvaluationDomain,
};
use ark_std::{collections::BTreeMap, format, string::ToString, vec, vec::Vec};
use core::ops::Neg;

use jf_primitives::{
    pcs::{PolynomialCommitmentScheme, StructuredReferenceString},
    rescue::RescueParameter,
};
use jf_relation::{
    constants::GATE_WIDTH,
    gadgets::{ecc::HasTEForm, EmulationConfig},
};
use jf_utils::par_utils::parallelizable_slice_iter;

use rayon::prelude::*;

/// (Aggregated) polynomial commitment evaluation info.
/// * `u` - a random combiner that was used to combine evaluations at point
///   `eval_point` and `next_eval_point`.
/// * `eval_point` - the point to be evaluated at.
/// * `next_eval_point` - the shifted point to be evaluated at.
/// * `eval` - the (aggregated) polynomial evaluation value.
/// * `comm_scalars_and_bases` - the scalars-and-bases form of the (aggregated)
///   polynomial commitment.
/// * `opening_proof` - (aggregated) proof of evaluations at point `eval_point`.
/// * `shifted_opening_proof` - (aggregated) proof of evaluations at point
///   `next_eval_point`.
#[derive(Debug, Clone)]
pub struct PcsInfo<PCS>
where
    PCS: PolynomialCommitmentScheme,
{
    pub u: PCS::Evaluation,
    pub comm_scalars_and_bases: ScalarsAndBases<PCS>,
    pub opening_proof: PCS::Proof,
}

impl<PCS: PolynomialCommitmentScheme> Default for PcsInfo<PCS> {
    fn default() -> Self {
        Self {
            u: PCS::Evaluation::default(),
            comm_scalars_and_bases: ScalarsAndBases::<PCS>::new(),
            opening_proof: PCS::Proof::default(),
        }
    }
}

pub(crate) struct FFTVerifier<PCS: PolynomialCommitmentScheme> {
    pub(crate) domain: Radix2EvaluationDomain<PCS::Evaluation>,
}

/// Function used to reproduce the end state of a transcript, used in recursive proving and verification.
pub fn reproduce_transcript<PCS, E, F, T>(
    vk_id: Option<VerificationKeyId>,
    public_input: E::ScalarField,
    proof: &Proof<PCS>,
) -> Result<T, PlonkError>
where
    PCS: PolynomialCommitmentScheme<
        Evaluation = E::ScalarField,
        Polynomial = DensePolynomial<E::ScalarField>,
        Point = E::ScalarField,
        Commitment = Affine<E>,
        Proof: TranscriptVisitor,
    >,
    F: RescueParameter,
    E: HasTEForm<BaseField = F>,
    E::ScalarField: EmulationConfig<F>,
    PCS::SRS: StructuredReferenceString<Item = Affine<E>>,
    T: Transcript,
{
    let mut transcript = T::new_transcript(b"PlonkProof");
    if let Some(id) = vk_id {
        transcript.push_message(b"vk_id", &E::ScalarField::from(id as u8))?;
    }
    transcript.push_message(b"public_input", &public_input)?;

    transcript.append_curve_points(b"witness_poly_comms", &proof.wires_poly_comms)?;

    let _ = transcript.squeeze_scalar_challenge::<E>(b"tau")?;

    if let Some(proof_lkup) = proof.plookup_proof.as_ref() {
        transcript.append_curve_points(b"h_poly_comms", &proof_lkup.h_poly_comms)?;
    }

    let _: [E::ScalarField; 2] = transcript
        .squeeze_scalar_challenges::<E>(b"beta gamma", 2)?
        .try_into()
        .map_err(|_| {
            PlonkError::InvalidParameters("Couldn't convert to fixed length array".to_string())
        })?;

    transcript.append_curve_point(b"perm_poly_comms", &proof.prod_perm_poly_comm)?;

    if let Some(proof_lkup) = proof.plookup_proof.as_ref() {
        transcript.append_curve_point(b"plookup_poly_comms", &proof_lkup.prod_lookup_poly_comm)?;
    }

    let _ = transcript.squeeze_scalar_challenge::<E>(b"alpha")?;

    transcript.append_curve_points(b"quot_poly_comms", &proof.split_quot_poly_comms)?;
    let _ = transcript.squeeze_scalar_challenge::<E>(b"zeta")?;

    transcript.append_visitor(&proof.poly_evals)?;

    if let Some(proof_lkup) = proof.plookup_proof.as_ref() {
        transcript.append_visitor(&proof_lkup.poly_evals)?;
    }

    let _ = transcript.squeeze_scalar_challenge::<E>(b"v")?;

    transcript.append_curve_point(b"q_comm", &proof.q_comm)?;

    let _ = transcript.squeeze_scalar_challenge::<E>(b"u")?;

    // As we continue to use the transcript in the recursive setting,
    // we need to append the opening proof to the transcript.
    transcript.append_visitor(&proof.opening_proof)?;
    Ok(transcript)
}

impl<E, F, PCS> FFTVerifier<PCS>
where
    PCS: PolynomialCommitmentScheme<
        Evaluation = E::ScalarField,
        Polynomial = DensePolynomial<E::ScalarField>,
        Point = E::ScalarField,
        Commitment = Affine<E>,
    >,
    F: RescueParameter,
    E: HasTEForm<BaseField = F>,
    E::ScalarField: EmulationConfig<F>,
    PCS::SRS: StructuredReferenceString<Item = Affine<E>>,
{
    /// Construct a Plonk verifier that uses a domain with size `domain_size`.
    pub(crate) fn new(domain_size: usize) -> Result<Self, PlonkError> {
        let domain = Radix2EvaluationDomain::<E::ScalarField>::new(domain_size)
            .ok_or(PlonkError::DomainCreationError)?;
        Ok(Self { domain })
    }

    /// Prepare the (aggregated) polynomial commitment evaluation information.
    pub(crate) fn prepare_pcs_info<T>(
        &self,
        verify_key: &VerifyingKey<PCS>,
        public_inputs: &[E::ScalarField],
        proof: &Proof<PCS>,
        extra_transcript_init_msg: &Option<Vec<u8>>,
    ) -> Result<PcsInfo<PCS>, PlonkError>
    where
        T: Transcript,
    {
        if public_inputs.len() != verify_key.num_inputs {
            return Err(ParameterError(format!(
                "the circuit public input length {} != the verification key public input length {}",
                public_inputs.len(),
                verify_key.num_inputs
            ))
            .into());
        }
        if verify_key.plookup_vk.is_some() != proof.plookup_proof.is_some() {
            return Err(ParameterError(
                "Mismatched proof type and verification key type".to_string(),
            )
            .into());
        }
        if verify_key.domain_size != self.domain.size() {
            return Err(ParameterError(format!(
                "the domain size of the verification key is different from {}",
                self.domain.size(),
            ))
            .into());
        }

        // compute challenges except 'u' and evaluations
        let challenges = Self::compute_challenges::<T>(
            verify_key,
            public_inputs,
            proof,
            extra_transcript_init_msg,
        )?;

        // pre-compute alpha related values
        let alpha_2 = challenges.alpha.square();
        let alpha_3 = alpha_2 * challenges.alpha;
        let alpha_4 = alpha_2 * alpha_2;
        let alpha_5 = alpha_2 * alpha_3;
        let alpha_6 = alpha_4 * alpha_2;
        // let alpha_7 = alpha_3 * alpha_4;
        let alpha_powers = vec![alpha_2, alpha_3, alpha_4, alpha_5, alpha_6];

        let vanish_eval = self.evaluate_vanishing_poly(&challenges.zeta);
        let (lagrange_1_eval, lagrange_n_eval) =
            self.evaluate_lagrange_1_and_n(&challenges.zeta, &vanish_eval);

        // compute the constant term of the linearization polynomial
        let lin_poly_constant = self.compute_lin_poly_constant_term(
            &challenges,
            verify_key,
            public_inputs,
            proof,
            &vanish_eval,
            &lagrange_1_eval,
            &lagrange_n_eval,
            &alpha_powers,
        )?;

        let mut comms_and_eval_points = BTreeMap::<MapKey<PCS>, Vec<PCS::Commitment>>::new();

        let zeta = challenges.zeta;
        let zeta_omega = self.domain.group_gen * zeta;

        // First we compute the polynomial z(x)
        let z_1_poly = DensePolynomial::from_coefficients_slice(&[zeta.neg(), E::ScalarField::ONE]);
        let z_2_poly =
            DensePolynomial::from_coefficients_slice(&[zeta_omega.neg(), E::ScalarField::ONE]);

        let z_poly = &z_1_poly * &z_2_poly;

        let g_base_zero = PCS::SRS::g(&verify_key.open_key);

        let d1_scalars_and_bases = self.linearization_scalars_and_bases(
            verify_key,
            &challenges,
            &vanish_eval,
            &lagrange_1_eval,
            &lagrange_n_eval,
            proof,
            &alpha_powers,
        )?;

        let z_1_poly_eval = z_1_poly.evaluate(&challenges.u);
        let z_2_poly_eval = z_2_poly.evaluate(&challenges.u);
        let mut g_base_scalars = vec![lin_poly_constant * z_2_poly_eval];
        if let Some(plookup_proof) = &proof.plookup_proof {
            // First we add all the commitments opened at 'zeta'.
            let mut polys_ref_and_evals = vec![proof.wires_poly_comms[0]];
            g_base_scalars.push(-(z_2_poly_eval * proof.poly_evals.wires_evals[0]));
            polys_ref_and_evals.push(proof.wires_poly_comms[1]);
            g_base_scalars.push(-(z_2_poly_eval * proof.poly_evals.wires_evals[1]));
            polys_ref_and_evals.push(proof.wires_poly_comms[2]);
            g_base_scalars.push(-(z_2_poly_eval * proof.poly_evals.wires_evals[2]));
            polys_ref_and_evals.push(proof.wires_poly_comms[5]);
            g_base_scalars.push(-(z_2_poly_eval * proof.poly_evals.wires_evals[5]));

            let num_wire_types = proof.wires_poly_comms.len();
            for (poly_comm, poly_eval) in verify_key
                .sigma_comms
                .iter()
                .take(num_wire_types - 1)
                .zip(proof.poly_evals.wire_sigma_evals.iter())
            {
                polys_ref_and_evals.push(*poly_comm);
                g_base_scalars.push(-(z_2_poly_eval * *poly_eval));
            }

            polys_ref_and_evals.push(verify_key.plookup_vk.as_ref().unwrap().q_dom_sep_comm);
            g_base_scalars.push(-(z_2_poly_eval * plookup_proof.poly_evals.q_dom_sep_eval));

            comms_and_eval_points.insert(MapKey(1, z_1_poly), polys_ref_and_evals);

            // Now all the commitments opened only at 'zeta * omega'.
            let mut polys_ref_and_evals = vec![proof.prod_perm_poly_comm];
            g_base_scalars.push(-(z_1_poly_eval * proof.poly_evals.perm_next_eval));
            polys_ref_and_evals.push(plookup_proof.h_poly_comms[1]);
            g_base_scalars.push(-(z_1_poly_eval * plookup_proof.poly_evals.h_2_next_eval));
            polys_ref_and_evals.push(plookup_proof.prod_lookup_poly_comm);
            g_base_scalars.push(-(z_1_poly_eval * plookup_proof.poly_evals.prod_next_eval));

            comms_and_eval_points.insert(MapKey(2, z_2_poly), polys_ref_and_evals);

            // Finally we add the commitments opened at both 'zeta' and 'zeta * omega'.
            let polys_ref_and_evals =
                Self::plookup_both_open_poly_comms_and_evals(proof, verify_key)?;

            let polys_ref_and_evals_two = polys_ref_and_evals
                .par_iter()
                .map(|(comm, _)| *comm)
                .collect::<Vec<Affine<E>>>();
            polys_ref_and_evals.iter().for_each(|(_, evals)| {
                let lagrange_eval = Self::lagrange_interpolate(&[zeta, zeta_omega], evals)
                    .unwrap()
                    .evaluate(&challenges.u);
                g_base_scalars.push(-(lagrange_eval));
            });
            comms_and_eval_points.insert(MapKey(3, z_poly.clone()), polys_ref_and_evals_two);
        } else {
            let num_wire_types = proof.wires_poly_comms.len();
            let comms_iter = proof
                .wires_poly_comms
                .iter()
                .chain(verify_key.sigma_comms.iter().take(num_wire_types - 1));
            let evals_iter = proof.poly_evals.wires_evals.iter().chain(
                proof
                    .poly_evals
                    .wire_sigma_evals
                    .iter()
                    .take(num_wire_types - 1),
            );

            let polys_ref_and_evals = comms_iter
                .zip(evals_iter)
                .map(|(comm, eval)| {
                    g_base_scalars.push(-(z_2_poly_eval * eval));
                    *comm
                })
                .collect::<Vec<PCS::Commitment>>();

            comms_and_eval_points.insert(MapKey(1, z_1_poly), polys_ref_and_evals);

            // Now the grand product commitment opened at 'zeta * omega'
            comms_and_eval_points.insert(MapKey(2, z_2_poly), vec![proof.prod_perm_poly_comm]);
            g_base_scalars.push(-(z_1_poly_eval * proof.poly_evals.perm_next_eval));
        }

        let mut combiner = challenges.v;
        let mut scalars_and_bases = ScalarsAndBases::<PCS>::new();

        for (map_key, points) in comms_and_eval_points.iter() {
            let z_i_u = (&z_poly / &map_key.1).evaluate(&challenges.u);

            for &point in points.iter() {
                scalars_and_bases.push(combiner * z_i_u, point)?;

                combiner *= challenges.v;
            }
        }

        let mut combiner = E::ScalarField::one();

        let g_scalar = g_base_scalars
            .iter()
            .fold(E::ScalarField::zero(), |acc, s| {
                let val = combiner * s;
                combiner *= challenges.v;
                acc + val
            });

        let neg_z_u = z_poly.evaluate(&challenges.u).neg();
        scalars_and_bases.push(neg_z_u, proof.q_comm)?;
        scalars_and_bases.push(g_scalar, g_base_zero)?;

        scalars_and_bases.merge(challenges.u - zeta_omega, &d1_scalars_and_bases)?;

        Ok(PcsInfo {
            u: challenges.u,
            comm_scalars_and_bases: scalars_and_bases,
            opening_proof: proof.opening_proof.clone(),
        })
    }

    /// Batchly verify multiple (aggregated) PCS opening proofs.
    ///
    /// We need to verify that
    /// - `e(Ai, [x]2) = e(Bi, [1]2) for i \in {0, .., m-1}`, where
    /// - `Ai = [open_proof_i] + u_i * [shifted_open_proof_i]` and
    /// - `Bi = eval_point_i * [open_proof_i] + u_i * next_eval_point_i *
    ///   [shifted_open_proof_i] + comm_i - eval_i * [1]1`.
    ///
    /// By Schwartz-Zippel lemma, it's equivalent to check that for a random r:
    /// - `e(A0 + ... + r^{m-1} * Am, [x]2) = e(B0 + ... + r^{m-1} * Bm, [1]2)`.
    pub(crate) fn verify_opening_proofs(
        open_key: &<PCS::SRS as StructuredReferenceString>::VerifierParam,
        pcs_info: &PcsInfo<PCS>,
    ) -> Result<bool, PlonkError> {
        let g_comm = pcs_info
            .comm_scalars_and_bases
            .multi_scalar_mul()
            .into_affine();

        let result = PCS::verify(
            open_key,
            &g_comm,
            &pcs_info.u,
            &E::ScalarField::zero(),
            &pcs_info.opening_proof,
        )?;

        Ok(result)
    }

    /// Compute verifier challenges `tau`, `beta`, `gamma`, `alpha`, `zeta`,
    /// 'v', 'u'.
    #[inline]
    pub(crate) fn compute_challenges<T>(
        verify_key: &VerifyingKey<PCS>,
        public_inputs: &[E::ScalarField],
        proof: &Proof<PCS>,
        extra_transcript_init_msg: &Option<Vec<u8>>,
    ) -> Result<Challenges<E::ScalarField>, PlonkError>
    where
        T: Transcript,
    {
        let mut transcript = T::new_transcript(b"PlonkProof");
        if let Some(msg) = extra_transcript_init_msg {
            transcript.push_message(EXTRA_TRANSCRIPT_MSG_LABEL, msg)?;
        }

        if verify_key.id.is_some() {
            transcript.append_visitor(verify_key)?;
        }

        for pub_input in public_inputs.iter() {
            transcript.push_message(b"public_input", pub_input)?;
        }

        transcript.append_curve_points(b"witness_poly_comms", &proof.wires_poly_comms)?;

        let tau = transcript.squeeze_scalar_challenge::<E>(b"tau")?;

        if let Some(proof_lkup) = proof.plookup_proof.as_ref() {
            transcript.append_curve_points(b"h_poly_comms", &proof_lkup.h_poly_comms)?;
        }

        let [beta, gamma]: [E::ScalarField; 2] = transcript
            .squeeze_scalar_challenges::<E>(b"beta gamma", 2)?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters("Couldn't convert to fixed length array".to_string())
            })?;

        transcript.append_curve_point(b"perm_poly_comms", &proof.prod_perm_poly_comm)?;

        if let Some(proof_lkup) = proof.plookup_proof.as_ref() {
            transcript
                .append_curve_point(b"plookup_poly_comms", &proof_lkup.prod_lookup_poly_comm)?;
        }

        let alpha = transcript.squeeze_scalar_challenge::<E>(b"alpha")?;

        transcript.append_curve_points(b"quot_poly_comms", &proof.split_quot_poly_comms)?;
        let zeta = transcript.squeeze_scalar_challenge::<E>(b"zeta")?;

        transcript.append_visitor(&proof.poly_evals)?;

        if let Some(proof_lkup) = proof.plookup_proof.as_ref() {
            transcript.append_visitor(&proof_lkup.poly_evals)?;
        }

        let v = transcript.squeeze_scalar_challenge::<E>(b"v")?;

        transcript.append_curve_point(b"q_comm", &proof.q_comm)?;

        let u = transcript.squeeze_scalar_challenge::<E>(b"u")?;
        Ok(Challenges {
            tau,
            alpha,
            beta,
            gamma,
            zeta,
            v,
            u,
        })
    }

    /// Compute the constant term of the linearization polynomial:
    /// For each instance j:
    ///
    /// r_plonk_j = PI - L1(x) * alpha^2 -
    ///             alpha * \prod_i=1..m-1 (w_{j,i} + beta * sigma_{j,i} +
    /// gamma) * (w_{j,m} + gamma) * z_j(xw)
    ///
    /// r_lookup_j = alpha^3 * Ln(x) * (h1_x_j - h2_wx_j) -
    ///              alpha^4 * L1(x) * alpha -
    ///              alpha^5 * Ln(x) -
    ///              alpha^6 * (x - g^{n-1}) * prod_poly_wx_j * [gamma(1+beta) +
    /// h1_x_j + beta * h1_wx_j] * [gamma(1+beta) + beta * h2_wx_j]
    ///
    /// r_0 = \sum_{j=1..m} alpha^{k_j} * (r_plonk_j + (r_lookup_j))
    /// where m is the number of instances, and k_j is the number of alpha power
    /// terms added to the first j-1 instances.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_lin_poly_constant_term(
        &self,
        challenges: &Challenges<E::ScalarField>,
        verify_key: &VerifyingKey<PCS>,
        public_inputs: &[E::ScalarField],
        proof: &Proof<PCS>,
        vanish_eval: &E::ScalarField,
        lagrange_1_eval: &E::ScalarField,
        lagrange_n_eval: &E::ScalarField,
        alpha_powers: &[E::ScalarField],
    ) -> Result<E::ScalarField, PlonkError> {
        let mut result = E::ScalarField::zero();

        let mut tmp = self.evaluate_pi_poly(
            public_inputs,
            &challenges.zeta,
            vanish_eval,
            verify_key.is_merged,
        )? - alpha_powers[0] * lagrange_1_eval;
        let num_wire_types = GATE_WIDTH
            + 1
            + match proof.plookup_proof.is_some() {
                true => 1,
                false => 0,
            };
        let first_w_evals = &proof.poly_evals.wires_evals[..num_wire_types - 1];
        let last_w_eval = &proof.poly_evals.wires_evals[num_wire_types - 1];
        let sigma_evals = &proof.poly_evals.wire_sigma_evals[..];
        tmp -= first_w_evals.iter().zip(sigma_evals.iter()).fold(
            challenges.alpha * proof.poly_evals.perm_next_eval * (challenges.gamma + last_w_eval),
            |acc, (w_eval, sigma_eval)| {
                acc * (challenges.gamma + w_eval + challenges.beta * sigma_eval)
            },
        );

        if let Some(proof_lk) = &proof.plookup_proof {
            let gamma_mul_beta_plus_one =
                challenges.gamma * (E::ScalarField::one() + challenges.beta);
            let evals = &proof_lk.poly_evals;

            let plookup_constant = *lagrange_n_eval
                * (evals.h_1_eval - evals.h_2_next_eval - alpha_powers[0])
                - challenges.alpha * lagrange_1_eval
                - alpha_powers[1]
                    * (challenges.zeta - self.domain.group_gen_inv)
                    * evals.prod_next_eval
                    * (gamma_mul_beta_plus_one
                        + evals.h_1_eval
                        + challenges.beta * evals.h_1_next_eval)
                    * (gamma_mul_beta_plus_one + challenges.beta * evals.h_2_next_eval);

            tmp += alpha_powers[1] * plookup_constant;
        }

        result += tmp;

        Ok(result)
    }

    /// Compute the bases and scalars in the batched polynomial commitment,
    /// which is a generalization of `[D]1` specified in Sec 8.3, Verifier
    /// algorithm step 9 of https://eprint.iacr.org/2019/953.pdf.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linearization_scalars_and_bases(
        &self,
        vk: &VerifyingKey<PCS>,
        challenges: &Challenges<E::ScalarField>,
        vanish_eval: &E::ScalarField,
        lagrange_1_eval: &E::ScalarField,
        lagrange_n_eval: &E::ScalarField,
        proof: &Proof<PCS>,
        alpha_powers: &[E::ScalarField],
    ) -> Result<ScalarsAndBases<PCS>, PlonkError> {
        // compute constants that are being reused
        let beta_plus_one = E::ScalarField::one() + challenges.beta;
        let gamma_mul_beta_plus_one = beta_plus_one * challenges.gamma;

        let mut scalars_and_bases = ScalarsAndBases::new();

        // Compute coefficient for the permutation product polynomial commitment.
        // coeff = L1(zeta) * alpha^2
        //       + alpha
        //       * (beta * zeta      + a_bar + gamma)
        //       * (beta * k1 * zeta + b_bar + gamma)
        //       * (beta * k2 * zeta + c_bar + gamma)
        // where a_bar, b_bar and c_bar are in w_evals
        let mut coeff = alpha_powers[0] * lagrange_1_eval;
        let w_evals = &proof.poly_evals.wires_evals;
        coeff += w_evals
            .iter()
            .zip(vk.k.iter())
            .fold(challenges.alpha, |acc, (w_eval, k)| {
                acc * (challenges.beta * k * challenges.zeta + challenges.gamma + w_eval)
            });

        // Add permutation product polynomial commitment.
        scalars_and_bases.push(coeff, proof.prod_perm_poly_comm)?;

        // Compute coefficient for the last wire sigma polynomial commitment.
        let num_wire_types = proof.wires_poly_comms.len();
        let sigma_evals = &proof.poly_evals.wire_sigma_evals;
        let coeff = w_evals
            .iter()
            .take(num_wire_types - 1)
            .zip(sigma_evals.iter())
            .fold(
                challenges.alpha * challenges.beta * proof.poly_evals.perm_next_eval,
                |acc, (w_eval, sigma_eval)| {
                    acc * (challenges.beta * sigma_eval + challenges.gamma + w_eval)
                },
            );

        // Add output wire sigma polynomial commitment.
        scalars_and_bases.push(
            -coeff,
            *vk.sigma_comms.last().ok_or(PlonkError::IndexError)?,
        )?;

        // Add selector polynomial commitments.
        // Compute coefficients for selector polynomial commitments.
        // The order: q_lc, q_mul, q_hash, q_o, q_c, q_ecc, q_x, q_x2, q_y, q_y2
        // TODO(binyi): get the order from a function.
        let mut q_scalars = [E::ScalarField::zero(); 2 * GATE_WIDTH + 9];
        q_scalars[0] = w_evals[0];
        q_scalars[1] = w_evals[1];
        q_scalars[2] = w_evals[2];
        q_scalars[3] = w_evals[3];
        q_scalars[4] = w_evals[0] * w_evals[1];
        q_scalars[5] = w_evals[2] * w_evals[3];
        q_scalars[6] = w_evals[0].pow([5]);
        q_scalars[7] = w_evals[1].pow([5]);
        q_scalars[8] = w_evals[2].pow([5]);
        q_scalars[9] = w_evals[3].pow([5]);
        q_scalars[10] = -w_evals[4];
        q_scalars[11] = E::ScalarField::one();
        q_scalars[12] = w_evals[0] * w_evals[1] * w_evals[2] * w_evals[3] * w_evals[4];
        q_scalars[13] = w_evals[0] * w_evals[3] * w_evals[2] * w_evals[3]
            + w_evals[1] * w_evals[2] * w_evals[2] * w_evals[3];
        q_scalars[14] = w_evals[0] * w_evals[2]
            + w_evals[1] * w_evals[3]
            + E::ScalarField::from(2u8) * w_evals[0] * w_evals[3]
            + E::ScalarField::from(2u8) * w_evals[1] * w_evals[2];
        q_scalars[15] = w_evals[2] * w_evals[2] * w_evals[3] * w_evals[3];
        q_scalars[16] = w_evals[0] * w_evals[0] * w_evals[1] + w_evals[0] * w_evals[1] * w_evals[1];

        for (&s, poly) in q_scalars.iter().zip(vk.selector_comms.iter()) {
            scalars_and_bases.push(s, *poly)?;
        }

        // Add splitted quotient commitments
        let zeta_to_n_plus_2 =
            (E::ScalarField::one() + vanish_eval) * challenges.zeta * challenges.zeta;
        let mut coeff = vanish_eval.neg();
        scalars_and_bases.push(
            coeff,
            *proof
                .split_quot_poly_comms
                .first()
                .ok_or(PlonkError::IndexError)?,
        )?;
        for poly in proof.split_quot_poly_comms.iter().skip(1) {
            coeff *= zeta_to_n_plus_2;
            scalars_and_bases.push(coeff, *poly)?;
        }

        // Add Plookup related commitments
        if let Some(lookup_proof) = proof.plookup_proof.as_ref() {
            let lookup_evals = &lookup_proof.poly_evals;
            let merged_lookup_x = eval_merged_lookup_witness::<E>(
                challenges.tau,
                w_evals[5],
                w_evals[0],
                w_evals[1],
                w_evals[2],
                lookup_evals.q_lookup_eval,
                lookup_evals.q_dom_sep_eval,
            );
            let merged_table_x = eval_merged_table::<E>(
                challenges.tau,
                lookup_evals.range_table_eval,
                lookup_evals.key_table_eval,
                lookup_evals.q_lookup_eval,
                w_evals[3],
                w_evals[4],
                lookup_evals.table_dom_sep_eval,
            );
            let merged_table_xw = eval_merged_table::<E>(
                challenges.tau,
                lookup_evals.range_table_next_eval,
                lookup_evals.key_table_next_eval,
                lookup_evals.q_lookup_next_eval,
                lookup_evals.w_3_next_eval,
                lookup_evals.w_4_next_eval,
                lookup_evals.table_dom_sep_next_eval,
            );

            // coefficient for prod_lookup_poly(X):
            // coeff_lin_poly = alpha^4 * L1(x) +
            //                  alpha^5 * Ln(x) +
            //                  alpha^6 * (x - w^{n-1}) * (1+beta) * (gamma + lookup_w_eval)
            //                  * (gamma(1+beta) + table_x + beta * table_xw),
            let coeff = alpha_powers[2] * lagrange_1_eval
                + alpha_powers[3] * lagrange_n_eval
                + alpha_powers[4]
                    * (challenges.zeta - self.domain.group_gen_inv)
                    * beta_plus_one
                    * (challenges.gamma + merged_lookup_x)
                    * (gamma_mul_beta_plus_one
                        + merged_table_x
                        + challenges.beta * merged_table_xw);
            scalars_and_bases.push(coeff, lookup_proof.prod_lookup_poly_comm)?;

            // coefficient for h2(X):
            // coeff_lin_poly = alpha_base * alpha^6 * (w^{n-1} - x)
            //                  * prod_lookup_poly_xw
            //                  * [gamma(1+beta) + h1_x + beta * h1_xw]
            let coeff = alpha_powers[4]
                * (self.domain.group_gen_inv - challenges.zeta)
                * lookup_evals.prod_next_eval
                * (gamma_mul_beta_plus_one
                    + lookup_evals.h_1_eval
                    + challenges.beta * lookup_evals.h_1_next_eval);
            scalars_and_bases.push(coeff, lookup_proof.h_poly_comms[1])?;
        }

        Ok(scalars_and_bases)
    }
}

type CommitsAndPoints<PCS> = Vec<(
    <PCS as PolynomialCommitmentScheme>::Commitment,
    Vec<<PCS as PolynomialCommitmentScheme>::Evaluation>,
)>;

/// Helper methods
impl<E, F, PCS> FFTVerifier<PCS>
where
    E: HasTEForm<BaseField = F>,
    F: RescueParameter,
    PCS: PolynomialCommitmentScheme<
        Evaluation = E::ScalarField,
        Polynomial = DensePolynomial<E::ScalarField>,
    >,
    PCS::Commitment: AffineRepr<Config = E, BaseField = F, ScalarField = E::ScalarField>,
{
    /// Evaluate vanishing polynomial at point `zeta`
    #[inline]
    pub(crate) fn evaluate_vanishing_poly(&self, zeta: &E::ScalarField) -> E::ScalarField {
        self.domain.evaluate_vanishing_polynomial(*zeta)
    }

    /// Evaluate the first and the last lagrange polynomial at point `zeta`
    /// given the vanishing polynomial evaluation `vanish_eval`.
    #[inline]
    pub(crate) fn evaluate_lagrange_1_and_n(
        &self,
        zeta: &E::ScalarField,
        vanish_eval: &E::ScalarField,
    ) -> (E::ScalarField, E::ScalarField) {
        let divisor =
            E::ScalarField::from(self.domain.size() as u32) * (*zeta - E::ScalarField::one());
        let lagrange_1_eval = *vanish_eval / divisor;
        let divisor =
            E::ScalarField::from(self.domain.size() as u32) * (*zeta - self.domain.group_gen_inv);
        let lagrange_n_eval = *vanish_eval * self.domain.group_gen_inv / divisor;
        (lagrange_1_eval, lagrange_n_eval)
    }

    /// Return the list of polynomials commitments to be opened at points 'zeta' and 'zeta * omega'
    /// Together with their evaluations at both points.
    fn plookup_both_open_poly_comms_and_evals(
        proof: &Proof<PCS>,
        vk: &VerifyingKey<PCS>,
    ) -> Result<CommitsAndPoints<PCS>, PlonkError> {
        Ok(vec![
            (
                vk.plookup_vk.as_ref().unwrap().range_table_comm,
                vec![
                    proof
                        .plookup_proof
                        .as_ref()
                        .unwrap()
                        .poly_evals
                        .range_table_eval,
                    proof
                        .plookup_proof
                        .as_ref()
                        .unwrap()
                        .poly_evals
                        .range_table_next_eval,
                ],
            ),
            (
                vk.plookup_vk.as_ref().unwrap().key_table_comm,
                vec![
                    proof
                        .plookup_proof
                        .as_ref()
                        .unwrap()
                        .poly_evals
                        .key_table_eval,
                    proof
                        .plookup_proof
                        .as_ref()
                        .unwrap()
                        .poly_evals
                        .key_table_next_eval,
                ],
            ),
            (
                proof.plookup_proof.as_ref().unwrap().h_poly_comms[0],
                vec![
                    proof.plookup_proof.as_ref().unwrap().poly_evals.h_1_eval,
                    proof
                        .plookup_proof
                        .as_ref()
                        .unwrap()
                        .poly_evals
                        .h_1_next_eval,
                ],
            ),
            (
                *vk.q_lookup_comm()?,
                vec![
                    proof
                        .plookup_proof
                        .as_ref()
                        .unwrap()
                        .poly_evals
                        .q_lookup_eval,
                    proof
                        .plookup_proof
                        .as_ref()
                        .unwrap()
                        .poly_evals
                        .q_lookup_next_eval,
                ],
            ),
            (
                proof.wires_poly_comms[3],
                vec![
                    proof.poly_evals.wires_evals[3],
                    proof
                        .plookup_proof
                        .as_ref()
                        .unwrap()
                        .poly_evals
                        .w_3_next_eval,
                ],
            ),
            (
                proof.wires_poly_comms[4],
                vec![
                    proof.poly_evals.wires_evals[4],
                    proof
                        .plookup_proof
                        .as_ref()
                        .unwrap()
                        .poly_evals
                        .w_4_next_eval,
                ],
            ),
            (
                vk.plookup_vk.as_ref().unwrap().table_dom_sep_comm,
                vec![
                    proof
                        .plookup_proof
                        .as_ref()
                        .unwrap()
                        .poly_evals
                        .table_dom_sep_eval,
                    proof
                        .plookup_proof
                        .as_ref()
                        .unwrap()
                        .poly_evals
                        .table_dom_sep_next_eval,
                ],
            ),
        ])
    }

    /// Evaluate public input polynomial at point `z`.
    /// Define the following as
    /// - H: The domain with generator g
    /// - n: The size of the domain H
    /// - Z_H: The vanishing polynomial for H.
    /// - v_i: A sequence of values, where v_i = g^i / n
    ///
    /// We then compute L_{i,H}(z) as `L_{i,H}(z) = Z_H(z) * v_i / (z - g^i)`
    /// The public input polynomial evaluation is:
    ///
    /// \sum_{i=0..l} L_{i,H}(z) * pub_input[i].
    ///
    /// For merged circuits, the evaluation is:
    /// \sum_{i=0..l/2} L_{i,H}(z) * pub_input[i] + \sum_{i=0..l/2} L_{n-i,H}(z)
    /// * pub_input[l/2+i]
    ///
    /// TODO: reuse the lagrange values
    pub(crate) fn evaluate_pi_poly(
        &self,
        pub_input: &[E::ScalarField],
        z: &E::ScalarField,
        vanish_eval: &E::ScalarField,
        circuit_is_merged: bool,
    ) -> Result<E::ScalarField, PlonkError> {
        // If z is a root of the vanishing polynomial, directly return zero.
        if vanish_eval.is_zero() {
            return Ok(E::ScalarField::zero());
        }
        let len = match circuit_is_merged {
            false => pub_input.len(),
            true => pub_input.len() / 2,
        };

        let vanish_eval_div_n = E::ScalarField::from(self.domain.size() as u32)
            .inverse()
            .ok_or(PlonkError::DivisionError)?
            * (*vanish_eval);
        let mut result = E::ScalarField::zero();
        for (i, val) in pub_input.iter().take(len).enumerate() {
            let lagrange_i =
                vanish_eval_div_n * self.domain.element(i) / (*z - self.domain.element(i));
            result += lagrange_i * val;
        }
        if circuit_is_merged {
            let n = self.domain.size();
            for (i, val) in pub_input.iter().skip(len).enumerate() {
                let lagrange_n_minus_i = vanish_eval_div_n * self.domain.element(n - i - 1)
                    / (*z - self.domain.element(n - i - 1));
                result += lagrange_n_minus_i * val;
            }
        }
        Ok(result)
    }

    #[inline]
    fn mul_poly(
        poly: &DensePolynomial<E::ScalarField>,
        coeff: &E::ScalarField,
    ) -> DensePolynomial<E::ScalarField> {
        DensePolynomial::<E::ScalarField>::from_coefficients_vec(
            parallelizable_slice_iter(&poly.coeffs)
                .map(|c| *coeff * c)
                .collect(),
        )
    }

    pub(crate) fn lagrange_interpolate(
        points: &[E::ScalarField],
        evals: &[E::ScalarField],
    ) -> Result<DensePolynomial<E::ScalarField>, PlonkError> {
        if points.len() != evals.len() {
            Err(PlonkError::InvalidParameters(
                "The number of points and evaluations should be the same".to_string(),
            ))
        } else {
            let lagrange_basis = Self::compute_lagrange_basis(points)?;
            let interpolation = lagrange_basis.iter().zip(evals.iter()).fold(
                DensePolynomial::<E::ScalarField>::zero(),
                |acc, (l_i, eval)| acc + Self::mul_poly(l_i, eval),
            );
            Ok(interpolation)
        }
    }

    fn compute_lagrange_basis(
        points: &[E::ScalarField],
    ) -> Result<Vec<DensePolynomial<E::ScalarField>>, PlonkError> {
        let monomials_vec = points
            .iter()
            .map(|point| {
                DensePolynomial::<E::ScalarField>::from_coefficients_slice(&[
                    point.neg(),
                    E::ScalarField::one(),
                ])
            })
            .collect::<Vec<DensePolynomial<E::ScalarField>>>();

        let vanishing_poly = monomials_vec.iter().fold(
            DensePolynomial::<E::ScalarField>::from_coefficients_vec(vec![E::ScalarField::one()]),
            |acc, poly| &acc * poly,
        );
        let mut lagrange_bases = Vec::<DensePolynomial<E::ScalarField>>::new();

        for (poly, point) in monomials_vec.iter().zip(points.iter()) {
            let tmp_poly = &vanishing_poly / poly;
            let scalar = tmp_poly.evaluate(point).inverse().unwrap();
            lagrange_bases.push(Self::mul_poly(&tmp_poly, &scalar));
        }

        Ok(lagrange_bases)
    }
}
