use ark_bn254::{g1::Config as BnConfig, Fq as Fq254, Fr as Fr254};
use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Polynomial};
use ark_std::{collections::BTreeMap, One, Zero};
use ark_std::{format, string::ToString, vec, vec::Vec};
use core::ops::Neg;
use jf_primitives::pcs::PolynomialCommitmentScheme;
use jf_relation::{constants::GATE_WIDTH, gadgets::ecc::PointVariable, Circuit, PlonkCircuit};
use jf_utils::fq_to_fr;

use crate::constants::EXTRA_TRANSCRIPT_MSG_LABEL;
use crate::errors::{PlonkError, SnarkError::ParameterError};
use crate::nightfall::{
    circuit::plonk_partial_verifier::{
        proof_to_var::Bn254ProofScalarsandBasesVar, Bn254OutputScalarsAndBasesVar,
        RecursiveOutputToScalarsAndBasesVar, VerifyingKeyScalarsAndBasesVar,
    },
    ipa_structs::{eval_merged_lookup_witness, eval_merged_table, Challenges, MapKey, Proof},
    ipa_verifier::FFTVerifier,
};
use crate::recursion::{circuits::Kzg, merge_functions::Bn254Output};
use crate::transcript::Transcript;

/// The vector representation of bases and corresponding scalars.
/// The bases are stored as `Variable`s while the scalars are stored in the clear.
#[derive(Debug, Clone)]
pub struct ScalarsAndBasesVar<PCS: PolynomialCommitmentScheme> {
    pub(crate) bases: Vec<PointVariable>,
    pub(crate) scalars: Vec<PCS::Evaluation>,
}

impl<PCS: PolynomialCommitmentScheme> ScalarsAndBasesVar<PCS> {
    pub(crate) fn new() -> Self {
        Self {
            bases: Vec::new(),
            scalars: Vec::new(),
        }
    }
    /// Insert a base point and the corresponding scalar.
    pub(crate) fn push(&mut self, scalar: PCS::Evaluation, base: PointVariable) {
        self.bases.push(base);
        self.scalars.push(scalar);
    }

    /// Add a list of scalars and bases into self, where each scalar is
    /// multiplied by a constant c.
    pub(crate) fn merge(&mut self, c: PCS::Evaluation, scalars_and_bases: &Self) {
        for (base, scalar) in scalars_and_bases
            .bases
            .iter()
            .zip(scalars_and_bases.scalars.iter())
        {
            self.push(c * scalar, *base);
        }
    }

    #[allow(dead_code)]
    /// Returns the scalars as a slice.
    pub(crate) fn scalars(&self) -> &[PCS::Evaluation] {
        &self.scalars
    }

    /// Returns the bases as a slice.
    pub(crate) fn bases(&self) -> &[PointVariable] {
        &self.bases
    }
}

type EvalsAndCommitsVar<PCS> = Vec<(
    Vec<<PCS as PolynomialCommitmentScheme>::Evaluation>,
    PointVariable,
)>;

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
///   `next_eval_point`. This is stored as a variable.
#[derive(Debug, Clone)]
pub struct PcsInfoBasesVar<PCS>
where
    PCS: PolynomialCommitmentScheme,
{
    /// random combiner
    pub u: PCS::Evaluation,
    /// the scalars-and-bases form of the (aggregated) polynomial commitment
    pub comm_scalars_and_bases: ScalarsAndBasesVar<PCS>,
    /// (aggregated) proof of evaluations at point `eval_point`
    pub opening_proof: PointVariable,
}

impl FFTVerifier<Kzg> {
    /// Prepare the (aggregated) polynomial commitment evaluation information.
    pub(crate) fn prepare_pcs_info_with_bases_var<T>(
        &self,
        vk_var: &VerifyingKeyScalarsAndBasesVar<Kzg>,
        public_inputs: &[Fr254],
        output: &Bn254Output,
        extra_transcript_init_msg: &Option<Vec<u8>>,
        circuit: &mut PlonkCircuit<Fq254>,
        blind: bool,
    ) -> Result<(Bn254OutputScalarsAndBasesVar, PcsInfoBasesVar<Kzg>), PlonkError>
    where
        T: Transcript,
    {
        if public_inputs.len() != vk_var.num_inputs {
            return Err(ParameterError(format!(
                "the circuit public input length {} != the verification key public input length {}",
                public_inputs.len(),
                vk_var.num_inputs
            ))
            .into());
        }
        if vk_var.plookup_vk.is_some() != output.proof.plookup_proof.is_some() {
            return Err(ParameterError(
                "Mismatched proof type and verification key type".to_string(),
            )
            .into());
        }
        if vk_var.domain_size != self.domain.size() {
            return Err(ParameterError(format!(
                "the domain size of the verification key is different from {}",
                self.domain.size(),
            ))
            .into());
        }

        let output_var = output.create_variables(circuit)?;

        // We extract the id value from the verification key variable.
        let vk_id = if let Some(id) = vk_var.id {
            Some(fq_to_fr::<Fq254, BnConfig>(&circuit.witness(id)?))
        } else {
            None
        };
        // compute challenges except 'u' and evaluations
        let challenges = Self::compute_challenges_with_bases_var::<T>(
            &vk_id,
            public_inputs,
            &output.proof,
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
        let lin_poly_constant = self.compute_lin_poly_constant_term_with_bases_var(
            &challenges,
            vk_var.is_merged,
            public_inputs,
            &output.proof,
            &vanish_eval,
            &lagrange_1_eval,
            &lagrange_n_eval,
            &alpha_powers,
        )?;

        let mut comms_and_eval_points = BTreeMap::<MapKey<Kzg>, Vec<PointVariable>>::new();

        let zeta = challenges.zeta;
        let zeta_omega = self.domain.group_gen * zeta;

        // First we compute the polynomial z(x)
        let z_1_poly = DensePolynomial::from_coefficients_slice(&[zeta.neg(), Fr254::ONE]);
        let z_2_poly = DensePolynomial::from_coefficients_slice(&[zeta_omega.neg(), Fr254::ONE]);

        let z_poly = &z_1_poly * &z_2_poly;

        let g_base_zero = vk_var.g;

        let d1_scalars_and_bases = self.linearization_scalars_and_bases_var(
            vk_var,
            &challenges,
            &vanish_eval,
            &lagrange_1_eval,
            &lagrange_n_eval,
            &output_var,
            &alpha_powers,
            blind,
        )?;

        let z_1_poly_eval = z_1_poly.evaluate(&challenges.u);
        let z_2_poly_eval = z_2_poly.evaluate(&challenges.u);
        let mut g_base_scalars = vec![lin_poly_constant * z_2_poly_eval];
        if let Some(plookup_proof) = &output_var.proof.plookup_proof {
            // First we add all the commitments opened at 'zeta'.
            let mut polys_ref_and_evals = vec![output_var.proof.wires_poly_comms[0]];
            g_base_scalars.push(-(z_2_poly_eval * output.proof.poly_evals.wires_evals[0]));
            polys_ref_and_evals.push(output_var.proof.wires_poly_comms[1]);
            g_base_scalars.push(-(z_2_poly_eval * output.proof.poly_evals.wires_evals[1]));
            polys_ref_and_evals.push(output_var.proof.wires_poly_comms[2]);
            g_base_scalars.push(-(z_2_poly_eval * output.proof.poly_evals.wires_evals[2]));
            polys_ref_and_evals.push(output_var.proof.wires_poly_comms[5]);
            g_base_scalars.push(-(z_2_poly_eval * output.proof.poly_evals.wires_evals[5]));

            let num_wire_types = output_var.proof.wires_poly_comms.len();
            for (poly_comm, poly_eval) in vk_var
                .sigma_comms
                .iter()
                .take(num_wire_types - 1)
                .zip(output.proof.poly_evals.wire_sigma_evals.iter())
            {
                polys_ref_and_evals.push(*poly_comm);
                g_base_scalars.push(-(z_2_poly_eval * *poly_eval));
            }

            polys_ref_and_evals.push(vk_var.plookup_vk.as_ref().unwrap().q_dom_sep_comm);
            g_base_scalars.push(-(z_2_poly_eval * plookup_proof.poly_evals.q_dom_sep_eval));

            comms_and_eval_points.insert(MapKey(1, z_1_poly), polys_ref_and_evals);

            // Now all the commitments opened only at 'zeta * omega'.
            let mut polys_ref_and_evals = vec![output_var.proof.prod_perm_poly_comm];
            g_base_scalars.push(-(z_1_poly_eval * output.proof.poly_evals.perm_next_eval));
            polys_ref_and_evals.push(plookup_proof.h_poly_comms[1]);
            g_base_scalars.push(-(z_1_poly_eval * plookup_proof.poly_evals.h_2_next_eval));
            polys_ref_and_evals.push(plookup_proof.prod_lookup_poly_comm);
            g_base_scalars.push(-(z_1_poly_eval * plookup_proof.poly_evals.prod_next_eval));

            comms_and_eval_points.insert(MapKey(2, z_2_poly), polys_ref_and_evals);

            // Finally we add the commitments opened at both 'zeta' and 'zeta * omega'.
            let polys_ref_and_evals =
                Self::plookup_open_poly_evals_and_comms_var(&output_var.proof, vk_var)?;

            let polys_ref_and_evals_two = polys_ref_and_evals
                .iter()
                .map(|(_, comm)| *comm)
                .collect::<Vec<PointVariable>>();
            polys_ref_and_evals.iter().try_for_each(|(evals, _)| {
                let lagrange_eval =
                    Self::lagrange_interpolate(&[zeta, zeta_omega], evals)?.evaluate(&challenges.u);
                g_base_scalars.push(-(lagrange_eval));
                Ok::<_, PlonkError>(())
            })?;
            comms_and_eval_points.insert(MapKey(3, z_poly.clone()), polys_ref_and_evals_two);
        } else {
            let num_wire_types = output_var.proof.wires_poly_comms.len();
            let comms_iter = output_var
                .proof
                .wires_poly_comms
                .iter()
                .chain(vk_var.sigma_comms.iter().take(num_wire_types - 1));
            let evals_iter = output.proof.poly_evals.wires_evals.iter().chain(
                output
                    .proof
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
                .collect::<Vec<PointVariable>>();

            comms_and_eval_points.insert(MapKey(1, z_1_poly), polys_ref_and_evals);

            // Now the grand product commitment opened at 'zeta * omega'
            comms_and_eval_points.insert(
                MapKey(2, z_2_poly),
                vec![output_var.proof.prod_perm_poly_comm],
            );
            g_base_scalars.push(-(z_1_poly_eval * output.proof.poly_evals.perm_next_eval));
        }

        let mut combiner = challenges.v;
        let mut scalars_and_bases = ScalarsAndBasesVar::<Kzg>::new();

        for (map_key, points) in comms_and_eval_points.iter() {
            let z_i_u = (&z_poly / &map_key.1).evaluate(&challenges.u);

            for &point in points.iter() {
                scalars_and_bases.push(combiner * z_i_u, point);

                combiner *= challenges.v;
            }
        }

        let mut combiner = Fr254::one();

        let g_scalar = g_base_scalars.iter().fold(Fr254::zero(), |acc, s| {
            let val = combiner * s;
            combiner *= challenges.v;
            acc + val
        });

        let neg_z_u = z_poly.evaluate(&challenges.u).neg();
        scalars_and_bases.push(neg_z_u, output_var.proof.q_comm);
        scalars_and_bases.push(g_scalar, g_base_zero);

        scalars_and_bases.merge(challenges.u - zeta_omega, &d1_scalars_and_bases);

        let pcs_info_var = PcsInfoBasesVar::<Kzg> {
            u: challenges.u,
            comm_scalars_and_bases: scalars_and_bases,
            opening_proof: output_var.proof.opening_proof,
        };

        Ok((output_var, pcs_info_var))
    }

    /// Compute verifier challenges `tau`, `beta`, `gamma`, `alpha`, `zeta`,
    /// 'v', 'u'.
    #[inline]
    pub(crate) fn compute_challenges_with_bases_var<T>(
        vk_id: &Option<Fr254>,
        public_inputs: &[Fr254],
        proof: &Proof<Kzg>,
        extra_transcript_init_msg: &Option<Vec<u8>>,
    ) -> Result<Challenges<Fr254>, PlonkError>
    where
        T: Transcript,
    {
        let mut transcript = T::new_transcript(b"PlonkProof");
        if let Some(msg) = extra_transcript_init_msg {
            transcript.push_message(EXTRA_TRANSCRIPT_MSG_LABEL, msg)?;
        }

        if let Some(vk_id) = vk_id {
            transcript.push_message(b"verifying key", vk_id)?;
        }

        for pub_input in public_inputs.iter() {
            transcript.push_message(b"public_input", pub_input)?;
        }

        transcript.append_curve_points(b"witness_poly_comms", &proof.wires_poly_comms)?;

        let tau = transcript.squeeze_scalar_challenge::<BnConfig>(b"tau")?;

        if let Some(proof_lkup) = proof.plookup_proof.as_ref() {
            transcript.append_curve_points(b"h_poly_comms", &proof_lkup.h_poly_comms)?;
        }

        let [beta, gamma]: [Fr254; 2] = transcript
            .squeeze_scalar_challenges::<BnConfig>(b"beta gamma", 2)?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters("Couldn't convert to fixed length array".to_string())
            })?;

        transcript.append_curve_point(b"perm_poly_comms", &proof.prod_perm_poly_comm)?;

        if let Some(proof_lkup) = proof.plookup_proof.as_ref() {
            transcript
                .append_curve_point(b"plookup_poly_comms", &proof_lkup.prod_lookup_poly_comm)?;
        }

        let alpha = transcript.squeeze_scalar_challenge::<BnConfig>(b"alpha")?;

        transcript.append_curve_points(b"quot_poly_comms", &proof.split_quot_poly_comms)?;
        let zeta = transcript.squeeze_scalar_challenge::<BnConfig>(b"zeta")?;

        transcript.append_visitor(&proof.poly_evals)?;

        if let Some(proof_lkup) = proof.plookup_proof.as_ref() {
            transcript.append_visitor(&proof_lkup.poly_evals)?;
        }

        let v = transcript.squeeze_scalar_challenge::<BnConfig>(b"v")?;

        transcript.append_curve_point(b"q_comm", &proof.q_comm)?;

        let u = transcript.squeeze_scalar_challenge::<BnConfig>(b"u")?;
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
    pub(crate) fn compute_lin_poly_constant_term_with_bases_var(
        &self,
        challenges: &Challenges<Fr254>,
        is_merged: bool,
        public_inputs: &[Fr254],
        proof: &Proof<Kzg>,
        vanish_eval: &Fr254,
        lagrange_1_eval: &Fr254,
        lagrange_n_eval: &Fr254,
        alpha_powers: &[Fr254],
    ) -> Result<Fr254, PlonkError> {
        let mut result = Fr254::zero();

        let mut tmp =
            self.evaluate_pi_poly(public_inputs, &challenges.zeta, vanish_eval, is_merged)?
                - alpha_powers[0] * lagrange_1_eval;
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
            let gamma_mul_beta_plus_one = challenges.gamma * (Fr254::one() + challenges.beta);
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
    pub(crate) fn linearization_scalars_and_bases_var(
        &self,
        vk_var: &VerifyingKeyScalarsAndBasesVar<Kzg>,
        challenges: &Challenges<Fr254>,
        vanish_eval: &Fr254,
        lagrange_1_eval: &Fr254,
        lagrange_n_eval: &Fr254,
        output_var: &Bn254OutputScalarsAndBasesVar,
        alpha_powers: &[Fr254],
        blind: bool,
    ) -> Result<ScalarsAndBasesVar<Kzg>, PlonkError> {
        // compute constants that are being reused
        let beta_plus_one = Fr254::one() + challenges.beta;
        let gamma_mul_beta_plus_one = beta_plus_one * challenges.gamma;

        let mut scalars_and_bases_var = ScalarsAndBasesVar::<Kzg>::new();

        // Compute coefficient for the permutation product polynomial commitment.
        // coeff = L1(zeta) * alpha^2
        //       + alpha
        //       * (beta * zeta      + a_bar + gamma)
        //       * (beta * k1 * zeta + b_bar + gamma)
        //       * (beta * k2 * zeta + c_bar + gamma)
        // where a_bar, b_bar and c_bar are in w_evals
        let mut coeff = alpha_powers[0] * lagrange_1_eval;
        let w_evals = &output_var.proof.poly_evals.wires_evals;
        coeff += w_evals
            .iter()
            .zip(vk_var.k.iter())
            .fold(challenges.alpha, |acc, (w_eval, k)| {
                acc * (challenges.beta * k * challenges.zeta + challenges.gamma + w_eval)
            });

        // Add permutation product polynomial commitment.
        scalars_and_bases_var.push(coeff, output_var.proof.prod_perm_poly_comm);

        // Compute coefficient for the last wire sigma polynomial commitment.
        let num_wire_types = output_var.proof.wires_poly_comms.len();
        let sigma_evals = &output_var.proof.poly_evals.wire_sigma_evals;
        let coeff = w_evals
            .iter()
            .take(num_wire_types - 1)
            .zip(sigma_evals.iter())
            .fold(
                challenges.alpha * challenges.beta * output_var.proof.poly_evals.perm_next_eval,
                |acc, (w_eval, sigma_eval)| {
                    acc * (challenges.beta * sigma_eval + challenges.gamma + w_eval)
                },
            );

        // Add output wire sigma polynomial commitment.
        scalars_and_bases_var.push(
            -coeff,
            *vk_var.sigma_comms.last().ok_or(PlonkError::IndexError)?,
        );

        // Add selector polynomial commitments.
        // Compute coefficients for selector polynomial commitments.
        // The order: q_lc, q_mul, q_hash, q_o, q_c, q_ecc, q_x, q_x2, q_y, q_y2
        // TODO(binyi): get the order from a function.
        let mut q_scalars = [Fr254::zero(); 2 * GATE_WIDTH + 9];
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
        q_scalars[11] = Fr254::one();
        q_scalars[12] = w_evals[0] * w_evals[1] * w_evals[2] * w_evals[3] * w_evals[4];
        q_scalars[13] = w_evals[0] * w_evals[3] * w_evals[2] * w_evals[3]
            + w_evals[1] * w_evals[2] * w_evals[2] * w_evals[3];
        q_scalars[14] = w_evals[0] * w_evals[2]
            + w_evals[1] * w_evals[3]
            + Fr254::from(2u8) * w_evals[0] * w_evals[3]
            + Fr254::from(2u8) * w_evals[1] * w_evals[2];
        q_scalars[15] = w_evals[2] * w_evals[2] * w_evals[3] * w_evals[3];
        q_scalars[16] = w_evals[0] * w_evals[0] * w_evals[1] + w_evals[0] * w_evals[1] * w_evals[1];

        for (&s, poly) in q_scalars.iter().zip(vk_var.selector_comms.iter()) {
            scalars_and_bases_var.push(s, *poly);
        }

        // Add splitted quotient commitments
        let zeta_to_n = Fr254::one() + vanish_eval;
        let scalar = if blind {
            zeta_to_n * challenges.zeta * challenges.zeta
        } else {
            zeta_to_n
        };
        let mut coeff = vanish_eval.neg();
        scalars_and_bases_var.push(
            coeff,
            *output_var
                .proof
                .split_quot_poly_comms
                .first()
                .ok_or(PlonkError::IndexError)?,
        );
        for poly in output_var.proof.split_quot_poly_comms.iter().skip(1) {
            coeff *= scalar;
            scalars_and_bases_var.push(coeff, *poly);
        }

        // Add Plookup related commitments
        if let Some(lookup_proof) = output_var.proof.plookup_proof.as_ref() {
            let lookup_evals = &lookup_proof.poly_evals;
            let merged_lookup_x = eval_merged_lookup_witness::<BnConfig>(
                challenges.tau,
                w_evals[5],
                w_evals[0],
                w_evals[1],
                w_evals[2],
                lookup_evals.q_lookup_eval,
                lookup_evals.q_dom_sep_eval,
            );
            let merged_table_x = eval_merged_table::<BnConfig>(
                challenges.tau,
                lookup_evals.range_table_eval,
                lookup_evals.key_table_eval,
                lookup_evals.q_lookup_eval,
                w_evals[3],
                w_evals[4],
                lookup_evals.table_dom_sep_eval,
            );
            let merged_table_xw = eval_merged_table::<BnConfig>(
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
            scalars_and_bases_var.push(coeff, lookup_proof.prod_lookup_poly_comm);

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
            scalars_and_bases_var.push(coeff, lookup_proof.h_poly_comms[1]);
        }

        Ok(scalars_and_bases_var)
    }

    /// Return the list of polynomials commitments to be opened at points 'zeta' and 'zeta * omega'
    /// Together with their evaluations at both points.
    fn plookup_open_poly_evals_and_comms_var(
        proof_var: &Bn254ProofScalarsandBasesVar,
        vk_var: &VerifyingKeyScalarsAndBasesVar<Kzg>,
    ) -> Result<EvalsAndCommitsVar<Kzg>, PlonkError> {
        let plookup_proof =
            proof_var
                .plookup_proof
                .as_ref()
                .ok_or(PlonkError::InvalidParameters(
                    "The proof does not contain a plookup proof var".to_string(),
                ))?;
        let plookup_vk = vk_var
            .plookup_vk
            .as_ref()
            .ok_or(PlonkError::InvalidParameters(
                "The verifying key does not contain a plookup verifying key var".to_string(),
            ))?;

        Ok(vec![
            (
                vec![
                    plookup_proof.poly_evals.range_table_eval,
                    plookup_proof.poly_evals.range_table_next_eval,
                ],
                plookup_vk.range_table_comm,
            ),
            (
                vec![
                    plookup_proof.poly_evals.key_table_eval,
                    plookup_proof.poly_evals.key_table_next_eval,
                ],
                plookup_vk.key_table_comm,
            ),
            (
                vec![
                    plookup_proof.poly_evals.h_1_eval,
                    plookup_proof.poly_evals.h_1_next_eval,
                ],
                plookup_proof.h_poly_comms[0],
            ),
            (
                vec![
                    plookup_proof.poly_evals.q_lookup_eval,
                    plookup_proof.poly_evals.q_lookup_next_eval,
                ],
                *vk_var.q_lookup_comm()?,
            ),
            (
                vec![
                    proof_var.poly_evals.wires_evals[3],
                    plookup_proof.poly_evals.w_3_next_eval,
                ],
                proof_var.wires_poly_comms[3],
            ),
            (
                vec![
                    proof_var.poly_evals.wires_evals[4],
                    plookup_proof.poly_evals.w_4_next_eval,
                ],
                proof_var.wires_poly_comms[4],
            ),
            (
                vec![
                    plookup_proof.poly_evals.table_dom_sep_eval,
                    plookup_proof.poly_evals.table_dom_sep_next_eval,
                ],
                plookup_vk.table_dom_sep_comm,
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        errors::PlonkError,
        nightfall::{
            ipa_snark::test::gen_circuit_for_test, ipa_structs::VerificationKeyId,
            ipa_verifier::FFTVerifier, FFTPlonk,
        },
        proof_system::{UniversalRecursiveSNARK, UniversalSNARK},
        transcript::RescueTranscript,
    };

    use super::*;
    use ark_bn254::Bn254;
    use itertools::izip;
    use jf_primitives::pcs::prelude::UnivariateKzgPCS;
    use jf_relation::{Arithmetization, PlonkType};

    #[test]
    fn test_prepare_pcs_info_with_bases_var() -> Result<(), PlonkError> {
        let rng = &mut jf_utils::test_rng();
        for (m, vk_id, blind) in izip!(
            (2..8),
            [
                Some(VerificationKeyId::Client),
                Some(VerificationKeyId::Deposit),
                None,
            ],
            [true, false],
        ) {
            let circuit = gen_circuit_for_test::<Fr254>(m, 3, PlonkType::UltraPlonk, true)?;
            let pi = circuit.public_input()?[0];

            let srs_size = circuit.srs_size(blind)?;
            let srs = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, srs_size)?;

            // Here we are assuming we are in the non-base case and our verification key is fixed.
            // Our `vk_id` is, therefore, `None`.
            let (pk, vk) = FFTPlonk::<Kzg>::preprocess(&srs, vk_id, &circuit, blind)?;

            let mut output = FFTPlonk::<Kzg>::recursive_prove::<_, _, RescueTranscript<Fr254>>(
                rng, &circuit, &pk, None, blind,
            )?;

            let fft_verifier = FFTVerifier::<Kzg>::new(vk.domain_size)?;

            let pcs_info = fft_verifier.prepare_pcs_info::<RescueTranscript<Fr254>>(
                &vk,
                &[pi],
                &output.proof,
                &None,
                blind,
            )?;

            let mut verifier_circuit = PlonkCircuit::<Fq254>::new_ultra_plonk(8);
            let vk_var = VerifyingKeyScalarsAndBasesVar::new(&mut verifier_circuit, &vk)?;

            let (mut output_var, pcs_info_var) = fft_verifier
                .prepare_pcs_info_with_bases_var::<RescueTranscript<Fr254>>(
                    &vk_var,
                    &[pi],
                    &output,
                    &None,
                    &mut verifier_circuit,
                    blind,
                )?;

            assert!(verifier_circuit.check_circuit_satisfiability(&[]).is_ok());

            // We check that `output_var` matches up with its non-circuit equivalent.
            for (point_var, point) in output_var
                .proof
                .wires_poly_comms
                .iter()
                .zip(output.proof.wires_poly_comms.iter())
            {
                assert_eq!(verifier_circuit.witness(point_var.get_x())?, point.x);
                assert_eq!(verifier_circuit.witness(point_var.get_y())?, point.y);
            }

            assert_eq!(
                verifier_circuit.witness(output_var.proof.prod_perm_poly_comm.get_x())?,
                output.proof.prod_perm_poly_comm.x,
            );
            assert_eq!(
                verifier_circuit.witness(output_var.proof.prod_perm_poly_comm.get_y())?,
                output.proof.prod_perm_poly_comm.y,
            );
            for (point_var, point) in output_var
                .proof
                .split_quot_poly_comms
                .iter()
                .zip(output.proof.split_quot_poly_comms.iter())
            {
                assert_eq!(verifier_circuit.witness(point_var.get_x())?, point.x);
                assert_eq!(verifier_circuit.witness(point_var.get_y())?, point.y);
            }
            assert_eq!(
                verifier_circuit.witness(output_var.proof.opening_proof.get_x())?,
                output.proof.opening_proof.proof.x,
            );
            assert_eq!(
                verifier_circuit.witness(output_var.proof.opening_proof.get_y())?,
                output.proof.opening_proof.proof.y,
            );
            assert_eq!(
                output_var.proof.poly_evals.wires_evals,
                output.proof.poly_evals.wires_evals,
            );
            assert_eq!(
                output_var.proof.poly_evals.wire_sigma_evals,
                output.proof.poly_evals.wire_sigma_evals,
            );
            assert_eq!(
                output_var.proof.poly_evals.perm_next_eval,
                output.proof.poly_evals.perm_next_eval,
            );

            let plookup_proof_var = if let Some(plookup_proof_var) = output_var.proof.plookup_proof
            {
                plookup_proof_var
            } else {
                return Err(PlonkError::InvalidParameters(
                    "The proof does not contain a plookup proof var".to_string(),
                ));
            };
            let plookup_proof = if let Some(plookup_proof) = output.proof.plookup_proof {
                plookup_proof
            } else {
                return Err(PlonkError::InvalidParameters(
                    "The proof does not contain a plookup proof".to_string(),
                ));
            };
            for (point_var, point) in plookup_proof_var
                .h_poly_comms
                .iter()
                .zip(plookup_proof.h_poly_comms.iter())
            {
                assert_eq!(verifier_circuit.witness(point_var.get_x())?, point.x);
                assert_eq!(verifier_circuit.witness(point_var.get_y())?, point.y);
            }
            assert_eq!(
                verifier_circuit.witness(plookup_proof_var.prod_lookup_poly_comm.get_x())?,
                plookup_proof.prod_lookup_poly_comm.x,
            );
            assert_eq!(
                verifier_circuit.witness(plookup_proof_var.prod_lookup_poly_comm.get_y())?,
                plookup_proof.prod_lookup_poly_comm.y,
            );
            assert_eq!(plookup_proof_var.poly_evals, plookup_proof.poly_evals,);

            assert_eq!(output_var.pi_hash, output.pi_hash);
            // We test equality of the transcripts by squeezing a challenge from each.
            assert_eq!(
                output_var
                    .transcript
                    .squeeze_scalar_challenge::<BnConfig>(b"test")?,
                output
                    .transcript
                    .squeeze_scalar_challenge::<BnConfig>(b"test")?,
            );

            // We check that `pcs_var` matches up with its non-circuit equivalent.
            assert_eq!(pcs_info_var.u, pcs_info.u);
            assert_eq!(
                pcs_info_var.comm_scalars_and_bases.scalars,
                pcs_info.comm_scalars_and_bases.scalars
            );
            for (point_var, point) in pcs_info_var
                .comm_scalars_and_bases
                .bases
                .iter()
                .zip(pcs_info.comm_scalars_and_bases.bases.iter())
            {
                assert_eq!(verifier_circuit.witness(point_var.get_x())?, point.x);
                assert_eq!(verifier_circuit.witness(point_var.get_y())?, point.y);
            }
            assert_eq!(
                verifier_circuit.witness(pcs_info_var.opening_proof.get_x())?,
                pcs_info.opening_proof.proof.x
            );
            assert_eq!(
                verifier_circuit.witness(pcs_info_var.opening_proof.get_y())?,
                pcs_info.opening_proof.proof.y
            );
        }
        Ok(())
    }
}
