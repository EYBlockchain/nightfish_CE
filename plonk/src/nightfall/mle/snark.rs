#![allow(dead_code)]
//! This module contains the MLEPlonk struct that can be used to prove Plonk-ish circuits
//! over a curve that does not have an FFT-friendly scalar field.

use ark_ec::{
    short_weierstrass::{Affine, Projective},
    CurveGroup, VariableBaseMSM,
};
use ark_ff::{Field, One, PrimeField, Zero};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{
    cfg_iter, format,
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    string::ToString,
    sync::Arc,
    vec,
    vec::Vec,
};
use jf_primitives::{
    pcs::{Accumulation, PolynomialCommitmentScheme},
    rescue::RescueParameter,
};
use jf_relation::{
    gadgets::{ecc::HasTEForm, EmulationConfig},
    Arithmetization,
};
use rayon::prelude::*;

use crate::{
    errors::{PlonkError, SnarkError},
    nightfall::{
        accumulation::{accumulation_structs::SplitAccumulator, MLAccumulator},
        mle::mle_structs::{FullMLEChallenges, MLEProofShared},
    },
    proof_system::RecursiveOutput,
    transcript::Transcript,
};

use super::{
    mle_structs::{
        GateInfo, MLEChallenges, MLELookupEvals, MLELookupProof, MLELookupProvingKey,
        MLELookupVerifyingKey, MLEProof, MLEProofEvals, MLEProvingKey, MLEVerifyingKey,
        PolynomialError, SAMLEProof,
    },
    subroutines::{
        gkr::batch_prove_gkr,
        gkr::batch_verify_gkr,
        lookupcheck::{LogUpTable, LookupCheck},
        permutationcheck::PermutationCheck,
        sumcheck::SumCheck,
        VPSumCheck,
    },
    utils::{
        add_vecs, build_eq_x_r, build_sumcheck_poly, build_zerocheck_eval, eq_eval, scale_mle,
    },
};

/// MLEPlonk struct used to preprocess, prove and verify circuits.
#[derive(Debug, Clone)]
pub struct MLEPlonk<PCS: PolynomialCommitmentScheme>(PhantomData<PCS>);

impl<PCS: PolynomialCommitmentScheme> MLEPlonk<PCS> {
    /// Create a new MLEPlonk struct.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<PCS: PolynomialCommitmentScheme> Default for MLEPlonk<PCS> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type for simplifying recursive proving outputs.
type InternalRecursionOutput<PCS, T> = (SAMLEProof<PCS>, T);

impl<PCS: PolynomialCommitmentScheme> MLEPlonk<PCS> {
    /// Preprocess a circuit. In this we produce the verifying and proving keys.
    /// TODO: Add support for customizing the gate info/a way to read the gate info from the circuit.
    pub fn preprocess_helper<F, C>(
        circuit: &C,
        srs: &PCS::SRS,
    ) -> Result<(MLEProvingKey<PCS>, MLEVerifyingKey<PCS>), PlonkError>
    where
        F: PrimeField,
        C: Arithmetization<F>,
        PCS: PolynomialCommitmentScheme<
            Evaluation = F,
            Polynomial = Arc<DenseMultilinearExtension<F>>,
        >,
    {
        let srs_size = circuit.srs_size(false)?;

        let params_to_extract = srs_size.ilog2();

        let selector_polys = circuit.compute_selector_mles()?[..17].to_vec();
        let permutation_polys = circuit.compute_extended_permutation_mles()?;

        // Compute lookup proving key if support lookup.
        let lookup_pk = if circuit.support_lookup() {
            let range_table_mle = circuit.compute_range_table_mle()?;
            let key_table_mle = circuit.compute_key_table_mle()?;
            let table_dom_sep_mle = circuit.compute_table_dom_sep_mle()?;
            let q_dom_sep_mle = circuit.compute_q_dom_sep_mle()?;
            let q_lookup_mle = circuit.compute_q_lookup_mle()?;
            Some(MLELookupProvingKey {
                range_table_mle,
                key_table_mle,
                table_dom_sep_mle,
                q_dom_sep_mle,
                q_lookup_mle,
            })
        } else {
            None
        };

        let (prover_param, verifier_param) =
            PCS::trim(srs, srs_size, Some(params_to_extract as usize))?;

        let turbo_gate: GateInfo<F> = GateInfo {
            max_degree: 6,
            products: vec![
                (F::from(1u8), vec![5, 0]),
                (F::from(1u8), vec![6, 1]),
                (F::from(1u8), vec![7, 2]),
                (F::from(1u8), vec![8, 3]),
                (F::from(1u8), vec![9, 0, 1]),
                (F::from(1u8), vec![10, 2, 3]),
                (F::from(1u8), vec![11, 0, 0, 0, 0, 0]),
                (F::from(1u8), vec![12, 1, 1, 1, 1, 1]),
                (F::from(1u8), vec![13, 2, 2, 2, 2, 2]),
                (F::from(1u8), vec![14, 3, 3, 3, 3, 3]),
                (-F::from(1u8), vec![15, 4]),
                (F::from(1u8), vec![16]),
                (F::from(1u8), vec![17, 0, 1, 2, 3, 4]),
                (F::from(1u8), vec![22]),
                (F::from(1u8), vec![18, 0, 3, 2, 3]),
                (F::from(1u8), vec![18, 1, 2, 2, 3]),
                (F::from(1u8), vec![19, 0, 2]),
                (F::from(1u8), vec![19, 1, 3]),
                (F::from(2u8), vec![19, 0, 3]),
                (F::from(2u8), vec![19, 1, 2]),
                (F::from(1u8), vec![20, 2, 2, 3, 3]),
                (F::from(1u8), vec![21, 0, 0, 1]),
                (F::from(1u8), vec![21, 1, 1, 0]),
            ],
        };

        let selector_commitments = cfg_iter!(&selector_polys)
            .map(|poly| PCS::commit(&prover_param, poly))
            .collect::<Result<Vec<PCS::Commitment>, _>>()?;

        let permutation_commitments = cfg_iter!(&permutation_polys)
            .map(|poly| PCS::commit(&prover_param, poly))
            .collect::<Result<Vec<PCS::Commitment>, _>>()?;

        let lookup_vk = if let Some(lookup_pk) = &lookup_pk {
            let range_table_comm = PCS::commit(&prover_param, &lookup_pk.range_table_mle)?;
            let key_table_comm = PCS::commit(&prover_param, &lookup_pk.key_table_mle)?;
            let table_dom_sep_comm = PCS::commit(&prover_param, &lookup_pk.table_dom_sep_mle)?;
            let q_dom_sep_comm = PCS::commit(&prover_param, &lookup_pk.q_dom_sep_mle)?;
            let q_lookup_comm = PCS::commit(&prover_param, &lookup_pk.q_lookup_mle)?;
            Some(MLELookupVerifyingKey {
                range_table_comm,
                key_table_comm,
                table_dom_sep_comm,
                q_dom_sep_comm,
                q_lookup_comm,
            })
        } else {
            None
        };

        let vk = MLEVerifyingKey::<PCS> {
            selector_commitments,
            permutation_commitments,
            lookup_verifying_key: lookup_vk,
            pcs_verifier_params: verifier_param,
            gate_info: turbo_gate,
            num_inputs: circuit.num_inputs(),
            num_vars: circuit.num_gates().ilog2(),
        };

        let pk = MLEProvingKey {
            selector_oracles: selector_polys,
            permutation_oracles: permutation_polys,
            lookup_proving_key: lookup_pk,
            verifying_key: vk.clone(),
            pcs_prover_params: prover_param,
        };
        Ok((pk, vk))
    }

    /// Prove a circuit.
    pub fn prove<C, F, P, T>(
        circuit: &C,
        pk: &MLEProvingKey<PCS>,
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<MLEProof<PCS>, PlonkError>
    where
        F: PrimeField + RescueParameter,
        C: Arithmetization<P::ScalarField>,
        PCS: Accumulation<
            Evaluation = P::ScalarField,
            Point = Vec<P::ScalarField>,
            Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
            Commitment = Affine<P>,
        >,
        P: HasTEForm<BaseField = F>,
        P::ScalarField: EmulationConfig<F>,
        T: Transcript,
    {
        let mut transcript: T = if let Some(msg) = extra_transcript_init_msg {
            T::new_with_initial_message::<_, P>(&msg)?
        } else {
            T::new_transcript(b"mle_plonk")
        };

        // Append public input to transcript.
        for public_input in circuit.public_input()? {
            transcript.push_message(b"public input", &public_input)?;
        }

        // Compute the wire polynomials.
        let wire_polys = circuit.compute_wire_mles()?;

        let num_vars = wire_polys[0].num_vars;

        // Compute the public input mle.
        let mut public_inputs = circuit.public_input()?;
        public_inputs.resize(circuit.num_gates(), P::ScalarField::zero());

        // We take the negative because we need to subtract the public input from the gate equation.
        // This is because I/O gates just declare the output wire to be the value of the public input.
        let public_input_poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            public_inputs,
        ));

        // Commit to the wire polynomials.
        let wire_comms = cfg_iter!(&wire_polys)
            .map(|poly| PCS::commit(&pk.pcs_prover_params, poly))
            .collect::<Result<Vec<PCS::Commitment>, _>>()?;

        // We know that the commitments we are using will always be points on an SW curve.
        transcript.append_curve_points(b"wires", &wire_comms)?;

        let [gamma, tau]: [P::ScalarField; 2] = transcript
            .squeeze_scalar_challenges::<P>(b"gamma tau", 2)?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters("Couldn't convert to fixed length array".to_string())
            })?;
        let perm_mles =
            circuit.compute_prod_permutation_mles(&wire_polys, &pk.permutation_oracles, &gamma)?;
        let [perm_p_poly, perm_q_poly] =
            <VPSumCheck<P> as PermutationCheck<P>>::prep_for_gkr(&perm_mles)?;
        let mut gkr_p_polys = vec![perm_p_poly];
        let mut gkr_q_polys = vec![perm_q_poly];

        // Run the lookup related subroutines if support lookup.
        let (m_poly, m_commit, [alpha, beta]) = if pk.lookup_proving_key.is_some() {
            let lookup_table = circuit.compute_merged_lookup_table_mle(tau)?;
            let lookup_wire = circuit.compute_lookup_sorted_vec_mles(tau, &lookup_table)?;
            let table = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                lookup_table,
            ));
            let logup_table = LogUpTable::<P::ScalarField>::new(table);
            let m_poly = LookupCheck::<P>::calculate_m_poly(&lookup_wire, &logup_table)?;
            let m_commit = PCS::commit(&pk.pcs_prover_params, &m_poly)?;

            transcript.append_curve_point(b"m_commit", &m_commit)?;
            let [alpha, beta]: [P::ScalarField; 2] = transcript
                .squeeze_scalar_challenges::<P>(b"alpha beta", 2)?
                .try_into()
                .map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Couldn't convert to fixed length array".to_string(),
                    )
                })?;

            let prepped_items =
                LookupCheck::<P>::reduce_to_gkr(&logup_table, lookup_wire, &m_poly, alpha, beta)?;

            gkr_p_polys.push(prepped_items.p_poly);
            gkr_q_polys.push(prepped_items.q_poly);
            (Some(m_poly), Some(m_commit), [alpha, beta])
        } else {
            let [alpha, beta]: [P::ScalarField; 2] = transcript
                .squeeze_scalar_challenges::<P>(b"alpha beta", 2)?
                .try_into()
                .map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Couldn't convert to fixed length array".to_string(),
                    )
                })?;
            (None, None, [alpha, beta])
        };
        let gkr_proof = batch_prove_gkr::<P, _>(&gkr_p_polys, &gkr_q_polys, &mut transcript)?;
        // Now we have done the GKR proof we run a final zero check on the underlying products of MLEs.

        // Since we return the individual multilinear extension evaluations when we verify the GKR proof we need only use the SumCheck
        // to check the evaluation claims for the numerator and denominator polynomials in th epermutation and the lookup wire and table.
        let gkr_point = &gkr_proof.sumcheck_proofs.last().as_ref().unwrap().point;

        let eq_poly = Arc::new(build_eq_x_r(gkr_point));
        let challenges = MLEChallenges {
            beta,
            gamma,
            alpha,
            tau,
        };

        let epsilon = transcript.squeeze_scalar_challenge::<P>(b"epsilon")?;

        let sumcheck_vp = build_sumcheck_poly(
            &wire_polys,
            &pk.selector_oracles,
            &pk.permutation_oracles,
            &public_input_poly,
            eq_poly.clone(),
            &pk.verifying_key.gate_info,
            &challenges,
            epsilon,
            pk.lookup_proving_key.as_ref(),
            m_poly.clone(),
        )?;

        let sumcheck_proof = <VPSumCheck<P> as SumCheck<P>>::prove(&sumcheck_vp, &mut transcript)?;
        // Extract the final point from the SumCheck proof. The unwrap is safe here because we ensure that we have at least one round.
        let zero_check_point = &sumcheck_proof.point;

        let mut pcs_accumulator = <MLAccumulator<PCS> as SplitAccumulator<PCS>>::new();

        // Calculate wire_evals and add the wires to the accumulator.
        let mut wire_evals = vec![];
        for (poly, comm) in wire_polys.iter().zip(wire_comms.iter()) {
            let value = poly
                .evaluate(zero_check_point)
                .ok_or(PlonkError::PolynomialError(
                    PolynomialError::ParameterError("Couldn't evaluate wire poly".to_string()),
                ))?;
            pcs_accumulator.push(poly.clone(), *comm, zero_check_point.clone(), value);
            wire_evals.push(value);
        }

        // Calculate selector_evals and add the selectors to the accumulator.
        let mut selector_evals = vec![];
        for (poly, comm) in pk
            .selector_oracles
            .iter()
            .zip(pk.verifying_key.selector_commitments.iter())
            .take(17)
        {
            let value = poly
                .evaluate(zero_check_point)
                .ok_or(PlonkError::PolynomialError(
                    PolynomialError::ParameterError("Couldn't evaluate selector poly".to_string()),
                ))?;
            pcs_accumulator.push(poly.clone(), *comm, zero_check_point.clone(), value);
            selector_evals.push(value);
        }

        // Calculate permutation_evals and add the permutations to the accumulator.
        let mut permutation_evals = vec![];
        for (poly, comm) in pk
            .permutation_oracles
            .iter()
            .zip(pk.verifying_key.permutation_commitments.iter())
        {
            let value = poly
                .evaluate(zero_check_point)
                .ok_or(PlonkError::PolynomialError(
                    PolynomialError::ParameterError(
                        "Couldn't evaluate permutation poly".to_string(),
                    ),
                ))?;
            pcs_accumulator.push(poly.clone(), *comm, zero_check_point.clone(), value);
            permutation_evals.push(value);
        }

        // Now we handle lookup related subroutines if support lookup.
        let lookup_proof = if let (Some(m_poly), Some(m_commit)) = (m_poly, m_commit) {
            let m_poly_eval =
                m_poly
                    .evaluate(zero_check_point)
                    .ok_or(PolynomialError::ParameterError(
                        "Could not evaluate m poly".to_string(),
                    ))?;

            pcs_accumulator.push(
                m_poly.clone(),
                m_commit,
                zero_check_point.clone(),
                m_poly_eval,
            );
            let range_table_eval = pk
                .lookup_proving_key
                .as_ref()
                .unwrap()
                .range_table_mle
                .evaluate(zero_check_point)
                .ok_or(PolynomialError::ParameterError(
                    "Could not evaluate range table mle".to_string(),
                ))?;
            pcs_accumulator.push(
                pk.lookup_proving_key
                    .as_ref()
                    .unwrap()
                    .range_table_mle
                    .clone(),
                pk.verifying_key
                    .lookup_verifying_key
                    .as_ref()
                    .unwrap()
                    .range_table_comm,
                zero_check_point.clone(),
                range_table_eval,
            );
            let key_table_eval = pk
                .lookup_proving_key
                .as_ref()
                .unwrap()
                .key_table_mle
                .evaluate(zero_check_point)
                .ok_or(PolynomialError::ParameterError(
                    "Could not evaluate key table mle".to_string(),
                ))?;
            pcs_accumulator.push(
                pk.lookup_proving_key
                    .as_ref()
                    .unwrap()
                    .key_table_mle
                    .clone(),
                pk.verifying_key
                    .lookup_verifying_key
                    .as_ref()
                    .unwrap()
                    .key_table_comm,
                zero_check_point.clone(),
                key_table_eval,
            );
            let table_dom_sep_eval = pk
                .lookup_proving_key
                .as_ref()
                .unwrap()
                .table_dom_sep_mle
                .evaluate(zero_check_point)
                .ok_or(PolynomialError::ParameterError(
                    "Could not evaluate table dom sep mle".to_string(),
                ))?;
            pcs_accumulator.push(
                pk.lookup_proving_key
                    .as_ref()
                    .unwrap()
                    .table_dom_sep_mle
                    .clone(),
                pk.verifying_key
                    .lookup_verifying_key
                    .as_ref()
                    .unwrap()
                    .table_dom_sep_comm,
                zero_check_point.clone(),
                table_dom_sep_eval,
            );
            let q_dom_sep_eval = pk
                .lookup_proving_key
                .as_ref()
                .unwrap()
                .q_dom_sep_mle
                .evaluate(zero_check_point)
                .ok_or(PolynomialError::ParameterError(
                    "Could not evaluate q dom sep mle".to_string(),
                ))?;
            pcs_accumulator.push(
                pk.lookup_proving_key
                    .as_ref()
                    .unwrap()
                    .q_dom_sep_mle
                    .clone(),
                pk.verifying_key
                    .lookup_verifying_key
                    .as_ref()
                    .unwrap()
                    .q_dom_sep_comm,
                zero_check_point.clone(),
                q_dom_sep_eval,
            );
            let q_lookup_eval = pk
                .lookup_proving_key
                .as_ref()
                .unwrap()
                .q_lookup_mle
                .evaluate(zero_check_point)
                .ok_or(PolynomialError::ParameterError(
                    "Could not evaluate q dom sep mle".to_string(),
                ))?;
            pcs_accumulator.push(
                pk.lookup_proving_key.as_ref().unwrap().q_lookup_mle.clone(),
                pk.verifying_key
                    .lookup_verifying_key
                    .as_ref()
                    .unwrap()
                    .q_lookup_comm,
                zero_check_point.clone(),
                q_lookup_eval,
            );

            let evals = MLELookupEvals::<PCS> {
                range_table_eval,
                key_table_eval,
                table_dom_sep_eval,
                q_dom_sep_eval,
                q_lookup_eval,
                m_poly_eval,
            };

            Some(MLELookupProof::<PCS> {
                m_poly_comm: m_commit,
                lookup_evals: evals,
            })
        } else {
            None
        };

        // Append evals and lookup_proof.poly_evals to the transcript.
        for eval in wire_evals
            .iter()
            .chain(&selector_evals)
            .chain(&permutation_evals)
        {
            transcript.push_message(b"eval", eval)?;
        }
        if let Some(lookup_proof) = lookup_proof.clone() {
            for eval in [
                lookup_proof.lookup_evals.m_poly_eval,
                lookup_proof.lookup_evals.range_table_eval,
                lookup_proof.lookup_evals.key_table_eval,
                lookup_proof.lookup_evals.table_dom_sep_eval,
                lookup_proof.lookup_evals.q_dom_sep_eval,
                lookup_proof.lookup_evals.q_lookup_eval,
            ]
            .iter()
            {
                transcript.push_message(b"lookup eval", eval)?;
            }
        }

        let evals = MLEProofEvals::<PCS> {
            wire_evals,
            selector_evals,
            permutation_evals,
        };

        let delta = transcript.squeeze_scalar_challenge::<P>(b"delta")?;

        let mut combiner = P::ScalarField::one();
        let batched_poly_evals = pcs_accumulator.polynomials().iter().try_fold(
            vec![P::ScalarField::zero(); 1 << num_vars],
            |acc, poly| {
                let scaled_poly = scale_mle(poly, combiner);
                combiner *= delta;
                add_vecs(&acc, &scaled_poly)
            },
        )?;
        let batched_poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            batched_poly_evals,
        ));

        let (opening_proof, _) = PCS::open(&pk.pcs_prover_params, &batched_poly, zero_check_point)?;
        Ok(MLEProof {
            wire_commitments: wire_comms,
            gkr_proof,
            sumcheck_proof,
            lookup_proof,
            evals,
            opening_proof,
        })
    }

    /// Prove for recursive purposes (so we don't perform the opening and provide a commitment to the public input).
    pub fn sa_prove<C, F, P, T>(
        circuit: &C,
        pk: &MLEProvingKey<PCS>,
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<InternalRecursionOutput<PCS, T>, PlonkError>
    where
        F: PrimeField + RescueParameter,
        C: Arithmetization<P::ScalarField>,
        PCS: Accumulation<
            Evaluation = P::ScalarField,
            Point = Vec<P::ScalarField>,
            Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
            Commitment = Affine<P>,
        >,
        P: HasTEForm<BaseField = F>,
        P::ScalarField: EmulationConfig<F>,
        T: Transcript,
    {
        // First we check if the circuit is suitable for recursive proving.
        if !circuit.is_recursive() {
            return Err(PlonkError::InvalidParameters(
                "Circuit is not suitable for recursive proving".to_string(),
            ));
        }

        let mut transcript: T = if let Some(msg) = extra_transcript_init_msg {
            T::new_with_initial_message::<_, P>(&msg)?
        } else {
            T::new_transcript(b"mle_plonk")
        };

        // Compute the wire polynomials.
        let wire_polys = circuit.compute_wire_mles()?;

        let num_vars = wire_polys[0].num_vars;

        // Compute the public input mle.
        let mut public_inputs = circuit.public_input()?;

        // Append the singular public input to the transcript.
        transcript.push_message(b"pi", &public_inputs[0])?;

        public_inputs.resize(circuit.num_gates(), P::ScalarField::zero());

        // We take the negative because we need to add the public input from the gate equation.
        let public_input_poly =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, public_inputs);
        let arc_pi_poly = Arc::new(public_input_poly.clone());

        // Commit to the wire polynomials.
        let wire_comms = cfg_iter!(&wire_polys)
            .map(|poly| PCS::commit(&pk.pcs_prover_params, poly))
            .collect::<Result<Vec<PCS::Commitment>, _>>()?;

        // We know that the commitments we are using will always be points on an SW curve.
        transcript.append_curve_points(b"wires", &wire_comms)?;

        let [gamma, tau]: [P::ScalarField; 2] = transcript
            .squeeze_scalar_challenges::<P>(b"gamma tau", 2)?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters("Couldn't convert to fixed length array".to_string())
            })?;

        let perm_mles =
            circuit.compute_prod_permutation_mles(&wire_polys, &pk.permutation_oracles, &gamma)?;

        let [perm_p_poly, perm_q_poly] =
            <VPSumCheck<P> as PermutationCheck<P>>::prep_for_gkr(&perm_mles)?;
        let mut gkr_p_polys = vec![perm_p_poly.clone()];
        let mut gkr_q_polys = vec![perm_q_poly.clone()];

        // Run the lookup related subroutines if support lookup.
        let (m_poly, m_commit, [alpha, beta]) = if pk.lookup_proving_key.is_some() {
            let lookup_table = circuit.compute_merged_lookup_table_mle(tau)?;
            let lookup_wire = circuit.compute_lookup_sorted_vec_mles(tau, &lookup_table)?;
            let table = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                lookup_table,
            ));
            let logup_table = LogUpTable::<P::ScalarField>::new(table);
            let m_poly = LookupCheck::<P>::calculate_m_poly(&lookup_wire, &logup_table)?;
            let m_commit = PCS::commit(&pk.pcs_prover_params, &m_poly)?;

            transcript.append_curve_point(b"m_commit", &m_commit)?;
            let [alpha, beta]: [P::ScalarField; 2] = transcript
                .squeeze_scalar_challenges::<P>(b"alpha beta", 2)?
                .try_into()
                .map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Couldn't convert to fixed length array".to_string(),
                    )
                })?;

            let prepped_items =
                LookupCheck::<P>::reduce_to_gkr(&logup_table, lookup_wire, &m_poly, alpha, beta)?;

            gkr_p_polys.push(prepped_items.p_poly);
            gkr_q_polys.push(prepped_items.q_poly);
            (Some(m_poly), Some(m_commit), [alpha, beta])
        } else {
            let [alpha, beta]: [P::ScalarField; 2] = transcript
                .squeeze_scalar_challenges::<P>(b"alpha beta", 2)?
                .try_into()
                .map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Couldn't convert to fixed length array".to_string(),
                    )
                })?;
            (None, None, [alpha, beta])
        };

        let gkr_proof = batch_prove_gkr::<P, _>(&gkr_p_polys, &gkr_q_polys, &mut transcript)?;

        // Now we have done the GKR proof we run a final zero check on the underlying products of MLEs.
        let epsilon = transcript.squeeze_scalar_challenge::<P>(b"epsilon")?;

        let gkr_point = &gkr_proof.sumcheck_proofs.last().as_ref().unwrap().point;

        let eq_poly = Arc::new(build_eq_x_r(gkr_point));
        let challenges = MLEChallenges {
            beta,
            gamma,
            alpha,
            tau,
        };

        let sumcheck_vp = build_sumcheck_poly(
            &wire_polys,
            &pk.selector_oracles,
            &pk.permutation_oracles,
            &arc_pi_poly,
            eq_poly.clone(),
            &pk.verifying_key.gate_info,
            &challenges,
            epsilon,
            pk.lookup_proving_key.as_ref(),
            m_poly.clone(),
        )?;

        let sumcheck_proof = <VPSumCheck<P> as SumCheck<P>>::prove(&sumcheck_vp, &mut transcript)?;

        // Extract the final point from the SumCheck proof. The unwrap is safe here because we ensure that we have at least one round.

        let zero_check_point = &sumcheck_proof.point;

        let mut wire_evals = sumcheck_proof.poly_evals[..5].to_vec();
        let selector_evals = sumcheck_proof.poly_evals[5..22].to_vec();

        let mut permutation_evals = sumcheck_proof.poly_evals[23..28].to_vec();
        let mut combiner_set = Vec::<Arc<DenseMultilinearExtension<P::ScalarField>>>::new();

        // Add the wires to the accumulator.

        for poly in wire_polys.iter() {
            combiner_set.push(poly.clone());
        }

        // Add the selectors to the accumulator.

        for poly in pk.selector_oracles.iter().take(17) {
            combiner_set.push(poly.clone());
        }

        // Add the permutations to the accumulator.
        for poly in pk.permutation_oracles.iter() {
            combiner_set.push(poly.clone());
        }

        // Now we handle lookup related subroutines if support lookup.
        let lookup_proof = if let (Some(m_poly), Some(m_commit)) = (m_poly, m_commit) {
            wire_evals.push(sumcheck_proof.poly_evals[27]);
            permutation_evals = [
                &sumcheck_proof.poly_evals[23..27],
                &sumcheck_proof.poly_evals[28..30],
            ]
            .concat();

            combiner_set.push(m_poly.clone());
            let m_poly_eval = sumcheck_proof.poly_evals[35];

            let range_table_eval = sumcheck_proof.poly_evals[32];
            combiner_set.push(
                pk.lookup_proving_key
                    .as_ref()
                    .unwrap()
                    .range_table_mle
                    .clone(),
            );
            let key_table_eval = sumcheck_proof.poly_evals[34];

            combiner_set.push(
                pk.lookup_proving_key
                    .as_ref()
                    .unwrap()
                    .key_table_mle
                    .clone(),
            );
            let table_dom_sep_eval = sumcheck_proof.poly_evals[33];
            combiner_set.push(
                pk.lookup_proving_key
                    .as_ref()
                    .unwrap()
                    .table_dom_sep_mle
                    .clone(),
            );
            let q_dom_sep_eval = sumcheck_proof.poly_evals[31];
            combiner_set.push(
                pk.lookup_proving_key
                    .as_ref()
                    .unwrap()
                    .q_dom_sep_mle
                    .clone(),
            );

            let q_lookup_eval = sumcheck_proof.poly_evals[30];
            combiner_set.push(pk.lookup_proving_key.as_ref().unwrap().q_lookup_mle.clone());

            let evals = MLELookupEvals::<PCS> {
                range_table_eval,
                key_table_eval,
                table_dom_sep_eval,
                q_dom_sep_eval,
                q_lookup_eval,
                m_poly_eval,
            };

            Some(MLELookupProof::<PCS> {
                m_poly_comm: m_commit,
                lookup_evals: evals,
            })
        } else {
            None
        };

        // Append evals and lookup_proof.poly_evals to the transcript.
        for eval in wire_evals
            .iter()
            .chain(&selector_evals)
            .chain(&permutation_evals)
        {
            transcript.push_message(b"eval", eval)?;
        }
        if let Some(lookup_proof) = lookup_proof.clone() {
            for eval in [
                lookup_proof.lookup_evals.m_poly_eval,
                lookup_proof.lookup_evals.range_table_eval,
                lookup_proof.lookup_evals.key_table_eval,
                lookup_proof.lookup_evals.table_dom_sep_eval,
                lookup_proof.lookup_evals.q_dom_sep_eval,
                lookup_proof.lookup_evals.q_lookup_eval,
            ]
            .iter()
            {
                transcript.push_message(b"lookup eval", eval)?;
            }
        }

        let evals = MLEProofEvals::<PCS> {
            wire_evals,
            selector_evals,
            permutation_evals,
        };

        let delta = transcript.squeeze_scalar_challenge::<P>(b"delta")?;

        let (_, poly_evals) = combiner_set.iter().try_fold(
            (
                P::ScalarField::one(),
                vec![P::ScalarField::zero(); 1 << num_vars],
            ),
            |acc, poly| {
                Result::<(P::ScalarField, Vec<P::ScalarField>), PlonkError>::Ok((
                    acc.0 * delta,
                    add_vecs(&acc.1, &scale_mle(poly, acc.0))?,
                ))
            },
        )?;

        let opening_point = zero_check_point.clone();

        Ok((
            SAMLEProof {
                wire_commitments: wire_comms,
                gkr_proof,
                sumcheck_proof,
                lookup_proof,
                evals,
                opening_point,
                polynomial: Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    num_vars, poly_evals,
                )),
            },
            transcript,
        ))
    }

    /// Verify an MLEPlonk Proof.
    pub fn verify<F, P, R, T>(
        proof: &MLEProof<PCS>,
        vk: &MLEVerifyingKey<PCS>,
        public_input: &[P::ScalarField],
        _rng: &mut R,
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<bool, PlonkError>
    where
        F: PrimeField + RescueParameter,
        PCS: PolynomialCommitmentScheme<
            Evaluation = P::ScalarField,
            Point = Vec<P::ScalarField>,
            Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
            Commitment = Affine<P>,
        >,
        P: HasTEForm<BaseField = F>,
        P::ScalarField: EmulationConfig<F>,
        R: RngCore + CryptoRng,
        T: Transcript,
    {
        let mut transcript: T = if let Some(msg) = extra_transcript_init_msg {
            T::new_with_initial_message::<_, P>(&msg)?
        } else {
            T::new_transcript(b"mle_plonk")
        };

        let num_vars = vk.num_vars as usize;
        let n = 1usize << num_vars;
        let shared = MLEProofShared::from(proof);
        check_proof_shape::<F, PCS, P>(&shared, vk, public_input, num_vars)?;

        let mut pi_evals = public_input.to_vec();
        pi_evals.resize(n, P::ScalarField::zero());
        let pi_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, pi_evals);

        let challenges =
            MLEChallenges::<P::ScalarField>::new(proof, public_input, &mut transcript)?;

        let gkr_deferred_check = batch_verify_gkr::<P, _>(&proof.gkr_proof, &mut transcript)?;

        // Now we check the initial value of the SumCheck proof against the values output by the GKR protocol.
        // The gate equation sums to zero over the boolean hypercube so that contributes nothing to the sum.
        // There are either 4 or 8 evaluations depending on if lookup is supported.
        // Evals are ordered as:
        // gkr_evals[0] = perm_numerator_1
        // gkr_evals[1] = perm_numerator_2
        // gkr_evals[2] = perm_denominator_1
        // gkr_evals[3] = perm_denominator_2
        // gkr_evals[4] = -beta
        // gkr_evals[5] = beta * m_poly
        // gkr_evals[6] = (beta * alpha - beta * lookup_wire)
        // gkr_evals[7] = (beta * alpha - beta * lookup_table)
        let gkr_evals = gkr_deferred_check.evals();

        let epsilon = transcript.squeeze_scalar_challenge::<P>(b"epsilon")?;

        let initial_sumcheck_eval = if gkr_evals.len() == 8 {
            let mut combiner = P::ScalarField::one();
            let perm_eval = gkr_evals[..4]
                .iter()
                .fold(P::ScalarField::zero(), |acc, eval| {
                    combiner *= epsilon;
                    acc + *eval * combiner
                });

            combiner *= epsilon;
            combiner /= challenges.beta;
            let beta_alpha = challenges.beta * challenges.alpha;
            perm_eval
                + combiner
                    * ((beta_alpha - gkr_evals[6])
                        + epsilon * (beta_alpha - gkr_evals[7])
                        + epsilon * epsilon * gkr_evals[5])
        } else if gkr_evals.len() == 4 {
            let mut combiner = P::ScalarField::one();
            gkr_evals[..4]
                .iter()
                .fold(P::ScalarField::zero(), |acc, eval| {
                    combiner *= epsilon;
                    acc + *eval * combiner
                })
        } else {
            return Err(PlonkError::SnarkError(SnarkError::ParameterError(
                "Invalid number of GKR evaluations".to_string(),
            )));
        };

        if initial_sumcheck_eval != proof.sumcheck_proof.eval {
            return Err(PlonkError::SnarkError(SnarkError::ParameterError(format!(
                "SumCheck check failed. Expected {}, got {}",
                proof.sumcheck_proof.eval, initial_sumcheck_eval
            ))));
        }

        let sumcheck_deferred_check =
            <VPSumCheck<P> as SumCheck<P>>::verify(&proof.sumcheck_proof, &mut transcript)?;
        let gkr_point = gkr_deferred_check.point();
        let zero_check_point = &sumcheck_deferred_check.point;

        let zc_eq_eval = eq_eval(gkr_point, zero_check_point)?;

        let pi_poly_eval =
            pi_poly
                .evaluate(zero_check_point)
                .ok_or(PolynomialError::ParameterError(
                    "Could not evaluate pi poly".to_string(),
                ))?;

        let mut comms = Vec::new();

        let mut evals = Vec::new();

        for (comm, eval) in proof
            .wire_commitments
            .iter()
            .zip(proof.evals.wire_evals.iter())
        {
            comms.push(*comm);

            evals.push(*eval);
        }

        for (comm, eval) in vk
            .selector_commitments
            .iter()
            .take(17)
            .zip(proof.evals.selector_evals.iter())
        {
            comms.push(*comm);

            evals.push(*eval)
        }

        for (comm, eval) in vk
            .permutation_commitments
            .iter()
            .zip(proof.evals.permutation_evals.iter())
        {
            comms.push(*comm);

            evals.push(*eval);
        }

        if let Some(lookup_proof) = &proof.lookup_proof {
            comms.push(lookup_proof.m_poly_comm);

            evals.push(lookup_proof.lookup_evals.m_poly_eval);

            comms.push(vk.lookup_verifying_key.as_ref().unwrap().range_table_comm);

            evals.push(lookup_proof.lookup_evals.range_table_eval);

            comms.push(vk.lookup_verifying_key.as_ref().unwrap().key_table_comm);

            evals.push(lookup_proof.lookup_evals.key_table_eval);

            comms.push(vk.lookup_verifying_key.as_ref().unwrap().table_dom_sep_comm);

            evals.push(lookup_proof.lookup_evals.table_dom_sep_eval);

            comms.push(vk.lookup_verifying_key.as_ref().unwrap().q_dom_sep_comm);

            evals.push(lookup_proof.lookup_evals.q_dom_sep_eval);

            comms.push(vk.lookup_verifying_key.as_ref().unwrap().q_lookup_comm);

            evals.push(lookup_proof.lookup_evals.q_lookup_eval);
        }

        // Append evals and lookup_proof.poly_evals to the transcript.
        for eval in evals.iter() {
            transcript.push_message(b"eval", eval)?;
        }

        let delta = transcript.squeeze_scalar_challenge::<P>(b"delta")?;

        let full_challenges = FullMLEChallenges::from_parts(challenges, delta, epsilon);

        let zero_check_calc_eval = build_zerocheck_eval(
            &proof.evals,
            proof
                .lookup_proof
                .as_ref()
                .map(|lookup_proof| &lookup_proof.lookup_evals),
            &vk.gate_info,
            &full_challenges,
            pi_poly_eval,
            zc_eq_eval,
        );

        if zero_check_calc_eval != sumcheck_deferred_check.eval {
            return Err(PlonkError::SnarkError(SnarkError::ParameterError(format!(
                "ZeroCheck check failed. Expected {}, got {}",
                sumcheck_deferred_check.eval, zero_check_calc_eval
            ))));
        }

        let delta_powers = (0..comms.len() as u64)
            .map(|i| delta.pow([i]))
            .collect::<Vec<P::ScalarField>>();

        let eval = evals
            .iter()
            .zip(delta_powers.iter())
            .fold(P::ScalarField::zero(), |acc, (eval, delta)| {
                acc + *eval * *delta
            });

        let comm = Projective::<P>::msm_bigint(
            &comms,
            &delta_powers
                .iter()
                .map(|d| d.into_bigint())
                .collect::<Vec<_>>(),
        )
        .into_affine();

        let result = PCS::verify(
            &vk.pcs_verifier_params,
            &comm,
            zero_check_point,
            &eval,
            &proof.opening_proof,
        )?;

        Ok(result)
    }

    /// Verify an MLEPlonk Proof.
    pub fn verify_recursive_proof<F, P, R, T>(
        recursion_output: &RecursiveOutput<PCS, Self, T>,
        opening_proof: &PCS::Proof,
        vk: &MLEVerifyingKey<PCS>,
        public_input: P::ScalarField,
        _rng: &mut R,
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<bool, PlonkError>
    where
        F: PrimeField + RescueParameter,
        PCS: Accumulation<
            Evaluation = P::ScalarField,
            Point = Vec<P::ScalarField>,
            Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
            Commitment = Affine<P>,
        >,
        P: HasTEForm<BaseField = F>,
        P::ScalarField: EmulationConfig<F>,
        R: RngCore + CryptoRng,
        T: Transcript + ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize,
    {
        let mut transcript: T = if let Some(msg) = extra_transcript_init_msg {
            T::new_with_initial_message::<_, P>(&msg)?
        } else {
            T::new_transcript(b"mle_plonk")
        };

        let proof = &recursion_output.proof;
        let num_vars = vk.num_vars as usize;
        let n = 1usize << num_vars;

        let shared = MLEProofShared::from(proof);
        check_proof_shape(&shared, vk, &[public_input], num_vars)?;

        let mut pi_evals = vec![public_input];
        pi_evals.resize(n, P::ScalarField::zero());
        let pi_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, pi_evals);

        let challenges = MLEChallenges::<P::ScalarField>::new_recursion(
            proof,
            &[public_input],
            &mut transcript,
        )?;

        let gkr_deferred_check = batch_verify_gkr::<P, _>(&proof.gkr_proof, &mut transcript)?;

        // Now we check the initial value of the SumCheck proof against the values output by the GKR protocol.
        // The gate equation sums to zero over the boolean hypercube so that contributes nothing to the sum.
        // There are either 4 or 8 evaluations depending on if lookup is supported.
        // Evals are ordered as:
        // gkr_evals[0] = perm_numerator_1
        // gkr_evals[1] = perm_numerator_2
        // gkr_evals[2] = perm_denominator_1
        // gkr_evals[3] = perm_denominator_2
        // gkr_evals[4] = -beta
        // gkr_evals[5] = beta * m_poly
        // gkr_evals[6] = (beta * alpha - beta * lookup_wire)
        // gkr_evals[7] = (beta * alpha - beta * lookup_table)
        let gkr_evals = gkr_deferred_check.evals();

        let epsilon = transcript.squeeze_scalar_challenge::<P>(b"epsilon")?;

        let initial_sumcheck_eval = if gkr_evals.len() == 8 {
            let mut combiner = P::ScalarField::one();
            let perm_eval = gkr_evals[..4]
                .iter()
                .fold(P::ScalarField::zero(), |acc, eval| {
                    combiner *= epsilon;
                    acc + *eval * combiner
                });

            combiner *= epsilon;
            combiner /= challenges.beta;
            let beta_alpha = challenges.beta * challenges.alpha;
            perm_eval
                + combiner
                    * ((beta_alpha - gkr_evals[6])
                        + epsilon * (beta_alpha - gkr_evals[7])
                        + epsilon * epsilon * gkr_evals[5])
        } else if gkr_evals.len() == 4 {
            let mut combiner = P::ScalarField::one();
            gkr_evals[..4]
                .iter()
                .fold(P::ScalarField::zero(), |acc, eval| {
                    combiner *= epsilon;
                    acc + *eval * combiner
                })
        } else {
            return Err(PlonkError::SnarkError(SnarkError::ParameterError(
                "Invalid number of GKR evaluations".to_string(),
            )));
        };

        if initial_sumcheck_eval != proof.sumcheck_proof.eval {
            return Err(PlonkError::SnarkError(SnarkError::ParameterError(format!(
                "SumCheck check failed. Expected {}, got {}",
                proof.sumcheck_proof.eval, initial_sumcheck_eval
            ))));
        }

        let sumcheck_deferred_check =
            <VPSumCheck<P> as SumCheck<P>>::verify(&proof.sumcheck_proof, &mut transcript)?;

        let zero_check_point = &sumcheck_deferred_check.point;

        let zc_eq_eval = eq_eval(zero_check_point, gkr_deferred_check.point())?;

        let pi_poly_eval =
            pi_poly
                .evaluate(zero_check_point)
                .ok_or(PolynomialError::ParameterError(
                    "Could not evaluate pi poly".to_string(),
                ))?;

        let mut comms = Vec::new();

        let mut evals = Vec::new();

        for (comm, eval) in proof
            .wire_commitments
            .iter()
            .zip(proof.evals.wire_evals.iter())
        {
            comms.push(*comm);

            evals.push(*eval);
        }

        for (comm, eval) in vk
            .selector_commitments
            .iter()
            .take(17)
            .zip(proof.evals.selector_evals.iter())
        {
            comms.push(*comm);

            evals.push(*eval)
        }

        for (comm, eval) in vk
            .permutation_commitments
            .iter()
            .zip(proof.evals.permutation_evals.iter())
        {
            comms.push(*comm);

            evals.push(*eval);
        }

        if let Some(lookup_proof) = &proof.lookup_proof {
            comms.push(lookup_proof.m_poly_comm);

            evals.push(lookup_proof.lookup_evals.m_poly_eval);

            comms.push(vk.lookup_verifying_key.as_ref().unwrap().range_table_comm);

            evals.push(lookup_proof.lookup_evals.range_table_eval);

            comms.push(vk.lookup_verifying_key.as_ref().unwrap().key_table_comm);

            evals.push(lookup_proof.lookup_evals.key_table_eval);

            comms.push(vk.lookup_verifying_key.as_ref().unwrap().table_dom_sep_comm);

            evals.push(lookup_proof.lookup_evals.table_dom_sep_eval);

            comms.push(vk.lookup_verifying_key.as_ref().unwrap().q_dom_sep_comm);

            evals.push(lookup_proof.lookup_evals.q_dom_sep_eval);

            comms.push(vk.lookup_verifying_key.as_ref().unwrap().q_lookup_comm);

            evals.push(lookup_proof.lookup_evals.q_lookup_eval);
        }

        // Append evals and lookup_proof.poly_evals to the transcript.
        for eval in evals.iter() {
            transcript.push_message(b"eval", eval)?;
        }

        let delta = transcript.squeeze_scalar_challenge::<P>(b"delta")?;

        let full_challenges = FullMLEChallenges::from_parts(challenges, delta, epsilon);

        let zero_check_calc_eval = build_zerocheck_eval(
            &proof.evals,
            proof
                .lookup_proof
                .as_ref()
                .map(|lookup_proof| &lookup_proof.lookup_evals),
            &vk.gate_info,
            &full_challenges,
            pi_poly_eval,
            zc_eq_eval,
        );

        if zero_check_calc_eval != sumcheck_deferred_check.eval {
            return Err(PlonkError::SnarkError(SnarkError::ParameterError(format!(
                "ZeroCheck check failed. Expected {}, got {}",
                sumcheck_deferred_check.eval, zero_check_calc_eval
            ))));
        }

        let delta_powers = (0..comms.len() as u64)
            .map(|i| delta.pow([i]))
            .collect::<Vec<P::ScalarField>>();

        let eval = evals
            .iter()
            .zip(delta_powers.iter())
            .fold(P::ScalarField::zero(), |acc, (eval, delta)| {
                acc + *eval * *delta
            });

        let comm = Projective::<P>::msm_bigint(
            &comms,
            &delta_powers
                .iter()
                .map(|d| d.into_bigint())
                .collect::<Vec<_>>(),
        )
        .into_affine();

        let result = PCS::verify(
            &vk.pcs_verifier_params,
            &comm,
            &proof.opening_point,
            &eval,
            opening_proof,
        )?;

        Ok(result)
    }
}

fn check_proof_shape<F, PCS, P>(
    proof: &MLEProofShared<PCS>,
    vk: &MLEVerifyingKey<PCS>,
    public_input: &[P::ScalarField],
    num_vars: usize,
) -> Result<(), PlonkError>
where
    PCS: PolynomialCommitmentScheme<
        Evaluation = P::ScalarField,
        Point = Vec<P::ScalarField>,
        Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
        Commitment = Affine<P>,
    >,
    P: HasTEForm<BaseField = F>,
    F: PrimeField + RescueParameter,
{
    // Public inputs must match VK
    if public_input.len() != vk.num_inputs {
        return Err(PlonkError::SnarkError(SnarkError::ParameterError(format!(
            "Unexpected number of public inputs: expected {}, got {}",
            vk.num_inputs,
            public_input.len()
        ))));
    }

    let expected_wires = vk.permutation_commitments.len();
    if proof.wire_commitments.len() != expected_wires {
        return Err(PlonkError::SnarkError(SnarkError::ParameterError(format!(
            "Unexpected number of wire commitments: expected {}, got {}",
            expected_wires,
            proof.wire_commitments.len()
        ))));
    }

    if proof.evals.wire_evals.len() != expected_wires {
        return Err(PlonkError::SnarkError(SnarkError::ParameterError(format!(
            "Unexpected number of wire evals: expected {}, got {}",
            expected_wires,
            proof.evals.wire_evals.len()
        ))));
    }

    let expected_selectors = vk.selector_commitments.len();
    if proof.evals.selector_evals.len() != expected_selectors {
        return Err(PlonkError::SnarkError(SnarkError::ParameterError(format!(
            "Unexpected number of selector evals: expected {}, got {}",
            expected_selectors,
            proof.evals.selector_evals.len()
        ))));
    }

    if proof.evals.permutation_evals.len() != vk.permutation_commitments.len() {
        return Err(PlonkError::SnarkError(SnarkError::ParameterError(format!(
            "Unexpected number of permutation evals: expected {}, got {}",
            vk.permutation_commitments.len(),
            proof.evals.permutation_evals.len()
        ))));
    }

    let lookup_expected = vk.lookup_verifying_key.is_some();
    match (lookup_expected, proof.lookup_proof.as_ref()) {
        (true, None) => {
            return Err(PlonkError::SnarkError(SnarkError::ParameterError(
                "Lookup was enabled in VK but lookup proof is missing".to_string(),
            )));
        },
        (false, Some(_)) => {
            return Err(PlonkError::SnarkError(SnarkError::ParameterError(
                "Lookup proof was provided but VK does not enable lookup".to_string(),
            )));
        },
        _ => {},
    }

    if proof.gkr_proof.challenge_point().len() != num_vars {
        return Err(PlonkError::SnarkError(SnarkError::ParameterError(format!(
            "GKR challenge point dimension mismatch: expected {}, got {}",
            num_vars,
            proof.gkr_proof.challenge_point().len()
        ))));
    }
    Ok(())
}

#[cfg(test)]
/// test module
pub mod tests {
    use ark_ec::{
        pairing::Pairing,
        short_weierstrass::{Affine, Projective},
    };

    use jf_primitives::rescue::sponge::RescueCRHF;
    use jf_relation::{Circuit, PlonkCircuit, PlonkType};
    use jf_utils::test_rng;
    use nf_curves::grumpkin::{Fq as FqGrumpkin, Grumpkin};

    use crate::{
        nightfall::{mle::zeromorph::zeromorph_protocol::Zeromorph, UnivariateIpaPCS},
        proof_system::UniversalRecursiveSNARK,
        transcript::RescueTranscript,
    };

    use super::*;

    #[test]
    fn test_mle_plonk_proof_system() -> Result<(), PlonkError> {
        test_recursive_plonk_proof_system_helper::<
            Grumpkin,
            FqGrumpkin,
            _,
            Zeromorph<UnivariateIpaPCS<Grumpkin>>,
        >(PlonkType::UltraPlonk)?;
        test_plonk_proof_system_helper::<
            Grumpkin,
            FqGrumpkin,
            _,
            Zeromorph<UnivariateIpaPCS<Grumpkin>>,
        >(PlonkType::UltraPlonk)?;
        test_plonk_proof_system_helper::<
            Grumpkin,
            FqGrumpkin,
            _,
            Zeromorph<UnivariateIpaPCS<Grumpkin>>,
        >(PlonkType::TurboPlonk)
    }

    fn test_plonk_proof_system_helper<E, F, P, PCS>(plonk_type: PlonkType) -> Result<(), PlonkError>
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
        F: RescueParameter,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
        PCS: Accumulation<
            Evaluation = P::ScalarField,
            Point = Vec<P::ScalarField>,
            Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
            Commitment = Affine<P>,
        >,
    {
        // 1. Simulate universal setup
        let rng = &mut test_rng();

        let srs = PCS::gen_srs_for_testing(rng, 15)?;

        // 2. Create circuits
        let circuits = (0..6)
            .map(|i| {
                let m = 2 + i / 3;
                let a0 = 1 + i % 3;
                gen_circuit_for_test::<P::ScalarField>(m, a0, plonk_type, false)
            })
            .collect::<Result<Vec<_>, PlonkError>>()?;
        // 3. Preprocessing
        let (pk1, vk1) = MLEPlonk::<PCS>::preprocess_helper::<
            P::ScalarField,
            PlonkCircuit<P::ScalarField>,
        >(&circuits[0], &srs)?;
        let (pk2, vk2) = MLEPlonk::<PCS>::preprocess_helper::<
            P::ScalarField,
            PlonkCircuit<P::ScalarField>,
        >(&circuits[3], &srs)?;
        // 4. Proving
        let mut proofs = vec![];

        for (i, cs) in circuits.iter().enumerate() {
            let pk_ref = if i < 3 { &pk1 } else { &pk2 };

            proofs.push(
                MLEPlonk::<PCS>::prove::<_, _, _, RescueTranscript<F>>(cs, pk_ref, None).unwrap(),
            );
        }

        // 5. Verification
        let public_inputs: Vec<Vec<E::ScalarField>> = circuits
            .iter()
            .map(|cs| cs.public_input())
            .collect::<Result<Vec<Vec<E::ScalarField>>, _>>(
        )?;
        for (i, proof) in proofs.iter().enumerate() {
            let vk_ref = if i < 3 { &vk1 } else { &vk2 };
            assert!(MLEPlonk::<PCS>::verify::<_, _, _, RescueTranscript<F>>(
                proof,
                vk_ref,
                &public_inputs[i],
                rng,
                None
            )
            .unwrap());
            // Inconsistent proof should fail the verification.
            let mut bad_pub_input = public_inputs[i].clone();
            bad_pub_input[0] = E::ScalarField::from(0u8);
            assert!(MLEPlonk::<PCS>::verify::<_, _, _, RescueTranscript<F>>(
                proof,
                vk_ref,
                &bad_pub_input,
                rng,
                None
            )
            .is_err());

            // Incorrect proof [W_z] = 0, [W_z*g] = 0
            // attack against some vulnerable implementation described in:
            // https://cryptosubtlety.medium.com/00-8d4adcf4d255
            let mut bad_proof = proof.clone();
            bad_proof.opening_proof = PCS::Proof::default();

            assert!(MLEPlonk::<PCS>::verify::<_, _, _, RescueTranscript<F>>(
                &bad_proof,
                vk_ref,
                &public_inputs[i],
                rng,
                None
            )
            .is_err());
        }

        Ok(())
    }

    fn test_recursive_plonk_proof_system_helper<E, F, P, PCS>(
        plonk_type: PlonkType,
    ) -> Result<(), PlonkError>
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
        F: RescueParameter,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
        PCS: Accumulation<
            Evaluation = P::ScalarField,
            Point = Vec<P::ScalarField>,
            Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
            Commitment = Affine<P>,
        >,
    {
        // 1. Simulate universal setup
        let rng = &mut test_rng();

        let srs = PCS::gen_srs_for_testing(rng, 10)?;

        // 2. Create circuits
        let circuits = (0..6)
            .map(|i| {
                let m = 2 + i / 3;
                let a0 = 1 + i % 3;
                gen_circuit_for_test::<P::ScalarField>(m, a0, plonk_type, true)
            })
            .collect::<Result<Vec<_>, PlonkError>>()?;
        // 3. Preprocessing
        let (pk1, vk1) = MLEPlonk::<PCS>::preprocess_helper::<
            P::ScalarField,
            PlonkCircuit<P::ScalarField>,
        >(&circuits[0], &srs)?;
        let (pk2, vk2) = MLEPlonk::<PCS>::preprocess_helper::<
            P::ScalarField,
            PlonkCircuit<P::ScalarField>,
        >(&circuits[3], &srs)?;
        // 4. Proving
        let mut proofs = vec![];

        for (i, cs) in circuits.iter().enumerate() {
            let pk_ref = if i < 3 { &pk1 } else { &pk2 };

            proofs.push(
                MLEPlonk::<PCS>::recursive_prove::<_, _, RescueTranscript<F>>(
                    rng, cs, pk_ref, None, false,
                )
                .unwrap(),
            );
        }

        // 5. Verification
        let public_inputs: Vec<Vec<E::ScalarField>> = circuits
            .iter()
            .map(|cs| cs.public_input())
            .collect::<Result<Vec<Vec<E::ScalarField>>, _>>(
        )?;
        for (i, proof) in proofs.iter().enumerate() {
            let vk_ref = if i < 3 { &vk1 } else { &vk2 };
            let pk_ref = if i < 3 { &pk1 } else { &pk2 };

            let (opening_proof, _) = PCS::open(
                &pk_ref.pcs_prover_params,
                &proof.proof.polynomial,
                &proof.proof.opening_point,
            )?;

            assert!(
                MLEPlonk::<PCS>::verify_recursive_proof::<_, _, _, RescueTranscript<F>>(
                    proof,
                    &opening_proof,
                    vk_ref,
                    public_inputs[i][0],
                    rng,
                    None
                )
                .unwrap()
            );
            // Inconsistent proof should fail the verification.

            assert!(
                MLEPlonk::<PCS>::verify_recursive_proof::<_, _, _, RescueTranscript<F>>(
                    proof,
                    &opening_proof,
                    vk_ref,
                    E::ScalarField::zero(),
                    rng,
                    None
                )
                .is_err()
            );

            // Incorrect proof [W_z] = 0, [W_z*g] = 0
            // attack against some vulnerable implementation described in:
            // https://cryptosubtlety.medium.com/00-8d4adcf4d255
            let default_opening = PCS::Proof::default();

            assert!(
                MLEPlonk::<PCS>::verify_recursive_proof::<_, _, _, RescueTranscript<F>>(
                    proof,
                    &default_opening,
                    vk_ref,
                    public_inputs[i][0],
                    rng,
                    None
                )
                .is_err()
            );
        }

        Ok(())
    }

    /// Different `m`s lead to different circuits.
    /// Different `a0`s lead to different witness values.
    /// For UltraPlonk circuits, `a0` should be less than or equal to `m+1`
    pub fn gen_circuit_for_test<F: RescueParameter>(
        m: usize,
        a0: usize,
        plonk_type: PlonkType,
        is_recursive: bool,
    ) -> Result<PlonkCircuit<F>, PlonkError> {
        let range_bit_len = 5;
        let mut cs: PlonkCircuit<F> = match plonk_type {
            PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
            PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(range_bit_len),
        };
        // Create variables
        let mut a = vec![];
        for i in a0..(a0 + 4 * m) {
            a.push(cs.create_variable(F::from(i as u64))?);
        }
        let b = [
            cs.create_public_variable(F::from(m as u64 * 2))?,
            cs.create_public_variable(F::from(a0 as u64 * 2 + m as u64 * 4 - 1))?,
        ];
        let c = cs.create_public_variable(
            (cs.witness(b[1])? + cs.witness(a[0])?) * (cs.witness(b[1])? - cs.witness(a[0])?),
        )?;

        // Create gates:
        // 1. a0 + ... + a_{4*m-1} = b0 * b1
        // 2. (b1 + a0) * (b1 - a0) = c
        // 3. b0 = 2 * m
        let mut acc = cs.zero();
        a.iter().for_each(|&elem| acc = cs.add(acc, elem).unwrap());
        let b_mul = cs.mul(b[0], b[1])?;
        cs.enforce_equal(acc, b_mul)?;
        let b1_plus_a0 = cs.add(b[1], a[0])?;
        let b1_minus_a0 = cs.sub(b[1], a[0])?;
        cs.mul_gate(b1_plus_a0, b1_minus_a0, c)?;
        cs.enforce_constant(b[0], F::from(m as u64 * 2))?;

        if plonk_type == PlonkType::UltraPlonk {
            // Create range gates
            // 1. range_table = {0, 1, ..., 31}
            // 2. a_i \in range_table for i = 0..m-1
            // 3. b0 \in range_table
            for &var in a.iter().take(m) {
                cs.add_range_check_variable(var)?;
            }
            cs.add_range_check_variable(b[0])?;

            // Create variable table lookup gates
            // 1. table = [(a0, a2), (a1, a3), (b0, a0)]
            let table_vars = [(a[0], a[2]), (a[1], a[3]), (b[0], a[0])];
            // 2. lookup_witness = [(1, a0+1, a0+3), (2, 2m, a0)]
            let key0 = cs.one();
            let key1 = cs.create_variable(F::from(2u8))?;
            let two_m = cs.create_public_variable(F::from(m as u64 * 2))?;
            let a1 = cs.add_constant(a[0], &F::one())?;
            let a3 = cs.add_constant(a[0], &F::from(3u8))?;
            let lookup_vars = [(key0, a1, a3), (key1, two_m, a[0])];
            cs.create_table_and_lookup_variables(&lookup_vars, &table_vars)?;
        }

        // Finalize the circuit.
        if is_recursive {
            cs.finalize_for_recursive_mle_arithmetization::<RescueCRHF<F>>()?;
        } else {
            cs.finalize_for_mle_arithmetization()?;
        }
        Ok(cs)
    }
}
