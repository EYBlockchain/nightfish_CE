//! This module contains the code for the final decider circuit in the recursive prover.
//! This circuit will partially verify two Grumpkin proofs together and accumulate them
//! together with the existing Grumpkin accumulators. It will then verify the opening proof
//! of this accumulator.

use super::{split_acc::SplitAccumulationInfo, Zmorph};
use crate::{
    nightfall::{
        accumulation::circuit::structs::EmulatedPCSInstanceVar,
        circuit::{
            plonk_partial_verifier::{
                emulated_eq_x_r_eval_circuit, EmulatedMLEChallenges, MLELookupEvaluationsVar,
                MLEProofEvaluationsVar, MLEVerifyingKeyVar, SAMLEProofVar,
            },
            subroutine_verifiers::{structs::EmulatedSumCheckProofVar, sumcheck::SumCheckGadget},
        },
        mle::mle_structs::{GateInfo, MLEVerifyingKey},
    },
    recursion::merge_functions::GrumpkinOutput,
    transcript::{rescue::RescueTranscriptVar, CircuitTranscript},
};
use ark_bn254::{Fq as Fq254, Fr as Fr254};
use ark_ff::{batch_inversion, Field};
use ark_std::{string::ToString, vec::Vec};
use jf_relation::{
    errors::CircuitError,
    gadgets::{ecc::Point, EmulatedVariable},
    PlonkCircuit,
};
use nf_curves::grumpkin::short_weierstrass::SWGrumpkin;

/// Function for taking the deferred evaluations output by verifying a GKR proof and combines them into the expected initial value of
/// the final SumCheck proof.
pub fn combine_emulated_gkr_evals(
    circuit: &mut PlonkCircuit<Fr254>,
    gkr_evals: &[EmulatedVariable<Fq254>],
    challenges: &EmulatedMLEChallenges<Fq254>,
) -> Result<EmulatedVariable<Fq254>, CircuitError> {
    // Now we check the initial value of the SumCheck proof against the values output by the GKR protocol.
    // The gate equation sums to zero over the boolean hypercube so that contributes nothing to the sum.
    // The rest should be:
    //         epsilon * (alpha - evals[2]) + epsilon^2 * (alpha - evals[3]) + epsilon^3 * (alpha - evals[6]) + epsilon^4 * (alpha - evals[7]) + epsilon^5 * evals[5]
    // Evals are ordered as:
    // evals[0] = -1
    // evals[1] = 1
    // evals[2] = (alpha - permutation numerator)
    // evals[3] = (alpha - permutation denominator)
    // evals[4] = -1
    // evals[5] = m_poly
    // evals[6] = (alpha - lookup_wire)
    // evals[7] = (alpha - lookup_table)
    let one_var = circuit.emulated_one();
    if gkr_evals.len() == 8 {
        let epsilon_sq = circuit.emulated_mul(&challenges.epsilon, &challenges.epsilon)?;
        let mut eval =
            circuit.emulated_mul_add(&challenges.epsilon, &gkr_evals[1], &gkr_evals[0])?;

        let tmp1 = circuit.emulated_mul(&challenges.epsilon, &gkr_evals[2])?;
        let tmp2 = circuit.emulated_mul(&epsilon_sq, &gkr_evals[3])?;
        let second_eval = circuit.emulated_add(&tmp1, &tmp2)?;
        eval = circuit.emulated_mul(&eval, &challenges.epsilon)?;
        eval = circuit.emulated_mul_add(&epsilon_sq, &second_eval, &eval)?;

        let beta_inv = circuit
            .emulated_witness(&challenges.beta)?
            .inverse()
            .ok_or(CircuitError::ParameterError(
                "Could not invert Beta challenge".to_string(),
            ))?;
        let beta_inv_var = circuit.create_emulated_variable(beta_inv)?;

        let beta_alpha = circuit.emulated_mul(&challenges.beta, &challenges.alpha)?;
        let coeff = circuit.emulated_mul(&challenges.epsilon, &beta_inv_var)?;
        circuit.emulated_mul_gate(&beta_inv_var, &challenges.beta, &one_var)?;
        let tmp_sub = circuit.emulated_sub(&beta_alpha, &gkr_evals[6])?;
        let third_eval = circuit.emulated_mul(&coeff, &tmp_sub)?;

        let epsilon_four = circuit.emulated_mul(&epsilon_sq, &epsilon_sq)?;

        eval = circuit.emulated_mul_add(&epsilon_four, &third_eval, &eval)?;

        let tmp_sub = circuit.emulated_sub(&beta_alpha, &gkr_evals[7])?;
        let fourth_eval = circuit.emulated_mul(&coeff, &tmp_sub)?;

        let epsilon_five = circuit.emulated_mul(&epsilon_four, &challenges.epsilon)?;

        eval = circuit.emulated_mul_add(&epsilon_five, &fourth_eval, &eval)?;

        let epsilon_seven = circuit.emulated_mul(&epsilon_sq, &epsilon_five)?;
        let final_coeff = circuit.emulated_mul(&epsilon_seven, &beta_inv_var)?;
        circuit.emulated_mul_add(&final_coeff, &gkr_evals[5], &eval)
    } else {
        Err(CircuitError::ParameterError(
            "Invalid number of GKR evaluations".to_string(),
        ))
    }
}

///Circuit corresponding to evaluating the gate equation.
/// Returns the variable corresponding to the evaluation.
pub fn emulated_eval_gate_equation(
    circuit: &mut PlonkCircuit<Fr254>,
    gate_info: &GateInfo<Fq254>,
    selector_evals_var: &[EmulatedVariable<Fq254>],
    wire_evals_var: &[EmulatedVariable<Fq254>],
    pub_input_poly_eval_var: EmulatedVariable<Fq254>,
) -> Result<EmulatedVariable<Fq254>, CircuitError> {
    let evals_var = [
        wire_evals_var,
        selector_evals_var,
        &[pub_input_poly_eval_var],
    ]
    .concat();
    let mut sum_var = circuit.emulated_zero();
    for (coeff, prod) in gate_info.products.iter() {
        let first_index = prod.first().ok_or(CircuitError::IndexError)?;
        let mut prod_var = evals_var[*first_index].clone();
        for index in prod.iter().skip(1) {
            prod_var = circuit.emulated_mul(&prod_var, &evals_var[*index])?;
        }
        prod_var = circuit.emulated_mul_constant(&prod_var, *coeff)?;
        sum_var = circuit.emulated_add(&sum_var, &prod_var)?;
    }
    Ok(sum_var)
}

/// Evaluate the permutation equation at a given point. Used by the verifier.
pub(crate) fn emulated_eval_permutation_equation(
    perm_evals: &[EmulatedVariable<Fq254>],
    wire_evals: &[EmulatedVariable<Fq254>],
    challenges: &EmulatedMLEChallenges<Fq254>,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<EmulatedVariable<Fq254>, CircuitError> {
    let EmulatedMLEChallenges { gamma, epsilon, .. } = challenges;
    let one_var = circuit.emulated_one();
    let zero_var = circuit.emulated_zero();
    let pairs = perm_evals
        .chunks(3)
        .zip(wire_evals.chunks(3))
        .map(|(perm_chunk, wire_chunk)| {
            let shift_wires = wire_chunk
                .iter()
                .map(|eval| circuit.emulated_sub(gamma, eval))
                .collect::<Result<Vec<EmulatedVariable<Fq254>>, CircuitError>>()?;
            let mut inv_wires = shift_wires
                .iter()
                .map(|eval| circuit.emulated_witness(eval))
                .collect::<Result<Vec<Fq254>, CircuitError>>()?;
            batch_inversion(&mut inv_wires);
            let denominator = shift_wires.iter().try_fold(one_var.clone(), |acc, eval| {
                circuit.emulated_mul(&acc, eval)
            })?;
            let inv_wire_vars = inv_wires
                .iter()
                .map(|&eval| circuit.create_emulated_variable(eval))
                .collect::<Result<Vec<EmulatedVariable<Fq254>>, CircuitError>>()?;
            let multiplicands = shift_wires
                .iter()
                .zip(inv_wire_vars.iter())
                .map(|(wire, inv_wire)| {
                    circuit.emulated_mul_gate(wire, inv_wire, &one_var)?;
                    circuit.emulated_mul(inv_wire, &denominator)
                })
                .collect::<Result<Vec<EmulatedVariable<Fq254>>, CircuitError>>()?;

            let numerator = perm_chunk
                .iter()
                .zip(multiplicands.iter())
                .try_fold(zero_var.clone(), |acc, (perm, wire)| {
                    circuit.emulated_mul_add(perm, wire, &acc)
                })?;
            Result::<[EmulatedVariable<Fq254>; 2], CircuitError>::Ok([numerator, denominator])
        })
        .collect::<Result<Vec<[EmulatedVariable<Fq254>; 2]>, CircuitError>>()?;

    let epsilon_sq = circuit.emulated_mul(epsilon, epsilon)?;
    let epsilon_cubed = circuit.emulated_mul(&epsilon_sq, epsilon)?;
    let epsilon_four = circuit.emulated_mul(&epsilon_sq, &epsilon_sq)?;

    let tmp1 = circuit.emulated_mul(&pairs[0][0], epsilon)?;
    let tmp2 = circuit.emulated_mul(&pairs[1][0], &epsilon_sq)?;
    let tmp3 = circuit.emulated_mul(&pairs[0][1], &epsilon_cubed)?;
    let tmp4 = circuit.emulated_mul(&pairs[1][1], &epsilon_four)?;
    circuit.emulated_batch_add(&[tmp1, tmp2, tmp3, tmp4])
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emulated_eval_lookup_equation(
    wire_evals: &[EmulatedVariable<Fq254>],
    range_table_eval: &EmulatedVariable<Fq254>,
    key_table_eval: &EmulatedVariable<Fq254>,
    table_dom_sep_eval: &EmulatedVariable<Fq254>,
    q_dom_sep_eval: &EmulatedVariable<Fq254>,
    q_lookup_eval: &EmulatedVariable<Fq254>,
    challenges: &EmulatedMLEChallenges<Fq254>,
    m_poly_eval: &EmulatedVariable<Fq254>,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<EmulatedVariable<Fq254>, CircuitError> {
    let EmulatedMLEChallenges { tau, epsilon, .. } = challenges;

    let epsilon_sq = circuit.emulated_mul(epsilon, epsilon)?;
    let epsilon_four = circuit.emulated_mul(&epsilon_sq, &epsilon_sq)?;
    let epsilon_five = circuit.emulated_mul(epsilon, &epsilon_four)?;

    let tmp1 = circuit.emulated_mul_add(&wire_evals[2], tau, &wire_evals[1])?;
    let tmp2 = circuit.emulated_mul_add(&tmp1, tau, &wire_evals[0])?;
    let tmp3 = circuit.emulated_mul_add(&tmp2, tau, q_dom_sep_eval)?;
    let tmp4 = circuit.emulated_mul(q_lookup_eval, &tmp3)?;
    let lookup_wire_eval = circuit.emulated_mul_add(&tmp4, tau, &wire_evals[5])?;

    let tmp1 = circuit.emulated_mul_add(&wire_evals[4], tau, &wire_evals[3])?;
    let tmp2 = circuit.emulated_mul_add(&tmp1, tau, key_table_eval)?;
    let tmp3 = circuit.emulated_mul_add(&tmp2, tau, table_dom_sep_eval)?;
    let tmp4 = circuit.emulated_mul(q_lookup_eval, &tmp3)?;
    let table_eval = circuit.emulated_mul_add(&tmp4, tau, range_table_eval)?;

    let tmp1 = circuit.emulated_mul_add(epsilon, m_poly_eval, &table_eval)?;
    let tmp2 = circuit.emulated_mul_add(&tmp1, epsilon, &lookup_wire_eval)?;

    circuit.emulated_mul(&epsilon_five, &tmp2)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_emulated_zerocheck_eval(
    evals: &MLEProofEvaluationsVar<Fq254>,
    lookup_evals: Option<&MLELookupEvaluationsVar<Fq254>>,
    gate_info: &GateInfo<Fq254>,
    challenges: &EmulatedMLEChallenges<Fq254>,
    pi_eval: &EmulatedVariable<Fq254>,
    eq_eval: &EmulatedVariable<Fq254>,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<EmulatedVariable<Fq254>, CircuitError> {
    let sc_eval = emulated_eval_gate_equation(
        circuit,
        gate_info,
        &evals.selector_evals[..17],
        &evals.wire_evals[..5],
        pi_eval.clone(),
    )?;

    let permutation_eval = emulated_eval_permutation_equation(
        &evals.permutation_evals,
        &evals.wire_evals,
        challenges,
        circuit,
    )?;

    let mut operands = ark_std::vec![sc_eval, permutation_eval];
    if let Some(lookup_evals) = lookup_evals {
        let lookup_eq_eval = emulated_eval_lookup_equation(
            &evals.wire_evals,
            &lookup_evals.range_table_eval,
            &lookup_evals.key_table_eval,
            &lookup_evals.table_dom_sep_eval,
            &lookup_evals.q_dom_sep_eval,
            &lookup_evals.q_lookup_eval,
            challenges,
            &lookup_evals.m_poly_eval,
            circuit,
        )?;

        operands.push(lookup_eq_eval);
    }
    let sc_eval = circuit.emulated_batch_add(&operands)?;
    circuit.emulated_mul(eq_eval, &sc_eval)
}

/// This function builds the batched evaluation for a single [`MLEPlonk`] proof.
/// It outputs the combined evaluation, together with the individual scalars.
pub(crate) fn build_emulated_mleplonk_eval(
    evals: &MLEProofEvaluationsVar<Fq254>,
    lookup_evals: Option<&MLELookupEvaluationsVar<Fq254>>,
    delta: &EmulatedVariable<Fq254>,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<(Vec<EmulatedVariable<Fq254>>, EmulatedVariable<Fq254>), CircuitError> {
    let mut evals_vec = evals.wire_evals.to_vec();
    evals_vec.extend_from_slice(&evals.selector_evals);
    evals_vec.extend_from_slice(&evals.permutation_evals);

    if let Some(lookup_evals) = lookup_evals {
        evals_vec.push(lookup_evals.m_poly_eval.clone());
        evals_vec.push(lookup_evals.range_table_eval.clone());
        evals_vec.push(lookup_evals.key_table_eval.clone());
        evals_vec.push(lookup_evals.table_dom_sep_eval.clone());
        evals_vec.push(lookup_evals.q_dom_sep_eval.clone());
        evals_vec.push(lookup_evals.q_lookup_eval.clone());
    }

    let mut scalars = Vec::<EmulatedVariable<Fq254>>::with_capacity(evals_vec.len());
    let zero_var = circuit.emulated_zero();
    let mut combiner = circuit.emulated_one();
    let eval = evals_vec.iter().try_fold(zero_var.clone(), |acc, eval| {
        let fixed_combiner = combiner.clone();
        scalars.push(combiner.clone());
        combiner = circuit.emulated_mul(&combiner, delta)?;
        circuit.emulated_mul_add(&fixed_combiner, eval, &acc)
    })?;

    Ok((scalars, eval))
}

pub(crate) type MLEProofMSMScalars = (Vec<EmulatedVariable<Fq254>>, EmulatedVariable<Fq254>);

/// This function verifies a single [`MLEPlonk`][crate::nightfall::mle::snark::MLEPlonk] proofs scalar arithmetic given the challenges
/// (that is it doesn't do any hashing). The output of this function if the scalars to be used in a fixed base MSM
/// to check the validity of the PI polynomial.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_mleplonk_emulated_scalar_arithmetic(
    circuit: &mut PlonkCircuit<Fr254>,
    proof: &SAMLEProofVar<Zmorph>,
    challenges: &EmulatedMLEChallenges<Fq254>,
    pi_eval: &EmulatedVariable<Fq254>,
    transcript: &mut RescueTranscriptVar<Fr254>,
    gate_info: &GateInfo<Fq254>,
) -> Result<MLEProofMSMScalars, CircuitError> {
    // First we verify the GKR proof and obtain the evaluations it outputs.
    let gkr_evals = proof
        .gkr_proof
        .verify_gkr_proof_emulated::<SWGrumpkin>(circuit, transcript)?;

    // Now we reconstruct the initial zerocheck evaluation and check it is correct.
    let initial_zerocheck_eval = combine_emulated_gkr_evals(circuit, &gkr_evals, challenges)?;

    circuit.enforce_emulated_var_equal(&initial_zerocheck_eval, &proof.sumcheck_proof.eval_var)?;

    // Now we verify the arithmetic of the SumCheck proof.
    let zerocheck_eval =
        circuit.verify_emulated_proof::<SWGrumpkin>(&proof.sumcheck_proof, transcript)?;

    let zc_point = &proof.sumcheck_proof.point_var;

    // Calculate the eq_eval.
    let eq_eval = emulated_eq_x_r_eval_circuit(
        circuit,
        zc_point,
        &proof
            .gkr_proof
            .sumcheck_proof_vars
            .last()
            .as_ref()
            .unwrap()
            .point_var,
    )?;

    // Calculate the same eval based on the supplied polynomial evaluations.
    let calculated_zerocheck_eval = build_emulated_zerocheck_eval(
        &proof.poly_evals_var,
        proof.lookup_proof_var.as_ref().map(|l| l.poly_evals()),
        gate_info,
        challenges,
        pi_eval,
        &eq_eval,
        circuit,
    )?;

    circuit.enforce_emulated_var_equal(&zerocheck_eval, &calculated_zerocheck_eval)?;

    // Finally calculate the final combined evaluation and return the scalars.
    build_emulated_mleplonk_eval(
        &proof.poly_evals_var,
        proof.lookup_proof_var.as_ref().map(|l| l.poly_evals()),
        &challenges.delta,
        circuit,
    )
}

type ScalarsAndEval = (Vec<EmulatedVariable<Fq254>>, EmulatedVariable<Fq254>);

/// We need a function that verifies the accumulation SumCheck for split accumulation
pub fn emulated_verify_split_accumulation(
    circuit: &mut PlonkCircuit<Fr254>,
    merged_proof_evals: &[EmulatedVariable<Fq254>],
    proof_eval_points: &[Vec<EmulatedVariable<Fq254>>],
    old_accs: &[EmulatedPCSInstanceVar<SWGrumpkin>],
    sumcheck_proof: &EmulatedSumCheckProofVar<Fq254>,
    transcript: &mut RescueTranscriptVar<Fr254>,
) -> Result<ScalarsAndEval, CircuitError> {
    if merged_proof_evals.len() * 2 != old_accs.len() {
        return Err(CircuitError::ParameterError(
            "The merged proof evals should be half the length of the old accumulator evals"
                .to_string(),
        ));
    }

    old_accs.iter().try_for_each(|acc| {
        transcript.append_point_variable(&acc.comm, circuit)?;
        transcript.push_emulated_variable(&acc.value, circuit)
    })?;

    let batch_challenge = transcript.squeeze_scalar_challenge::<SWGrumpkin>(circuit)?;
    let batch_challenge = circuit.to_emulated_variable(batch_challenge)?;
    let zero_var = circuit.emulated_zero();
    let one_var = circuit.emulated_one();
    let mut combiner = one_var.clone();
    let initial_sumcheck_eval = merged_proof_evals
        .iter()
        .chain(old_accs.iter().map(|acc| &acc.value))
        .try_fold(zero_var.clone(), |acc, eval| {
            let fixed_combiner = combiner.clone();
            combiner = circuit.emulated_mul(&combiner, &batch_challenge)?;
            circuit.emulated_mul_add(eval, &fixed_combiner, &acc)
        })?;

    circuit.enforce_emulated_var_equal(&initial_sumcheck_eval, &sumcheck_proof.eval_var)?;

    let accumulated_eval =
        circuit.verify_emulated_proof::<SWGrumpkin>(sumcheck_proof, transcript)?;

    let mut combiner = one_var.clone();

    let accumulation_scalars = proof_eval_points
        .iter()
        .chain(
            &old_accs
                .iter()
                .map(|acc| acc.point.clone())
                .collect::<Vec<Vec<EmulatedVariable<Fq254>>>>(),
        )
        .map(|point| {
            let eq_eval = emulated_eq_x_r_eval_circuit(circuit, point, &sumcheck_proof.point_var)?;
            let fixed_combiner = combiner.clone();
            combiner = circuit.emulated_mul(&combiner, &batch_challenge)?;
            circuit.emulated_mul(&fixed_combiner, &eq_eval)
        })
        .collect::<Result<Vec<EmulatedVariable<Fq254>>, CircuitError>>()?;

    Ok((accumulation_scalars, accumulated_eval))
}

type MLEScalarsAndAccEval = (Vec<EmulatedVariable<Fq254>>, EmulatedVariable<Fq254>);
/// This function takes in two ['RecursiveOutput']s and some pre-calculated challenges and produces the scalars that should be used to calculate their final commitment.
/// It then combines all the scalars in such a way that their hash is equal to the public input hash of the proof from the other curve.
pub fn emulated_combine_mle_proof_scalars(
    outputs: &[GrumpkinOutput],
    acc_info: &SplitAccumulationInfo,
    vk: &MLEVerifyingKey<Zmorph>,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<MLEScalarsAndAccEval, CircuitError> {
    let vk_var = MLEVerifyingKeyVar::<Zmorph>::new(circuit, vk)?;
    let mut scalar_list = Vec::new();
    let mut evals = Vec::new();
    let mut points = Vec::new();
    let mut transcripts = Vec::new();
    let one_var = circuit.emulated_one();
    for output in outputs.iter() {
        let proof_var = SAMLEProofVar::<Zmorph>::from_struct(circuit, &output.proof)?;
        let pi = circuit.create_emulated_variable(output.pi_hash)?;
        let mut transcript_var = RescueTranscriptVar::<Fr254>::new_transcript(circuit);
        let challenges = EmulatedMLEChallenges::<Fq254>::compute_challenges_vars(
            circuit,
            &vk_var,
            &pi,
            &proof_var,
            &mut transcript_var,
        )?;

        let zero_eval =
            proof_var
                .sumcheck_proof
                .point_var
                .iter()
                .try_fold(one_var.clone(), |acc, point| {
                    let tmp1 = circuit.emulated_mul(&acc, point)?;
                    circuit.emulated_sub(&acc, &tmp1)
                })?;

        let pi_eval = circuit.emulated_mul(&pi, &zero_eval)?;

        let (scalars, eval) = verify_mleplonk_emulated_scalar_arithmetic(
            circuit,
            &proof_var,
            &challenges,
            &pi_eval,
            &mut transcript_var,
            &vk.gate_info,
        )?;

        scalar_list.push(scalars);
        evals.push(eval);
        points.push(proof_var.sumcheck_proof.point_var.clone());
        transcripts.push(transcript_var);
    }

    let mut transcript = transcripts[0].clone();
    transcript.merge(&transcripts[1])?;

    let old_acc_vars = acc_info
        .old_accumulators()
        .iter()
        .map(|acc| {
            let comm = circuit.create_point_variable(&Point::<Fr254>::from(acc.comm))?;
            let point = acc
                .point
                .iter()
                .map(|&p| circuit.create_emulated_variable(p))
                .collect::<Result<Vec<EmulatedVariable<Fq254>>, CircuitError>>()?;
            let value = circuit.create_emulated_variable(acc.value)?;
            Ok(EmulatedPCSInstanceVar::new(comm, value, point))
        })
        .collect::<Result<Vec<EmulatedPCSInstanceVar<SWGrumpkin>>, CircuitError>>()?;

    let acc_sumcheck_proof =
        circuit.proof_to_emulated_var::<SWGrumpkin>(acc_info.sumcheck_proof())?;

    let (accumulation_scalars, accumulated_eval) = emulated_verify_split_accumulation(
        circuit,
        &evals,
        &points,
        &old_acc_vars,
        &acc_sumcheck_proof,
        &mut transcript,
    )?;

    let combined_scalars =
        combine_emulated_scalar_lists(&scalar_list, &accumulation_scalars[..2], circuit)?;

    let out_scalars = [combined_scalars.as_slice(), &accumulation_scalars[2..]].concat();

    Ok((out_scalars, accumulated_eval))
}

fn combine_emulated_scalar_lists(
    scalar_list: &[Vec<EmulatedVariable<Fq254>>],
    acc_scalars: &[EmulatedVariable<Fq254>],
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<Vec<EmulatedVariable<Fq254>>, CircuitError> {
    if scalar_list.is_empty() {
        return Err(CircuitError::ParameterError(
            "The scalar list should not be empty".to_string(),
        ));
    }

    if scalar_list.len() != acc_scalars.len() {
        return Err(CircuitError::ParameterError(
            "The number of scalar lists should be equal to the number of accumulator scalars"
                .to_string(),
        ));
    }

    let mut combined_scalars = scalar_list[0]
        .iter()
        .map(|s| circuit.emulated_mul(s, &acc_scalars[0]))
        .collect::<Result<Vec<EmulatedVariable<Fq254>>, CircuitError>>()?;

    for (scalars, acc_scalar) in scalar_list.iter().zip(acc_scalars.iter()).skip(1) {
        let new_scalars = scalars
            .iter()
            .map(|s| circuit.emulated_mul(s, acc_scalar))
            .collect::<Result<Vec<EmulatedVariable<Fq254>>, CircuitError>>()?;

        combined_scalars[6..29]
            .iter_mut()
            .zip(new_scalars[6..29].iter())
            .try_for_each(|(a, b)| {
                *a = circuit.emulated_add(a, b)?;
                Result::<(), CircuitError>::Ok(())
            })?;

        combined_scalars[30..35]
            .iter_mut()
            .zip(new_scalars[30..35].iter())
            .try_for_each(|(a, b)| {
                *a = circuit.emulated_add(a, b)?;
                Result::<(), CircuitError>::Ok(())
            })?;

        let extended_scalars = [&new_scalars[..6], &[new_scalars[29].clone()]].concat();
        combined_scalars.extend_from_slice(&extended_scalars);
    }

    Ok(combined_scalars)
}

#[cfg(test)]
mod tests {
    use ark_ec::{short_weierstrass::Projective, CurveGroup, VariableBaseMSM};

    use ark_poly::{evaluations::multivariate::DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{sync::Arc, vec::Vec, UniformRand};
    use jf_primitives::pcs::PolynomialCommitmentScheme;
    use jf_relation::{Arithmetization, PlonkType};

    use nf_curves::grumpkin::short_weierstrass::SWGrumpkin;

    use super::*;
    use crate::{
        errors::PlonkError,
        nightfall::{
            accumulation::accumulation_structs::PCSWitness,
            mle::{snark::tests::gen_circuit_for_test, MLEPlonk},
        },
        proof_system::{RecursiveOutput, UniversalSNARK},
        transcript::RescueTranscript,
    };
    use ark_ff::PrimeField;
    use jf_relation::Circuit;

    #[test]
    fn test_scalar_combiner() -> Result<(), PlonkError> {
        let rng = &mut jf_utils::test_rng();
        for m in 2..8 {
            let circuit_one = gen_circuit_for_test::<Fq254>(m, 3, PlonkType::UltraPlonk, true)?;
            let circuit_two = gen_circuit_for_test::<Fq254>(m, 4, PlonkType::UltraPlonk, true)?;

            let num_vars = circuit_one.num_gates().ilog2() as usize;
            let srs_size = circuit_one.srs_size()?;
            let srs = MLEPlonk::<Zmorph>::universal_setup_for_testing(srs_size, rng).unwrap();

            let (pk, vk) = MLEPlonk::<Zmorph>::preprocess(&srs, None, &circuit_one)?;

            let circuits = [circuit_one, circuit_two];

            let outputs: [RecursiveOutput<Zmorph, MLEPlonk<Zmorph>, RescueTranscript<Fr254>>; 2] =
                circuits
                    .iter()
                    .map(|circuit| {
                        MLEPlonk::<Zmorph>::recursive_prove::<_, _, RescueTranscript<Fr254>>(
                            rng, circuit, &pk, None,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .try_into()
                    .unwrap();
            for output in outputs.iter() {
                let (opening_proof, _eval) = Zmorph::open(
                    &pk.pcs_prover_params,
                    &output.proof.polynomial,
                    &output.proof.opening_point,
                )
                .unwrap();
                MLEPlonk::<Zmorph>::verify_recursive_proof::<_, _, _, RescueTranscript<Fr254>>(
                    output,
                    &opening_proof,
                    &vk,
                    output.pi_hash,
                    rng,
                )
                .unwrap();
            }
            let mut accs = Vec::new();

            for _ in 0..4 {
                let poly = Arc::new(DenseMultilinearExtension::<Fq254>::rand(num_vars, rng));
                let eval_point = (0..num_vars)
                    .map(|_| Fq254::rand(rng))
                    .collect::<Vec<Fq254>>();

                let comm = Zmorph::commit(&pk.pcs_prover_params, &poly)?;
                let eval = poly.evaluate(&eval_point).unwrap();

                accs.push(PCSWitness::<Zmorph>::new(poly, comm, eval, eval_point));
            }

            let accumulators: [PCSWitness<Zmorph>; 4] = accs.try_into().unwrap();

            let split_acc_info = SplitAccumulationInfo::perform_accumulation(
                &outputs,
                &accumulators,
                &pk.pcs_prover_params,
            )?;

            let mut verifier_circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(8);

            let (combined_scalars, combined_eval) = emulated_combine_mle_proof_scalars(
                &outputs,
                &split_acc_info,
                &vk,
                &mut verifier_circuit,
            )?;

            let m_comm = outputs[0].proof.lookup_proof.as_ref().unwrap().m_poly_comm;

            let mut comms = outputs[0].proof.wire_commitments.clone();
            comms.extend_from_slice(&vk.selector_commitments[..17]);
            comms.extend_from_slice(&vk.permutation_commitments);
            comms.push(m_comm);
            comms.push(vk.lookup_verifying_key.as_ref().unwrap().range_table_comm);
            comms.push(vk.lookup_verifying_key.as_ref().unwrap().key_table_comm);
            comms.push(vk.lookup_verifying_key.as_ref().unwrap().table_dom_sep_comm);
            comms.push(vk.lookup_verifying_key.as_ref().unwrap().q_dom_sep_comm);
            comms.push(vk.lookup_verifying_key.as_ref().unwrap().q_lookup_comm);

            comms.extend_from_slice(&outputs[1].proof.wire_commitments);
            comms.push(outputs[1].proof.lookup_proof.as_ref().unwrap().m_poly_comm);

            let real_scalars = combined_scalars
                .iter()
                .map(|s| verifier_circuit.emulated_witness(s).unwrap())
                .collect::<Vec<Fq254>>();

            let (proof_scalars, acc_scalars) = real_scalars.split_at(combined_scalars.len() - 4);
            let proof_bigints = proof_scalars
                .iter()
                .map(|s| s.into_bigint())
                .collect::<Vec<_>>();

            let proof_comm = Projective::<SWGrumpkin>::msm_bigint(&comms, &proof_bigints);
            let acc_comm = Projective::<SWGrumpkin>::msm_bigint(
                &split_acc_info
                    .old_accumulators()
                    .iter()
                    .map(|acc| acc.comm)
                    .collect::<Vec<_>>(),
                &acc_scalars
                    .iter()
                    .map(|s| s.into_bigint())
                    .collect::<Vec<_>>(),
            );

            let calculated_comm = (proof_comm + acc_comm).into_affine();

            let (opening_proof, eval) = Zmorph::open(
                &pk.pcs_prover_params,
                &split_acc_info.new_accumulator().poly,
                &split_acc_info.new_accumulator().point,
            )
            .unwrap();

            let calc_eval = verifier_circuit.emulated_witness(&combined_eval).unwrap();
            assert_eq!(calc_eval, eval);

            verifier_circuit.check_circuit_satisfiability(&[]).unwrap();

            ark_std::println!("verifier circuit size: {}", verifier_circuit.num_gates());

            assert!(Zmorph::verify(
                &vk.pcs_verifier_params,
                &calculated_comm,
                &split_acc_info.new_accumulator().point,
                &calc_eval,
                &opening_proof
            )
            .unwrap());
        }
        Ok(())
    }
}
