//! This module contains code for performing all the scalar arithmetic involved in verifying a [`MLEPlonk`] proof.
//! We verify the scalar arithmetic as standard and then we check that the scalars used in the previous circuits MSM are correct.
#![allow(dead_code)]
use ark_ff::{batch_inversion, PrimeField};
use ark_std::{string::ToString, vec, vec::Vec, One};

use crate::{
    nightfall::{
        circuit::{
            plonk_partial_verifier::{
                eq_x_r_eval_circuit, MLEChallengesVar, MLELookupEvaluationsNativeVar,
                MLEProofEvalsNativeVar, SAMLEProofNative,
            },
            subroutine_verifiers::{structs::SumCheckProofVar, sumcheck::SumCheckGadget},
        },
        mle::{
            mle_structs::{GateInfo, MLEVerifyingKey},
            MLEPlonk,
        },
    },
    proof_system::RecursiveOutput,
    transcript::RescueTranscript,
};
use ark_bn254::{Fq as Fq254, Fr as Fr254};
use jf_primitives::rescue::RescueParameter;
use jf_relation::{errors::CircuitError, Circuit, PlonkCircuit, Variable};

use super::{
    challenges::{MLEProofChallenges, MLEProofChallengesVar},
    split_acc::SplitAccumulationInfo,
    Zmorph,
};

/// Function for taking the deferred evaluations output by verifying a GKR proof and combines them into the expected initial value of
/// the final SumCheck proof.
pub fn combine_gkr_evals<F: PrimeField>(
    circuit: &mut PlonkCircuit<F>,
    gkr_evals: &[Variable],
    challenges: &MLEChallengesVar,
) -> Result<Variable, CircuitError> {
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

    if gkr_evals.len() == 8 {
        let epsilon_sq = circuit.mul(challenges.epsilon, challenges.epsilon)?;
        let mut eval = circuit.mul_add(
            &[
                circuit.one(),
                gkr_evals[0],
                challenges.epsilon,
                gkr_evals[1],
            ],
            &[F::one(), F::one()],
        )?;

        let second_eval = circuit.mul_add(
            &[challenges.epsilon, gkr_evals[2], epsilon_sq, gkr_evals[3]],
            &[F::one(), F::one()],
        )?;
        eval = circuit.mul_add(
            &[eval, challenges.epsilon, epsilon_sq, second_eval],
            &[F::one(), F::one()],
        )?;
        let beta_inv =
            circuit
                .witness(challenges.beta)?
                .inverse()
                .ok_or(CircuitError::ParameterError(
                    "Could not invert Beta challenge".to_string(),
                ))?;
        let beta_inv_var = circuit.create_variable(beta_inv)?;

        let beta_alpha = circuit.mul(challenges.beta, challenges.alpha)?;
        let coeff = circuit.mul(challenges.epsilon, beta_inv_var)?;
        circuit.mul_gate(beta_inv_var, challenges.beta, circuit.one())?;
        let third_eval = circuit.mul_add(
            &[coeff, beta_alpha, coeff, gkr_evals[6]],
            &[F::one(), -F::one()],
        )?;
        let epsilon_four = circuit.mul(epsilon_sq, epsilon_sq)?;

        eval = circuit.mul_add(
            &[eval, circuit.one(), epsilon_four, third_eval],
            &[F::one(), F::one()],
        )?;

        let fourth_eval = circuit.mul_add(
            &[coeff, beta_alpha, coeff, gkr_evals[7]],
            &[F::one(), -F::one()],
        )?;

        let epsilon_five = circuit.power_5_gen(challenges.epsilon)?;

        eval = circuit.mul_add(
            &[eval, circuit.one(), epsilon_five, fourth_eval],
            &[F::one(), F::one()],
        )?;

        let epsilon_seven = circuit.mul(epsilon_sq, epsilon_five)?;
        let final_coeff = circuit.mul(epsilon_seven, beta_inv_var)?;
        circuit.mul_add(
            &[eval, circuit.one(), final_coeff, gkr_evals[5]],
            &[F::one(), F::one()],
        )
    } else if gkr_evals.len() == 4 {
        let epsilon_sq = circuit.mul(challenges.epsilon, challenges.epsilon)?;
        let eval = circuit.mul_add(
            &[
                circuit.one(),
                gkr_evals[0],
                challenges.epsilon,
                gkr_evals[1],
            ],
            &[F::one(), F::one()],
        )?;

        let second_eval = circuit.mul_add(
            &[challenges.epsilon, gkr_evals[2], epsilon_sq, gkr_evals[3]],
            &[F::one(), F::one()],
        )?;
        circuit.mul_add(
            &[eval, challenges.epsilon, epsilon_sq, second_eval],
            &[F::one(), F::one()],
        )
    } else {
        Err(CircuitError::ParameterError(
            "Invalid number of GKR evaluations".to_string(),
        ))
    }
}

///Circuit corresponding to evaluating the gate equation.
/// Returns the variable corresponding to the evaluation.
pub fn eval_gate_equation<F: PrimeField>(
    circuit: &mut PlonkCircuit<F>,
    gate_info: &GateInfo<F>,
    selector_evals_var: &[Variable],
    wire_evals_var: &[Variable],
    pub_input_poly_eval_var: Variable,
) -> Result<Variable, CircuitError> {
    let evals_var = [
        wire_evals_var,
        selector_evals_var,
        &[pub_input_poly_eval_var],
    ]
    .concat();
    let mut sum_vars = Vec::<Variable>::new();
    for (coeff, prod) in gate_info.products.iter() {
        let mut prod_var = circuit.mul_constant(evals_var[prod[0]], coeff)?;
        for index in prod.iter().skip(1) {
            prod_var = circuit.mul(prod_var, evals_var[*index])?;
        }
        sum_vars.push(prod_var);
    }
    circuit.lin_comb(&vec![F::one(); sum_vars.len()], &F::zero(), &sum_vars)
}

/// Evaluate the permutation equation at a given point. Used by the verifier.
pub(crate) fn eval_permutation_equation<F: PrimeField>(
    perm_evals: &[Variable],
    wire_evals: &[Variable],
    challenges: &MLEChallengesVar,
    circuit: &mut PlonkCircuit<F>,
) -> Result<Variable, CircuitError> {
    let MLEChallengesVar { gamma, epsilon, .. } = *challenges;

    let pairs = perm_evals
        .chunks(3)
        .zip(wire_evals.chunks(3))
        .map(|(perm_chunk, wire_chunk)| {
            let shift_wires = wire_chunk
                .iter()
                .map(|&eval| circuit.sub(gamma, eval))
                .collect::<Result<Vec<Variable>, CircuitError>>()?;
            let mut inv_wires = shift_wires
                .iter()
                .map(|&eval| circuit.witness(eval))
                .collect::<Result<Vec<F>, CircuitError>>()?;
            batch_inversion(&mut inv_wires);
            let denominator = shift_wires
                .iter()
                .try_fold(circuit.one(), |acc, eval| circuit.mul(acc, *eval))?;
            let inv_wire_vars = inv_wires
                .iter()
                .map(|&eval| circuit.create_variable(eval))
                .collect::<Result<Vec<Variable>, CircuitError>>()?;
            let multiplicands = shift_wires
                .iter()
                .zip(inv_wire_vars.iter())
                .map(|(&wire, &inv_wire)| {
                    circuit.mul_gate(wire, inv_wire, circuit.one())?;
                    circuit.mul(inv_wire, denominator)
                })
                .collect::<Result<Vec<Variable>, CircuitError>>()?;

            let numerator = perm_chunk.iter().zip(multiplicands.iter()).try_fold(
                circuit.zero(),
                |acc, (perm, wire)| {
                    circuit.mul_add(&[acc, circuit.one(), *perm, *wire], &[F::one(), F::one()])
                },
            )?;
            Result::<[Variable; 2], CircuitError>::Ok([numerator, denominator])
        })
        .collect::<Result<Vec<[Variable; 2]>, CircuitError>>()?;

    let epsilon_sq = circuit.mul(epsilon, epsilon)?;
    let epsilon_cubed = circuit.mul(epsilon_sq, epsilon)?;
    let epsilon_four = circuit.mul(epsilon_sq, epsilon_sq)?;

    let tmp1 = circuit.mul_add(
        &[pairs[0][0], epsilon, pairs[1][0], epsilon_sq],
        &[F::one(), F::one()],
    )?;
    let tmp2 = circuit.mul_add(
        &[pairs[0][1], epsilon_cubed, pairs[1][1], epsilon_four],
        &[F::one(), F::one()],
    )?;

    circuit.add(tmp1, tmp2)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_lookup_equation<F: PrimeField>(
    wire_evals: &[Variable],
    range_table_eval: Variable,
    key_table_eval: Variable,
    table_dom_sep_eval: Variable,
    q_dom_sep_eval: Variable,
    q_lookup_eval: Variable,
    challenges: &MLEChallengesVar,
    m_poly_eval: Variable,
    circuit: &mut PlonkCircuit<F>,
) -> Result<Variable, CircuitError> {
    let MLEChallengesVar { tau, epsilon, .. } = *challenges;

    let epsilon_five = circuit.power_5_gen(epsilon)?;
    let tmp_one = circuit.mul_add(
        &[wire_evals[1], circuit.one(), wire_evals[2], tau],
        &[F::one(), F::one()],
    )?;
    let tmp_two = circuit.mul_add(
        &[wire_evals[0], circuit.one(), tmp_one, tau],
        &[F::one(), F::one()],
    )?;
    let tmp_three = circuit.mul_add(
        &[q_dom_sep_eval, circuit.one(), tmp_two, tau],
        &[F::one(), F::one()],
    )?;
    let tmp_four = circuit.mul(q_lookup_eval, tmp_three)?;
    let lookup_wire_eval = circuit.mul_add(
        &[wire_evals[5], circuit.one(), tmp_four, tau],
        &[F::one(), F::one()],
    )?;

    let tmp_one = circuit.mul_add(
        &[wire_evals[3], circuit.one(), wire_evals[4], tau],
        &[F::one(), F::one()],
    )?;
    let tmp_two = circuit.mul_add(
        &[key_table_eval, circuit.one(), tmp_one, tau],
        &[F::one(), F::one()],
    )?;
    let tmp_three = circuit.mul_add(
        &[table_dom_sep_eval, circuit.one(), tmp_two, tau],
        &[F::one(), F::one()],
    )?;
    let tmp_four = circuit.mul(q_lookup_eval, tmp_three)?;
    let table_eval = circuit.mul_add(
        &[range_table_eval, circuit.one(), tmp_four, tau],
        &[F::one(), F::one()],
    )?;

    let tmp_one = circuit.mul_add(
        &[table_eval, circuit.one(), epsilon, m_poly_eval],
        &[F::one(), F::one()],
    )?;
    let tmp_two = circuit.mul_add(
        &[lookup_wire_eval, circuit.one(), epsilon, tmp_one],
        &[F::one(), F::one()],
    )?;
    circuit.mul(epsilon_five, tmp_two)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_zerocheck_eval<F: PrimeField>(
    evals: &MLEProofEvalsNativeVar,
    lookup_evals: Option<&MLELookupEvaluationsNativeVar>,
    gate_info: &GateInfo<F>,
    challenges: &MLEChallengesVar,
    pi_eval: Variable,
    eq_eval: Variable,
    circuit: &mut PlonkCircuit<F>,
) -> Result<Variable, CircuitError> {
    let mut sc_eval = eval_gate_equation(
        circuit,
        gate_info,
        &evals.selector_evals[..17],
        &evals.wire_evals[..5],
        pi_eval,
    )?;

    sc_eval = circuit.mul(eq_eval, sc_eval)?;

    let permuatation_eval = eval_permutation_equation(
        &evals.permutation_evals,
        &evals.wire_evals,
        challenges,
        circuit,
    )?;

    sc_eval = circuit.mul_add(
        &[sc_eval, circuit.one(), permuatation_eval, eq_eval],
        &[F::one(), F::one()],
    )?;

    if let Some(lookup_evals) = lookup_evals {
        let lookup_eq_eval = eval_lookup_equation::<F>(
            &evals.wire_evals,
            lookup_evals.range_table_eval,
            lookup_evals.key_table_eval,
            lookup_evals.table_dom_sep_eval,
            lookup_evals.q_dom_sep_eval,
            lookup_evals.q_lookup_eval,
            challenges,
            lookup_evals.m_poly_eval,
            circuit,
        )?;

        sc_eval = circuit.mul_add(
            &[sc_eval, circuit.one(), lookup_eq_eval, eq_eval],
            &[F::one(), F::one()],
        )?;
    }
    Ok(sc_eval)
}

/// This function builds the batched evaluation for a single [`MLEPlonk`] proof.
/// It outputs the combined evaluation, together with the individual scalars.
pub(crate) fn build_mleplonk_eval<F: PrimeField>(
    evals: &MLEProofEvalsNativeVar,
    lookup_evals: Option<&MLELookupEvaluationsNativeVar>,
    delta: Variable,
    circuit: &mut PlonkCircuit<F>,
) -> Result<(Vec<Variable>, Variable), CircuitError> {
    let mut evals_vec = evals.wire_evals.to_vec();
    evals_vec.extend_from_slice(&evals.selector_evals);
    evals_vec.extend_from_slice(&evals.permutation_evals);

    if let Some(lookup_evals) = lookup_evals {
        evals_vec.push(lookup_evals.m_poly_eval);
        evals_vec.push(lookup_evals.range_table_eval);
        evals_vec.push(lookup_evals.key_table_eval);
        evals_vec.push(lookup_evals.table_dom_sep_eval);
        evals_vec.push(lookup_evals.q_dom_sep_eval);
        evals_vec.push(lookup_evals.q_lookup_eval);
    }

    let mut scalars = Vec::<Variable>::with_capacity(evals_vec.len());
    let mut combiner = circuit.one();
    let eval = evals_vec.iter().try_fold(circuit.zero(), |acc, eval| {
        let fixed_combiner = combiner;
        scalars.push(combiner);
        combiner = circuit.mul(combiner, delta)?;
        circuit.mul_add(
            &[acc, circuit.one(), fixed_combiner, *eval],
            &[F::one(), F::one()],
        )
    })?;

    Ok((scalars, eval))
}

pub(crate) type MLEProofMSMScalars = (Vec<Variable>, Variable);

/// This function verifies a single [`MLEPlonk`][crate::nightfall::mle::snark::MLEPlonk] proofs scalar arithmetic given the challenges
/// (that is it doesn't do any hashing). The output of this function if the scalars to be used in a fixed base MSM
/// to check the validity of the PI polynomial.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_mleplonk_scalar_arithmetic<F: PrimeField + RescueParameter>(
    circuit: &mut PlonkCircuit<F>,
    proof: &SAMLEProofNative<F>,
    challenges: &MLEChallengesVar,
    lambdas: &[Variable],
    r_challenges: &[Variable],
    gkr_sumcheck_challenges: &[Vec<Variable>],
    pi_eval: Variable,
    gate_info: &GateInfo<F>,
) -> Result<MLEProofMSMScalars, CircuitError> {
    // First we verify the GKR proof and obtain the evaluations it outputs.
    let gkr_evals = proof.gkr_proof().verify_gkr_proof_with_challenges(
        circuit,
        lambdas,
        r_challenges,
        gkr_sumcheck_challenges,
    )?;

    // Now we reconstruct the initial zerocheck evaluation and check it is correct.
    let initial_zerocheck_eval = combine_gkr_evals(circuit, &gkr_evals, challenges)?;

    circuit.enforce_equal(initial_zerocheck_eval, proof.sumcheck_proof().eval_var)?;

    // Now we verify the arithmetic of the SumCheck proof.
    let zerocheck_eval = circuit.verify_sum_check_with_challenges(proof.sumcheck_proof())?;

    let zc_point = &proof.sumcheck_proof().point_var;

    // Calculate the eq_eval.
    let eq_eval = eq_x_r_eval_circuit(
        circuit,
        zc_point,
        &proof
            .gkr_proof()
            .sumcheck_proofs()
            .last()
            .unwrap()
            .point_var,
    )?;

    // Calculate the same eval based on the supplied polynomial evaluations.
    let calculated_zerocheck_eval = build_zerocheck_eval(
        proof.poly_evals(),
        proof.lookup_proof().as_ref().map(|l| l.poly_evals()),
        gate_info,
        challenges,
        pi_eval,
        eq_eval,
        circuit,
    )?;

    circuit.enforce_equal(zerocheck_eval, calculated_zerocheck_eval)?;

    // Finally calculate the final combined evaluation and return the scalars.
    build_mleplonk_eval(
        proof.poly_evals(),
        proof.lookup_proof().as_ref().map(|l| l.poly_evals()),
        challenges.delta,
        circuit,
    )
}

type ScalarsAndEval = (Vec<Variable>, Variable);

/// We need a function that verifies the accumulation SumCheck for split accumulation
pub fn verify_split_accumulation<F: RescueParameter>(
    circuit: &mut PlonkCircuit<F>,
    merged_proof_evals: &[Variable],
    old_acc_evals: &[Variable],
    sumcheck_proof: &SumCheckProofVar<F>,
    eval_points: &[&[Variable]],
    batching_scalar: Variable,
) -> Result<ScalarsAndEval, CircuitError> {
    if merged_proof_evals.len() * 2 != old_acc_evals.len() {
        return Err(CircuitError::ParameterError(
            "The merged proof evals should be half the length of the old accumulator evals"
                .to_string(),
        ));
    }

    // We should have the same number of eval points as the total polys being accumulated.
    if eval_points.len() != merged_proof_evals.len() + old_acc_evals.len() {
        return Err(CircuitError::ParameterError(
            "The number of eval points should be equal to the number of polys being accumulated"
                .to_string(),
        ));
    }
    let mut combiner = circuit.one();
    let initial_sumcheck_eval = merged_proof_evals
        .iter()
        .chain(old_acc_evals.iter())
        .try_fold(circuit.zero(), |acc, eval| {
            let fixed_combiner = combiner;
            combiner = circuit.mul(combiner, batching_scalar)?;
            circuit.mul_add(
                &[acc, circuit.one(), *eval, fixed_combiner],
                &[F::one(), F::one()],
            )
        })?;

    circuit.enforce_equal(initial_sumcheck_eval, sumcheck_proof.eval_var)?;

    let accumulated_eval = circuit.verify_sum_check_with_challenges(sumcheck_proof)?;

    let mut combiner = circuit.one();

    let accumulation_scalars = eval_points
        .iter()
        .map(|point| {
            let eq_eval = eq_x_r_eval_circuit(circuit, point, &sumcheck_proof.point_var)?;
            let fixed_combiner = combiner;
            combiner = circuit.mul(combiner, batching_scalar)?;
            circuit.mul(fixed_combiner, eq_eval)
        })
        .collect::<Result<Vec<Variable>, CircuitError>>()?;

    Ok((accumulation_scalars, accumulated_eval))
}

type MLEScalarsAndAccEval = (Vec<Variable>, Variable);
/// This function takes in two ['RecursiveOutput']s and some pre-calculated challenges and produces the scalars that should be used to calculate their final commitment.
/// It then combines all the scalars in such a way that their hash is equal to the public input hash of the proof from the other curve.
pub fn combine_mle_proof_scalars(
    outputs: &[RecursiveOutput<Zmorph, MLEPlonk<Zmorph>, RescueTranscript<Fr254>>],
    challenges: &[MLEProofChallenges<Fq254>],
    pi_hashes: &[Variable],
    acc_info: &SplitAccumulationInfo,
    vk: &MLEVerifyingKey<Zmorph>,
    circuit: &mut PlonkCircuit<Fq254>,
) -> Result<MLEScalarsAndAccEval, CircuitError> {
    if outputs.len() != challenges.len() {
        return Err(CircuitError::ParameterError(
            "The number of outputs should be equal to the number of challenges".to_string(),
        ));
    }
    if outputs.len() != pi_hashes.len() {
        return Err(CircuitError::ParameterError(
            "The number of outputs should be equal to the number of public input hashes"
                .to_string(),
        ));
    }

    let mut scalar_list = Vec::new();
    let mut evals = Vec::new();
    let mut points = Vec::new();
    for ((output, pi_hash), proof_challenges) in
        outputs.iter().zip(pi_hashes).zip(challenges.iter())
    {
        let proof_var = SAMLEProofNative::from_struct(circuit, &output.proof)?;
        let proof_challenges_var = MLEProofChallengesVar::from_struct(circuit, proof_challenges)?;

        let zero_eval =
            proof_var
                .sumcheck_proof()
                .point_var
                .iter()
                .try_fold(circuit.one(), |acc, point| {
                    circuit.mul_add(
                        &[acc, circuit.one(), acc, *point],
                        &[Fq254::one(), -Fq254::one()],
                    )
                })?;

        let pi_eval = circuit.mul(*pi_hash, zero_eval)?;

        let (scalars, eval) = verify_mleplonk_scalar_arithmetic(
            circuit,
            &proof_var,
            proof_challenges_var.challenges(),
            proof_challenges_var.gkr_lambda_challenges(),
            proof_challenges_var.gkr_r_challenges(),
            proof_challenges_var.gkr_sumcheck_challenges(),
            pi_eval,
            &vk.gate_info,
        )?;

        scalar_list.push(scalars);
        evals.push(eval);
        points.push(proof_var.sumcheck_proof().point_var.clone());
    }

    let old_acc_evals = acc_info
        .old_accumulators()
        .iter()
        .map(|acc| circuit.create_variable(acc.value))
        .collect::<Result<Vec<Variable>, CircuitError>>()?;

    let acc_sumcheck_proof = circuit.sum_check_proof_to_var(acc_info.sumcheck_proof())?;

    let acc_eval_points = acc_info
        .old_accumulators()
        .iter()
        .map(|acc| {
            acc.point
                .iter()
                .map(|p| circuit.create_variable(*p))
                .collect::<Result<Vec<Variable>, CircuitError>>()
        })
        .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?;

    let eval_points = points
        .iter()
        .chain(acc_eval_points.iter())
        .map(|p| p.as_slice())
        .collect::<Vec<&[Variable]>>();

    let batching_scalar = circuit.create_variable(acc_info.batch_challenge)?;
    let (accumulation_scalars, accumulated_eval) = verify_split_accumulation(
        circuit,
        &evals,
        &old_acc_evals,
        &acc_sumcheck_proof,
        &eval_points,
        batching_scalar,
    )?;

    let combined_scalars = combine_scalar_lists(&scalar_list, &accumulation_scalars[..2], circuit)?;

    let out_scalars = [combined_scalars.as_slice(), &accumulation_scalars[2..]].concat();

    Ok((out_scalars, accumulated_eval))
}

fn combine_scalar_lists(
    scalar_list: &[Vec<Variable>],
    acc_scalars: &[Variable],
    circuit: &mut PlonkCircuit<Fq254>,
) -> Result<Vec<Variable>, CircuitError> {
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
        .map(|s| circuit.mul(*s, acc_scalars[0]))
        .collect::<Result<Vec<Variable>, CircuitError>>()?;

    for (scalars, acc_scalar) in scalar_list.iter().zip(acc_scalars.iter()).skip(1) {
        let new_scalars = scalars
            .iter()
            .map(|s| circuit.mul(*s, *acc_scalar))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;

        combined_scalars[6..29]
            .iter_mut()
            .zip(new_scalars[6..29].iter())
            .try_for_each(|(a, b)| {
                *a = circuit.add(*a, *b)?;
                Result::<(), CircuitError>::Ok(())
            })?;

        combined_scalars[30..35]
            .iter_mut()
            .zip(new_scalars[30..35].iter())
            .try_for_each(|(a, b)| {
                *a = circuit.add(*a, *b)?;
                Result::<(), CircuitError>::Ok(())
            })?;

        let extended_scalars = [&new_scalars[..6], &[new_scalars[29]]].concat();
        combined_scalars.extend_from_slice(&extended_scalars);
    }

    Ok(combined_scalars)
}

#[cfg(test)]
mod tests {
    use ark_ec::{
        short_weierstrass::{Affine, Projective},
        CurveGroup, VariableBaseMSM,
    };

    use ark_poly::{evaluations::multivariate::DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{sync::Arc, vec, vec::Vec, UniformRand};
    use jf_primitives::pcs::{Accumulation, PolynomialCommitmentScheme};
    use jf_relation::{
        gadgets::{ecc::HasTEForm, EmulationConfig},
        Arithmetization, PlonkType,
    };
    use jf_utils::test_rng;
    use nf_curves::grumpkin::{short_weierstrass::SWGrumpkin, Grumpkin};

    use crate::{
        errors::PlonkError,
        nightfall::{
            accumulation::accumulation_structs::PCSWitness,
            circuit::{
                plonk_partial_verifier::SAMLEProofVar,
                subroutine_verifiers::gkr::tests::extract_gkr_challenges,
            },
            mle::{
                mle_structs::MLEChallenges, snark::tests::gen_circuit_for_test,
                zeromorph::Zeromorph, MLEPlonk,
            },
            UnivariateIpaPCS,
        },
        proof_system::{UniversalRecursiveSNARK, UniversalSNARK},
        recursion::circuits::challenges::reconstruct_mle_challenges,
        transcript::{rescue::RescueTranscriptVar, RescueTranscript, Transcript},
    };

    use super::*;

    #[test]
    fn test_mle_scalar_arithmetic() -> Result<(), PlonkError> {
        test_mle_scalar_arithmetic_helper::<Zeromorph<UnivariateIpaPCS<Grumpkin>>, _, _>(
            PlonkType::UltraPlonk,
        )?;
        Ok(())
    }
    fn test_mle_scalar_arithmetic_helper<PCS, P, F>(plonk_type: PlonkType) -> Result<(), PlonkError>
    where
        PCS: Accumulation<
            Commitment = Affine<P>,
            Point = Vec<P::ScalarField>,
            Evaluation = P::ScalarField,
            Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
        >,
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField + RescueParameter,
        F: PrimeField + RescueParameter + EmulationConfig<P::BaseField>,
    {
        // 1. Simulate universal setup
        let rng = &mut test_rng();

        // 2. Create circuits
        let circuits = (0..6)
            .map(|i| {
                let m = 5;
                let a0 = 6 + i;
                gen_circuit_for_test(m, a0, plonk_type, true)
            })
            .collect::<Result<Vec<_>, PlonkError>>()?;
        // 3. Preprocessing
        let n = circuits[3].srs_size(false)?;
        let max_degree = n + 2;
        let srs = MLEPlonk::<PCS>::universal_setup_for_testing(max_degree, rng)?;

        let (pk1, _vk1) =
            <MLEPlonk<PCS> as UniversalSNARK<PCS>>::preprocess(&srs, None, &circuits[3], false)
                .unwrap();

        // 4. Proving
        let mut proofs = vec![];
        let mut public_inputs = vec![];
        for cs in circuits.iter() {
            let pk_ref = &pk1;

            proofs.push(
                MLEPlonk::<PCS>::recursive_prove::<_, _, RescueTranscript<P::BaseField>>(
                    rng, cs, pk_ref, None, false,
                )
                .unwrap(),
            );
            public_inputs.push(cs.public_input().unwrap());
        }

        for (proof, public_input) in proofs.iter().zip(public_inputs.iter()) {
            let mut transcript = RescueTranscript::<P::BaseField>::new_transcript(b"mle_plonk");
            let mle_challenges = MLEChallenges::<P::ScalarField>::new_recursion(
                &proof.proof,
                &[public_input[0]],
                &mut transcript,
            )
            .unwrap();
            let inner_proof = &proof.proof;
            let (lambdas, r_challenges) =
                extract_gkr_challenges::<P, RescueTranscript<P::BaseField>>(
                    &inner_proof.gkr_proof,
                    &mut transcript,
                )
                .unwrap();
            let gkr_sumcheck_challenges = inner_proof
                .gkr_proof
                .sumcheck_proofs()
                .iter()
                .map(|p| p.point.clone())
                .collect::<Vec<Vec<P::ScalarField>>>();

            let (opening_proof, _eval) = PCS::open(
                &pk1.pcs_prover_params,
                &inner_proof.polynomial,
                &inner_proof.opening_point,
            )
            .unwrap();
            MLEPlonk::<PCS>::verify_recursive_proof::<_, _, _, RescueTranscript<P::BaseField>>(
                proof,
                &opening_proof,
                &_vk1,
                public_input[0],
                rng,
                None,
            )
            .unwrap();

            let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(8);

            let challenge_vars = MLEChallengesVar::from_struct(&mut circuit, &mle_challenges)?;

            let lambda_vars = lambdas
                .iter()
                .map(|l| circuit.create_variable(*l))
                .collect::<Result<Vec<Variable>, CircuitError>>()?;
            let r_challenges = r_challenges
                .iter()
                .map(|r| circuit.create_variable(*r))
                .collect::<Result<Vec<Variable>, CircuitError>>()?;
            let gkr_sumcheck_challenges = gkr_sumcheck_challenges
                .iter()
                .map(|p| {
                    p.iter()
                        .map(|j| circuit.create_variable(*j))
                        .collect::<Result<Vec<Variable>, CircuitError>>()
                })
                .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?;

            let proof_native = SAMLEProofNative::from_struct(&mut circuit, &proof.proof)?;

            let gate_info = &pk1.verifying_key.gate_info;
            let mut pi_poly = vec![public_input[0]];
            let num_vars = gkr_sumcheck_challenges.last().as_ref().unwrap().len();
            pi_poly.resize(1 << num_vars, F::zero());

            let pi_pol = DenseMultilinearExtension::<F>::from_evaluations_vec(num_vars, pi_poly);

            let pi_eval = pi_pol.evaluate(&proof.proof.sumcheck_proof.point).unwrap();
            let pi_eval = circuit.create_variable(pi_eval)?;

            let (scalars, combined_eval) = verify_mleplonk_scalar_arithmetic(
                &mut circuit,
                &proof_native,
                &challenge_vars,
                &lambda_vars,
                &r_challenges,
                &gkr_sumcheck_challenges,
                pi_eval,
                gate_info,
            )?;

            let scalars = scalars
                .iter()
                .map(|s| circuit.witness(*s))
                .collect::<Result<Vec<F>, CircuitError>>()?;
            let bigints = scalars.iter().map(|s| s.into_bigint()).collect::<Vec<_>>();
            let combined_eval = circuit.witness(combined_eval)?;
            circuit.check_circuit_satisfiability(&[]).unwrap();

            let lookup_commitments = if let Some(lookup_proof) = inner_proof.lookup_proof.as_ref() {
                let lookup_vk = _vk1.lookup_verifying_key.as_ref().unwrap();
                vec![
                    lookup_proof.m_poly_comm,
                    lookup_vk.range_table_comm,
                    lookup_vk.key_table_comm,
                    lookup_vk.table_dom_sep_comm,
                    lookup_vk.q_dom_sep_comm,
                    lookup_vk.q_lookup_comm,
                ]
            } else {
                vec![]
            };

            let proof_bases = [
                inner_proof.wire_commitments.as_slice(),
                &_vk1.selector_commitments[..17],
                _vk1.permutation_commitments.as_slice(),
                lookup_commitments.as_slice(),
            ]
            .concat();

            let combined_comm = Projective::<P>::msm_bigint(&proof_bases, &bigints).into_affine();

            assert!(PCS::verify(
                &_vk1.pcs_verifier_params,
                &combined_comm,
                &inner_proof.opening_point,
                &combined_eval,
                &opening_proof
            )
            .unwrap());
        }

        Ok(())
    }

    #[test]
    fn test_scalar_combiner() -> Result<(), PlonkError> {
        let rng = &mut jf_utils::test_rng();
        for m in 2..8 {
            let circuit_one = gen_circuit_for_test::<Fq254>(m, 3, PlonkType::UltraPlonk, true)?;
            let circuit_two = gen_circuit_for_test::<Fq254>(m, 4, PlonkType::UltraPlonk, true)?;

            let num_vars = circuit_one.num_gates().ilog2() as usize;
            let srs_size = circuit_one.srs_size(false)?;
            let srs = MLEPlonk::<Zmorph>::universal_setup_for_testing(srs_size, rng).unwrap();

            let (pk, vk) = MLEPlonk::<Zmorph>::preprocess(&srs, None, &circuit_one, false)?;

            let circuits = [circuit_one, circuit_two];

            let outputs: [RecursiveOutput<Zmorph, MLEPlonk<Zmorph>, RescueTranscript<Fr254>>; 2] =
                circuits
                    .iter()
                    .map(|circuit| {
                        MLEPlonk::<Zmorph>::recursive_prove::<_, _, RescueTranscript<Fr254>>(
                            rng, circuit, &pk, None, false,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .try_into()
                    .unwrap();
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

            let mut challenges_circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(8);
            let mle_proof_challenges = outputs
                .iter()
                .map(|output| {
                    let pi_hash = challenges_circuit
                        .create_emulated_variable(output.pi_hash)
                        .unwrap();
                    let proof_var =
                        SAMLEProofVar::from_struct(&mut challenges_circuit, &output.proof).unwrap();
                    let (stuff, _) = reconstruct_mle_challenges::<
                        _,
                        _,
                        Zmorph,
                        MLEPlonk<Zmorph>,
                        RescueTranscript<Fr254>,
                        RescueTranscriptVar<Fr254>,
                    >(
                        &proof_var, &mut challenges_circuit, &pi_hash, &None
                    )
                    .unwrap();
                    stuff
                })
                .collect::<Vec<MLEProofChallenges<Fq254>>>();

            let mut verifier_circuit = PlonkCircuit::<Fq254>::new_ultra_plonk(8);

            let pi_hashes: Vec<Variable> = outputs
                .iter()
                .map(|o| verifier_circuit.create_variable(o.pi_hash))
                .collect::<Result<Vec<Variable>, _>>()
                .unwrap();
            let (combined_scalars, combined_eval) = combine_mle_proof_scalars(
                &outputs,
                &mle_proof_challenges,
                &pi_hashes,
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
                .map(|s| verifier_circuit.witness(*s).unwrap())
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

            let calc_eval = verifier_circuit.witness(combined_eval).unwrap();
            assert_eq!(calc_eval, eval);

            verifier_circuit.check_circuit_satisfiability(&[]).unwrap();

            ark_std::println!("verifier circuit size: {}", verifier_circuit.num_gates());
            challenges_circuit
                .check_circuit_satisfiability(&[])
                .unwrap();
            ark_std::println!("challenge circuit size: {}", challenges_circuit.num_gates());
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
