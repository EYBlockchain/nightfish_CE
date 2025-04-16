// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Circuits for the building blocks in Plonk verifiers.

use crate::nightfall::mle::mle_structs::GateInfo;

use ark_ff::PrimeField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::{string::ToString, vec, vec::Vec};
use jf_primitives::rescue::RescueParameter;
use jf_relation::{
    errors::CircuitError,
    gadgets::{EmulatedVariable, EmulationConfig},
    Circuit, PlonkCircuit, Variable,
};

use super::{
    poly::{
        self, compute_lin_poly_constant_term_circuit_native, evaluate_lagrange_poly_helper_native,
        linearization_scalars_circuit_native,
    },
    ChallengesVar, PlookupEvalsVarNative, ProofEvalsVarNative,
    {DEPOSIT_DOMAIN_SIZE, TRANSFER_DOMAIN_SIZE},
};

/// Function to compute the scalars used in partial verification over the native field
pub fn compute_scalars_for_native_field<F: PrimeField + RescueParameter, const IS_BASE: bool>(
    circuit: &mut PlonkCircuit<F>,
    pi: Variable,
    challenges: &ChallengesVar,
    proof_evals: &ProofEvalsVarNative,
    lookup_evals: Option<PlookupEvalsVarNative>,
    vk_k: &[Variable],
    domain_size: usize,
) -> Result<Vec<Variable>, CircuitError> {
    // In lookup scalars are combined in the order
    // zeta: w[0], w[1], w[2], w[5], sigma[0], sigma[1], sigma[2], sigma[3], sigma[4], q_dom_sep, pi_eval
    // zeta_omega: prod_perm_poly_next_eval, h_2_next_eval, lookup_prod_next_eval
    // zeta and zeta_omega: range_table, key_table, h_1, q_lookup, w[3], w[4], table_dom_sep
    // If normal turboplonk then the order is
    // zeta: w[0], w[1], w[2], w[3], w[4], sigma[0], sigma[1], sigma[2], sigma[3],
    // zeta_omega: prod_perm_poly_next_eval,

    // If we are in the non-base case, `domain_size` depends only on the layer of the recursion.
    // We can, therefore, treat `domain_size` and `domain.group_gen` as scalars.
    // If we are in the base case, `domain_size` can be either 2^15 or 2^18,
    // depending on whether the corresponding client circuit is a deposit or not.

    let (domain_size_var, gen_var, gen_inv_var) = if IS_BASE {
        // In the base case, `domain_size` must be either 2^15 or 2^18.
        if domain_size != TRANSFER_DOMAIN_SIZE && domain_size != DEPOSIT_DOMAIN_SIZE {
            return Err(CircuitError::ParameterError(
                "Invalid domain size for base case".to_string(),
            ));
        }
        let is_transfer_var =
            circuit.create_boolean_variable(domain_size == TRANSFER_DOMAIN_SIZE)?;
        let transfer_domain = Radix2EvaluationDomain::<F>::new(TRANSFER_DOMAIN_SIZE).unwrap();
        let deposit_domain = Radix2EvaluationDomain::<F>::new(DEPOSIT_DOMAIN_SIZE).unwrap();
        let transfer_domain_size_var =
            circuit.create_constant_variable(F::from(transfer_domain.size))?;
        let deposit_domain_size_var =
            circuit.create_constant_variable(F::from(deposit_domain.size))?;
        let transfer_gen_var = circuit.create_constant_variable(transfer_domain.group_gen)?;
        let deposit_gen_var = circuit.create_constant_variable(deposit_domain.group_gen)?;
        let domain_size_var = circuit.conditional_select(
            is_transfer_var,
            deposit_domain_size_var,
            transfer_domain_size_var,
        )?;
        let gen_var =
            circuit.conditional_select(is_transfer_var, deposit_gen_var, transfer_gen_var)?;
        let gen_inv = circuit.witness(gen_var)?.inverse().unwrap_or(F::zero());
        let gen_inv_var = circuit.create_variable(gen_inv)?;
        circuit.mul_gate(gen_var, gen_inv_var, circuit.one())?;
        (domain_size_var, gen_var, gen_inv_var)
    } else {
        // In the non-base case, `domain_size` cannot be either TRANSFER_DOMAIN_SIZE or DEPOSIT_DOMAIN_SIZE.
        if domain_size == TRANSFER_DOMAIN_SIZE || domain_size == DEPOSIT_DOMAIN_SIZE {
            return Err(CircuitError::ParameterError(
                "Invalid domain size for non-base case".to_string(),
            ));
        }
        // In the non-base case, we treat `domain_size` as fixed.
        let domain = Radix2EvaluationDomain::<F>::new(domain_size).unwrap();
        let domain_size_var = circuit.create_constant_variable(F::from(domain.size))?;
        let gen_var = circuit.create_constant_variable(domain.group_gen)?;
        let gen_inv_var = circuit.create_constant_variable(domain.group_gen_inv)?;
        (domain_size_var, gen_var, gen_inv_var)
    };

    let evals = poly::evaluate_poly_helper_native::<F, IS_BASE>(
        circuit,
        challenges.zeta,
        gen_inv_var,
        domain_size_var,
    )?;

    let lin_poly_const = compute_lin_poly_constant_term_circuit_native(
        circuit,
        gen_inv_var,
        challenges,
        proof_evals,
        pi,
        &evals,
        &lookup_evals,
    )?;

    let mut d_1_coeffs = linearization_scalars_circuit_native(
        circuit,
        vk_k,
        challenges,
        challenges.zeta,
        &evals,
        proof_evals,
        &lookup_evals,
        gen_inv_var,
    )?;

    if lookup_evals.is_none() {
        d_1_coeffs = d_1_coeffs[..24].to_vec();
    }
    let zeta_omega_var = circuit.mul(challenges.zeta, gen_var)?;
    let denom = circuit.sub(challenges.zeta, zeta_omega_var)?;
    let denom_val = circuit.witness(denom)?;
    let inverse = denom_val.inverse().unwrap_or(F::one());
    let inverse_var = circuit.create_variable(inverse)?;
    let c = if denom_val == F::zero() {
        circuit.zero()
    } else {
        circuit.one()
    };

    circuit.mul_gate(inverse_var, denom, c)?;
    let u_minus_zeta = circuit.sub(challenges.u, challenges.zeta)?;
    let u_minus_zeta_omega = circuit.sub(challenges.u, zeta_omega_var)?;
    let q_commitment_scalar = circuit.mul_add(
        &[
            u_minus_zeta_omega,
            u_minus_zeta,
            circuit.zero(),
            circuit.zero(),
        ],
        &[-F::one(), F::zero()],
    )?;
    let mut evals_list = vec![];
    let mut coeffs_list = vec![];
    let num_wire_types = proof_evals.wires_evals.len();
    if let Some(lookup_evals) = lookup_evals.as_ref() {
        evals_list.push(circuit.mul(u_minus_zeta_omega, proof_evals.wires_evals[0])?);
        evals_list.push(circuit.mul(u_minus_zeta_omega, proof_evals.wires_evals[1])?);
        evals_list.push(circuit.mul(u_minus_zeta_omega, proof_evals.wires_evals[2])?);
        evals_list.push(circuit.mul(u_minus_zeta_omega, proof_evals.wires_evals[5])?);
        for sigma_eval in proof_evals.wire_sigma_evals.iter().take(num_wire_types - 1) {
            evals_list.push(circuit.mul(u_minus_zeta_omega, *sigma_eval)?);
        }
        evals_list.push(circuit.mul(u_minus_zeta_omega, lookup_evals.q_dom_sep_eval)?);

        evals_list.push(circuit.mul(u_minus_zeta, proof_evals.perm_next_eval)?);
        evals_list.push(circuit.mul(u_minus_zeta, lookup_evals.h_2_next_eval)?);
        evals_list.push(circuit.mul(u_minus_zeta, lookup_evals.prod_next_eval)?);

        let range_eval = evaluate_lagrange_poly_helper_native(
            circuit,
            challenges.zeta,
            zeta_omega_var,
            inverse_var,
            &[
                lookup_evals.range_table_eval,
                lookup_evals.range_table_next_eval,
            ],
            challenges.u,
        )?;

        let key_eval = evaluate_lagrange_poly_helper_native(
            circuit,
            challenges.zeta,
            zeta_omega_var,
            inverse_var,
            &[
                lookup_evals.key_table_eval,
                lookup_evals.key_table_next_eval,
            ],
            challenges.u,
        )?;

        let h_1_eval = evaluate_lagrange_poly_helper_native(
            circuit,
            challenges.zeta,
            zeta_omega_var,
            inverse_var,
            &[lookup_evals.h_1_eval, lookup_evals.h_1_next_eval],
            challenges.u,
        )?;

        let q_lookup_eval = evaluate_lagrange_poly_helper_native(
            circuit,
            challenges.zeta,
            zeta_omega_var,
            inverse_var,
            &[lookup_evals.q_lookup_eval, lookup_evals.q_lookup_next_eval],
            challenges.u,
        )?;

        let w_3_eval = evaluate_lagrange_poly_helper_native(
            circuit,
            challenges.zeta,
            zeta_omega_var,
            inverse_var,
            &[proof_evals.wires_evals[3], lookup_evals.w_3_next_eval],
            challenges.u,
        )?;

        let w_4_eval = evaluate_lagrange_poly_helper_native(
            circuit,
            challenges.zeta,
            zeta_omega_var,
            inverse_var,
            &[proof_evals.wires_evals[4], lookup_evals.w_4_next_eval],
            challenges.u,
        )?;

        let table_dom_sep_eval = evaluate_lagrange_poly_helper_native(
            circuit,
            challenges.zeta,
            zeta_omega_var,
            inverse_var,
            &[
                lookup_evals.table_dom_sep_eval,
                lookup_evals.table_dom_sep_next_eval,
            ],
            challenges.u,
        )?;

        evals_list.push(range_eval);
        evals_list.push(key_eval);
        evals_list.push(h_1_eval);
        evals_list.push(q_lookup_eval);
        evals_list.push(w_3_eval);
        evals_list.push(w_4_eval);
        evals_list.push(table_dom_sep_eval);

        for _ in 0..10 {
            coeffs_list.push(u_minus_zeta_omega);
        }

        for _ in 0..3 {
            coeffs_list.push(u_minus_zeta);
        }

        for _ in 0..7 {
            coeffs_list.push(circuit.one());
        }

        for eval in evals_list.iter_mut() {
            *eval = circuit.sub(circuit.zero(), *eval)?;
        }
    } else {
        for wire_eval in proof_evals.wires_evals.iter() {
            evals_list.push(circuit.mul(u_minus_zeta_omega, *wire_eval)?);
        }
        for sigma_eval in proof_evals.wire_sigma_evals.iter().take(4) {
            evals_list.push(circuit.mul(u_minus_zeta_omega, *sigma_eval)?);
        }

        evals_list.push(circuit.mul(u_minus_zeta, proof_evals.perm_next_eval)?);
        for eval in evals_list.iter_mut() {
            *eval = circuit.sub(circuit.zero(), *eval)?;
        }
        for _ in 0..9 {
            coeffs_list.push(u_minus_zeta_omega);
        }
        coeffs_list.push(u_minus_zeta);
    }
    let mut combiner = challenges.v;

    for (eval, coeff) in evals_list.iter_mut().zip(coeffs_list.iter_mut()) {
        *eval = circuit.mul(*eval, combiner)?;
        *coeff = circuit.mul(*coeff, combiner)?;
        combiner = circuit.mul(combiner, challenges.v)?;
    }

    for d_1_coeff in d_1_coeffs.iter_mut() {
        *d_1_coeff = circuit.mul(*d_1_coeff, u_minus_zeta_omega)?;
    }

    evals_list.push(circuit.mul(u_minus_zeta_omega, lin_poly_const)?);

    let g_scalar = circuit.sum(&evals_list)?;

    let mut result = [
        coeffs_list.as_slice(),
        &[q_commitment_scalar, g_scalar],
        d_1_coeffs.as_slice(),
    ]
    .concat();

    if lookup_evals.is_some() {
        result[10] = circuit.add(result[10], result[22])?;

        result[11] = circuit.add(result[11], result[48])?;
        result[12] = circuit.add(result[12], result[47])?;

        result.remove(22);
    }

    Ok(result[..46].to_vec())
}

///Circuit corresponding to evaluating the gate equation.
/// Returns the variable corresponding to the evaluation.
pub fn emulated_eval_gate_equation_circuit<F, E>(
    circuit: &mut PlonkCircuit<F>,
    gate_info: &GateInfo<E>,
    selector_evals_var: &[EmulatedVariable<E>],
    wire_evals_var: &[EmulatedVariable<E>],
    pub_input_poly_eval_var: &EmulatedVariable<E>,
) -> Result<EmulatedVariable<E>, CircuitError>
where
    F: PrimeField,
    E: PrimeField + EmulationConfig<F>,
{
    let zero_var = circuit.emulated_zero();

    let neg_pub_input_poly_eval_var = circuit.emulated_sub(&zero_var, pub_input_poly_eval_var)?;
    let evals_var = [
        wire_evals_var,
        selector_evals_var,
        &[neg_pub_input_poly_eval_var],
    ]
    .concat();
    let mut sum_var = zero_var.clone();
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

/// Circuit corresponding to evaluating the permutation gate equation.
/// Returns the variable corresponding to the evaluation.
#[allow(clippy::too_many_arguments)]
pub fn emulated_eval_perm_gate_circuit<F, E>(
    circuit: &mut PlonkCircuit<F>,
    prod_evals_var: &[EmulatedVariable<E>],
    frac_evals_var: &[EmulatedVariable<E>],
    wire_evals_var: &[EmulatedVariable<E>],
    id_evals_var: &[EmulatedVariable<E>],
    perm_evals_var: &[EmulatedVariable<E>],
    alpha_var: &EmulatedVariable<E>,
    beta_var: &EmulatedVariable<E>,
    gamma_var: &EmulatedVariable<E>,
    x1_var: &EmulatedVariable<E>,
) -> Result<EmulatedVariable<E>, CircuitError>
where
    F: PrimeField,
    E: PrimeField + EmulationConfig<F>,
{
    let one_var = circuit.emulated_one();

    let first_sub_var = circuit.emulated_sub(&prod_evals_var[1], &frac_evals_var[1])?;
    let second_sub_var = circuit.emulated_sub(&prod_evals_var[2], &frac_evals_var[2])?;

    let p1_tmp2 = circuit.emulated_mul(x1_var, &first_sub_var)?;
    let p1_eval_var = circuit.emulated_add(&frac_evals_var[1], &p1_tmp2)?;

    let p2_tmp1 = circuit.emulated_mul(x1_var, &second_sub_var)?;
    let p2_eval_var = circuit.emulated_add(&frac_evals_var[2], &p2_tmp1)?;

    let mut f_prod_eval_var = one_var.clone();
    for (w_eval_var, id_eval_var) in wire_evals_var.iter().zip(id_evals_var.iter()) {
        let tmp1 = circuit.emulated_mul(beta_var, id_eval_var)?;
        let tmp2 = circuit.emulated_add(&tmp1, gamma_var)?;
        let tmp3 = circuit.emulated_add(w_eval_var, &tmp2)?;
        f_prod_eval_var = circuit.emulated_mul(&f_prod_eval_var, &tmp3)?;
    }
    let mut g_prod_eval_var = one_var.clone();
    for (w_eval_var, p_eval_var) in wire_evals_var.iter().zip(perm_evals_var.iter()) {
        let tmp1 = circuit.emulated_mul(beta_var, p_eval_var)?;
        let tmp2 = circuit.emulated_add(&tmp1, gamma_var)?;
        let tmp3 = circuit.emulated_add(w_eval_var, &tmp2)?;
        g_prod_eval_var = circuit.emulated_mul(&g_prod_eval_var, &tmp3)?;
    }
    let first_mul_add_tmp = circuit.emulated_mul(&p1_eval_var, &p2_eval_var)?;
    let first_mul_add_var = circuit.emulated_sub(&prod_evals_var[0], &first_mul_add_tmp)?;

    let second_mul_add_tmp = circuit.emulated_mul(&frac_evals_var[0], &g_prod_eval_var)?;
    let second_mul_add_var = circuit.emulated_sub(&second_mul_add_tmp, &f_prod_eval_var)?;

    let res_tmp = circuit.emulated_mul(alpha_var, &second_mul_add_var)?;
    let res_var = circuit.emulated_add(&first_mul_add_var, &res_tmp)?;

    Ok(res_var)
}

/// Circuit corresponding to evaluating the permutation gate equation.
/// Returns the variable corresponding to the evaluation.
#[allow(clippy::too_many_arguments)]
pub fn emulated_eval_lookup_equation_circuit<F, E>(
    circuit: &mut PlonkCircuit<F>,
    taus_var: &[EmulatedVariable<E>; 3],
    wire_evals_var: &[EmulatedVariable<E>],
    range_table_eval_var: &EmulatedVariable<E>,
    key_table_eval_var: &EmulatedVariable<E>,
    table_dom_sep_eval_var: &EmulatedVariable<E>,
    q_dom_sep_eval_var: &EmulatedVariable<E>,
    m_poly_eval_var: &EmulatedVariable<E>,
    lk_alpha_var: &EmulatedVariable<E>,
    lambda_var: &EmulatedVariable<E>,
) -> Result<EmulatedVariable<E>, CircuitError>
where
    F: PrimeField,
    E: PrimeField + EmulationConfig<F>,
{
    let first_mul_add_tmp1 = circuit.emulated_mul(q_dom_sep_eval_var, &taus_var[0])?;
    let first_mul_add_tmp2 = circuit.emulated_mul(&wire_evals_var[0], &taus_var[1])?;
    let first_mul_add_var = circuit.emulated_add(&first_mul_add_tmp1, &first_mul_add_tmp2)?;

    let taus_mul_var = circuit.emulated_mul(&taus_var[0], &taus_var[1])?;
    let second_mul_add_tmp1 = circuit.emulated_mul(&taus_mul_var, &wire_evals_var[1])?;
    let second_mul_add_tmp2 = circuit.emulated_mul(&taus_var[2], &wire_evals_var[2])?;
    let second_mul_add_var = circuit.emulated_add(&second_mul_add_tmp1, &second_mul_add_tmp2)?;

    let mut lookup_wire_eval_var = circuit.emulated_add(&wire_evals_var[5], &first_mul_add_var)?;
    lookup_wire_eval_var = circuit.emulated_add(&lookup_wire_eval_var, &second_mul_add_var)?;

    let third_mul_add_tmp1 = circuit.emulated_mul(table_dom_sep_eval_var, &taus_var[0])?;
    let third_mul_add_tmp2 = circuit.emulated_mul(&taus_var[1], key_table_eval_var)?;
    let third_mul_add_var = circuit.emulated_add(&third_mul_add_tmp1, &third_mul_add_tmp2)?;

    let fourth_mul_add_tmp1 = circuit.emulated_mul(&taus_mul_var, &wire_evals_var[3])?;
    let fourth_mul_add_tmp2 = circuit.emulated_mul(&taus_var[2], &wire_evals_var[4])?;
    let fourth_mul_add_var = circuit.emulated_add(&fourth_mul_add_tmp1, &fourth_mul_add_tmp2)?;

    let mut table_eval_var = circuit.emulated_add(range_table_eval_var, &third_mul_add_var)?;
    table_eval_var = circuit.emulated_add(&table_eval_var, &fourth_mul_add_var)?;

    let first_sub_var = circuit.emulated_sub(lk_alpha_var, &lookup_wire_eval_var)?;
    let second_sub_var = circuit.emulated_sub(lk_alpha_var, &table_eval_var)?;

    let fifth_mul_add_tmp1 = circuit.emulated_mul(m_poly_eval_var, &first_sub_var)?;
    let fifth_mul_add_var = circuit.emulated_sub(&fifth_mul_add_tmp1, &second_sub_var)?;

    let lambda_mul_first_sub_var = circuit.emulated_mul(lambda_var, &first_sub_var)?;
    let res_tmp = circuit.emulated_mul(&lambda_mul_first_sub_var, &second_sub_var)?;
    let res = circuit.emulated_add(&fifth_mul_add_var, &res_tmp)?;
    Ok(res)
}

#[cfg(test)]
mod test {
    use super::super::MLEVerifyingKeyVar;
    use super::*;

    use crate::nightfall::mle::{
        mle_structs::MLEChallenges, snark::tests::gen_circuit_for_test, MLEPlonk,
    };
    use crate::{
        nightfall::circuit::plonk_partial_verifier::{EmulatedMLEChallenges, SAMLEProofVar},
        proof_system::UniversalSNARK,
        transcript::{RescueTranscript, Transcript},
    };
    use ark_ec::short_weierstrass::Affine;
    use ark_ec::{pairing::Pairing, short_weierstrass::Projective};

    use ark_poly::DenseMultilinearExtension;
    use ark_std::{string::ToString, sync::Arc};

    use jf_primitives::{
        pcs::{Accumulation, PolynomialCommitmentScheme},
        rescue::RescueParameter,
    };

    use jf_relation::{gadgets::ecc::HasTEForm, Circuit, PlonkType};
    use jf_utils::test_rng;

    use nf_curves::grumpkin::{Fq as FqGrumpkin, Grumpkin};

    use crate::{
        nightfall::{mle::zeromorph::zeromorph_protocol::Zeromorph, UnivariateIpaPCS},
        transcript::{rescue::RescueTranscriptVar, CircuitTranscript},
    };

    const RANGE_BIT_LEN_FOR_TEST: usize = 16;

    #[test]
    fn test_compute_mle_proof_challenges_var() -> Result<(), CircuitError> {
        test_compute_mle_proof_challenges_var_helper::<
            Grumpkin,
            FqGrumpkin,
            _,
            Zeromorph<UnivariateIpaPCS<Grumpkin>>,
        >()
    }

    fn test_compute_mle_proof_challenges_var_helper<E, F, P, PCS>() -> Result<(), CircuitError>
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
        F: RescueParameter + EmulationConfig<E::ScalarField> + PrimeField,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        PCS: PolynomialCommitmentScheme<
            Evaluation = P::ScalarField,
            Point = Vec<P::ScalarField>,
            Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
            Commitment = Affine<P>,
        >,
        PCS: Accumulation,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
    {
        // 1. Simulate universal setup
        let rng = &mut test_rng();

        let srs = PCS::gen_srs_for_testing(rng, 10)
            .map_err(|_| CircuitError::ParameterError("SRS generation failed".to_string()))?;

        // 2. Create circuits
        let circuits = (0..6)
            .map(|i| {
                let m = 2 + i / 3;
                let a0 = 1 + i % 3;
                gen_circuit_for_test(m, a0, PlonkType::UltraPlonk, true).map_err(|_| {
                    CircuitError::ParameterError("Circuit generation failed".to_string())
                })
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;

        let public_inputs = circuits
            .iter()
            .map(|c| c.public_input().unwrap())
            .collect::<Vec<Vec<_>>>();
        // 3. Preprocessing
        let (pk1, vk1) = MLEPlonk::<PCS>::preprocess_helper(&circuits[0], &srs)
            .map_err(|_| CircuitError::ParameterError("Preprocessing failed".to_string()))?;
        let (pk2, vk2) = MLEPlonk::<PCS>::preprocess_helper(&circuits[3], &srs)
            .map_err(|_| CircuitError::ParameterError("Preprocessing failed".to_string()))?;

        // 4. Proving
        let mut proofs = vec![];

        for (i, cs) in circuits.iter().enumerate() {
            let pk_ref = if i < 3 { &pk1 } else { &pk2 };

            proofs.push(
                MLEPlonk::<PCS>::recursive_prove::<_, _, RescueTranscript<F>>(
                    rng, cs, pk_ref, None,
                )
                .unwrap(),
            );
        }

        // 5. Verification

        for (i, proof) in proofs.iter().enumerate() {
            let vk_ref = if i < 3 { &vk1 } else { &vk2 };

            let pi = public_inputs[i][0];
            let mut transcript = <RescueTranscript<F> as Transcript>::new_transcript(b"mle_plonk");
            let mle_challenges =
                MLEChallenges::new_recursion(&proof.proof, &[pi], vk_ref, &mut transcript)
                    .map_err(|_| {
                        CircuitError::ParameterError("MLE challenge generation failed".to_string())
                    })?;

            let mut plonk_circuit = PlonkCircuit::<F>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);
            let mle_proof_var = SAMLEProofVar::from_struct::<P>(&mut plonk_circuit, &proof.proof)?;
            let vk_var = MLEVerifyingKeyVar::new::<F, P>(&mut plonk_circuit, vk_ref)?;

            let mut transcript_var = RescueTranscriptVar::<F>::new_transcript(&mut plonk_circuit);
            let pi_var = plonk_circuit.create_emulated_variable(pi)?;
            let mle_challenges_var =
                EmulatedMLEChallenges::<E::ScalarField>::compute_challenges_vars::<PCS, P>(
                    &mut plonk_circuit,
                    &vk_var,
                    &pi_var,
                    &mle_proof_var,
                    &mut transcript_var,
                )?;
            assert!(plonk_circuit.check_circuit_satisfiability(&[]).is_ok());
            assert_eq!(
                mle_challenges.beta,
                plonk_circuit.emulated_witness(&mle_challenges_var.beta)?
            );
            assert_eq!(
                mle_challenges.gamma,
                plonk_circuit.emulated_witness(&mle_challenges_var.gamma)?
            );
            assert_eq!(
                mle_challenges.alpha,
                plonk_circuit.emulated_witness(&mle_challenges_var.alpha)?
            );
            assert_eq!(
                mle_challenges.tau,
                plonk_circuit.emulated_witness(&mle_challenges_var.tau)?
            );

            assert_eq!(
                mle_challenges.delta,
                plonk_circuit.emulated_witness(&mle_challenges_var.delta)?
            );
        }
        Ok(())
    }

    // #[test]
    // fn test_verification_circuits() {
    //     test_verification_circuits_helper::<
    //         Grumpkin,
    //         FqGrumpkin,
    //         _,
    //         Zeromorph<UnivariateIpaPCS<Grumpkin>>,
    //     >(PlonkType::UltraPlonk)
    //     .expect("test failed for Grumpkin");
    // }

    // fn test_verification_circuits_helper<E, F, P, PCS>(
    //     plonk_type: PlonkType,
    // ) -> Result<(), CircuitError>
    // where
    //     E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
    //     F: RescueParameter,
    //     P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
    //     E::ScalarField: EmulationConfig<F>,
    //     PCS: PolynomialCommitmentScheme<
    //         Evaluation = P::ScalarField,
    //         Point = Vec<P::ScalarField>,
    //         Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
    //         Commitment = Affine<P>,
    //     >,
    //     PCS: Accumulation,
    // {
    //     // 1. Simulate universal setup
    //     let rng = &mut test_rng();

    //     let srs = PCS::gen_srs_for_testing(rng, 10)
    //         .map_err(|_| CircuitError::ParameterError("SRS generation failed".to_string()))?;

    //     // 2. Create circuits
    //     let circuits = (0..6)
    //         .map(|i| {
    //             let m = 2 + i / 3;
    //             let a0 = 1 + i % 3;
    //             gen_circuit_for_test::<P::ScalarField>(m, a0, plonk_type).map_err(|_| {
    //                 CircuitError::ParameterError("Circuit generation failed".to_string())
    //             })
    //         })
    //         .collect::<Result<Vec<_>, CircuitError>>()?;
    //     // 3. Preprocessing
    //     let (pk1, vk1) = MLEPlonk::<PCS>::preprocess_helper::<
    //         P::ScalarField,
    //         PlonkCircuit<P::ScalarField>,
    //     >(&circuits[0], &srs)
    //     .map_err(|_| CircuitError::ParameterError("Key generation failed".to_string()))?;
    //     let (pk2, vk2) = MLEPlonk::<PCS>::preprocess_helper::<
    //         P::ScalarField,
    //         PlonkCircuit<P::ScalarField>,
    //     >(&circuits[3], &srs)
    //     .map_err(|_| CircuitError::ParameterError("Key generation failed".to_string()))?;
    //     // 4. Proving
    //     let mut proofs = vec![];

    //     for (i, cs) in circuits.iter().enumerate() {
    //         let pk_ref = if i < 3 { &pk1 } else { &pk2 };

    //         proofs
    //             .push(MLEPlonk::<PCS>::prove::<_, _, _, RescueTranscript<F>>(cs, pk_ref).unwrap());
    //     }

    //     // 5. Verification
    //     let public_inputs: Vec<Vec<E::ScalarField>> = circuits
    //         .iter()
    //         .map(|cs| cs.public_input())
    //         .collect::<Result<Vec<Vec<E::ScalarField>>, _>>(
    //     )?;
    //     for (i, proof) in proofs.iter().enumerate() {
    //         let vk_ref = if i < 3 { &vk1 } else { &vk2 };

    //         let mut transcript = <RescueTranscript<F> as Transcript>::new_transcript(b"mle_plonk");

    //         let num_vars = proof.sumcheck_proof.point.len();
    //         let n = 1usize << num_vars;

    //         let mut plonk_circuit =
    //             PlonkCircuit::<E::ScalarField>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);

    //         let mut pi_evals = public_inputs[i].to_vec();
    //         pi_evals.resize(n, P::ScalarField::zero());
    //         let pi_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, pi_evals);
    //         let pi_poly_var =
    //             DenseMultilinearExtensionVar::new::<E::ScalarField>(&mut plonk_circuit, &pi_poly)?;

    //         let MLEChallenges {
    //             beta,
    //             gamma,
    //             alpha,
    //             taus,
    //             lk_alpha,
    //         } = MLEChallenges::<P::ScalarField>::new(
    //             proof,
    //             &public_inputs[i],
    //             vk_ref,
    //             &mut transcript,
    //         )
    //         .map_err(|_| {
    //             CircuitError::ParameterError("MLE challenge generation failed".to_string())
    //         })?;

    //         let beta_var = plonk_circuit.create_variable(beta)?;
    //         let gamma_var = plonk_circuit.create_variable(gamma)?;
    //         let alpha_var = plonk_circuit.create_variable(alpha)?;
    //         let mut taus_var = [0 as Variable; 3];

    //         for (tau, tau_var) in taus.iter().zip(taus_var.iter_mut()) {
    //             *tau_var = plonk_circuit.create_variable(*tau)?;
    //         }
    //         let lk_alpha_var = plonk_circuit.create_variable(lk_alpha)?;

    //         let deferred_check =
    //             <VPSumCheck<P> as ZeroCheck<P>>::verify(&proof.sumcheck_proof, &mut transcript)?;

    //         let zero_check_point = &deferred_check.point;

    //         let x1 = *zero_check_point
    //             .last()
    //             .ok_or(CircuitError::ParameterError("Could not get x1".to_string()))?;

    //         let x1_var = plonk_circuit.create_variable(x1)?;

    //         let pi_poly_eval =
    //             pi_poly
    //                 .evaluate(&deferred_check.point)
    //                 .ok_or(CircuitError::ParameterError(
    //                     "Could not evaluate pi poly".to_string(),
    //                 ))?;

    //         let mut defferred_check_point_var = Vec::<Variable>::new();
    //         for coord_var in deferred_check.point.iter() {
    //             defferred_check_point_var.push(plonk_circuit.create_variable(*coord_var)?);
    //         }
    //         let pi_poly_eval_var = mle_evaluation_circuit::<E::ScalarField>(
    //             &mut plonk_circuit,
    //             &pi_poly_var,
    //             &defferred_check_point_var,
    //         )?;

    //         let gate_eval = eval_gate_equation(
    //             &vk_ref.gate_info,
    //             &proof.evals.selector_evals,
    //             &proof.evals.wire_evals[..5],
    //             pi_poly_eval,
    //         );

    //         let mut selector_evals_var = Vec::<Variable>::new();
    //         for selector_eval in proof.evals.selector_evals.iter() {
    //             selector_evals_var.push(plonk_circuit.create_variable(*selector_eval)?);
    //         }

    //         let mut wire_evals_var = Vec::<Variable>::new();
    //         for wire_eval in proof.evals.wire_evals[..5].iter() {
    //             wire_evals_var.push(plonk_circuit.create_variable(*wire_eval)?);
    //         }

    //         let gate_eval_var = eval_gate_equation_circuit(
    //             &mut plonk_circuit,
    //             &vk_ref.gate_info,
    //             &selector_evals_var,
    //             &wire_evals_var,
    //             pi_poly_eval_var,
    //         )?;

    //         let mut eval = P::ScalarField::zero();
    //         for (i, zp) in zero_check_point.iter().enumerate() {
    //             eval += *zp * P::ScalarField::from(1u64 << i);
    //         }

    //         let mut identity_evals = vec![];
    //         let field_n = P::ScalarField::from(1u64 << num_vars);
    //         for i in 0..proof.evals.permutation_evals.len() {
    //             identity_evals.push(eval + (P::ScalarField::from(i as u64) * field_n));
    //         }

    //         let perm_eval = eval_perm_gate(
    //             &proof.evals.product_eval,
    //             &proof.evals.frac_eval,
    //             &proof.evals.wire_evals,
    //             &identity_evals,
    //             &proof.evals.permutation_evals,
    //             alpha,
    //             beta,
    //             gamma,
    //             x1,
    //         )?;

    //         let mut product_eval_var = Vec::<Variable>::new();
    //         for product_eval in proof.evals.product_eval.iter() {
    //             product_eval_var.push(plonk_circuit.create_variable(*product_eval)?);
    //         }
    //         let mut frac_eval_var = Vec::<Variable>::new();
    //         for frac_eval in proof.evals.frac_eval.iter() {
    //             frac_eval_var.push(plonk_circuit.create_variable(*frac_eval)?);
    //         }
    //         let mut wire_evals_var = Vec::<Variable>::new();
    //         for wire_eval in proof.evals.wire_evals.iter() {
    //             wire_evals_var.push(plonk_circuit.create_variable(*wire_eval)?);
    //         }

    //         let mut id_eval = plonk_circuit.zero();
    //         for (i, zp) in proof.sumcheck_proof.point.iter().enumerate() {
    //             let zc_var = plonk_circuit.create_variable(*zp)?;
    //             let tmp = plonk_circuit.mul_constant(zc_var, &P::ScalarField::from(1u64 << i))?;
    //             id_eval = plonk_circuit.add(id_eval, tmp)?;
    //         }

    //         let mut identity_evals_var = vec![id_eval];
    //         let n_var = plonk_circuit.create_variable(P::ScalarField::from(1u64 << num_vars))?;
    //         for i in 1..proof.evals.permutation_evals.len() {
    //             let tmp = plonk_circuit.mul_constant(n_var, &P::ScalarField::from(i as u64))?;
    //             identity_evals_var.push(plonk_circuit.add(id_eval, tmp)?)
    //         }

    //         let mut permutation_evals_var = Vec::<Variable>::new();
    //         for permutation_eval in proof.evals.permutation_evals.iter() {
    //             permutation_evals_var.push(plonk_circuit.create_variable(*permutation_eval)?);
    //         }

    //         let perm_eval_var = eval_perm_gate_circuit(
    //             &mut plonk_circuit,
    //             &product_eval_var,
    //             &frac_eval_var,
    //             &wire_evals_var,
    //             &identity_evals_var,
    //             &permutation_evals_var,
    //             alpha_var,
    //             beta_var,
    //             gamma_var,
    //             x1_var,
    //         )?;

    //         let eval = gate_eval + alpha * perm_eval;
    //         let eval_var = plonk_circuit.mul_add(
    //             &[gate_eval_var, plonk_circuit.one(), alpha_var, perm_eval_var],
    //             &[E::ScalarField::one(), E::ScalarField::one()],
    //         )?;

    //         assert_eq!(eval, plonk_circuit.witness(eval_var)?);

    //         if let Some(lookup_proof) = &proof.lookup_proof {
    //             let gkr_deferred_check = LookupCheck::<P>::verify(
    //                 &proof.lookup_proof.as_ref().unwrap().gkr_proof.0,
    //                 proof.lookup_proof.as_ref().unwrap().gkr_proof.1,
    //                 &mut transcript,
    //             )?;

    //             let lookup_eval = eval_lookup_equation(
    //                 &taus,
    //                 lookup_proof.lookup_evals.wire_polys_evals.as_slice(),
    //                 lookup_proof.lookup_evals.range_table_eval,
    //                 lookup_proof.lookup_evals.key_table_eval,
    //                 lookup_proof.lookup_evals.table_dom_sep_eval,
    //                 lookup_proof.lookup_evals.q_dom_sep_eval,
    //                 lookup_proof.lookup_evals.m_poly_eval,
    //                 lk_alpha,
    //                 *gkr_deferred_check.lambda(),
    //             );

    //             let mut wire_polys_evals_var = Vec::<Variable>::new();
    //             for wire_poly_eval in lookup_proof.lookup_evals.wire_polys_evals.iter() {
    //                 wire_polys_evals_var.push(plonk_circuit.create_variable(*wire_poly_eval)?);
    //             }

    //             let range_table_eval_var =
    //                 plonk_circuit.create_variable(lookup_proof.lookup_evals.range_table_eval)?;
    //             let key_table_eval_var =
    //                 plonk_circuit.create_variable(lookup_proof.lookup_evals.key_table_eval)?;
    //             let table_dom_sep_eval_var =
    //                 plonk_circuit.create_variable(lookup_proof.lookup_evals.table_dom_sep_eval)?;
    //             let q_dom_sep_eval_var =
    //                 plonk_circuit.create_variable(lookup_proof.lookup_evals.q_dom_sep_eval)?;
    //             let m_poly_eval_var =
    //                 plonk_circuit.create_variable(lookup_proof.lookup_evals.m_poly_eval)?;
    //             let lambda_var = plonk_circuit.create_variable(*gkr_deferred_check.lambda())?;

    //             let lookup_eval_var = eval_lookup_equation_circuit(
    //                 &mut plonk_circuit,
    //                 &taus_var,
    //                 &wire_polys_evals_var,
    //                 range_table_eval_var,
    //                 key_table_eval_var,
    //                 table_dom_sep_eval_var,
    //                 q_dom_sep_eval_var,
    //                 m_poly_eval_var,
    //                 lk_alpha_var,
    //                 lambda_var,
    //             )?;
    //             assert_eq!(lookup_eval, plonk_circuit.witness(lookup_eval_var)?);
    //             assert!(plonk_circuit.check_circuit_satisfiability(&[]).is_ok());
    //         }
    //     }
    //     Ok(())
    // }

    // #[test]
    // fn test_partial_verify_mle_plonk() -> Result<(), PlonkError> {
    //     partial_verify_mle_helper::<Zeromorph<UnivariateIpaPCS<Grumpkin>>, _, _>()
    // }

    // fn partial_verify_mle_helper<PCS, P, F>() -> Result<(), PlonkError>
    // where
    //     PCS: Accumulation<
    //         Commitment = Affine<P>,
    //         Evaluation = P::ScalarField,
    //         Point = Vec<P::ScalarField>,
    //         Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
    //     >,
    //     P: HasTEForm<BaseField = F>,
    //     F: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
    //     P::ScalarField: PrimeField + RescueParameter + EmulationConfig<F>,
    // {
    //     // 1. Simulate universal setup
    //     let rng = &mut test_rng();

    //     let srs = PCS::gen_srs_for_testing(rng, 10)?;

    //     // 2. Create circuits
    //     let circuits = (0..6)
    //         .map(|i| {
    //             let m = 2 + i / 3;
    //             let a0 = 1 + i % 3;
    //             gen_circuit_for_test::<P::ScalarField>(m, a0, PlonkType::UltraPlonk)
    //         })
    //         .collect::<Result<Vec<_>, PlonkError>>()?;
    //     // 3. Preprocessing
    //     let (pk1, vk1) = MLEPlonk::<PCS>::preprocess_helper::<
    //         P::ScalarField,
    //         PlonkCircuit<P::ScalarField>,
    //     >(&circuits[0], &srs)?;
    //     let (pk2, vk2) = MLEPlonk::<PCS>::preprocess_helper::<
    //         P::ScalarField,
    //         PlonkCircuit<P::ScalarField>,
    //     >(&circuits[3], &srs)?;
    //     // 4. Proving
    //     let mut proofs = vec![];

    //     for (i, cs) in circuits.iter().enumerate() {
    //         let pk_ref = if i < 3 { &pk1 } else { &pk2 };

    //         proofs.push(
    //             MLEPlonk::<PCS>::recursive_prove::<_, _, RescueTranscript<F>>(
    //                 rng, cs, pk_ref, None,
    //             )
    //             .unwrap(),
    //         );
    //     }

    //     // 5. Verification
    //     let public_inputs: Vec<Vec<P::ScalarField>> = circuits
    //         .iter()
    //         .map(|cs| cs.public_input())
    //         .collect::<Result<Vec<Vec<P::ScalarField>>, _>>(
    //     )?;

    //     for (i, (proof, public_input)) in proofs.iter().zip(public_inputs.iter()).enumerate() {
    //         let (pk, vk) = if i < 3 { (&pk1, &vk1) } else { (&pk2, &vk2) };
    //         let mut circuit = PlonkCircuit::<P::BaseField>::new_ultra_plonk(16);
    //         let vk_var = MLEVerifyingKeyVar::new::<F, P>(&mut circuit, vk)?;
    //         let pi = public_input
    //             .iter()
    //             .map(|pi| circuit.create_emulated_variable(*pi))
    //             .collect::<Result<Vec<EmulatedVariable<P::ScalarField>>, _>>()?;
    //         partial_verify_mle_plonk(&mut circuit, &vk_var, proof, &pi, &vk.gate_info)?;

    //         let mut acc = proof.proof.accumulator.clone();
    //         let opening_proof = acc.multi_open(&pk.pcs_prover_params).unwrap();

    //         MLEPlonk::<PCS>::verify_recursive_proof::<F, P, _, RescueTranscript<F>>(
    //             proof,
    //             &opening_proof,
    //             vk,
    //             public_input,
    //             rng,
    //         )?;
    //         circuit.check_circuit_satisfiability(&[]).unwrap();
    //     }
    //     Ok(())
    // }
}
