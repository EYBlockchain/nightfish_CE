// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Circuits for the polynomial evaluations within Plonk verifiers.

use ark_ff::PrimeField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::{format, iter, string::ToString, vec, vec::Vec};
use jf_primitives::rescue::RescueParameter;
use jf_relation::{
    errors::CircuitError,
    gadgets::{ecc::HasTEForm, EmulatedVariable, EmulationConfig},
    Circuit, PlonkCircuit, Variable,
};
use num_bigint::BigUint;

use super::{ChallengesVar, PlookupEvalsVarNative, ProofEvalsVarNative};

/// This helper function generate the variables for the following data
/// - Circuit evaluation of vanishing polynomial at point `zeta` i.e., output =
///   zeta ^ domain_size - 1 mod Fr::modulus
/// - Evaluations of the first and the last lagrange polynomial at point `zeta`
///
/// Note that outputs and zeta are both Fr element
/// so this needs to be carried out over a non-native circuit
/// using parameter m.
/// The output is lifted to Fq and in the EmulatedVariable form for:
///
/// - zeta^n
/// - zeta^n - 1
/// - lagrange evaluation at 1
///
/// Note that evaluation at n is commented out as we don't need it for
/// partial verification circuit.
#[allow(dead_code)]
pub(super) fn evaluate_poly_helper<E, F>(
    circuit: &mut PlonkCircuit<F>,
    zeta_emul_var: &EmulatedVariable<E::ScalarField>,
    domain_size: usize,
) -> Result<[EmulatedVariable<E::ScalarField>; 3], CircuitError>
where
    E: HasTEForm<BaseField = F>,
    F: PrimeField + EmulationConfig<E::ScalarField> + RescueParameter,
    E::ScalarField: EmulationConfig<F> + PrimeField + RescueParameter,
{
    // constants
    let domain_size_val = E::ScalarField::from(domain_size as u64);

    // ================================
    // compute zeta^n - 1
    // ================================

    // compute zeta^n for n = domain_size a power of 2
    let mut ctr = 1;
    let mut zeta_n_emul_var = zeta_emul_var.clone();
    while ctr < domain_size {
        ctr <<= 1;
        zeta_n_emul_var = circuit.emulated_mul(&zeta_n_emul_var, &zeta_n_emul_var)?;
    }

    // to compute zeta^n -1 we need to compute it over Fr
    // we cannot simply do
    //  let zeta_n_minus_one_var = circuit.sub(zeta_n_var, circuit.one())?;
    // since it may be overflowing if zeta_n = 0
    //
    // Option 1: to write the subtraction in non-native field
    //
    // Option 2, which is what is implemented here
    // - if zeta_n = 0, output Fr::modulus - 1
    // - else output zeta_n -1
    // this circuit should still be cheaper than non-native circuit.
    //
    //
    // Question(ZZ): second thought, this should be fine since we know that
    // zeta !=0 mod fr with 1 - 1/|Fr| probability as zeta is a output from
    // RO. Nonetheless, I am implementing it with non-native field first.
    // We may switch to native field if this is fine...
    //

    // zeta^n = zeta_n_minus_1 + 1

    let one_emul_var = circuit.emulated_one();
    let zeta_n_minus_one_emul_var = circuit.emulated_sub(&zeta_n_emul_var, &one_emul_var)?;

    // ================================
    // evaluate lagrange at 1
    //  lagrange_1_eval = (zeta^n - 1) / (zeta - 1) / domain_size
    //
    // which is proven via
    //  domain_size * lagrange_1_eval * (zeta - 1) = zeta^n - 1 mod Fr::modulus
    // ================================

    // lagrange_1_eval
    let zeta_minus_one_emul_var = circuit.emulated_sub(zeta_emul_var, &one_emul_var)?;
    let divisor_emul_var =
        circuit.emulated_mul_constant(&zeta_minus_one_emul_var, domain_size_val)?;

    let zeta_n_minus_one = circuit.emulated_witness(&zeta_n_minus_one_emul_var)?;
    let divisor = circuit.emulated_witness(&divisor_emul_var)?;

    let lagrange_1_eval = zeta_n_minus_one / divisor;
    let lagrange_1_eval_emul_var = circuit.create_emulated_variable(lagrange_1_eval)?;
    // Constrain the lagrange_1_eval to be correct.
    circuit.emulated_mul_gate(
        &divisor_emul_var,
        &lagrange_1_eval_emul_var,
        &zeta_n_minus_one_emul_var,
    )?;

    Ok([
        zeta_n_emul_var,
        zeta_n_minus_one_emul_var,
        lagrange_1_eval_emul_var,
    ])
}

/// This helper function generate the variables for the following data
/// - Circuit evaluation of vanishing polynomial at point `zeta` i.e., output =
///   zeta ^ domain_size - 1 mod Fr::modulus
/// - Evaluations of the first and the last lagrange polynomial at point `zeta`
///
/// Note that outputs and zeta are both Fr element
/// so this needs to be carried out over a non-native circuit
/// using parameter m.
/// The output is lifted to Fq and in the EmulatedVariable form for:
///
/// - zeta^n
/// - zeta^n - 1
/// - lagrange evaluation at 1
///
/// Note that evaluation at n is commented out as we don't need it for
/// partial verification circuit.
pub(super) fn evaluate_poly_helper_native<F>(
    circuit: &mut PlonkCircuit<F>,
    zeta_var: Variable,
    gen_inv: F,
    domain_size: usize,
) -> Result<[Variable; 4], CircuitError>
where
    F: PrimeField + RescueParameter,
{
    // ================================
    // compute zeta^n - 1
    // ================================
    // In the non-base case, `domain_size` is considered constant. It only depends on the layer of recursion.
    let mut zeta_n_var = zeta_var;
    let mut ctr = 1;
    while ctr < domain_size {
        ctr *= 2;
        zeta_n_var = circuit.mul(zeta_n_var, zeta_n_var)?;
    }

    // zeta^n = zeta_n_minus_1 + 1
    let zeta_n_minus_one_var = circuit.add_constant(zeta_n_var, &-F::from(1u8))?;

    // ================================
    // evaluate lagrange at 1
    //  lagrange_1_eval = (zeta^n - 1) / (zeta - 1) / domain_size
    //
    // which is proven via
    //  domain_size * lagrange_1_eval * (zeta - 1) = zeta^n - 1 mod Fr::modulus
    // ================================

    let domain_size = F::from(domain_size as u64);

    // lagrange_1_eval

    let divisor_var = circuit.lin_comb(&[domain_size], &-domain_size, &[zeta_var])?;

    let zeta_n_minus_one = circuit.witness(zeta_n_minus_one_var)?;
    let divisor = circuit.witness(divisor_var)?;

    let lagrange_1_eval = zeta_n_minus_one / divisor;
    let lagrange_1_eval_var = circuit.create_variable(lagrange_1_eval)?;
    // Constrain the lagrange_1_eval to be correct.
    circuit.mul_gate(divisor_var, lagrange_1_eval_var, zeta_n_minus_one_var)?;

    // Compute lagrange_n_eval
    let divisor_var = circuit.lin_comb(&[domain_size], &(-domain_size * gen_inv), &[zeta_var])?;
    let numerator_var = circuit.mul_constant(zeta_n_minus_one_var, &gen_inv)?;
    let divisor = circuit.witness(divisor_var)?;
    let numerator = circuit.witness(numerator_var)?;
    let lagrange_n_eval = numerator / divisor;
    let lagrange_n_eval_var = circuit.create_variable(lagrange_n_eval)?;
    // Constrain the lagrange_n_eval to be correct.
    circuit.mul_gate(divisor_var, lagrange_n_eval_var, numerator_var)?;

    Ok([
        zeta_n_var,
        zeta_n_minus_one_var,
        lagrange_1_eval_var,
        lagrange_n_eval_var,
    ])
}

pub(super) fn evaluate_poly_helper_native_base<F>(
    circuit: &mut PlonkCircuit<F>,
    zeta_var: Variable,
    gen_inv_var: Variable,
    domain_size_var: Variable,
    max_domain_size: usize,
) -> Result<[Variable; 4], CircuitError>
where
    F: PrimeField + RescueParameter,
{
    // ================================
    // compute zeta^n - 1
    // ================================
    let zeta_n_var = zeta_to_n(circuit, zeta_var, domain_size_var, max_domain_size)?;

    // zeta^n = zeta_n_minus_1 + 1
    let zeta_n_minus_one_var = circuit.add_constant(zeta_n_var, &-F::from(1u8))?;

    // ================================
    // evaluate lagrange at 1
    //  lagrange_1_eval = (zeta^n - 1) / (zeta - 1) / domain_size
    //
    // which is proven via
    //  domain_size * lagrange_1_eval * (zeta - 1) = zeta^n - 1 mod Fr::modulus
    // ================================

    // lagrange_1_eval

    let divisor_var = circuit.mul_add(
        &[domain_size_var, zeta_var, domain_size_var, circuit.one()],
        &[F::one(), -F::one()],
    )?;

    let zeta_n_minus_one = circuit.witness(zeta_n_minus_one_var)?;
    let divisor = circuit.witness(divisor_var)?;

    let lagrange_1_eval = zeta_n_minus_one / divisor;
    let lagrange_1_eval_var = circuit.create_variable(lagrange_1_eval)?;
    // Constrain the lagrange_1_eval to be correct.
    circuit.mul_gate(divisor_var, lagrange_1_eval_var, zeta_n_minus_one_var)?;

    // Compute lagrange_n_eval
    let divisor_var = circuit.mul_add(
        &[domain_size_var, zeta_var, domain_size_var, gen_inv_var],
        &[F::one(), -F::one()],
    )?;
    let numerator_var = circuit.mul(zeta_n_minus_one_var, gen_inv_var)?;
    let divisor = circuit.witness(divisor_var)?;
    let numerator = circuit.witness(numerator_var)?;
    let lagrange_n_eval = numerator / divisor;
    let lagrange_n_eval_var = circuit.create_variable(lagrange_n_eval)?;
    // Constrain the lagrange_n_eval to be correct.
    circuit.mul_gate(divisor_var, lagrange_n_eval_var, numerator_var)?;

    Ok([
        zeta_n_var,
        zeta_n_minus_one_var,
        lagrange_1_eval_var,
        lagrange_n_eval_var,
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
/// The public input polynomial evaluation for the merged circuit is:
///
/// \sum_{i=0..l/2} L_{i,H}(z) * pub_input[i] +
/// \sum_{i=0..l/2} L_{n-i,H}(z) * pub_input[l/2+i]
pub fn evaluate_pi_poly_circuit_native<F>(
    circuit: &mut PlonkCircuit<F>,
    domain_size: usize,
    pub_inputs_var: &[Variable],
    zeta_var: &Variable,
) -> Result<Variable, CircuitError>
where
    F: PrimeField + RescueParameter,
{
    // compute zeta^n for n = domain_size a power of 2
    let mut ctr = 1;
    let mut zeta_n_var = *zeta_var;
    while ctr < domain_size {
        ctr <<= 1;
        zeta_n_var = circuit.mul(zeta_n_var, zeta_n_var)?;
    }

    // vanish_poly_at_zeta = zeta^n - 1
    let vanish_eval_var: usize = circuit.add_constant(zeta_n_var, &-F::from(1u8))?;

    // compute v_i = g^i / n in the clear
    let domain = Radix2EvaluationDomain::<F>::new(domain_size).unwrap();

    let pi_len = pub_inputs_var.len();
    let v_i = (0..pub_inputs_var.len())
        .map(|x| domain.element(x) / F::from(domain_size as u64))
        .collect::<Vec<F>>();

    // compute L_{i,H}(zeta) = Z_H(zeta) * v_i / (zeta - g^i)
    // where Z_H(z) is the vanishing evaluation.
    // we sum over l, the length of the public inputs.
    let mut lagrange_eval_emul_var: Vec<Variable> = Vec::new();
    let zeta = circuit.witness(*zeta_var)?;
    for (i, v_item) in v_i.iter().enumerate().take(pi_len) {
        // compute L_{i,H}(zeta) and related values in the clear
        let g_i = domain.element(i);
        let eval_i = circuit.witness(vanish_eval_var)? * v_item / (zeta - g_i);
        let eval_i_var = circuit.create_variable(eval_i)?;

        let wires = [
            eval_i_var,
            *zeta_var,
            circuit.zero(),
            circuit.zero(),
            vanish_eval_var,
        ];
        circuit.quad_poly_gate(
            &wires,
            &[-g_i, F::zero(), F::zero(), F::zero()],
            &[F::one(), F::zero()],
            *v_item,
            F::zero(),
        )?;

        // finish
        lagrange_eval_emul_var.push(eval_i_var);
    }

    // \sum_{i=0..l/2} L_{i,H}(z) * pub_input[i] + \sum_{i=0..l/2} L_{n-i,H}(z)
    // * pub_input[l/2+i]
    let mut prod_to_sum = Vec::<Variable>::new();
    for i in 0..(pi_len / 2) {
        let wires = [
            lagrange_eval_emul_var[2 * i],
            pub_inputs_var[2 * i],
            lagrange_eval_emul_var[2 * i + 1],
            pub_inputs_var[2 * i + 1],
        ];
        prod_to_sum.push(circuit.mul_add(&wires, &[F::one(), F::one()])?);
    }
    if pi_len % 2 == 1 {
        prod_to_sum.push(circuit.mul(
            lagrange_eval_emul_var[pi_len - 1],
            pub_inputs_var[pi_len - 1],
        )?);
    }
    circuit.sum(&prod_to_sum)
}

/// Evaluate public input polynomial at point `z` using batch additions.
///
/// We compute Lagrange coefficients l_{i,H}(z) = Z_H(z) * (g^i / n) / (z - g^i) for i in 0..l,
/// then form the dot-product \sum_i l_{i,H}(z) * pub_input[i] via a single batch add of all products.
pub fn evaluate_pi_poly_circuit_emulated<E, F>(
    circuit: &mut PlonkCircuit<F>,
    domain_size: usize,
    pub_inputs_var: &[EmulatedVariable<E::ScalarField>],
    zeta_var: &EmulatedVariable<E::ScalarField>,
) -> Result<EmulatedVariable<E::ScalarField>, CircuitError>
where
    E: HasTEForm<BaseField = F>,
    F: PrimeField + EmulationConfig<E::ScalarField> + RescueParameter,
    E::ScalarField: PrimeField + EmulationConfig<F> + RescueParameter,
{
    // compute zeta^n for n = domain_size a power of 2
    let mut ctr = 1;
    let mut zeta_n_var = zeta_var.clone();
    while ctr < domain_size {
        ctr <<= 1;
        zeta_n_var = circuit.emulated_mul(&zeta_n_var, &zeta_n_var)?;
    }

    // vanish_poly_at_zeta = zeta^n - 1
    let vanish_eval_var = circuit.emulated_add_constant(&zeta_n_var, -E::ScalarField::from(1u8))?;

    let zeta_eval = circuit.emulated_witness(zeta_var)?;

    // Prepare domain powers and v_i = g^i / n
    let domain = Radix2EvaluationDomain::<E::ScalarField>::new(domain_size).unwrap();
    let n = E::ScalarField::from(domain_size as u64);
    let mut lagrange_vars = Vec::with_capacity(pub_inputs_var.len());

    // Compute each l_i and enforce (zeta - g^i)*l_i = Z_H(zeta)*v_i
    for i in 0..pub_inputs_var.len() {
        let g_i = domain.element(i);
        let v_i = g_i / n;
        // clear-text l_i
        let l_i = circuit.emulated_witness(&vanish_eval_var)? * v_i / (zeta_eval - g_i);
        let l_var = circuit.create_emulated_variable(l_i)?;

        // constraint: (zeta - g_i) * l_var == vanish_eval * v_i
        let diff = circuit.emulated_sub_constant(zeta_var, g_i)?;
        let lhs = circuit.emulated_mul(&diff, &l_var)?;
        circuit.emulated_mul_constant_gate(&vanish_eval_var, v_i, &lhs)?;

        lagrange_vars.push(l_var);
    }

    // Build all products l_i * pub_input[i]
    let mut products = Vec::with_capacity(pub_inputs_var.len());
    for (l_var, p_var) in lagrange_vars.iter().zip(pub_inputs_var.iter()) {
        products.push(circuit.emulated_mul(l_var, p_var)?);
    }

    // Sum all products in one batch add
    circuit.emulated_batch_add(&products)
}

/// Compute the constant term of the linearization polynomial:
/// For each instance j:
///
/// r_plonk_j
///  = PI - L1(x) * alpha^2 - alpha *
///  \prod_i=1..m-1 (w_{j,i} + beta * sigma_{j,i} + gamma)
///  * (w_{j,m} + gamma) * z_j(xw)
///
/// return r_0 = \sum_{j=1..m} alpha^{k_j} * r_plonk_j
/// where m is the number of instances, and k_j is the number of alpha power
/// terms added to the first j-1 instances.
///
/// - input evals: zeta^n, zeta^n-1, L_1(zeta) and L_n(zeta)
///
/// Note that this function cannot evaluate plookup verification circuits.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_lin_poly_constant_term_circuit_native<F>(
    circuit: &mut PlonkCircuit<F>,
    gen_inv: &F,
    challenges: &ChallengesVar,
    proof_evals: &ProofEvalsVarNative,
    pi: &Variable,
    evals: &[Variable; 4],
    lookup_evals: &Option<PlookupEvalsVarNative>,
) -> Result<Variable, CircuitError>
where
    F: PrimeField + RescueParameter,
{
    let zeta_var = challenges.zeta;

    // r_plonk
    //  = PI - L1(x) * alpha^2 - alpha *
    //  \prod_i=1..m-1 (w_{j,i} + beta * sigma_{j,i} + gamma)
    //  * (w_{j,m} + gamma) * z_j(xw)
    //
    // r_0 = r_plonk + r_lookup
    // where m is the number of instances, and k_j is the number of alpha power
    // terms added to the first j-1 instances.

    // =====================================================
    // r_plonk
    //  = - L1(x) * alpha^2 - alpha *
    //  \prod_i=1..m-1 (w_{i} + beta * sigma_{i} + gamma)
    //  * (w_{m} + gamma) * z(xw)
    // =====================================================

    // \prod_i=1..m-1 (w_{i} + beta * sigma_{i} + gamma)
    let num_wire_types = proof_evals.wires_evals.len();
    let mut prod = challenges.alphas[0];
    for (w_j_i_var, sigma_j_i_var) in proof_evals.wires_evals[..num_wire_types - 1]
        .iter()
        .zip(proof_evals.wire_sigma_evals.iter())
    {
        let wires = [
            challenges.gamma,
            *w_j_i_var,
            challenges.beta,
            *sigma_j_i_var,
        ];
        let sum = circuit.gen_quad_poly(
            &wires,
            &[F::one(), F::one(), F::zero(), F::zero()],
            &[F::zero(), F::one()],
            F::zero(),
        )?;
        prod = circuit.mul(prod, sum)?;
    }

    // tmp = (w_{m} + gamma) * z(xw)
    let wires = [
        proof_evals.wires_evals[num_wire_types - 1],
        proof_evals.perm_next_eval,
        challenges.gamma,
        proof_evals.perm_next_eval,
    ];
    let tmp = circuit.mul_add(&wires, &[F::one(), F::one()])?;

    // tmp = alpha *
    //  \prod_i=1..m-1 (w_{i} + beta * sigma_{i} + gamma)
    //  * (w_{m} + gamma) * z(xw)
    prod = circuit.mul(tmp, prod)?;

    // r_plonk
    let pi_eval = circuit.mul(*pi, evals[2])?;
    let wires = [pi_eval, prod, evals[2], challenges.alphas[1]];
    let non_lookup = circuit.gen_quad_poly(
        &wires,
        &[F::one(), -F::one(), F::zero(), F::zero()],
        &[F::zero(), -F::one()],
        F::zero(),
    )?;

    if let Some(lookup_evals) = lookup_evals {
        // We compute L_n(zeta) * (h_1 - h_1_next - alpha^2) - alpha * L_1(zeta)
        let wires = [
            lookup_evals.h_1_eval,
            lookup_evals.h_2_next_eval,
            challenges.alphas[1],
            circuit.zero(),
        ];
        let tmp = circuit.lc(&wires, &[F::one(), -F::one(), -F::one(), F::zero()])?;
        let term_one = circuit.mul_add(
            &[evals[3], tmp, evals[2], challenges.alphas[0]],
            &[F::one(), -F::one()],
        )?;

        // Now alpha^3 * (zeta  - domain_gen_inv) * prod_lookup_next * (gamma * ( 1 + beta) + h_1 + beta * h_1_next) * (gamma * (1 + beta) + beta * h_2_next)
        let wires = [
            challenges.alphas[2],
            zeta_var,
            circuit.zero(),
            circuit.zero(),
        ];
        let mut init = circuit.gen_quad_poly(
            &wires,
            &[-*gen_inv, F::zero(), F::zero(), F::zero()],
            &[F::one(), F::zero()],
            F::zero(),
        )?;
        init = circuit.mul(init, lookup_evals.prod_next_eval)?;
        let g_mul_one_b = circuit.mul_add(
            &[
                challenges.gamma,
                circuit.one(),
                challenges.gamma,
                challenges.beta,
            ],
            &[F::one(), F::one()],
        )?;
        let wires = [
            g_mul_one_b,
            lookup_evals.h_1_eval,
            challenges.beta,
            lookup_evals.h_1_next_eval,
        ];
        let tmp1 = circuit.gen_quad_poly(
            &wires,
            &[F::one(), F::one(), F::zero(), F::zero()],
            &[F::zero(), F::one()],
            F::zero(),
        )?;
        let tmp2 = circuit.mul_add(
            &[
                g_mul_one_b,
                circuit.one(),
                challenges.beta,
                lookup_evals.h_2_next_eval,
            ],
            &[F::one(), F::one()],
        )?;
        let mut term_two = circuit.mul(tmp1, tmp2)?;
        term_two = circuit.mul(term_two, init)?;

        let final_sum = circuit.sub(term_one, term_two)?;
        circuit.mul_add(
            &[final_sum, challenges.alphas[2], non_lookup, circuit.one()],
            &[F::one(), F::one()],
        )
    } else {
        Ok(non_lookup)
    }
}

/// Same as `compute_lin_poly_constant_term_circuit_native` but takes `gen_inv` as a variable.
/// This is used in the base layer of recursion where `gen_inv` cannot be hardcoded.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_lin_poly_constant_term_circuit_native_base<F>(
    circuit: &mut PlonkCircuit<F>,
    gen_inv_var: &Variable,
    challenges: &ChallengesVar,
    proof_evals: &ProofEvalsVarNative,
    pi: &Variable,
    evals: &[Variable; 4],
    lookup_evals: &Option<PlookupEvalsVarNative>,
) -> Result<Variable, CircuitError>
where
    F: PrimeField + RescueParameter,
{
    let zeta_var = challenges.zeta;

    // r_plonk
    //  = PI - L1(x) * alpha^2 - alpha *
    //  \prod_i=1..m-1 (w_{j,i} + beta * sigma_{j,i} + gamma)
    //  * (w_{j,m} + gamma) * z_j(xw)
    //
    // r_0 = r_plonk + r_lookup
    // where m is the number of instances, and k_j is the number of alpha power
    // terms added to the first j-1 instances.

    // =====================================================
    // r_plonk
    //  = - L1(x) * alpha^2 - alpha *
    //  \prod_i=1..m-1 (w_{i} + beta * sigma_{i} + gamma)
    //  * (w_{m} + gamma) * z(xw)
    // =====================================================

    // \prod_i=1..m-1 (w_{i} + beta * sigma_{i} + gamma)
    let num_wire_types = proof_evals.wires_evals.len();
    let mut prod = challenges.alphas[0];
    for (w_j_i_var, sigma_j_i_var) in proof_evals.wires_evals[..num_wire_types - 1]
        .iter()
        .zip(proof_evals.wire_sigma_evals.iter())
    {
        let wires = [
            challenges.gamma,
            *w_j_i_var,
            challenges.beta,
            *sigma_j_i_var,
        ];
        let sum = circuit.gen_quad_poly(
            &wires,
            &[F::one(), F::one(), F::zero(), F::zero()],
            &[F::zero(), F::one()],
            F::zero(),
        )?;
        prod = circuit.mul(prod, sum)?;
    }

    // tmp = (w_{m} + gamma) * z(xw)
    let wires = [
        proof_evals.wires_evals[num_wire_types - 1],
        proof_evals.perm_next_eval,
        challenges.gamma,
        proof_evals.perm_next_eval,
    ];
    let tmp = circuit.mul_add(&wires, &[F::one(), F::one()])?;

    // tmp = alpha *
    //  \prod_i=1..m-1 (w_{i} + beta * sigma_{i} + gamma)
    //  * (w_{m} + gamma) * z(xw)
    prod = circuit.mul(tmp, prod)?;

    // r_plonk
    let pi_eval = circuit.mul(*pi, evals[2])?;
    let wires = [pi_eval, prod, evals[2], challenges.alphas[1]];
    let non_lookup = circuit.gen_quad_poly(
        &wires,
        &[F::one(), -F::one(), F::zero(), F::zero()],
        &[F::zero(), -F::one()],
        F::zero(),
    )?;

    if let Some(lookup_evals) = lookup_evals {
        // We compute L_n(zeta) * (h_1 - h_1_next - alpha^2) - alpha * L_1(zeta)
        let wires = [
            lookup_evals.h_1_eval,
            lookup_evals.h_2_next_eval,
            challenges.alphas[1],
            circuit.zero(),
        ];
        let tmp = circuit.lc(&wires, &[F::one(), -F::one(), -F::one(), F::zero()])?;
        let term_one = circuit.mul_add(
            &[evals[3], tmp, evals[2], challenges.alphas[0]],
            &[F::one(), -F::one()],
        )?;

        // Now alpha^3 * (zeta  - domain_gen_inv) * prod_lookup_next * (gamma * ( 1 + beta) + h_1 + beta * h_1_next) * (gamma * (1 + beta) + beta * h_2_next)
        let mut init = circuit.mul_add(
            &[
                challenges.alphas[2],
                zeta_var,
                challenges.alphas[2],
                *gen_inv_var,
            ],
            &[F::one(), -F::one()],
        )?;
        init = circuit.mul(init, lookup_evals.prod_next_eval)?;
        let g_mul_one_b = circuit.mul_add(
            &[
                challenges.gamma,
                circuit.one(),
                challenges.gamma,
                challenges.beta,
            ],
            &[F::one(), F::one()],
        )?;
        let wires = [
            g_mul_one_b,
            lookup_evals.h_1_eval,
            challenges.beta,
            lookup_evals.h_1_next_eval,
        ];
        let tmp1 = circuit.gen_quad_poly(
            &wires,
            &[F::one(), F::one(), F::zero(), F::zero()],
            &[F::zero(), F::one()],
            F::zero(),
        )?;
        let tmp2 = circuit.mul_add(
            &[
                g_mul_one_b,
                circuit.one(),
                challenges.beta,
                lookup_evals.h_2_next_eval,
            ],
            &[F::one(), F::one()],
        )?;
        let mut term_two = circuit.mul(tmp1, tmp2)?;
        term_two = circuit.mul(term_two, init)?;

        let final_sum = circuit.sub(term_one, term_two)?;
        circuit.mul_add(
            &[final_sum, challenges.alphas[2], non_lookup, circuit.one()],
            &[F::one(), F::one()],
        )
    } else {
        Ok(non_lookup)
    }
}

/// Function used to evaluate polynomial that interpolates the evaluations fo polynomials used in plonk proofs over the native field.
pub(super) fn evaluate_lagrange_poly_helper_native<F>(
    circuit: &mut PlonkCircuit<F>,
    zeta_var: Variable,
    zeta_omega_var: Variable,
    inverse_var: Variable,
    poly_evals: &[Variable; 2],
    point: Variable,
) -> Result<Variable, CircuitError>
where
    F: PrimeField + RescueParameter,
{
    let term_one = circuit.mul_add(
        &[poly_evals[0], point, poly_evals[0], zeta_omega_var],
        &[F::one(), -F::one()],
    )?;
    let term_two = circuit.mul_add(
        &[poly_evals[1], point, poly_evals[1], zeta_var],
        &[F::one(), -F::one()],
    )?;
    circuit.mul_add(
        &[term_one, inverse_var, term_two, inverse_var],
        &[F::one(), -F::one()],
    )
}

/// Compute the bases and scalars in the batched polynomial commitment,
/// which is a generalization of `[D]1` specified in Sec 8.3, Verifier
/// algorithm step 9 of https://eprint.iacr.org/2019/953.pdf.
///
/// - input evals: zeta^n, zeta^n-1 and L_1(zeta), L_n(zeta)
///
/// Variables are returned in the order:
/// perm_coeff, sigma_coeff, q_scalars, quotient_scalars, lookup_prod_coeff, h_2_coeff
/// where the last two are zero if the proof was not an Ultraplonk proof.
#[allow(clippy::too_many_arguments)]
pub fn linearization_scalars_circuit_native<F>(
    circuit: &mut PlonkCircuit<F>,
    vk_k: &[F],
    challenges: &ChallengesVar,
    zeta: Variable,
    evals: &[Variable; 4],
    poly_evals: &ProofEvalsVarNative,
    lookup_evals: &Option<PlookupEvalsVarNative>,
    gen_inv: &F,
) -> Result<Vec<Variable>, CircuitError>
where
    F: PrimeField + RescueParameter,
{
    let wire_evals = &poly_evals.wires_evals;
    let sigma_evals = &poly_evals.wire_sigma_evals;
    // First we calculate the permutation poly coefficient
    let mut init = circuit.gen_quad_poly(
        &[challenges.beta, zeta, wire_evals[0], challenges.gamma],
        &[F::zero(), F::zero(), F::one(), F::one()],
        &[F::one(), F::zero()],
        F::zero(),
    )?;
    for (wire_eval, k) in wire_evals.iter().skip(1).zip(vk_k.iter().skip(1)) {
        let tmp = circuit.gen_quad_poly(
            &[challenges.beta, zeta, *wire_eval, challenges.gamma],
            &[F::zero(), F::zero(), F::one(), F::one()],
            &[*k, F::zero()],
            F::zero(),
        )?;
        init = circuit.mul(init, tmp)?;
    }

    let perm_coeff = circuit.mul_add(
        &[challenges.alphas[1], evals[2], challenges.alphas[0], init],
        &[F::one(), F::one()],
    )?;

    // Calculate the coefficient of the final sigma commitment
    let mut init = circuit.mul(challenges.alphas[0], poly_evals.perm_next_eval)?;
    init = circuit.mul(init, challenges.beta)?;

    let num_wire_types = wire_evals.len();
    for (wire_eval, sigma_eval) in wire_evals
        .iter()
        .take(num_wire_types - 1)
        .zip(sigma_evals.iter())
    {
        let tmp = circuit.gen_quad_poly(
            &[challenges.beta, *sigma_eval, *wire_eval, challenges.gamma],
            &[F::zero(), F::zero(), F::one(), F::one()],
            &[F::one(), F::zero()],
            F::zero(),
        )?;

        init = circuit.mul(init, tmp)?;
    }
    let sigma_coeff = circuit.sub(circuit.zero(), init)?;

    // Calculate the coefficients of the selector polynomial commitments
    let mut q_scalars = Vec::<Variable>::with_capacity(17);
    q_scalars.extend_from_slice(&wire_evals[0..4]);
    q_scalars.push(circuit.mul(wire_evals[0], wire_evals[1])?);
    q_scalars.push(circuit.mul(wire_evals[2], wire_evals[3])?);
    q_scalars.push(circuit.power_5_gen(wire_evals[0])?);
    q_scalars.push(circuit.power_5_gen(wire_evals[1])?);
    q_scalars.push(circuit.power_5_gen(wire_evals[2])?);
    q_scalars.push(circuit.power_5_gen(wire_evals[3])?);
    q_scalars.push(circuit.mul_constant(wire_evals[4], &(-F::one()))?);
    q_scalars.push(circuit.one());
    let tmp = circuit.mul(q_scalars[4], q_scalars[5])?;
    q_scalars.push(circuit.mul(tmp, wire_evals[4])?);
    let ad_wires = circuit.mul(wire_evals[0], wire_evals[3])?;
    let ac_wires = circuit.mul(wire_evals[0], wire_evals[2])?;
    let cd_wires = circuit.mul(wire_evals[2], wire_evals[3])?;
    let bc_wires = circuit.mul(wire_evals[1], wire_evals[2])?;
    let bd_wires = circuit.mul(wire_evals[1], wire_evals[3])?;
    let cc_wires = circuit.mul(wire_evals[2], wire_evals[2])?;
    let dd_wires = circuit.mul(wire_evals[3], wire_evals[3])?;
    let ab_wires = circuit.mul(wire_evals[0], wire_evals[1])?;
    q_scalars.push(circuit.mul_add(
        &[ad_wires, cd_wires, bc_wires, cd_wires],
        &[F::one(), F::one()],
    )?);
    q_scalars.push(circuit.lc(
        &[ac_wires, bd_wires, ad_wires, bc_wires],
        &[F::one(), F::one(), F::from(2u8), F::from(2u8)],
    )?);
    q_scalars.push(circuit.mul(cc_wires, dd_wires)?);
    q_scalars.push(circuit.mul_add(
        &[ab_wires, wire_evals[0], ab_wires, wire_evals[1]],
        &[F::one(), F::one()],
    )?);

    // Now calculate lookup scalars if they are present
    let (lookup_prod_coeff, h_2_coeff) = if let Some(lookup_evals) = lookup_evals {
        let g_mul_one_plus_b = circuit.mul_add(
            &[
                circuit.one(),
                challenges.gamma,
                challenges.beta,
                challenges.gamma,
            ],
            &[F::one(), F::one()],
        )?;
        // First we calculate the lookup_product poly coefficient.
        // To do this we calculate the merged lookup wire eal and the merged table eval and merged table next eval
        let mut mlw = circuit.mul_add(
            &[circuit.one(), wire_evals[1], challenges.tau, wire_evals[2]],
            &[F::one(), F::one()],
        )?;
        mlw = circuit.mul_add(
            &[challenges.tau, mlw, wire_evals[0], circuit.one()],
            &[F::one(), F::one()],
        )?;
        mlw = circuit.mul_add(
            &[
                challenges.tau,
                mlw,
                lookup_evals.q_dom_sep_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;
        mlw = circuit.mul(challenges.tau, mlw)?;
        mlw = circuit.mul_add(
            &[
                lookup_evals.q_lookup_eval,
                mlw,
                wire_evals[5],
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;

        let mut mlt = circuit.mul_add(
            &[circuit.one(), wire_evals[3], challenges.tau, wire_evals[4]],
            &[F::one(), F::one()],
        )?;
        mlt = circuit.mul_add(
            &[
                challenges.tau,
                mlt,
                lookup_evals.key_table_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;
        mlt = circuit.mul_add(
            &[
                challenges.tau,
                mlt,
                lookup_evals.table_dom_sep_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;
        mlt = circuit.mul(challenges.tau, mlt)?;
        mlt = circuit.mul_add(
            &[
                lookup_evals.q_lookup_eval,
                mlt,
                lookup_evals.range_table_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;

        let mut mltn = circuit.mul_add(
            &[
                circuit.one(),
                lookup_evals.w_3_next_eval,
                challenges.tau,
                lookup_evals.w_4_next_eval,
            ],
            &[F::one(), F::one()],
        )?;
        mltn = circuit.mul_add(
            &[
                challenges.tau,
                mltn,
                lookup_evals.key_table_next_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;
        mltn = circuit.mul_add(
            &[
                challenges.tau,
                mltn,
                lookup_evals.table_dom_sep_next_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;
        mltn = circuit.mul(challenges.tau, mltn)?;
        mltn = circuit.mul_add(
            &[
                lookup_evals.q_lookup_next_eval,
                mltn,
                lookup_evals.range_table_next_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;

        let reuse_temp = circuit.gen_quad_poly(
            &[challenges.alphas[2], zeta, circuit.zero(), circuit.zero()],
            &[-*gen_inv, F::zero(), F::zero(), F::zero()],
            &[F::one(), F::zero()],
            F::zero(),
        )?;

        let mut term_one = circuit.mul_add(
            &[reuse_temp, challenges.beta, reuse_temp, circuit.one()],
            &[F::one(), F::one()],
        )?;
        term_one = circuit.mul_add(
            &[term_one, challenges.gamma, term_one, mlw],
            &[F::one(), F::one()],
        )?;
        let tmp = circuit.gen_quad_poly(
            &[g_mul_one_plus_b, mlt, challenges.beta, mltn],
            &[F::one(), F::one(), F::zero(), F::zero()],
            &[F::zero(), F::one()],
            F::zero(),
        )?;
        term_one = circuit.mul(term_one, tmp)?;

        let term_two = circuit.mul_add(
            &[
                challenges.alphas[0],
                evals[2],
                challenges.alphas[1],
                evals[3],
            ],
            &[F::one(), F::one()],
        )?;

        let lookup_prod_coeff = circuit.mul_add(
            &[
                challenges.alphas[2],
                term_one,
                challenges.alphas[2],
                term_two,
            ],
            &[F::one(), F::one()],
        )?;

        // Now we calculate the coefficient of h_2
        term_one = circuit.mul_add(
            &[
                reuse_temp,
                lookup_evals.prod_next_eval,
                circuit.zero(),
                circuit.zero(),
            ],
            &[-F::one(), F::zero()],
        )?;
        let tmp = circuit.gen_quad_poly(
            &[
                g_mul_one_plus_b,
                lookup_evals.h_1_eval,
                challenges.beta,
                lookup_evals.h_1_next_eval,
            ],
            &[F::one(), F::one(), F::zero(), F::zero()],
            &[F::zero(), F::one()],
            F::zero(),
        )?;
        term_one = circuit.mul(term_one, tmp)?;

        let h_2_coeff = circuit.mul(challenges.alphas[2], term_one)?;
        (lookup_prod_coeff, h_2_coeff)
    } else {
        (circuit.zero(), circuit.zero())
    };

    // Now calculate the coefficients of the quotient commitments
    let zeta_square = circuit.mul(zeta, zeta)?;
    let vanish_eval = evals[1];
    let zeta_n_plus_two = circuit.mul(zeta_square, evals[0])?;
    let mut quotient_coeffs = Vec::with_capacity(5);
    let mut combiner = circuit.sub(circuit.zero(), vanish_eval)?;
    quotient_coeffs.push(combiner);
    for _ in 0..(num_wire_types - 1) {
        combiner = circuit.mul(combiner, zeta_n_plus_two)?;
        quotient_coeffs.push(combiner);
    }

    let result = [
        vec![perm_coeff, sigma_coeff],
        q_scalars,
        quotient_coeffs,
        vec![lookup_prod_coeff, h_2_coeff],
    ]
    .concat();

    Ok(result)
}

/// Same as `linearization_scalars_circuit_native` but takes `gen_inv` as a variable.
#[allow(clippy::too_many_arguments)]
pub fn linearization_scalars_circuit_native_base<F>(
    circuit: &mut PlonkCircuit<F>,
    vk_k: &[Variable],
    challenges: &ChallengesVar,
    zeta: Variable,
    evals: &[Variable; 4],
    poly_evals: &ProofEvalsVarNative,
    lookup_evals: &Option<PlookupEvalsVarNative>,
    gen_inv_var: &Variable,
) -> Result<Vec<Variable>, CircuitError>
where
    F: PrimeField + RescueParameter,
{
    let wire_evals = &poly_evals.wires_evals;
    let sigma_evals = &poly_evals.wire_sigma_evals;
    // First we calculate the permutation poly coefficient
    let mut init = circuit.gen_quad_poly(
        &[challenges.beta, zeta, wire_evals[0], challenges.gamma],
        &[F::zero(), F::zero(), F::one(), F::one()],
        &[F::one(), F::zero()],
        F::zero(),
    )?;
    for (wire_eval, k) in wire_evals.iter().skip(1).zip(vk_k.iter().skip(1)) {
        let tmp = circuit.mul(challenges.beta, *k)?;
        let tmp = circuit.gen_quad_poly(
            &[tmp, zeta, *wire_eval, challenges.gamma],
            &[F::zero(), F::zero(), F::one(), F::one()],
            &[F::one(), F::zero()],
            F::zero(),
        )?;
        init = circuit.mul(init, tmp)?;
    }

    let perm_coeff = circuit.mul_add(
        &[challenges.alphas[1], evals[2], challenges.alphas[0], init],
        &[F::one(), F::one()],
    )?;

    // Calculate the coefficient of the final sigma commitment
    let mut init = circuit.mul(challenges.alphas[0], poly_evals.perm_next_eval)?;
    init = circuit.mul(init, challenges.beta)?;

    let num_wire_types = wire_evals.len();
    for (wire_eval, sigma_eval) in wire_evals
        .iter()
        .take(num_wire_types - 1)
        .zip(sigma_evals.iter())
    {
        let tmp = circuit.gen_quad_poly(
            &[challenges.beta, *sigma_eval, *wire_eval, challenges.gamma],
            &[F::zero(), F::zero(), F::one(), F::one()],
            &[F::one(), F::zero()],
            F::zero(),
        )?;

        init = circuit.mul(init, tmp)?;
    }
    let sigma_coeff = circuit.sub(circuit.zero(), init)?;

    // Calculate the coefficients of the selector polynomial commitments
    let mut q_scalars = Vec::<Variable>::with_capacity(17);
    q_scalars.extend_from_slice(&wire_evals[0..4]);
    q_scalars.push(circuit.mul(wire_evals[0], wire_evals[1])?);
    q_scalars.push(circuit.mul(wire_evals[2], wire_evals[3])?);
    q_scalars.push(circuit.power_5_gen(wire_evals[0])?);
    q_scalars.push(circuit.power_5_gen(wire_evals[1])?);
    q_scalars.push(circuit.power_5_gen(wire_evals[2])?);
    q_scalars.push(circuit.power_5_gen(wire_evals[3])?);
    q_scalars.push(circuit.mul_constant(wire_evals[4], &(-F::one()))?);
    q_scalars.push(circuit.one());
    let tmp = circuit.mul(q_scalars[4], q_scalars[5])?;
    q_scalars.push(circuit.mul(tmp, wire_evals[4])?);
    let ad_wires = circuit.mul(wire_evals[0], wire_evals[3])?;
    let ac_wires = circuit.mul(wire_evals[0], wire_evals[2])?;
    let cd_wires = circuit.mul(wire_evals[2], wire_evals[3])?;
    let bc_wires = circuit.mul(wire_evals[1], wire_evals[2])?;
    let bd_wires = circuit.mul(wire_evals[1], wire_evals[3])?;
    let cc_wires = circuit.mul(wire_evals[2], wire_evals[2])?;
    let dd_wires = circuit.mul(wire_evals[3], wire_evals[3])?;
    let ab_wires = circuit.mul(wire_evals[0], wire_evals[1])?;
    q_scalars.push(circuit.mul_add(
        &[ad_wires, cd_wires, bc_wires, cd_wires],
        &[F::one(), F::one()],
    )?);
    q_scalars.push(circuit.lc(
        &[ac_wires, bd_wires, ad_wires, bc_wires],
        &[F::one(), F::one(), F::from(2u8), F::from(2u8)],
    )?);
    q_scalars.push(circuit.mul(cc_wires, dd_wires)?);
    q_scalars.push(circuit.mul_add(
        &[ab_wires, wire_evals[0], ab_wires, wire_evals[1]],
        &[F::one(), F::one()],
    )?);

    // Now calculate lookup scalars if they are present
    let (lookup_prod_coeff, h_2_coeff) = if let Some(lookup_evals) = lookup_evals {
        let g_mul_one_plus_b = circuit.mul_add(
            &[
                circuit.one(),
                challenges.gamma,
                challenges.beta,
                challenges.gamma,
            ],
            &[F::one(), F::one()],
        )?;
        // First we calculate the lookup_product poly coefficient.
        // To do this we calculate the merged lookup wire eal and the merged table eval and merged table next eval
        let mut mlw = circuit.mul_add(
            &[circuit.one(), wire_evals[1], challenges.tau, wire_evals[2]],
            &[F::one(), F::one()],
        )?;
        mlw = circuit.mul_add(
            &[challenges.tau, mlw, wire_evals[0], circuit.one()],
            &[F::one(), F::one()],
        )?;
        mlw = circuit.mul_add(
            &[
                challenges.tau,
                mlw,
                lookup_evals.q_dom_sep_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;
        mlw = circuit.mul(challenges.tau, mlw)?;
        mlw = circuit.mul_add(
            &[
                lookup_evals.q_lookup_eval,
                mlw,
                wire_evals[5],
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;

        let mut mlt = circuit.mul_add(
            &[circuit.one(), wire_evals[3], challenges.tau, wire_evals[4]],
            &[F::one(), F::one()],
        )?;
        mlt = circuit.mul_add(
            &[
                challenges.tau,
                mlt,
                lookup_evals.key_table_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;
        mlt = circuit.mul_add(
            &[
                challenges.tau,
                mlt,
                lookup_evals.table_dom_sep_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;
        mlt = circuit.mul(challenges.tau, mlt)?;
        mlt = circuit.mul_add(
            &[
                lookup_evals.q_lookup_eval,
                mlt,
                lookup_evals.range_table_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;

        let mut mltn = circuit.mul_add(
            &[
                circuit.one(),
                lookup_evals.w_3_next_eval,
                challenges.tau,
                lookup_evals.w_4_next_eval,
            ],
            &[F::one(), F::one()],
        )?;
        mltn = circuit.mul_add(
            &[
                challenges.tau,
                mltn,
                lookup_evals.key_table_next_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;
        mltn = circuit.mul_add(
            &[
                challenges.tau,
                mltn,
                lookup_evals.table_dom_sep_next_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;
        mltn = circuit.mul(challenges.tau, mltn)?;
        mltn = circuit.mul_add(
            &[
                lookup_evals.q_lookup_next_eval,
                mltn,
                lookup_evals.range_table_next_eval,
                circuit.one(),
            ],
            &[F::one(), F::one()],
        )?;

        let reuse_temp = circuit.mul_add(
            &[
                challenges.alphas[2],
                zeta,
                challenges.alphas[2],
                *gen_inv_var,
            ],
            &[F::one(), -F::one()],
        )?;
        let mut term_one = circuit.mul_add(
            &[reuse_temp, challenges.beta, reuse_temp, circuit.one()],
            &[F::one(), F::one()],
        )?;
        term_one = circuit.mul_add(
            &[term_one, challenges.gamma, term_one, mlw],
            &[F::one(), F::one()],
        )?;
        let tmp = circuit.gen_quad_poly(
            &[g_mul_one_plus_b, mlt, challenges.beta, mltn],
            &[F::one(), F::one(), F::zero(), F::zero()],
            &[F::zero(), F::one()],
            F::zero(),
        )?;
        term_one = circuit.mul(term_one, tmp)?;

        let term_two = circuit.mul_add(
            &[
                challenges.alphas[0],
                evals[2],
                challenges.alphas[1],
                evals[3],
            ],
            &[F::one(), F::one()],
        )?;

        let lookup_prod_coeff = circuit.mul_add(
            &[
                challenges.alphas[2],
                term_one,
                challenges.alphas[2],
                term_two,
            ],
            &[F::one(), F::one()],
        )?;

        // Now we calculate the coefficient of h_2
        term_one = circuit.mul_add(
            &[
                reuse_temp,
                lookup_evals.prod_next_eval,
                circuit.zero(),
                circuit.zero(),
            ],
            &[-F::one(), F::zero()],
        )?;
        let tmp = circuit.gen_quad_poly(
            &[
                g_mul_one_plus_b,
                lookup_evals.h_1_eval,
                challenges.beta,
                lookup_evals.h_1_next_eval,
            ],
            &[F::one(), F::one(), F::zero(), F::zero()],
            &[F::zero(), F::one()],
            F::zero(),
        )?;
        term_one = circuit.mul(term_one, tmp)?;

        let h_2_coeff = circuit.mul(challenges.alphas[2], term_one)?;
        (lookup_prod_coeff, h_2_coeff)
    } else {
        (circuit.zero(), circuit.zero())
    };

    // Now calculate the coefficients of the quotient commitments
    let zeta_square = circuit.mul(zeta, zeta)?;
    let vanish_eval = evals[1];
    let zeta_n_plus_two = circuit.mul(zeta_square, evals[0])?;
    let mut quotient_coeffs = Vec::with_capacity(5);
    let mut combiner = circuit.sub(circuit.zero(), vanish_eval)?;
    quotient_coeffs.push(combiner);
    for _ in 0..(num_wire_types - 1) {
        combiner = circuit.mul(combiner, zeta_n_plus_two)?;
        quotient_coeffs.push(combiner);
    }

    let result = [
        vec![perm_coeff, sigma_coeff],
        q_scalars,
        quotient_coeffs,
        vec![lookup_prod_coeff, h_2_coeff],
    ]
    .concat();

    Ok(result)
}

/// In-circuit evaluation of the public-input polynomial
/// π(X₁,…,X_μ) = P(X₁,…,X_t) · ∏_{k=t+1}^μ (1 − X_k)
///
/// - `mle_var`   – first 2^t coefficients of the MLE
/// - `point_var` – μ-dimensional verifier point (ρ₁,…,ρ_μ)
///
/// Gate-cost in the circuit:  O(2ᵗ + μ)
pub fn emulated_sparse_mle_evaluation_circuit<F, E>(
    circuit: &mut PlonkCircuit<F>,
    mle_var: &[EmulatedVariable<E>],
    point_var: &[EmulatedVariable<E>],
) -> Result<EmulatedVariable<E>, CircuitError>
where
    F: PrimeField,
    E: PrimeField + EmulationConfig<F>,
{
    let two_t = mle_var.len();
    if two_t == 0 || two_t & (two_t - 1) != 0 {
        return Err(CircuitError::ParameterError(
            "mle_var.len() must be a power of two".into(),
        ));
    }

    // t = log₂(2ᵗ)
    let t = two_t.trailing_zeros() as usize;
    let mu = point_var.len();
    if mu < t {
        return Err(CircuitError::ParameterError(format!(
            "point length μ = {} < t = {}",
            mu, t
        )));
    }

    let mut p_eval = emulated_mle_evaluation_circuit::<F, E>(circuit, mle_var, &point_var[..t])?;

    // Q(ρ_{t+1},…,ρ_μ) = ∏ (1 − ρ_k)
    for rho_k in &point_var[t..] {
        let one_minus_rho_k = circuit.emulated_sub(&circuit.emulated_one(), rho_k)?;
        p_eval = circuit.emulated_mul(&p_eval, &one_minus_rho_k)?;
    }

    Ok(p_eval)
}

/// Circuit to evaluate a polynomial oracle at a given point,
/// returns the variable of the evaluation.
pub fn emulated_mle_evaluation_circuit<F, E>(
    circuit: &mut PlonkCircuit<F>,
    mle_var: &[EmulatedVariable<E>],
    point_var: &[EmulatedVariable<E>],
) -> Result<EmulatedVariable<E>, CircuitError>
where
    F: PrimeField,
    E: PrimeField + EmulationConfig<F>,
{
    let nv = point_var.len();

    if mle_var.len().ilog2() as usize != nv {
        return Err(CircuitError::ParameterError(
            "num_vars != point.len()".to_string(),
        ));
    }
    let mut mut_evals_var = mle_var.to_vec();
    // Mimick `evaluate` function of `DenseMultilinearExtension` which calls the `fix_variables` function.
    for i in 1..nv + 1 {
        for b in 0..(1 << (nv - i)) {
            let right_sub_left_var =
                circuit.emulated_sub(&mut_evals_var[(b << 1) + 1], &mut_evals_var[b << 1])?;
            let tmp = circuit.emulated_mul(&point_var[i - 1], &right_sub_left_var)?;
            mut_evals_var[b] = circuit.emulated_add(&mut_evals_var[b << 1], &tmp)?;
        }
    }
    Ok(mut_evals_var[0].clone())
}

/// Circuit used to perform eq_x_r_eval over a native field.
pub(crate) fn eq_x_r_eval_circuit<F>(
    circuit: &mut PlonkCircuit<F>,
    x: &[Variable],
    r: &[Variable],
) -> Result<Variable, CircuitError>
where
    F: PrimeField,
{
    if x.len() != r.len() {
        return Err(CircuitError::ParameterError(format!(
            "x.len(): {} != r.len(): {}",
            x.len(),
            r.len()
        )));
    }
    let mut eq_eval = circuit.gen_quad_poly(
        &[x[0], r[0], circuit.zero(), circuit.zero()],
        &[-F::one(), -F::one(), F::zero(), F::zero()],
        &[F::from(2u8), F::zero()],
        F::one(),
    )?;
    for (x_var, r_var) in x.iter().zip(r.iter()).skip(1) {
        let tmp = circuit.gen_quad_poly(
            &[*x_var, *r_var, circuit.zero(), circuit.zero()],
            &[-F::one(), -F::one(), F::zero(), F::zero()],
            &[F::from(2u8), F::zero()],
            F::one(),
        )?;
        eq_eval = circuit.mul(eq_eval, tmp)?;
    }

    Ok(eq_eval)
}

/// Circuit used to perform eq_x_r_eval over a native field.
pub(crate) fn emulated_eq_x_r_eval_circuit<F, E>(
    circuit: &mut PlonkCircuit<F>,
    x: &[EmulatedVariable<E>],

    r: &[EmulatedVariable<E>],
) -> Result<EmulatedVariable<E>, CircuitError>
where
    F: PrimeField,
    E: PrimeField + EmulationConfig<F>,
{
    if x.len() != r.len() {
        return Err(CircuitError::ParameterError(format!(
            "x.len(): {} != r.len(): {}",
            x.len(),
            r.len()
        )));
    }
    let r_var = r.first().ok_or(CircuitError::IndexError)?;
    let x_var = x.first().ok_or(CircuitError::IndexError)?;
    let tmp1 = circuit.emulated_sub_constant(r_var, E::one())?;
    let tmp2 = circuit.emulated_sub_constant(x_var, E::one())?;
    let tmp3 = circuit.emulated_mul(&tmp1, &tmp2)?;
    let tmp4 = circuit.emulated_mul(r_var, x_var)?;
    let mut eq_eval = circuit.emulated_add(&tmp3, &tmp4)?;

    for (x_var, r_var) in x.iter().skip(1).zip(r.iter().skip(1)) {
        let tmp1 = circuit.emulated_sub_constant(r_var, E::one())?;
        let tmp2 = circuit.emulated_sub_constant(x_var, E::one())?;
        let tmp3 = circuit.emulated_mul(&tmp1, &tmp2)?;
        let tmp4 = circuit.emulated_mul(r_var, x_var)?;
        let tmp5 = circuit.emulated_add(&tmp3, &tmp4)?;
        eq_eval = circuit.emulated_mul(&eq_eval, &tmp5)?;
    }

    Ok(eq_eval)
}

/// This function takes in a `Variable` `zeta_var`, a `domain_size_var` and a `max_domain_size` and returns
/// a `Variable` representing `zeta^n`, where `domain_size_var` represents `2^n`.
pub(crate) fn zeta_to_n<F>(
    circuit: &mut PlonkCircuit<F>,
    zeta_var: Variable,
    domain_size_var: Variable,
    max_domain_size: usize,
) -> Result<Variable, CircuitError>
where
    F: PrimeField,
{
    let domain_size: BigUint = circuit.witness(domain_size_var)?.into_bigint().into();

    let log_domain = domain_size.bits() - 1;
    let max_log_size = max_domain_size.ilog2() as u64;
    if log_domain > max_log_size {
        return Err(CircuitError::ParameterError(format!(
            "domain size {} is larger than max domain size {}",
            domain_size, max_domain_size
        )));
    }
    let two_pows = (0..max_log_size + 1)
        .map(|i| F::from(1u64 << i))
        .collect::<Vec<F>>();

    let bool_vars = (0..max_log_size + 1)
        .map(|i| {
            let bool_var = circuit.create_boolean_variable(i == log_domain)?;
            Ok(bool_var.0)
        })
        .collect::<Result<Vec<Variable>, CircuitError>>()?;

    circuit.lin_comb_gate(&two_pows, &F::zero(), &bool_vars, &domain_size_var)?;

    let zeta_pow_vars = iter::successors(Some(Ok(zeta_var)), |prev| match prev {
        Ok(var) => Some(circuit.mul(*var, *var)),
        _ => None,
    })
    .take(max_log_size as usize + 1)
    .collect::<Result<Vec<Variable>, CircuitError>>()?;

    let mut zeta_n_var = circuit.mul_add(
        &[
            bool_vars[0],
            zeta_pow_vars[0],
            bool_vars[1],
            zeta_pow_vars[1],
        ],
        &[F::one(), F::one()],
    )?;
    for (bool_var, zeta_pow_var) in bool_vars.iter().zip(zeta_pow_vars.iter()).skip(2) {
        zeta_n_var = circuit.mul_add(
            &[*bool_var, *zeta_pow_var, zeta_n_var, circuit.one()],
            &[F::one(), F::one()],
        )?;
    }

    Ok(zeta_n_var)
}

#[cfg(test)]
mod test {

    use super::*;

    use ark_bn254::{g1::Config as BnConfig, Fr as Fr254};

    use ark_poly::{
        DenseMultilinearExtension, DenseUVPolynomial, MultilinearExtension, Polynomial,
        Radix2EvaluationDomain,
    };
    use ark_std::{rand::Rng, One, UniformRand};

    use jf_primitives::rescue::RescueParameter;
    use jf_relation::gadgets::ecc::HasTEForm;
    use jf_utils::test_rng;

    const RANGE_BIT_LEN_FOR_TEST: usize = 16;

    #[test]
    fn test_evaluate_poly() {
        test_evaluate_poly_helper::<BnConfig>();
    }

    fn test_evaluate_poly_helper<E: HasTEForm>()
    where
        E::ScalarField: EmulationConfig<E::BaseField> + PrimeField + RescueParameter,
        E::BaseField: RescueParameter + PrimeField + EmulationConfig<E::ScalarField>,
    {
        let mut rng = test_rng();

        let mut circuit = PlonkCircuit::<E::BaseField>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);
        let zeta = E::ScalarField::rand(&mut rng);
        let zeta_var = circuit.create_emulated_variable(zeta).unwrap();

        for domain_size in [64, 128, 256, 512, 1024] {
            // compute the result in the clear
            let domain = Radix2EvaluationDomain::<E::ScalarField>::new(domain_size).unwrap();
            let vanish_eval = domain.evaluate_vanishing_polynomial(zeta);
            let zeta_n = vanish_eval + E::ScalarField::one();
            let divisor = E::ScalarField::from(domain_size as u32) * (zeta - E::ScalarField::one());
            let lagrange_1_eval = vanish_eval / divisor;

            let eval_results =
                evaluate_poly_helper::<E, _>(&mut circuit, &zeta_var, domain_size).unwrap();

            // check the correctness
            assert_eq!(zeta_n, circuit.emulated_witness(&eval_results[0]).unwrap(),);

            assert_eq!(
                vanish_eval,
                circuit.emulated_witness(&eval_results[1]).unwrap(),
            );

            assert_eq!(
                lagrange_1_eval,
                circuit.emulated_witness(&eval_results[2]).unwrap(),
            );
        }
    }

    #[test]
    fn test_pi_poly_native_vs_emulated_correctness() {
        evaluate_pi_poly_circuit_native_emulated_helper::<BnConfig>();
    }

    /// Test that native and emulated versions agree and match clear evaluation
    fn evaluate_pi_poly_circuit_native_emulated_helper<E: HasTEForm>()
    where
        E::ScalarField: EmulationConfig<E::BaseField> + PrimeField + RescueParameter,
        E::BaseField: RescueParameter + PrimeField + EmulationConfig<E::ScalarField>,
    {
        let mut rng = test_rng();

        for domain_size in [256, 512, 1024] {
            let degrees: Vec<usize> = (0..3).map(|_| rng.gen_range(0..domain_size)).collect();
            for degree in degrees {
                // setup native
                let mut nat_circ =
                    PlonkCircuit::<E::ScalarField>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);
                let zeta = E::ScalarField::rand(&mut rng);
                let zeta_var = nat_circ.create_variable(zeta).unwrap();
                let pub_inputs: Vec<E::ScalarField> = (0..=degree)
                    .map(|_| E::ScalarField::rand(&mut rng))
                    .collect();
                let pub_vars: Vec<_> = pub_inputs
                    .iter()
                    .map(|&v| nat_circ.create_variable(v).unwrap())
                    .collect();

                let domain = Radix2EvaluationDomain::<E::ScalarField>::new(domain_size).unwrap();

                let native_out = evaluate_pi_poly_circuit_native(
                    &mut nat_circ,
                    domain_size,
                    &pub_vars,
                    &zeta_var,
                )
                .unwrap();
                let native_eval = nat_circ.witness(native_out).unwrap();

                // setup emulated
                let mut emc_circ =
                    PlonkCircuit::<E::BaseField>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);
                let zeta_em_var = emc_circ.create_emulated_variable(zeta).unwrap();
                let pub_em_vars: Vec<_> = pub_inputs
                    .iter()
                    .map(|&v| emc_circ.create_emulated_variable(v).unwrap())
                    .collect();

                let em_out = evaluate_pi_poly_circuit_emulated::<E, E::BaseField>(
                    &mut emc_circ,
                    domain_size,
                    &pub_em_vars,
                    &zeta_em_var,
                )
                .unwrap();
                let em_eval = emc_circ.emulated_witness(&em_out).unwrap();

                // compare against clear evaluation
                let mut coeffs = pub_inputs.to_vec();
                domain.ifft_in_place(&mut coeffs);
                let clear_eval =
                    ark_poly::univariate::DensePolynomial::from_coefficients_vec(coeffs)
                        .evaluate(&zeta);

                assert_eq!(native_eval, clear_eval, "native vs clear mismatch");
                assert_eq!(em_eval, clear_eval, "emulated vs clear mismatch");
                assert_eq!(native_eval, em_eval, "native vs emulated mismatch");
            }
        }
    }

    #[test]
    fn test_evaluate_mle_native_emulated() {
        test_evaluate_mle_native_emulated_helper::<BnConfig>();
    }

    fn test_evaluate_mle_native_emulated_helper<E: HasTEForm>()
    where
        E::ScalarField: EmulationConfig<E::BaseField> + PrimeField + RescueParameter,
        E::BaseField: RescueParameter + PrimeField + EmulationConfig<E::ScalarField>,
    {
        let mut rng = test_rng();
        for n_vars in [3, 6, 10] {
            let domain_size = 1 << n_vars;
            let degrees: Vec<usize> = (0..3).map(|_| rng.gen_range(1..domain_size)).collect();

            for degree in degrees {
                let mut circuit =
                    PlonkCircuit::<E::BaseField>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);

                let points: Vec<E::ScalarField> = (0..n_vars)
                    .map(|_| E::ScalarField::rand(&mut rng))
                    .collect();
                let public_inputs: Vec<E::ScalarField> = (0..=degree)
                    .map(|_| E::ScalarField::rand(&mut rng))
                    .collect();

                let mut padded_public_inputs = public_inputs.clone();
                padded_public_inputs.resize(domain_size, E::ScalarField::from(0u8));

                // native evaluation, in the clear
                let poly = DenseMultilinearExtension::<E::ScalarField>::from_evaluations_vec(
                    n_vars,
                    padded_public_inputs.clone(),
                );

                let clear_eval = poly.evaluate(&points).unwrap();

                // emulated evaluation
                let mut mle_var: Vec<EmulatedVariable<_>> = public_inputs
                    .iter()
                    .map(|val| circuit.create_emulated_variable(*val).unwrap())
                    .collect();
                mle_var.resize(domain_size, circuit.emulated_zero());

                let points_var: Vec<EmulatedVariable<E::ScalarField>> = points
                    .iter()
                    .map(|coord| circuit.create_emulated_variable(*coord).unwrap())
                    .collect();

                let eval =
                    emulated_mle_evaluation_circuit(&mut circuit, &mle_var, &points_var).unwrap();

                // sparse MLE evaluation
                let mut mle_var: Vec<EmulatedVariable<_>> = public_inputs
                    .iter()
                    .map(|val| circuit.create_emulated_variable(*val).unwrap())
                    .collect();

                mle_var.resize(1 << (degree.ilog2() + 1), circuit.emulated_zero());

                let points: Vec<EmulatedVariable<E::ScalarField>> = points
                    .iter()
                    .map(|coord| circuit.create_emulated_variable(*coord).unwrap())
                    .collect();
                let sparse_eval =
                    emulated_sparse_mle_evaluation_circuit(&mut circuit, &mle_var, &points)
                        .unwrap();

                // check MLE evaluation vs clear evaluation
                assert_eq!(
                    circuit.emulated_witness(&eval).unwrap(),
                    clear_eval,
                    "MLE evaluation does not match clear evaluation"
                );

                // check MLE evaluation vs sparse MLE evaluation
                assert_eq!(
                    circuit.emulated_witness(&eval).unwrap(),
                    circuit.emulated_witness(&sparse_eval).unwrap(),
                    "MLE evaluation does not match sparse MLE evaluation"
                );
            }
        }
    }

    #[test]
    fn test_zeta_to_n() -> Result<(), CircuitError> {
        test_zeta_to_n_helper::<Fr254>()
    }

    fn test_zeta_to_n_helper<F: PrimeField>() -> Result<(), CircuitError> {
        let mut rng = test_rng();
        let max_domain_size = 1024;
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        for n in 6..11 {
            let domain_size = 1_u32 << n;
            let domain_size_var = circuit.create_variable(F::from(domain_size))?;
            let zeta = F::rand(&mut rng);
            let zeta_var = circuit.create_variable(zeta)?;

            let zeta_n_var = zeta_to_n(&mut circuit, zeta_var, domain_size_var, max_domain_size)?;
            let zeta_n = circuit.witness(zeta_n_var)?;

            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
            assert_eq!(zeta_n, zeta.pow([n]));
        }
        Ok(())
    }
}
