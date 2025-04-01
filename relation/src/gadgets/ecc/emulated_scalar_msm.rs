// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! This module implements multi-scalar-multiplication circuits.

use super::{
    create_point_lookup_gates_custom_keys, decompose_for_msm, decompose_scalar_var_signed,
    HasTEForm, Point, PointVariable,
};
use crate::{
    errors::CircuitError,
    gadgets::{EmulatedVariable, EmulationConfig},
    BoolVar, Circuit, PlonkCircuit, Variable,
};
use ark_ec::{
    short_weierstrass::{Affine as SWAffine, Projective, SWCurveConfig as SWConfig},
    CurveConfig, CurveGroup,
};

use num_bigint::BigUint;
use num_traits::sign;
use rayon::prelude::*;

use ark_ff::{BigInteger, Field, One, PrimeField, Zero};
use ark_std::{cfg_iter, format, vec, vec::Vec};
use jf_utils::fq_to_fr;

/// Compute the emulated scalar multi-scalar-multiplications in circuit.
pub trait EmulMultiScalarMultiplicationCircuit<F, P>
where
    F: PrimeField,
    P: SWConfig<BaseField = F>,
    <P as CurveConfig>::ScalarField: PrimeField + EmulationConfig<F>,
{
    /// Compute the multi-scalar-multiplications.
    /// Use pippenger when the circuit supports lookup;
    /// Use naive method otherwise.
    /// Return error if the number bases does not match the number of scalars.
    fn msm(
        &mut self,
        bases: &[PointVariable],
        scalars: &[EmulatedVariable<P::ScalarField>],
    ) -> Result<PointVariable, CircuitError>;

    /// Compute the multi-scalar-multiplications where each scalar has at most
    /// `scalar_bit_length` bits.
    fn msm_with_var_scalar_length(
        &mut self,
        bases: &[PointVariable],
        scalars: &[EmulatedVariable<P::ScalarField>],
        scalar_bit_length: usize,
    ) -> Result<PointVariable, CircuitError>;

    /// Compute the multi-scalar-multiplications with fixed bases.
    fn fixed_base_msm(
        &mut self,
        bases: &[SWAffine<P>],
        scalars: &[EmulatedVariable<P::ScalarField>],
    ) -> Result<PointVariable, CircuitError>;
}

impl<F, P> EmulMultiScalarMultiplicationCircuit<F, P> for PlonkCircuit<F>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
    <P as CurveConfig>::ScalarField: PrimeField + EmulationConfig<F>,
{
    fn msm(
        &mut self,
        bases: &[PointVariable],
        scalars: &[EmulatedVariable<P::ScalarField>],
    ) -> Result<PointVariable, CircuitError> {
        let scalar_bit_length = <P as CurveConfig>::ScalarField::MODULUS_BIT_SIZE as usize;
        EmulMultiScalarMultiplicationCircuit::<F, P>::msm_with_var_scalar_length(
            self,
            bases,
            scalars,
            scalar_bit_length,
        )
    }

    fn msm_with_var_scalar_length(
        &mut self,
        bases: &[PointVariable],
        scalars: &[EmulatedVariable<P::ScalarField>],
        scalar_bit_length: usize,
    ) -> Result<PointVariable, CircuitError> {
        if bases.len() != scalars.len() {
            return Err(CircuitError::ParameterError(format!(
                "bases length ({}) does not match scalar length ({})",
                bases.len(),
                scalars.len()
            )));
        }

        if self.support_lookup() {
            msm_pippenger::<F, P>(self, bases, scalars, scalar_bit_length)
        } else {
            msm_naive::<F, P>(self, bases, scalars, scalar_bit_length)
        }
    }

    fn fixed_base_msm(
        &mut self,
        bases: &[SWAffine<P>],
        scalars: &[EmulatedVariable<P::ScalarField>],
    ) -> Result<PointVariable, CircuitError> {
        if bases.len() != scalars.len() {
            return Err(CircuitError::ParameterError(format!(
                "bases length ({}) does not match scalar length ({})",
                bases.len(),
                scalars.len()
            )));
        }
        let scalar_bit_length = <P as CurveConfig>::ScalarField::MODULUS_BIT_SIZE as usize;
        msm_pippenger_fixed::<F, P>(self, bases, scalars, scalar_bit_length)
    }
}

// A naive way to implement msm by computing them individually.
// Used for double checking the correctness; also as a fall-back solution
// to Pippenger.
//
// Some typical result on BW6-761 curve is shown below (i.e. the circuit
// simulates BLS12-377 curve operations). More results are available in the test
// function.
//
// number of basis: 1
// #variables: 1867
// #constraints: 1865
//
// number of basis: 2
// #variables: 3734
// #constraints: 3730
//
// number of basis: 4
// #variables: 7468
// #constraints: 7460
//
// number of basis: 8
// #variables: 14936
// #constraints: 14920
//
// number of basis: 16
// #variables: 29872
// #constraints: 29840
//
// number of basis: 32
// #variables: 59744
// #constraints: 59680
//
// number of basis: 64
// #variables: 119488
// #constraints: 119360
//
// number of basis: 128
// #variables: 238976
// #constraints: 238720
fn msm_naive<F, P>(
    circuit: &mut PlonkCircuit<F>,
    bases: &[PointVariable],
    scalars: &[EmulatedVariable<P::ScalarField>],
    scalar_bit_length: usize,
) -> Result<PointVariable, CircuitError>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
    <P as CurveConfig>::ScalarField: PrimeField + EmulationConfig<F>,
{
    for scalar in scalars.iter() {
        circuit.check_vars_bound(&scalar.0[..])?;
    }

    for base in bases.iter() {
        circuit.check_point_var_bound(base)?;
    }

    let scalar_0_bits_le = circuit.emulated_unpack(&scalars[0], scalar_bit_length)?;
    let mut res = circuit.variable_base_binary_scalar_mul::<P>(&scalar_0_bits_le, &bases[0])?;

    for (base, scalar) in bases.iter().zip(scalars.iter()).skip(1) {
        let scalar_bits_le = circuit.emulated_unpack(scalar, scalar_bit_length)?;
        let tmp = circuit.variable_base_binary_scalar_mul::<P>(&scalar_bits_le, base)?;
        res = circuit.ecc_add::<P>(&res, &tmp)?;
    }

    Ok(res)
}

// A variant of Pippenger MSM.
//
// Note, it is assumed that none of the bases is the neutral element.
//
// Some typical result on BW6-761 curve is shown below (i.e. the circuit
// simulates BLS12-377 curve operations). More results are available in the test
// function.
//
// number of basis: 1
// #variables: 887
// #constraints: 783
//
// number of basis: 2
// #variables: 1272
// #constraints: 1064
//
// number of basis: 4
// #variables: 2042
// #constraints: 1626
//
// number of basis: 8
// #variables: 3582
// #constraints: 2750
//
// number of basis: 16
// #variables: 6662
// #constraints: 4998
//
// number of basis: 32
// #variables: 12822
// #constraints: 9494
//
// number of basis: 64
// #variables: 25142
// #constraints: 18486
//
// number of basis: 128
// #variables: 49782
// #constraints: 36470
fn msm_pippenger<F, P>(
    circuit: &mut PlonkCircuit<F>,
    bases: &[PointVariable],
    scalars: &[EmulatedVariable<P::ScalarField>],
    scalar_bit_length: usize,
) -> Result<PointVariable, CircuitError>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
    <P as CurveConfig>::ScalarField: PrimeField + EmulationConfig<F>,
{
    // ================================================
    // check inputs
    // ================================================
    for (scalar, base) in scalars.iter().zip(bases.iter()) {
        circuit.check_vars_bound(&scalar.0[..])?;
        circuit.check_point_var_bound(base)?;
    }

    if P::has_glv() {
        // ================================================
        // Find scalar length for GLV method
        // ================================================

        let scalar_bit_length = scalar_bit_length.next_power_of_two() / 2;

        // ================================================
        // set up parameters
        // ================================================

        let c = if scalar_bit_length < 32 {
            3
        } else {
            ln_without_floats(scalar_bit_length)
        };
        // ================================================
        // compute lookup tables and window sums
        // ================================================

        // Each window is of size `c`.
        // We divide up the bits 0..scalar_bit_length into windows of size `c`, and
        // in parallel process each such window.
        let mut window_sums = Vec::new();
        let mut skews_one = vec![];
        let mut skews_two = vec![];
        let mut neg_bases_one = vec![];
        let mut neg_bases_two = vec![];
        for (base_var, scalar_var) in bases.iter().zip(scalars.iter()) {
            let (scalar_one, scalar_two, sign_var) =
                circuit.scalar_decomposition_gate::<P>(scalar_var)?;

            // decompose scalar into c-bit scalars
            let (decomposed_scalar_one_vars, skew_one) =
                decompose_scalar_var_signed::<P, F>(circuit, scalar_one, c, scalar_bit_length)?;

            let (decomposed_scalar_two_vars, skew_two) =
                decompose_scalar_var_signed::<P, F>(circuit, scalar_two, c, scalar_bit_length)?;

            // We need to contrain the last scalar in each decomposition to be positive and
            // ensure that we don't exceed the `scalar_bit_length`. All the other scalars
            // will be constrained appropriately by the lookup table.
            if scalar_bit_length % c != 0 {
                for decomposed_scalar_vars in
                    &[&decomposed_scalar_one_vars, &decomposed_scalar_two_vars]
                {
                    let last_scalar_var = decomposed_scalar_vars.last().unwrap();
                    circuit.range_gate_with_lookup(*last_scalar_var, scalar_bit_length % c)?;
                }
            }

            // We need the inverse of both the base and the endomorphism applied to the base.
            let neg_base = circuit.inverse_point(base_var)?;
            let temp_endo_point = circuit.endomorphism_circuit::<P>(base_var)?;
            let temp_neg_endo_point = circuit.endomorphism_circuit::<P>(&neg_base)?;
            // If `sign_var` represents a minus, we swap the roles of endo_point and neg_endo_point.
            let endo_point = circuit.binary_point_vars_select(
                sign_var,
                &temp_neg_endo_point,
                &temp_endo_point,
            )?;
            let neg_endo_point = circuit.binary_point_vars_select(
                sign_var,
                &temp_endo_point,
                &temp_neg_endo_point,
            )?;
            // We add `neg_base` and `neg_end_point` to the our vector of negated bases.
            neg_bases_one.push(neg_base);
            neg_bases_two.push(neg_endo_point);

            // create point table [0 * base, 1 * base, ..., (2^c-1) * base]
            // let mut table_point_vars_one = vec![point_zero_var, *base_var];
            let mut table_point_vars_one = vec![*base_var];
            let mut table_point_vars_two = vec![endo_point];
            let double_base_var = circuit.ecc_add_no_neutral::<P>(base_var, base_var)?;

            for i in 1usize..(1 << (c - 1)) {
                let base_point_var = circuit
                    .ecc_add_no_neutral::<P>(&table_point_vars_one[i - 1], &double_base_var)?;
                let endo_point_var = circuit.endomorphism_circuit::<P>(&base_point_var)?;
                table_point_vars_one.push(base_point_var);
                table_point_vars_two.push(endo_point_var);
            }

            let mut neg_points_one = vec![neg_base];
            let mut neg_points_two = vec![neg_endo_point];
            if !P::has_te_form() {
                for point in table_point_vars_one.iter().skip(1) {
                    let point_x = point.get_x();
                    let point_y = point.get_y();
                    let new_y = circuit.sub(circuit.zero(), point_y)?;
                    let neg_point = PointVariable::SW(point_x, new_y);
                    let neg_endo_point = circuit.endomorphism_circuit::<P>(&neg_point)?;
                    neg_points_one.push(neg_point);
                    neg_points_two.push(neg_endo_point);
                }
                table_point_vars_one.extend_from_slice(&neg_points_one);
                table_point_vars_two.extend_from_slice(&neg_points_two);
            } else {
                for point in table_point_vars_one.iter().skip(1) {
                    let neg_point = circuit.inverse_point(point)?;
                    let neg_endo_point = circuit.endomorphism_circuit::<P>(&neg_point)?;
                    neg_points_one.push(neg_point);
                    neg_points_two.push(neg_endo_point);
                }
                table_point_vars_one.extend_from_slice(&neg_points_one);
                table_point_vars_two.extend_from_slice(&neg_points_two);
            }

            // create lookup point variables
            let mut lookup_point_vars_one = Vec::new();
            for &scalar_var in decomposed_scalar_one_vars.iter() {
                let lookup_point =
                    compute_scalar_mul_value_signed::<F, P>(circuit, scalar_var, base_var)?;
                let lookup_point_var = circuit.create_point_variable(&lookup_point)?;
                lookup_point_vars_one.push(lookup_point_var);
            }

            // create lookup point variables
            let mut lookup_point_vars_two = Vec::new();
            for &scalar_var in decomposed_scalar_two_vars.iter() {
                let lookup_point =
                    compute_scalar_mul_value_signed::<F, P>(circuit, scalar_var, &endo_point)?;
                let lookup_point_var = circuit.create_point_variable(&lookup_point)?;
                lookup_point_vars_two.push(lookup_point_var);
            }

            create_point_lookup_gates_custom_keys::<P, F>(
                circuit,
                &table_point_vars_one,
                &decomposed_scalar_one_vars,
                &lookup_point_vars_one,
            )?;

            create_point_lookup_gates_custom_keys::<P, F>(
                circuit,
                &table_point_vars_two,
                &decomposed_scalar_two_vars,
                &lookup_point_vars_two,
            )?;

            let lookup_point_vars = lookup_point_vars_one
                .iter()
                .zip(lookup_point_vars_two.iter())
                .map(|(a, b)| circuit.ecc_add_no_neutral::<P>(a, b))
                .collect::<Result<Vec<_>, _>>()?;
            // update window sums
            if window_sums.is_empty() {
                window_sums = lookup_point_vars;
            } else {
                for (window_sum_mut, lookup_point_var) in
                    window_sums.iter_mut().zip(lookup_point_vars.iter())
                {
                    *window_sum_mut =
                        circuit.ecc_add_no_neutral::<P>(window_sum_mut, lookup_point_var)?;
                }
            }
            skews_one.push(skew_one);
            skews_two.push(skew_two);
        }

        // ================================================
        // performing additions
        // ================================================
        // We store the sum for the lowest window.
        let lowest = window_sums.first().unwrap();
        let mut last = *window_sums.last().unwrap();

        for _ in 0..c {
            last = circuit.ecc_add_no_neutral::<P>(&last, &last)?;
        }
        // We're traversing windows from high to low.
        let b = &window_sums[1..]
            .iter()
            .rev()
            .skip(1)
            .fold(last, |mut total, sum_i| {
                // total += sum_i
                total = circuit.ecc_add_no_neutral::<P>(&total, sum_i).unwrap();
                for _ in 0..c {
                    // double
                    total = circuit.ecc_add_no_neutral::<P>(&total, &total).unwrap();
                }
                total
            });

        let mut final_point = circuit.ecc_add_no_neutral::<P>(lowest, b)?;
        for (skew, base) in skews_one.into_iter().zip(neg_bases_one.iter()) {
            let point = circuit.ecc_add_no_neutral::<P>(&final_point, base)?;
            final_point = circuit.binary_point_vars_select(skew, &final_point, &point)?;
        }

        for (skew, base) in skews_two.into_iter().zip(neg_bases_two.iter()) {
            let point = circuit.ecc_add_no_neutral::<P>(&final_point, base)?;
            final_point = circuit.binary_point_vars_select(skew, &final_point, &point)?;
        }
        Ok(final_point)
    } else {
        // ================================================
        // set up parameters
        // ================================================
        let c = if scalar_bit_length < 32 {
            3
        } else {
            ln_without_floats(scalar_bit_length)
        };

        // ================================================
        // compute lookup tables and window sums
        // ================================================
        let point_zero_var = circuit.neutral_point_variable::<P>();
        // Each window is of size `c`.
        // We divide up the bits 0..scalar_bit_length into windows of size `c`, and
        // in parallel process each such window.
        let mut window_sums = Vec::new();
        for (base_var, scalar_var) in bases.iter().zip(scalars.iter()) {
            // decompose scalar into c-bit scalars
            let decomposed_scalar_vars =
                decompose_emulated_scalar_var(circuit, scalar_var, c, scalar_bit_length)?;

            // create point table [0 * base, 1 * base, ..., (2^c-1) * base]
            let mut table_point_vars = vec![point_zero_var; 1 << c];
            table_point_vars[1] = *base_var;

            for i in 2usize..(1 << c) {
                let point_var =
                    circuit.ecc_add_no_neutral::<P>(base_var, &table_point_vars[i - 1])?;
                table_point_vars[i] = point_var;
            }

            // create lookup point variables
            let mut lookup_point_vars = Vec::new();
            for &scalar_var in decomposed_scalar_vars.iter() {
                let lookup_point = compute_scalar_mul_value::<F, P>(circuit, scalar_var, base_var)?;
                let lookup_point_var = circuit.create_point_variable(&lookup_point)?;
                lookup_point_vars.push(lookup_point_var);
            }

            create_point_lookup_gates(
                circuit,
                &table_point_vars,
                &decomposed_scalar_vars,
                &lookup_point_vars,
            )?;

            // update window sums
            if window_sums.is_empty() {
                window_sums = lookup_point_vars;
            } else {
                for (window_sum_mut, lookup_point_var) in
                    window_sums.iter_mut().zip(lookup_point_vars.iter())
                {
                    *window_sum_mut = circuit.ecc_add::<P>(window_sum_mut, lookup_point_var)?;
                }
            }
        }

        // ================================================
        // performing additions
        // ================================================
        // We store the sum for the lowest window.
        let lowest = *window_sums.first().unwrap();

        // We're traversing windows from high to low.
        let b = &window_sums[1..]
            .iter()
            .rev()
            .fold(point_zero_var, |mut total, sum_i| {
                // total += sum_i
                total = circuit.ecc_add::<P>(&total, sum_i).unwrap();
                for _ in 0..c {
                    // double
                    total = circuit.ecc_double::<P>(&total).unwrap();
                }
                total
            });

        circuit.ecc_add::<P>(&lowest, b)
    }
}

// Again, it is assumed that none of the bases is the neutral element.
fn msm_pippenger_fixed<F, P>(
    circuit: &mut PlonkCircuit<F>,
    bases: &[SWAffine<P>],
    scalars: &[EmulatedVariable<P::ScalarField>],
    scalar_bit_length: usize,
) -> Result<PointVariable, CircuitError>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
    <P as CurveConfig>::ScalarField: PrimeField + EmulationConfig<F>,
{
    // ================================================
    // check inputs
    // ================================================
    for scalar in scalars.iter() {
        circuit.check_vars_bound(&scalar.0[..])?;
    }

    // ================================================
    // set up parameters
    // ================================================
    let mut c = if scalar_bit_length < 32 {
        3
    } else {
        ln_without_floats(scalar_bit_length)
    };

    // For now naively increment c until it divides the scalar limb length.
    while P::ScalarField::B % c != 0 {
        c += 1;
    }
    // ================================================
    // compute lookup tables and window sums
    // ================================================

    // Each window is of size `c`.
    // We divide up the bits 0..scalar_bit_length into windows of size `c`, and
    // in parallel process each such window.
    let mut window_sums = Vec::new();
    // We will later add 1 to all our scalars to avoid using the neutral element in the lookup table.
    let bases_vec = cfg_iter!(bases)
        .map(|base| {
            (1u32..=(1 << c))
                .map(|j| (*base * P::ScalarField::from(j)).into_affine())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<Vec<SWAffine<P>>>>();

    // We need the following sum to subtract from the final result, since we added 1 to all our scalars.
    let bases_sum = cfg_iter!(bases)
        .copied()
        .sum::<Projective<P>>()
        .into_affine();
    let mut num_scalars = 0;
    for (base_vec, scalar_var) in bases_vec.iter().zip(scalars.iter()) {
        // decompose scalar into c-bit scalars

        let decomposed_scalar_vars =
            decompose_emulated_scalar_var(circuit, scalar_var, c, scalar_bit_length)?;

        num_scalars = decomposed_scalar_vars.len();

        // create lookup point variables
        let mut lookup_point_vars_one = Vec::new();

        let scalar_vals = decomposed_scalar_vars
            .iter()
            .map(|&scalar_var| {
                let f_val = circuit.witness(scalar_var)?;
                Ok(fq_to_fr::<F, P>(&f_val))
            })
            .collect::<Result<Vec<P::ScalarField>, _>>()?;

        // We add 1 onto every scalar to avoid having to deal with the neutral point
        let lookup_points_one = cfg_iter!(scalar_vals)
            .map(|scalar_val| {
                let affine_point =
                    (base_vec[0] * (*scalar_val + P::ScalarField::one())).into_affine();
                Point::<P::BaseField>::from(affine_point)
            })
            .collect::<Vec<_>>();

        for lk_point in lookup_points_one {
            let lookup_point_var = circuit.create_point_variable(&lk_point)?;
            lookup_point_vars_one.push(lookup_point_var);
        }

        // create point table [1 * base, ..., 2^c * base]
        let table_points = base_vec
            .iter()
            .map(|p| Point::<F>::from(*p))
            .collect::<Vec<_>>();

        create_fixed_point_lookup_gates(
            circuit,
            &table_points,
            &decomposed_scalar_vars,
            &lookup_point_vars_one,
        )?;

        // update window sums
        if window_sums.is_empty() {
            window_sums = lookup_point_vars_one;
        } else {
            for (window_sum_mut, lookup_point_var) in
                window_sums.iter_mut().zip(lookup_point_vars_one.iter())
            {
                *window_sum_mut =
                    circuit.ecc_add_no_neutral::<P>(window_sum_mut, lookup_point_var)?;
            }
        }
    }

    // ================================================
    // performing additions
    // ================================================
    // We store the sum for the lowest window.
    let sum = (0..num_scalars).fold(P::ScalarField::zero(), |acc, i| {
        acc + P::ScalarField::from(2u8).pow([(c * i) as u64])
    });
    let final_point = (bases_sum * -sum).into_affine();
    let final_var = circuit.create_constant_point_variable(&Point::<F>::from(final_point))?;
    let lowest = *window_sums.first().unwrap();
    let mut last = *window_sums.last().unwrap();

    for _ in 0..c {
        last = circuit.ecc_add_no_neutral::<P>(&last, &last)?;
    }

    // We're traversing windows from high to low.
    let b = &window_sums[1..]
        .iter()
        .rev()
        .skip(1)
        .fold(last, |mut total, sum_i| {
            // total += sum_i
            total = circuit.ecc_add_no_neutral::<P>(&total, sum_i).unwrap();
            for _ in 0..c {
                // double
                total = circuit.ecc_add_no_neutral::<P>(&total, &total).unwrap();
            }
            total
        });

    let tmp = circuit.ecc_add_no_neutral::<P>(&lowest, b)?;
    circuit.ecc_add_no_neutral::<P>(&tmp, &final_var)
}

#[inline]
fn create_point_lookup_gates<F>(
    circuit: &mut PlonkCircuit<F>,
    table_point_vars: &[PointVariable],
    lookup_scalar_vars: &[Variable],
    lookup_point_vars: &[PointVariable],
) -> Result<(), CircuitError>
where
    F: PrimeField,
{
    let table_vars: Vec<(Variable, Variable)> = table_point_vars
        .iter()
        .map(|p| (p.get_x(), p.get_y()))
        .collect();
    let lookup_vars: Vec<(Variable, Variable, Variable)> = lookup_scalar_vars
        .iter()
        .zip(lookup_point_vars.iter())
        .map(|(&s, pt)| (s, pt.get_x(), pt.get_y()))
        .collect();
    circuit.create_table_and_lookup_variables(&lookup_vars, &table_vars)
}

#[inline]
fn create_fixed_point_lookup_gates<F>(
    circuit: &mut PlonkCircuit<F>,
    table_points: &[Point<F>],
    lookup_scalar_vars: &[Variable],
    lookup_point_vars: &[PointVariable],
) -> Result<(), CircuitError>
where
    F: PrimeField,
{
    let table_vars: Vec<(F, Variable, Variable)> = table_points
        .iter()
        .enumerate()
        .map(|(index, p)| {
            let y_var = circuit.create_variable(p.get_y())?;
            let scalar = circuit.create_variable(F::from(index as u32))?;
            Ok((p.get_x(), y_var, scalar))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let lookup_vars: Vec<(Variable, Variable, Variable)> = lookup_scalar_vars
        .iter()
        .zip(lookup_point_vars.iter())
        .map(|(&s, pt)| (pt.get_x(), pt.get_y(), s))
        .collect();
    circuit.create_table_and_lookup_variables_custom_keys(&lookup_vars, &table_vars)
}

#[inline]
/// Decompose a `scalar_bit_length`-bit scalar `s` into many c-bit scalar
/// variables `{s0, ..., s_m}` such that `s = \sum_{j=0..m} 2^{cj} * s_j`
/// Note the `s_j`s are not contrained here. That needs to be done elsewhere.
fn decompose_emulated_scalar_var<E, F>(
    circuit: &mut PlonkCircuit<F>,
    scalar_var: &EmulatedVariable<E>,
    c: usize,
    scalar_bit_length: usize,
) -> Result<Vec<Variable>, CircuitError>
where
    F: PrimeField,
    E: EmulationConfig<F>,
{
    if E::B % c == 0 {
        // create witness
        let m = (scalar_bit_length - 1) / c + 1;
        let mut scalar_val = circuit.emulated_witness(scalar_var)?.into_bigint();
        let decomposed_scalar_vars = (0..m)
            .map(|_| {
                // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
                let scalar_u64 = scalar_val.as_ref()[0] % (1 << c);
                // We right-shift by c bits, thus getting rid of the
                // lower bits.
                scalar_val.divn(c as u32);

                circuit.create_variable(F::from(scalar_u64))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // create circuit
        let range_size = F::from((1 << c) as u32);
        let scalars_per_limb = E::B / c;
        for (i, limb) in scalar_var.0.iter().enumerate() {
            let scalars_le = decomposed_scalar_vars
                .iter()
                .skip(i * scalars_per_limb)
                .take(scalars_per_limb)
                .copied()
                .collect::<Vec<_>>();
            circuit.decomposition_gate(scalars_le, *limb, range_size)?;
        }

        Ok(decomposed_scalar_vars)
    } else {
        // create witness
        let m = (scalar_bit_length - 1) / c + 1;
        let mut scalar_val = circuit.emulated_witness(scalar_var)?.into_bigint();
        let decomposed_scalar_vars = (0..m)
            .map(|_| {
                // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
                let scalar_u64 = scalar_val.as_ref()[0] % (1 << c);
                // We right-shift by c bits, thus getting rid of the
                // lower bits.
                scalar_val.divn(c as u32);

                circuit.create_variable(F::from(scalar_u64))
            })
            .collect::<Result<Vec<_>, _>>()?;

        for scalar_var in decomposed_scalar_vars.clone() {
            circuit.enforce_in_range(scalar_var, c)?;
        }
        // create circuit
        circuit.emulated_decomposition_gate(&decomposed_scalar_vars, scalar_var, c)?;
        Ok(decomposed_scalar_vars)
    }
}

#[inline]
/// Decompose a `scalar_bit_length`-bit scalar `s` into many c-bit scalar
/// variables `{s0, ..., s_m}` such that `s = \sum_{j=0..m} 2^{cj} * s_j`
pub fn decompose_scalar_var_signed_emulated<P, F>(
    circuit: &mut PlonkCircuit<F>,
    scalar_var: Variable,
    c: usize,
    scalar_bit_length: usize,
) -> Result<(Vec<(bool, Variable)>, BoolVar), CircuitError>
where
    P: HasTEForm<BaseField = F>,
    F: PrimeField,
{
    // create witness

    let mut scalar_val_field = circuit.witness(scalar_var)?;

    let scalar_val_biguint: BigUint = scalar_val_field.into();
    let skew = if scalar_val_biguint % 2u8 == BigUint::zero() {
        scalar_val_field += F::one();
        F::one()
    } else {
        F::zero()
    };

    let new_window_scalars =
        decompose_for_msm(&scalar_val_field.into_bigint(), c, scalar_bit_length)
            .iter()
            .rev()
            .scan(0i64, |state, x| {
                let x = x + *state;
                *state = 0;
                if x % 2 == 0 {
                    *state -= 1i64 << c;
                    Some(x + 1)
                } else {
                    Some(x)
                }
            })
            .collect::<Vec<_>>();

    let field_elems_with_sign = new_window_scalars
        .iter()
        .rev()
        .map(|x| {
            let sign = match x.signum() {
                0 => false,
                1 => false,
                -1 => true,
                _ => unreachable!(),
            };

            (sign, F::from(x.unsigned_abs()))
        })
        .collect::<Vec<_>>();

    // create circuit
    let range_size = F::from((1 << c) as u32);
    let skew_var = circuit.create_boolean_variable_unchecked(skew)?;
    let scalar_field_vars = field_elems_with_sign
        .iter()
        .map(|(sign, x)| Ok((*sign, circuit.create_variable(*x)?)))
        .collect::<Result<Vec<_>, _>>()?;

    let mut decomposed_scalar_vars = field_elems_with_sign
        .iter()
        .map(|(sign, x)| {
            let x = if *sign { -*x } else { *x };
            circuit.create_variable(x)
        })
        .collect::<Result<Vec<_>, _>>()?;
    decomposed_scalar_vars[0] = circuit.sub(decomposed_scalar_vars[0], skew_var.into())?;
    circuit.decomposition_gate(decomposed_scalar_vars, scalar_var, range_size)?;

    Ok((scalar_field_vars, skew_var))
}

/// Create point lookup gates with custom keys.
#[inline]
pub fn create_point_lookup_gates_custom_keys_emulated<F>(
    circuit: &mut PlonkCircuit<F>,
    table_point_vars: &[PointVariable],
    lookup_scalar_vars: &[(bool, Variable)],
    lookup_point_vars: &[PointVariable],
) -> Result<(), CircuitError>
where
    F: PrimeField,
{
    let table_vars: Vec<(F, Variable, Variable)> = table_point_vars
        .iter()
        .enumerate()
        .map(|(index, p)| {
            let key = if index < table_point_vars.len() / 2 {
                F::from(2 * index as u32 + 1)
            } else {
                let value = index - (table_point_vars.len() / 2);
                -F::from(2 * value as u32 + 1)
            };
            (key, p.get_x(), p.get_y())
        })
        .collect();

    let mut lookup_scalars = vec![];
    for (sign, s) in lookup_scalar_vars.iter() {
        let scalar_val = if *sign {
            -circuit.witness(*s)?
        } else {
            circuit.witness(*s)?
        };
        let scalar = circuit.create_variable(scalar_val)?;
        lookup_scalars.push(scalar);
    }

    let lookup_vars: Vec<(Variable, Variable, Variable)> = lookup_scalars
        .iter()
        .zip(lookup_point_vars.iter())
        .map(|(&s, pt)| (s, pt.get_x(), pt.get_y()))
        .collect();
    circuit.create_table_and_lookup_variables_custom_keys(&lookup_vars, &table_vars)
}

#[inline]
/// Compute the value of scalar multiplication `witness(scalar_var) *
/// witness(base_var)`. This function does not add any constraints.
fn compute_scalar_mul_value<F, P>(
    circuit: &PlonkCircuit<F>,
    scalar_var: Variable,
    base_var: &PointVariable,
) -> Result<Point<F>, CircuitError>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
{
    let curve_point: SWAffine<P> = circuit.point_witness(base_var)?.into();
    let scalar = fq_to_fr::<F, P>(&circuit.witness(scalar_var)?);
    let res = curve_point * scalar;
    Ok(res.into_affine().into())
}

#[inline]
/// Compute the value of scalar multiplication `witness(scalar_var) *
/// witness(base_var)`. This function does not add any constraints.
pub(crate) fn compute_scalar_mul_value_signed<F, P>(
    circuit: &PlonkCircuit<F>,
    scalar_var: Variable,
    base_var: &PointVariable,
) -> Result<Point<F>, CircuitError>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
{
    let mut big_uint: BigUint = circuit.witness(scalar_var)?.into();
    let sign = if big_uint.clone() * 2u8 > F::MODULUS.into() {
        big_uint = F::MODULUS.into() - big_uint;
        true
    } else {
        false
    };
    let curve_point: SWAffine<P> = circuit.point_witness(base_var)?.into();
    let mut scalar: P::ScalarField = big_uint.into();
    scalar = if sign { -scalar } else { scalar };
    let res = curve_point * scalar;
    Ok(res.into_affine().into())
}

/// The result of this function is only approximately `ln(a)`
/// [`Explanation of usage`]
///
/// [`Explanation of usage`]: https://github.com/scipr-lab/zexe/issues/79#issue-556220473
fn ln_without_floats(a: usize) -> usize {
    // log2(a) * ln(2)
    (ark_std::log2(a) * 69 / 100) as usize
}

#[cfg(test)]
mod tests {

    use core::marker::PhantomData;

    use super::*;
    use crate::{Circuit, PlonkType};

    use ark_ec::{
        scalar_mul::variable_base::VariableBaseMSM, short_weierstrass::Projective as SWProjective,
    };

    use ark_ff::UniformRand;
    use ark_std::vec::Vec;
    use nf_curves::grumpkin::{short_weierstrass::SWGrumpkin, Fq as FqGrump};

    const RANGE_BIT_LEN_FOR_TEST: usize = 8;

    #[test]
    fn test_emulated_variable_base_multi_scalar_mul() -> Result<(), CircuitError> {
        // test_emulated_variable_base_multi_scalar_mul_helper::<Fq377, Param377>(
        //     PlonkType::TurboPlonk,
        // )?;
        test_emulated_variable_base_multi_scalar_mul_helper::<FqGrump, SWGrumpkin>(
            PlonkType::UltraPlonk,
        )?;
        // test_emulated_variable_base_multi_scalar_mul_helper::<Fq377, Param377>(
        //     PlonkType::UltraPlonk,
        // )?;

        // // uncomment the following code to dump the circuit comparison to screen
        // assert!(false);

        Ok(())
    }

    fn test_emulated_variable_base_multi_scalar_mul_helper<F, P>(
        plonk_type: PlonkType,
    ) -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
        <P as CurveConfig>::ScalarField: PrimeField + EmulationConfig<F>,
    {
        let mut rng = jf_utils::test_rng();

        for dim in [1, 2, 4, 8, 16, 32, 64, 128] {
            for _ in 0..20 {
                let mut circuit: PlonkCircuit<F> = match plonk_type {
                    PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
                    PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST),
                };

                // bases and scalars
                let bases: Vec<SWAffine<P>> =
                    (0..dim).map(|_| SWAffine::<P>::rand(&mut rng)).collect();
                let scalars: Vec<P::ScalarField> =
                    (0..dim).map(|_| P::ScalarField::rand(&mut rng)).collect();
                let scalar_reprs: Vec<<P::ScalarField as PrimeField>::BigInt> =
                    scalars.iter().map(|x| x.into_bigint()).collect();
                let res = SWProjective::<P>::msm_bigint(&bases, &scalar_reprs).into_affine();
                let res_point = res.into();

                // corresponding wires
                let bases_point: Vec<Point<F>> = bases.iter().map(|x| (*x).into()).collect();
                let bases_vars: Vec<PointVariable> = bases_point
                    .iter()
                    .map(|x| circuit.create_point_variable(x))
                    .collect::<Result<Vec<_>, _>>()?;
                let scalar_vars: Vec<EmulatedVariable<_>> = scalars
                    .iter()
                    .map(|x| circuit.create_emulated_variable(*x))
                    .collect::<Result<Vec<_>, _>>()?;

                // compute circuit
                let res_var = EmulMultiScalarMultiplicationCircuit::<F, P>::msm(
                    &mut circuit,
                    &bases_vars,
                    &scalar_vars,
                )?;

                assert_eq!(circuit.point_witness(&res_var)?, res_point);

                // uncomment the following code to dump the circuit comparison to screen
                ark_std::println!("number of basis: {}", dim);
                ark_std::println!("#variables: {}", circuit.num_vars(),);
                ark_std::println!("#constraints: {}\n", circuit.num_gates(),);

                // wrong witness should fail
                *circuit.witness_mut(2) = F::rand(&mut rng);
                assert!(circuit.check_circuit_satisfiability(&[]).is_err());
                // un-matching basis & scalars
                assert!(EmulMultiScalarMultiplicationCircuit::<F, P>::msm(
                    &mut circuit,
                    &bases_vars[0..dim - 1],
                    &scalar_vars
                )
                .is_err());

                // Check variable out of bound error.
                let var_number = circuit.num_vars();
                assert!(EmulMultiScalarMultiplicationCircuit::<F, P>::msm(
                    &mut circuit,
                    &[PointVariable::TE(var_number, var_number)],
                    &scalar_vars
                )
                .is_err());
                let emul_var = EmulatedVariable::<P::ScalarField>(
                    vec![var_number],
                    PhantomData::<P::ScalarField>,
                );
                assert!(EmulMultiScalarMultiplicationCircuit::<F, P>::msm(
                    &mut circuit,
                    &bases_vars,
                    &[emul_var]
                )
                .is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn test_emulated_fixed_base_multi_scalar_mul() -> Result<(), CircuitError> {
        test_emulated_fixed_base_multi_scalar_mul_helper::<FqGrump, SWGrumpkin>(
            PlonkType::UltraPlonk,
        )?;

        // // uncomment the following code to dump the circuit comparison to screen
        // assert!(false);

        Ok(())
    }

    fn test_emulated_fixed_base_multi_scalar_mul_helper<F, P>(
        plonk_type: PlonkType,
    ) -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
        <P as CurveConfig>::ScalarField: PrimeField + EmulationConfig<F>,
    {
        let mut rng = jf_utils::test_rng();

        for dim in [1, 2, 4, 8, 16, 32, 64, 128] {
            for _ in 0..20 {
                let mut circuit: PlonkCircuit<F> = match plonk_type {
                    PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
                    PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST),
                };

                // bases and scalars
                let bases: Vec<SWAffine<P>> =
                    (0..dim).map(|_| SWAffine::<P>::rand(&mut rng)).collect();
                let scalars: Vec<P::ScalarField> =
                    (0..dim).map(|_| P::ScalarField::rand(&mut rng)).collect();
                let scalar_reprs: Vec<<P::ScalarField as PrimeField>::BigInt> =
                    scalars.iter().map(|x| x.into_bigint()).collect();
                let res = SWProjective::<P>::msm_bigint(&bases, &scalar_reprs).into_affine();
                let res_point = res.into();

                // corresponding wires
                let scalar_vars: Vec<EmulatedVariable<_>> = scalars
                    .iter()
                    .map(|x| circuit.create_emulated_variable(*x))
                    .collect::<Result<Vec<_>, _>>()?;

                // compute circuit
                let res_var = EmulMultiScalarMultiplicationCircuit::<F, P>::fixed_base_msm(
                    &mut circuit,
                    &bases,
                    &scalar_vars,
                )?;

                assert_eq!(circuit.point_witness(&res_var)?, res_point);
                circuit.check_circuit_satisfiability(&[]).unwrap();
                // uncomment the following code to dump the circuit comparison to screen
                ark_std::println!("number of basis: {}", dim);
                ark_std::println!("#variables: {}", circuit.num_vars(),);
                ark_std::println!("#constraints: {}\n", circuit.num_gates(),);

                // wrong witness should fail
                *circuit.witness_mut(2) = F::rand(&mut rng);
                assert!(circuit.check_circuit_satisfiability(&[]).is_err());
                // un-matching basis & scalars
                assert!(
                    EmulMultiScalarMultiplicationCircuit::<F, P>::fixed_base_msm(
                        &mut circuit,
                        &bases[0..dim - 1],
                        &scalar_vars
                    )
                    .is_err()
                );

                // Check variable out of bound error.
                let var_number = circuit.num_vars();
                let emul_var = EmulatedVariable::<P::ScalarField>(
                    vec![var_number],
                    PhantomData::<P::ScalarField>,
                );
                assert!(
                    EmulMultiScalarMultiplicationCircuit::<F, P>::fixed_base_msm(
                        &mut circuit,
                        &bases,
                        &[emul_var]
                    )
                    .is_err()
                );
            }
        }
        Ok(())
    }
}
