// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! This module implements multi-scalar-multiplication circuits.

use super::{compute_scalar_mul_value_signed, HasTEForm, Point, PointVariable};
use crate::{errors::CircuitError, BoolVar, Circuit, PlonkCircuit, Variable};
use ark_ec::{
    scalar_mul,
    short_weierstrass::{Affine as SWAffine, SWCurveConfig as SWConfig},
    AffineRepr, CurveConfig, CurveGroup,
};
use ark_ff::{BigInteger, One, PrimeField, Zero};
use ark_std::string::ToString;
use ark_std::{format, vec, vec::Vec};
use itertools::{izip, MultiUnzip};
use jf_utils::fq_to_fr;
use num_bigint::BigInt;
use num_bigint::BigUint;
use num_traits::sign;

/// Compute the multi-scalar-multiplications in circuit.
pub trait MultiScalarMultiplicationCircuit<F, P>
where
    F: PrimeField,
    P: SWConfig<BaseField = F>,
{
    /// Compute the multi-scalar-multiplications.
    /// Use pippenger when the circuit supports lookup;
    /// Use naive method otherwise.
    /// Return error if the number bases does not match the number of scalars.
    fn msm(
        &mut self,
        bases: &[PointVariable],
        scalars: &[Variable],
    ) -> Result<PointVariable, CircuitError>;

    /// Compute the multi-scalar-multiplications where each scalar has at most
    /// `scalar_bit_length` bits.
    fn msm_with_var_scalar_length(
        &mut self,
        bases: &[PointVariable],
        scalars: &[Variable],
        scalar_bit_length: usize,
    ) -> Result<PointVariable, CircuitError>;
}

impl<F, P> MultiScalarMultiplicationCircuit<F, P> for PlonkCircuit<F>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
{
    fn msm(
        &mut self,
        bases: &[PointVariable],
        scalars: &[Variable],
    ) -> Result<PointVariable, CircuitError> {
        if bases.len() != scalars.len() {
            return Err(CircuitError::ParameterError(format!(
                "bases length ({}) does not match scalar length ({})",
                bases.len(),
                scalars.len()
            )));
        }
        for (&scalar, base) in scalars.iter().zip(bases.iter()) {
            self.check_var_bound(scalar)?;
            self.check_point_var_bound(base)?;
        }
        // The scalar field modulus must be less than the base field modulus.
        let base_field_mod: BigUint = F::MODULUS.into();
        let scalar_field_mod: BigUint = P::ScalarField::MODULUS.into();
        if base_field_mod <= scalar_field_mod {
            return Err(CircuitError::ParameterError(format!(
                "base field modulus ({:?}) must be strictly greater than scalar field modulus ({:?})",
                F::MODULUS,
                P::ScalarField::MODULUS
            )));
        }
        let scalar_bit_length = <P as CurveConfig>::ScalarField::MODULUS_BIT_SIZE as usize;
        // We need the `scalar_bit_length` to be less that the bit-size of `F`.
        // If they are equal, we reduce the `scalar_bit_length` by `1` and
        // "negate" each `base` and `scalar`, where appropriate, to ensure this
        // bit-length is satisfied.
        let mut new_bases = Vec::<PointVariable>::new();
        let mut new_scalars = Vec::<Variable>::new();
        let mut new_scalar_bit_length = scalar_bit_length;
        if scalar_bit_length == F::MODULUS_BIT_SIZE as usize {
            new_scalar_bit_length -= 1;
            let scalarfield_modulus: BigUint = P::ScalarField::MODULUS.into();
            let f_mod = F::from(scalarfield_modulus);
            for (base, scalar) in bases.iter().zip(scalars.iter()) {
                // We negate the base and the scalar
                let neg_base = self.inverse_point(base)?;
                let neg_scalar = self.lin_comb(&[-F::one()], &f_mod, &[*scalar])?;
                let scalar_big_uint: BigUint = self.witness(*scalar)?.into();
                let sign = scalar_big_uint >= BigUint::from(1u8) << new_scalar_bit_length;
                let sign_var = self.create_boolean_variable(sign)?;
                let new_base = self.binary_point_vars_select(sign_var, base, &neg_base)?;
                // If the scalar is too big, we swap the base and scalar for their negatives
                let new_scalar = self.conditional_select(sign_var, *scalar, neg_scalar)?;
                new_bases.push(new_base);
                new_scalars.push(new_scalar);
            }
        } else {
            new_bases = bases.to_vec();
            new_scalars = scalars.to_vec();
        }

        MultiScalarMultiplicationCircuit::<F, P>::msm_with_var_scalar_length(
            self,
            &new_bases,
            &new_scalars,
            new_scalar_bit_length,
        )
    }

    fn msm_with_var_scalar_length(
        &mut self,
        bases: &[PointVariable],
        scalars: &[Variable],
        scalar_bit_length: usize,
    ) -> Result<PointVariable, CircuitError> {
        if bases.len() != scalars.len() {
            return Err(CircuitError::ParameterError(format!(
                "bases length ({}) does not match scalar length ({})",
                bases.len(),
                scalars.len()
            )));
        }
        if F::MODULUS_BIT_SIZE as usize <= scalar_bit_length {
            return Err(CircuitError::ParameterError(format!(
                "base field bit size ({:?}) must be strictly greater than scalar bit length ({:?})",
                F::MODULUS_BIT_SIZE,
                scalar_bit_length
            )));
        }
        for (&scalar, base) in scalars.iter().zip(bases.iter()) {
            self.check_var_bound(scalar)?;
            self.check_point_var_bound(base)?;
        }

        if self.support_lookup() {
            msm_pippenger::<F, P>(self, bases, scalars, scalar_bit_length)
        } else {
            msm_naive::<F, P>(self, bases, scalars, scalar_bit_length)
        }
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
    scalars: &[Variable],
    scalar_bit_length: usize,
) -> Result<PointVariable, CircuitError>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
{
    if F::MODULUS_BIT_SIZE as usize <= scalar_bit_length {
        return Err(CircuitError::ParameterError(format!(
            "base field bit size ({:?}) must be strictly greater than scalar bit length ({:?})",
            F::MODULUS_BIT_SIZE,
            scalar_bit_length
        )));
    }
    for (&scalar, base) in scalars.iter().zip(bases.iter()) {
        circuit.check_var_bound(scalar)?;
        circuit.check_point_var_bound(base)?;
    }

    let scalar_0_bits_le = circuit.unpack(scalars[0], scalar_bit_length)?;
    let mut res = circuit.variable_base_binary_scalar_mul::<P>(&scalar_0_bits_le, &bases[0])?;

    for (base, scalar) in bases.iter().zip(scalars.iter()).skip(1) {
        let scalar_bits_le = circuit.unpack(*scalar, scalar_bit_length)?;
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
    scalars: &[Variable],
    scalar_bit_length: usize,
) -> Result<PointVariable, CircuitError>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
{
    if F::MODULUS_BIT_SIZE as usize <= scalar_bit_length {
        return Err(CircuitError::ParameterError(format!(
            "base field bit size ({:?}) must be strictly greater than scalar bit length ({:?})",
            F::MODULUS_BIT_SIZE,
            scalar_bit_length
        )));
    }
    // ================================================
    // check inputs
    // ================================================
    for (&scalar, base) in scalars.iter().zip(bases.iter()) {
        circuit.check_var_bound(scalar)?;
        circuit.check_point_var_bound(base)?;
    }

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
    let mut skews = Vec::new();
    let mut neg_bases = Vec::new();
    for (base_var, &scalar_var) in bases.iter().zip(scalars.iter()) {
        // decompose scalar into c-bit scalars
        let (decomposed_scalar_vars, skew) =
            decompose_scalar_var_signed::<P, F>(circuit, scalar_var, c, scalar_bit_length)?;
        // We need to contrain the last scalar in the decomposition to be positive and
        // ensure that we don't exceed the `scalar_bit_length`. All the other scalars
        // will be constrained appropriately by the lookup table.
        if scalar_bit_length % c != 0 {
            let last_scalar_var = decomposed_scalar_vars.last().unwrap();
            circuit.range_gate_with_lookup(*last_scalar_var, scalar_bit_length % c)?;
        }

        // create point table [1 * base, 3 * base, ..., (2^c - 1) * base]
        let double_base_var = circuit.ecc_add_no_neutral::<P>(base_var, base_var)?;
        let mut table_point_vars = vec![*base_var];

        for i in 1usize..(1 << (c - 1)) {
            let point =
                circuit.ecc_add_no_neutral::<P>(&table_point_vars[i - 1], &double_base_var)?;
            table_point_vars.push(point);
        }

        let mut neg_points = vec![];

        if !P::has_te_form() {
            for (j, point) in table_point_vars.iter().enumerate() {
                let point_x = point.get_x();
                let point_y = point.get_y();
                let new_y = circuit.sub(circuit.zero(), point_y)?;
                let neg_point = PointVariable::SW(point_x, new_y);
                neg_points.push(neg_point);
                if j == 0 {
                    neg_bases.push(neg_point);
                }
            }
            table_point_vars.extend_from_slice(&neg_points);
        } else {
            for (j, point) in table_point_vars.iter().enumerate() {
                let neg_point = circuit.inverse_point(point)?;
                neg_points.push(neg_point);
                if j == 0 {
                    neg_bases.push(neg_point);
                }
            }
            table_point_vars.extend_from_slice(&neg_points);
        }

        // create lookup point variables
        let mut lookup_point_vars = Vec::new();
        for &scalar_var in decomposed_scalar_vars.iter() {
            let lookup_point =
                compute_scalar_mul_value_signed::<F, P>(circuit, scalar_var, base_var)?;
            let lookup_point_var = circuit.create_point_variable(&lookup_point)?;
            lookup_point_vars.push(lookup_point_var);
        }

        create_point_lookup_gates_custom_keys::<P, F>(
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
                *window_sum_mut =
                    circuit.ecc_add_no_neutral::<P>(window_sum_mut, lookup_point_var)?;
            }
        }
        skews.push(skew);
    }

    // ================================================
    // performing additions
    // ================================================
    // We store the sum for the lowest window.
    let lowest = *window_sums.first().unwrap();
    let mut last_point = *window_sums.last().unwrap();

    for _ in 0..c {
        last_point = circuit.ecc_double::<P>(&last_point)?;
    }

    // We're traversing windows from high to low.
    let b = &window_sums[1..]
        .iter()
        .rev()
        .skip(1)
        .fold(last_point, |mut total, sum_i| {
            // total += sum_i
            total = circuit.ecc_add_no_neutral::<P>(&total, sum_i).unwrap();

            for _ in 0..c {
                // double
                total = circuit.ecc_add_no_neutral::<P>(&total, &total).unwrap();
            }
            total
        });
    let mut final_point = circuit.ecc_add_no_neutral::<P>(&lowest, b)?;
    for (skew, base) in skews.into_iter().zip(neg_bases.iter()) {
        let point = circuit.ecc_add_no_neutral::<P>(&final_point, base)?;
        final_point = circuit.binary_point_vars_select(skew, &final_point, &point)?;
    }
    Ok(final_point)
}

/// Create point lookup gates with custom keys.
#[inline]
pub fn create_point_lookup_gates_custom_keys<P, F>(
    circuit: &mut PlonkCircuit<F>,
    table_point_vars: &[PointVariable],
    lookup_scalar_vars: &[Variable],
    lookup_point_vars: &[PointVariable],
) -> Result<(), CircuitError>
where
    P: HasTEForm<BaseField = F>,
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

    let lookup_vars: Vec<(Variable, Variable, Variable)> = lookup_scalar_vars
        .iter()
        .zip(lookup_point_vars.iter())
        .map(|(&s, pt)| (s, pt.get_x(), pt.get_y()))
        .collect();
    circuit.create_table_and_lookup_variables_custom_keys(&lookup_vars, &table_vars)
}

/*#[inline]
/// Decompose a `scalar_bit_length`-bit scalar `s` into many c-bit scalar
/// variables `{s0, ..., s_m}` such that `s = \sum_{j=0..m} 2^{cj} * s_j`
pub fn decompose_scalar_var<F>(
    circuit: &mut PlonkCircuit<F>,
    scalar_var: Variable,
    c: usize,
    scalar_bit_length: usize,
) -> Result<Vec<Variable>, CircuitError>
where
    F: PrimeField,
{
    // create witness
    let m = (scalar_bit_length - 1) / c + 1;
    let mut scalar_val = circuit.witness(scalar_var)?.into_bigint();
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

    for var in decomposed_scalar_vars.clone() {
        circuit.enforce_in_range(var, c)?;
    }

    // create circuit
    let range_size = F::from((1 << c) as u32);
    circuit.decomposition_gate(decomposed_scalar_vars.clone(), scalar_var, range_size)?;

    Ok(decomposed_scalar_vars)
}*/

#[inline]
/// Decompose a `scalar_bit_length`-bit scalar `s` into many c-bit scalar
/// variables `{s0, ..., s_m}` such that `s = \sum_{j=0..m} 2^{cj} * s_j`
/// Here, each `s_j`, with the possible exception of `s_0`, satisfies
/// `-2^c < s_j < 2^c` with `|s_j|` odd.
/// We do not constrain the `s_j`s here. This will need to be done in the
/// separately in a lookup table at the same time the elliptic curve
/// points are constrained for scalar multiplication.
pub fn decompose_scalar_var_signed<P, F>(
    circuit: &mut PlonkCircuit<F>,
    scalar_var: Variable,
    c: usize,
    scalar_bit_length: usize,
) -> Result<(Vec<Variable>, BoolVar), CircuitError>
where
    P: HasTEForm<BaseField = F>,
    F: PrimeField,
{
    // create witness
    let mut scalar_val_field = circuit.witness(scalar_var)?;

    let scalar_val_biguint: BigUint = scalar_val_field.into();
    // We need our scalar to be odd. To ensure this, we add a skew.
    let skew = scalar_val_biguint % 2u8 == BigUint::zero();
    scalar_val_field += F::from(skew);

    // `new_window_scalars` is the signed representation of the scalar.
    let signed_scalars = decompose_for_msm(&scalar_val_field.into_bigint(), c, scalar_bit_length)
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

    let mut field_sum = F::zero();
    let mut coeff = F::one();
    for scalar in signed_scalars.iter().rev() {
        let mut field_scalar = F::from(scalar.unsigned_abs());
        if *scalar < 0 {
            field_scalar = -field_scalar;
        }
        field_sum += coeff * field_scalar;
        coeff *= F::from(1u64 << c);
    }

    let mut signed_vars = signed_scalars
        .iter()
        .rev()
        .map(|x| {
            let sign = match x.signum() {
                0 => false,
                1 => false,
                -1 => true,
                _ => unreachable!(),
            };
            let abs_x = x.unsigned_abs();
            let val = match sign {
                true => -F::from(abs_x),
                false => F::from(abs_x),
            };
            circuit.create_variable(val)
        })
        .collect::<Result<Vec<Variable>, CircuitError>>()?;

    // create circuit
    let range_size = F::from((1 << c) as u32);
    let skew_var = circuit.create_boolean_variable(skew)?;
    let first_var = signed_vars[0];
    signed_vars[0] = circuit.sub(signed_vars[0], skew_var.into())?;
    circuit.decomposition_gate(signed_vars.clone(), scalar_var, range_size)?;
    signed_vars[0] = first_var;

    Ok((signed_vars, skew_var))
}

/// Returns the windowed non-adjacent form of `self`, for a window of size `w`.
pub(crate) fn decompose_for_msm<B: BigInteger>(
    input: &B,
    c: usize,
    scalar_bit_length: usize,
) -> Vec<i64> {
    // w > 2 due to definition of wNAF, and w < 64 to make sure that `i64`
    // can fit each signed digit

    let mut res = vec![];
    let mut e = *input;

    let m = (scalar_bit_length as u32 - 1) / c as u32 + 1;
    for _ in 0..m {
        let z = (e.as_ref()[0] % (1u64 << c)) as u32;

        let z = z as i64;
        res.push(z);
        e.divn(c as u32);
    }

    res
}

/*#[inline]
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
    let point_var = circuit.point_witness(base_var)?;
    match point_var {
        Point::TE(..) => {
            let curve_point: SWAffine<P> = point_var.into();
            let proj_point = curve_point.into_group();
            let scalar = fq_to_fr::<F, P>(&circuit.witness(scalar_var)?);
            let res = proj_point * scalar;
            let result = res.into_affine().into();
            Ok(result)
        },
        Point::SW(x, y) => {
            if x == F::zero() && y == F::one() {
                Ok(Point::SW(F::zero(), F::one()))
            } else {
                let curve_point: SWAffine<P> = point_var.into();
                let proj_point = curve_point.into_group();
                let scalar = fq_to_fr::<F, P>(&circuit.witness(scalar_var)?);
                let res = proj_point * scalar;
                let result = res.into_affine().into();
                Ok(result)
            }
        },
    }
}*/

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

    use super::*;
    use crate::{gadgets::ecc::HasTEForm, Circuit, PlonkType};
    use ark_bls12_377::{g1::Config as Param377, Fq as Fq377};
    use ark_bn254::{g1::Config as BnConfig, Fq as Fq254};
    use ark_ec::{short_weierstrass::Projective as SWProjective, VariableBaseMSM};
    use ark_ed_on_bls12_381::{EdwardsConfig as ParamEd381, Fq as FqEd381};
    use ark_ff::UniformRand;
    use ark_std::vec::Vec;
    use jf_utils::fr_to_fq;
    use nf_curves::{
        ed_on_bls_12_381_bandersnatch::{EdwardsConfig as Param381b, Fq as FqEd381b},
        ed_on_bn254::{BabyJubjub as ParamEd254, Fq as FqEd254},
    };

    const RANGE_BIT_LEN_FOR_TEST: usize = 8;

    #[test]
    fn test_variable_base_multi_scalar_mul() -> Result<(), CircuitError> {
        test_variable_base_multi_scalar_mul_helper::<FqEd254, ParamEd254>(PlonkType::TurboPlonk)?;
        test_variable_base_multi_scalar_mul_helper::<FqEd254, ParamEd254>(PlonkType::UltraPlonk)?;
        test_variable_base_multi_scalar_mul_helper::<Fq254, BnConfig>(PlonkType::TurboPlonk)?;
        test_variable_base_multi_scalar_mul_helper::<Fq254, BnConfig>(PlonkType::UltraPlonk)?;
        test_variable_base_multi_scalar_mul_helper::<FqEd381, ParamEd381>(PlonkType::TurboPlonk)?;
        test_variable_base_multi_scalar_mul_helper::<FqEd381, ParamEd381>(PlonkType::UltraPlonk)?;
        test_variable_base_multi_scalar_mul_helper::<FqEd381b, Param381b>(PlonkType::TurboPlonk)?;
        test_variable_base_multi_scalar_mul_helper::<FqEd381b, Param381b>(PlonkType::UltraPlonk)?;
        test_variable_base_multi_scalar_mul_helper::<Fq377, Param377>(PlonkType::TurboPlonk)?;
        test_variable_base_multi_scalar_mul_helper::<Fq377, Param377>(PlonkType::UltraPlonk)?;

        // // uncomment the following code to dump the circuit comparison to screen
        // assert!(false);

        Ok(())
    }

    fn test_variable_base_multi_scalar_mul_helper<F, P>(
        plonk_type: PlonkType,
    ) -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();

        for dim in [1, 2, 4, 8, 16, 32, 64, 128] {
            let mut circuit: PlonkCircuit<F> = match plonk_type {
                PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
                PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST),
            };

            // bases and scalars
            let bases: Vec<SWAffine<P>> = (0..dim).map(|_| SWAffine::<P>::rand(&mut rng)).collect();
            let scalars: Vec<P::ScalarField> =
                (0..dim).map(|_| P::ScalarField::rand(&mut rng)).collect();
            let scalar_reprs: Vec<<P::ScalarField as PrimeField>::BigInt> =
                scalars.iter().map(|x| x.into_bigint()).collect();
            let res = SWProjective::<P>::msm_bigint(&bases, &scalar_reprs);
            let res_point: Point<F> = res.into_affine().into();

            // corresponding wires
            let base_point: Vec<Point<F>> = bases.iter().map(|x| (*x).into()).collect();
            let bases_vars: Vec<PointVariable> = base_point
                .iter()
                .map(|x| circuit.create_point_variable(x))
                .collect::<Result<Vec<_>, _>>()?;
            let scalar_vars: Vec<Variable> = scalars
                .iter()
                .map(|x| circuit.create_variable(fr_to_fq::<F, P>(x)))
                .collect::<Result<Vec<_>, _>>()?;

            // compute circuit
            let res_var = MultiScalarMultiplicationCircuit::<F, P>::msm(
                &mut circuit,
                &bases_vars,
                &scalar_vars,
            )?;

            assert_eq!(circuit.point_witness(&res_var)?, res_point);

            // uncomment the following code to dump the circuit comparison to screen
            // ark_std::println!("number of basis: {}", dim);
            // ark_std::println!("#variables: {}", circuit.num_vars(),);
            // ark_std::println!("#constraints: {}\n", circuit.num_gates(),);

            // wrong witness should fail
            *circuit.witness_mut(2) = F::rand(&mut rng);
            assert!(circuit.check_circuit_satisfiability(&[]).is_err());
            // un-matching basis & scalars
            assert!(MultiScalarMultiplicationCircuit::<F, P>::msm(
                &mut circuit,
                &bases_vars[0..dim - 1],
                &scalar_vars
            )
            .is_err());

            // Check variable out of bound error.
            let var_number = circuit.num_vars();
            assert!(MultiScalarMultiplicationCircuit::<F, P>::msm(
                &mut circuit,
                &[PointVariable::TE(var_number, var_number)],
                &scalar_vars
            )
            .is_err());
            assert!(MultiScalarMultiplicationCircuit::<F, P>::msm(
                &mut circuit,
                &bases_vars,
                &[var_number]
            )
            .is_err());
        }
        Ok(())
    }
}
