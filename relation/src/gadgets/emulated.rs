// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Emulate arithmetic operations on a large prime field.
//! To emulate arithmetic operations on F_q when the native field is F_p where p
//! < q, we represent the elements in F_q using CRT modulus [p, 2^T] where p *
//! 2^T > q^2 + q. This constraint is required to emulate the F_q multiplication
//! by checking a * b - k * q = c (mod 2^T * p) without any overflow. The second
//! componenet, with modulus 2^T, will be divided into limbs each with B bits
//! where 2^{2B} < p.

use super::utils::next_multiple;
use crate::{
    constants::GATE_WIDTH, errors::CircuitError, BoolVar, Circuit, PlonkCircuit, Variable,
};
use ark_ff::{BigInteger, PrimeField};

use ark_std::{format, string::ToString, vec, vec::Vec, One, Zero};
use core::{marker::PhantomData, num};
use itertools::izip;
use jf_utils::{bytes_to_field_elements, field_switching, to_bytes};
use num_bigint::BigUint;

/// Parameters needed for emulating field operations over [`F`].
pub trait EmulationConfig<F: PrimeField>: PrimeField {
    /// Log2 of the other CRT modulus is 2^T.
    const T: usize;
    /// Bit length of each limbs.
    const B: usize;
    /// `B * NUM_LIMBS` should equals to `T`.
    const NUM_LIMBS: usize;
}

/// A struct that can be serialized into `Vec` of field elements.
pub trait SerializableEmulatedStruct<F: PrimeField> {
    /// Serialize into a `Vec` of field elements.
    fn serialize_to_native_elements(&self) -> Vec<F>;
}

fn biguint_to_limbs<F: PrimeField>(val: &BigUint, b: usize, num_limbs: usize) -> Vec<F> {
    let mut result = vec![];
    let b_pow = BigUint::one() << b;
    let mut val = val.clone();

    // Since q < 2^T, no need to perform mod 2^T
    for _ in 0..num_limbs {
        result.push(F::from(&val % &b_pow));
        val /= &b_pow;
    }
    result
}

/// Convert an element in the emulated field to a list of native field elements.
pub fn from_emulated_field<E, F>(val: E) -> Vec<F>
where
    E: EmulationConfig<F>,
    F: PrimeField,
{
    biguint_to_limbs(&val.into(), E::B, E::NUM_LIMBS)
}

/// Inverse conversion of the [`from_emulated_field`]
pub fn to_emulated_field<E, F>(vals: &[F]) -> Result<E, CircuitError>
where
    E: EmulationConfig<F>,
    F: PrimeField,
{
    if vals.len() != E::NUM_LIMBS {
        return Err(CircuitError::FieldAlgebraError(
            "Malformed structure for emulated field element conversion.".to_string(),
        ));
    }
    let b_pow = BigUint::one() << E::B;
    Ok(E::from(
        vals.iter().rfold(BigUint::zero(), |result, &val| {
            result * &b_pow + <F as Into<BigUint>>::into(val)
        }),
    ))
}

/// The variable represents an element in the emulated field.
/// The limbs (elements of F) represent the emulated field element
/// λ via λ = Σ_i 2^{i * E::B} * a_i, with each a_i < 2^{E::B}.
///
/// Note this representation of λ is not unique. Generally this is OK.
/// Whenever we require it to be the canonical representation of λ
/// (for example, when taking modulo |F|) is when we use
/// `enforce_valid_emulated_var`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmulatedVariable<E: PrimeField>(pub(crate) Vec<Variable>, pub PhantomData<E>);

impl<E: PrimeField> EmulatedVariable<E> {
    /// Return the list of variables that simulate the field element
    pub fn to_vec(&self) -> Vec<Variable> {
        self.0.clone()
    }
}

impl<F: PrimeField> PlonkCircuit<F> {
    /// Returns zero as an emulated constant
    pub fn emulated_zero<E: EmulationConfig<F>>(&self) -> EmulatedVariable<E> {
        EmulatedVariable::<E>(
            (0..E::NUM_LIMBS)
                .map(|_| self.zero())
                .collect::<Vec<Variable>>(),
            PhantomData,
        )
    }

    /// Returns one as an emulated constant
    pub fn emulated_one<E: EmulationConfig<F>>(&self) -> EmulatedVariable<E> {
        let mut one = self.emulated_zero();
        one.0[0] = self.one();
        one
    }

    /*/// Checks if an `EmulatedVariable` is of the correct form.
    /// This involves checking that the integer represented is less
    /// than `E::MODULUS`. We use this sparingly as it is rather costly.
    ///
    /// Note this does not check all of the limbs are within the allowed
    /// bit length. This is done with `create_emulated_variable`.
    fn enforce_valid_emulated_var<E: EmulationConfig<F>>(
        &mut self,
        var: &EmulatedVariable<E>,
    ) -> Result<(), CircuitError> {
        let modulus: BigUint = E::MODULUS.into();
        let modulus_limbs = biguint_to_limbs::<F>(&modulus, E::B, E::NUM_LIMBS);
        let mut eq_vars = Vec::<BoolVar>::new();
        let mut lt_vars = Vec::<BoolVar>::new();
        for (&var, &constant) in var.0.iter().zip(modulus_limbs.iter()) {
            let diff_var = self.add_constant(var, &-constant)?;
            let eq_var = self.is_zero(diff_var)?;
            eq_vars.push(eq_var);

            let lt_var = self.is_lt_constant(var, constant)?;
            lt_vars.push(lt_var);
        }
        // `res_var` records if the variable is less than `E::MODULUS`.
        let mut res_var = BoolVar(self.zero());
        for (&eq_var, &lt_var) in eq_vars.iter().zip(lt_vars.iter()) {
            res_var = BoolVar(self.conditional_select(eq_var, lt_var.into(), res_var.into())?);
        }

        self.enforce_true(res_var.0)
    }*/

    /// Return the witness point for the circuit
    pub fn emulated_witness<E: EmulationConfig<F>>(
        &self,
        var: &EmulatedVariable<E>,
    ) -> Result<E, CircuitError> {
        let values = var
            .0
            .iter()
            .map(|&v| self.witness(v))
            .collect::<Result<Vec<_>, CircuitError>>()?;
        to_emulated_field(&values)
    }

    /// Add an emulated variable
    pub fn create_emulated_variable<E: EmulationConfig<F>>(
        &mut self,
        val: E,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        let var = self.create_emulated_variable_unchecked(val)?;
        // We constrain the limbs as tightly as we can.
        for (i, v) in var.0.iter().enumerate() {
            match i.cmp(&(E::MODULUS_BIT_SIZE as usize / E::B)) {
                ark_std::cmp::Ordering::Less => {
                    self.enforce_in_range(*v, E::B)?;
                },
                ark_std::cmp::Ordering::Equal => {
                    let final_bit_len = E::MODULUS_BIT_SIZE as usize % E::B;
                    self.enforce_in_range(*v, final_bit_len)?;
                },
                ark_std::cmp::Ordering::Greater => {
                    self.enforce_constant(*v, F::zero())?;
                },
            }
        }
        Ok(var)
    }

    /// Add an emulated variable without enforcing the limb size checks
    pub fn create_emulated_variable_unchecked<E: EmulationConfig<F>>(
        &mut self,
        val: E,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        Ok(EmulatedVariable::<E>(
            from_emulated_field(val)
                .into_iter()
                .map(|v| self.create_variable(v))
                .collect::<Result<Vec<_>, CircuitError>>()?,
            PhantomData,
        ))
    }

    /// Add a constant emulated variable
    pub fn create_constant_emulated_variable<E: EmulationConfig<F>>(
        &mut self,
        val: E,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        Ok(EmulatedVariable::<E>(
            from_emulated_field(val)
                .into_iter()
                .map(|v| self.create_constant_variable(v))
                .collect::<Result<Vec<_>, CircuitError>>()?,
            PhantomData,
        ))
    }

    /// Add a public emulated variable
    pub fn create_public_emulated_variable<E: EmulationConfig<F>>(
        &mut self,
        val: E,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        Ok(EmulatedVariable::<E>(
            from_emulated_field(val)
                .into_iter()
                .map(|v| self.create_public_variable(v))
                .collect::<Result<Vec<_>, CircuitError>>()?,
            PhantomData,
        ))
    }

    /// Converts an emulated variable to a form that aligns with a transcript.
    /// If the transcript is defined over the field `F` we work out the byte length
    /// of the modulus of `F` (rounded down) and then convert the underlying emulated element from `E` into
    /// this many elements of `F` and constrain them to be equal to the variables used in the `EmuatedVariable`.
    ///
    /// NOTE: This function is currently only working for Bn254 fields, needs to be generalized.
    pub fn convert_for_transcript<E: EmulationConfig<F>>(
        &mut self,
        var: &EmulatedVariable<E>,
    ) -> Result<Vec<Variable>, CircuitError> {
        let field_limb_bits = ((F::MODULUS_BIT_SIZE as usize - 1) / 8) * 8;

        // Make variables for the field element limbs.
        let element = self.emulated_witness(var)?;
        let mut writer = Vec::new();
        element.serialize_compressed(&mut writer).map_err(|_| {
            CircuitError::InternalError("Failed to serialize emulated element".to_string())
        })?;
        let field_limbs = bytes_to_field_elements::<_, F>(writer);
        // We discard the first element because this just tells us how many field element limbs there are.
        let field_limb_vars = field_limbs[1..]
            .iter()
            .map(|&v| self.create_variable(v))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;

        // We run through the `field_limb_vars` and constrain them to be decomposed into the appropriate `Variable`s of `var`.
        // Of course some of the `Variable`s must be decomposed in two as they straddle two different `field_limb_vars`.
        // `emulated_index` records the index of the `Variable` in `var` that we are currently working on.
        let mut emulated_index = 0;
        // `carry_var` records the `Variable` that carries over from the previous `field_limb_var`.
        let mut carry_var = self.zero();
        // `carry_bit_size` records the number of bits that are carried over from the previous `field_limb_var`.
        let mut carry_bit_size = 0;
        for (i, field_var) in field_limb_vars.iter().enumerate() {
            // `emu_var_vec` will store the `Variable`s that our `field_var` is decomposed into.
            let mut emu_var_vec = Vec::<Variable>::new();
            // `coeff_vec` will store the coefficients of the `Variables`s in `emu_var_vec`.
            let mut coeff_vec = Vec::<F>::new();
            let mut coeff = F::one();
            // if `carry_bit_size > 0`, there's a `Variable` to carry.
            if carry_bit_size > 0 {
                emu_var_vec.push(carry_var);
                coeff_vec.push(coeff);
                coeff *= F::from(2u32).pow([carry_bit_size as u64]);
            }
            // We keep adding the emulated variables until they no longer fit in the field limb.
            while (emulated_index + 1) * E::B <= (i + 1) * field_limb_bits
                && emulated_index < E::NUM_LIMBS
            {
                emu_var_vec.push(var.0[emulated_index]);
                coeff_vec.push(coeff);
                coeff *= F::from(2u32).pow([E::B as u64]);
                emulated_index += 1;
            }
            // If we need to add part of the next `Variable` we need to decompose it and constrain everything appropriately.
            if emulated_index * E::B < (i + 1) * field_limb_bits && emulated_index < E::NUM_LIMBS {
                let power = ((i + 1) * field_limb_bits - emulated_index * E::B) as u32;
                let val: BigUint = self.witness(var.0[emulated_index])?.into_bigint().into();

                // `lower_var` is the lower part of the `Variable` that straddles two `field_limb_vars`.
                // This must be included in the decomposition of the current `field_var`.
                let lower_val = val.clone() % (BigUint::from(2u32).pow(power));
                let lower_var = self.create_variable(F::from(lower_val))?;
                emu_var_vec.push(lower_var);
                self.enforce_in_range(lower_var, power as usize)?;
                coeff_vec.push(coeff);

                let carry_val = val / (BigUint::from(2u32).pow(power));
                carry_var = self.create_variable(F::from(carry_val))?;
                self.enforce_in_range(carry_var, E::B - power as usize)?;

                // `lower_var` and `carry_var` must decompose the `Variable` that straddles two `field_limb_vars`.
                self.lin_comb_gate(
                    &[F::one(), F::from(2u8).pow([power as u64])],
                    &F::zero(),
                    &[lower_var, carry_var],
                    &var.0[emulated_index],
                )?;
                carry_bit_size = E::B - power as usize;
                emulated_index += 1;
            } else {
                // In this situation there is no carry.
                carry_bit_size = 0;
            }
            self.lin_comb_gate(&coeff_vec, &F::zero(), &emu_var_vec, field_var)?;
            if emulated_index == E::NUM_LIMBS {
                break;
            }
        }
        Ok(field_limb_vars)
    }

    /// Constrain that a*b=c in the emulated field.
    /// Checking that a * b - k * E::MODULUS = c.
    /// This function doesn't perform emulated variable validaty check on the
    /// input a, b and c. We assume at least that `a`, `b` and `c` have bounded limbs.
    pub fn emulated_mul_gate<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: &EmulatedVariable<E>,
        c: &EmulatedVariable<E>,
    ) -> Result<(), CircuitError> {
        self.check_vars_bound(&a.0)?;
        self.check_vars_bound(&b.0)?;
        self.check_vars_bound(&c.0)?;

        let val_a: BigUint = self.emulated_witness(a)?.into();
        let val_b: BigUint = self.emulated_witness(b)?.into();
        let val_k = E::from(&val_a * &val_b / E::MODULUS.into());
        let k = self.create_emulated_variable(val_k)?;
        let a_limbs = biguint_to_limbs::<F>(&val_a, E::B, E::NUM_LIMBS);
        let b_limbs = biguint_to_limbs::<F>(&val_b, E::B, E::NUM_LIMBS);
        let k_limbs = from_emulated_field(val_k);
        let b_pow = F::from(2u32).pow([E::B as u64]);
        let val_expected = E::from(val_a) * E::from(val_b);
        let val_expected_limbs = from_emulated_field(val_expected);

        let neg_modulus = biguint_to_limbs::<F>(
            &(BigUint::from(2u32).pow(E::T as u32) - E::MODULUS.into()),
            E::B,
            E::NUM_LIMBS,
        );

        // enforcing a * b - k * E::MODULUS = c mod 2^t

        // first compare the first limb
        let mut val_carry_out =
            (a_limbs[0] * b_limbs[0] + k_limbs[0] * neg_modulus[0] - val_expected_limbs[0]) / b_pow;
        let mut carry_out = self.create_variable(val_carry_out)?;
        // checking that the carry_out has at most [`E::B`] + 1 bits
        self.enforce_in_range(carry_out, E::B + 1)?;
        // enforcing that a0 * b0 - k0 * modulus[0] - carry_out * 2^E::B = c0
        self.quad_poly_gate(
            &[a.0[0], b.0[0], k.0[0], carry_out, c.0[0]],
            &[F::zero(), F::zero(), neg_modulus[0], -b_pow],
            &[F::one(), F::zero()],
            F::one(),
            F::zero(),
        )?;

        for i in 1..E::NUM_LIMBS {
            // compare the i-th limb

            // calculate the next carry out
            let val_next_carry_out = ((0..=i)
                .map(|j| k_limbs[j] * neg_modulus[i - j] + a_limbs[j] * b_limbs[i - j])
                .sum::<F>()
                + val_carry_out
                - val_expected_limbs[i])
                / b_pow;
            let next_carry_out = self.create_variable(val_next_carry_out)?;

            // range checking for this carry out.
            // let a = 2^B - 1. The maximum possible value of `next_carry_out` is ((i + 1) *
            // 2 * a^2 + a) / 2^B.
            let num_vals = 2u64 * (i as u64) + 2;
            let log_num_vals = (u64::BITS - num_vals.leading_zeros()) as usize;
            self.enforce_in_range(next_carry_out, E::B + log_num_vals)?;

            // k * E::MODULUS part, waiting for summation
            let mut stack = (0..=i)
                .map(|j| (k.0[j], neg_modulus[i - j]))
                .collect::<Vec<_>>();
            // carry out from last limb
            stack.push((carry_out, F::one()));
            stack.push((next_carry_out, -b_pow));

            // part of the summation \sum_j a_i * b_{i-j}
            for j in (0..i).step_by(2) {
                let t = self.mul_add(
                    &[a.0[j], b.0[i - j], a.0[j + 1], b.0[i - j - 1]],
                    &[F::one(), F::one()],
                )?;
                stack.push((t, F::one()));
            }

            // last item of the summation \sum_j a_i * b_{i-j}
            if i % 2 == 0 {
                let t1 = stack.pop().unwrap();
                let t2 = stack.pop().unwrap();
                let t = self.gen_quad_poly(
                    &[a.0[i], b.0[0], t1.0, t2.0],
                    &[F::zero(), F::zero(), t1.1, t2.1],
                    &[F::one(), F::zero()],
                    F::zero(),
                )?;
                stack.push((t, F::one()));
            }

            // linear combination of all items in the stack
            while stack.len() > 4 {
                let t1 = stack.pop().unwrap();
                let t2 = stack.pop().unwrap();
                let t3 = stack.pop().unwrap();
                let t4 = stack.pop().unwrap();
                let t = self.lc(&[t1.0, t2.0, t3.0, t4.0], &[t1.1, t2.1, t3.1, t4.1])?;
                stack.push((t, F::one()));
            }
            let t1 = stack.pop().unwrap_or((self.zero(), F::zero()));
            let t2 = stack.pop().unwrap_or((self.zero(), F::zero()));
            let t3 = stack.pop().unwrap_or((self.zero(), F::zero()));
            let t4 = stack.pop().unwrap_or((self.zero(), F::zero()));

            // checking that the summation equals to i-th limb of c
            self.lc_gate(&[t1.0, t2.0, t3.0, t4.0, c.0[i]], &[t1.1, t2.1, t3.1, t4.1])?;

            val_carry_out = val_next_carry_out;
            carry_out = next_carry_out;
        }

        // enforcing a * b - k * E::MODULUS = c mod F::MODULUS
        let a_mod = self.mod_to_native_field(a)?;
        let b_mod = self.mod_to_native_field(b)?;
        let k_mod = self.mod_to_native_field(&k)?;
        let c_mod = self.mod_to_native_field(c)?;
        let e_mod_f = F::from(E::MODULUS.into());
        self.quad_poly_gate(
            &[a_mod, b_mod, k_mod, self.zero(), c_mod],
            &[F::zero(), F::zero(), -e_mod_f, F::zero()],
            &[F::one(), F::zero()],
            F::one(),
            F::zero(),
        )?;

        Ok(())
    }

    /// Return an [`EmulatedVariable`] which equals a*b + c.
    pub fn emulated_mul_add<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: &EmulatedVariable<E>,
        c: &EmulatedVariable<E>,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        let ab_var = self.emulated_mul(a, b)?;
        self.emulated_add(&ab_var, c)
    }

    /// Return an [`EmulatedVariable`] which equals to a*b.
    pub fn emulated_mul<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: &EmulatedVariable<E>,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        let c = self.emulated_witness(a)? * self.emulated_witness(b)?;
        let c = self.create_emulated_variable(c)?;
        self.emulated_mul_gate(a, b, &c)?;
        Ok(c)
    }

    /// Constrain that a*b=c in the emulated field for a constant b.
    /// This function doesn't perform emulated variable validaty check on the
    /// input a and c. We assume that they are already performed elsewhere.
    pub fn emulated_mul_constant_gate<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: E,
        c: &EmulatedVariable<E>,
    ) -> Result<(), CircuitError> {
        self.check_vars_bound(&a.0)?;
        self.check_vars_bound(&c.0)?;

        let val_a: BigUint = self.emulated_witness(a)?.into();
        let val_b: BigUint = b.into();
        let val_k = E::from(&val_a * &val_b / E::MODULUS.into());
        let k = self.create_emulated_variable(val_k)?;
        let a_limbs = biguint_to_limbs::<F>(&val_a, E::B, E::NUM_LIMBS);
        let b_limbs = biguint_to_limbs::<F>(&val_b, E::B, E::NUM_LIMBS);
        let k_limbs = from_emulated_field(val_k);
        let b_pow = F::from(2u32).pow([E::B as u64]);
        let val_expected = E::from(val_a) * b;
        let val_expected_limbs = from_emulated_field(val_expected);

        let neg_modulus = biguint_to_limbs::<F>(
            &(BigUint::from(2u32).pow(E::T as u32) - E::MODULUS.into()),
            E::B,
            E::NUM_LIMBS,
        );

        // enforcing a * b - k * E::MODULUS = c mod 2^t

        // first compare the first limb
        let mut val_carry_out =
            (a_limbs[0] * b_limbs[0] + k_limbs[0] * neg_modulus[0] - val_expected_limbs[0]) / b_pow;
        let mut carry_out = self.create_variable(val_carry_out)?;
        // checking that the carry_out has at most [`E::B`] bits
        self.enforce_in_range(carry_out, E::B + 1)?;
        // enforcing that a0 * b0 - k0 * modulus[0] - carry_out * 2^E::B = c0
        self.lc_gate(
            &[a.0[0], k.0[0], carry_out, self.zero(), c.0[0]],
            &[b_limbs[0], neg_modulus[0], -b_pow, F::zero()],
        )?;

        for i in 1..E::NUM_LIMBS {
            // compare the i-th limb

            // calculate the next carry out
            let val_next_carry_out = ((0..=i)
                .map(|j| k_limbs[j] * neg_modulus[i - j] + a_limbs[j] * b_limbs[i - j])
                .sum::<F>()
                + val_carry_out
                - val_expected_limbs[i])
                / b_pow;
            let next_carry_out = self.create_variable(val_next_carry_out)?;

            // range checking for this carry out.
            let num_vals = 2u64 * (i as u64) + 2;
            let log_num_vals = (u64::BITS - num_vals.leading_zeros()) as usize;
            self.enforce_in_range(next_carry_out, E::B + log_num_vals)?;

            // k * E::MODULUS part, waiting for summation
            let mut stack = (0..=i)
                .map(|j| (k.0[j], neg_modulus[i - j]))
                .collect::<Vec<_>>();
            // a * b part
            (0..=i).for_each(|j| stack.push((a.0[j], b_limbs[i - j])));
            // carry out from last limb
            stack.push((carry_out, F::one()));
            stack.push((next_carry_out, -b_pow));

            // linear combination of all items in the stack
            while stack.len() > 4 {
                let t1 = stack.pop().unwrap();
                let t2 = stack.pop().unwrap();
                let t3 = stack.pop().unwrap();
                let t4 = stack.pop().unwrap();
                let t = self.lc(&[t1.0, t2.0, t3.0, t4.0], &[t1.1, t2.1, t3.1, t4.1])?;
                stack.push((t, F::one()));
            }
            let t1 = stack.pop().unwrap_or((self.zero(), F::zero()));
            let t2 = stack.pop().unwrap_or((self.zero(), F::zero()));
            let t3 = stack.pop().unwrap_or((self.zero(), F::zero()));
            let t4 = stack.pop().unwrap_or((self.zero(), F::zero()));

            // checking that the summation equals to i-th limb of c
            self.lc_gate(&[t1.0, t2.0, t3.0, t4.0, c.0[i]], &[t1.1, t2.1, t3.1, t4.1])?;

            val_carry_out = val_next_carry_out;
            carry_out = next_carry_out;
        }

        // enforcing a * b - k * E::MODULUS = c mod F::MODULUS
        let a_mod = self.mod_to_native_field(a)?;
        let b_mod = F::from(val_b);
        let k_mod = self.mod_to_native_field(&k)?;
        let c_mod = self.mod_to_native_field(c)?;
        let e_mod_f = F::from(E::MODULUS.into());
        self.lc_gate(
            &[a_mod, k_mod, self.zero(), self.zero(), c_mod],
            &[b_mod, -e_mod_f, F::zero(), F::zero()],
        )?;

        Ok(())
    }

    /// Return an [`EmulatedVariable`] which equals to a*b.
    pub fn emulated_mul_constant<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: E,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        let c = self.emulated_witness(a)? * b;
        let c = self.create_emulated_variable(c)?;
        self.emulated_mul_constant_gate(a, b, &c)?;
        Ok(c)
    }

    /// Constrain that a+b=c in the emulated field.
    /// Checking whether a + b = k * E::MODULUS + c
    /// This function doesn't perform emulated variable validaty check on the
    /// input a, b and c. We assume at least that `a`, `b` and `c` have bounded limbs.
    pub fn emulated_add_gate<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: &EmulatedVariable<E>,
        c: &EmulatedVariable<E>,
    ) -> Result<(), CircuitError> {
        self.check_vars_bound(&a.0)?;
        self.check_vars_bound(&b.0)?;
        self.check_vars_bound(&c.0)?;

        let val_a: BigUint = self.emulated_witness(a)?.into();
        let val_b: BigUint = self.emulated_witness(b)?.into();
        let modulus: BigUint = E::MODULUS.into();
        let b_pow = BigUint::from(2u32).pow(E::B as u32);
        let add_no_mod = &val_a + &val_b;
        let k = if add_no_mod >= modulus { 1u32 } else { 0u32 };
        let var_k = self.create_boolean_variable(add_no_mod >= modulus)?.0;
        let modulus_limbs = biguint_to_limbs::<F>(&modulus, E::B, E::NUM_LIMBS);

        let add_no_mod_limbs = biguint_to_limbs::<F>(&add_no_mod, E::B, E::NUM_LIMBS)
            .into_iter()
            .map(|val| self.create_variable(val))
            .collect::<Result<Vec<_>, CircuitError>>()?;

        // Checking whether a + b = add_no_mod_limbs
        let mut carry_out = self.zero();
        for (n, (a, b, c)) in izip!(&a.0, &b.0, &add_no_mod_limbs).enumerate() {
            let next_carry_out =
                F::from(<F as Into<BigUint>>::into(self.witness(*a)? + self.witness(*b)?) / &b_pow);
            // If we are on the final limb, there should be no next_carry.
            let next_carry_out = if n == E::NUM_LIMBS - 1 {
                self.zero()
            } else {
                let next_carry = self.create_variable(next_carry_out)?;
                self.enforce_bool(next_carry)?;
                next_carry
            };

            let wires = [*a, *b, carry_out, next_carry_out, *c];
            let coeffs = [F::one(), F::one(), F::one(), -F::from(b_pow.clone())];
            self.lc_gate(&wires, &coeffs)?;
            carry_out = next_carry_out;

            self.enforce_in_range(*c, E::B)?;
        }

        // Checking whether k * E::MODULUS + c = add_no_mod_limbs
        carry_out = self.zero();
        for (n, (a, b, c)) in izip!(modulus_limbs, &c.0, &add_no_mod_limbs).enumerate() {
            let next_carry_out =
                F::from(<F as Into<BigUint>>::into(a * F::from(k) + self.witness(*b)?) / &b_pow);
            // If we are on the final limb, there should be no next_carry.
            let next_carry_out = if n == E::NUM_LIMBS - 1 {
                self.zero()
            } else {
                let next_carry = self.create_variable(next_carry_out)?;
                self.enforce_bool(next_carry)?;
                next_carry
            };

            let wires = [var_k, *b, carry_out, next_carry_out, *c];
            let coeffs = [a, F::one(), F::one(), -F::from(b_pow.clone())];
            self.lc_gate(&wires, &coeffs)?;
            carry_out = next_carry_out;
        }
        Ok(())
    }

    /// Return an [`EmulatedVariable`] which equals to a+b.
    pub fn emulated_add<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: &EmulatedVariable<E>,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        let c = self.emulated_witness(a)? + self.emulated_witness(b)?;
        let c = self.create_emulated_variable(c)?;
        self.emulated_add_gate(a, b, &c)?;
        Ok(c)
    }

    /// Return an [`EmulatedVariable`] which equals to a-b.
    pub fn emulated_sub<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: &EmulatedVariable<E>,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        let c = self.emulated_witness(a)? - self.emulated_witness(b)?;
        let c = self.create_emulated_variable(c)?;
        self.emulated_add_gate(&c, b, a)?;
        Ok(c)
    }

    /// Constrain that a+b=c in the emulated field.
    /// This function doesn't perform emulated variable validaty check on the
    /// input a and c. We assume that they are already performed elsewhere.
    pub fn emulated_add_constant_gate<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: E,
        c: &EmulatedVariable<E>,
    ) -> Result<(), CircuitError> {
        self.check_vars_bound(&a.0)?;
        self.check_vars_bound(&c.0)?;

        let val_a: BigUint = self.emulated_witness(a)?.into();
        let val_b: BigUint = b.into();
        let q: BigUint = E::MODULUS.into();
        let b_pow = BigUint::from(2u32).pow(E::B as u32);
        let add_no_mod = &val_a + &val_b;
        let k = if add_no_mod >= q { 1u32 } else { 0u32 };
        let var_k = self.create_boolean_variable(add_no_mod >= q)?.0;
        let q_limbs = biguint_to_limbs::<F>(&q, E::B, E::NUM_LIMBS);
        let b_limbs = biguint_to_limbs::<F>(&val_b, E::B, E::NUM_LIMBS);

        let add_no_mod_limbs = biguint_to_limbs::<F>(&add_no_mod, E::B, E::NUM_LIMBS)
            .into_iter()
            .map(|val| self.create_variable(val))
            .collect::<Result<Vec<_>, CircuitError>>()?;

        // Checking whether a + b = add_no_mod_limbs
        let mut carry_out = self.zero();
        for (a, b, c) in izip!(&a.0, b_limbs, &add_no_mod_limbs) {
            let next_carry_out =
                F::from(<F as Into<BigUint>>::into(self.witness(*a)? + b) / &b_pow);
            let next_carry_out = self.create_variable(next_carry_out)?;
            self.enforce_bool(next_carry_out)?;

            let wires = [*a, self.one(), carry_out, next_carry_out, *c];
            let coeffs = [F::one(), b, F::one(), -F::from(b_pow.clone())];
            self.lc_gate(&wires, &coeffs)?;
            carry_out = next_carry_out;

            self.enforce_in_range(*c, E::B)?;
        }

        // Checking whether k * q + c = add_no_mod_limbs
        carry_out = self.zero();
        for (a, b, c) in izip!(q_limbs, &c.0, &add_no_mod_limbs) {
            let next_carry_out =
                F::from(<F as Into<BigUint>>::into(a * F::from(k) + self.witness(*b)?) / &b_pow);
            let next_carry_out = self.create_variable(next_carry_out)?;
            self.enforce_bool(next_carry_out)?;

            let wires = [var_k, *b, carry_out, next_carry_out, *c];
            let coeffs = [a, F::one(), F::one(), -F::from(b_pow.clone())];
            self.lc_gate(&wires, &coeffs)?;
            carry_out = next_carry_out;
        }
        Ok(())
    }

    /// Return an [`EmulatedVariable`] which equals to a + b where b is a
    /// constant.
    pub fn emulated_add_constant<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: E,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        let c = self.emulated_witness(a)? + b;
        let c = self.create_emulated_variable(c)?;
        self.emulated_add_constant_gate(a, b, &c)?;
        Ok(c)
    }

    /// Return an [`EmulatedVariable`] which equals to a - b where b is a
    /// constant.
    pub fn emulated_sub_constant<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: E,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        let c = self.emulated_witness(a)? - b;
        let c = self.create_emulated_variable(c)?;
        self.emulated_add_constant_gate(&c, b, a)?;
        Ok(c)
    }
    /// Obtain an emulated variable of the conditional selection from 2 emulated
    /// variables. `b` is a boolean variable that indicates selection of P_b
    /// from (P0, P1).
    /// Return error if invalid input parameters are provided.
    pub fn conditional_select_emulated<E: EmulationConfig<F>>(
        &mut self,
        b: BoolVar,
        p0: &EmulatedVariable<E>,
        p1: &EmulatedVariable<E>,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        self.check_var_bound(b.into())?;
        self.check_vars_bound(&p0.0[..])?;
        self.check_vars_bound(&p1.0[..])?;

        let mut vals = vec![];
        for (&x_0, &x_1) in p0.0.iter().zip(p1.0.iter()) {
            let selected = self.conditional_select(b, x_0, x_1)?;
            vals.push(selected);
        }

        Ok(EmulatedVariable::<E>(vals, PhantomData::<E>))
    }

    /// Constrain two emulated variables to be the same.
    /// Return error if the input variables are invalid.
    pub fn enforce_emulated_var_equal<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: &EmulatedVariable<E>,
    ) -> Result<(), CircuitError> {
        self.check_vars_bound(&a.0[..])?;
        self.check_vars_bound(&b.0[..])?;
        for (&a, &b) in a.0.iter().zip(b.0.iter()) {
            self.enforce_equal(a, b)?;
        }
        Ok(())
    }

    /// Obtain a bool variable representing whether two input emulated variables
    /// are equal. Return error if variables are invalid.
    pub fn is_emulated_var_equal<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: &EmulatedVariable<E>,
    ) -> Result<BoolVar, CircuitError> {
        self.check_vars_bound(&a.0[..])?;
        self.check_vars_bound(&b.0[..])?;
        let c = self.emulated_sub(a, b)?;
        self.is_emulated_var_zero(&c)
    }

    /// Obtain a bool variable representing whether the input emulated variable
    /// is zero. We need to accept an `EmulatedVariable` representing the value
    /// `E::MODULUS` as zero. Return error if variables are invalid.
    pub fn is_emulated_var_zero<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
    ) -> Result<BoolVar, CircuitError> {
        self.check_vars_bound(&a.0[..])?;
        let zero_bools =
            a.0.iter()
                .map(|&a| self.is_zero(a))
                .collect::<Result<Vec<_>, _>>()?;
        let zero_bool = self.logic_and_all(&zero_bools)?;

        let modulus: BigUint = E::MODULUS.into();
        let modulus_limbs = biguint_to_limbs::<F>(&modulus, E::B, E::NUM_LIMBS);
        let e_bools =
            a.0.iter()
                .zip(modulus_limbs.iter())
                .map(|(&a, &e)| {
                    let b = self.add_constant(a, &-e)?;
                    self.is_zero(b)
                })
                .collect::<Result<Vec<_>, _>>()?;
        let e_bool = self.logic_and_all(&e_bools)?;

        self.logic_or(zero_bool, e_bool)
    }

    /// This function attempts to replicate the [`bytes_to_field_elements`]
    /// function used in transcripts. We do not need to constrain anything
    /// as the hash wil be incorrect if the field elem was not correct.
    pub fn emulated_var_to_field_vars<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
    ) -> Result<Vec<Variable>, CircuitError> {
        self.check_vars_bound(&a.0[..])?;
        let a_value = self.emulated_witness(a)?;

        let a_bytes = to_bytes!(&a_value).map_err(|_| {
            CircuitError::ParameterError(
                "Could not convert emulated field element to bytes".to_string(),
            )
        })?;
        let a_bytes_field_elems = bytes_to_field_elements::<_, F>(a_bytes);
        let a_field_elems_vars = a_bytes_field_elems
            .iter()
            .map(|elem| self.create_variable(*elem))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        Ok(a_field_elems_vars)
    }

    /// Given an emulated field element `a`, return `a mod F::MODULUS` in the
    /// native field. This does not ensure that the `EmulatedVariable` is represents
    /// an integer less than `E::MODULUS`. Usually this check is required, in which
    /// case use `mod_to_native_field`.
    pub fn mod_to_native_field<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
    ) -> Result<Variable, CircuitError> {
        let b_pow = F::from(2u32).pow([E::B as u64]);
        let double_b_pow = b_pow * b_pow;
        let triple_b_pow = double_b_pow * b_pow;
        let zero = self.zero();
        let a0 = a.0.first().unwrap_or(&zero);
        let a1 = a.0.get(1).unwrap_or(&zero);
        let a2 = a.0.get(2).unwrap_or(&zero);
        let a3 = a.0.get(3).unwrap_or(&zero);

        let mut result = self.lc(
            &[*a0, *a1, *a2, *a3],
            &[F::one(), b_pow, double_b_pow, triple_b_pow],
        )?;

        if E::NUM_LIMBS > 4 {
            let mut cur_pow = triple_b_pow * b_pow;
            for i in (4..E::NUM_LIMBS).step_by(3) {
                let a0 = a.0.get(i).unwrap_or(&zero);
                let a1 = a.0.get(i + 1).unwrap_or(&zero);
                let a2 = a.0.get(i + 2).unwrap_or(&zero);
                result = self.lc(
                    &[result, *a0, *a1, *a2],
                    &[F::one(), cur_pow, cur_pow * b_pow, cur_pow * double_b_pow],
                )?;
                cur_pow *= triple_b_pow;
            }
        }
        Ok(result)
    }

    /// Opposite of `mod_to_native_field`
    pub fn to_emulated_variable<E: EmulationConfig<F>>(
        &mut self,
        a: Variable,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        self.check_var_bound(a)?;
        let a_val = self.witness(a)?;

        let emulated = E::from_le_bytes_mod_order(&a_val.into_bigint().to_bytes_le());
        let emul_var = self.create_emulated_variable(emulated)?;
        let out = self.mod_to_native_field(&emul_var)?;
        self.enforce_equal(a, out)?;
        Ok(emul_var)
    }

    /// Obtain the `bit_len`-long binary representation of emulated variable `a`
    /// Return a list of variables [b0, ..., b_`bit_len`] which is the binary
    /// representation of `a`.
    /// Return error if the `a` is not the range of [0, 2^`bit_len`).
    pub fn emulated_unpack<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        bit_len: usize,
    ) -> Result<Vec<BoolVar>, CircuitError> {
        if bit_len < E::MODULUS_BIT_SIZE as usize
            && self.emulated_witness(a)? >= E::from(2u32).pow([bit_len as u64])
        {
            return Err(CircuitError::ParameterError(
                "Failed to unpack variable to a range of smaller than 2^bit_len".to_string(),
            ));
        }
        self.emulated_range_gate_internal(a, bit_len)
    }

    // internal of a range check gate
    pub(crate) fn emulated_range_gate_internal<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        bit_len: usize,
    ) -> Result<Vec<BoolVar>, CircuitError> {
        self.check_vars_bound(&a.0[..])?;
        if bit_len == 0 {
            return Err(CircuitError::ParameterError(
                "Only allows positive bit length for range upper bound".to_string(),
            ));
        }

        let a_bits_le: Vec<bool> = self.emulated_witness(a)?.into_bigint().to_bits_le();
        if bit_len > a_bits_le.len() {
            return Err(CircuitError::ParameterError(format!(
                "Maximum field bit size: {}, requested range upper bound bit len: {}",
                a_bits_le.len(),
                bit_len
            )));
        }
        // convert to variable in the circuit from the vector of boolean as binary
        // representation
        let a_bits_le: Vec<BoolVar> = a_bits_le
            .iter()
            .take(bit_len) // since little-endian, truncate would remove MSBs
            .map(|&b| {
                self.create_boolean_variable(b)
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;

        self.emulated_binary_decomposition_gate(a_bits_le.clone(), a)?;

        Ok(a_bits_le)
    }

    fn emulated_binary_decomposition_gate<E: EmulationConfig<F>>(
        &mut self,
        a_bits_le: Vec<BoolVar>,
        a: &EmulatedVariable<E>,
    ) -> Result<(), CircuitError> {
        let a_chunks_le: Vec<Variable> = a_bits_le.into_iter().map(|b| b.into()).collect();
        self.emulated_decomposition_gate(&a_chunks_le, a, 1)?;
        Ok(())
    }

    /// a general decomposition gate (not necessarily binary decomposition)
    /// where emulated variable `a` is enforced to be decomposed to `a_chunks_le` which consists
    /// of chunks (multiple bits) in little-endian order and each chunk \in [0,..,`2^c`)
    /// Note, the chunks are not constrained here. This must be done elsewhere.
    pub(crate) fn emulated_decomposition_gate<E: EmulationConfig<F>>(
        &mut self,
        a_chunks_le: &[Variable],
        a: &EmulatedVariable<E>,
        c: usize,
    ) -> Result<(), CircuitError> {
        // We run through the limbs of `a` and constrain them to be decomposed into the appropriate `Variable`s of `a_chunks_le`.
        // Of course some of the `Variable`s must be decomposed in two as they straddle two different limbs of `a`.
        let chunk_len = a_chunks_le.len();
        // `chunk_index` records the index of the `Variable` in `a_chunks_le` that we are currently working on.
        let mut chunk_index = 0;
        // `carry_var` records the `Variable` that carries over from the previous limb of `a`.
        let mut carry_var = self.zero();
        // `carry_bit_size` records the number of bits that are carried over from the previous limb of `a`.
        let mut carry_bit_size = 0;
        for (i, a_var) in a.0.iter().enumerate() {
            // `chunk_var_vec` will store the `Variable`s that our `a_var` is decomposed into.
            let mut chunk_var_vec = Vec::<Variable>::new();
            // `coeff_vec` will store the coefficients of the `Variables`s in `chunk_var_vec`.
            let mut coeff_vec = Vec::<F>::new();
            let mut coeff = F::one();
            // if `carry_bit_size > 0`, there's a `Variable` to carry.
            if carry_bit_size > 0 {
                chunk_var_vec.push(carry_var);
                coeff_vec.push(coeff);
                coeff *= F::from(2u32).pow([carry_bit_size as u64]);
            }
            // We keep adding the `a_chunk_le` variables until they no longer fit in the `Variable` of `a`.
            while (chunk_index + 1) * c <= (i + 1) * E::B && chunk_index < chunk_len {
                chunk_var_vec.push(a_chunks_le[chunk_index]);
                coeff_vec.push(coeff);
                coeff *= F::from(2u32).pow([c as u64]);
                chunk_index += 1;
            }
            // If we need to add part of the next `Variable` we need to decompose it and constrain everything appropriately.
            if (chunk_index * c) < (i + 1) * E::B && chunk_index < chunk_len {
                let power = ((i + 1) * E::B - chunk_index * c) as u32;
                let val: BigUint = self.witness(a_chunks_le[chunk_index])?.into_bigint().into();

                // `lower_var` is the lower part of the `Variable` that straddles two `Variable`s of `a`.
                // This must be included in the decomposition of the current `a_var`.
                let lower_val = val.clone() % (BigUint::from(2u32).pow(power));
                let lower_var = self.create_variable(F::from(lower_val))?;
                chunk_var_vec.push(lower_var);
                self.enforce_in_range(lower_var, power as usize)?;
                coeff_vec.push(coeff);

                let carry_val = val / (BigUint::from(2u32).pow(power));
                carry_var = self.create_variable(F::from(carry_val))?;
                self.enforce_in_range(carry_var, E::B - power as usize)?;

                // `lower_var` and `carry_var` must decompose the `Variable` that straddles two `Variable`s of `a`.
                self.lin_comb_gate(
                    &[F::one(), F::from(2u8).pow([power as u64])],
                    &F::zero(),
                    &[lower_var, carry_var],
                    &a_chunks_le[chunk_index],
                )?;
                self.check_circuit_satisfiability(&[])?;
                carry_bit_size = c - power as usize;
                chunk_index += 1;
            } else {
                // In this situation there is no carry.
                carry_bit_size = 0;
            }
            self.lin_comb_gate(&coeff_vec, &F::zero(), &chunk_var_vec, a_var)?;
            self.check_circuit_satisfiability(&[])?;
            if chunk_index == chunk_len {
                break;
            }
        }
        // We must constrain all the limbs of `a` not dealt with above to be zero.
        let start_limb = (c * chunk_len - 1) / E::B + 1;
        for i in start_limb..E::NUM_LIMBS {
            self.enforce_constant(a.0[i], F::zero())?;
        }
        Ok(())
    }

    /// Constrain a variable to be within the [0, 2^`bit_len`) range
    /// Return error if the variable is invalid.
    pub fn emulated_enforce_in_range<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        bit_len: usize,
    ) -> Result<(), CircuitError> {
        self.emulated_range_gate_internal(a, bit_len)?;

        Ok(())
    }
}

impl EmulationConfig<ark_bn254::Fr> for ark_bls12_377::Fq {
    const T: usize = 500;
    const B: usize = 100;
    const NUM_LIMBS: usize = 5;
}

impl EmulationConfig<ark_bn254::Fr> for ark_bn254::Fq {
    const T: usize = 288;
    const B: usize = 96;
    const NUM_LIMBS: usize = 3;
}

impl EmulationConfig<ark_bn254::Fq> for ark_bn254::Fr {
    const T: usize = 288;
    const B: usize = 96;
    const NUM_LIMBS: usize = 3;
}

impl EmulationConfig<ark_bls12_377::Fq> for ark_bls12_377::Fr {
    const T: usize = 300;
    const B: usize = 100;
    const NUM_LIMBS: usize = 3;
}

impl EmulationConfig<ark_bls12_381::Fq> for ark_bls12_381::Fr {
    const T: usize = 300;
    const B: usize = 100;
    const NUM_LIMBS: usize = 3;
}

impl EmulationConfig<ark_bw6_761::Fq> for ark_bw6_761::Fr {
    const T: usize = 400;
    const B: usize = 100;
    const NUM_LIMBS: usize = 4;
}

impl<F: PrimeField> EmulationConfig<F> for F {
    const T: usize = F::MODULUS_BIT_SIZE as usize;
    const B: usize = F::MODULUS_BIT_SIZE as usize;
    const NUM_LIMBS: usize = 1;
}

#[cfg(test)]
mod tests {
    use super::EmulationConfig;
    use crate::{errors::CircuitError, gadgets::from_emulated_field, test, Circuit, PlonkCircuit};
    use ark_bls12_377::{Fq as Fq377, Fr as Fr377};
    use ark_bls12_381::{Fq as Fq381, Fr as Fr381};
    use ark_bn254::{Fq as Fq254, Fr as Fr254};
    use ark_bw6_761::{Fq as Fq761, Fr as Fr761};
    use ark_ff::{BigInteger, MontFp, PrimeField};
    use ark_std::{num, vec::Vec, UniformRand};
    use itertools::Itertools;
    use jf_utils::{bytes_to_field_elements, test_rng, to_bytes};
    use num_bigint::BigUint;

    #[test]
    fn test_basics() {
        test_basics_helper::<Fq377, Fr254>();
        test_basics_helper::<Fr254, Fq254>();
        test_basics_helper::<Fq254, Fr254>();
        test_basics_helper::<Fr377, Fq377>();
        test_basics_helper::<Fr381, Fq381>();
        test_basics_helper::<Fr761, Fq761>();
    }

    fn test_basics_helper<E, F>()
    where
        E: EmulationConfig<F>,
        F: PrimeField,
    {
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        let var_x = circuit.create_emulated_variable(E::one()).unwrap();
        let overflow = E::from(F::MODULUS.into() * 2u64 + 1u64);
        let var_y = circuit.create_emulated_variable(overflow).unwrap();
        assert_eq!(circuit.emulated_witness(&var_x).unwrap(), E::one());
        assert_eq!(circuit.emulated_witness(&var_y).unwrap(), overflow);
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
    }

    /*#[test]
    fn test_validity_check() {
        test_validity_check_helper::<Fq377, Fr254>();
        test_validity_check_helper::<Fr254, Fq254>();
        test_validity_check_helper::<Fq254, Fr254>();
        test_validity_check_helper::<Fr377, Fq377>();
        test_validity_check_helper::<Fr381, Fq381>();
        test_validity_check_helper::<Fr761, Fq761>();
    }

    fn test_validity_check_helper<E, F>()
    where
        E: EmulationConfig<F>,
        F: PrimeField,
    {
        let rng = &mut test_rng();
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(8);
        // Randomly generated emulated variables should all pass the validity check
        for _ in 0..20 {
            let val = E::rand(rng);
            let emu_var = circuit.create_emulated_variable(val).unwrap();
            circuit.enforce_valid_emulated_var(&emu_var).unwrap();
        }
        circuit.check_circuit_satisfiability(&[]).unwrap();

        // `E::MODULUS - 1` should pass the validity check but adding
        // something small and non-zero to any limb should not pass
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(8);
        let minus_one = -E::one();
        let emu_var = circuit.create_emulated_variable(minus_one).unwrap();
        circuit.enforce_valid_emulated_var(&emu_var).unwrap();
        circuit.check_circuit_satisfiability(&[]).unwrap();
        for _ in 0..20 {
            let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(8);

            let index = usize::rand(rng) % E::NUM_LIMBS;
            let rand_u8 = u8::rand(rng);
            let mut emu_var = circuit.create_emulated_variable(minus_one).unwrap();
            emu_var.0[index] = circuit
                .add_constant(emu_var.0[index], &F::from(rand_u8 + 1))
                .unwrap();

            circuit.enforce_valid_emulated_var(&emu_var).unwrap();
            assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        }
    }*/

    #[test]
    fn test_emulated_add() {
        test_emulated_add_helper::<Fq377, Fr254>();
        test_emulated_add_helper::<Fr254, Fq254>();
        test_emulated_add_helper::<Fq254, Fr254>();
        test_emulated_add_helper::<Fr377, Fq377>();
        test_emulated_add_helper::<Fr381, Fq381>();
        test_emulated_add_helper::<Fr761, Fq761>();
    }

    fn test_emulated_add_helper<E, F>()
    where
        E: EmulationConfig<F>,
        F: PrimeField,
    {
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(8);
        let var_x = circuit.create_public_emulated_variable(E::one()).unwrap();
        let overflow = E::from(E::MODULUS.into() - 1u64);
        let var_y = circuit.create_emulated_variable(overflow).unwrap();
        let var_z = circuit.emulated_add(&var_x, &var_y).unwrap();
        assert_eq!(circuit.emulated_witness(&var_x).unwrap(), E::one());
        assert_eq!(circuit.emulated_witness(&var_y).unwrap(), overflow);
        assert_eq!(circuit.emulated_witness(&var_z).unwrap(), E::zero());

        let var_z = circuit.emulated_add_constant(&var_z, overflow).unwrap();
        assert_eq!(circuit.emulated_witness(&var_z).unwrap(), overflow);

        let x = from_emulated_field(E::one());
        assert!(circuit.check_circuit_satisfiability(&x).is_ok());

        let var_z = circuit.create_emulated_variable(E::one()).unwrap();
        circuit.emulated_add_gate(&var_x, &var_y, &var_z).unwrap();
        assert!(circuit.check_circuit_satisfiability(&x).is_err());
    }

    #[test]
    fn test_emulated_mul() {
        test_emulated_mul_helper::<Fq377, Fr254>();
        test_emulated_mul_helper::<Fr254, Fq254>();
        test_emulated_mul_helper::<Fq254, Fr254>();
        test_emulated_mul_helper::<Fr377, Fq377>();
        test_emulated_mul_helper::<Fr381, Fq381>();
        test_emulated_mul_helper::<Fr761, Fq761>();

        // test for issue (https://github.com/EspressoSystems/jellyfish/issues/306)
        let x : Fq377= MontFp!("218393408942992446968589193493746660101651787560689350338764189588519393175121782177906966561079408675464506489966");
        let y : Fq377 = MontFp!("122268283598675559488486339158635529096981886914877139579534153582033676785385790730042363341236035746924960903179");

        let mut circuit = PlonkCircuit::<Fr254>::new_turbo_plonk();
        let var_x = circuit.create_emulated_variable(x).unwrap();
        let _ = circuit.emulated_mul_constant(&var_x, y).unwrap();
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
    }

    fn test_emulated_mul_helper<E, F>()
    where
        E: EmulationConfig<F>,
        F: PrimeField,
    {
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(20);
        let x = E::from(6732u64);
        let y = E::from(E::MODULUS.into() - 12387u64);
        let expected = x * y;
        let var_x = circuit.create_public_emulated_variable(x).unwrap();
        let var_y = circuit.create_emulated_variable(y).unwrap();
        let var_z = circuit.emulated_mul(&var_x, &var_y).unwrap();
        assert_eq!(circuit.emulated_witness(&var_x).unwrap(), x);
        assert_eq!(circuit.emulated_witness(&var_y).unwrap(), y);
        assert_eq!(circuit.emulated_witness(&var_z).unwrap(), expected);
        assert!(circuit
            .check_circuit_satisfiability(&from_emulated_field(x))
            .is_ok());

        let var_y_z = circuit.emulated_mul(&var_y, &var_z).unwrap();
        assert_eq!(circuit.emulated_witness(&var_y_z).unwrap(), expected * y);
        assert!(circuit
            .check_circuit_satisfiability(&from_emulated_field(x))
            .is_ok());

        let var_z = circuit.emulated_mul_constant(&var_z, expected).unwrap();
        assert_eq!(
            circuit.emulated_witness(&var_z).unwrap(),
            expected * expected
        );
        assert!(circuit
            .check_circuit_satisfiability(&from_emulated_field(x))
            .is_ok());

        let var_z = circuit.create_emulated_variable(E::one()).unwrap();
        circuit.emulated_mul_gate(&var_x, &var_y, &var_z).unwrap();
        assert!(circuit
            .check_circuit_satisfiability(&from_emulated_field(x))
            .is_err());
    }

    #[test]
    fn test_select() {
        test_select_helper::<Fq377, Fr254>();
        test_select_helper::<Fr254, Fq254>();
        test_select_helper::<Fq254, Fr254>();
        test_select_helper::<Fr377, Fq377>();
        test_select_helper::<Fr381, Fq381>();
        test_select_helper::<Fr761, Fq761>();
    }

    fn test_select_helper<E, F>()
    where
        E: EmulationConfig<F>,
        F: PrimeField,
    {
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        let var_x = circuit.create_emulated_variable(E::one()).unwrap();
        let overflow = E::from(E::MODULUS.into() - 1u64);
        let var_y = circuit.create_emulated_variable(overflow).unwrap();
        let b = circuit.create_boolean_variable(true).unwrap();
        let var_z = circuit
            .conditional_select_emulated(b, &var_x, &var_y)
            .unwrap();
        assert_eq!(circuit.emulated_witness(&var_z).unwrap(), overflow);
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        *circuit.witness_mut(var_z.0[0]) = F::zero();
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
    }

    #[test]
    fn test_enforce_equal() {
        test_enforce_equal_helper::<Fq377, Fr254>();
        test_enforce_equal_helper::<Fr254, Fq254>();
        test_enforce_equal_helper::<Fq254, Fr254>();
        test_enforce_equal_helper::<Fr377, Fq377>();
        test_enforce_equal_helper::<Fr381, Fq381>();
        test_enforce_equal_helper::<Fr761, Fq761>();
    }

    fn test_enforce_equal_helper<E, F>()
    where
        E: EmulationConfig<F>,
        F: PrimeField,
    {
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        let var_x = circuit.create_emulated_variable(E::one()).unwrap();
        let overflow = E::from(E::MODULUS.into() - 1u64);
        let var_y = circuit.create_emulated_variable(overflow).unwrap();
        let var_z = circuit.create_emulated_variable(overflow).unwrap();
        circuit.enforce_emulated_var_equal(&var_y, &var_z).unwrap();
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        circuit.enforce_emulated_var_equal(&var_x, &var_y).unwrap();
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
    }

    #[test]
    fn test_emulated_var_to_field_vec() {
        test_emulated_var_to_field_vec_helper::<Fq377, Fr254>();
        test_emulated_var_to_field_vec_helper::<Fr254, Fq254>();
        test_emulated_var_to_field_vec_helper::<Fq254, Fr254>();
        test_emulated_var_to_field_vec_helper::<Fr377, Fq377>();
        test_emulated_var_to_field_vec_helper::<Fr381, Fq381>();
        test_emulated_var_to_field_vec_helper::<Fr761, Fq761>();
    }

    fn test_emulated_var_to_field_vec_helper<E, F>()
    where
        E: EmulationConfig<F>,
        F: PrimeField,
    {
        let rng = &mut test_rng();
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        for _ in 0..100 {
            let x = E::rand(rng);
            let var_x = circuit.create_emulated_variable(x).unwrap();
            let var_x_vec = circuit.emulated_var_to_field_vars(&var_x).unwrap();
            let x_vec_f = var_x_vec
                .iter()
                .map(|var| circuit.witness(*var).unwrap())
                .collect::<Vec<F>>();
            let x_bytes = to_bytes!(&x).unwrap();
            let bytes_to_field = bytes_to_field_elements(x_bytes);
            for (calc, expect) in x_vec_f.iter().zip(bytes_to_field.iter()) {
                assert_eq!(*calc, *expect);
            }
        }
    }

    #[test]
    fn test_unpack() {
        test_emulated_unpack_helper::<Fq377, Fr254>();
        test_emulated_unpack_helper::<Fr254, Fq254>();
        test_emulated_unpack_helper::<Fq254, Fr254>();
        test_emulated_unpack_helper::<Fr377, Fq377>();
        test_emulated_unpack_helper::<Fr381, Fq381>();
        test_emulated_unpack_helper::<Fr761, Fq761>()
    }

    fn test_emulated_unpack_helper<E: EmulationConfig<F>, F: PrimeField>() {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let a = circuit.create_emulated_variable(E::one()).unwrap();
        let b = circuit.create_emulated_variable(E::from(1023u32)).unwrap();

        circuit.emulated_enforce_in_range(&a, 1).unwrap();
        let a_le = circuit.emulated_unpack(&a, 3).unwrap();
        assert_eq!(a_le.len(), 3);
        let b_le = circuit.emulated_unpack(&b, 10).unwrap();
        assert_eq!(b_le.len(), 10);
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        assert!(circuit.emulated_unpack(&b, 9).is_err());
    }

    // Doesn't work with the Fq377, Fr254 configuration. Maybe try to fix this in the future
    #[test]
    fn test_emulated_decomposition() -> Result<(), CircuitError> {
        test_emulated_decomposition_helper::<Fr254, Fq254>()?;
        test_emulated_decomposition_helper::<Fq254, Fr254>()?;
        test_emulated_decomposition_helper::<Fr377, Fq377>()?;
        test_emulated_decomposition_helper::<Fr381, Fq381>()?;
        test_emulated_decomposition_helper::<Fr761, Fq761>()
    }

    fn test_emulated_decomposition_helper<E: EmulationConfig<F>, F: PrimeField>(
    ) -> Result<(), CircuitError> {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_ultra_plonk(8);
            let raw_scalar = E::rand(rng);
            let scalar_var = circuit.create_emulated_variable(raw_scalar)?;
            let scalar_bit_length = E::MODULUS_BIT_SIZE as usize;
            let c = usize::rand(rng) % 10 + 2;
            // create witness
            let m = (scalar_bit_length - 1) / c + 1;
            let mut scalar_val = circuit.emulated_witness(&scalar_var)?.into_bigint();
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
            circuit.emulated_decomposition_gate(&decomposed_scalar_vars, &scalar_var, c)?;

            circuit.check_circuit_satisfiability(&[])?;
        }
        Ok(())
    }

    // Doesn't work with the Fq377, Fr254 configuration. Maybe try to fix this in the future
    #[test]
    fn test_convert_to_transcript() -> Result<(), CircuitError> {
        test_convert_to_transcript_helper::<Fr254, Fq254>()?;
        test_convert_to_transcript_helper::<Fq254, Fr254>()?;
        test_convert_to_transcript_helper::<Fr377, Fq377>()?;
        test_convert_to_transcript_helper::<Fr381, Fq381>()?;
        test_convert_to_transcript_helper::<Fr761, Fq761>()
    }

    fn test_convert_to_transcript_helper<E: EmulationConfig<F>, F: PrimeField>(
    ) -> Result<(), CircuitError> {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let raw_scalar = E::rand(rng);
            let scalar_bytes = to_bytes!(&raw_scalar).unwrap();
            let scalar_field_elems = bytes_to_field_elements::<_, F>(scalar_bytes);
            let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_ultra_plonk(8);
            let scalar_var = circuit.create_emulated_variable(raw_scalar)?;

            let function_vars = circuit.convert_for_transcript(&scalar_var)?;
            for (calc, expect) in function_vars.iter().zip(scalar_field_elems[1..].iter()) {
                assert_eq!(circuit.witness(*calc).unwrap(), *expect);
            }
            circuit.check_circuit_satisfiability(&[])?;
        }
        Ok(())
    }
}
