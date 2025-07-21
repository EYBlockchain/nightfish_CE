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

use ark_std::{format, println, string::ToString, vec, vec::Vec, One, Zero};
use core::{marker::PhantomData, num, ops::Sub, panic};
use itertools::izip;
use jf_utils::{bytes_to_field_elements, field_switching, to_bytes};
use num_bigint::{BigInt, BigUint, Sign, ToBigInt, ToBigUint};
use num_integer::Integer;
use num_traits::{FromPrimitive, Signed};

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

    /// Changes the value represented by an emulated variable
    pub fn set_emulated_witness<E: EmulationConfig<F>>(
        &mut self,
        var: &EmulatedVariable<E>,
        val: E,
    ) {
        let values = from_emulated_field(val);
        for (v, &witness) in var.0.iter().zip(values.iter()) {
            *self.witness_mut(*v) = witness;
        }
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
                coeff_vec.push(coeff);

                let carry_val = val / (BigUint::from(2u32).pow(power));
                carry_var = self.create_variable(F::from(carry_val))?;

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

        // enforcing a * b - k * E::MODULUS = c mod 2^T

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
        // Here our use of `mod_to_native_field` is valid as we don't
        // actually care what `a`, `b`, `c` and `k` are modulo `F::MODULUS`.
        // We are using `mod_to_native_field` to constrain an integer relationship
        // by constraining everything modulo `F::MODULUS` and modulo `2^T`.
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

    // Precompute all cross-products x_i * y_j as circuit variables.
    fn cross_products<E: EmulationConfig<F>>(
        &mut self,
        x: &EmulatedVariable<E>,
        y: &EmulatedVariable<E>,
    ) -> Result<Vec<Variable>, CircuitError> {
        let n = E::NUM_LIMBS;
        let mut products = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let p = self.mul(x.0[i], y.0[j])?;
                products.push(p);
            }
        }
        Ok(products)
    }

    /// 1)  
    ///     ∑_{i,j}(B^{i+j} mod q)·x_i·y_j
    ///   − ∑_i (B^i mod q)·z_i
    ///   − (u + k_min)·q ≡ 0   (mod p)
    /// 2) for each auxiliary modulus m ∈ M:
    ///
    ///   ∑_{i,j}((B^{i+j}%q)%m)·x_i·y_j
    /// − ∑_i((B^i%q)%m)·z_i
    /// − u·(q%m)
    /// − (k_min·q)%m
    /// − (v_m + l_min)·m
    ///   ≡ 0  (mod p)
    pub fn enforce_main_identity<E: EmulationConfig<F>>(
        &mut self,
        x: &EmulatedVariable<E>,
        y: &EmulatedVariable<E>,
        z: &EmulatedVariable<E>,
        u: Variable,
        k_min: &BigInt,
        pxy: Vec<Variable>,
    ) -> Result<(), CircuitError> {
        // 1) grab q (the emulated field modulus) as BigUint and F
        let q_big: BigUint = BigUint::from_bytes_le(&E::MODULUS.to_bytes_le());
        let q_f: F = F::from(q_big.clone());
        let b_big_u = BigUint::from(2u32).pow(E::B as u32);

        // 2) build up one big linear combination
        let mut coeffs = Vec::with_capacity(E::NUM_LIMBS * E::NUM_LIMBS + E::NUM_LIMBS + 1);
        let mut vars = Vec::with_capacity(coeffs.capacity());

        // 2a) + Σ B^{i+j}·(x_i·y_j)
        for idx in 0..pxy.len() {
            // idx = i * n + j
            let i = idx / E::NUM_LIMBS;
            let j = idx % E::NUM_LIMBS;
            let c = F::from(b_big_u.pow((i + j) as u32) % &q_big);
            coeffs.push(c);
            vars.push(pxy[idx]);
        }

        // 2b) − Σ B^i·z_i
        for i in 0..E::NUM_LIMBS {
            let c_i = F::from(b_big_u.pow(i as u32) % &q_big);
            coeffs.push(-c_i);
            vars.push(z.0[i]);
        }

        // 2c) − q·u
        coeffs.push(-q_f);
        vars.push(u);

        // 2d) constant = −(k_min·q) in F, even if k_min<0
        let p_bigint = BigInt::from_bytes_le(Sign::Plus, &F::MODULUS.to_bytes_le());
        let k_mod = k_min.mod_floor(&p_bigint); // in [0,p)
        let k_u = k_mod.to_biguint().unwrap();
        let k_fe = F::from(k_u);
        let constant = -(q_f * k_fe);

        // 3) one gate: Σ coeffs[i]·vars[i] + constant ≡ 0
        let expr = self.lin_comb(&coeffs, &constant, &vars)?;
        self.enforce_constant(expr, F::zero())?;
        println!("lc len main = {}", coeffs.len());
        Ok(())
    }

    /// 2) for each auxiliary modulus m ∈ M:
    ///
    ///   ∑_{i,j}((B^{i+j}%q)%m)·x_i·y_j
    /// − ∑_i((B^i%q)%m)·z_i
    /// − u·(q%m)
    /// − (k_min·q)%m
    /// − (v_m + l_min)·m
    ///   ≡ 0  (mod p)
    pub fn enforce_aux_mod_id<E: EmulationConfig<F>>(
        &mut self,
        x: &EmulatedVariable<E>,
        y: &EmulatedVariable<E>,
        z: &EmulatedVariable<E>,
        u: Variable,
        k_min: BigInt,
        m: BigUint,
        l_min: BigInt,
        v_m: Variable,
        pxy: Vec<Variable>,
    ) -> Result<(), CircuitError> {
        // 1) constants
        let q_big = BigUint::from_bytes_le(&E::MODULUS.to_bytes_le());
        let q_mod_m = &q_big % &m;
        let q_mod_m_fe = F::from(q_mod_m.clone());
        let m_fe = F::from(m.clone());

        let p_bigint = BigInt::from_bytes_le(Sign::Plus, &F::MODULUS.to_bytes_le());
        let b_big_u = BigUint::from(2u32).pow(E::B as u32);

        // 2) build linear-combination of vars
        let mut coeffs = Vec::with_capacity(
            E::NUM_LIMBS * E::NUM_LIMBS  // x·y
    + E::NUM_LIMBS                // z
    + 1, // u
        );
        let mut vars = Vec::with_capacity(coeffs.capacity());

        // + Σ (B^{i+j}%q)%m · (x_i·y_j)
        for idx in 0..pxy.len() {
            // idx = i * n + j
            let i = idx / E::NUM_LIMBS;
            let j = idx % E::NUM_LIMBS;
            let c = F::from(b_big_u.pow((i + j) as u32) % &q_big % &m);
            coeffs.push(c);
            vars.push(pxy[idx]);
        }

        // − Σ (B^i%q)%m · z_i
        for i in 0..E::NUM_LIMBS {
            let c = F::from(b_big_u.pow(i as u32) % &q_big % &m);
            coeffs.push(-c);
            vars.push(z.0[i]);
        }

        // − (q % m)·u
        coeffs.push(-q_mod_m_fe);
        vars.push(u);

        // 3) constant = −((k_min·q)%m) − (v_m + l_min)·m
        // (k_min·q)%m
        let kq_mod_m = (k_min * BigInt::from_biguint(Sign::Plus, q_big.clone()))
            % (&BigInt::from_biguint(Sign::Plus, m.clone()));
        let mut kq_u = kq_mod_m % (&p_bigint);
        if kq_u.is_negative() {
            kq_u += &p_bigint; // ensure kq_u is non-negative
        }
        println!("kq_u = {}", kq_u);
        println!("p_bigint = {}", p_bigint);
        let kq = F::from(kq_u.to_biguint().unwrap());

        // l_min·m  mod p
        let mut l_min = l_min % (&p_bigint);
        if l_min.is_negative() {
            l_min += &p_bigint; // ensure l_min is non-negative
        }
        let l_min = F::from(l_min.to_biguint().unwrap());

        coeffs.push(-m_fe); // coefficient = -m
        vars.push(v_m); // v_m_var: the Variable you created for v_m

        let constant: F = -(kq + l_min * m_fe);

        // 4) enforce
        let expr = self.lin_comb(&coeffs, &constant, &vars)?;

        self.enforce_constant(expr, F::zero())?;
        println!("lc len aux = {}", coeffs.len());
        Ok(())
    }

    /// Enforce all of:
    ///   1) ∑_{i,j}(B^{i+j} mod q)·x_i·y_j
    ///    - ∑_i (B^i mod q)·z_i
    ///    - (u + k_min)·q ≡ 0  (mod p)
    ///   2) For each m ∈ E::M:
    ///      ∑ ((B^{i+j}%q)%m)x_i y_j
    ///    - ∑ ((B^i%q)%m)z_i
    ///    - u·(q%m)
    ///    - (k_min·q)%m
    ///    - (v_m + l_min)·m ≡ 0 (mod p)
    ///   3) Range‐checks:
    ///      • each z_i in [0,B)
    ///      • u in [0,U_MAX)
    ///      • each v_m in [0,V_MAX)
    pub fn emulated_mul_gate_2<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: &EmulatedVariable<E>,
        c: &EmulatedVariable<E>,
    ) -> Result<(), CircuitError> {
        // 0) ensure inputs are limb-bounded
        self.check_vars_bound(&a.0)?;
        self.check_vars_bound(&b.0)?;
        self.check_vars_bound(&c.0)?;

        // 1) get BigUint witnesses
        let val_a: BigUint = self.emulated_witness(a)?.into();
        let val_b: BigUint = self.emulated_witness(b)?.into();
        let val_c: BigUint = self.emulated_witness(c)?.into();

        let n = E::NUM_LIMBS;
        let base = E::B as u32;

        let x_limbs: Vec<BigUint> = biguint_to_limbs::<F>(&val_a, base as usize, n)
            .into_iter()
            .map(Into::into)
            .collect();
        let y_limbs: Vec<BigUint> = biguint_to_limbs::<F>(&val_b, base as usize, n)
            .into_iter()
            .map(Into::into)
            .collect();
        let z_limbs: Vec<BigUint> = biguint_to_limbs::<F>(&val_c, base as usize, n)
            .into_iter()
            .map(Into::into)
            .collect();

        // --- inline print_bounds() computations with signed integers ---
        let b_big_u = BigUint::from(2u32).pow(base);
        let b_big = b_big_u.to_bigint().unwrap();
        let q_big = E::MODULUS.into();
        let q_int = q_big.to_bigint().unwrap();
        let p_big_u = F::MODULUS.into();
        println!("p = {}", p_big_u);

        // 1) signed lower‐bound
        let mut power = BigInt::one();
        let mut acc = BigInt::zero();
        for _ in 0..n {
            acc += (&power % &q_int);
            power *= &b_big;
        }
        let signed_lb = -((&b_big - 1u32) * acc); // now truly negative

        // 2) signed upper‐bound
        let mut dbl = BigInt::zero();
        for i in 0..n {
            for j in 0..n {
                let idx = (i + j) as u32;
                let term = BigInt::from(b_big_u.clone().pow(idx) % &q_big);
                dbl += term;
            }
        }
        let signed_ub = (&b_big - 1u32).pow(2) * dbl; // positive

        // 3) floor‐divide by q to get k_min, k_max
        let k_min = signed_lb.clone() / (&q_int); // negative or zero
        let k_max = signed_ub.clone() / (&q_int); // non‐negative

        // 4) u_max = 2^t where t = ceil(log2(k_max - k_min))
        let span = (&k_max - &k_min).to_biguint().unwrap();

        let u_max_bits = if span.is_zero() { 0 } else { log2up(&span) };
        let u_max_i = BigInt::one() << u_max_bits;

        // 5) first_bound, second_bound, max_bound
        let first_b = -signed_lb + (&u_max_i + &k_min) * &q_int;
        let second_b = signed_ub - (&k_min) * &q_int;
        let max_b = first_b.abs().max(second_b.clone());

        // 6) choose bit‐length for m = 2^num_bits where num_bits = ceil(log2(max_b / p))
        let max_over_p = (&max_b.to_biguint().unwrap() / &p_big_u);
        let mut m_bits = if max_over_p.is_zero() {
            0
        } else {
            log2up(&max_over_p)
        };

        println!("max_b = {}", max_b);
        println!("first_b = {}", first_b);
        println!("second_b = {}", second_b);
        println!("m_bits = {}", m_bits);

        let m_i = BigInt::one() << m_bits;
        let m_big = m_i.to_biguint().unwrap();

        // 7) recompute lb_m, ub_m mod m
        let mut pow_m = BigInt::one();
        let mut acc_m = BigInt::zero();
        for _ in 0..n {
            acc_m += (&pow_m % &q_int) % &m_i;
            pow_m *= &b_big;
        }
        let lb_m = (&b_big - 1u32) * acc_m;

        let mut dbl_m = BigInt::zero();
        for i in 0..n {
            for j in 0..n {
                let idx = (i + j) as u32;
                let term = BigInt::from(b_big_u.clone().pow(idx) % &q_big);
                dbl_m += term % &m_i;
            }
        }
        let ub_m = (&b_big - 1u32).pow(2) * dbl_m;

        // 8) l_min, l_max
        let l_min = -(&lb_m + &u_max_i * (&q_int % &m_i) + ((&k_min * &q_int) % &m_i)) / (&m_i);
        let l_max = (&ub_m - ((&k_min * &q_int) % &m_i)) / (&m_i);

        // --- 2) compute t = xy_term − z_term  (signed) ---
        let mut sum_xy = BigInt::zero();
        for i in 0..n {
            for j in 0..n {
                let coef = BigInt::from(b_big_u.clone().pow((i + j) as u32) % &q_big);
                let xi = BigInt::from(x_limbs[i].clone());
                let yj = BigInt::from(y_limbs[j].clone());
                sum_xy += coef * xi * yj;
            }
        }
        let xy_term = sum_xy;

        let mut sum_z = BigInt::zero();
        let mut acc_p = BigInt::one();
        for i in 0..n {
            let zi = BigInt::from(z_limbs[i].clone());
            sum_z += (&acc_p % &q_int) * zi;
            acc_p *= &b_big;
        }
        let z_term = sum_z;

        let t = xy_term - z_term;

        // --- 3) k = ceil( t / q ) in signed world ---
        let k_int = t.div_ceil(&q_int);
        assert_eq!(t, k_int.clone() * &q_int + (&t % &q_int));
        let val_u_int = &k_int - &k_min; // u = k - k_min
        let val_u = val_u_int.to_biguint().unwrap();
        println!("k = {}", k_int);
        println!("u = {}", val_u);
        let u = self.create_variable(val_u.clone().into())?;

        // --- 4) auxiliary‐mod‐m: l = ceil( total / m ) ---
        // compute total = sum_xy_m - sum_z_m - u*(q mod m) + (k_min*q mod m)
        let q_mod_m = &q_int % &m_i;
        let mut sum_xy_m = BigInt::zero();
        let mut sum_z_m = BigInt::zero();
        for i in 0..n {
            for j in 0..n {
                let coef = BigInt::from(b_big_u.pow((i + j) as u32) % &q_big) % &m_i;
                println!("coef_qm({},{}) = {}", i, j, coef);
                let coef2 = BigInt::from(b_big_u.pow((i + j) as u32) % &q_big);
                println!("coef_q({},{}) = {}", i, j, coef2);
                let xi = BigInt::from(x_limbs[i].clone());
                let yj = BigInt::from(y_limbs[j].clone());
                sum_xy_m += coef * xi * yj;
            }
            let coef_z = BigInt::from(b_big_u.pow((i) as u32) % &q_big) % &m_i;
            let zi = BigInt::from(z_limbs[i].clone());
            sum_z_m += coef_z * zi;
        }
        let total = sum_xy_m - sum_z_m - (&val_u_int * &q_mod_m) - ((&k_min * &q_int) % &m_i);

        // ceil‐divide by m
        let l_int = total.div_ceil(&m_i);
        assert_eq!(total, l_int.clone() * &m_i + (&total % &m_i));

        println!("TOTAL = {}", total);
        println!("l_int*m = {}", &l_int * &m_i);
        let val_v_int = &l_int - &l_min; // v = l - l_min
        println!("l = {}", l_int);
        println!("v = {}", val_v_int);
        let val_v = (val_v_int).to_biguint().unwrap();
        let v = self.create_variable(val_v.into())?;
        let span = (&l_max - &l_min).to_biguint().unwrap();
        let v_max_bits = if span.is_zero() { 0 } else { log2up(&span) };
        let v_max_i = BigInt::one() << v_max_bits;

        let m_mod = |x: &BigInt| x % (&m_i);

        // recompute the per-limb sums modulo m
        let mut pow = BigInt::one();
        let mut sum1 = BigInt::zero();
        for _ in 0..n {
            sum1 += m_mod(&(&pow % &q_int));
            pow *= &b_big;
        }
        let lb_coeff = (&b_big - 1u32) * &sum1;

        let mut sum2 = BigInt::zero();
        for i in 0..n {
            for j in 0..n {
                let c = BigInt::from(b_big_u.pow((i + j) as u32) % &q_big);
                sum2 += m_mod(&c);
            }
        }
        let ub_coeff = (&b_big - 1u32).pow(2) * &sum2;
        println!("ub_coeff = {}", ub_coeff);

        let qm = BigInt::from(q_big.clone()) % &m_i;
        let kqmod = (&k_min * &BigInt::from(q_big.clone())) % (&m_i);

        // the two wrap-bounds from eq.(4):
        let bound_low = -(&lb_coeff + &u_max_i * &qm + &kqmod + &v_max_i * &m_i + &l_min * &m_i);
        let bound_high = &ub_coeff - &kqmod - &l_min * &m_i;

        let p_big_u = BigUint::from_bytes_le(&F::MODULUS.to_bytes_le());
        let p_int = BigInt::from_biguint(Sign::Plus, p_big_u.clone());
        println!("bound_low = {}", bound_low);
        println!("bound_high = {}", bound_high);
        // check p > both bounds in absolute value
        if !(&p_int > &bound_high.abs() && &p_int > &bound_low.abs()) {
            println!("bound_low = {}", bound_low);
            println!("bound_high = {}", bound_high);
            println!("p = {}", p_int);
            panic!("p is too small to prevent wrap-around in the auxiliary mod-m identity");
        };

        let m_i = BigInt::one() << m_bits;
        let m_big = m_i.to_biguint().unwrap();

        let q_mod_m = BigInt::from_biguint(Sign::Plus, q_big.clone() % &m_big);

        // --- B) compute each term as a BigInt ---
        // term1 = Σ_{i,j} ((B^{i+j} mod q) % m) * x_i * y_j
        let mut term1 = BigInt::zero();
        for i in 0..n {
            for j in 0..n {
                let coeff = //BigInt::from_biguint(Sign::Plus, b_big_u.pow((i + j) as u32) % &q_big).mod_floor(&m_i);
                BigInt::from(b_big_u.pow((i + j) as u32) % &q_big) % &m_i;
                term1 += coeff
                    * BigInt::from_biguint(Sign::Plus, x_limbs[i].clone())
                    * BigInt::from_biguint(Sign::Plus, y_limbs[j].clone());
            }
        }

        // term2 = Σ_i ((B^i mod q) % m) * z_i
        let mut term2 = BigInt::zero();
        for i in 0..n {
            let coeff =
               // BigInt::from_biguint(Sign::Plus, b_big_u.pow(i as u32) % &q_big).mod_floor(&m_i);
               BigInt::from(b_big_u.pow((i) as u32) % &q_big) % &m_i;
            term2 += coeff * BigInt::from_biguint(Sign::Plus, z_limbs[i].clone());
        }

        // term3 = u * (q % m)
        let term3 = val_u_int.clone() * &q_mod_m;

        // term4 = (k_min * q) % m
        let term4 = (k_min.clone() * &q_int) % (&m_i);

        // term5 = (v_m + l_min) * m
        let term5 = (BigInt::from(val_v_int.clone()) + l_min.clone()) * &m_i;

        let R = term1.clone() - term2.clone() - term3.clone() - term4.clone();
        let R =
            term1.clone() - term2.clone() - (&val_u_int * &q_mod_m) - ((&k_min * &q_int) % &m_i);
        let ell = val_v_int + &l_min; // ℓ = v + ℓₘᵢₙ
        println!("R = {}", R);
        println!("ℓ·m = {}", &ell * &m_i);
        println!("Difference S = {}", R - &ell * &m_i);

        // S = term1 - term2 - term3 - term4 - term5
        let S = term1.clone() - term2.clone() - term3.clone() - term4.clone() - term5.clone();

        // finally, check S ≡ 0 (mod p)
        let r = S.mod_floor(&p_int.to_bigint().unwrap());
        println!(
            "aux‐mod identity failed: S mod p = {} (expected 0). \
                terms: [{} - {} - {} - {} - {}] = {}",
            r, term1, term2, term3, term4, term5, S
        );

        self.enforce_in_range(u, u_max_bits as usize)?;

        // precompute products
        let pxy = self.cross_products::<E>(a, b)?;

        // 5) main identity mod native p
        self.enforce_main_identity::<E>(a, b, c, u, &k_min, pxy.clone())?;

        // 6) auxiliary-mod-m identity

        self.enforce_in_range(v, v_max_bits as usize)?;

        self.enforce_aux_mod_id::<E>(
            a,
            b,
            c,
            u,
            k_min.clone(),
            m_big,
            l_min.clone(),
            v,
            pxy.clone(),
        )?;

        Ok(())
    }

    /// convenience wrapper: multiply and gate‐check in one call
    pub fn emulated_mul<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: &EmulatedVariable<E>,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        // 1) allocate the “product” witness
        let c = self.emulated_witness(a)? * self.emulated_witness(b)?;
        let c = self.create_emulated_variable(c)?;
        // 2) enforce a·b = c via all the limb‐sums + mod q and mod m checks
        self.emulated_mul_gate(a, b, &c)?;
        Ok(c)
    }

    fn needs_normalise<E: EmulationConfig<F>>(
        &self,
        x: &EmulatedVariable<E>,
    ) -> Result<bool, CircuitError> {
        let base_big = BigUint::from(2u32).pow(E::B as u32); // B
        let bound = &base_big * (&base_big - 1u8); // B²

        for limb in &x.0 {
            let v: BigUint = self.witness(*limb)?.into();
            if v >= bound {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn lazy_add<E: EmulationConfig<F>>(
        &mut self,
        lhs: &EmulatedVariable<E>,
        rhs: &EmulatedVariable<E>,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        let mut limbs = Vec::with_capacity(E::NUM_LIMBS);

        for i in 0..E::NUM_LIMBS {
            let s = self.add(lhs.0[i], rhs.0[i])?; // zᵢ = aᵢ + bᵢ
                                                   // range-check   zᵢ < B²  (i.e.  2·E::B  bits)
            self.enforce_in_range(s, E::B + 1)?;
            limbs.push(s);
        }
        Ok(EmulatedVariable::<E>(limbs, PhantomData))
    }

    pub fn emulated_batch_add_precise<E: EmulationConfig<F>>(
        &mut self,
        inputs: &[EmulatedVariable<E>],
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        assert!(!inputs.is_empty(), "batch add needs at least one operand");

        // start with the first (already normalised) operand
        let mut acc = inputs[0].clone();

        for x in &inputs[1..] {
            // 1) lazy add with larger per-limb range ( < B² )
            acc = self.lazy_add::<E>(&acc, x)?;

            // 2) decide _precisely_ whether we must normalise now
            if self.needs_normalise::<E>(&acc)? {
                acc = self.normalise::<E>(&acc)?; // linear identity (5)
            }
        }
        // make sure the final result is well-formed
        if self.needs_normalise::<E>(&acc)? {
            acc = self.normalise::<E>(&acc)?;
        }
        Ok(acc) // limbs in [0,B)
    }

    pub fn emulated_batch_add_precise_gate<E: EmulationConfig<F>>(
        &mut self,
        inputs: &[EmulatedVariable<E>],
        out: &EmulatedVariable<E>,
    ) -> Result<(), CircuitError> {
        let sum = self.emulated_batch_add_precise::<E>(inputs)?;
        for i in 0..E::NUM_LIMBS {
            self.enforce_equal(sum.0[i], out.0[i])?;
        }
        Ok(())
    }

    /// Given *possibly* non-well-formed `x`, produce well-formed `z`
    /// so that  Σ (Bᶦ % q)·xᵢ − Σ (Bᶦ % q)·zᵢ  =  k·q           (eq. 5)
    /// and each zᵢ ∈ [0, B).  Follows the same blueprint as `enforce_main_identity`
    /// but degree 1, hence cheaper.
    /*pub fn normalise<E: EmulationConfig<F>>(
        &mut self,
        x: &EmulatedVariable<E>,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        // 1) Witness integer values and decompose a *fresh* well-formed `z`.
        let x_big: BigUint = self.emulated_witness(x)?.into();
        let z_big = (&x_big) % &E::MODULUS.into(); // canonical repr.
        let z = self.create_emulated_variable(E::from(z_big.clone()))?;

        // 2) Build the linear combination  Σ cᵢ·xᵢ − Σ cᵢ·zᵢ − k·q   (mod p)
        let q_big: BigUint = E::MODULUS.into();
        let q_f: F = F::from(q_big.clone());
        let b_big = BigUint::from(2u32).pow(E::B as u32);

        let mut coeffs = Vec::with_capacity(2 * E::NUM_LIMBS + 1);
        let mut vars = Vec::with_capacity(coeffs.capacity());

        for i in 0..E::NUM_LIMBS {
            let c = F::from(b_big.pow(i as u32) % &q_big); // (Bᶦ % q)
            coeffs.push(c);
            vars.push(x.0[i]); //  +c·xᵢ
            coeffs.push(-c);
            vars.push(z.0[i]); //  −c·zᵢ
        }

        // k := ceil( Σ cᵢ·xᵢ / q )  in Z
        let sum_x: BigInt = (0..E::NUM_LIMBS)
            .map(|i| {
                BigInt::from(b_big.pow(i as u32) % &q_big)
                    * BigInt::from(<F as Into<BigUint>>::into(self.witness(x.0[i]).unwrap()))
            })
            .sum();
        let k_int = sum_x.div_ceil(&BigInt::from_biguint(Sign::Plus, q_big.clone()));
        let k_fe = F::from(k_int.to_biguint().unwrap());
        let k_var = self.create_variable(k_fe)?;

        coeffs.push(-q_f); // −q·k
        vars.push(k_var);

        // 3) Enforce Σ coeffs·vars = 0   (mod p)
        let expr = self.lin_comb(&coeffs, &F::zero(), &vars)?;
        self.enforce_constant(expr, F::zero())?;

        // 4) Range-check new limbs  zᵢ ∈ [0, B)
        self.check_vars_bound(&z.0)?;

        Ok(z)
    }*/

    /// Turn a (possibly non-well-formed) element `x` into a well-formed `z`
    /// with limbs in `[0, B)` and prove the full identity
    ///
    ///   Σ cᵢ·xᵢ − Σ cᵢ·zᵢ  =  k·q                       (mod p)
    ///   Σ (cᵢ mod m)·xᵢ − Σ (cᵢ mod m)·zᵢ
    ///                 − k·(q mod m) − vₘ·m  = 0         (mod p)
    ///
    /// for every auxiliary modulus `m` listed in `E::M`.
    pub fn normalise<E: EmulationConfig<F>>(
        &mut self,
        x: &EmulatedVariable<E>,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        /* --------------------------------------------------------------- *
         * 0) witnesses and constants                                      *
         * --------------------------------------------------------------- */
        let q_big: BigUint = E::MODULUS.into();
        let q_f: F = F::from(q_big.clone());
        let b_big = BigUint::from(2u32).pow(E::B as u32); // B

        /* --------------------------------------------------------------- *
         * 1) Create canonical z  (limbs in [0,B))                         *
         * --------------------------------------------------------------- */
        let x_big: BigUint = self.emulated_witness(x)?.into();
        let z_big = &x_big % &q_big; // canonical rep
        let z = self.create_emulated_variable(E::from(z_big))?;

        /* --------------------------------------------------------------- *
         * 2) Build native-field (mod p) linear identity                   *
         * --------------------------------------------------------------- */
        let mut coeffs = Vec::with_capacity(2 * E::NUM_LIMBS + 1);
        let mut vars = Vec::with_capacity(coeffs.capacity());

        // Σ cᵢ·xᵢ − Σ cᵢ·zᵢ
        for i in 0..E::NUM_LIMBS {
            let c = F::from(b_big.pow(i as u32) % &q_big); // cᵢ=(Bᶦ mod q)
            coeffs.extend_from_slice(&[c, -c]);
            vars.extend_from_slice(&[x.0[i], z.0[i]]);
        }

        // k := ceil( Σ cᵢ·xᵢ  / q )
        let sum_x: BigInt = (0..E::NUM_LIMBS)
            .map(|i| {
                BigInt::from(b_big.pow(i as u32) % &q_big)
                    * BigInt::from_bytes_le(Sign::Plus, &self.witness(x.0[i]).unwrap().into_bigint().to_bytes_le())
            })
            .sum();
        let k_int = sum_x.div_ceil(&BigInt::from_biguint(Sign::Plus, q_big.clone()));
        let k_fe = F::from(k_int.to_biguint().unwrap());
        let k_var = self.create_variable(k_fe)?;
        coeffs.push(-q_f);
        vars.push(k_var); // −k·q

        // enforce native identity
        let expr = self.lin_comb(&coeffs, &F::zero(), &vars)?;
        self.enforce_constant(expr, F::zero())?;

        /* --------------------------------------------------------------- *
         * 3) Auxiliary-mod-m identities                                   *
         * --------------------------------------------------------------- */
        let p_int = BigInt::from_biguint(Sign::Plus, F::MODULUS.into());

        let m_big = BigInt::pow(&BigInt::from_isize(2isize).unwrap(), 107);
        let m_fe = F::from(m_big.clone().to_biguint().unwrap());
        let qm_big = &q_big.mod_floor(&m_big.to_biguint().unwrap()); // q mod m
        let qm_fe = F::from(qm_big.clone());

        /* ----- build Σ (cᵢ mod m)·xᵢ − Σ (cᵢ mod m)·zᵢ ------------- */
        let mut coeffs_m = Vec::with_capacity(2 * E::NUM_LIMBS + 2);
        let mut vars_m = Vec::with_capacity(coeffs_m.capacity());

        for i in 0..E::NUM_LIMBS {
            let c_mod_m =
                F::from(b_big.pow(i as u32) % &q_big.mod_floor(&m_big.to_biguint().unwrap()));
            coeffs_m.extend_from_slice(&[c_mod_m, -c_mod_m]);
            vars_m.extend_from_slice(&[x.0[i], z.0[i]]);
        }

        // −k·(q mod m)
        coeffs_m.push(-qm_fe);
        vars_m.push(k_var);


        // TODO clean + wrap around check

        /* ----- witness vₘ = ceil( S / m ) -------------------------- */
        // Compute S in ℤ
        let s_int: BigInt = (0..E::NUM_LIMBS)
            .map(|i| {
                let c = BigInt::from(
                    b_big.pow(i as u32)
                        % &q_big.mod_floor(&m_big.to_biguint().unwrap())
                        % &m_big.to_biguint().unwrap(),
                );
                let xi = BigInt::from_bytes_le(Sign::Plus, &self.witness(x.0[i]).unwrap().into_bigint().to_bytes_le());
                let zi = BigInt::from_bytes_le(Sign::Plus, &self.witness(z.0[i]).unwrap().into_bigint().to_bytes_le());
                c * (xi.sub(zi))
            })
            .sum::<BigInt>()
            - &k_int * BigInt::from_biguint(Sign::Plus, qm_big.clone());

        let v_int = s_int.div_ceil(&BigInt::from_biguint(
            Sign::Plus,
            m_big.clone().to_biguint().unwrap(),
        ));
        let v_fe = F::from(v_int.to_biguint().unwrap());
        let v_var = self.create_variable(v_fe)?;

        // −vₘ·m
        coeffs_m.push(-m_fe);
        vars_m.push(v_var);

        // enforce mod-m identity  (still modulo native p)
        let expr_m = self.lin_comb(&coeffs_m, &F::zero(), &vars_m)?;
        self.enforce_constant(expr_m, F::zero())?;

        // OPTIONAL: range-check vₘ   (use your bound for V_MAX)
        // self.enforce_in_range(v_var, V_MAX_BITS)?;

        /* --------------------------------------------------------------- *
         * 4) finally, z’s limbs are canonical                             *
         * --------------------------------------------------------------- */
        self.check_vars_bound(&z.0)?;
        Ok(z)
    }

    /*/// Return an [`EmulatedVariable`] which equals to a*b.
    pub fn emulated_mul<E: EmulationConfig<F>>(
        &mut self,
        a: &EmulatedVariable<E>,
        b: &EmulatedVariable<E>,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        let c = self.emulated_witness(a)? * self.emulated_witness(b)?;
        let c = self.create_emulated_variable(c)?;
        self.emulated_mul_gate(a, b, &c)?;
        Ok(c)
    }*/

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

        // enforcing a * b - k * E::MODULUS = c mod 2^T

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
        // Here our use of `mod_to_native_field` is valid as we don't
        // actually care what `a`, `b`, `c` and `k` are modulo `F::MODULUS`.
        // We are using `mod_to_native_field` to constrain an integer relationship
        // by constraining everything modulo `F::MODULUS` and modulo `2^T`.
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

    /// Given an emulated field element `a`, return `a mod F::MODULUS` in the native
    /// field. This does not ensure that the `EmulatedVariable` represents an
    /// integer less than `E::MODULUS`. This is because it is assumed that `a`
    /// represents an integer less than `2^n`, the smallest power of 2 greater than
    /// `E::MODULUS`. This ensures that this function will return a `Variable`
    /// representing either `a mod F::MODULUS` or `a + E::MODULUS mod F::MODULUS`.
    /// Since `mod_to_native_field` is nearly always used on a randomly squeezed
    /// scalar, this is good enough.
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
    const B: usize = 64;
    const NUM_LIMBS: usize = 6;
}

impl EmulationConfig<ark_bn254::Fr> for ark_bn254::Fq {
    /*const T: usize = 52 * 5; //288;
                             //const B: usize = 96;
                             //const NUM_LIMBS: usize = 3;
    const B: usize = 52;
    const NUM_LIMBS: usize = 5;*/
    //const T: usize = 42 * 6;
    //const B: usize = 43;
    //const NUM_LIMBS: usize = 6;
    //const T: usize = 64 * 4;
    //const B: usize = 64;
    //const NUM_LIMBS: usize = 4;
    const T: usize = 288;
    const B: usize = 96;
    const NUM_LIMBS: usize = 3;
}

impl EmulationConfig<ark_bn254::Fq> for ark_bn254::Fr {
    const T: usize = 43 * 6;
    //const B: usize = 128;
    //const NUM_LIMBS: usize = 2;
    //const B: usize = 85;
    //const NUM_LIMBS: usize = 3;
    const B: usize = 43;
    const NUM_LIMBS: usize = 6;
}

impl EmulationConfig<ark_bls12_377::Fq> for ark_bls12_377::Fr {
    const T: usize = 300;
    const B: usize = 64;
    const NUM_LIMBS: usize = 4;
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

pub fn log2up(x: &BigUint) -> u64 {
    println!("log2up({})", x);
    (x - BigUint::from(1u8)).bits()
}

#[cfg(test)]
mod tests {
    use super::log2up;
    use super::EmulationConfig;
    use crate::gadgets::EmulatedVariable;
    use crate::{
        errors::CircuitError,
        gadgets::{emulated::biguint_to_limbs, from_emulated_field},
        test, Circuit, PlonkCircuit,
    };
    use ark_bls12_377::{Fq as Fq377, Fr as Fr377};
    use ark_bls12_381::{Fq as Fq381, Fr as Fr381};
    use ark_bn254::{Fq as Fq254, Fr as Fr254};
    use ark_bw6_761::{Fq as Fq761, Fr as Fr761};
    use ark_ff::{BigInteger, MontFp, PrimeField};
    use ark_std::{num, vec::Vec, UniformRand};
    use itertools::Itertools;
    use jf_utils::{bytes_to_field_elements, test_rng, to_bytes};
    use num_bigint::BigUint;
    use num_bigint::ToBigInt;
    use num_integer::Integer;

    #[test]
    fn print_bounds() {
        let n = 3;
        let base = 90;
        let b = BigUint::from(2u32).pow(base);
        let p: BigUint = Fr254::MODULUS.into();
        let q: BigUint = Fq254::MODULUS.into();
        let lb = (b.clone() - BigUint::from(1u8))
            * (0..n)
                .fold(
                    (BigUint::from(0u8), BigUint::from(1u8)),
                    |(acc, power), _| {
                        let term = &power % q.clone();
                        let next_power = &power * b.clone();
                        (acc + term, next_power)
                    },
                )
                .0;
        ark_std::println!("lb: {}", lb);
        let ub = (b.clone() - BigUint::from(1u8))
            * (b.clone() - BigUint::from(1u8))
            * (0..n).fold(BigUint::from(0u8), |acc, i| {
                let row_sum = (0..n).fold(BigUint::from(0u8), |inner_acc, j| {
                    let idx = i + j;
                    let val = b.pow(idx) % q.clone();
                    inner_acc + val
                });
                acc + row_sum
            });

        let k_min = lb.clone() / q.clone();
        let k_max = ub.clone() / q.clone();
        let u_max = BigUint::from(1u8) << log2up(&(k_min.clone() + k_max.clone()));

        let first_bound = lb + (u_max.clone() - k_min.clone()) * q.clone();
        let second_bound = ub + k_min.clone() * q.clone();
        let max_bound = ark_std::cmp::max(first_bound, second_bound);
        let num_bits = log2up(&(max_bound / p));
        ark_std::println!("num_bits: {}", num_bits);
        let m = BigUint::from(1u8) << num_bits;

        let lb_m = (b.clone() - BigUint::from(1u8))
            * (0..n)
                .fold(
                    (BigUint::from(0u8), BigUint::from(1u8)),
                    |(acc, power), _| {
                        let term = &power % q.clone() % m.clone();
                        let next_power = &power * b.clone();
                        (acc + term, next_power)
                    },
                )
                .0;

        let ub_m = (b.clone() - BigUint::from(1u8))
            * (b.clone() - BigUint::from(1u8))
            * (0..n).fold(BigUint::from(0u8), |acc, i| {
                let row_sum = (0..n).fold(BigUint::from(0u8), |inner_acc, j| {
                    let idx = i + j;
                    let val = b.pow(idx) % q.clone() % m.clone();
                    inner_acc + val
                });
                acc + row_sum
            });

        let l_min = (lb_m.clone() + u_max.clone() * (q.clone() % m.clone())
            - ((k_min.clone() * q.clone()) % m.clone()))
        .div_floor(&m);
        let l_max = (ub_m.clone() + ((k_min.clone() * q.clone()) % m.clone())).div_floor(&m);

        let x : Fq377= MontFp!("218393408942992446968589193493746660101651787560689350338764189588519393175121782177906966561079408675464506489966");
        let y : Fq377 = MontFp!("122268283598675559488486339158635529096981886914877139579534153582033676785385790730042363341236035746924960903179");
        let z = x * y;

        let x_limbs = biguint_to_limbs::<Fr254>(&x.0.into(), base as usize, n as usize);
        let y_limbs = biguint_to_limbs::<Fr254>(&y.0.into(), base as usize, n as usize);
        let z_limbs = biguint_to_limbs::<Fr254>(&z.0.into(), base as usize, n as usize);

        ark_std::println!("x_limbs: {:?}", x_limbs);
        ark_std::println!("y_limbs: {:?}", y_limbs);
        ark_std::println!("z_limbs: {:?}", z_limbs);

        let t = (b.clone() - BigUint::from(1u8))
            * (b.clone() - BigUint::from(1u8))
            * (0..n).fold(BigUint::from(0u8), |acc, i| {
                let row_sum = (0..n).fold(BigUint::from(0u8), |inner_acc, j| {
                    let idx = i + j;
                    let val = (b.pow(idx) % q.clone())
                        * BigUint::from(x_limbs[i as usize].0)
                        * BigUint::from(y_limbs[j as usize].0);
                    inner_acc + val
                });
                acc + row_sum
            })
            - (b.clone() - BigUint::from(1u8))
                * (0..n)
                    .fold(
                        (BigUint::from(0u8), BigUint::from(1u8)),
                        |(acc, power), i| {
                            let term = &power % q.clone() * BigUint::from(z_limbs[i as usize].0);
                            let next_power = &power * b.clone();
                            (acc + term, next_power)
                        },
                    )
                    .0;

        let k = t.div_floor(&q);

        ark_std::println!("k_min: {}", k_min);
        ark_std::println!("k_max: {}", k_max);
        ark_std::println!("k: {}", k);

        let u = k.clone() - k_min.clone();
        let t_ = (b.clone() - BigUint::from(1u8))
            * (b.clone() - BigUint::from(1u8))
            * (0..n).fold(BigUint::from(0u8), |acc, i| {
                let row_sum = (0..n).fold(BigUint::from(0u8), |inner_acc, j| {
                    let idx = i + j;
                    let val = ((b.pow(idx) % q.clone()) % m.clone())
                        * BigUint::from(x_limbs[i as usize].0)
                        * BigUint::from(y_limbs[j as usize].0);
                    inner_acc + val
                });
                acc + row_sum
            })
            - (b.clone() - BigUint::from(1u8))
                * (0..n)
                    .fold(
                        (BigUint::from(0u8), BigUint::from(1u8)),
                        |(acc, power), i| {
                            let term = ((&power % q.clone()) % m.clone())
                                * BigUint::from(z_limbs[i as usize].0);
                            let next_power = &power * b.clone();
                            (acc + term, next_power)
                        },
                    )
                    .0
            - u * (q.clone() % m.clone())
            + (k_min.clone() * q.clone()) % m.clone();

        let l = t_.div_floor(&m.clone());
        ark_std::println!("l_min: {}", l_min);
        ark_std::println!("l_max: {}", l_max);
        ark_std::println!("l: {}", l);

        ark_std::println!("m: {}", m);
        ark_std::println!("q: {}", q);
    }

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
        //test_emulated_add_helper::<Fq377, Fr254>();
        //test_emulated_add_helper::<Fr254, Fq254>();
        test_emulated_add_helper::<Fq254, Fr254>();
        //test_emulated_add_helper::<Fr377, Fq377>();
        //test_emulated_add_helper::<Fr381, Fq381>();
        //test_emulated_add_helper::<Fr761, Fq761>();
    }

    fn to_big<F: PrimeField, E: EmulationConfig<F>>(v: E) -> BigUint {
        <E as Into<BigUint>>::into(v)
    }

    fn test_emulated_add_helper<E, F>()
    where
        E: EmulationConfig<F>,
        F: PrimeField,
    {
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(16);
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

        //// ========
        ///

        const N_OPERANDS: usize = 100; // any N ≤ B works
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(8);

        // first operand is public so that we have public inputs to feed later
        let pub_val = E::one();
        let var_pub = circuit.create_public_emulated_variable(pub_val).unwrap();

        // remaining private operands: 2, 3, 4 …
        let mut operands = ark_std::vec![var_pub];
        let mut acc_big = to_big(pub_val); // running integer sum

        for i in 1..=N_OPERANDS {
            let val = E::from((i as u64) + 1); // 2, 3, 4, …
            acc_big += to_big(val.clone());
            let var = circuit.create_emulated_variable(val).unwrap();
            operands.push(var);
        }

        // expected result  Σ operands  (mod q)
        let q_big: BigUint = E::MODULUS.into();
        acc_big %= &q_big;
        let expected = E::from(acc_big.clone());
        let var_result = circuit.create_emulated_variable(expected).unwrap();
        let batch_constraints = circuit.num_gates(); // or .row_count(), .gate_count() …
        ark_std::println!("batch-add gate  →  {batch_constraints} constraints");
        // constrain with the new gate
        circuit
            .emulated_batch_add_precise_gate(&operands, &var_result)
            .unwrap();

        // feed public inputs (limbs of the single public operand)
        let public_inputs = from_emulated_field(pub_val);
        assert!(circuit.check_circuit_satisfiability(&public_inputs).is_ok());

        ////////
        ///
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(16);

        // first operand is public so that we have public inputs to feed later
        let pub_val = E::one();

        // first operand is public so that we have public inputs to feed later
        let pub_val = E::one();
        let var_pub = circuit.create_public_emulated_variable(pub_val).unwrap();

        // remaining private operands: 2, 3, 4 …
        let mut operands = ark_std::vec![var_pub];
        let mut acc_big = to_big(pub_val); // running integer sum

        for i in 1..=N_OPERANDS {
            let val = E::from((i as u64) + 1); // 2, 3, 4, …
            acc_big += to_big(val.clone());
            let var = circuit.create_emulated_variable(val).unwrap();
            operands.push(var);
        }

        // chain of ordinary add_gates
        let mut acc_var = operands[0].clone();
        for op in &operands[1..] {
            let tmp = circuit.emulated_add(&acc_var, op).unwrap();
            acc_var = tmp; // new accumulator
        }

        let seq_constraints = circuit.num_gates();
        ark_std::println!("sequential add_gates →  {seq_constraints} constraints");

        // feed public inputs (limbs of the single public operand)
        let public_inputs = from_emulated_field(pub_val);
        assert!(circuit.check_circuit_satisfiability(&public_inputs).is_ok());
    }

    #[test]
    fn test_emulated_mul() {
        test_emulated_mul_helper::<Fq254, Fr254>();
        /*test_emulated_mul_helper::<Fr254, Fq254>();
        test_emulated_mul_helper::<Fq377, Fr254>();
        test_emulated_mul_helper::<Fr377, Fq377>();
        test_emulated_mul_helper::<Fr381, Fq381>();
        test_emulated_mul_helper::<Fr761, Fq761>();*/

        // test for issue (https://github.com/EspressoSystems/jellyfish/issues/306)
        /*let x : Fq377= MontFp!("218393408942992446968589193493746660101651787560689350338764189588519393175121782177906966561079408675464506489966");
        let y : Fq377 = MontFp!("122268283598675559488486339158635529096981886914877139579534153582033676785385790730042363341236035746924960903179");

        let mut circuit = PlonkCircuit::<Fr254>::new_turbo_plonk();
        let var_x = circuit.create_emulated_variable(x).unwrap();
        let _ = circuit.emulated_mul_constant(&var_x, y).unwrap();
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());*/
    }

    fn test_emulated_mul_helper<E, F>()
    where
        E: EmulationConfig<F>,
        F: PrimeField,
    {
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(16);
        let x = E::from(6732u64);
        let y = E::from(E::MODULUS.into() - 12387u64);
        let expected = x * y;
        let var_x = circuit.create_public_emulated_variable(x).unwrap();
        let var_y = circuit.create_emulated_variable(y).unwrap();
        let sz = circuit.num_gates();
        let var_z = circuit.emulated_mul(&var_x, &var_y).unwrap();
        assert_eq!(circuit.emulated_witness(&var_x).unwrap(), x);
        assert_eq!(circuit.emulated_witness(&var_y).unwrap(), y);
        assert_eq!(circuit.emulated_witness(&var_z).unwrap(), expected);
        ark_std::println!("circuit size = {}", circuit.num_gates() - sz);
        assert!(circuit
            .check_circuit_satisfiability(&from_emulated_field(x))
            .is_ok());

        /*let var_y_z = circuit.emulated_mul(&var_y, &var_z).unwrap();
        assert_eq!(circuit.emulated_witness(&var_y_z).unwrap(), expected * y);
        assert!(circuit
            .check_circuit_satisfiability(&from_emulated_field(x))
            .is_ok());
        //ark_std::println!("circuit size = {}", circuit.num_gates() - sz);

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
            .is_err());*/
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
