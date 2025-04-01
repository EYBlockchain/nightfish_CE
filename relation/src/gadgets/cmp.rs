// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Comparison gadgets for circuit

use crate::{errors::CircuitError, gates::QuadPolyGate, BoolVar, Circuit, PlonkCircuit, Variable};
use ark_ff::{BigInteger, PrimeField};
use ark_std::{boxed::Box, collections, vec::Vec};
use itertools::izip;
use num_bigint::BigUint;

impl<F: PrimeField> PlonkCircuit<F> {
    /// Constrain that `a` < `b`.
    pub fn enforce_lt(&mut self, a: Variable, b: Variable) -> Result<(), CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        self.check_var_bound(b)?;
        self.enforce_lt_internal(a, b)
    }

    /// Constrain that `a` <= `b`
    pub fn enforce_leq(&mut self, a: Variable, b: Variable) -> Result<(), CircuitError>
    where
        F: PrimeField,
    {
        let c = self.is_lt(b, a)?;
        self.enforce_constant(c.0, F::zero())
    }

    /// Constrain that `a` > `b`.
    pub fn enforce_gt(&mut self, a: Variable, b: Variable) -> Result<(), CircuitError>
    where
        F: PrimeField,
    {
        self.enforce_lt(b, a)
    }

    /// Constrain that `a` >= `b`.
    pub fn enforce_geq(&mut self, a: Variable, b: Variable) -> Result<(), CircuitError>
    where
        F: PrimeField,
    {
        let c = self.is_lt(a, b)?;
        self.enforce_constant(c.into(), F::zero())
    }

    /// Returns a `BoolVar` indicating whether `a` < `b`.
    pub fn is_lt(&mut self, a: Variable, b: Variable) -> Result<BoolVar, CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        self.check_var_bound(b)?;
        self.is_lt_internal(a, b)
    }

    /// Returns a `BoolVar` indicating whether `a` > `b`.
    pub fn is_gt(&mut self, a: Variable, b: Variable) -> Result<BoolVar, CircuitError>
    where
        F: PrimeField,
    {
        self.is_lt(b, a)
    }

    /// Returns a `BoolVar` indicating whether `a` <= `b`.
    pub fn is_leq(&mut self, a: Variable, b: Variable) -> Result<BoolVar, CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        self.check_var_bound(b)?;
        let c = self.is_lt_internal(b, a)?;
        self.logic_neg(c)
    }

    /// Returns a `BoolVar` indicating whether `a` >= `b`.
    pub fn is_geq(&mut self, a: Variable, b: Variable) -> Result<BoolVar, CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        self.check_var_bound(b)?;
        let c = self.is_lt_internal(a, b)?;
        self.logic_neg(c)
    }

    /// Returns a `BoolVar` indicating whether the variable `a` is less than a
    /// given constant `val`.
    pub fn is_lt_constant(&mut self, a: Variable, val: F) -> Result<BoolVar, CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        let neg_a = self.lin_comb(&[-F::one()], &-F::one(), &[a])?;
        let neg_val = -F::one() - val;
        self.is_gt_constant(neg_a, neg_val)
    }

    /// Returns a `BoolVar` indicating whether the variable `a` is less than or
    /// equal to a given constant `val`.
    pub fn is_leq_constant(&mut self, a: Variable, val: F) -> Result<BoolVar, CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        let c = self.is_gt_constant(a, val)?;
        self.logic_neg(c)
    }

    /// Returns a `BoolVar` indicating whether the variable `a` is greater than
    /// a given constant `val`.
    pub fn is_gt_constant(&mut self, a: Variable, val: F) -> Result<BoolVar, CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        self.is_gt_constant_internal(a, &val)
    }

    /// Returns a `BoolVar` indicating whether the variable `a` is greater than
    /// or equal to a given constant `val`.
    pub fn is_geq_constant(&mut self, a: Variable, val: F) -> Result<BoolVar, CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        let c = self.is_lt_constant(a, val)?;
        self.logic_neg(c)
    }

    /// Enforce the variable `a` to be less than a
    /// given constant `val`.
    pub fn enforce_lt_constant(&mut self, a: Variable, val: F) -> Result<(), CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        let c = self.is_lt_constant(a, val)?;
        self.enforce_true(c.into())
    }

    /// Enforce the variable `a` to be less than or
    /// equal to a given constant `val`.
    pub fn enforce_leq_constant(&mut self, a: Variable, val: F) -> Result<(), CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        let c = self.is_gt_constant(a, val)?;
        self.enforce_false(c.into())
    }

    /// Enforce the variable `a` to be greater than
    /// a given constant `val`.
    pub fn enforce_gt_constant(&mut self, a: Variable, val: F) -> Result<(), CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        let c = self.is_gt_constant(a, val)?;
        self.enforce_true(c.into())
    }

    /*/// Returns a [`BoolVar`] indicating whether the variable `a` is less than `b` using the range wire.
    pub fn is_lt_lookup(&mut self, a: Variable, b: Variable) -> Result<BoolVar, CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        self.check_var_bound(b)?;
        self.is_lt_lookup_internal(a, b)
    }*/

    /*/// Enforce that `a` is less than `b` using the range wire.
    pub fn enforce_lt_lookup(&mut self, a: Variable, b: Variable) -> Result<(), CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        self.check_var_bound(b)?;
        let res = self.is_lt_internal(a, b)?;
        self.enforce_true(res.into())
    }*/

    /// Enforce the variable `a` to be greater than
    /// or equal a given constant `val`.
    pub fn enforce_geq_constant(&mut self, a: Variable, val: F) -> Result<(), CircuitError>
    where
        F: PrimeField,
    {
        self.check_var_bound(a)?;
        let c = self.is_lt_constant(a, val)?;
        self.enforce_false(c.into())
    }
}

/// Private helper functions for comparison gate
impl<F: PrimeField> PlonkCircuit<F> {
    /// Returns 2 `BoolVar`s.
    /// First indicates whether `a` <= (q-1)/2 and `b` > (q-1)/2.
    /// Second indicates whether `a` and `b` are both <= (q-1)/2
    /// or both > (q-1)/2.
    fn msb_check_internal(
        &mut self,
        a: Variable,
        b: Variable,
    ) -> Result<(BoolVar, BoolVar), CircuitError> {
        let a_gt_const = self.is_gt_constant_internal(a, &F::from(F::MODULUS_MINUS_ONE_DIV_TWO))?;
        let b_gt_const = self.is_gt_constant_internal(b, &F::from(F::MODULUS_MINUS_ONE_DIV_TWO))?;
        let a_leq_const = self.logic_neg(a_gt_const)?;
        // Check whether `a` <= (q-1)/2 and `b` > (q-1)/2
        let msb_check = self.logic_and(a_leq_const, b_gt_const)?;
        // Check whether `a` and `b` are both <= (q-1)/2 or
        // are both > (q-1)/2
        let msb_eq = self.is_equal(a_gt_const.into(), b_gt_const.into())?;
        Ok((msb_check, msb_eq))
    }

    /// Return a variable indicating whether `a` < `b`.
    fn is_lt_internal(&mut self, a: Variable, b: Variable) -> Result<BoolVar, CircuitError> {
        let (msb_check, msb_eq) = self.msb_check_internal(a, b)?;
        // check whether (a-b) > (q-1)/2
        let c = self.sub(a, b)?;
        let cmp_result = self.is_gt_constant_internal(c, &F::from(F::MODULUS_MINUS_ONE_DIV_TWO))?;
        let cmp_result = self.logic_and(msb_eq, cmp_result)?;

        self.logic_or(msb_check, cmp_result)
    }

    /// Constrain that `a` < `b`
    fn enforce_lt_internal(&mut self, a: Variable, b: Variable) -> Result<(), CircuitError> {
        let (msb_check, msb_eq) = self.msb_check_internal(a, b)?;
        // check whether (a-b) <= (q-1)/2
        let c = self.sub(a, b)?;
        let cmp_result = self.is_gt_constant_internal(c, &F::from(F::MODULUS_MINUS_ONE_DIV_TWO))?;
        let cmp_result = self.logic_and(msb_eq, cmp_result)?;

        self.logic_or_gate(msb_check, cmp_result)
    }

    /// Helper function to check whether `a` is greater than a given
    /// constant. Let N = F::MODULUS_BIT_SIZE, it assumes that the
    /// constant < 2^N. And it uses at most N AND/OR gates.
    fn is_gt_constant_internal(
        &mut self,
        a: Variable,
        constant: &F,
    ) -> Result<BoolVar, CircuitError> {
        if self.support_lookup() {
            return self.is_gt_constant_lookup_internal(a, constant);
        }
        let a_bits_le = self.unpack(a, F::MODULUS_BIT_SIZE as usize)?;
        let const_bits_le = constant.into_bigint().to_bits_le();

        // Iterating from LSB to MSB. Skip the front consecutive 1's.
        // Put an OR gate for bit 0 and an AND gate for bit 1.
        let mut zipped = const_bits_le
            .into_iter()
            .chain(ark_std::iter::repeat(false))
            .take(a_bits_le.len())
            .zip(a_bits_le.iter())
            .skip_while(|(b, _)| *b);
        if let Some((_, &var)) = zipped.next() {
            zipped.try_fold(var, |current, (b, a)| -> Result<BoolVar, CircuitError> {
                if b {
                    self.logic_and(*a, current)
                } else {
                    self.logic_or(*a, current)
                }
            })
        } else {
            // the constant is all one
            Ok(BoolVar(self.zero()))
        }
    }

    // Function designed specifically for when lookups are enabled to
    // determine whether a variable is greater than a constant.
    fn is_gt_constant_lookup_internal(
        &mut self,
        a: Variable,
        constant: &F,
    ) -> Result<BoolVar, CircuitError> {
        if *constant + F::one() == F::zero() {
            return Ok(BoolVar(self.zero()));
        }

        let big_int_const_0: BigUint = constant.into_bigint().into() + BigUint::from(1u8);
        let num_bits_0 = big_int_const_0.bits() as usize - 1;
        let big_int_power_constant_0: BigUint = BigUint::from(1_u8) << num_bits_0;
        let power_constant_0 = F::from(big_int_power_constant_0.clone());
        let shifted_a = self.add_constant(a, &(power_constant_0 - *constant - F::one()))?;

        let big_int_const_1: BigUint = F::MODULUS.into() - big_int_const_0;
        let num_bits_1 = big_int_const_1.bits() as usize - 1;
        let big_int_power_constant_1: BigUint = BigUint::from(1_u8) << num_bits_1;
        let power_constant_1 = F::from(big_int_power_constant_1.clone());
        let minus_a_minus_one = self.lin_comb(
            &[-F::one(), F::zero(), F::zero(), F::zero()],
            &-F::one(),
            &[a, self.zero(), self.zero(), self.zero()],
        )?;
        let shifted_minus_a = self.add_constant(
            minus_a_minus_one,
            &(power_constant_1 + *constant + F::one()),
        )?;

        // (Below we set n_0 := num_bits_0, n_1 := num_bits_1 and C := constant.)
        // We check if `a < 2^n_0` or if `a + 2^n_0 - C - 1 < 2^n_0`.
        // Then `a <= C` if and only if one of these inequality holds.
        // We do this by using conditional select to make sure a passing inequality is always checked if `a <= C` holds.
        // This avoids us calling `range_check_with_lookup` twice.
        let a_big_uint: BigUint = self.witness(a)?.into();
        let sel_0 = if a_big_uint < big_int_power_constant_0 {
            self.create_boolean_variable(false)?
        } else {
            self.create_boolean_variable(true)?
        };
        let a_0 = self.conditional_select(sel_0, a, shifted_a)?;
        let bool_0 = self.range_check_with_lookup(a_0, num_bits_0)?;

        // Next we check if `-a-1 < 2^n_1` or if `-a + 2^n_1 + C < 2^n_1`.
        // Then `a > C` if and only if one of these inequality holds.
        // Again, we do this by using conditional select to make sure a passing inequality is always checked if `a > C` holds.
        let minus_a_minus_one_uint: BigUint = self.witness(minus_a_minus_one)?.into();
        let sel_1 = if minus_a_minus_one_uint < big_int_power_constant_1 {
            self.create_boolean_variable(false)?
        } else {
            self.create_boolean_variable(true)?
        };
        let a_1 = self.conditional_select(sel_1, minus_a_minus_one, shifted_minus_a)?;
        let bool_1 = self.range_check_with_lookup(a_1, num_bits_1)?;

        // We enforce that precisely one of the above checks returns a true but not both.
        // This is necessary to avoid having two false results.
        // (Recall that `range_check_with_lookup` can only be trusted if it returns true.)
        self.add_gate(bool_0.into(), bool_1.into(), self.one())?;
        Ok(bool_1)
    }
}

#[cfg(test)]
#[allow(non_local_definitions)]
mod test {
    use crate::{errors::CircuitError, BoolVar, Circuit, PlonkCircuit};
    use ark_bls12_377::Fq as Fq377;
    use ark_ed_on_bls12_377::Fq as FqEd377;
    use ark_ed_on_bls12_381::Fq as FqEd381;
    use ark_ed_on_bn254::Fq as FqEd254;
    use ark_ff::{
        fields::{Field, MontBackend, MontConfig},
        Fp, PrimeField,
    };
    use ark_std::{cmp::Ordering, One, UniformRand, Zero};
    use itertools::multizip;
    use num_bigint::BigUint;

    #[derive(MontConfig)]
    #[modulus = "11"]
    #[generator = "2"]
    pub struct F11Config;
    pub type F11 = Fp<MontBackend<F11Config, 1>, 1>;

    #[derive(MontConfig)]
    #[modulus = "13"]
    #[generator = "2"]
    pub struct F13Config;
    pub type F13 = Fp<MontBackend<F13Config, 1>, 1>;

    #[derive(MontConfig)]
    #[modulus = "17"]
    #[generator = "3"]
    pub struct F17Config;
    pub type F17 = Fp<MontBackend<F17Config, 1>, 1>;

    #[test]
    fn test_cmp_gates() -> Result<(), CircuitError> {
        test_cmp_helper::<F11>()?;
        test_cmp_helper::<F13>()?;
        test_cmp_helper::<F17>()?;
        test_cmp_helper::<FqEd254>()?;
        test_cmp_helper::<FqEd377>()?;
        test_cmp_helper::<FqEd381>()?;
        test_cmp_helper::<Fq377>()
    }

    #[test]
    fn test_lookup_constant_cmp_gates() -> Result<(), CircuitError> {
        lookup_gt_constant_helper::<F11>()?;
        lookup_gt_constant_helper::<F13>()?;
        lookup_gt_constant_helper::<F17>()?;
        lookup_gt_constant_helper::<FqEd254>()?;
        lookup_gt_constant_helper::<FqEd377>()?;
        lookup_gt_constant_helper::<FqEd381>()?;
        lookup_gt_constant_helper::<Fq377>()
    }

    #[test]
    fn test_lookup_cmp_gates() -> Result<(), CircuitError> {
        lookup_gt_helper::<F11>()?;
        lookup_gt_helper::<F13>()?;
        lookup_gt_helper::<F17>()?;
        lookup_gt_helper::<FqEd254>()?;
        lookup_gt_helper::<FqEd377>()?;
        lookup_gt_helper::<FqEd381>()?;
        lookup_gt_helper::<Fq377>()
    }

    fn test_cmp_helper<F: PrimeField>() -> Result<(), CircuitError> {
        let list = [
            (F::from(5u32), F::from(5u32)),
            (F::from(1u32), F::from(2u32)),
            (
                F::from(F::MODULUS_MINUS_ONE_DIV_TWO).add(F::one()),
                F::from(2u32),
            ),
            (
                F::from(F::MODULUS_MINUS_ONE_DIV_TWO).add(F::one()),
                F::from(F::MODULUS_MINUS_ONE_DIV_TWO).mul(F::from(2u32)),
            ),
        ];
        multizip((
            list,
            [Ordering::Less, Ordering::Greater],
            [false, true],
            [false, true],
        )).try_for_each(
                |((a, b), ordering, should_also_check_equality,
                 is_b_constant)|
                 -> Result<(), CircuitError> {
                    test_enforce_cmp_helper(&a, &b, ordering, should_also_check_equality, is_b_constant)?;
                    test_enforce_cmp_helper(&b, &a, ordering, should_also_check_equality, is_b_constant)?;
                    test_is_cmp_helper(&a, &b, ordering, should_also_check_equality, is_b_constant)?;
                    test_is_cmp_helper(&b, &a, ordering, should_also_check_equality, is_b_constant)
                },
            )
    }

    fn test_is_cmp_helper<F: PrimeField>(
        a: &F,
        b: &F,
        ordering: Ordering,
        should_also_check_equality: bool,
        is_b_constant: bool,
    ) -> Result<(), CircuitError> {
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        let expected_result = if a.cmp(b) == ordering
            || (a.cmp(b) == Ordering::Equal && should_also_check_equality)
        {
            F::one()
        } else {
            F::zero()
        };
        let a = circuit.create_variable(*a)?;
        let c: BoolVar = if is_b_constant {
            match ordering {
                Ordering::Less => {
                    if should_also_check_equality {
                        circuit.is_leq_constant(a, *b)?
                    } else {
                        circuit.is_lt_constant(a, *b)?
                    }
                },
                Ordering::Greater => {
                    if should_also_check_equality {
                        circuit.is_geq_constant(a, *b)?
                    } else {
                        circuit.is_gt_constant(a, *b)?
                    }
                },
                // Equality test will be handled elsewhere, comparison gate test will not enter here
                Ordering::Equal => circuit.create_boolean_variable_unchecked(expected_result)?,
            }
        } else {
            let b = circuit.create_variable(*b)?;
            match ordering {
                Ordering::Less => {
                    if should_also_check_equality {
                        circuit.is_leq(a, b)?
                    } else {
                        circuit.is_lt(a, b)?
                    }
                },
                Ordering::Greater => {
                    if should_also_check_equality {
                        circuit.is_geq(a, b)?
                    } else {
                        circuit.is_gt(a, b)?
                    }
                },
                // Equality test will be handled elsewhere, comparison gate test will not enter here
                Ordering::Equal => circuit.create_boolean_variable_unchecked(expected_result)?,
            }
        };
        assert!(circuit.witness(c.into())?.eq(&expected_result));
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        Ok(())
    }
    fn test_enforce_cmp_helper<F: PrimeField>(
        a: &F,
        b: &F,
        ordering: Ordering,
        should_also_check_equality: bool,
        is_b_constant: bool,
    ) -> Result<(), CircuitError> {
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        let expected_result =
            a.cmp(b) == ordering || (a.cmp(b) == Ordering::Equal && should_also_check_equality);
        let a = circuit.create_variable(*a)?;
        if is_b_constant {
            match ordering {
                Ordering::Less => {
                    if should_also_check_equality {
                        circuit.enforce_leq_constant(a, *b)?
                    } else {
                        circuit.enforce_lt_constant(a, *b)?
                    }
                },
                Ordering::Greater => {
                    if should_also_check_equality {
                        circuit.enforce_geq_constant(a, *b)?
                    } else {
                        circuit.enforce_gt_constant(a, *b)?
                    }
                },
                // Equality test will be handled elsewhere, comparison gate test will not enter here
                Ordering::Equal => (),
            }
        } else {
            let b = circuit.create_variable(*b)?;
            match ordering {
                Ordering::Less => {
                    if should_also_check_equality {
                        circuit.enforce_leq(a, b)?
                    } else {
                        circuit.enforce_lt(a, b)?
                    }
                },
                Ordering::Greater => {
                    if should_also_check_equality {
                        circuit.enforce_geq(a, b)?
                    } else {
                        circuit.enforce_gt(a, b)?
                    }
                },
                // Equality test will be handled elsewhere, comparison gate test will not enter here
                Ordering::Equal => (),
            }
        };
        if expected_result {
            assert!(circuit.check_circuit_satisfiability(&[]).is_ok())
        } else {
            assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        }
        Ok(())
    }

    fn lookup_gt_helper<F: PrimeField>() -> Result<(), CircuitError> {
        let mut rng = ark_std::test_rng();
        for _ in 0..500 {
            let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(12);
            let a_val = F::rand(&mut rng);
            let b_val = F::rand(&mut rng);
            let expected = a_val < b_val;
            let a = circuit.create_variable(a_val)?;
            let b = circuit.create_variable(b_val)?;
            let res = circuit.is_lt(a, b)?;
            let eq_res = circuit.is_lt(a, a)?;

            assert_eq!(circuit.witness(res.0)?, F::from(expected));
            assert_eq!(circuit.witness(eq_res.0)?, F::zero());

            circuit.check_circuit_satisfiability(&[])?;
        }
        Ok(())
    }

    fn lookup_gt_constant_helper<F: PrimeField>() -> Result<(), CircuitError> {
        let mut rng = ark_std::test_rng();
        for _ in 0..500 {
            let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(12);
            let a_val = F::rand(&mut rng);
            let b_val = F::rand(&mut rng);

            let expected = a_val > b_val;
            let a = circuit.create_variable(a_val)?;
            let res = circuit.is_gt_constant_lookup_internal(a, &b_val)?;
            assert_eq!(circuit.witness(res.0)?, F::from(expected));

            circuit.check_circuit_satisfiability(&[])?;
        }
        Ok(())
    }
}
