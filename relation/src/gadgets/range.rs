// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Implemtation of circuit lookup related gadgets

use super::utils::next_multiple;
use crate::{
    constants::GATE_WIDTH, errors::CircuitError, BoolVar, Circuit, PlonkCircuit, Variable,
};
use ark_ff::{BigInteger, PrimeField};
use ark_std::{format, string::ToString, vec::Vec};

impl<F: PrimeField> PlonkCircuit<F> {
    /// Constrain a variable to be within the [0, 2^`bit_len`) range
    /// Return error if the variable is invalid.
    pub fn enforce_in_range(&mut self, a: Variable, bit_len: usize) -> Result<(), CircuitError> {
        if self.support_lookup() {
            self.range_gate_with_lookup(a, bit_len)?;
        } else {
            self.range_gate_internal(a, bit_len)?;
        }
        Ok(())
    }

    /// Return a boolean variable indicating whether variable `a` is in the
    /// range [0, 2^`bit_len`). Return error if the variable is invalid.
    /// TODO: optimize the gate for UltraPlonk.
    pub fn is_in_range(&mut self, a: Variable, bit_len: usize) -> Result<BoolVar, CircuitError> {
        let a_bit_le: Vec<BoolVar> = self.unpack(a, F::MODULUS_BIT_SIZE as usize)?;
        let a_bit_le: Vec<Variable> = a_bit_le.into_iter().map(|b| b.into()).collect();
        // a is in range if and only if the bits in `a_bit_le[bit_len..]` are all
        // zeroes.
        let higher_bit_sum = self.sum(&a_bit_le[bit_len..])?;
        self.is_zero(higher_bit_sum)
    }

    /// Obtain the `bit_len`-long binary representation of variable `a`
    /// Return a list of variables [b0, ..., b_`bit_len`] which is the binary
    /// representation of `a`.
    /// Return error if the `a` is not the range of [0, 2^`bit_len`).
    pub fn unpack(&mut self, a: Variable, bit_len: usize) -> Result<Vec<BoolVar>, CircuitError> {
        if bit_len < F::MODULUS_BIT_SIZE as usize
            && self.witness(a)? >= F::from(2u32).pow([bit_len as u64])
        {
            return Err(CircuitError::ParameterError(
                "Failed to unpack variable to a range of smaller than 2^bit_len".to_string(),
            ));
        }
        self.range_gate_internal(a, bit_len)
    }

    /// a general decomposition gate (not necessarily binary decomposition)
    /// where `a` are enforced to decomposed to `a_chunks_le` which consists
    /// of chunks (multiple bits) in little-endian order and
    /// each chunk \in [0, `range_size`)
    /// Note, the chunks are not constrained here. This must be done elsewhere.
    pub(crate) fn decomposition_gate(
        &mut self,
        a_chunks_le: Vec<Variable>,
        a: Variable,
        range_size: F,
    ) -> Result<(), CircuitError> {
        let mut coeffs = Vec::<F>::new();
        let mut coeff = F::one();
        for _ in 0..a_chunks_le.len() {
            coeffs.push(coeff);
            coeff *= range_size;
        }

        self.lin_comb_gate(&coeffs, &F::zero(), &a_chunks_le, &a)?;
        Ok(())
    }
}

/// Private helper function for range gate
impl<F: PrimeField> PlonkCircuit<F> {
    // internal of a range check gate
    pub(crate) fn range_gate_internal(
        &mut self,
        a: Variable,
        bit_len: usize,
    ) -> Result<Vec<BoolVar>, CircuitError> {
        self.check_var_bound(a)?;
        if bit_len == 0 {
            return Err(CircuitError::ParameterError(
                "Only allows positive bit length for range upper bound".to_string(),
            ));
        }

        let a_bits_le: Vec<bool> = self.witness(a)?.into_bigint().to_bits_le();
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

        self.binary_decomposition_gate(a_bits_le.clone(), a)?;

        Ok(a_bits_le)
    }

    fn binary_decomposition_gate(
        &mut self,
        a_bits_le: Vec<BoolVar>,
        a: Variable,
    ) -> Result<(), CircuitError> {
        let a_chunks_le: Vec<Variable> = a_bits_le.into_iter().map(|b| b.into()).collect();
        self.decomposition_gate(a_chunks_le, a, 2u8.into())?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use core::num;

    use crate::{
        errors::CircuitError, gadgets::test_utils::test_variable_independence_for_circuit, Circuit,
        PlonkCircuit,
    };
    use ark_bls12_377::Fq as Fq377;
    use ark_bn254::{Fq as Fq254, Fr as Fr254};
    use ark_ed_on_bls12_377::Fq as FqEd377;
    use ark_ed_on_bls12_381::Fq as FqEd381;
    use ark_ed_on_bn254::Fq as FqEd254;
    use ark_ff::PrimeField;

    #[test]
    fn test_unpack() -> Result<(), CircuitError> {
        test_unpack_helper::<Fr254>()?;
        test_unpack_helper::<Fq254>()?;
        test_unpack_helper::<FqEd254>()?;
        test_unpack_helper::<FqEd377>()?;
        test_unpack_helper::<FqEd381>()?;
        test_unpack_helper::<Fq377>()
    }

    fn test_unpack_helper<F: PrimeField>() -> Result<(), CircuitError> {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let a = circuit.create_variable(F::one())?;
        let b = circuit.create_variable(F::from(1023u32))?;

        circuit.enforce_in_range(a, 1)?;
        let a_le = circuit.unpack(a, 3)?;
        assert_eq!(a_le.len(), 3);
        let b_le = circuit.unpack(b, 10)?;
        assert_eq!(b_le.len(), 10);
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        assert!(circuit.unpack(b, 9).is_err());
        Ok(())
    }

    #[test]
    fn test_range_gate() -> Result<(), CircuitError> {
        test_range_gate_helper::<Fr254>()?;
        test_range_gate_helper::<Fq254>()?;
        test_range_gate_helper::<FqEd254>()?;
        test_range_gate_helper::<FqEd377>()?;
        test_range_gate_helper::<FqEd381>()?;
        test_range_gate_helper::<Fq377>()
    }
    fn test_range_gate_helper<F: PrimeField>() -> Result<(), CircuitError> {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let a = circuit.create_variable(F::one())?;
        let b = circuit.create_variable(F::from(1023u32))?;

        circuit.enforce_in_range(a, 1)?;
        circuit.enforce_in_range(a, 3)?;
        circuit.enforce_in_range(b, 10)?;
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        circuit.enforce_in_range(b, 9)?;
        assert!(circuit.enforce_in_range(a, 0).is_err());
        // non-positive bit length is undefined, thus fail
        assert!(circuit.enforce_in_range(a, 0).is_err());
        // bit length bigger than that of a field element (bit length takes 256 or 381
        // bits)
        let bit_len = (F::MODULUS_BIT_SIZE as usize / 8 + 1) * 8;
        assert!(circuit.enforce_in_range(a, bit_len + 1).is_err());
        // if mess up the wire value, should fail
        *circuit.witness_mut(b) = F::from(1024u32);
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit.enforce_in_range(circuit.num_vars(), 10).is_err());

        // build two fixed circuits with different variable assignments, checking that
        // the arithmetized extended permutation polynomial is variable
        // independent
        let circuit_1 = build_range_gate_circuit(F::from(314u32))?;
        let circuit_2 = build_range_gate_circuit(F::from(489u32))?;
        test_variable_independence_for_circuit(circuit_1, circuit_2)?;

        Ok(())
    }

    fn build_range_gate_circuit<F: PrimeField>(a: F) -> Result<PlonkCircuit<F>, CircuitError> {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let a_var = circuit.create_variable(a)?;
        circuit.enforce_in_range(a_var, 10)?;
        if ark_std::any::TypeId::of::<F>() != ark_std::any::TypeId::of::<Fq254>() {
            circuit.finalize_for_arithmetization()?;
        } else {
            circuit.finalize_for_mle_arithmetization()?;
        }
        Ok(circuit)
    }
}
