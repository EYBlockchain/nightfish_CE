// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Gates and gadgets implementations
#![allow(unused_imports)]
pub mod ecc;
pub mod ultraplonk;

pub mod arithmetic;
pub mod cmp;
pub mod emulated;
pub mod logic;
pub mod range;

pub use emulated::*;

// Helper functions
mod utils;

/// Utils for test
pub mod test_utils {
    use crate::{errors::CircuitError, Arithmetization, Circuit, PlonkCircuit};
    use ark_bn254::Fq as Fq254;
    use ark_ff::PrimeField;

    /// two circuit with the same statement should have the same extended
    /// permutation polynomials even with different variable assignment
    pub fn test_variable_independence_for_circuit<F: PrimeField>(
        circuit_1: PlonkCircuit<F>,
        circuit_2: PlonkCircuit<F>,
    ) -> Result<(), CircuitError> {
        assert_eq!(circuit_1.num_gates(), circuit_2.num_gates());
        assert_eq!(circuit_1.num_vars(), circuit_2.num_vars());
        // Check extended permutation polynomials
        if ark_std::any::TypeId::of::<F>() != ark_std::any::TypeId::of::<Fq254>() {
            let sigma_polys_1 = circuit_1.compute_extended_permutation_polynomials()?;
            let sigma_polys_2 = circuit_2.compute_extended_permutation_polynomials()?;
            sigma_polys_1
                .iter()
                .zip(sigma_polys_2.iter())
                .for_each(|(p1, p2)| assert_eq!(p1, p2));
        } else {
            let sigma_polys_1 = circuit_1.compute_extended_permutation_mles()?;
            let sigma_polys_2 = circuit_2.compute_extended_permutation_mles()?;
            sigma_polys_1
                .iter()
                .zip(sigma_polys_2.iter())
                .for_each(|(p1, p2)| assert_eq!(p1, p2));
        }
        Ok(())
    }
}
