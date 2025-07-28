use ark_std::println;

use crate::poseidon::{sponge::CRHF_RATE, PoseidonParams, STATE_SIZE};
use ark_std::{string::ToString, vec::Vec};
use jf_relation::{errors::CircuitError, Circuit, PlonkCircuit, Variable};

use super::PoseidonHashGadget;

#[derive(Clone, Debug)]
/// Array of variables representing a Poseidon state (4 field elements).
pub struct PoseidonStateVar(pub(crate) [Variable; STATE_SIZE]);

pub trait SpongePoseidonHashGadget<F: PoseidonParams>: PoseidonHashGadget<F> {
    fn absorb(
        &mut self,
        state_var: &PoseidonStateVar,
        input_var: &[Variable],
    ) -> Result<PoseidonStateVar, CircuitError>;

    fn squeeze(
        &mut self,
        state_vars: &PoseidonStateVar,
        num_elements: usize,
    ) -> Result<Vec<Variable>, CircuitError>;
}

impl<F: PoseidonParams> SpongePoseidonHashGadget<F> for PlonkCircuit<F> {
    fn absorb(
        &mut self,
        state_vars: &PoseidonStateVar,
        input_vars: &[Variable],
    ) -> Result<PoseidonStateVar, CircuitError> {
        let mut output_vars = state_vars.0;
        println!("absorbing {} input variables", input_vars.len());
        for chunk in input_vars.chunks(CRHF_RATE) {
            // To be consistent with `Poseidon::hash()`, we need start adding elements in position `STATE_SIZE - CRHF_RATE`.
            for (output_var, input_var) in output_vars
                .iter_mut()
                .skip(STATE_SIZE - CRHF_RATE)
                .zip(chunk.iter())
            {
                *output_var = self.add(*output_var, *input_var)?;
            }
            println!(
                "calling poseidon_perm with {} state variables",
                output_vars.len()
            );
            output_vars = self.poseidon_perm(&output_vars)?.try_into().map_err(|_| {
                CircuitError::ParameterError(
                    "Failed to convert Poseidon state to array".to_string(),
                )
            })?;
        }
        Ok(PoseidonStateVar(output_vars))
    }

    fn squeeze(
        &mut self,
        state_vars: &PoseidonStateVar,
        num_elements: usize,
    ) -> Result<Vec<Variable>, CircuitError> {
        let mut result_vars = Vec::<Variable>::new();
        let mut state_vars_vec = state_vars.0;
        // SQUEEZE PHASE
        let mut remaining = num_elements;
        // extract current rate before calling Poseidon permutation again
        loop {
            let extract = remaining.min(CRHF_RATE);
            result_vars.extend_from_slice(&state_vars_vec[0..extract]);
            // always permute
            state_vars_vec = self
                .poseidon_perm(&state_vars_vec)?
                .try_into()
                .map_err(|_| {
                    CircuitError::ParameterError(
                        "Failed to convert Poseidon state to array".to_string(),
                    )
                })?;
            remaining -= extract;
            if remaining == 0 {
                break;
            }
        }
        Ok(result_vars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_crypto_primitives::sponge::{CryptographicSponge, FieldBasedCryptographicSponge};
    use ark_std::{test_rng, UniformRand};
    use jf_relation::Circuit;

    use crate::poseidon::{sponge::PoseidonSponge, PoseidonPerm};

    #[test]
    // The purpose of this test is to compare the sponge poseidon hash result from plonk circuit with
    // result calculated outside of circuit.
    fn test_sponge_poseidon_hash_plonk_gadget() -> Result<(), CircuitError> {
        let max_input_length = 25;
        let min_input_length = 10;
        let max_squeeze_length = 9;
        let min_squeeze_length = 3;
        let sponge_perm = PoseidonPerm::perm().unwrap();
        let mut rng = test_rng();
        for _ in 0..25 {
            let mut circuit: PlonkCircuit<Fr> = PlonkCircuit::new_turbo_plonk();
            let input_length =
                (usize::rand(&mut rng) % (max_input_length - min_input_length)) + min_input_length;
            let mut pos_state_vars = PoseidonStateVar([circuit.zero(); STATE_SIZE]);
            let input_vals = (0..input_length)
                .map(|_| Fr::rand(&mut rng))
                .collect::<Vec<Fr>>();
            let input_vars = input_vals
                .iter()
                .map(|&x| circuit.create_variable(x))
                .collect::<Result<Vec<Variable>, CircuitError>>()?;

            pos_state_vars = circuit.absorb(&pos_state_vars, &input_vars)?;
            let squeeze_length = (usize::rand(&mut rng)
                % (max_squeeze_length - min_squeeze_length))
                + min_squeeze_length;
            let output_vars = circuit.squeeze(&pos_state_vars, squeeze_length)?;

            let mut sponge = PoseidonSponge::<Fr, CRHF_RATE>::new(&sponge_perm);
            sponge.absorb(&input_vals);
            let output_vals = sponge.squeeze_native_field_elements(squeeze_length);

            circuit.check_circuit_satisfiability(&[])?;
            for (output_var, output_val) in output_vars.iter().zip(output_vals.iter()) {
                assert_eq!(circuit.witness(*output_var)?, *output_val);
            }
        }
        Ok(())
    }
}
