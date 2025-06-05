// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! This module implements rescue circuit with non-native arithmetics.
//! The overall structure of the module mimics what is in rescue.rs
//! The major adjustment is to move from `Variable`s (that are native to
//! a plonk circuit) to `FpElemVar`s that are non-native to the circuit.

use crate::rescue::{
    Permutation, RescueMatrix, RescueParameter, RescueVector, PRP, ROUNDS, STATE_SIZE,
};
use ark_ff::PrimeField;
use ark_std::{format, string::ToString, vec, vec::Vec};
use itertools::Itertools;
use jf_relation::{
    errors::CircuitError::{self, ParameterError},
    gadgets::{EmulatedVariable, EmulationConfig},
    PlonkCircuit,
};
use jf_utils::compute_len_to_next_multiple;

use super::{PermutationGadget, RescueGadget, SpongeStateVar};

/// Array of Emulayedvariables representing a Rescue state (4 field elements), and also
/// the modulus of the non-native evaluating field.
#[derive(Clone, Debug)]
pub struct RescueEmulatedStateVar<E: PrimeField>(pub(crate) [EmulatedVariable<E>; STATE_SIZE]);

/// Type wrapper for the RescueGadget over the non-native field.
pub type RescueEmulatedGadget<E, F> = dyn RescueGadget<RescueEmulatedStateVar<E>, E, F>;

impl<E: PrimeField, F: PrimeField> SpongeStateVar<E, F> for RescueEmulatedStateVar<E> {
    type Native = E;
    type NonNative = F;
    type Var = EmulatedVariable<E>;
}

impl<E, F> RescueGadget<RescueEmulatedStateVar<E>, E, F> for PlonkCircuit<F>
where
    F: PrimeField,
    E: EmulationConfig<F> + RescueParameter,
{
    fn rescue_permutation(
        &mut self,
        input_var: RescueEmulatedStateVar<E>,
    ) -> Result<RescueEmulatedStateVar<E>, CircuitError> {
        let permutation = Permutation::<E>::default();
        let keys = permutation.round_keys_ref();
        let keys = keys
            .iter()
            .map(|key| RescueVector::from(key.elems().as_slice()))
            .collect_vec();
        let mds_matrix = permutation.mds_matrix_ref();

        self.permutation_with_const_round_keys(input_var, mds_matrix, keys.as_slice())
    }

    fn prp(
        &mut self,
        key_var: &RescueEmulatedStateVar<E>,
        input_var: &RescueEmulatedStateVar<E>,
    ) -> Result<RescueEmulatedStateVar<E>, CircuitError> {
        let prp_instance = PRP::<E>::default();
        let mds_states = prp_instance.mds_matrix_ref();
        let keys_vars = self.key_schedule(mds_states, key_var, &prp_instance)?;
        self.prp_with_round_keys(input_var, mds_states, &keys_vars)
    }

    fn rescue_sponge_with_padding(
        &mut self,
        data_vars: &[EmulatedVariable<E>],
        num_output: usize,
    ) -> Result<Vec<EmulatedVariable<E>>, CircuitError> {
        if data_vars.is_empty() {
            return Err(ParameterError("empty data vars".to_string()));
        }
        let zero_var = self.emulated_zero();
        let rate = STATE_SIZE - 1;
        let data_len = compute_len_to_next_multiple(data_vars.len() + 1, rate);

        let data_vars: Vec<EmulatedVariable<E>> = [
            data_vars,
            &[self.emulated_one()],
            vec![zero_var; data_len - data_vars.len() - 1].as_ref(),
        ]
        .concat();

        RescueEmulatedGadget::<E, F>::rescue_sponge_no_padding(self, &data_vars, num_output)
    }

    fn rescue_sponge_no_padding(
        &mut self,
        data_vars: &[EmulatedVariable<E>],
        num_output: usize,
    ) -> Result<Vec<EmulatedVariable<E>>, CircuitError> {
        if (data_vars.is_empty()) || (data_vars.len() % (STATE_SIZE - 1) != 0) {
            return Err(ParameterError("data_vars".to_string()));
        }

        let rate = STATE_SIZE - 1;

        let zero_var = self.emulated_zero();

        // ABSORB PHASE
        let mut state_var = RescueEmulatedStateVar([
            data_vars[0].clone(),
            data_vars[1].clone(),
            data_vars[2].clone(),
            zero_var.clone(),
        ]);
        state_var = self.rescue_permutation(state_var)?;

        for block in data_vars[rate..].chunks_exact(rate) {
            state_var = self.add_state(
                &state_var,
                &RescueEmulatedStateVar([
                    block[0].clone(),
                    block[1].clone(),
                    block[2].clone(),
                    zero_var.clone(),
                ]),
            )?;
            state_var = self.rescue_permutation(state_var)?;
        }
        // SQUEEZE PHASE
        let mut result = vec![];
        let mut remaining = num_output;
        // extract current rate before calling PRP again
        loop {
            let extract = remaining.min(rate);
            result.extend_from_slice(&state_var.0[0..extract]);
            remaining -= extract;
            if remaining == 0 {
                break;
            }
            state_var = self.rescue_permutation(state_var)?;
        }

        Ok(result)
    }

    fn rescue_full_state_keyed_sponge_with_zero_padding(
        &mut self,
        key: EmulatedVariable<E>,
        data_vars: &[EmulatedVariable<E>],
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        if data_vars.is_empty() {
            return Err(ParameterError("empty data vars".to_string()));
        }

        let zero_var = self.emulated_zero();
        let data_vars = [
            data_vars,
            vec![
                zero_var;
                compute_len_to_next_multiple(data_vars.len(), STATE_SIZE) - data_vars.len()
            ]
            .as_ref(),
        ]
        .concat();

        RescueEmulatedGadget::<E, F>::rescue_full_state_keyed_sponge_no_padding(
            self, key, &data_vars,
        )
    }

    fn rescue_full_state_keyed_sponge_no_padding(
        &mut self,
        key: EmulatedVariable<E>,
        data_vars: &[EmulatedVariable<E>],
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        if data_vars.len() % STATE_SIZE != 0 || data_vars.is_empty() {
            return Err(ParameterError(format!(
                "Bad input length for FSKS circuit: {:}, it must be positive multiple of STATE_SIZE",
                data_vars.len()
            )));
        }

        let zero_var = self.emulated_zero();

        // set key
        let mut state = RescueEmulatedStateVar([zero_var.clone(), zero_var.clone(), zero_var, key]);

        // absorb phase
        let chunks = data_vars.chunks_exact(STATE_SIZE);
        for chunk in chunks {
            let chunk_var = RescueEmulatedStateVar([
                chunk[0].clone(),
                chunk[1].clone(),
                chunk[2].clone(),
                chunk[3].clone(),
            ]);
            state = PermutationGadget::<RescueEmulatedStateVar<E>, E, F>::add_state(
                self, &state, &chunk_var,
            )?;
            state = self.rescue_permutation(state)?;
        }
        // squeeze phase, but only a single output, can return directly from state
        Ok(state.0[0].clone())
    }

    fn create_rescue_state_variable(
        &mut self,
        state: &RescueVector<E>,
    ) -> Result<RescueEmulatedStateVar<E>, CircuitError> {
        // create vars for states
        let mut state_vec = Vec::new();
        for f in state.vec.iter() {
            state_vec.push(self.create_emulated_variable(*f)?);
        }

        Ok(RescueEmulatedStateVar(state_vec.try_into().map_err(
            |_| ParameterError("Failed to convert to fixed size array".to_string()),
        )?))
    }

    fn key_schedule(
        &mut self,
        mds: &RescueMatrix<E>,
        key_var: &RescueEmulatedStateVar<E>,
        prp_instance: &PRP<E>,
    ) -> Result<Vec<RescueEmulatedStateVar<E>>, CircuitError> {
        let mut aux = *prp_instance.init_vec_ref();
        let key_injection_vec = prp_instance.key_injection_vec_ref();

        let mut key_state_var = self.add_constant_state(key_var, &aux)?;
        let mut result = vec![key_state_var.clone()];

        for (r, key_injection_item) in key_injection_vec.iter().enumerate() {
            aux.linear(mds, key_injection_item);
            if r % 2 == 0 {
                key_state_var = self.pow_alpha_inv_state(&key_state_var)?;
                key_state_var = self.affine_transform(&key_state_var, mds, key_injection_item)?;
            } else {
                key_state_var =
                    self.non_linear_transform(&key_state_var, mds, key_injection_item)?;
            }
            result.push(key_state_var.clone());
        }

        Ok(result)
    }

    /// Return the variable corresponding to the output of the of the Rescue
    /// PRP where the rounds keys have already been computed "dynamically"
    /// * `input_var` - variable corresponding to the plain text
    /// * `mds_states` - Rescue MDS matrix
    /// * `key_vars` - variables corresponding to the scheduled keys
    /// * `returns` -
    fn prp_with_round_keys(
        &mut self,
        input_var: &RescueEmulatedStateVar<E>,
        mds: &RescueMatrix<E>,
        keys_vars: &[RescueEmulatedStateVar<E>],
    ) -> Result<RescueEmulatedStateVar<E>, CircuitError> {
        if (keys_vars.len() != 2 * ROUNDS + 1) || (mds.len() != STATE_SIZE) {
            return Err(CircuitError::ParameterError("data_vars".to_string()));
        }

        let zero_state = RescueVector::from(&[E::zero(); STATE_SIZE]);
        let mut state_var = self.add_state(input_var, &keys_vars[0])?;
        for (r, key_var) in keys_vars.iter().skip(1).enumerate() {
            if r % 2 == 0 {
                state_var = self.pow_alpha_inv_state(&state_var)?;
                state_var = self.affine_transform(&state_var, mds, &zero_state)?;
            } else {
                state_var = self.non_linear_transform(&state_var, mds, &zero_state)?;
            }

            state_var = self.add_state(&state_var, key_var)?;
        }
        Ok(state_var)
    }
}

impl<E, F> PermutationGadget<RescueEmulatedStateVar<E>, E, F> for PlonkCircuit<F>
where
    F: PrimeField,
    E: EmulationConfig<F> + RescueParameter,
{
    fn check_var_bound_rescue_state(
        &self,
        _emulated_rescue_state: &RescueEmulatedStateVar<E>,
    ) -> Result<(), CircuitError> {
        Ok(())
    }

    fn add_constant_state(
        &mut self,
        input_var: &RescueEmulatedStateVar<E>,
        constant: &RescueVector<E>,
    ) -> Result<RescueEmulatedStateVar<E>, CircuitError> {
        let res_vec = input_var
            .0
            .iter()
            .zip(constant.elems().iter())
            .map(|(var, elem)| self.emulated_add_constant(var, *elem))
            .collect::<Result<Vec<EmulatedVariable<E>>, CircuitError>>()?;
        Ok(RescueEmulatedStateVar(res_vec.try_into().map_err(
            |_| ParameterError("Failed to convert to fixed size array".to_string()),
        )?))
    }

    fn pow_alpha_inv_state(
        &mut self,
        input_var: &RescueEmulatedStateVar<E>,
    ) -> Result<RescueEmulatedStateVar<E>, CircuitError> {
        let res_vec = input_var
            .0
            .iter()
            .map(|var| {
                PermutationGadget::<RescueEmulatedStateVar<E>, E, F>::pow_alpha_inv(
                    self,
                    var.clone(),
                )
            })
            .collect::<Result<Vec<EmulatedVariable<E>>, CircuitError>>()?;
        Ok(RescueEmulatedStateVar(res_vec.try_into().map_err(
            |_| ParameterError("Failed to convert to fixed size array".to_string()),
        )?))
    }

    fn affine_transform(
        &mut self,
        input_var: &RescueEmulatedStateVar<E>,
        matrix: &RescueMatrix<E>,
        constant: &RescueVector<E>,
    ) -> Result<RescueEmulatedStateVar<E>, CircuitError> {
        let mut res_vec = Vec::<EmulatedVariable<E>>::new();

        for i in 0..STATE_SIZE {
            let mut sum = self.emulated_mul_constant(&input_var.0[0], matrix.vec(i).vec[0])?;
            for j in 1..STATE_SIZE {
                let mul_var = self.emulated_mul_constant(&input_var.0[j], matrix.vec(i).vec[j])?;
                sum = self.emulated_add(&sum, &mul_var)?;
            }
            sum = self.emulated_add_constant(&sum, constant.elems()[i])?;
            res_vec.push(sum);
        }

        Ok(RescueEmulatedStateVar(res_vec.try_into().map_err(
            |_| ParameterError("Failed to convert to fixed size array".to_string()),
        )?))
    }

    fn non_linear_transform(
        &mut self,
        input_var: &RescueEmulatedStateVar<E>,
        matrix: &RescueMatrix<E>,
        constant: &RescueVector<E>,
    ) -> Result<RescueEmulatedStateVar<E>, CircuitError> {
        let mut power_vec = Vec::<EmulatedVariable<E>>::new();
        for var in &input_var.0 {
            let power_var = if E::A == 5 {
                let sq_var = self.emulated_mul(var, var)?;
                let fourth_var = self.emulated_mul(&sq_var, &sq_var)?;
                self.emulated_mul(&fourth_var, var)
            } else if E::A == 11 {
                let sq_var = self.emulated_mul(var, var)?;
                let cube_var = self.emulated_mul(var, &sq_var)?;
                let fourth_var = self.emulated_mul(&sq_var, &sq_var)?;
                let output_eighth_var = self.emulated_mul(&fourth_var, &fourth_var)?;
                self.emulated_mul(&output_eighth_var, &cube_var)
            } else {
                Err(CircuitError::ParameterError(
                    "incorrect Rescue parameters".to_string(),
                ))
            }?;
            power_vec.push(power_var);
        }

        let state_var =
            RescueEmulatedStateVar(power_vec.try_into().map_err(|_| {
                ParameterError("Failed to convert to fixed size array".to_string())
            })?);

        self.affine_transform(&state_var, matrix, constant)
    }

    fn pow_alpha_inv(
        &mut self,
        input_var: EmulatedVariable<E>,
    ) -> Result<EmulatedVariable<E>, CircuitError> {
        let input_val = self.emulated_witness(&input_var)?;

        let output_val = input_val.pow(E::A_INV);
        let output_var = self.create_emulated_variable(output_val)?;
        if E::A == 5 {
            let output_sq_var = self.emulated_mul(&output_var, &output_var)?;
            let output_fourth_var = self.emulated_mul(&output_sq_var, &output_sq_var)?;
            self.emulated_mul_gate(&output_fourth_var, &output_var, &input_var)?;
            Ok(output_var)
        } else if E::A == 11 {
            let output_sq_var = self.emulated_mul(&output_var, &output_var)?;
            let output_cube_var = self.emulated_mul(&output_var, &output_sq_var)?;
            let output_fourth_var = self.emulated_mul(&output_sq_var, &output_sq_var)?;
            let output_eighth_var = self.emulated_mul(&output_fourth_var, &output_fourth_var)?;
            self.emulated_mul_gate(&output_eighth_var, &output_cube_var, &input_var)?;
            Ok(output_var)
        } else {
            Err(CircuitError::ParameterError(
                "incorrect Rescue parameters".to_string(),
            ))
        }
    }

    fn add_state(
        &mut self,
        left_state_var: &RescueEmulatedStateVar<E>,
        right_state_var: &RescueEmulatedStateVar<E>,
    ) -> Result<RescueEmulatedStateVar<E>, CircuitError> {
        let mut res_vec = Vec::<EmulatedVariable<E>>::new();

        for (left_var, right_var) in left_state_var.0.iter().zip(right_state_var.0.iter()) {
            res_vec.push(self.emulated_add(left_var, right_var)?);
        }
        Ok(RescueEmulatedStateVar(res_vec.try_into().map_err(
            |_| ParameterError("Failed to convert to fixed size array".to_string()),
        )?))
    }

    fn permutation_with_const_round_keys(
        &mut self,
        input_var: RescueEmulatedStateVar<E>,
        mds: &RescueMatrix<E>,
        round_keys: &[RescueVector<E>],
    ) -> Result<RescueEmulatedStateVar<E>, CircuitError> {
        if (round_keys.len() != 2 * ROUNDS + 1) || (mds.len() != STATE_SIZE) {
            return Err(CircuitError::ParameterError("data_vars".to_string()));
        }

        let mut state_var = self.add_constant_state(&input_var, &round_keys[0])?;
        for (r, key) in round_keys.iter().skip(1).enumerate() {
            if r % 2 == 0 {
                state_var = self.pow_alpha_inv_state(&state_var)?;
                state_var = self.affine_transform(&state_var, mds, key)?;
            } else {
                state_var = self.non_linear_transform(&state_var, mds, key)?;
            }
        }
        Ok(state_var)
    }
}

#[cfg(test)]
mod tests {

    use super::{PermutationGadget, RescueEmulatedGadget, RescueEmulatedStateVar, RescueGadget};
    use crate::rescue::{
        sponge::{RescueCRHF, RescuePRFCore},
        Permutation, RescueMatrix, RescueParameter, RescueVector, CRHF_RATE, PRP, STATE_SIZE,
    };
    use ark_bls12_377::Fq as Fq377;
    use ark_bn254::{Fq as Fq254, Fr as Fr254};
    use ark_ed_on_bls12_377::Fq as FqEd377;
    use ark_ff::PrimeField;
    use ark_std::{vec, vec::Vec};
    use itertools::Itertools;
    use jf_relation::{
        gadgets::{EmulatedVariable, EmulationConfig},
        Circuit, PlonkCircuit,
    };

    const RANGE_BIT_LEN_FOR_TEST: usize = 8;

    fn gen_state_matrix_constant<E: PrimeField>(
    ) -> (RescueVector<E>, RescueMatrix<E>, RescueVector<E>) {
        let state_in =
            RescueVector::from(&[E::from(12u32), E::from(2u32), E::from(8u32), E::from(9u32)]);

        let matrix = RescueMatrix::from(&[
            RescueVector::from(&[E::from(2u32), E::from(3u32), E::from(4u32), E::from(5u32)]),
            RescueVector::from(&[E::from(3u32), E::from(3u32), E::from(3u32), E::from(3u32)]),
            RescueVector::from(&[E::from(5u32), E::from(3u32), E::from(5u32), E::from(5u32)]),
            RescueVector::from(&[E::from(1u32), E::from(0u32), E::from(2u32), E::from(17u32)]),
        ]);

        let constant =
            RescueVector::from(&[E::from(2u32), E::from(3u32), E::from(4u32), E::from(5u32)]);

        (state_in, matrix, constant)
    }

    fn check_state<E, F>(
        circuit: &PlonkCircuit<F>,
        out_var: &RescueEmulatedStateVar<E>,
        out_value: &RescueVector<E>,
    ) where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        for i in 0..STATE_SIZE {
            assert_eq!(
                circuit.emulated_witness(&out_var.0[i]).unwrap(),
                out_value.elems()[i]
            );
        }
    }

    fn check_circuit_satisfiability<E, F>(
        circuit: &mut PlonkCircuit<F>,
        out_value: Vec<E>,
        out_var: RescueEmulatedStateVar<E>,
    ) where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        for (i, v) in out_value.iter().enumerate() {
            circuit.set_emulated_witness(&out_var.0[i], E::from(888_u32));
            assert!(circuit.check_circuit_satisfiability(&[]).is_err());
            circuit.set_emulated_witness(&out_var.0[i], *v);
            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        }
    }

    #[test]
    fn test_add_constant_state() {
        test_add_constant_state_helper::<FqEd377, Fq377>();
        test_add_constant_state_helper::<Fr254, Fq254>();
        test_add_constant_state_helper::<Fq254, Fr254>()
    }
    fn test_add_constant_state_helper<E, F>()
    where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);

        let state = RescueVector::from(&[E::from(12_u32), E::one(), E::one(), E::one()]);
        let constant = RescueVector::from(&[E::zero(), E::one(), E::one(), E::one()]);

        let input_var = circuit.create_rescue_state_variable(&state).unwrap();
        let out_var = circuit.add_constant_state(&input_var, &constant).unwrap();

        let out_value: Vec<E> = (0..STATE_SIZE)
            .map(|i| constant.elems()[i] + state.elems()[i])
            .collect();

        check_state(
            &circuit,
            &out_var,
            &RescueVector::from(out_value.as_slice()),
        );

        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        // Alter the input state
        circuit.set_emulated_witness(&input_var.0[0], E::from(0_u32));
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());

        // Restablish the input state
        circuit.set_emulated_witness(&input_var.0[0], state.elems()[0]);
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        // Alter the output state
        circuit.set_emulated_witness(&out_var.0[1], E::from(888_u32));
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
    }

    #[test]
    fn test_state_inversion() {
        test_state_inversion_helper::<FqEd377, Fq377>();
        test_state_inversion_helper::<Fr254, Fq254>();
        test_state_inversion_helper::<Fq254, Fr254>()
    }
    fn test_state_inversion_helper<E, F>()
    where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);

        let state =
            RescueVector::from(&[E::from(12u32), E::from(2u32), E::from(8u32), E::from(9u32)]);

        let input_var = circuit.create_rescue_state_variable(&state).unwrap();
        let out_var = circuit.pow_alpha_inv_state(&input_var).unwrap();

        let out_value: Vec<E> = (0..STATE_SIZE)
            .map(|i| state.elems()[i].pow(E::A_INV))
            .collect();

        check_state(
            &circuit,
            &out_var,
            &RescueVector::from(out_value.as_slice()),
        );

        check_circuit_satisfiability(&mut circuit, out_value, out_var);
    }

    #[test]
    fn test_affine_transformation() {
        test_affine_transformation_helper::<FqEd377, Fq377>();
        test_affine_transformation_helper::<Fr254, Fq254>();
        test_affine_transformation_helper::<Fq254, Fr254>()
    }

    fn test_affine_transformation_helper<E, F>()
    where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();

        let (state_in, matrix, constant) = gen_state_matrix_constant();

        let input_var: RescueEmulatedStateVar<E> =
            circuit.create_rescue_state_variable(&state_in).unwrap();

        let out_var = circuit
            .affine_transform(&input_var, &matrix, &constant)
            .unwrap();

        let mut out_value = state_in;
        out_value.linear(&matrix, &constant);

        check_state(&circuit, &out_var, &out_value);

        check_circuit_satisfiability(&mut circuit, out_value.elems(), out_var);
    }

    #[test]
    fn test_non_linear_transformation() {
        test_non_linear_transformation_helper::<FqEd377, Fq377>();
        test_non_linear_transformation_helper::<Fr254, Fq254>();
        test_non_linear_transformation_helper::<Fq254, Fr254>()
    }

    fn test_non_linear_transformation_helper<E, F>()
    where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();

        let (state_in, matrix, constant) = gen_state_matrix_constant();

        let input_var: RescueEmulatedStateVar<E> =
            circuit.create_rescue_state_variable(&state_in).unwrap();

        let out_var = circuit
            .non_linear_transform(&input_var, &matrix, &constant)
            .unwrap();

        let mut out_value = state_in;
        out_value.non_linear(&matrix, &constant);

        check_state(&circuit, &out_var, &out_value);

        check_circuit_satisfiability(&mut circuit, out_value.elems(), out_var);
    }

    #[test]
    fn test_rescue_perm() {
        test_rescue_perm_helper::<FqEd377, Fq377>();
        test_rescue_perm_helper::<Fr254, Fq254>();
        test_rescue_perm_helper::<Fq254, Fr254>()
    }

    fn test_rescue_perm_helper<E, F>()
    where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();

        let state_in =
            RescueVector::from(&[E::from(1u32), E::from(2u32), E::from(3u32), E::from(4u32)]);

        let state_in_var = circuit.create_rescue_state_variable(&state_in).unwrap();

        let perm = Permutation::default();
        let state_out = perm.eval(&state_in);

        let out_var = circuit.rescue_permutation(state_in_var).unwrap();

        check_state(&circuit, &out_var, &state_out);

        check_circuit_satisfiability(&mut circuit, state_out.elems(), out_var);
    }

    #[test]
    fn test_add_state() {
        test_add_state_helper::<FqEd377, Fq377>();
        test_add_state_helper::<Fr254, Fq254>();
        test_add_state_helper::<Fq254, Fr254>()
    }

    fn test_add_state_helper<E, F>()
    where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();

        let state1 = RescueVector::from(&[
            E::from(12_u32),
            E::from(7_u32),
            E::from(4_u32),
            E::from(3_u32),
        ]);

        let state2 = RescueVector::from(&[
            E::from(1_u32),
            E::from(2_u32),
            E::from(2555_u32),
            E::from(888_u32),
        ]);

        let input1_var = circuit.create_rescue_state_variable(&state1).unwrap();
        let input2_var = circuit.create_rescue_state_variable(&state2).unwrap();
        let out_var = circuit.add_state(&input1_var, &input2_var).unwrap();

        let out_value: Vec<E> = (0..STATE_SIZE)
            .map(|i| state1.elems()[i] + state2.elems()[i])
            .collect();

        check_state(
            &circuit,
            &out_var,
            &RescueVector::from(out_value.as_slice()),
        );

        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        // Alter the input state
        circuit.set_emulated_witness(&input1_var.0[0], E::from(0_u32));
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());

        // Re-establish the input state
        circuit.set_emulated_witness(&input1_var.0[0], state1.elems()[0]);
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        // Alter the input state
        circuit.set_emulated_witness(&input2_var.0[0], E::from(0_u32));
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());

        // Re-establish the input state
        circuit.set_emulated_witness(&input2_var.0[0], state2.elems()[0]);
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        // Alter the output state
        circuit.set_emulated_witness(&out_var.0[1], E::from(777_u32));
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
    }

    #[test]
    fn test_prp() {
        test_prp_helper::<FqEd377, Fq377>();
        test_prp_helper::<Fr254, Fq254>();
        test_prp_helper::<Fq254, Fr254>()
    }

    fn test_prp_helper<E, F>()
    where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);
        let prp = PRP::default();
        let mut prng = jf_utils::test_rng();
        let key_vec = RescueVector::from(&[
            E::rand(&mut prng),
            E::rand(&mut prng),
            E::rand(&mut prng),
            E::rand(&mut prng),
        ]);
        let input_vec = RescueVector::from(&[
            E::rand(&mut prng),
            E::rand(&mut prng),
            E::rand(&mut prng),
            E::rand(&mut prng),
        ]);
        let key_var = circuit.create_rescue_state_variable(&key_vec).unwrap();
        let input_var = circuit.create_rescue_state_variable(&input_vec).unwrap();
        let out_var = circuit.prp(&key_var, &input_var).unwrap();

        let out_val = prp.prp(&key_vec, &input_vec);

        // Check consistency between witness[input_var] and input_vec
        check_state(&circuit, &input_var, &input_vec);

        // Check consistency between witness[key_var] and key_vec
        check_state(&circuit, &key_var, &key_vec);

        // Check consistency between witness[out_var] and rescue cipher output
        check_state(&circuit, &out_var, &out_val);

        // Check good witness
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        // Check bad witness
        // Alter the input state
        circuit.set_emulated_witness(&key_var.0[0], E::from(0_u32));
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
    }

    #[test]
    fn test_rescue_sponge_no_padding_single_output() {
        test_rescue_sponge_no_padding_single_output_helper::<FqEd377, Fq377>();
        test_rescue_sponge_no_padding_single_output_helper::<Fr254, Fq254>();
        test_rescue_sponge_no_padding_single_output_helper::<Fq254, Fr254>()
    }

    fn test_rescue_sponge_no_padding_single_output_helper<E, F>()
    where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);

        let mut prng = jf_utils::test_rng();
        let data = (0..112 * CRHF_RATE)
            .map(|_| E::rand(&mut prng))
            .collect_vec();
        let data_vars = data
            .iter()
            .map(|&x| circuit.create_emulated_variable(x).unwrap())
            .collect_vec();

        let expected_sponge = RescueCRHF::sponge_no_padding(&data, 1).unwrap()[0];
        let sponge_var = &RescueEmulatedGadget::<E, F>::rescue_sponge_no_padding(
            &mut circuit,
            data_vars.as_slice(),
            1,
        )
        .unwrap()[0];

        assert_eq!(
            expected_sponge,
            circuit.emulated_witness(sponge_var).unwrap()
        );

        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        circuit.set_emulated_witness(sponge_var, E::from(1_u32));
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());

        // If the data length is not a multiple of RATE==3 then an error is triggered
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();

        let size = 2 * CRHF_RATE + 1; // Non multiple of RATE
        let data = (0..size).map(|_| E::rand(&mut prng)).collect_vec();
        let data_vars = data
            .iter()
            .map(|&x| circuit.create_emulated_variable(x).unwrap())
            .collect_vec();

        assert!(RescueEmulatedGadget::<E, F>::rescue_sponge_no_padding(
            &mut circuit,
            data_vars.as_slice(),
            1
        )
        .is_err());
    }

    #[test]
    fn test_rescue_sponge_no_padding() {
        test_rescue_sponge_no_padding_helper::<FqEd377, Fq377>();
        test_rescue_sponge_no_padding_helper::<Fr254, Fq254>();
        test_rescue_sponge_no_padding_helper::<Fq254, Fr254>()
    }

    fn test_rescue_sponge_no_padding_helper<E, F>()
    where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);

        let rate = 3;

        let input_vec = vec![E::from(11_u32), E::from(144_u32), E::from(87_u32)];
        let input_var = [
            circuit.create_emulated_variable(input_vec[0]).unwrap(),
            circuit.create_emulated_variable(input_vec[1]).unwrap(),
            circuit.create_emulated_variable(input_vec[2]).unwrap(),
        ];

        for output_len in 1..10 {
            let out_var = RescueEmulatedGadget::<E, F>::rescue_sponge_no_padding(
                &mut circuit,
                &input_var,
                output_len,
            )
            .unwrap();

            // Check consistency between inputs
            for i in 0..rate {
                assert_eq!(
                    input_vec[i],
                    circuit.emulated_witness(&input_var[i]).unwrap()
                );
            }

            // Check consistency between outputs
            let expected_hash = RescueCRHF::sponge_no_padding(&input_vec, output_len).unwrap();

            for (e, f) in out_var.iter().zip(expected_hash.iter()) {
                assert_eq!(*f, circuit.emulated_witness(e).unwrap());
            }

            // Check constraints
            // good path
            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

            // bad path: incorrect output
            let w = circuit.emulated_witness(&out_var[0]).unwrap();
            circuit.set_emulated_witness(&out_var[0], E::from(1_u32));
            assert!(circuit.check_circuit_satisfiability(&[]).is_err());
            circuit.set_emulated_witness(&out_var[0], w);
        }

        // bad path: incorrect number of inputs
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);
        let input_vec = [
            E::from(11_u32),
            E::from(144_u32),
            E::from(87_u32),
            E::from(45_u32),
        ];
        let input_var = [
            circuit.create_emulated_variable(input_vec[0]).unwrap(),
            circuit.create_emulated_variable(input_vec[1]).unwrap(),
            circuit.create_emulated_variable(input_vec[2]).unwrap(),
            circuit.create_emulated_variable(input_vec[3]).unwrap(),
        ];
        assert!(RescueEmulatedGadget::<E, F>::rescue_sponge_no_padding(
            &mut circuit,
            &input_var,
            1
        )
        .is_err());
    }

    #[test]
    fn test_rescue_sponge_with_padding() {
        test_rescue_sponge_with_padding_helper::<FqEd377, Fq377>();
        test_rescue_sponge_with_padding_helper::<Fr254, Fq254>();
        test_rescue_sponge_with_padding_helper::<Fq254, Fr254>()
    }
    fn test_rescue_sponge_with_padding_helper<E, F>()
    where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        for input_len in 1..10 {
            for output_len in 1..10 {
                let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(RANGE_BIT_LEN_FOR_TEST);

                let input_vec: Vec<E> = (0..input_len).map(|i| E::from((i + 10) as u32)).collect();
                let input_var: Vec<EmulatedVariable<E>> = input_vec
                    .iter()
                    .map(|x| circuit.create_emulated_variable(*x).unwrap())
                    .collect();

                let num_gates = circuit.num_gates();

                let out_var = RescueEmulatedGadget::<E, F>::rescue_sponge_with_padding(
                    &mut circuit,
                    &input_var,
                    output_len,
                )
                .unwrap();

                ark_std::println!(
                    "num_gates: {}, input_len: {}, output_len: {}",
                    circuit.num_gates() - num_gates,
                    input_len,
                    output_len
                );

                // Check consistency between inputs
                for i in 0..input_len {
                    assert_eq!(
                        input_vec[i],
                        circuit.emulated_witness(&input_var[i]).unwrap()
                    );
                }

                // Check consistency between outputs
                let expected_hash = RescueCRHF::sponge_with_bit_padding(&input_vec, output_len);

                for (&e, f) in expected_hash.iter().zip(out_var.iter()) {
                    assert_eq!(e, circuit.emulated_witness(f).unwrap());
                }

                // Check constraints
                // good path
                assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
                // bad path: incorrect output
                let w = circuit.emulated_witness(&out_var[0]).unwrap();
                circuit.set_emulated_witness(&out_var[0], E::from(1_u32));
                assert!(circuit.check_circuit_satisfiability(&[]).is_err());
                circuit.set_emulated_witness(&out_var[0], w);
            }
        }
    }

    #[test]
    fn test_fsks() {
        test_fsks_helper::<FqEd377, Fq377>();
        test_fsks_helper::<Fr254, Fq254>();
        test_fsks_helper::<Fq254, Fr254>()
    }
    fn test_fsks_helper<E, F>()
    where
        F: PrimeField,
        E: EmulationConfig<F> + RescueParameter,
    {
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        let mut prng = jf_utils::test_rng();
        let key = E::rand(&mut prng);
        let key_var = circuit.create_emulated_variable(key).unwrap();
        let input_len = 8;
        let data: Vec<E> = (0..input_len).map(|_| E::rand(&mut prng)).collect_vec();
        let data_vars: Vec<EmulatedVariable<E>> = data
            .iter()
            .map(|&x| circuit.create_emulated_variable(x).unwrap())
            .collect_vec();

        let expected_fsks_output =
            RescuePRFCore::full_state_keyed_sponge_no_padding(&key, &data, 1).unwrap();

        let fsks_var = RescueEmulatedGadget::<E, F>::rescue_full_state_keyed_sponge_no_padding(
            &mut circuit,
            key_var.clone(),
            &data_vars,
        )
        .unwrap();

        // Check prf output consistency
        assert_eq!(
            expected_fsks_output[0],
            circuit.emulated_witness(&fsks_var).unwrap()
        );

        // Check constraints
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        circuit.set_emulated_witness(&fsks_var, E::from(1_u32));
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());

        // make data_vars of bad length
        let mut data_vars = data_vars;
        data_vars.push(circuit.emulated_zero());
        assert!(
            RescueEmulatedGadget::<E, F>::rescue_full_state_keyed_sponge_no_padding(
                &mut circuit,
                key_var,
                &data_vars
            )
            .is_err()
        );
    }
}
