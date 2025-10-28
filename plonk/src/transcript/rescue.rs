// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! This module is a defines rescue transcript.
use super::{CircuitTranscript, Transcript};
use crate::errors::PlonkError;

use ark_ff::{BigInteger, PrimeField};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{any::TypeId, cmp::max, marker::PhantomData, vec::Vec};
use jf_primitives::{
    circuit::rescue::RescueNativeGadget,
    crhf::{VariableLengthRescueCRHF, CRHF},
    rescue::{sponge::RescueCRHF, RescueParameter, STATE_SIZE},
};
use jf_relation::{errors::CircuitError, gadgets::ecc::HasTEForm, Circuit, PlonkCircuit, Variable};
use jf_utils::bytes_to_field_elements;

/// Transcript with rescue hash function.
///
/// It is currently implemented simply as
/// - an append only vector of field elements
/// - a state that is initialized with 0
///
/// We keep appending new elements to the transcript vector,
/// and when a challenge is to be generated,
/// we reset the state with the fresh challenge.
///
/// 1. state: \[F: STATE_SIZE\] = hash(state|transcript)
/// 2. challenge = state\[0\]
/// 3. transcript = vec!\[challenge\]
#[derive(Debug, Clone, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct RescueTranscript<F>
where
    F: RescueParameter + ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize,
{
    transcript: Vec<F>,
    state: [F; STATE_SIZE],
}

impl<F> Transcript for RescueTranscript<F>
where
    F: RescueParameter,
{
    fn new_transcript(_label: &'static [u8]) -> Self {
        RescueTranscript {
            transcript: Vec::new(),
            state: [F::zero(); STATE_SIZE],
        }
    }

    fn push_message<S: CanonicalSerialize + ?Sized + 'static>(
        &mut self,
        _label: &'static [u8],
        msg: &S,
    ) -> Result<(), PlonkError> {
        // We remove the labels for better efficiency
        if TypeId::of::<S>() == TypeId::of::<F>() {
            let mut writer = Vec::new();
            msg.serialize_compressed(&mut writer)?;
            let msg = F::from_le_bytes_mod_order(writer.as_slice());
            self.transcript.push(msg);
            Ok(())
        } else {
            let mut writer = Vec::new();
            msg.serialize_compressed(&mut writer)?;

            let f = bytes_to_field_elements(writer.as_slice());

            self.transcript.extend_from_slice(&f[1..]);
            Ok(())
        }
    }

    fn squeeze_scalar_challenge<E>(
        &mut self,
        _label: &'static [u8],
    ) -> Result<E::ScalarField, PlonkError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField,
    {
        // 1. state: [F: STATE_SIZE] = hash(state|transcript)
        // 2. challenge = state[0] in Fr
        // 3. transcript = Vec::new()

        let input = [self.state.as_ref(), self.transcript.as_ref()].concat();
        let tmp: [F; STATE_SIZE] = VariableLengthRescueCRHF::evaluate(&input)?;

        // Find the byte length of the scalar field (minus one).
        let field_bytes_length = (E::ScalarField::MODULUS_BIT_SIZE as usize - 1) / 8;
        let challenge = E::ScalarField::from_le_bytes_mod_order(
            tmp[0]
                .into_bigint()
                .to_bytes_le()
                .iter()
                .take(field_bytes_length)
                .copied()
                .collect::<Vec<u8>>()
                .as_slice(),
        );

        self.state.copy_from_slice(&tmp);
        self.transcript = Vec::new();

        Ok(challenge)
    }
    fn squeeze_scalar_challenges<E>(
        &mut self,
        _label: &'static [u8],
        number_of_challenges: usize,
    ) -> Result<Vec<E::ScalarField>, PlonkError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField,
    {
        // 1. state: [F: STATE_SIZE] = hash(state|transcript)
        // 2. challenge = state[0] in Fr
        // 3. transcript = Vec::new()
        let num_outputs = max(4, number_of_challenges);
        let input = [self.state.as_ref(), self.transcript.as_ref()].concat();
        let tmp = RescueCRHF::sponge_with_bit_padding(&input, num_outputs);

        // Find the byte length of the scalar field (minus one).
        let field_bytes_length = (E::ScalarField::MODULUS_BIT_SIZE as usize - 1) / 8;

        let challenges = tmp
            .iter()
            .take(number_of_challenges)
            .map(|g_elem| {
                E::ScalarField::from_le_bytes_mod_order(
                    g_elem
                        .into_bigint()
                        .to_bytes_le()
                        .iter()
                        .take(field_bytes_length)
                        .copied()
                        .collect::<Vec<u8>>()
                        .as_slice(),
                )
            })
            .collect::<Vec<_>>();

        self.state.copy_from_slice(&tmp[tmp.len() - STATE_SIZE..]);
        self.transcript = Vec::new();

        Ok(challenges)
    }

    fn merge(&mut self, other: &Self) -> Result<(), PlonkError> {
        self.transcript.extend_from_slice(&other.state);
        self.transcript.extend_from_slice(&other.transcript);
        Ok(())
    }
}

/// Circuit variable used for a Plonk transcript making use of emulated variables.
#[derive(Clone, Debug)]
pub struct RescueTranscriptVar<F: RescueParameter> {
    pub(crate) transcript_var: Vec<Variable>,
    state_var: [Variable; 4],
    _phantom: PhantomData<F>,
}

impl<F> CircuitTranscript<F> for RescueTranscriptVar<F>
where
    F: PrimeField + RescueParameter,
{
    fn new_transcript(circuit: &mut PlonkCircuit<F>) -> Self {
        Self {
            transcript_var: Vec::new(),
            state_var: [circuit.zero(); 4],
            _phantom: PhantomData,
        }
    }

    fn push_message<S: CanonicalSerialize + ?Sized + 'static, E>(
        &mut self,
        _label: &'static [u8],
        msg: &S,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), PlonkError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField,
    {
        // We remove the labels for better efficiency
        if TypeId::of::<S>() == TypeId::of::<F>() {
            let mut writer = Vec::new();
            msg.serialize_compressed(&mut writer)?;
            let msg = F::from_le_bytes_mod_order(writer.as_slice());
            let msg_var = circuit.create_variable(msg)?;
            self.transcript_var.push(msg_var);
            Ok(())
        } else {
            let mut writer = Vec::new();
            msg.serialize_compressed(&mut writer)?;

            let f = bytes_to_field_elements(writer.as_slice());

            for e in f[1..].iter() {
                let e_var = circuit.create_variable(*e)?;
                self.transcript_var.push(e_var);
            }
            Ok(())
        }
    }

    fn push_variable(&mut self, var: &Variable) -> Result<(), CircuitError> {
        self.transcript_var.push(*var);
        Ok(())
    }

    fn squeeze_scalar_challenge<E>(
        &mut self,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<Variable, CircuitError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField,
    {
        // step 1. state: [F: STATE_SIZE] = hash(state|transcript)
        let input_var = [self.state_var.as_ref(), self.transcript_var.as_ref()].concat();
        let res_var =
            RescueNativeGadget::<F>::rescue_sponge_with_padding(circuit, &input_var, 4).unwrap();
        let out_var = res_var[0];

        // Find the byte length of the scalar field (minus one).
        let field_bytes_length = (E::ScalarField::MODULUS_BIT_SIZE as usize - 1) / 8;

        let value = circuit.witness(out_var)?;
        let bytes = value.into_bigint().to_bytes_le();
        let (challenge, leftover) = bytes.split_at(field_bytes_length);

        let challenge_var = circuit.create_variable(F::from_le_bytes_mod_order(challenge))?;
        let leftover_var = circuit.create_variable(F::from_le_bytes_mod_order(leftover))?;

        let bits = field_bytes_length * 8;
        let leftover_bits = F::MODULUS_BIT_SIZE as usize - bits;
        let coeff = F::from(2u32).pow([bits as u64]);

        circuit.enforce_in_range(challenge_var, bits)?;
        circuit.enforce_in_range(leftover_var, leftover_bits)?;

        circuit.lc_gate(
            &[
                challenge_var,
                leftover_var,
                circuit.zero(),
                circuit.zero(),
                out_var,
            ],
            &[F::one(), coeff, F::zero(), F::zero()],
        )?;

        self.state_var.copy_from_slice(&res_var[0..STATE_SIZE]);
        self.transcript_var = Vec::new();

        Ok(challenge_var)
    }

    fn squeeze_scalar_challenges<E>(
        &mut self,
        number_of_challenges: usize,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<Vec<Variable>, CircuitError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField,
    {
        let num_outputs = max(4, number_of_challenges);

        // step 1. state: [F: STATE_SIZE] = hash(state|transcript)
        let input_var = [self.state_var.as_ref(), self.transcript_var.as_ref()].concat();
        let res_vars =
            RescueNativeGadget::<F>::rescue_sponge_with_padding(circuit, &input_var, num_outputs)
                .unwrap();

        // step 2. challenge = state[0] in Fr

        // Here we check if the scalar field is smaller than the base field. If so, we truncate the output.
        // Find the byte length of the scalar field (minus one).
        let field_bytes_length = (E::ScalarField::MODULUS_BIT_SIZE as usize - 1) / 8;

        let bits = field_bytes_length * 8;
        let leftover_bits = F::MODULUS_BIT_SIZE as usize - bits;
        let coeff = F::from(2u32).pow([bits as u64]);

        let mut challenges = Vec::new();
        for res in res_vars.iter().take(number_of_challenges) {
            let value = circuit.witness(*res)?;
            let bytes = value.into_bigint().to_bytes_le();
            let (challenge, leftover) = bytes.split_at(field_bytes_length);

            let challenge_var = circuit.create_variable(F::from_le_bytes_mod_order(challenge))?;
            let leftover_var = circuit.create_variable(F::from_le_bytes_mod_order(leftover))?;

            circuit.enforce_in_range(challenge_var, bits)?;
            circuit.enforce_in_range(leftover_var, leftover_bits)?;

            circuit.lc_gate(
                &[
                    challenge_var,
                    leftover_var,
                    circuit.zero(),
                    circuit.zero(),
                    *res,
                ],
                &[F::one(), coeff, F::zero(), F::zero()],
            )?;

            challenges.push(challenge_var);
        }

        // 3. transcript = vec![challenge]
        // finish and update the states
        self.state_var
            .copy_from_slice(&res_vars[res_vars.len() - STATE_SIZE..]);
        self.transcript_var = Vec::new();

        Ok(challenges)
    }

    fn merge(&mut self, other: &Self) -> Result<(), CircuitError> {
        self.transcript_var.extend_from_slice(&other.state_var);
        self.transcript_var.extend_from_slice(&other.transcript_var);
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use ark_bls12_377::Bls12_377;
    use ark_bn254::{Fq as Fq254, Fr as Fr254};
    use ark_ec::{
        pairing::Pairing,
        short_weierstrass::{Affine, Projective},
    };
    use ark_std::UniformRand;

    use jf_relation::gadgets::{ecc::HasTEForm, EmulationConfig};
    use nf_curves::grumpkin::Grumpkin;

    #[test]
    fn bytes_to_field_elems_test() {
        let rng = &mut jf_utils::test_rng();

        let fq_elem = Fq254::rand(rng);
        ark_std::println!("fq elem: {}", fq_elem);
        let mut writer = Vec::new();
        fq_elem.serialize_compressed(&mut writer).unwrap();

        let field_elems = bytes_to_field_elements::<_, Fr254>(writer.as_slice());

        for elem in field_elems {
            ark_std::println!("field elem: {}", elem);
        }
    }

    #[test]
    fn test_rescue_transcript_challenge_circuit() {
        test_rescue_transcript_challenge_circuit_helper::<Grumpkin, _, _>();
        test_rescue_transcript_challenge_circuit_helper::<Bls12_377, _, _>()
    }
    fn test_rescue_transcript_challenge_circuit_helper<E, F, P>()
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
        F: RescueParameter,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        <E as Pairing>::ScalarField: EmulationConfig<F> + RescueParameter,
    {
        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(16);

        let label = "testing".as_ref();

        let mut transcipt_var = RescueTranscriptVar::new_transcript(&mut circuit);
        let mut transcript = <RescueTranscript<F> as Transcript>::new_transcript(label);

        for _ in 0..10 {
            for i in 0..10 {
                let msg = ark_std::format!("message {}", i);
                let vals = bytes_to_field_elements::<_, F>(msg.as_bytes());
                let message_vars: Vec<Variable> = vals
                    .iter()
                    .map(|x| circuit.create_variable(*x).unwrap())
                    .collect();

                for val in vals.iter() {
                    transcript.push_message(label, val).unwrap()
                }

                transcipt_var.push_variables(&message_vars).unwrap();
            }

            let challenge = transcript.squeeze_scalar_challenge::<P>(label).unwrap();

            let challenge_var = transcipt_var
                .squeeze_scalar_challenge::<P>(&mut circuit)
                .unwrap();

            assert_eq!(
                circuit.witness(challenge_var).unwrap(),
                F::from_le_bytes_mod_order(&challenge.into_bigint().to_bytes_le())
            );

            let challenge = transcript.squeeze_scalar_challenge::<P>(label).unwrap();

            let challenge_var = transcipt_var
                .squeeze_scalar_challenge::<P>(&mut circuit)
                .unwrap();

            assert_eq!(
                circuit.witness(challenge_var).unwrap(),
                F::from_le_bytes_mod_order(&challenge.into_bigint().to_bytes_le())
            );

            let challenges = transcript
                .squeeze_scalar_challenges::<P>(label, 10)
                .unwrap();

            let challenge_vars = transcipt_var
                .squeeze_scalar_challenges::<P>(10, &mut circuit)
                .unwrap();

            for (challenge, challenge_var) in challenges.iter().zip(challenge_vars.iter()) {
                assert_eq!(
                    circuit.witness(*challenge_var).unwrap(),
                    F::from_le_bytes_mod_order(&challenge.into_bigint().to_bytes_le())
                );
            }
        }
    }
}
