//! This module contains the code for performing scalar arithmetic to do with FFT Proofs.

use ark_bn254::{g1::Config as BnConfig, Fr as Fr254};

use ark_std::{string::ToString, vec::Vec};

use jf_relation::{errors::CircuitError, Circuit, PlonkCircuit, Variable};

use crate::{
    nightfall::{
        accumulation::accumulation_structs::AtomicInstance,
        circuit::plonk_partial_verifier::{
            compute_scalars_for_native_field, ChallengesVar, PlookupEvalsVarNative,
            ProofEvalsVarNative, ProofVarNative,
        },
        ipa_structs::VerifyingKey,
        FFTPlonk,
    },
    proof_system::RecursiveOutput,
    transcript::{rescue::RescueTranscriptVar, CircuitTranscript, RescueTranscript},
};
use jf_relation::{errors::CircuitError, gadgets::ecc::Point, Circuit, PlonkCircuit, Variable};

use super::Kzg;

/// Struct representing the information needed before proceeding to accumulation of the proofs.
#[derive(Clone, Debug)]
pub struct PCSInfoCircuit {
    /// Te scalars used in the MSM
    pub(crate) scalars: Vec<Variable>,
    /// The transcript so it can be merged with the other transcript.
    pub(crate) transcript: RescueTranscriptVar<Fr254>,
    /// The opening point for the proof
    pub(crate) opening_point: Variable,
}

impl PCSInfoCircuit {
    /// Creates a new instance of the struct.
    pub fn new(
        scalars: Vec<Variable>,
        transcript: RescueTranscriptVar<Fr254>,
        opening_point: Variable,
    ) -> Self {
        Self {
            scalars,
            transcript,
            opening_point,
        }
    }

    /// Getter for the scalars
    pub fn scalars(&self) -> &[Variable] {
        &self.scalars
    }

    /// Getter for the transcript
    pub fn transcript(&self) -> &RescueTranscriptVar<Fr254> {
        &self.transcript
    }

    /// Getter for the opening point
    pub fn opening_point(&self) -> Variable {
        self.opening_point
    }
}

/// This function takes as input an FFT proof and verifies its transcript and produces the scalars that should be used to calculate its final commitment.
pub fn partial_verify_fft_plonk<const IS_BASE: bool>(
    output: &RecursiveOutput<Kzg, FFTPlonk<Kzg>, RescueTranscript<Fr254>>,
    vk: &VerifyingKey<Kzg>,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<PCSInfoCircuit, CircuitError> {
    // First we convert the `output` into the relevant circuit variables.
    let proof = ProofVarNative::from_struct(circuit, &output.proof)?;

    let proof_evals = ProofEvalsVarNative::from_struct(circuit, &output.proof.poly_evals)?;

    let lookup_evals = if let Some(lookup_proof) = &output.proof.plookup_proof {
        Some(PlookupEvalsVarNative::from_struct(
            circuit,
            &lookup_proof.poly_evals,
        )?)
    } else {
        None
    };
    let pi_hash = circuit.create_variable(output.pi_hash)?;

    let mut transcript = RescueTranscriptVar::new_transcript(circuit);

    let vk_id = if let Some(id) = vk.id {
        Some(circuit.create_variable(Fr254::from(id as u8))?)
    } else {
        None
    };

    // Generate the challenges
    let challenges = ChallengesVar::compute_challenges::<Kzg, _, _, _>(
        circuit,
        vk_id,
        &pi_hash,
        &proof,
        &mut transcript,
    )?;

    // Output the scalars
    let vk_k =
        vk.k.iter()
            .map(|k| circuit.create_variable(*k))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
    let scalars = compute_scalars_for_native_field::<Fr254, IS_BASE>(
        circuit,
        pi_hash,
        &challenges,
        &proof_evals,
        lookup_evals,
        &vk_k,
        vk.domain_size,
    )?;

    Ok(PCSInfoCircuit::new(scalars, transcript, challenges.u))
}

/// This function takes in two [`RecursiveOutput`]s and verifies their transcripts and produces the scalars that should be used to calculate their final commitment.
/// It then combines all the scalars in such a way that their hash is equal to the public input hash of the proof from the other curve.
pub fn calculate_recursion_scalars(
    outputs: &[RecursiveOutput<Kzg, FFTPlonk<Kzg>, RescueTranscript<Fr254>>],
    old_accs: &[AtomicInstance<Kzg>],
    vk: &VerifyingKey<Kzg>,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<Vec<Variable>, CircuitError> {
    if outputs.is_empty() {
        return Err(CircuitError::ParameterError(
            "No outputs provided".to_string(),
        ));
    }
    // First prepare the pcs_infos for each proof
    let pcs_infos = outputs
        .iter()
        .map(|output| partial_verify_fft_plonk::<false>(output, vk, circuit))
        .collect::<Result<Vec<_>, _>>()?;

    // Since we have checked that outputs is non-empty, we can safely unwrap the first element
    let mut transcript = pcs_infos[0].transcript.clone();
    pcs_infos
        .iter()
        .skip(1)
        .try_for_each(|pcs_info| transcript.merge(&pcs_info.transcript))?;

    // Generate the challenge
    let batching_challenge = transcript.squeeze_scalar_challenge::<BnConfig>(circuit)?;

    // Now we rescale the scalars and combine as necessary
    let combined_scalars = combine_fft_scalars(
        &pcs_infos
            .iter()
            .map(|pcs_info| pcs_info.scalars())
            .collect::<Vec<_>>(),
        batching_challenge,
        circuit,
    )?;

    let mut challenge_powers = (0..outputs.len() + old_accs.len() - 1)
        .scan(circuit.one(), |state, _| {
            if let Ok(challenge_power) = circuit.mul(*state, batching_challenge) {
                *state = challenge_power;
                Some(challenge_power)
            } else {
                None
            }
        })
        .collect::<Vec<Variable>>();

    challenge_powers.insert(0, circuit.one());

    let points = pcs_infos
        .iter()
        .map(|pcs_info| pcs_info.opening_point())
        .collect::<Vec<_>>();
    let final_instance_scalars = points
        .iter()
        .zip(challenge_powers.iter())
        .map(|(point, challenge_power)| circuit.mul(*point, *challenge_power))
        .collect::<Result<Vec<_>, _>>()?;

    Ok([
        combined_scalars.as_slice(),
        final_instance_scalars.as_slice(),
        &challenge_powers[2..],
        &challenge_powers[..2],
    ]
    .concat())
}

/// This function takes in two [`RecursiveOutput`]s and verifies their transcripts and produces the scalars that should be used to calculate their final commitment.
/// It then combines all the scalars in suc a way that their hash is equal to the public input hash of the proof from the other curve.
pub fn calculate_recursion_scalars_base(
    outputs: &[RecursiveOutput<Kzg, FFTPlonk<Kzg>, RescueTranscript<Fr254>>],
    old_accs: &[AtomicInstance<Kzg>],
    vks: &[VerifyingKey<Kzg>],
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<Vec<Variable>, CircuitError> {
    if outputs.is_empty() {
        return Err(CircuitError::ParameterError(
            "No outputs provided".to_string(),
        ));
    }
    // First prepare the pcs_infos for each proof
    let pcs_infos = outputs
        .iter()
        .zip(vks.iter())
        .map(|(output, vk)| partial_verify_fft_plonk::<true>(output, vk, circuit))
        .collect::<Result<Vec<_>, _>>()?;

    // Since we have checked that outputs is non-empty, we can safely unwrap the first element
    let mut transcript = pcs_infos[0].transcript.clone();
    pcs_infos
        .iter()
        .skip(1)
        .try_for_each(|pcs_info| transcript.merge(&pcs_info.transcript))?;

    // Generate the challenge
    let batching_challenge = transcript.squeeze_scalar_challenge::<BnConfig>(circuit)?;

    // Now we rescale the scalars and combine as necessary
    let combined_scalars = combine_fft_scalars_base(
        &pcs_infos
            .iter()
            .map(|pcs_info| pcs_info.scalars())
            .collect::<Vec<_>>(),
        batching_challenge,
        circuit,
    )?;

    let mut challenge_powers = (0..outputs.len() + old_accs.len() - 1)
        .scan(circuit.one(), |state, _| {
            if let Ok(challenge_power) = circuit.mul(*state, batching_challenge) {
                *state = challenge_power;
                Some(challenge_power)
            } else {
                None
            }
        })
        .collect::<Vec<Variable>>();

    challenge_powers.insert(0, circuit.one());

    let points = pcs_infos
        .iter()
        .map(|pcs_info| pcs_info.opening_point())
        .collect::<Vec<_>>();
    let final_instance_scalars = points
        .iter()
        .zip(challenge_powers.iter())
        .map(|(point, challenge_power)| circuit.mul(*point, *challenge_power))
        .collect::<Result<Vec<_>, _>>()?;

    Ok([
        combined_scalars.as_slice(),
        final_instance_scalars.as_slice(),
        &challenge_powers[2..],
        &challenge_powers[..2],
    ]
    .concat())
}

fn combine_fft_scalars(
    scalar_lists: &[&[Variable]],
    batching_challenge: Variable,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<Vec<Variable>, CircuitError> {
    if scalar_lists.is_empty() {
        return Err(CircuitError::ParameterError(
            "No scalar lists provided".to_string(),
        ));
    }

    let mut combined_scalars = scalar_lists[0].to_vec();

    for prescalars in scalar_lists.iter().skip(1) {
        let scalars = prescalars
            .iter()
            .map(|s| circuit.mul(*s, batching_challenge))
            .collect::<Result<Vec<Variable>, _>>()?;
        combined_scalars[4..10]
            .iter_mut()
            .zip(scalars[4..10].iter())
            .try_for_each(|(a, b)| {
                *a = circuit.add(*a, *b)?;
                Result::<_, CircuitError>::Ok(())
            })?;

        combined_scalars[16] = circuit.add(combined_scalars[16], scalars[16])?;
        combined_scalars[13..15]
            .iter_mut()
            .zip(scalars[13..15].iter())
            .try_for_each(|(a, b)| {
                *a = circuit.add(*a, *b)?;
                Result::<_, CircuitError>::Ok(())
            })?;
        combined_scalars[19] = circuit.add(combined_scalars[19], scalars[19])?;
        combined_scalars[21..40]
            .iter_mut()
            .zip(scalars[21..40].iter())
            .try_for_each(|(a, b)| {
                *a = circuit.add(*a, *b)?;
                Result::<_, CircuitError>::Ok(())
            })?;

        let appended_scalars = [
            &scalars[0..4],
            &scalars[10..13],
            &[scalars[15], scalars[17], scalars[18], scalars[20]],
            &scalars[40..],
        ]
        .concat();

        combined_scalars.extend_from_slice(&appended_scalars);
    }
    Ok(combined_scalars)
}

fn combine_fft_scalars_base(
    scalar_lists: &[&[Variable]],
    batching_challenge: Variable,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<Vec<Variable>, CircuitError> {
    if scalar_lists.is_empty() {
        return Err(CircuitError::ParameterError(
            "No scalar lists provided".to_string(),
        ));
    }

    let mut combined_scalars = scalar_lists[0].to_vec();

    for prescalars in scalar_lists.iter().skip(1) {
        let appended_scalars = prescalars
            .iter()
            .map(|s| circuit.mul(batching_challenge, *s))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;

        combined_scalars.extend_from_slice(&appended_scalars);
    }
    Ok(combined_scalars)
}

#[cfg(test)]
mod tests {

    use crate::{
        errors::PlonkError,
        nightfall::{ipa_snark::test::gen_circuit_for_test, ipa_verifier::FFTVerifier},
        proof_system::UniversalSNARK,
    };
    use ark_bn254::{g1::Config as BnConfig, Bn254};
    use ark_ec::{
        short_weierstrass::{Affine, Projective},
        CurveGroup, VariableBaseMSM,
    };
    use ark_ff::PrimeField;
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
    use ark_std::Zero;
    use jf_primitives::pcs::{
        prelude::{UnivariateKzgPCS, UnivariateKzgProof},
        PolynomialCommitmentScheme,
    };
    use jf_relation::{Arithmetization, PlonkType};

    use super::*;

    #[test]
    fn test_partial_verifier() -> Result<(), PlonkError> {
        let rng = &mut jf_utils::test_rng();
        for m in 2..8 {
            let circuit = gen_circuit_for_test::<Fr254>(m, 3, PlonkType::UltraPlonk, true)?;
            let pi = circuit.public_input()?[0];

            let srs_size = circuit.srs_size()?;
            let srs = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, srs_size)?;

            let (pk, vk) = FFTPlonk::<Kzg>::preprocess(&srs, &circuit)?;

            let output = FFTPlonk::<Kzg>::recursive_prove::<_, _, RescueTranscript<Fr254>>(
                rng, &circuit, &pk, None,
            )?;

            let mut verifier_circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(8);
            let pcs_info_circuit =
                partial_verify_fft_plonk::<false>(&output, &vk, &mut verifier_circuit)?;

            let fft_verifier = FFTVerifier::<Kzg>::new(vk.domain_size)?;

            let pcs_info = fft_verifier.prepare_pcs_info::<RescueTranscript<Fr254>>(
                &vk,
                &[pi],
                &output.proof,
                &None,
            )?;

            let g_comm = pcs_info
                .comm_scalars_and_bases
                .multi_scalar_mul()
                .into_affine();
            let mut comms = pcs_info.comm_scalars_and_bases.bases()[..47].to_vec();

            let _ = comms.remove(22);
            let scalars = pcs_info_circuit
                .scalars()
                .iter()
                .map(|s| verifier_circuit.witness(*s))
                .collect::<Result<Vec<_>, _>>()?;

            let scalars_bigints = scalars.iter().map(|s| s.into_bigint()).collect::<Vec<_>>();

            let computed_g_comm =
                Projective::<BnConfig>::msm_bigint(&comms, &scalars_bigints).into_affine();

            assert_eq!(g_comm, computed_g_comm);
        }
        Ok(())
    }

    #[test]
    fn test_scalar_combiner() -> Result<(), PlonkError> {
        let rng = &mut jf_utils::test_rng();
        for m in 2..8 {
            let circuit_one = gen_circuit_for_test::<Fr254>(m, 3, PlonkType::UltraPlonk, true)?;
            let circuit_two = gen_circuit_for_test::<Fr254>(m, 4, PlonkType::UltraPlonk, true)?;
            let pi_one = circuit_one.public_input()?[0];
            let pi_two = circuit_two.public_input()?[0];

            let srs_size = circuit_one.srs_size()?;

            let srs = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, srs_size)?;

            let (pk, vk) = FFTPlonk::<Kzg>::preprocess(&srs, &circuit_one)?;

            let circuits = [circuit_one, circuit_two];
            let pis = [pi_one, pi_two];

            let outputs = circuits
                .iter()
                .map(|circuit| {
                    FFTPlonk::<Kzg>::recursive_prove::<_, _, RescueTranscript<Fr254>>(
                        rng, circuit, &pk, None,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut accs = Vec::new();

            for _ in 0..4 {
                let mut poly = DensePolynomial::<Fr254>::rand(srs_size, rng);

                poly[0] = Fr254::zero();
                let comm: Affine<BnConfig> = Kzg::commit(&pk.commit_key, &poly)?;
                let (proof, eval): (UnivariateKzgProof<Bn254>, Fr254) =
                    Kzg::open(&pk.commit_key, &poly, &Fr254::zero())?;
                assert_eq!(eval, Fr254::zero());
                assert!(Kzg::verify(
                    &vk.open_key,
                    &comm,
                    &Fr254::zero(),
                    &Fr254::zero(),
                    &proof
                )?);

                accs.push(AtomicInstance::new(comm, eval, Fr254::zero(), proof));
            }

            let mut verifier_circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(8);
            let recursion_scalars =
                calculate_recursion_scalars(&outputs, &accs, &vk, &mut verifier_circuit)?;
            let (instance_scalars, proof_scalars) =
                recursion_scalars.split_at(recursion_scalars.len() - 6);

            let non_acc_bigints = instance_scalars
                .iter()
                .map(|s| verifier_circuit.witness(*s).unwrap().into_bigint())
                .collect::<Vec<_>>();

            let acc_bigints = proof_scalars[..4]
                .iter()
                .map(|s| verifier_circuit.witness(*s).unwrap().into_bigint())
                .collect::<Vec<_>>();

            let proof_bigints = proof_scalars[4..]
                .iter()
                .chain(proof_scalars[..4].iter())
                .map(|s| verifier_circuit.witness(*s).unwrap().into_bigint())
                .collect::<Vec<_>>();

            let fft_verifier = FFTVerifier::<Kzg>::new(vk.domain_size)?;

            let pcs_infos = outputs
                .iter()
                .zip(pis.iter())
                .map(|(output, &pi)| {
                    fft_verifier.prepare_pcs_info::<RescueTranscript<Fr254>>(
                        &vk,
                        &[pi],
                        &output.proof,
                        &None,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            pcs_infos.iter().for_each(|pcs_info| {
                assert!(FFTVerifier::<Kzg>::verify_opening_proofs(&vk.open_key, pcs_info).unwrap());
            });

            let opening_proofs = outputs
                .iter()
                .map(|output| output.proof.opening_proof.proof)
                .chain(accs.iter().map(|acc| acc.opening_proof.proof))
                .collect::<Vec<_>>();

            let comms_lists = pcs_infos
                .iter()
                .map(|pcs_info| {
                    let mut comms = pcs_info.comm_scalars_and_bases.bases()[..47].to_vec();

                    let _ = comms.remove(22);
                    comms
                })
                .collect::<Vec<Vec<_>>>();

            let mut comms = comms_lists[0].clone();

            let appended_comms = [
                &comms_lists[1][0..4],
                &comms_lists[1][10..13],
                &[
                    comms_lists[1][15],
                    comms_lists[1][17],
                    comms_lists[1][18],
                    comms_lists[1][20],
                ],
                &comms_lists[1][40..],
            ]
            .concat();
            comms.extend_from_slice(&appended_comms);
            comms.extend_from_slice(&opening_proofs[..2]);

            let proof_part =
                Projective::<BnConfig>::msm_bigint(&comms, &non_acc_bigints).into_affine();
            let acc_part = Projective::<BnConfig>::msm_bigint(
                &accs.iter().map(|a| a.comm).collect::<Vec<_>>(),
                &acc_bigints,
            )
            .into_affine();

            let new_proof =
                Projective::<BnConfig>::msm_bigint(&opening_proofs, &proof_bigints).into_affine();

            let instance = proof_part + acc_part;

            let pcs_proof = UnivariateKzgProof::<Bn254> { proof: new_proof };

            assert!(Kzg::verify(
                &vk.open_key,
                &instance.into_affine(),
                &Fr254::zero(),
                &Fr254::zero(),
                &pcs_proof
            )
            .unwrap());
        }
        Ok(())
    }
}
