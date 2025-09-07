//! This module contains the code for performing scalar arithmetic to do with FFT Proofs.

use ark_bn254::{g1::Config as BnConfig, Fq as Fq254, Fr as Fr254};

use crate::{
    nightfall::{
        accumulation::accumulation_structs::AtomicInstance,
        circuit::plonk_partial_verifier::{
            compute_scalars_for_native_field, ChallengesVar, PlookupEvalsVarNative,
            ProofEvalsVarNative, ProofScalarsVarNative, ProofVarNative,
            VerifyingKeyNativeScalarsVar,
        },
        ipa_structs::VerifyingKey,
        FFTPlonk,
    },
    proof_system::RecursiveOutput,
    transcript::{rescue::RescueTranscriptVar, CircuitTranscript, RescueTranscript},
};
use ark_std::{string::ToString, vec::Vec};
use itertools::izip;
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
pub fn partial_verify_fft_plonk(
    scalar_var: &ProofScalarsVarNative,
    base_vars: &ProofVarNative<BnConfig>,
    vk: &VerifyingKey<Kzg>,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<PCSInfoCircuit, CircuitError> {
    let ProofScalarsVarNative{
        evals: proof_evals,
        lookup_evals,
        pi_hash
     } = scalar_var;

    let mut transcript = RescueTranscriptVar::new_transcript(circuit);

    // Generate the challenges
    // As this is the non-base version, the verification key is fixed and we do not pass in the vk_id to be added to the transcript.
    let challenges = ChallengesVar::compute_challenges::<Kzg, _, _, _>(
        circuit,
        None,
        &pi_hash,
        &base_vars,
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

/// This function takes as input an FFT proof and verifies its transcript and produces the scalars that should be used to calculate its final commitment.
/// This version is used within the base_bn254_circuit. Since verification keys come from the client proofs, they are inputted as variables. 
pub fn partial_verify_fft_plonk_base(
    scalar_var: &ProofScalarsVarNative,
    base_vars: &ProofVarNative<BnConfig>,
    vk_var: &VerifyingKeyNativeScalarsVar,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<PCSInfoCircuit, CircuitError> {
    let ProofScalarsVarNative{
        evals: proof_evals,
        lookup_evals,
        pi_hash
     } = scalar_var;

    let mut transcript = RescueTranscriptVar::new_transcript(circuit);

    // Generate the challenges
    let challenges = ChallengesVar::compute_challenges::<Kzg, _, _, _>(
        circuit,
        Some(vk_var.id),
        pi_hash,
        base_vars,
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
    scalar_vars: &[ProofScalarsVarNative; 2],
    base_vars: &[ProofVarNative<BnConfig>; 2],
    vk: &VerifyingKey<Kzg>,
    old_accs: &[AtomicInstance<Kzg>; 4],
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<Vec<Variable>, CircuitError> {
    // First prepare the pcs_infos for each proof
    let pcs_infos = scalar_vars
        .iter()
        .zip(base_vars.iter())
        .map(|(scalar_var, base_var)| {
            partial_verify_fft_plonk(scalar_var, base_var, vk, circuit)
        })
        .collect::<Result<Vec<PCSInfoCircuit>, CircuitError>>()?
        .try_into()
        .map_err(|_| CircuitError::ParameterError("pcs_infos must have length 4".to_string()))?;

    // Now we transform the 'old_accs' into the relevant circuit variables
    let acc_vars: Vec<(_, _, _, _)> = old_accs
        .iter()
        .map(|acc| {
            let comm_point = Point::<Fq254>::from(acc.comm);
            let emulated_comm = circuit.create_emulated_point_variable(&comm_point)?;
            let point = circuit.create_variable(acc.point)?;
            let eval = circuit.create_variable(acc.value)?;
            let opening_proof_point = Point::<Fq254>::from(acc.opening_proof.proof);
            let opening_proof = circuit.create_emulated_point_variable(&opening_proof_point)?;
            Result::<_, CircuitError>::Ok((emulated_comm, point, eval, opening_proof))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Since we have checked that outputs is non-empty, we can safely unwrap the first element
    let mut transcript = pcs_infos[0].transcript.clone();
    pcs_infos
        .iter()
        .skip(1)
        .try_for_each(|pcs_info| transcript.merge(&pcs_info.transcript))?;

    // Append old_accs to the transcript
    acc_vars
        .iter()
        .try_for_each(|(comm, _, _, opening_proof)| {
            transcript.append_point_variable(comm, circuit)?;
            transcript.append_point_variable(opening_proof, circuit)
        })?;

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

    let mut challenge_powers = (0..5)
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
    scalar_vars: &[ProofScalarsVarNative; 2],
    base_vars: &[ProofVarNative<BnConfig>; 2],
    vk_vars: &[VerifyingKeyNativeScalarsVar; 2],
    old_accs: &[AtomicInstance<Kzg>; 4],
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<Vec<Variable>, CircuitError> {
    // First prepare the pcs_infos for each proof
    let pcs_infos = izip!(scalar_vars, base_vars, vk_vars)
        .map(|(scalar_var, base_var, vk_var)| {
            partial_verify_fft_plonk_base(scalar_var, base_var, vk_var, circuit)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Now we transform the 'old_accs' into the relevant circuit variables
    let acc_vars: Vec<(_, _, _, _)> = old_accs
        .iter()
        .map(|acc| {
            let comm_point = Point::<Fq254>::from(acc.comm);
            let emulated_comm = circuit.create_emulated_point_variable(&comm_point)?;
            let point = circuit.create_variable(acc.point)?;
            let eval = circuit.create_variable(acc.value)?;
            let opening_proof_point = Point::<Fq254>::from(acc.opening_proof.proof);
            let opening_proof = circuit.create_emulated_point_variable(&opening_proof_point)?;
            Result::<_, CircuitError>::Ok((emulated_comm, point, eval, opening_proof))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Since we have checked that outputs is non-empty, we can safely unwrap the first element
    let mut transcript = pcs_infos[0].transcript.clone();
    pcs_infos
        .iter()
        .skip(1)
        .try_for_each(|pcs_info| transcript.merge(&pcs_info.transcript))?;

    // Append old_accs to the transcript
    acc_vars
        .iter()
        .try_for_each(|(comm, _, _, opening_proof)| {
            transcript.append_point_variable(comm, circuit)?;
            transcript.append_point_variable(opening_proof, circuit)
        })?;

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

    let mut challenge_powers = (0..5)
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
        nightfall::{
            ipa_snark::test::gen_circuit_for_test, ipa_structs::VerificationKeyId,
            ipa_verifier::FFTVerifier,
        },
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
        errors::PCSError,
        prelude::{UnivariateKzgPCS, UnivariateKzgProof},
        PolynomialCommitmentScheme,
    };
    use jf_relation::{Arithmetization, PlonkType};

    use super::*;

    #[test]
    fn test_partial_verifier() -> Result<(), PlonkError> {
        let rng = &mut jf_utils::test_rng();
        for (m, vk_id) in (2..8).zip(
            [
                None,
                Some(VerificationKeyId::Client),
                Some(VerificationKeyId::Deposit),
            ]
            .iter(),
        ) {
            let circuit = gen_circuit_for_test::<Fr254>(m, 3, PlonkType::UltraPlonk, true)?;
            let pi = circuit.public_input()?[0];

            let srs_size = circuit.srs_size()?;
            let srs = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, srs_size)?;

            let (pk, vk) = FFTPlonk::<Kzg>::preprocess(&srs, *vk_id, &circuit)?;

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
        for (m, vk_id) in (2..8).zip(
            [
                None,
                Some(VerificationKeyId::Client),
                Some(VerificationKeyId::Deposit),
            ]
            .iter(),
        ) {
            let circuit_one = gen_circuit_for_test::<Fr254>(m, 3, PlonkType::UltraPlonk, true)?;
            let circuit_two = gen_circuit_for_test::<Fr254>(m, 4, PlonkType::UltraPlonk, true)?;
            let pi_one = circuit_one.public_input()?[0];
            let pi_two = circuit_two.public_input()?[0];

            let srs_size = circuit_one.srs_size()?;

            let srs = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, srs_size)?;

            let (pk, vk) = FFTPlonk::<Kzg>::preprocess(&srs, *vk_id, &circuit_one)?;

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
            let accs: [AtomicInstance<Kzg>; 4] = (0..4)
                .into_iter()
                .map(|_| {
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

                    Ok(AtomicInstance::new(comm, eval, Fr254::zero(), proof))
                })
                .collect::<Result<Vec<AtomicInstance<Kzg>>, PCSError>>()?
                .try_into()
                .map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })?;

            let mut verifier_circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(8);

            // Create variables from the proofs
            let output_var_pairs = outputs
                .iter()
                .map(|output| {
                    let proof_evals = ProofScalarsVarNative::from_struct(&output, &mut verifier_circuit)?;
                    let proof = ProofVarNative::from_struct(&mut verifier_circuit, &output.proof)?;
                    Ok((proof_evals, proof))
                })
                .collect::<Result<Vec<(ProofScalarsVarNative, ProofVarNative<BnConfig>)>, CircuitError>>()?;
            let output_scalar_vars: [ProofScalarsVarNative; 2] = output_var_pairs
                .clone()
                .into_iter()
                .map(|(output_scalar_var, _)| output_scalar_var)
                .collect::<Vec<ProofScalarsVarNative>>()
                .try_into()
                .map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })?;
            let output_base_vars: [ProofVarNative<BnConfig>; 2] = output_var_pairs
                .into_iter()
                .map(|(_, output_base_var)| output_base_var)
                .collect::<Vec<ProofVarNative<BnConfig>>>()
                .try_into()
                .map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })?;

            let recursion_scalars = calculate_recursion_scalars(
                &output_scalar_vars,
                &output_base_vars,
                &vk,
                &accs,
                &mut verifier_circuit,
            )?;
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
