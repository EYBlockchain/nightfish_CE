//! This module contains the code for performing scalar arithmetic to do with FFT Proofs.

use ark_bn254::{g1::Config as BnConfig, Fr as Fr254};

use ark_std::{string::ToString, vec::Vec};

use jf_relation::{errors::CircuitError, Circuit, PlonkCircuit, Variable};

use crate::{
    nightfall::{
        circuit::plonk_partial_verifier::{
            compute_scalars_for_native_field, compute_scalars_for_native_field_base, ChallengesVar,
            ProofScalarsVarNative, ProofVarNative, VerifyingKeyNativeScalarsVar,
        },
        ipa_structs::VerifyingKey,
    },
    transcript::{rescue::RescueTranscriptVar, CircuitTranscript},
};
use itertools::izip;

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
    blind: bool,
) -> Result<PCSInfoCircuit, CircuitError> {
    let ProofScalarsVarNative {
        evals: proof_evals,
        lookup_evals,
        pi_hash,
    } = scalar_var;

    let mut transcript = RescueTranscriptVar::new_transcript(circuit);

    // Generate the challenges
    // As this is the non-base version, the verification key is fixed and we do not pass in the vk_id to be added to the transcript.
    let challenges = ChallengesVar::compute_challenges::<Kzg, _, _, _>(
        circuit,
        None,
        pi_hash,
        base_vars,
        &mut transcript,
    )?;

    let scalars = compute_scalars_for_native_field::<Fr254>(
        circuit,
        pi_hash,
        &challenges,
        proof_evals,
        lookup_evals,
        &vk.k,
        vk.domain_size,
        blind,
    )?;

    Ok(PCSInfoCircuit::new(scalars, transcript, challenges.u))
}

/// This function takes as input an FFT proof and verifies its transcript and produces the scalars that should be used to calculate its final commitment.
/// This version is used within the base_bn254_circuit. Since verification keys come from the client proofs, they are inputted as variables.
pub(crate) fn partial_verify_fft_plonk_base(
    scalar_var: &ProofScalarsVarNative,
    base_vars: &ProofVarNative<BnConfig>,
    vk_var: &VerifyingKeyNativeScalarsVar,
    max_domain_size: usize,
    circuit: &mut PlonkCircuit<Fr254>,
    blind: bool,
) -> Result<PCSInfoCircuit, CircuitError> {
    let ProofScalarsVarNative {
        evals: proof_evals,
        lookup_evals,
        pi_hash,
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

    let scalars = compute_scalars_for_native_field_base::<Fr254>(
        circuit,
        pi_hash,
        &challenges,
        proof_evals,
        lookup_evals,
        vk_var,
        max_domain_size,
        blind,
    )?;

    Ok(PCSInfoCircuit::new(scalars, transcript, challenges.u))
}

/// This function takes in two [`RecursiveOutput`]s and verifies their transcripts and produces the scalars that should be used to calculate their final commitment.
/// It then combines all the scalars in such a way that their hash is equal to the public input hash of the proof from the other curve.
pub(crate) fn calculate_recursion_scalars(
    scalar_vars: &[ProofScalarsVarNative; 2],
    base_vars: &[ProofVarNative<BnConfig>; 2],
    vk: &VerifyingKey<Kzg>,
    circuit: &mut PlonkCircuit<Fr254>,
    blind: bool,
) -> Result<Vec<Variable>, CircuitError> {
    // First prepare the pcs_infos for each proof
    let pcs_infos: [PCSInfoCircuit; 2] = scalar_vars
        .iter()
        .zip(base_vars.iter())
        .map(|(scalar_var, base_var)| {
            partial_verify_fft_plonk(scalar_var, base_var, vk, circuit, blind)
        })
        .collect::<Result<Vec<PCSInfoCircuit>, CircuitError>>()?
        .try_into()
        .map_err(|_| CircuitError::ParameterError("pcs_infos must have length 2".to_string()))?;

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

    // We need 6 challenge powers, to batch the 4 old accumulators and the 2 new proofs.
    let mut challenge_powers = (0..4)
        .scan(batching_challenge, |state, _| {
            if let Ok(challenge_power) = circuit.mul(*state, batching_challenge) {
                *state = challenge_power;
                Some(challenge_power)
            } else {
                None
            }
        })
        .collect::<Vec<Variable>>();

    // We only now add the first two powers, r^0 and r^1, to the start of the list to avoid extra calls to `mul`
    challenge_powers.insert(0, batching_challenge);
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
pub(crate) fn calculate_recursion_scalars_base(
    scalar_vars: &[ProofScalarsVarNative; 2],
    base_vars: &[ProofVarNative<BnConfig>; 2],
    vk_vars: &[VerifyingKeyNativeScalarsVar; 2],
    max_domain_size: usize,
    circuit: &mut PlonkCircuit<Fr254>,
    blind: bool,
) -> Result<Vec<Variable>, CircuitError> {
    // First prepare the pcs_infos for each proof
    let pcs_infos = izip!(scalar_vars, base_vars, vk_vars)
        .map(|(scalar_var, base_var, vk_var)| {
            partial_verify_fft_plonk_base(
                scalar_var,
                base_var,
                vk_var,
                max_domain_size,
                circuit,
                blind,
            )
        })
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

    // We need 6 challenge powers, to batch the 4 old accumulators and the 2 new proofs.
    let mut challenge_powers = (0..4)
        .scan(batching_challenge, |state, _| {
            if let Ok(challenge_power) = circuit.mul(*state, batching_challenge) {
                *state = challenge_power;
                Some(challenge_power)
            } else {
                None
            }
        })
        .collect::<Vec<Variable>>();

    // We only now add the first two powers, r^0 and r^1, to the start of the list to avoid extra calls to `mul`
    challenge_powers.insert(0, batching_challenge);
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
            circuit::plonk_partial_verifier::{
                Bn254OutputScalarsAndBasesVar, PcsInfoBasesVar, VerifyingKeyScalarsAndBasesVar,
            },
            ipa_snark::test::gen_circuit_for_test,
            ipa_structs::VerificationKeyId,
            ipa_verifier::FFTVerifier,
        },
        proof_system::{UniversalRecursiveSNARK, UniversalSNARK},
        recursion::{
            merge_functions::{combine_fft_proof_scalars, combine_fft_proof_scalars_round_one},
            AtomicInstance, FFTPlonk,
        },
        transcript::{RescueTranscript, Transcript},
    };
    use ark_bn254::{g1::Config as BnConfig, Bn254, Fq as Fq254};
    use ark_ec::{
        short_weierstrass::{Affine, Projective},
        AffineRepr, CurveGroup, VariableBaseMSM,
    };
    use ark_ff::PrimeField;
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
    use ark_std::{One, Zero};
    use core::iter;
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
        for m in 2..8 {
            let circuit = gen_circuit_for_test::<Fr254>(m, 3, PlonkType::UltraPlonk, true)?;
            let pi = circuit.public_input()?[0];

            let srs_size = circuit.srs_size(true)?;
            let srs = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, srs_size)?;

            // Here we are assuming we are in the non-base case and our verification key is fixed.
            // Our `vk_id` is, therefore, `None`.
            let (pk, vk) = FFTPlonk::<Kzg>::preprocess(&srs, None, &circuit, true)?;

            let output = FFTPlonk::<Kzg>::recursive_prove::<_, _, RescueTranscript<Fr254>>(
                rng, &circuit, &pk, None, true,
            )?;

            let mut verifier_circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(8);
            let base_var = ProofVarNative::from_struct(&output.proof, &mut verifier_circuit)?;
            let pi_hash = verifier_circuit.create_variable(output.pi_hash)?;
            let scalar_var = ProofScalarsVarNative::from_struct(&base_var, pi_hash)?;

            let pcs_info_circuit =
                partial_verify_fft_plonk(&scalar_var, &base_var, &vk, &mut verifier_circuit, true)?;

            let fft_verifier = FFTVerifier::<Kzg>::new(vk.domain_size)?;

            let pcs_info = fft_verifier.prepare_pcs_info::<RescueTranscript<Fr254>>(
                &vk,
                &[pi],
                &output.proof,
                &None,
                true,
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
    fn test_partial_verifier_base() -> Result<(), PlonkError> {
        let rng = &mut jf_utils::test_rng();
        for (m, vk_id) in (2..8).zip(
            [
                Some(VerificationKeyId::Client),
                Some(VerificationKeyId::Deposit),
            ]
            .iter(),
        ) {
            let circuit = gen_circuit_for_test::<Fr254>(m, 3, PlonkType::UltraPlonk, true)?;
            let pi = circuit.public_input()?[0];

            let srs_size = circuit.srs_size(true)?;
            let srs = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, srs_size)?;

            // Here we are assuming we are in the base case. Our `vk_id` is, therefore, non-`None`.
            let (pk, vk) = FFTPlonk::<Kzg>::preprocess(&srs, *vk_id, &circuit, true)?;

            let output = FFTPlonk::<Kzg>::recursive_prove::<_, _, RescueTranscript<Fr254>>(
                rng, &circuit, &pk, None, true,
            )?;

            let mut verifier_circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(8);
            let vk_var = VerifyingKeyNativeScalarsVar::new(&mut verifier_circuit, &vk)?;

            let pi_hash = verifier_circuit.create_variable(output.pi_hash)?;
            let base_var = ProofVarNative::from_struct(&output.proof, &mut verifier_circuit)?;
            let scalar_var = ProofScalarsVarNative::from_struct(&base_var, pi_hash)?;

            // Here we assume a max domain size of 2^10
            let pcs_info_circuit = partial_verify_fft_plonk_base(
                &scalar_var,
                &base_var,
                &vk_var,
                1 << 10,
                &mut verifier_circuit,
                true,
            )?;

            let fft_verifier = FFTVerifier::<Kzg>::new(vk.domain_size)?;

            let pcs_info = fft_verifier.prepare_pcs_info::<RescueTranscript<Fr254>>(
                &vk,
                &[pi],
                &output.proof,
                &None,
                true,
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

            let srs_size = circuit_one.srs_size(true)?;

            let srs = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, srs_size)?;

            let (pk, vk) = FFTPlonk::<Kzg>::preprocess(&srs, None, &circuit_one, true)?;

            let circuits = [circuit_one, circuit_two];
            let pis = [pi_one, pi_two];

            let outputs = circuits
                .iter()
                .map(|circuit| {
                    FFTPlonk::<Kzg>::recursive_prove::<_, _, RescueTranscript<Fr254>>(
                        rng, circuit, &pk, None, true,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            // We construct some random accumulators to simulate the old accumulators from the previous layer.
            // We ensure they have a zero constant term so that they can be opened to zero at zero.
            let old_accs: [AtomicInstance<Kzg>; 4] = (0..4)
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

            // We construct an intermediate circuit to process the bases computations of the verification.
            let mut bases_verifier_circuit = PlonkCircuit::<Fq254>::new_ultra_plonk(8);
            // This is not the first round, so we constrain the `VerifyingKeyBasesVar`s to be constant.
            let vk_var =
                VerifyingKeyScalarsAndBasesVar::new_constant(&mut bases_verifier_circuit, &vk)?;
            let vk_bases_var = [vk_var.clone(), vk_var];

            // We store the scalars and bases from the proofs into the relevant struct.
            // In particular, the bases are stored as `Variable`s in the circuit, as we
            // need to reuse them later in the circuit and in the next circuit.
            // We also prepare the relevant info from each of the proofs.
            let output_pcs_info_var_pair: [(Bn254OutputScalarsAndBasesVar, PcsInfoBasesVar<Kzg>); 2] = outputs.iter()
                .zip(vk_bases_var.iter())
                .map(|(output, vk)| {
                    let verifier = FFTVerifier::new(vk.domain_size)?;
                    verifier.prepare_pcs_info_with_bases_var::<RescueTranscript<Fr254>>(
                        vk,
                        &[output.pi_hash],
                        output,
                        &None,
                        &mut bases_verifier_circuit,
                        true,
                    )
                })
                .collect::<Result<Vec<(Bn254OutputScalarsAndBasesVar, PcsInfoBasesVar<Kzg>)>, PlonkError>>()?
                .try_into()
                .map_err(|_| PlonkError::InvalidParameters("outputs must have length 2".to_string()))?;

            let output_vars = output_pcs_info_var_pair
                .iter()
                .map(|(output, _)| output.clone())
                .collect::<Vec<Bn254OutputScalarsAndBasesVar>>();
            let pcs_info_vars = output_pcs_info_var_pair
                .iter()
                .map(|(_, pcs_info)| pcs_info.clone())
                .collect::<Vec<PcsInfoBasesVar<Kzg>>>();

            // Unwrap is safe here because we checked tht the slice was non-empty at the beginning.
            let transcript = &mut output_vars[0].transcript.clone();

            transcript.merge(&output_vars[1].transcript)?;

            let r = transcript.squeeze_scalar_challenge::<BnConfig>(b"r")?;

            // Calculate the various powers of r needed, they start at 1 and end at r^(pcs_infos.len()).
            let r_powers = iter::successors(Some(Fr254::one()), |x| Some(*x * r))
                .take(6)
                .collect::<Vec<Fr254>>();

            let (scalars, instance_base_vars) =
                combine_fft_proof_scalars(&pcs_info_vars, &r_powers);

            // We construct out final circuit to process the scalars computations of the verification.
            let mut scalars_verifier_circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(8);

            // Create variables from the proofs
            let output_var_pairs = outputs
                .iter()
                .map(|output| {
                    let proof = ProofVarNative::from_struct(&output.proof, &mut scalars_verifier_circuit)?;
                    let pi_hash = scalars_verifier_circuit.create_variable(output.pi_hash)?;
                    let proof_evals = ProofScalarsVarNative::from_struct(&proof, pi_hash)?;
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
                &mut scalars_verifier_circuit,
                true,
            )?;
            let (instance_scalar_vars, proof_scalar_vars) =
                recursion_scalars.split_at(recursion_scalars.len() - 6);

            // We check the witness values of the scalars are as expected.
            assert_eq!(instance_scalar_vars.len(), scalars.len());
            assert_eq!(proof_scalar_vars.len(), r_powers.len());

            for (var, &expected) in instance_scalar_vars.iter().zip(scalars.iter()) {
                let witness = scalars_verifier_circuit.witness(*var)?;
                assert_eq!(witness, expected);
            }
            for (var, &expected) in proof_scalar_vars
                .iter()
                .skip(4)
                .zip(r_powers.iter().take(2))
            {
                let witness = scalars_verifier_circuit.witness(*var)?;
                assert_eq!(witness, expected);
            }
            for (var, &expected) in proof_scalar_vars
                .iter()
                .take(4)
                .zip(r_powers.iter().skip(2))
            {
                let witness = scalars_verifier_circuit.witness(*var)?;
                assert_eq!(witness, expected);
            }

            let non_acc_bigints = instance_scalar_vars
                .iter()
                .map(|s| scalars_verifier_circuit.witness(*s).unwrap().into_bigint())
                .collect::<Vec<_>>();

            let acc_bigints = proof_scalar_vars[..4]
                .iter()
                .map(|s| scalars_verifier_circuit.witness(*s).unwrap().into_bigint())
                .collect::<Vec<_>>();

            let proof_bigints = proof_scalar_vars[4..]
                .iter()
                .chain(proof_scalar_vars[..4].iter())
                .map(|s| scalars_verifier_circuit.witness(*s).unwrap().into_bigint())
                .collect::<Vec<_>>();

            let opening_proofs = outputs
                .iter()
                .map(|output| output.proof.opening_proof.proof)
                .chain(old_accs.iter().map(|acc| acc.opening_proof.proof))
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
                        true,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            pcs_infos.iter().for_each(|pcs_info| {
                assert!(FFTVerifier::<Kzg>::verify_opening_proofs(&vk.open_key, pcs_info).unwrap());
            });

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

            // We now check the commitments match up with those computed in-circuit
            assert_eq!(comms.len(), instance_base_vars.len());
            for (comm, var) in comms.iter().zip(instance_base_vars.iter()) {
                let x_val = bases_verifier_circuit.witness(var.get_x())?;
                let y_val = bases_verifier_circuit.witness(var.get_y())?;
                assert_eq!(x_val, *comm.x().unwrap());
                assert_eq!(y_val, *comm.y().unwrap());
            }

            let proof_part =
                Projective::<BnConfig>::msm_bigint(&comms, &non_acc_bigints).into_affine();
            let acc_part = Projective::<BnConfig>::msm_bigint(
                &old_accs.iter().map(|a| a.comm).collect::<Vec<_>>(),
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

    #[test]
    fn test_scalar_combiner_base() -> Result<(), PlonkError> {
        let rng = &mut jf_utils::test_rng();
        for (m, vk_id_one, vk_id_two) in izip!(
            (2..8),
            [
                Some(VerificationKeyId::Client),
                Some(VerificationKeyId::Deposit),
            ],
            [
                Some(VerificationKeyId::Client),
                Some(VerificationKeyId::Deposit),
            ],
        ) {
            let circuit_one = gen_circuit_for_test::<Fr254>(m, 3, PlonkType::UltraPlonk, true)?;
            let circuit_two = gen_circuit_for_test::<Fr254>(m, 4, PlonkType::UltraPlonk, true)?;
            let pi_one = circuit_one.public_input()?[0];
            let pi_two = circuit_two.public_input()?[0];

            let srs_size = circuit_one.srs_size(true)?;

            let srs = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(rng, srs_size)?;

            let (pk_one, vk_one) =
                FFTPlonk::<Kzg>::preprocess(&srs, vk_id_one, &circuit_one, true)?;
            let (pk_two, vk_two) =
                FFTPlonk::<Kzg>::preprocess(&srs, vk_id_two, &circuit_two, true)?;

            let circuits = [circuit_one, circuit_two];
            let pks = [pk_one.clone(), pk_two.clone()];
            let vks = [vk_one.clone(), vk_two.clone()];
            let pis = [pi_one, pi_two];

            let outputs = circuits
                .iter()
                .zip(pks)
                .map(|(circuit, pk)| {
                    FFTPlonk::<Kzg>::recursive_prove::<_, _, RescueTranscript<Fr254>>(
                        rng, circuit, &pk, None, true,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let old_accs: [AtomicInstance<Kzg>; 4] = (0..4)
                .map(|_| {
                    let mut poly = DensePolynomial::<Fr254>::rand(srs_size, rng);

                    poly[0] = Fr254::zero();
                    let comm: Affine<BnConfig> = Kzg::commit(&pk_one.commit_key, &poly)?;
                    let (proof, eval): (UnivariateKzgProof<Bn254>, Fr254) =
                        Kzg::open(&pk_one.commit_key, &poly, &Fr254::zero())?;
                    assert_eq!(eval, Fr254::zero());
                    assert!(Kzg::verify(
                        &vk_one.open_key,
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

            // We construct an intermediate circuit to process the bases computations of the verification.
            let mut bases_verifier_circuit = PlonkCircuit::<Fq254>::new_ultra_plonk(8);
            // This is the first round, so the `VerifyingKeyBasesVar`s can be different.
            let vk_bases_var = vks
                .iter()
                .map(|vk| VerifyingKeyScalarsAndBasesVar::new(&mut bases_verifier_circuit, vk))
                .collect::<Result<Vec<VerifyingKeyScalarsAndBasesVar<Kzg>>, CircuitError>>()?;

            // We store the scalars and bases from the proofs into the relevant struct.
            // In particular, the bases are stored as `Variable`s in the circuit, as we
            // need to reuse them later in the circuit and in the next circuit.
            // We also prepare the relevant info from each of the proofs.
            let output_pcs_info_var_pair: [(Bn254OutputScalarsAndBasesVar, PcsInfoBasesVar<Kzg>); 2] = outputs.iter()
                .zip(vk_bases_var.iter())
                .map(|(output, vk)| {
                    let verifier = FFTVerifier::new(vk.domain_size)?;
                    verifier.prepare_pcs_info_with_bases_var::<RescueTranscript<Fr254>>(
                        vk,
                        &[output.pi_hash],
                        output,
                        &None,
                        &mut bases_verifier_circuit,
                        true,
                    )
                })
                .collect::<Result<Vec<(Bn254OutputScalarsAndBasesVar, PcsInfoBasesVar<Kzg>)>, PlonkError>>()?
                .try_into()
                .map_err(|_| PlonkError::InvalidParameters("outputs must have length 2".to_string()))?;

            let output_vars = output_pcs_info_var_pair
                .iter()
                .map(|(output, _)| output.clone())
                .collect::<Vec<Bn254OutputScalarsAndBasesVar>>();
            let pcs_info_vars = output_pcs_info_var_pair
                .iter()
                .map(|(_, pcs_info)| pcs_info.clone())
                .collect::<Vec<PcsInfoBasesVar<Kzg>>>();

            let transcript = &mut output_vars[0].transcript.clone();
            transcript.merge(&output_vars[1].transcript)?;

            let r = transcript.squeeze_scalar_challenge::<BnConfig>(b"r")?;

            // Calculate the various powers of r needed, they start at 1 and end at r^(pcs_infos.len()).
            let r_powers = iter::successors(Some(Fr254::one()), |x| Some(*x * r))
                .take(6)
                .collect::<Vec<Fr254>>();

            let (scalars, instance_base_vars) =
                combine_fft_proof_scalars_round_one(&pcs_info_vars, &r_powers);

            let mut scalars_verifier_circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(8);
            let vk_var_one =
                VerifyingKeyNativeScalarsVar::new(&mut scalars_verifier_circuit, &vk_one)?;
            let vk_var_two =
                VerifyingKeyNativeScalarsVar::new(&mut scalars_verifier_circuit, &vk_two)?;

            // Create variables from the proofs
            let output_var_pairs = outputs
                .iter()
                .map(|output| {
                    let proof = ProofVarNative::from_struct(&output.proof, &mut scalars_verifier_circuit)?;
                    let pi_hash = scalars_verifier_circuit.create_variable(output.pi_hash)?;
                    let proof_evals = ProofScalarsVarNative::from_struct(&proof, pi_hash)?;
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

            // Here we assume a max domain size of 2^10
            let recursion_scalars = calculate_recursion_scalars_base(
                &output_scalar_vars,
                &output_base_vars,
                &[vk_var_one, vk_var_two],
                1 << 10,
                &mut scalars_verifier_circuit,
                true,
            )?;
            let (instance_scalar_vars, proof_scalar_vars) =
                recursion_scalars.split_at(recursion_scalars.len() - 6);

            // We check the witness values of the scalars are as expected.
            assert_eq!(instance_scalar_vars.len(), scalars.len());
            assert_eq!(proof_scalar_vars.len(), r_powers.len());

            for (var, &expected) in instance_scalar_vars.iter().zip(scalars.iter()) {
                let witness = scalars_verifier_circuit.witness(*var)?;
                assert_eq!(witness, expected);
            }
            for (var, &expected) in proof_scalar_vars
                .iter()
                .skip(4)
                .zip(r_powers.iter().take(2))
            {
                let witness = scalars_verifier_circuit.witness(*var)?;
                assert_eq!(witness, expected);
            }
            for (var, &expected) in proof_scalar_vars
                .iter()
                .take(4)
                .zip(r_powers.iter().skip(2))
            {
                let witness = scalars_verifier_circuit.witness(*var)?;
                assert_eq!(witness, expected);
            }

            let non_acc_bigints = instance_scalar_vars
                .iter()
                .map(|s| scalars_verifier_circuit.witness(*s).unwrap().into_bigint())
                .collect::<Vec<_>>();

            let acc_bigints = proof_scalar_vars[..4]
                .iter()
                .map(|s| scalars_verifier_circuit.witness(*s).unwrap().into_bigint())
                .collect::<Vec<_>>();

            let proof_bigints = proof_scalar_vars[4..]
                .iter()
                .chain(proof_scalar_vars[..4].iter())
                .map(|s| scalars_verifier_circuit.witness(*s).unwrap().into_bigint())
                .collect::<Vec<_>>();

            let opening_proofs = outputs
                .iter()
                .map(|output| output.proof.opening_proof.proof)
                .chain(old_accs.iter().map(|acc| acc.opening_proof.proof))
                .collect::<Vec<_>>();

            let pcs_infos = izip!(outputs, pis, vks)
                .map(|(output, pi, vk)| {
                    let fft_verifier = FFTVerifier::<Kzg>::new(vk.domain_size)?;
                    fft_verifier.prepare_pcs_info::<RescueTranscript<Fr254>>(
                        &vk,
                        &[pi],
                        &output.proof,
                        &None,
                        true,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            pcs_infos.iter().for_each(|pcs_info| {
                assert!(
                    FFTVerifier::<Kzg>::verify_opening_proofs(&vk_one.open_key, pcs_info).unwrap()
                );
            });

            let comms_lists = pcs_infos
                .iter()
                .map(|pcs_info| {
                    let mut comms = pcs_info.comm_scalars_and_bases.bases()[..47].to_vec();

                    let _ = comms.remove(22);
                    comms
                })
                .collect::<Vec<Vec<_>>>();

            let mut comms = comms_lists[0].clone();
            comms.extend_from_slice(&comms_lists[1]);
            comms.extend_from_slice(&opening_proofs[..2]);

            // We now check the commitments match up with those computed in-circuit
            assert_eq!(comms.len(), instance_base_vars.len());
            for (comm, var) in comms.iter().zip(instance_base_vars.iter()) {
                let x_val = bases_verifier_circuit.witness(var.get_x())?;
                let y_val = bases_verifier_circuit.witness(var.get_y())?;
                assert_eq!(x_val, *comm.x().unwrap());
                assert_eq!(y_val, *comm.y().unwrap());
            }

            let proof_part =
                Projective::<BnConfig>::msm_bigint(&comms, &non_acc_bigints).into_affine();
            let acc_part = Projective::<BnConfig>::msm_bigint(
                &old_accs.iter().map(|a| a.comm).collect::<Vec<_>>(),
                &acc_bigints,
            )
            .into_affine();

            let new_proof =
                Projective::<BnConfig>::msm_bigint(&opening_proofs, &proof_bigints).into_affine();

            let instance = proof_part + acc_part;

            let pcs_proof = UnivariateKzgProof::<Bn254> { proof: new_proof };

            assert!(Kzg::verify(
                &vk_one.open_key,
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
