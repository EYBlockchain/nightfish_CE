//! This file defines functions to take in two `RecursiveOutputs` and perform the necessary accumulation steps,
//! they return the scalars necessary for the various MSMs in the next circuit.

use core::iter;

use ark_bn254::{g1::Config as BnConfig, Bn254, Fq as Fq254, Fr as Fr254};
use ark_ec::short_weierstrass::Affine;
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};

use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{cfg_iter, string::ToString, sync::Arc, vec, vec::Vec};
use jf_primitives::{
    circuit::rescue::RescueNativeGadget,
    pcs::{
        prelude::{UnivariateKzgPCS, UnivariateKzgProof},
        PolynomialCommitmentScheme,
    },
};

use itertools::izip;
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{
            EmulMultiScalarMultiplicationCircuit, MultiScalarMultiplicationCircuit, Point,
            PointVariable,
        },
        EmulatedVariable,
    },
    Circuit, PlonkCircuit, Variable,
};
use jf_utils::{bytes_to_field_elements, fq_to_fr, fr_to_fq};
use nf_curves::grumpkin::{short_weierstrass::SWGrumpkin, Grumpkin};
use rayon::prelude::*;
use sha3::{Digest, Keccak256};

use crate::{
    errors::PlonkError,
    nightfall::{
        accumulation::accumulation_structs::{AtomicInstance, PCSWitness},
        circuit::{
            plonk_partial_verifier::{MLEVerifyingKeyVar, SAMLEProofVar},
            verify_zeromorph::verify_zeromorph_circuit,
        },
        ipa_structs::{VerifyingKey, VK},
        ipa_verifier::{FFTVerifier, PcsInfo},
        mle::{
            mle_structs::{MLEProvingKey, MLEVerifyingKey},
            zeromorph::Zeromorph,
            MLEPlonk,
        },
        FFTPlonk, UnivariateIpaPCS,
    },
    proof_system::RecursiveOutput,
    recursion::circuits::{
        challenges::reconstruct_mle_challenges,
        emulated_mle_arithmetic::emulated_combine_mle_proof_scalars,
    },
    transcript::{rescue::RescueTranscriptVar, CircuitTranscript, RescueTranscript, Transcript},
};

use super::circuits::{
    challenges::MLEProofChallenges,
    fft_arithmetic::{calculate_recursion_scalars, calculate_recursion_scalars_base},
    mle_arithmetic::combine_mle_proof_scalars,
    split_acc::SplitAccumulationInfo,
    Zmorph,
};

pub(crate) type Bn254Output = RecursiveOutput<Kzg, FFTPlonk<Kzg>, RescueTranscript<Fr254>>;

type Verifier = FFTVerifier<Kzg>;

type Kzg = UnivariateKzgPCS<Bn254>;

type ZeromorphPCS = Zeromorph<UnivariateIpaPCS<Grumpkin>>;

pub(crate) type GrumpkinOutput =
    RecursiveOutput<ZeromorphPCS, MLEPlonk<ZeromorphPCS>, RescueTranscript<Fr254>>;

/// This struct represent the information needed to accumulate Bn254 proofs.
/// Each Grumpkin circuit will take in a `Bn254RecursiveInfo` struct and use it to build the necessary constraints.
///
/// These constraints are, in order:
/// 1. Perform the MSMs for accumulation
/// 2. Verify scalar arithmetic from 4 previous Grumpkin proofs.
/// 3. Update the public input hash by verifying that the pi_hash of the Bn254 proofs is the expected one.
///
/// For this we need the following information:
/// 1. The two [`RecursiveOutput`] structs from the previous Bn254 circuits.
/// 2. The four [`RecursiveOutput`] structs from the previous Grumpkin circuits.
/// 3. The four old Bn254 accumulators.
/// 4. Any Implementation specific public input relating to the two proofs.
/// 5. The Challenges used for verifying MLE arithmetic in the Grumpkin proofs.
/// 6. The information for verifying arithmetic to do with split accumulation.
#[derive(Debug, Clone, Default)]
pub struct Bn254RecursiveInfo {
    /// The two Bn254 outputs from the previous round.
    pub bn254_outputs: [Bn254Output; 2],
    /// The four Grumpkin outputs from the round two previous.
    pub grumpkin_outputs: [GrumpkinOutput; 4],
    /// The four old Bn254 accumulators.
    pub old_accumulators: [AtomicInstance<Kzg>; 4],
    /// The two Grumpkin accumulators that will be forwarded by this proof.
    pub forwarded_acumulators: [PCSWitness<ZeromorphPCS>; 2],
    /// The implementation specific public inputs.
    pub specific_pi: [Vec<Fr254>; 2],
    /// The prepped challenges for the MLE arithmetic in the Grumpkin proofs.
    pub challenges: [MLEProofChallenges<Fq254>; 4],
    /// Split accumulation information.
    pub split_acc_info: [SplitAccumulationInfo; 2],
}

impl Bn254RecursiveInfo {
    /// Create a new instance of the `Bn254RecursiveInfo` struct.
    pub fn new(
        bn254_outputs: [Bn254Output; 2],
        grumpkin_outputs: [GrumpkinOutput; 4],
        old_accumulators: [AtomicInstance<Kzg>; 4],
        forwarded_acumulators: [PCSWitness<ZeromorphPCS>; 2],
        specific_pi: [Vec<Fr254>; 2],
        challenges: [MLEProofChallenges<Fq254>; 4],
        split_acc_info: [SplitAccumulationInfo; 2],
    ) -> Self {
        Self {
            bn254_outputs,
            grumpkin_outputs,
            old_accumulators,
            forwarded_acumulators,
            specific_pi,
            challenges,
            split_acc_info,
        }
    }

    /// Constructs an instance of the struct from 2 [`Bn254Output`] and 2 [`Bn254CircuitOutput`]
    pub fn from_parts(
        bn254_outputs: [Bn254Output; 2],
        circuit_outputs: [Bn254CircuitOutput; 2],
    ) -> Self {
        let [co0, co1] = circuit_outputs;
        let Bn254CircuitOutput {
            specific_pi: specific_pi_0,
            accumulators: [old_acc00, old_acc01],
            output_accumulator: out_acc_0,
            challenges: [mle_challenges00, mle_challenges01],
            split_acc_info: sai0,
            grumpkin_outputs: [out00, out01],
        } = co0;
        let Bn254CircuitOutput {
            specific_pi: specific_pi_1,
            accumulators: [old_acc10, old_acc11],
            output_accumulator: out_acc_1,
            challenges: [mle_challenges10, mle_challenges11],
            split_acc_info: sai1,
            grumpkin_outputs: [out10, out11],
        } = co1;

        let grumpkin_outputs = [out00, out01, out10, out11];
        let old_accumulators = [old_acc00, old_acc01, old_acc10, old_acc11];
        let forwarded_accumulators = [out_acc_0, out_acc_1];
        let specific_pi = [specific_pi_0, specific_pi_1];
        let challenges = [
            mle_challenges00,
            mle_challenges01,
            mle_challenges10,
            mle_challenges11,
        ];
        let split_acc_info = [sai0, sai1];

        Self::new(
            bn254_outputs,
            grumpkin_outputs,
            old_accumulators,
            forwarded_accumulators,
            specific_pi,
            challenges,
            split_acc_info,
        )
    }
}

/// This struct is the output of the function that builds the Grumpkin circuit.
#[derive(Debug, Clone)]
pub struct GrumpkinCircuitOutput {
    /// The two Bn254 recursive outputs that all the other values are calculated from.
    pub bn254_outputs: [Bn254Output; 2],
    /// The implementation specific public input from this circuit
    pub specific_pi: Vec<Fq254>,
    /// The Grumpkin accumulators that this circuit forwarded
    pub accumulators: [PCSWitness<ZeromorphPCS>; 2],
    /// The Bn254 accumulator output/verified by this circuit
    pub output_accumulator: AtomicInstance<Kzg>,
    /// The accumulators used in the MSM for future transcript verification
    pub transcript_accumulators: [AtomicInstance<Kzg>; 4],
}

impl GrumpkinCircuitOutput {
    /// Constructs a new instance of the struct
    pub fn new(
        bn254_outputs: [Bn254Output; 2],
        specific_pi: Vec<Fq254>,
        accumulators: [PCSWitness<ZeromorphPCS>; 2],
        output_accumulator: AtomicInstance<Kzg>,
        transcript_accumulators: [AtomicInstance<Kzg>; 4],
    ) -> Self {
        Self {
            bn254_outputs,
            specific_pi,
            accumulators,
            output_accumulator,
            transcript_accumulators,
        }
    }
}

/// This struct represent the information needed to accumulate Grumpkin proofs.
/// Each Bn254 circuit will take in a `GrumpkinRecursiveInfo` struct and use it to build the necessary constraints.
///
/// These constraints are, in order:
/// 1. Transcript Hashing for the previous Bn254 proofs
/// 2. Verify scalar arithmetic from 4 previous Bn254 proofs.
/// 3. Use this together with any forwarded information to calculate the pi hashes of the two Grumpkin proofs
/// 4. Verify the transcripts and hashing for the Grumpkin proofs
/// 5. Perform the Grumpkin accumulation MSM
/// 6. Update the public input hashes
///
/// For this we need the following information:
/// 1. The two [`RecursiveOutput`] structs from the previous Grumpkin circuits.
/// 2. The four [`RecursiveOutput`] structs from the previous Bn254 circuits.
/// 3. The four old Grumpkin accumulators.
/// 4. Any Implementation specific public input relating to the two proofs.
#[derive(Debug, Clone, Default)]
pub struct GrumpkinRecursiveInfo {
    /// The two Grumpkin outputs from the previous round.
    pub grumpkin_outputs: [GrumpkinOutput; 2],
    /// The four Bn254 outputs from the round two previous.
    pub bn254_outputs: [Bn254Output; 4],
    /// The four old Grumpkin accumulators.
    pub old_accumulators: [PCSWitness<ZeromorphPCS>; 4],
    /// The two Bn254 accumulators that will be forwarded by this proof.
    pub forwarded_acumulators: [AtomicInstance<Kzg>; 2],
    /// The eight old Bn254 accumulators (2 for each `bn254_output`) for transcript verification
    pub transcript_accumulators: [AtomicInstance<Kzg>; 8],
    /// The implementation specific public inputs.
    pub specific_pi: [Vec<Fr254>; 2],
}

impl GrumpkinRecursiveInfo {
    /// Create a new instance of the `Bn254RecursiveInfo` struct.
    pub fn new(
        grumpkin_outputs: [GrumpkinOutput; 2],
        bn254_outputs: [Bn254Output; 4],
        old_accumulators: [PCSWitness<ZeromorphPCS>; 4],
        forwarded_acumulators: [AtomicInstance<Kzg>; 2],
        transcript_accumulators: [AtomicInstance<Kzg>; 8],
        specific_pi: [Vec<Fr254>; 2],
    ) -> Self {
        Self {
            grumpkin_outputs,
            bn254_outputs,
            old_accumulators,
            forwarded_acumulators,
            transcript_accumulators,
            specific_pi,
        }
    }

    /// Constructs an instance of the struct from 2 [`GrumpkinOutput`] and 2 [`GrumpkinCircuitOutput`]
    pub fn from_parts(
        grumpkin_outputs: [GrumpkinOutput; 2],
        circuit_outputs: [GrumpkinCircuitOutput; 2],
    ) -> Self {
        let [co0, co1] = circuit_outputs;
        let GrumpkinCircuitOutput {
            bn254_outputs: [bn25400, bn25401],
            specific_pi: specific_pi_0,
            accumulators: [old_acc00, old_acc01],
            output_accumulator: output_acc0,
            transcript_accumulators: [ta0, ta1, ta2, ta3],
        } = co0;

        let GrumpkinCircuitOutput {
            bn254_outputs: [bn25410, bn25411],
            specific_pi: specific_pi_1,
            accumulators: [old_acc10, old_acc11],
            output_accumulator: output_acc1,
            transcript_accumulators: [ta4, ta5, ta6, ta7],
        } = co1;

        let bn254_outputs = [bn25400, bn25401, bn25410, bn25411];
        let mut old_accumulators = [old_acc00, old_acc01, old_acc10, old_acc11];
        let forwarded_accumulators = [output_acc0, output_acc1];

        // We resize the accumulators if the new proof is larger (should only happen when going from base to merge circuits).
        let proof_num_vars = grumpkin_outputs[0].proof.polynomial.num_vars;
        let acc_num_vars = old_accumulators[0].poly.num_vars;
        if proof_num_vars > acc_num_vars {
            old_accumulators.iter_mut().for_each(|acc| {
                let mut evals = acc.poly.to_evaluations();
                evals.resize(1 << proof_num_vars, Fq254::zero());
                acc.poly = Arc::new(DenseMultilinearExtension::<Fq254>::from_evaluations_vec(
                    proof_num_vars,
                    evals,
                ));
                acc.point.push(Fq254::zero());
            });
        }
        let transcript_accumulators = [ta0, ta1, ta2, ta3, ta4, ta5, ta6, ta7];
        let specific_pi = [
            specific_pi_0
                .iter()
                .map(fq_to_fr::<Fq254, BnConfig>)
                .collect::<Vec<Fr254>>(),
            specific_pi_1
                .iter()
                .map(fq_to_fr::<Fq254, BnConfig>)
                .collect::<Vec<Fr254>>(),
        ];
        Self::new(
            grumpkin_outputs,
            bn254_outputs,
            old_accumulators,
            forwarded_accumulators,
            transcript_accumulators,
            specific_pi,
        )
    }
}

/// This struct is the output of the function that builds the Grumpkin circuit.
#[derive(Debug, Clone)]
pub struct Bn254CircuitOutput {
    /// The implementation specific public input from this circuit
    pub specific_pi: Vec<Fr254>,
    /// The Bn254 accumulators that this circuit forwarded
    pub accumulators: [AtomicInstance<Kzg>; 2],
    /// The Grumpkin accumulator output/verified by this circuit
    pub output_accumulator: PCSWitness<ZeromorphPCS>,
    /// The challenges verified by this circuit for the next grumpkin circuit
    pub challenges: [MLEProofChallenges<Fq254>; 2],
    /// The accumulation info required for the next Grumpkin circuit.
    pub split_acc_info: SplitAccumulationInfo,
    /// The two [`GrumpkinOutput`] all the other information is computed from.
    pub grumpkin_outputs: [GrumpkinOutput; 2],
}

impl Bn254CircuitOutput {
    /// Constructs a new instance of the struct
    pub fn new(
        specific_pi: Vec<Fr254>,
        accumulators: [AtomicInstance<Kzg>; 2],
        output_accumulator: PCSWitness<ZeromorphPCS>,
        challenges: [MLEProofChallenges<Fq254>; 2],
        split_acc_info: SplitAccumulationInfo,
        grumpkin_outputs: [GrumpkinOutput; 2],
    ) -> Self {
        Self {
            specific_pi,
            accumulators,
            output_accumulator,
            challenges,
            split_acc_info,
            grumpkin_outputs,
        }
    }
}

/// This function takes in two [`RecursiveOutput<UnivariateKzgPcs<Bn254>, FFTPlonk<UnivariateKzgPcs<Bn254>>, RescueTranscript<Fr254>>`] structs and prepares
/// the accumulator to be verified along with the scalars for the relevant MSMs to be performed in the next circuit.
///
/// In order to minimise the size of the MSMs that we have to compute in the next circuit we combine some of the scalars together where possible.
///
/// We recalculate the PI commitments separately and then add them on at the end, this is because we can use a fixed base MSM for these as
/// the SRS is fixed from the start. Now because the number of public inputs in both cases will be the same we can do both in one MSM.
///
/// Since all of the selector commitments are the same for both proofs we can combine all of their scalars together to eliminate another 21 bases from the MSM.
/// Using an almost identical argument we combine all the scalars for the permutation commitments, eliminating another 5 bases.
///
/// The generic parameter `IS_FIRST_ROUND` is used to determine if the first round of the recursion is being prepared. If it is the first round we do not
/// expect any accumulators to be passed in with the proofs.
///
/// For the accumulators we structure public inputs so that the forwarded accumulators are always the last two sets of public inputs.
/// Since atomic accumulators have fixed length (and instance point and a proof point) we can work out which of the public inputs relate to forwarded accumulators using this fact.
pub fn prove_bn254_accumulation<const IS_FIRST_ROUND: bool>(
    bn254info: &Bn254RecursiveInfo,
    vk_bn254: &[VerifyingKey<Kzg>; 2],
    vk_grumpkin: &MLEVerifyingKey<ZeromorphPCS>,
    specific_pi_fn: impl Fn(
        &[Vec<Variable>],
        &mut PlonkCircuit<Fq254>,
    ) -> Result<Vec<Variable>, CircuitError>,
    circuit: &mut PlonkCircuit<Fq254>,
) -> Result<GrumpkinCircuitOutput, PlonkError> {
    if !IS_FIRST_ROUND && (vk_bn254[0].hash() != vk_bn254[1].hash()) {
        return Err(PlonkError::InvalidParameters(
            "Can only have differing verification keys in the first round".to_string(),
        ));
    }
    let outputs = &bn254info.bn254_outputs;
    // First things first we prepare the relevant info from each of the proofs.
    let pcs_infos = cfg_iter!(outputs)
        .zip(cfg_iter!(vk_bn254))
        .map(|(output, vk)| {
            let verifier = Verifier::new(vk.domain_size)?;
            verifier.prepare_pcs_info::<RescueTranscript<Fr254>>(
                vk,
                &[output.pi_hash],
                &output.proof,
                &None,
            )
        })
        .collect::<Result<Vec<PcsInfo<Kzg>>, PlonkError>>()?;

    // Now we merge the transcripts from the two proofs. we do this to avoid having to re-append all the commitments to a new transcript.
    // TODO:  Double check that this still provides adequate security.

    // Unwrap is safe here because we checked tht the slice was non-empty at the beginning.
    let transcript = &mut outputs[0].transcript.clone();

    transcript.merge(&outputs[1].transcript)?;

    // Append the old bn254 accumulators to the transcript.
    bn254info.old_accumulators.iter().try_for_each(|acc| {
        transcript.append_curve_point(b"comm", &acc.comm)?;
        transcript.append_curve_point(b"opening proof", &acc.opening_proof.proof)
    })?;

    let r = transcript.squeeze_scalar_challenge::<BnConfig>(b"r")?;

    // Calculate the various powers of r needed, they start at 1 and end at r^(pcs_infos.len()).
    let r_powers = iter::successors(Some(Fr254::one()), |x| Some(*x * r))
        .take(6)
        .collect::<Vec<Fr254>>();

    let (scalars, mut bases) = if !IS_FIRST_ROUND {
        combine_fft_proof_scalars(&pcs_infos, &r_powers)
    } else {
        combine_fft_proof_scalars_round_one(&pcs_infos, &r_powers)
    };

    // Append the extra accumulator commitments to `bases` for the atomic accumulation.
    bases.extend_from_slice(
        &bn254info
            .old_accumulators
            .iter()
            .map(|acc| acc.comm)
            .collect::<Vec<_>>(),
    );

    let proof_bases = outputs
        .iter()
        .map(|output| output.proof.opening_proof.proof)
        .chain(
            bn254info
                .old_accumulators
                .iter()
                .map(|acc| acc.opening_proof.proof),
        )
        .collect::<Vec<_>>();
    // Now perform the MSM for the accumulation, we only do this in the circuit as the procedure for proving and verifying is identical.
    let scalar_vars = scalars
        .iter()
        .map(|s| circuit.create_variable(fr_to_fq::<Fq254, BnConfig>(s)))
        .collect::<Result<Vec<Variable>, CircuitError>>()?;
    let proof_scalar_vars = r_powers
        .iter()
        .map(|s| circuit.create_variable(fr_to_fq::<Fq254, BnConfig>(s)))
        .collect::<Result<Vec<Variable>, CircuitError>>()?;
    let instance_scalar_vars = [scalar_vars.as_slice(), &proof_scalar_vars[2..]].concat();

    let instance_base_vars = bases
        .iter()
        .map(|base| circuit.create_point_variable(&Point::<Fq254>::from(*base)))
        .collect::<Result<Vec<PointVariable>, CircuitError>>()?;
    let proof_base_vars = proof_bases
        .iter()
        .map(|base| circuit.create_point_variable(&Point::<Fq254>::from(*base)))
        .collect::<Result<Vec<PointVariable>, CircuitError>>()?;

    let acc_instance = MultiScalarMultiplicationCircuit::<Fq254, BnConfig>::msm(
        circuit,
        &instance_base_vars,
        &instance_scalar_vars,
    )?;

    let acc_proof = MultiScalarMultiplicationCircuit::<Fq254, BnConfig>::msm(
        circuit,
        &proof_base_vars[1..],
        &proof_scalar_vars[1..],
    )?;

    // Now we verify scalar arithmetic for the four previous Grumpkin proofs and the pi_hash.
    if !IS_FIRST_ROUND {
        let scalars_and_acc_evals: Vec<(Vec<Variable>, Vec<Variable>)> = izip!(
            bn254info.grumpkin_outputs.chunks_exact(2),
            bn254info.challenges.chunks_exact(2),
            bn254info.split_acc_info.iter()
        )
        .map(|(output_pair, challenges_pair, split_acc_info)| {
            let (scalars, eval) = combine_mle_proof_scalars(
                output_pair,
                challenges_pair,
                split_acc_info,
                vk_grumpkin,
                circuit,
            )?;

            // Find the byte length of the scalar field (minus one).
            let field_bytes_length = (Fq254::MODULUS_BIT_SIZE as usize - 1) / 8;

            let value = circuit.witness(eval)?;
            let bytes = value.into_bigint().to_bytes_le();
            let (low, high) = bytes.split_at(field_bytes_length);

            let low_var = circuit.create_variable(Fq254::from_le_bytes_mod_order(low))?;

            let high_var = circuit.create_variable(Fq254::from_le_bytes_mod_order(high))?;
            let bits = field_bytes_length * 8;
            let coeff = Fq254::from(2u32).pow([bits as u64]);

            circuit.lc_gate(
                &[low_var, high_var, circuit.zero(), circuit.zero(), eval],
                &[Fq254::one(), coeff, Fq254::zero(), Fq254::zero()],
            )?;
            Ok((scalars, vec![low_var, high_var]))
        })
        .collect::<Result<Vec<(Vec<Variable>, Vec<Variable>)>, CircuitError>>()?;

        let impl_pi_vars: Vec<Vec<Variable>> = bn254info
            .specific_pi
            .iter()
            .map(|pi| {
                pi.iter()
                    .map(|x| circuit.create_variable(fr_to_fq::<Fq254, BnConfig>(x)))
                    .collect::<Result<Vec<Variable>, CircuitError>>()
            })
            .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?;

        let old_acc_vars = bn254info
            .old_accumulators
            .chunks_exact(2)
            .map(|acc_pair| {
                Ok(acc_pair
                    .iter()
                    .map(|acc| {
                        let x = bytes_to_field_elements::<_, Fq254>(
                            acc.comm.x.into_bigint().to_bytes_le(),
                        )[1..]
                            .to_vec();

                        let y = bytes_to_field_elements::<_, Fq254>(
                            acc.comm.y.into_bigint().to_bytes_le(),
                        )[1..]
                            .to_vec();

                        let proof_x = bytes_to_field_elements::<_, Fq254>(
                            acc.opening_proof.proof.x.into_bigint().to_bytes_le(),
                        )[1..]
                            .to_vec();

                        let proof_y = bytes_to_field_elements::<_, Fq254>(
                            acc.opening_proof.proof.y.into_bigint().to_bytes_le(),
                        )[1..]
                            .to_vec();

                        x.into_iter()
                            .chain(y)
                            .chain(proof_x)
                            .chain(proof_y)
                            .map(|x| circuit.create_variable(x))
                            .collect::<Result<Vec<Variable>, CircuitError>>()
                    })
                    .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<Variable>>())
            })
            .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?;

        let forwarded_accs: Vec<Vec<Variable>> = bn254info
            .forwarded_acumulators
            .iter()
            .map(|acc| {
                let comm_x = circuit.create_variable(fq_to_fr::<Fr254, SWGrumpkin>(&acc.comm.x))?;
                let comm_y = circuit.create_variable(fq_to_fr::<Fr254, SWGrumpkin>(&acc.comm.y))?;

                Ok(vec![comm_x, comm_y])
            })
            .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?;

        let old_pi_hashes: Vec<Vec<Variable>> = bn254info
            .grumpkin_outputs
            .chunks_exact(2)
            .map(|output_pair| {
                output_pair
                    .iter()
                    .map(|output| circuit.create_variable(output.pi_hash))
                    .collect::<Result<Vec<Variable>, CircuitError>>()
            })
            .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?;

        let pi_hashes = izip!(
            scalars_and_acc_evals.iter(),
            impl_pi_vars.iter(),
            old_acc_vars.iter(),
            forwarded_accs.iter(),
            old_pi_hashes.iter(),
        )
        .map(
            |((scalars, acc_eval), pi, old_acc, forwarded_acc, old_hashes)| {
                let prepped_scalars = scalars
                    .iter()
                    .map(|&var| convert_to_hash_form_fq254(circuit, var))
                    .collect::<Result<Vec<[Variable; 2]>, CircuitError>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<Variable>>();
                let in_vars = [
                    pi.as_slice(),
                    prepped_scalars.as_slice(),
                    old_acc.as_slice(),
                    forwarded_acc.as_slice(),
                    acc_eval.as_slice(),
                    old_hashes.as_slice(),
                ]
                .concat();

                let pi_hash_pre =
                    RescueNativeGadget::<Fq254>::rescue_sponge_with_padding(circuit, &in_vars, 1)?
                        [0];

                let value = circuit.witness(pi_hash_pre)?;
                let bytes = value.into_bigint().to_bytes_le();
                let (challenge, leftover) = bytes.split_at(31);

                let pi_hash = circuit.create_variable(Fq254::from_le_bytes_mod_order(challenge))?;
                let leftover_var =
                    circuit.create_variable(Fq254::from_le_bytes_mod_order(leftover))?;

                circuit.enforce_in_range(pi_hash, 8 * 31)?;
                circuit.enforce_in_range(leftover_var, 6)?;

                let coeff = Fq254::from(2u32).pow([248u64]);

                circuit.lc_gate(
                    &[
                        pi_hash,
                        leftover_var,
                        circuit.zero(),
                        circuit.zero(),
                        pi_hash_pre,
                    ],
                    &[Fq254::one(), coeff, Fq254::zero(), Fq254::zero()],
                )?;
                Ok(pi_hash)
            },
        )
        .collect::<Result<Vec<Variable>, CircuitError>>()?;
        // For checking correctness during testing
        #[cfg(test)]
        {
            for (circuit_hash, actual_hash) in pi_hashes.iter().zip(bn254info.bn254_outputs.iter())
            {
                assert_eq!(
                    circuit.witness(*circuit_hash).unwrap(),
                    fr_to_fq::<Fq254, BnConfig>(&actual_hash.pi_hash)
                );
            }
        }

        // Now do the specific pi checks.
        let specific_pi_vars: Vec<Variable> = specific_pi_fn(&impl_pi_vars, circuit)?;

        specific_pi_vars
            .iter()
            .try_for_each(|pi| circuit.set_variable_public(*pi))?;

        instance_scalar_vars
            .iter()
            .try_for_each(|var| circuit.set_variable_public(*var))?;
        proof_scalar_vars[..2]
            .iter()
            .try_for_each(|var| circuit.set_variable_public(*var))?;
        scalars_and_acc_evals
            .iter()
            .zip(bn254info.forwarded_acumulators.iter())
            .try_for_each(|((_, acc_eval), forwarded_acc)| {
                let _ = circuit
                    .create_public_variable(fq_to_fr::<Fr254, SWGrumpkin>(&forwarded_acc.comm.x))?;
                let _ = circuit
                    .create_public_variable(fq_to_fr::<Fr254, SWGrumpkin>(&forwarded_acc.comm.y))?;

                let low_var = acc_eval[0];
                let high_var = acc_eval[1];

                let field_bytes_length = (Fq254::MODULUS_BIT_SIZE as usize - 1) / 8;

                let bits = field_bytes_length * 8;
                let coeff = Fq254::from(2u32).pow([bits as u64]);
                let actual_eval = circuit.lc(
                    &[low_var, high_var, circuit.zero(), circuit.zero()],
                    &[Fq254::one(), coeff, Fq254::zero(), Fq254::zero()],
                )?;
                circuit.set_variable_public(actual_eval)
            })?;
        acc_instance
            .get_coords()
            .iter()
            .try_for_each(|&var| circuit.set_variable_public(var))?;
        acc_proof
            .get_coords()
            .iter()
            .try_for_each(|&var| circuit.set_variable_public(var))?;

        pi_hashes
            .iter()
            .try_for_each(|x| circuit.set_variable_public(*x))?;

        let specific_pi_out = specific_pi_vars
            .iter()
            .map(|var| circuit.witness(*var))
            .collect::<Result<Vec<Fq254>, CircuitError>>()?;

        let acc_instance_point = circuit.point_witness(&acc_instance)?;
        let acc_proof_point = circuit.point_witness(&acc_proof)?;

        let acc_instance_real: Affine<BnConfig> = acc_instance_point.into();
        let acc_proof_real: Affine<BnConfig> = acc_proof_point.into();

        let kzg_proof = UnivariateKzgProof::<Bn254> {
            proof: acc_proof_real,
        };
        let output_accumulator =
            AtomicInstance::<Kzg>::new(acc_instance_real, Fr254::zero(), Fr254::zero(), kzg_proof);
        Ok(GrumpkinCircuitOutput::new(
            bn254info.bn254_outputs.clone(),
            specific_pi_out,
            bn254info.forwarded_acumulators.clone(),
            output_accumulator,
            bn254info.old_accumulators.clone(),
        ))
    } else {
        let impl_pi_vars = bn254info
            .specific_pi
            .iter()
            .map(|pi_vec| {
                pi_vec
                    .iter()
                    .map(|s| circuit.create_variable(fr_to_fq::<Fq254, BnConfig>(s)))
                    .collect::<Result<Vec<Variable>, CircuitError>>()
            })
            .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?;

        let pi_hashes = impl_pi_vars
            .iter()
            .map(|pi_vars| {
                let pi_hash_pre =
                    RescueNativeGadget::<Fq254>::rescue_sponge_with_padding(circuit, pi_vars, 1)?
                        [0];

                let value = circuit.witness(pi_hash_pre)?;
                let bytes = value.into_bigint().to_bytes_le();
                let (challenge, leftover) = bytes.split_at(31);

                let pi_hash = circuit.create_variable(Fq254::from_le_bytes_mod_order(challenge))?;
                let leftover_var =
                    circuit.create_variable(Fq254::from_le_bytes_mod_order(leftover))?;

                circuit.enforce_in_range(pi_hash, 8 * 31)?;
                circuit.enforce_in_range(leftover_var, 6)?;

                let coeff = Fq254::from(2u32).pow([248u64]);

                circuit.lc_gate(
                    &[
                        pi_hash,
                        leftover_var,
                        circuit.zero(),
                        circuit.zero(),
                        pi_hash_pre,
                    ],
                    &[Fq254::one(), coeff, Fq254::zero(), Fq254::zero()],
                )?;

                Ok(pi_hash)
            })
            .collect::<Result<Vec<Variable>, CircuitError>>()?;

        // For checking correctness during testing
        #[cfg(test)]
        {
            for (circuit_hash, actual_hash) in pi_hashes.iter().zip(bn254info.bn254_outputs.iter())
            {
                assert_eq!(
                    circuit.witness(*circuit_hash).unwrap(),
                    fr_to_fq::<Fq254, BnConfig>(&actual_hash.pi_hash)
                );
            }
        }
        // Do any specific pi required
        let specific_pi_vars = specific_pi_fn(&impl_pi_vars, circuit)?;

        specific_pi_vars
            .iter()
            .try_for_each(|pi| circuit.set_variable_public(*pi))?;

        instance_scalar_vars
            .iter()
            .try_for_each(|var| circuit.set_variable_public(*var))?;

        proof_scalar_vars[..2]
            .iter()
            .try_for_each(|var| circuit.set_variable_public(*var))?;

        // Since this is the first round of recursion there will be no forwarded accumulators
        // so we skip straight to the output accumulator.

        acc_instance
            .get_coords()
            .iter()
            .try_for_each(|&var| circuit.set_variable_public(var))?;
        acc_proof
            .get_coords()
            .iter()
            .try_for_each(|&var| circuit.set_variable_public(var))?;

        pi_hashes
            .iter()
            .try_for_each(|x| circuit.set_variable_public(*x))?;

        let specific_pi = specific_pi_vars
            .iter()
            .map(|&pi| circuit.witness(pi))
            .collect::<Result<Vec<Fq254>, CircuitError>>()?;

        let acc_instance_point = circuit.point_witness(&acc_instance)?;
        let acc_proof_point = circuit.point_witness(&acc_proof)?;

        let acc_instance_real: Affine<BnConfig> = acc_instance_point.into();
        let acc_proof_real: Affine<BnConfig> = acc_proof_point.into();

        let kzg_proof = UnivariateKzgProof::<Bn254> {
            proof: acc_proof_real,
        };
        let output_accumulator =
            AtomicInstance::<Kzg>::new(acc_instance_real, Fr254::zero(), Fr254::zero(), kzg_proof);

        Ok(GrumpkinCircuitOutput::new(
            bn254info.bn254_outputs.clone(),
            specific_pi,
            bn254info.forwarded_acumulators.clone(),
            output_accumulator,
            bn254info.old_accumulators.clone(),
        ))
    }
}

type CombineScalars = (Vec<Fr254>, Vec<Affine<BnConfig>>);

/// This function takes in a slice of [`PcsInfo`] structs and combines the scalars to minimize the number of bases used in the MSM in the next circuit.
///
/// NOTE: Currently this function is very fragile and will break if any of the proving system is changed. In the future we should
/// aim to make this something that is read from the gate info or such.
fn combine_fft_proof_scalars(pcs_infos: &[PcsInfo<Kzg>], r_powers: &[Fr254]) -> CombineScalars {
    let mut scalars_list: Vec<Vec<Fr254>> = vec![];
    let mut comms_list = vec![];
    let mut opening_proofs = vec![];

    for pcs_info in pcs_infos.iter() {
        let mut real_scalars = pcs_info.comm_scalars_and_bases.scalars[..47].to_vec();

        real_scalars[10] += pcs_info.comm_scalars_and_bases.scalars[22];
        real_scalars[11] += pcs_info.comm_scalars_and_bases.scalars[48];
        real_scalars[12] += pcs_info.comm_scalars_and_bases.scalars[47];

        let _ = real_scalars.remove(22);
        let mut comms = pcs_info.comm_scalars_and_bases.bases()[..47].to_vec();
        let _ = comms.remove(22);

        scalars_list.push(real_scalars);
        comms_list.push(comms);
        opening_proofs.push(pcs_info.opening_proof.proof);
    }

    // Now we iterate through the lists and combine relevant scalars, we retain the selector, permutation scalars in the first list
    if scalars_list.is_empty() {
        (vec![], vec![])
    } else {
        // Unwrap is safe here because we have checked if the vec is empty
        let mut scalars = scalars_list.first().cloned().unwrap();
        // Multiply each list by the relevant power
        let scalars_list = scalars_list
            .iter()
            .zip(r_powers.iter())
            .skip(1)
            .map(|(list, r)| list.iter().map(|elem| *elem * *r).collect::<Vec<Fr254>>())
            .collect::<Vec<Vec<Fr254>>>();
        for list in scalars_list.iter() {
            scalars[4..10]
                .iter_mut()
                .zip(list[4..10].iter())
                .for_each(|(a, b)| *a += b);

            scalars[13..15]
                .iter_mut()
                .zip(list[13..15].iter())
                .for_each(|(a, b)| *a += b);

            scalars[21..40]
                .iter_mut()
                .zip(list[21..40].iter())
                .for_each(|(a, b)| *a += b);

            scalars[16] += list[16];
            scalars[19] += list[19];

            let appended_scalars = [
                &list[0..4],
                &list[10..13],
                &[list[15], list[17], list[18], list[20]],
                &list[40..],
            ]
            .concat();

            scalars.extend_from_slice(&appended_scalars);
        }
        scalars.push(pcs_infos[0].u);
        scalars.push(pcs_infos[1].u * r_powers[1]);
        // With the bases we remove the pi_bases and g_base now since this list will only be used to tell the circuit which variable bases are being used.
        let mut comms = comms_list.first().cloned().unwrap();

        for list in comms_list.iter().skip(1) {
            let appended_comms = [
                &list[0..4],
                &list[10..13],
                &[list[15], list[17], list[18], list[20]],
                &list[40..],
            ]
            .concat();

            comms.extend_from_slice(&appended_comms);
        }

        comms.extend(opening_proofs);
        (scalars, comms)
    }
}

/// This function takes in a slice of [`PcsInfo`] structs and combines the scalars for use in the first round of recursive proving where both proofs could have differing VKs.
///
/// NOTE: Currently this function is very fragile and will break if any of the proving system is changed. In the future we should
/// aim to make this something that is read from the gate info or such.
fn combine_fft_proof_scalars_round_one(
    pcs_infos: &[PcsInfo<Kzg>],
    r_powers: &[Fr254],
) -> CombineScalars {
    let mut scalars_list: Vec<Vec<Fr254>> = vec![];
    let mut comms_list = vec![];
    let mut opening_proofs = vec![];

    for pcs_info in pcs_infos.iter() {
        let mut real_scalars = pcs_info.comm_scalars_and_bases.scalars[..47].to_vec();

        real_scalars[10] += pcs_info.comm_scalars_and_bases.scalars[22];
        real_scalars[11] += pcs_info.comm_scalars_and_bases.scalars[48];
        real_scalars[12] += pcs_info.comm_scalars_and_bases.scalars[47];

        let _ = real_scalars.remove(22);
        let mut comms = pcs_info.comm_scalars_and_bases.bases()[..47].to_vec();
        let _ = comms.remove(22);

        scalars_list.push(real_scalars);
        comms_list.push(comms);
        opening_proofs.push(pcs_info.opening_proof.proof);
    }

    // Now we iterate through the lists and combine relevant scalars, we retain the selector, permutation scalars in the first list
    if scalars_list.is_empty() {
        (vec![], vec![])
    } else {
        // Unwrap is safe here because we have checked if the vec is empty
        let mut scalars = scalars_list.first().cloned().unwrap();
        // Multiply each list by the relevant power
        let scalars_list = scalars_list
            .iter()
            .zip(r_powers.iter())
            .skip(1)
            .map(|(list, r)| list.iter().map(|elem| *elem * *r).collect::<Vec<Fr254>>())
            .collect::<Vec<Vec<Fr254>>>();
        for list in scalars_list.iter() {
            scalars.extend_from_slice(list);
        }
        scalars.push(pcs_infos[0].u);
        scalars.push(pcs_infos[1].u * r_powers[1]);

        // With the bases we remove the pi_bases and g_base now since this list will only be used to tell the circuit which variable bases are being used.
        let mut comms = comms_list.first().cloned().unwrap();

        for list in comms_list.iter().skip(1) {
            comms.extend_from_slice(list);
        }

        comms.extend(opening_proofs);
        (scalars, comms)
    }
}

/// This function takes in two [`RecursiveOutput<Zeromorph<UnivariateIpaPCS<Grumpkin>>>, MLEPlonk<Zeromorph<UnivariateIpaPCS<Grumpkin>>>>, RescueTranscript<Fr254>>`] structs and prepares
/// the accumulator to be verified along with the scalars for the relevant MSMs to be performed in the next circuit.
///
/// In order to minimise the size of the MSMs that we have to compute in the next circuit we combine some of the scalars together where possible.
///
/// We recalculate the PI commitments separately and then add them on at the end, this is because we can use a fixed base MSM for these as
/// the SRS is fixed from the start. Now because the number of public inputs in both cases will be the same we can do both in one MSM.
///
/// Since all of the selector commitments are the same for both proofs we can combine all of their scalars together to eliminate another 21 bases from the MSM.
/// Using an almost identical argument we combine all the scalars for the permutation commitments, eliminating another 5 bases.
///
/// The generic parameter `IS_FIRST_ROUND` is used to determine if the first round of the recursion is being prepared. If it is the first round we do not
/// expect any accumulators to be passed in with the proofs.
///
/// For the accumulators we structure public inputs so that the forwarded accumulators are always the last two sets of public inputs.
/// Since atomic accumulators have fixed length (and instance point and a proof point) we can work out which of the public inputs relate to forwarded accumulators using this fact.
pub fn prove_grumpkin_accumulation<const IS_BASE: bool>(
    grumpkin_info: &GrumpkinRecursiveInfo,
    bn254_vks: &[VerifyingKey<Kzg>; 4],
    pk_grumpkin: &MLEProvingKey<ZeromorphPCS>,
    specific_pi_fn: impl Fn(
        &[Vec<Variable>],
        &mut PlonkCircuit<Fr254>,
    ) -> Result<Vec<Variable>, CircuitError>,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<Bn254CircuitOutput, PlonkError> {
    // Calculate the two sets of scalars used in the previous Grumpkin proofs
    let recursion_scalars = if !IS_BASE {
        izip!(
            grumpkin_info.bn254_outputs.chunks_exact(2),
            grumpkin_info.transcript_accumulators.chunks_exact(4)
        )
        .map(|(outputs, old_accs)| {
            calculate_recursion_scalars(outputs, old_accs, &bn254_vks[0], circuit)
        })
        .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?
    } else {
        izip!(
            grumpkin_info.bn254_outputs.chunks_exact(2),
            grumpkin_info.transcript_accumulators.chunks_exact(4),
            bn254_vks.chunks_exact(2),
        )
        .map(|(outputs, old_accs, vks)| {
            calculate_recursion_scalars_base(outputs, old_accs, vks, circuit)
        })
        .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?
    };
    // Make a vk variable
    let vk_var = MLEVerifyingKeyVar::new(circuit, &pk_grumpkin.verifying_key)?;

    // Now we make variables for the specific pi as we will have to use these in the following for loop and later on.
    let impl_specific_pi = grumpkin_info
        .specific_pi
        .iter()
        .map(|pi_vec| {
            pi_vec
                .iter()
                .map(|pi| circuit.create_variable(*pi))
                .collect::<Result<Vec<Variable>, CircuitError>>()
        })
        .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?;

    let mut bn254_acc_vars = Vec::<Variable>::new();
    let mut pi_hash_vars = Vec::<Variable>::new();

    // Now we reform the pi_hashes for both grumpkin proof and extract the scalars from them.
    let next_grumpkin_challenges: Vec<(MLEProofChallenges<Fq254>, RescueTranscriptVar<Fr254>)> =
        izip!(
            grumpkin_info.bn254_outputs.chunks_exact(2),
            grumpkin_info.grumpkin_outputs.iter(),
            impl_specific_pi.iter(),
            grumpkin_info.forwarded_acumulators.iter(),
            grumpkin_info.old_accumulators.chunks_exact(2),
            recursion_scalars.iter()
        )
        .map(
            |(
                bn254_outputs,
                output,
                impl_pi,
                bn254_accumulator,
                grumpkin_accumulators,
                recursion_scalars,
            )| {
                let recursion_scalars_prepped = recursion_scalars
                    .iter()
                    .map(|&var| convert_to_hash_form(circuit, var))
                    .collect::<Result<Vec<[Variable; 2]>, CircuitError>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<Variable>>();

                let isp_prepped = impl_pi
                    .iter()
                    .map(|&var| convert_to_hash_form(circuit, var))
                    .collect::<Result<Vec<[Variable; 2]>, CircuitError>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<Variable>>();

                let grumpkin_accs = if !IS_BASE {
                    grumpkin_accumulators
                        .iter()
                        .map(|acc| {
                            let comm_x = circuit.create_variable(acc.comm.x)?;
                            let comm_y = circuit.create_variable(acc.comm.y)?;
                            let concat_vec = [comm_x, comm_y];
                            let mut prepped_vec = concat_vec
                                .iter()
                                .map(|&var| convert_to_hash_form(circuit, var))
                                .collect::<Result<Vec<[Variable; 2]>, CircuitError>>()?
                                .into_iter()
                                .flatten()
                                .collect::<Vec<Variable>>();
                            let eval_bytes = acc.value.into_bigint().to_bytes_le();
                            let [low_eval, high_eval]: [Fr254; 2] =
                                bytes_to_field_elements::<_, Fr254>(eval_bytes)[1..]
                                    .try_into()
                                    .map_err(|_| {
                                        CircuitError::ParameterError(
                                            "Could not convert slice to fixed length array"
                                                .to_string(),
                                        )
                                    })?;
                            let low_var = circuit.create_variable(low_eval)?;
                            let high_var = circuit.create_variable(high_eval)?;
                            prepped_vec.push(low_var);
                            prepped_vec.push(high_var);
                            Result::<Vec<Variable>, CircuitError>::Ok(prepped_vec)
                        })
                        .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?
                        .into_iter()
                        .flatten()
                        .collect::<Vec<Variable>>()
                } else {
                    vec![]
                };

                let bn254_acc = [
                    bn254_accumulator.comm.x,
                    bn254_accumulator.comm.y,
                    bn254_accumulator.opening_proof.proof.x,
                    bn254_accumulator.opening_proof.proof.y,
                ]
                .iter()
                .map(|f| {
                    let eval_bytes = f.into_bigint().to_bytes_le();
                    let [low_eval, high_eval]: [Fr254; 2] =
                        bytes_to_field_elements::<_, Fr254>(eval_bytes)[1..]
                            .try_into()
                            .map_err(|_| {
                                CircuitError::ParameterError(
                                    "Could not convert slice to fixed length array".to_string(),
                                )
                            })?;
                    let low_var = circuit.create_variable(low_eval)?;
                    let high_var = circuit.create_variable(high_eval)?;
                    Result::<[Variable; 2], CircuitError>::Ok([low_var, high_var])
                })
                .collect::<Result<Vec<[Variable; 2]>, CircuitError>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<Variable>>();

                bn254_acc_vars.extend_from_slice(&bn254_acc);

                let bn254_pi_hashes = bn254_outputs
                    .iter()
                    .map(|bn| circuit.create_variable(bn.pi_hash))
                    .collect::<Result<Vec<Variable>, CircuitError>>()?;

                let bn_pi_hashes_prepped = bn254_pi_hashes
                    .iter()
                    .map(|&var| convert_to_hash_form(circuit, var))
                    .collect::<Result<Vec<[Variable; 2]>, CircuitError>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<Variable>>();

                let data_vars = [
                    isp_prepped,
                    recursion_scalars_prepped,
                    grumpkin_accs,
                    bn254_acc,
                    bn_pi_hashes_prepped,
                ];

                let calc_pi_hash = RescueNativeGadget::<Fr254>::rescue_sponge_with_padding(
                    circuit,
                    &data_vars.concat(),
                    1,
                )?[0];

                let value = circuit.witness(calc_pi_hash)?;
                let bytes = value.into_bigint().to_bytes_le();
                let (challenge, leftover) = bytes.split_at(31);

                let pi_hash = circuit.create_variable(Fr254::from_le_bytes_mod_order(challenge))?;

                let leftover_var =
                    circuit.create_variable(Fr254::from_le_bytes_mod_order(leftover))?;

                circuit.enforce_in_range(pi_hash, 8 * 31)?;
                circuit.enforce_in_range(leftover_var, 6)?;

                let coeff = Fr254::from(2u32).pow([248u64]);

                circuit.lc_gate(
                    &[
                        pi_hash,
                        leftover_var,
                        circuit.zero(),
                        circuit.zero(),
                        calc_pi_hash,
                    ],
                    &[Fr254::one(), coeff, Fr254::zero(), Fr254::zero()],
                )?;

                pi_hash_vars.push(pi_hash);

                // For checking correctness during testing
                #[cfg(test)]
                {
                    assert_eq!(
                        circuit.witness(pi_hash).unwrap(),
                        fr_to_fq::<Fr254, SWGrumpkin>(&output.pi_hash)
                    );
                }

                let next_grumpkin_challenges = reconstruct_mle_challenges::<
                    _,
                    _,
                    _,
                    _,
                    RescueTranscript<Fr254>,
                    RescueTranscriptVar<Fr254>,
                >(output, &vk_var, circuit)?;
                Ok(next_grumpkin_challenges)
            },
        )
        .collect::<Result<Vec<(MLEProofChallenges<Fq254>, RescueTranscriptVar<Fr254>)>, CircuitError>>()?;

    let mut transcript = next_grumpkin_challenges[0].1.clone();

    transcript.merge(&next_grumpkin_challenges[1].1)?;

    let deltas = next_grumpkin_challenges
        .iter()
        .map(|challenges| challenges.0.challenges.delta)
        .collect::<Vec<Fq254>>();
    let split_acc_info = SplitAccumulationInfo::perform_accumulation(
        &grumpkin_info.grumpkin_outputs,
        &grumpkin_info.old_accumulators,
        &pk_grumpkin.pcs_prover_params,
    )?;

    let (acc_comm, msm_scalars) = split_acc_info.verify_split_accumulation(
        &grumpkin_info.grumpkin_outputs,
        &grumpkin_info.old_accumulators,
        &deltas,
        &pk_grumpkin.verifying_key,
        &mut transcript,
        circuit,
    )?;

    let specific_pi = specific_pi_fn(&impl_specific_pi, circuit)?;

    // Make relevant variables public in the order:
    // specific_pi
    // msm scalars
    // forwarded accumulators
    // new accumulator
    // old_pi_hashes

    specific_pi
        .iter()
        .try_for_each(|pi| circuit.set_variable_public(*pi))?;

    // Convert the msm scalars into a form that uses fewer variables and set them public
    msm_scalars.iter().try_for_each(|scalar| {
        let transcript_form = circuit.convert_for_transcript(scalar)?;
        transcript_form
            .iter()
            .try_for_each(|var| circuit.set_variable_public(*var))
    })?;

    for var in bn254_acc_vars.iter() {
        circuit.set_variable_public(*var)?;
    }

    acc_comm
        .get_coords()
        .iter()
        .try_for_each(|coord| circuit.set_variable_public(*coord))?;

    let _ = bytes_to_field_elements::<_, Fr254>(
        split_acc_info
            .new_accumulator()
            .value
            .into_bigint()
            .to_bytes_le(),
    )[1..]
        .iter()
        .map(|p| circuit.create_public_variable(*p))
        .collect::<Result<Vec<Variable>, CircuitError>>()?;

    // Finally pi hashes are constructed to fit into either field
    for var in pi_hash_vars.iter() {
        circuit.set_variable_public(*var)?;
    }

    let specific_pi_field = specific_pi
        .into_iter()
        .map(|pi| circuit.witness(pi))
        .collect::<Result<Vec<Fr254>, CircuitError>>()?;

    let challenges: [MLEProofChallenges<Fq254>; 2] = next_grumpkin_challenges
        .into_iter()
        .map(|(challenges, _)| challenges)
        .collect::<Vec<MLEProofChallenges<Fq254>>>()
        .try_into()
        .map_err(|_| {
            CircuitError::ParameterError(
                "Could not create an array of length 2 of MLEProofChallenges".to_string(),
            )
        })?;
    Ok(Bn254CircuitOutput::new(
        specific_pi_field,
        grumpkin_info.forwarded_acumulators.clone(),
        split_acc_info.new_accumulator.clone(),
        challenges,
        split_acc_info,
        grumpkin_info.grumpkin_outputs.clone(),
    ))
}

/// This function builds the decider circuit.
pub fn decider_circuit(
    grumpkin_info: &GrumpkinRecursiveInfo,
    vk_bn254: &VerifyingKey<Kzg>,
    pk_grumpkin: &MLEProvingKey<Zmorph>,
    extra_data: &[Fr254],
    specific_pi_fn: impl Fn(
        &[Vec<Variable>],
        &mut PlonkCircuit<Fr254>,
    ) -> Result<Vec<Variable>, CircuitError>,
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<Vec<Fr254>, PlonkError> {
    // Calculate the two sets of scalars used in the previous Grumpkin proofs
    let recursion_scalars = izip!(
        grumpkin_info.bn254_outputs.chunks_exact(2),
        grumpkin_info.transcript_accumulators.chunks_exact(4)
    )
    .map(|(outputs, old_accs)| calculate_recursion_scalars(outputs, old_accs, vk_bn254, circuit))
    .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?;

    // Make a vk variable
    let vk_var = MLEVerifyingKeyVar::new(circuit, &pk_grumpkin.verifying_key)?;

    // Now we make variables for the specific pi as we will have to use these in the following for loop and later on.
    let impl_specific_pi = grumpkin_info
        .specific_pi
        .iter()
        .map(|pi_vec| {
            pi_vec
                .iter()
                .map(|pi| circuit.create_variable(*pi))
                .collect::<Result<Vec<Variable>, CircuitError>>()
        })
        .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?;

    // Now we reform the pi_hashes for both grumpkin proof and extract the scalars from them.
    izip!(
        grumpkin_info.bn254_outputs.chunks_exact(2),
        grumpkin_info.grumpkin_outputs.iter(),
        impl_specific_pi.iter(),
        grumpkin_info.forwarded_acumulators.iter(),
        grumpkin_info.old_accumulators.chunks_exact(2),
        recursion_scalars.iter()
    )
    .try_for_each(
        |(
            bn254_outputs,
            output,
            impl_pi,
            bn254_accumulator,
            grumpkin_accumulators,
            recursion_scalars,
        )| {
            let recursion_scalars_prepped = recursion_scalars
                .iter()
                .map(|&var| convert_to_hash_form(circuit, var))
                .collect::<Result<Vec<[Variable; 2]>, CircuitError>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<Variable>>();

            let isp_prepped = impl_pi
                .iter()
                .map(|&var| convert_to_hash_form(circuit, var))
                .collect::<Result<Vec<[Variable; 2]>, CircuitError>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<Variable>>();

            let grumpkin_accs = grumpkin_accumulators
                .iter()
                .map(|acc| {
                    let comm_x = circuit.create_variable(acc.comm.x)?;
                    let comm_y = circuit.create_variable(acc.comm.y)?;

                    let concat_vec = [comm_x, comm_y];
                    let mut prepped_vec = concat_vec
                        .iter()
                        .map(|&var| convert_to_hash_form(circuit, var))
                        .collect::<Result<Vec<[Variable; 2]>, CircuitError>>()?
                        .into_iter()
                        .flatten()
                        .collect::<Vec<Variable>>();
                    let eval_bytes = acc.value.into_bigint().to_bytes_le();
                    let [low_eval, high_eval]: [Fr254; 2] =
                        bytes_to_field_elements::<_, Fr254>(eval_bytes)[1..]
                            .try_into()
                            .map_err(|_| {
                                CircuitError::ParameterError(
                                    "Could not convert slice to fixed length array".to_string(),
                                )
                            })?;
                    let low_var = circuit.create_variable(low_eval)?;
                    let high_var = circuit.create_variable(high_eval)?;
                    prepped_vec.push(low_var);
                    prepped_vec.push(high_var);
                    Result::<Vec<Variable>, CircuitError>::Ok(prepped_vec)
                })
                .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<Variable>>();

            let bn254_acc = [
                bn254_accumulator.comm.x,
                bn254_accumulator.comm.y,
                bn254_accumulator.opening_proof.proof.x,
                bn254_accumulator.opening_proof.proof.y,
            ]
            .iter()
            .map(|f| {
                let eval_bytes = f.into_bigint().to_bytes_le();
                let [low_eval, high_eval]: [Fr254; 2] =
                    bytes_to_field_elements::<_, Fr254>(eval_bytes)[1..]
                        .try_into()
                        .map_err(|_| {
                            CircuitError::ParameterError(
                                "Could not convert slice to fixed length array".to_string(),
                            )
                        })?;
                let low_var = circuit.create_variable(low_eval)?;
                let high_var = circuit.create_variable(high_eval)?;
                Result::<[Variable; 2], CircuitError>::Ok([low_var, high_var])
            })
            .collect::<Result<Vec<[Variable; 2]>, CircuitError>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<Variable>>();

            let bn254_pi_hashes = bn254_outputs
                .iter()
                .map(|bn| circuit.create_variable(bn.pi_hash))
                .collect::<Result<Vec<Variable>, CircuitError>>()?;

            let bn_pi_hashes_prepped = bn254_pi_hashes
                .iter()
                .map(|&var| convert_to_hash_form(circuit, var))
                .collect::<Result<Vec<[Variable; 2]>, CircuitError>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<Variable>>();

            let data_vars = [
                isp_prepped,
                recursion_scalars_prepped,
                grumpkin_accs,
                bn254_acc,
                bn_pi_hashes_prepped,
            ]
            .concat();
            let calc_pi_hash =
                RescueNativeGadget::<Fr254>::rescue_sponge_with_padding(circuit, &data_vars, 1)?[0];

            let value = circuit.witness(calc_pi_hash)?;
            let bytes = value.into_bigint().to_bytes_le();
            let (challenge, leftover) = bytes.split_at(31);

            let pi_hash = circuit.create_variable(Fr254::from_le_bytes_mod_order(challenge))?;

            let leftover_var = circuit.create_variable(Fr254::from_le_bytes_mod_order(leftover))?;

            let coeff = Fr254::from(2u32).pow([248u64]);

            circuit.enforce_in_range(pi_hash, 8 * 31)?;
            circuit.enforce_in_range(leftover_var, 6)?;

            circuit.lc_gate(
                &[
                    pi_hash,
                    leftover_var,
                    circuit.zero(),
                    circuit.zero(),
                    calc_pi_hash,
                ],
                &[Fr254::one(), coeff, Fr254::zero(), Fr254::zero()],
            )?;

            let pi_hash_emul = circuit.create_emulated_variable(output.pi_hash)?;
            let pi_native = circuit.mod_to_native_field(&pi_hash_emul)?;

            circuit.enforce_equal(pi_native, pi_hash)
        },
    )?;
    let split_acc_info = SplitAccumulationInfo::perform_accumulation(
        &grumpkin_info.grumpkin_outputs,
        &grumpkin_info.old_accumulators,
        &pk_grumpkin.pcs_prover_params,
    )?;
    let (msm_scalars, acc_eval) = emulated_combine_mle_proof_scalars(
        &grumpkin_info.grumpkin_outputs,
        &split_acc_info,
        &pk_grumpkin.verifying_key,
        circuit,
    )?;

    // Create the variables for the commitments in the two proofs
    let proof_one =
        SAMLEProofVar::<Zmorph>::from_struct(circuit, &grumpkin_info.grumpkin_outputs[0].proof)?;
    let proof_two =
        SAMLEProofVar::<Zmorph>::from_struct(circuit, &grumpkin_info.grumpkin_outputs[1].proof)?;

    let acc_comms = grumpkin_info
        .old_accumulators
        .iter()
        .map(|acc| circuit.create_point_variable(&Point::<Fr254>::from(acc.comm)))
        .collect::<Result<Vec<PointVariable>, CircuitError>>()?;

    // We have already checked that lookup is supported so the following unwrap is safe.
    let lookup_vk = pk_grumpkin
        .verifying_key
        .lookup_verifying_key
        .as_ref()
        .unwrap();
    let range_table_comm =
        circuit.create_point_variable(&Point::<Fr254>::from(lookup_vk.range_table_comm))?;
    let key_table_comm =
        circuit.create_point_variable(&Point::<Fr254>::from(lookup_vk.key_table_comm))?;
    let table_dom_sep_comm =
        circuit.create_point_variable(&Point::<Fr254>::from(lookup_vk.table_dom_sep_comm))?;
    let q_dom_sep_comm =
        circuit.create_point_variable(&Point::<Fr254>::from(lookup_vk.q_dom_sep_comm))?;
    let q_lookup_comm =
        circuit.create_point_variable(&Point::<Fr254>::from(lookup_vk.q_lookup_comm))?;

    let lookup_bases = &[
        range_table_comm,
        key_table_comm,
        table_dom_sep_comm,
        q_dom_sep_comm,
        q_lookup_comm,
    ];

    let proof_msm_bases = [
        proof_one.wire_commitments_var.as_slice(),
        vk_var.selector_commitments_var.as_slice(),
        vk_var.permutation_commitments_var.as_slice(),
        &[proof_one.lookup_proof_var.as_ref().unwrap().m_poly_comm_var],
        lookup_bases,
        proof_two.wire_commitments_var.as_slice(),
        &[proof_two.lookup_proof_var.as_ref().unwrap().m_poly_comm_var],
        acc_comms.as_slice(),
    ]
    .concat();

    let accumulated_comm = EmulMultiScalarMultiplicationCircuit::<Fr254, SWGrumpkin>::msm(
        circuit,
        &proof_msm_bases,
        &msm_scalars,
    )?;

    let (opening_proof, _) = Zmorph::open(
        &pk_grumpkin.pcs_prover_params,
        &split_acc_info.new_accumulator().poly,
        &split_acc_info.new_accumulator().point,
    )?;

    let point = split_acc_info
        .new_accumulator()
        .point
        .iter()
        .map(|p| circuit.create_emulated_variable(*p))
        .collect::<Result<Vec<EmulatedVariable<Fq254>>, CircuitError>>()?;

    let impl_spec_pi = grumpkin_info
        .specific_pi
        .iter()
        .chain([extra_data.to_vec()].iter())
        .map(|pi_vec| {
            pi_vec
                .iter()
                .map(|pi| circuit.create_variable(*pi))
                .collect::<Result<Vec<Variable>, CircuitError>>()
        })
        .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?;
    let specific_pi = specific_pi_fn(&impl_spec_pi, circuit)?;

    verify_zeromorph_circuit(
        circuit,
        &pk_grumpkin.verifying_key.pcs_verifier_params,
        &accumulated_comm,
        &point,
        &acc_eval,
        opening_proof,
    )?;

    // Now to make on chain verification easier we perform a Keccak hash of the specific pi with forwarded accumulators and set it as the public input
    let field_pi_out = specific_pi
        .iter()
        .map(|pi| circuit.witness(*pi))
        .collect::<Result<Vec<Fr254>, CircuitError>>()?;
    let field_pi = field_pi_out
        .iter()
        .flat_map(|f| f.into_bigint().to_bytes_be())
        .collect::<Vec<u8>>();

    let acc_elems = grumpkin_info
        .forwarded_acumulators
        .iter()
        .flat_map(|acc| {
            let point = Point::<Fq254>::from(acc.comm);
            let opening_proof = Point::<Fq254>::from(acc.opening_proof.proof);
            point
                .coords()
                .iter()
                .chain(opening_proof.coords().iter())
                .flat_map(|coord| coord.into_bigint().to_bytes_be())
                .collect::<Vec<u8>>()
        })
        .collect::<Vec<u8>>();

    let mut hasher = Keccak256::new();
    hasher.update([field_pi, acc_elems].concat());
    let buf = hasher.finalize();

    // Generate challenge from state bytes using little-endian order
    let pi_hash = Fr254::from_be_bytes_mod_order(&buf);

    circuit.create_public_variable(pi_hash)?;
    Ok(field_pi_out)
}

/// Function for transforming a variable into the form it would be in when absorbed into a `RecursionHasher`
fn convert_to_hash_form(
    circuit: &mut PlonkCircuit<Fr254>,
    var: Variable,
) -> Result<[Variable; 2], CircuitError> {
    let f: Fr254 = circuit.witness(var)?;
    let bytes = f.into_bigint().to_bytes_le();
    let [low_elem, high_elem]: [Fr254; 2] = bytes_to_field_elements::<_, Fr254>(bytes)[1..]
        .try_into()
        .map_err(|_| {
            CircuitError::ParameterError(
                "Could not convert slice to fixed length array".to_string(),
            )
        })?;

    let coeff = Fr254::from(2u8).pow([248u64]);

    let low_var = circuit.create_variable(low_elem)?;
    let high_var = circuit.create_variable(high_elem)?;

    circuit.lc_gate(
        &[low_var, high_var, circuit.zero(), circuit.zero(), var],
        &[Fr254::one(), coeff, Fr254::zero(), Fr254::zero()],
    )?;

    Ok([low_var, high_var])
}

fn convert_to_hash_form_fq254(
    circuit: &mut PlonkCircuit<Fq254>,
    var: Variable,
) -> Result<[Variable; 2], CircuitError> {
    let f: Fq254 = circuit.witness(var)?;
    let bytes = f.into_bigint().to_bytes_le();
    let [low_elem, high_elem]: [Fq254; 2] = bytes_to_field_elements::<_, Fq254>(bytes)[1..]
        .try_into()
        .map_err(|_| {
            CircuitError::ParameterError(
                "Could not convert slice to fixed length array".to_string(),
            )
        })?;

    let coeff = Fq254::from(2u8).pow([248u64]);

    let low_var = circuit.create_variable(low_elem)?;
    let high_var = circuit.create_variable(high_elem)?;

    circuit.lc_gate(
        &[low_var, high_var, circuit.zero(), circuit.zero(), var],
        &[Fq254::one(), coeff, Fq254::zero(), Fq254::zero()],
    )?;

    Ok([low_var, high_var])
}
