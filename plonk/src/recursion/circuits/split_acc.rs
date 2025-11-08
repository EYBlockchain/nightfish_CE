//! Module containing code for verifying split accumulation using a cycle of curves.

use ark_bn254::{Fq as Fq254, Fr as Fr254};

use ark_poly::DenseMultilinearExtension;
use ark_std::{cfg_iter, iter, string::ToString, sync::Arc, vec, vec::Vec, One, Zero};

use jf_primitives::pcs::PolynomialCommitmentScheme;
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{EmulMultiScalarMultiplicationCircuit, Point, PointVariable},
        EmulatedVariable,
    },
    Circuit, PlonkCircuit, Variable,
};

use crate::{
    errors::PlonkError,
    nightfall::{
        accumulation::accumulation_structs::PCSWitness,
        circuit::{
            plonk_partial_verifier::SAMLEProofVar,
            subroutine_verifiers::{
                structs::{EmulatedSumCheckProofVar, SumCheckProofVar},
                sumcheck::SumCheckGadget,
            },
        },
        mle::{
            mle_structs::MLEVerifyingKey,
            subroutines::{sumcheck::SumCheck, PolyOracle, SumCheckProof, VPSumCheck},
            utils::{add_vecs, build_eq_x_r, scale_mle},
            virtual_polynomial::VirtualPolynomial,
            MLEPlonk,
        },
        UnivariateUniversalIpaParams,
    },
    proof_system::RecursiveOutput,
    transcript::{rescue::RescueTranscriptVar, CircuitTranscript, RescueTranscript, Transcript},
};
use nf_curves::grumpkin::{short_weierstrass::SWGrumpkin, Grumpkin};
use rayon::prelude::*;

use super::Zmorph;

/// We use this struct to store information about Split accumulation challenges.
pub struct SplitAccumulationInfoVar {
    /// The sumcheck proof used for the split accumulation
    pub sumcheck_proof: SumCheckProofVar<Fq254>,
    /// The challenge used to batch terms in the split accumulation
    pub batch_challenge: Variable,
    /// The old accumulators evaluation points used in the split accumulation
    pub old_accumulators_points: [Vec<Variable>; 4],
    /// The old accumulators evaluations used in the split accumulation
    pub old_accumulators_evals: [Variable; 4],
    /// The new accumulator produced by the split accumulation
    pub new_accumulator: PCSWitness<Zmorph>,
}

/// We use this struct to store information about Split accumulation challenges.
#[derive(Clone, Debug, Default)]
pub struct SplitAccumulationInfo {
    /// The sumcheck proof used for the split accumulation
    pub sumcheck_proof: SumCheckProof<Fq254, PolyOracle<Fq254>>,
    /// The challenge used to batch terms in the split accumulation
    pub batch_challenge: Fq254,
    /// The old accumulators used in the split accumulation
    pub old_accumulators: [PCSWitness<Zmorph>; 4],
    /// The new accumulator produced by the split accumulation
    pub new_accumulator: PCSWitness<Zmorph>,
}

impl SplitAccumulationInfo {
    /// Create a new instance of the struct from constituent parts.
    pub fn new(
        sumcheck_proof: SumCheckProof<Fq254, PolyOracle<Fq254>>,
        batch_challenge: Fq254,
        old_accumulators: [PCSWitness<Zmorph>; 4],
        new_accumulator: PCSWitness<Zmorph>,
    ) -> Self {
        Self {
            sumcheck_proof,
            batch_challenge,
            old_accumulators,
            new_accumulator,
        }
    }

    /// Converts the SplitAccumulationInfo into its variable representation in a PlonkCircuit.
    pub fn to_variables(
        &self,
        circuit: &mut PlonkCircuit<Fq254>,
    ) -> Result<SplitAccumulationInfoVar, CircuitError> {
        let old_accumulators_points: [Vec<Variable>; 4] = self
            .old_accumulators
            .iter()
            .map(|acc: &PCSWitness<Zmorph>| {
                acc.point
                    .iter()
                    .copied()
                    .map(|p| circuit.create_variable(p))
                    .collect::<Result<Vec<Variable>, CircuitError>>()
            })
            .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?
            .try_into()
            .map_err(|v: Vec<Vec<Variable>>| {
                CircuitError::ParameterError(ark_std::format!(
                    "expected 4 accumulators, got {}",
                    v.len()
                ))
            })?;

        let old_accumulators_evals: [Variable; 4] = self
            .old_accumulators
            .iter()
            .map(|acc: &PCSWitness<Zmorph>| circuit.create_variable(acc.value))
            .collect::<Result<Vec<Variable>, CircuitError>>()?
            .try_into()
            .unwrap();

        let batch_challenge = circuit.create_variable(self.batch_challenge)?;

        let sumcheck_proof = circuit.sum_check_proof_to_var(&self.sumcheck_proof)?;

        Ok(SplitAccumulationInfoVar {
            sumcheck_proof,
            batch_challenge,
            old_accumulators_points,
            old_accumulators_evals,
            new_accumulator: self.new_accumulator.clone(),
        })
    }

    /// Getter for the sumcheck proof
    pub fn sumcheck_proof(&self) -> &SumCheckProof<Fq254, PolyOracle<Fq254>> {
        &self.sumcheck_proof
    }

    /// Getter for the batch challenge
    pub fn batch_challenge(&self) -> Fq254 {
        self.batch_challenge
    }

    /// Getter for the old accumulators
    pub fn old_accumulators(&self) -> &[PCSWitness<Zmorph>; 4] {
        &self.old_accumulators
    }

    /// Getter for the new accumulator
    pub fn new_accumulator(&self) -> &PCSWitness<Zmorph> {
        &self.new_accumulator
    }

    /// Function to create a new instance of the struct from two [`RecursiveOutput`]s and 4 old accumulators.
    pub fn perform_accumulation(
        outputs: &[RecursiveOutput<Zmorph, MLEPlonk<Zmorph>, RescueTranscript<Fr254>>; 2],
        accumulators: &[PCSWitness<Zmorph>; 4],
        commit_key: &UnivariateUniversalIpaParams<Grumpkin>,
    ) -> Result<Self, PlonkError> {
        // First we are going to combine the two transcripts into one and squeeze out the batch challenge.
        let mut transcript = outputs[0].transcript.clone();
        transcript.merge(&outputs[1].transcript)?;

        let batching_challenge: Fq254 =
            transcript.squeeze_scalar_challenge::<SWGrumpkin>(b"batching challenge")?;
        let bc_sq = batching_challenge * batching_challenge;
        let bc_cube = bc_sq * batching_challenge;
        let bc_fourth = bc_sq * bc_sq;
        let bc_five = bc_fourth * batching_challenge;
        let witness_polys = outputs
            .iter()
            .map(|output| output.proof.polynomial.clone())
            .chain(accumulators.iter().map(|acc| acc.poly.clone()))
            .collect::<Vec<Arc<DenseMultilinearExtension<Fq254>>>>();
        let num_vars = witness_polys[0].num_vars;

        let opening_points = outputs
            .iter()
            .map(|output| &output.proof.opening_point)
            .chain(accumulators.iter().map(|acc| &acc.point))
            .collect::<Vec<&Vec<Fq254>>>();

        let eq_polys = cfg_iter!(opening_points)
            .map(|point| Arc::new(build_eq_x_r(point)))
            .collect::<Vec<_>>();

        let polys = [witness_polys.clone(), eq_polys].concat();

        let products = vec![
            (Fq254::one(), vec![0, 6]),
            (batching_challenge, vec![1, 7]),
            (bc_sq, vec![2, 8]),
            (bc_cube, vec![3, 9]),
            (bc_fourth, vec![4, 10]),
            (bc_five, vec![5, 11]),
        ];

        let vp = VirtualPolynomial::new(2, num_vars, polys, products);
        let sumcheck_proof =
            <VPSumCheck<SWGrumpkin> as SumCheck<SWGrumpkin>>::prove(&vp, &mut transcript)?;

        let scalars = [
            Fq254::one(),
            batching_challenge,
            bc_sq,
            bc_cube,
            bc_fourth,
            bc_five,
        ]
        .iter()
        .zip(sumcheck_proof.poly_evals[6..].iter())
        .map(|(s, e)| *s * e)
        .collect::<Vec<Fq254>>();

        let accumulated_poly_evals = cfg_iter!(scalars)
            .zip(cfg_iter!(witness_polys))
            .map(|(s, p)| scale_mle(p, *s))
            .try_fold(
                || vec![Fq254::zero(); 1 << num_vars],
                |acc, p| add_vecs(&acc, &p),
            )
            .try_reduce(
                || vec![Fq254::zero(); 1 << num_vars],
                |acc, p| add_vecs(&acc, &p),
            )?;
        let accumulated_poly = Arc::new(DenseMultilinearExtension::<Fq254>::from_evaluations_vec(
            num_vars,
            accumulated_poly_evals,
        ));

        let accumulated_value = scalars
            .iter()
            .zip(sumcheck_proof.poly_evals[..6].iter())
            .fold(Fq254::zero(), |acc, (s, e1)| acc + s * e1);

        let commitment = Zmorph::commit(commit_key, &accumulated_poly)?;
        let new_accumulator = PCSWitness::<Zmorph>::new(
            accumulated_poly,
            commitment,
            accumulated_value,
            sumcheck_proof.point.clone(),
        );
        Ok(Self::new(
            sumcheck_proof,
            batching_challenge,
            accumulators.clone(),
            new_accumulator,
        ))
    }

    /// Used to verify the accumulation performed in [`Self::perform_accumulation`] in a Bn254 circuit.
    #[allow(clippy::type_complexity)]
    pub fn verify_split_accumulation(
        &self,
        proof_vars: &[SAMLEProofVar<Zmorph>; 2],
        acc_point_vars: &[PointVariable; 4],
        deltas: &[EmulatedVariable<Fq254>],
        vk: &MLEVerifyingKey<Zmorph>,
        transcript: &mut RescueTranscriptVar<Fr254>,
        circuit: &mut PlonkCircuit<Fr254>,
    ) -> Result<
        (
            PointVariable,
            Vec<EmulatedVariable<Fq254>>,
            EmulatedVariable<Fq254>,
        ),
        CircuitError,
    > {
        // If the proofs don't support lookup we throw an error
        if vk.lookup_verifying_key.is_none() {
            return Err(CircuitError::NotSupported(
                "Only support UltraPlonk proofs".to_string(),
            ));
        }

        // Create the sumcheck proof variable
        let sumcheck_proof: EmulatedSumCheckProofVar<Fq254> =
            circuit.proof_to_emulated_var::<SWGrumpkin>(self.sumcheck_proof())?;

        let _ = transcript.squeeze_scalar_challenge::<SWGrumpkin>(circuit)?;

        // Verify the hashing in the SumCheck.
        let _ = circuit.recover_sumcheck_challenges::<SWGrumpkin, RescueTranscriptVar<Fr254>>(
            &sumcheck_proof,
            transcript,
        )?;

        // Calculate the coefficients for the accumulation.
        let batch_challenge: EmulatedVariable<Fq254> =
            circuit.create_emulated_variable(self.batch_challenge)?;
        let accumulation_coeffs = {
            let initial_var = circuit.emulated_one();
            iter::successors(Some(initial_var), |x| {
                circuit.emulated_mul(x, &batch_challenge).ok()
            })
            .take(6)
            .zip(self.sumcheck_proof().poly_evals[6..].iter())
            .map(|(x, y)| (x, *y))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(x, y)| {
                let y = circuit.create_emulated_variable(y).unwrap();
                circuit.emulated_mul(&x, &y)
            })
            .collect::<Result<Vec<EmulatedVariable<Fq254>>, CircuitError>>()?
        };

        // Create the variables for the commitments in the two proofs
        let proof_one = &proof_vars[0];
        let proof_two = &proof_vars[1];

        let proof_scalars = compute_mle_scalars(&accumulation_coeffs[..2], deltas, circuit)?;

        // Create the bases from the verifying key.
        let selector_variables = vk
            .selector_commitments
            .iter()
            .map(|comm| circuit.create_point_variable(&Point::<Fr254>::from(*comm)))
            .collect::<Result<Vec<PointVariable>, CircuitError>>()?;
        let permutation_variables = vk
            .permutation_commitments
            .iter()
            .map(|comm| circuit.create_point_variable(&Point::<Fr254>::from(*comm)))
            .collect::<Result<Vec<PointVariable>, CircuitError>>()?;

        // We have already checked that lookup is supported so the following unwrap is safe.
        let lookup_vk = vk.lookup_verifying_key.as_ref().unwrap();
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
            selector_variables.as_slice(),
            permutation_variables.as_slice(),
            &[proof_one.lookup_proof_var.as_ref().unwrap().m_poly_comm_var],
            lookup_bases,
            proof_two.wire_commitments_var.as_slice(),
            &[proof_two.lookup_proof_var.as_ref().unwrap().m_poly_comm_var],
        ]
        .concat();

        let final_msm_scalars = proof_scalars
            .iter()
            .chain(accumulation_coeffs[2..].iter())
            .cloned()
            .collect::<Vec<_>>();

        let final_bases = [proof_msm_bases, acc_point_vars.into()].concat();

        let acc_com = EmulMultiScalarMultiplicationCircuit::<Fr254, SWGrumpkin>::msm(
            circuit,
            &final_bases,
            &final_msm_scalars,
        )?;

        // For sanity check during testing
        #[cfg(test)]
        {
            let point_witness = circuit.point_witness(&acc_com)?;

            assert_eq!(self.new_accumulator.comm.x, point_witness.get_x());
            assert_eq!(self.new_accumulator.comm.y, point_witness.get_y());
        }
        Ok((acc_com, final_msm_scalars, batch_challenge))
    }
}

/// This function computes the scalars used for the MSM that is performed in a Bn254 recursion circuit.
/// We work on the assumption that the proof uses lookups because if it isn't the circuit before was probably too big to prove anyway.
fn compute_mle_scalars(
    accumulation_scalars: &[EmulatedVariable<Fq254>],
    deltas: &[EmulatedVariable<Fq254>],
    circuit: &mut PlonkCircuit<Fr254>,
) -> Result<Vec<EmulatedVariable<Fq254>>, CircuitError> {
    // The individual proof scalars are structured in the order:
    // wire commits (6) [0..6]
    // selector commitments (17) [6..23]
    // permutation commitments (6) [23..29]
    // multiplicity poly commitment (1) [29]
    // lookup verifying key commitments (5) [30..35]

    if (accumulation_scalars.len() != 2) & (deltas.len() != 2) {
        return Err(CircuitError::InternalError(
            "Should only have 2 accumulation scalars and deltas".to_string(),
        ));
    }
    let scalars = accumulation_scalars
        .iter()
        .zip(deltas.iter())
        .map(|(acc_scalar, delta)| {
            (0..35).try_fold(Vec::with_capacity(35), |mut result, i| {
                let current = if i == 0 {
                    acc_scalar.clone()
                } else {
                    circuit.emulated_mul(delta, result.last().unwrap())?
                };
                result.push(current);
                Ok(result)
            })
        })
        .collect::<Result<Vec<Vec<EmulatedVariable<Fq254>>>, CircuitError>>()?;

    let mut out_scalars = scalars[0].clone();

    out_scalars[6..29]
        .iter_mut()
        .zip(scalars[1][6..29].iter())
        .try_for_each(|(out, s)| {
            *out = circuit.emulated_add(out, s)?;
            Ok::<(), CircuitError>(())
        })?;
    out_scalars[30..35]
        .iter_mut()
        .zip(scalars[1][30..35].iter())
        .try_for_each(|(out, s)| {
            *out = circuit.emulated_add(out, s)?;
            Ok::<(), CircuitError>(())
        })?;

    out_scalars.extend_from_slice(&scalars[1][0..6]);
    out_scalars.push(scalars[1][29].clone());
    Ok(out_scalars)
}
