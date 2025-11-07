//! This module contains the code for producing all the challenge squeezed from a transcript during a MLE proof.
//! We need this so that we don't have to do any hashing inside Grumpkin circuits, instead we can do it all inside the Bn254 circuit and
//! then pass the challenges across via public inputs.

use ark_ec::short_weierstrass::Affine;
use ark_ff::{BigInteger, PrimeField};
use ark_std::{string::ToString, vec::Vec};
use jf_primitives::{pcs::Accumulation, rescue::RescueParameter};
use jf_relation::{
    errors::CircuitError,
    gadgets::{ecc::HasTEForm, EmulatedVariable, EmulationConfig},
    Circuit, PlonkCircuit, Variable,
};

use crate::{
    errors::PlonkError,
    nightfall::{
        circuit::{
            plonk_partial_verifier::{MLEChallengesVar, SAMLEProofVar},
            subroutine_verifiers::sumcheck::SumCheckGadget,
        },
        mle::mle_structs::{MLEChallenges, SAMLEProof},
    },
    recursion::UniversalRecursiveSNARK,
    transcript::{CircuitTranscript, Transcript},
};

/// This struct contains all the challenges for the various protocols used during an [`MLEPlonk`] proof.
#[derive(Clone, Debug, Default)]
pub struct MLEProofChallenges<F: PrimeField> {
    /// The plonk challenges themselves.
    pub challenges: MLEChallenges<F>,
    /// The r challenges used in the GKR proof for combining claims about p0 and p1 or q0 and q1.
    pub gkr_r_challenges: Vec<F>,
    /// The lambda challenges used in the GKR proof for separating the claims about p0 and p1 or q0 and q1.
    pub gkr_lambda_challenges: Vec<F>,
    /// The challenges generated during each SumCheck proof in the GKR proof.
    /// Each of these vectors gets progressively longer until the final one has length equal to log(n) where n is the the total number of gates in the circuit
    /// that was proved.
    pub gkr_sumcheck_challenges: Vec<Vec<F>>,
    /// The challenges generated in the final SumCheck proof.
    pub final_sumcheck_challenges: Vec<F>,
}

// Challenges are designed to fit into either field without performing a modular reduction so we implement this conversion.
impl<Fq: PrimeField, Fr: PrimeField> From<&MLEProofChallenges<Fr>> for MLEProofChallenges<Fq> {
    fn from(challenges: &MLEProofChallenges<Fr>) -> Self {
        let gkr_r_challenges = challenges
            .gkr_r_challenges
            .iter()
            .map(|c| Fq::from_le_bytes_mod_order(&c.into_bigint().to_bytes_le()))
            .collect();
        let gkr_lambda_challenges = challenges
            .gkr_lambda_challenges
            .iter()
            .map(|c| Fq::from_le_bytes_mod_order(&c.into_bigint().to_bytes_le()))
            .collect();
        let gkr_sumcheck_challenges = challenges
            .gkr_sumcheck_challenges
            .iter()
            .map(|v| {
                v.iter()
                    .map(|c| Fq::from_le_bytes_mod_order(&c.into_bigint().to_bytes_le()))
                    .collect()
            })
            .collect();
        let final_sumcheck_challenges = challenges
            .final_sumcheck_challenges
            .iter()
            .map(|c| Fq::from_le_bytes_mod_order(&c.into_bigint().to_bytes_le()))
            .collect();

        let mle_challenges: MLEChallenges<Fq> = (&challenges.challenges).into();
        Self {
            challenges: mle_challenges,
            gkr_r_challenges,
            gkr_lambda_challenges,
            gkr_sumcheck_challenges,
            final_sumcheck_challenges,
        }
    }
}

impl<F: PrimeField> MLEProofChallenges<F> {
    /// Create a new [`MLEProofChallenges`] struct from the given challenges.
    pub fn new(
        challenges: MLEChallenges<F>,
        gkr_r_challenges: Vec<F>,
        gkr_lambda_challenges: Vec<F>,
        gkr_sumcheck_challenges: Vec<Vec<F>>,
        final_sumcheck_challenges: Vec<F>,
    ) -> Self {
        Self {
            challenges,
            gkr_r_challenges,
            gkr_lambda_challenges,
            gkr_sumcheck_challenges,
            final_sumcheck_challenges,
        }
    }

    /// Getter for the challenges.
    pub fn challenges(&self) -> &MLEChallenges<F> {
        &self.challenges
    }
    /// Getter for the r challenges.
    pub fn gkr_r_challenges(&self) -> &[F] {
        &self.gkr_r_challenges
    }
    /// Getter for the lambda challenges.
    pub fn gkr_lambda_challenges(&self) -> &[F] {
        &self.gkr_lambda_challenges
    }
    /// Getter for the GKR sumcheck challenges.
    pub fn gkr_sumcheck_challenges(&self) -> &[Vec<F>] {
        &self.gkr_sumcheck_challenges
    }
    /// Getter for the final sumcheck challenges.
    pub fn final_sumcheck_challenges(&self) -> &[F] {
        &self.final_sumcheck_challenges
    }
}

/// This function takes in a [`RecursiveOutput`] and returns the challenges that were squeezed from the transcript during the proof.
/// It is assumed that this circuit is defined over the same field that commitments are. By returning the challenges we mean that it sets the associate variables to be public.
pub fn reconstruct_mle_challenges<P, F, PCS, Scheme, T, C>(
    proof_var: &SAMLEProofVar<PCS>,
    circuit: &mut PlonkCircuit<F>,
    pi_hash: &EmulatedVariable<P::ScalarField>,
    initialisation_msg: &Option<Vec<u8>>,
) -> Result<(MLEProofChallenges<P::ScalarField>, C), CircuitError>
where
    PCS: Accumulation<
        Commitment = Affine<P>,
        Evaluation = P::ScalarField,
        Point = Vec<P::ScalarField>,
    >,
    Scheme: UniversalRecursiveSNARK<PCS, RecursiveProof = SAMLEProof<PCS>>,
    P: HasTEForm<BaseField = F>,
    P::ScalarField: PrimeField + RescueParameter + EmulationConfig<F>,
    P::BaseField: PrimeField,
    F: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
    T: Transcript + ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize,
    C: CircuitTranscript<F>,
{
    // First lets instantiate the transcript and make the variable version of the proof and pi_commitment.
    let mut transcript: C = if let Some(msg) = initialisation_msg {
        C::new_with_initial_message::<_, P>(msg, circuit)?
    } else {
        C::new_transcript(circuit)
    };

    // Now we begin by recovering the circuit version of the MLEChallenges struct.
    let mle_challenges =
        MLEChallengesVar::compute_challenges(circuit, pi_hash, proof_var, &mut transcript)?;

    let mle_challenges_field = mle_challenges.to_field(circuit)?;
    // Next we need to know the number of variables the polynomials used in the proof had, this is the same as `proof_var.opening_point_var.len()`.
    let num_vars = proof_var.opening_point_var.len();

    // We extract the flattened version of all the GKR challenges and sort them into their respective vectors.
    let flat_gkr_challenges = proof_var
        .gkr_proof
        .extract_challenges::<P, F, C>(circuit, &mut transcript)?;

    let gkr_r_challenges = flat_gkr_challenges[0..num_vars + 1].to_vec();

    let gkr_lambda_challenges = flat_gkr_challenges[num_vars + 1..2 * num_vars + 1].to_vec();

    let mut gkr_sumcheck_challenges = Vec::new();
    let mut gkr_sumcheck_challenges_field = Vec::new();
    let mut j = 0usize;
    for i in 1..=num_vars {
        let round_i_sumcheck_challenges: Vec<usize> = flat_gkr_challenges[2 * num_vars + 1..]
            .iter()
            .skip(j)
            .take(i)
            .copied()
            .collect();
        j += i;

        let round_i_sumcheck_challenges_field = round_i_sumcheck_challenges
            .iter()
            .map(|c| circuit.witness(*c))
            .collect::<Result<Vec<_>, _>>()?;
        gkr_sumcheck_challenges.push(round_i_sumcheck_challenges);
        gkr_sumcheck_challenges_field.push(round_i_sumcheck_challenges_field);
    }

    // Recover the final SumCheck challenges.
    let final_sumcheck_challenges =
        circuit.recover_sumcheck_challenges::<P, C>(&proof_var.sumcheck_proof, &mut transcript)?;

    // Convert everything to field elements so it can be passed across the boundary.
    let gkr_r_challenges_field = gkr_r_challenges
        .iter()
        .map(|c| circuit.witness(*c))
        .collect::<Result<Vec<_>, _>>()?;

    let gkr_lambda_challenges_field = gkr_lambda_challenges
        .iter()
        .map(|c| circuit.witness(*c))
        .collect::<Result<Vec<_>, _>>()?;

    let final_sumcheck_challenges_field = final_sumcheck_challenges
        .iter()
        .map(|c| circuit.witness(*c))
        .collect::<Result<Vec<_>, _>>()?;

    let mle_proof_challenges: MLEProofChallenges<P::ScalarField> =
        (&MLEProofChallenges::<P::BaseField>::new(
            mle_challenges_field,
            gkr_r_challenges_field,
            gkr_lambda_challenges_field,
            gkr_sumcheck_challenges_field,
            final_sumcheck_challenges_field,
        ))
            .into();

    Ok((mle_proof_challenges, transcript))
}

// We wish to be able to recover the challenges from a vector of field elements.
// If we denote by n the number of variables in the multilinear extensions used in the circuit then the total number of challenges is given by:
//          7 + (n + 1) + n + n(n + 1)/2 + n = n^2 + 3n + 7.
// So we can recover the challenges without knowing the number of variables in the circuit.
impl<F> TryFrom<&[F]> for MLEProofChallenges<F>
where
    F: PrimeField,
{
    type Error = PlonkError;
    fn try_from(challenges: &[F]) -> Result<Self, Self::Error> {
        let discriminant = 9 + 28 * challenges.len() as i64;
        let disc_sqrt = (discriminant as f64).sqrt() as i64;

        if disc_sqrt * disc_sqrt != discriminant {
            return Err(PlonkError::InvalidParameters("The length of challenges provided does not give an integer solution to the quadratic equation".to_string()));
        }

        let num_vars = (disc_sqrt as usize - 3) / 2;

        let gkr_r_challenges = challenges[7..num_vars + 8].to_vec();
        let gkr_lambda_challenges = challenges[num_vars + 8..2 * num_vars + 8].to_vec();
        let mut gkr_sumcheck_challenges = Vec::new();
        let mut j = 0usize;
        for i in 1..=num_vars {
            let round_i_sumcheck_challenges: Vec<F> = challenges[2 * num_vars + 8..]
                .iter()
                .skip(j)
                .take(i)
                .copied()
                .collect();
            j += i;
            gkr_sumcheck_challenges.push(round_i_sumcheck_challenges);
        }
        // 2 * num_vars + 8 + num_vars * (num_vars + 1)/2 = num_vars * (num_vars + 5)/2 + 8
        let total_gkr_challenges = num_vars * (num_vars + 5) / 2 + 8;
        let final_sumcheck_challenges = challenges[total_gkr_challenges..].to_vec();

        // Check that we have the correct length here.
        if final_sumcheck_challenges.len() != num_vars {
            return Err(PlonkError::InvalidParameters("The number of final sumcheck challenges is not equal to the number of variables in the circuit".to_string()));
        }

        let mle_challenges = MLEChallenges::try_from(challenges[0..7].to_vec())?;
        Ok(Self {
            challenges: mle_challenges,
            gkr_r_challenges,
            gkr_lambda_challenges,
            gkr_sumcheck_challenges,
            final_sumcheck_challenges,
        })
    }
}

/// Struct for converting everything in the [`MLEProofChallenges`] struct to [`Variable`]'s.
#[derive(Clone, Debug)]
pub struct MLEProofChallengesVar {
    /// The plonk challenges themselves.
    pub challenges: MLEChallengesVar,
    /// The r challenges used in the GKR proof for combining claims about p0 and p1 or q0 and q1.
    pub gkr_r_challenges: Vec<Variable>,
    /// The lambda challenges used in the GKR proof for separating the claims about p0 and p1 or q0 and q1.
    pub gkr_lambda_challenges: Vec<Variable>,
    /// The challenges generated during each SumCheck proof in the GKR proof.
    /// Each of these vectors gets progressively longer until the final one has length equal to log(n) where n is the the total number of gates in the circuit
    /// that was proved.
    pub gkr_sumcheck_challenges: Vec<Vec<Variable>>,
    /// The challenges generated in the final SumCheck proof.
    pub final_sumcheck_challenges: Vec<Variable>,
}

impl MLEProofChallengesVar {
    /// Create a new instance of the struct from the given challenges.
    pub fn new(
        challenges: MLEChallengesVar,
        gkr_r_challenges: Vec<Variable>,
        gkr_lambda_challenges: Vec<Variable>,
        gkr_sumcheck_challenges: Vec<Vec<Variable>>,
        final_sumcheck_challenges: Vec<Variable>,
    ) -> Self {
        Self {
            challenges,
            gkr_r_challenges,
            gkr_lambda_challenges,
            gkr_sumcheck_challenges,
            final_sumcheck_challenges,
        }
    }

    /// Create an instance of the struct from the given [`MLEProofChallenges`] struct.
    pub fn from_struct<F>(
        circuit: &mut PlonkCircuit<F>,
        challenges: &MLEProofChallenges<F>,
    ) -> Result<Self, CircuitError>
    where
        F: PrimeField,
    {
        let mle_challenges = MLEChallengesVar::from_struct(circuit, &challenges.challenges)?;

        let gkr_r_challenges = challenges
            .gkr_r_challenges()
            .iter()
            .map(|c| circuit.create_variable(*c))
            .collect::<Result<Vec<Variable>, _>>()?;

        let gkr_lambda_challenges = challenges
            .gkr_lambda_challenges()
            .iter()
            .map(|c| circuit.create_variable(*c))
            .collect::<Result<Vec<Variable>, _>>()?;

        let gkr_sumcheck_challenges = challenges
            .gkr_sumcheck_challenges()
            .iter()
            .map(|v| {
                v.iter()
                    .map(|c| circuit.create_variable(*c))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<Vec<Variable>>, _>>()?;

        let final_sumcheck_challenges = challenges
            .final_sumcheck_challenges()
            .iter()
            .map(|c| circuit.create_variable(*c))
            .collect::<Result<Vec<Variable>, _>>()?;

        Ok(Self::new(
            mle_challenges,
            gkr_r_challenges,
            gkr_lambda_challenges,
            gkr_sumcheck_challenges,
            final_sumcheck_challenges,
        ))
    }

    /// Getter for the challenges.
    pub fn challenges(&self) -> &MLEChallengesVar {
        &self.challenges
    }
    /// Getter for the r challenges.
    pub fn gkr_r_challenges(&self) -> &[Variable] {
        &self.gkr_r_challenges
    }
    /// Getter for the lambda challenges.
    pub fn gkr_lambda_challenges(&self) -> &[Variable] {
        &self.gkr_lambda_challenges
    }
    /// Getter for the GKR sumcheck challenges.
    pub fn gkr_sumcheck_challenges(&self) -> &[Vec<Variable>] {
        &self.gkr_sumcheck_challenges
    }
    /// Getter for the final sumcheck challenges.
    pub fn final_sumcheck_challenges(&self) -> &[Variable] {
        &self.final_sumcheck_challenges
    }
}
