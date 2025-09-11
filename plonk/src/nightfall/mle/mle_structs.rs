//! This module contains structs used for proving and verifying Plonk proofs constructed
//! using multilinear extensions.

use ark_ec::{short_weierstrass::Affine, AffineRepr};
use ark_poly::DenseMultilinearExtension;

use crate::{
    errors::PlonkError,
    nightfall::ipa_structs::{VerificationKeyId, VK},
    transcript::{Transcript, TranscriptVisitor},
};
use ark_ff::{BigInteger, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};
use ark_std::{
    fmt::{Display, Formatter, Result as FmtResult},
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
    write,
};
use jf_primitives::{
    pcs::{prelude::PCSError, Accumulation, PolynomialCommitmentScheme, StructuredReferenceString},
    rescue::RescueParameter,
};

use jf_relation::gadgets::{ecc::HasTEForm, EmulationConfig};
use jf_utils::to_bytes;
use sha3::{Digest, Keccak256};

use super::subroutines::{gkr::GKRProof, PolyOracle, SumCheckProof};

/// Used to encase errors when working with virtual polynomials.
#[derive(Debug)]
pub enum PolynomialError {
    /// Used if some parameter is not valid
    ParameterError(String),
    /// Returned if we reach a point in the code that should be unreachable
    UnreachableError,
    /// Returned if we encounter an error when working with the polynomial commitment scheme
    PCSErrors(PCSError),
    /// Returned if we encounter an error when serializing or deserializing
    SerializationErrors(SerializationError),
}

impl From<PCSError> for PolynomialError {
    fn from(e: PCSError) -> Self {
        Self::PCSErrors(e)
    }
}
impl From<SerializationError> for PolynomialError {
    fn from(e: SerializationError) -> Self {
        Self::SerializationErrors(e)
    }
}

impl Display for PolynomialError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            PolynomialError::ParameterError(s) => write!(f, "ParameterError: {}", s),
            PolynomialError::UnreachableError => write!(f, "UnreachableError"),
            PolynomialError::PCSErrors(e) => write!(f, "PCSErrors: {}", e),
            PolynomialError::SerializationErrors(e) => {
                write!(f, "SerializationErrors: {}", e)
            },
        }
    }
}

/// This struct contains information about the gate equation.
/// For instance if you had a gate that `q_1 * w_1 + q_2 * w_2`
/// then `max_degree` would be 2 and `products` would be `[[2,0], [3,1]]`.
/// For clarity we order the variables as `[w_1, w_2, q_1, q_2]`.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct GateInfo<F: PrimeField> {
    /// The maximum degree of any monomial in the mle's.
    pub max_degree: usize,
    /// List of products whose sum is the entire gate equation.
    pub products: Vec<(F, Vec<usize>)>,
}
/// Proving Key used for MLE Plonk proofs.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct MLEProvingKey<PCS: PolynomialCommitmentScheme> {
    /// The mle's used for selector oracles.
    pub selector_oracles: Vec<Arc<DenseMultilinearExtension<PCS::Evaluation>>>,
    /// The mle's used for permutation oracles.
    pub permutation_oracles: Vec<Arc<DenseMultilinearExtension<PCS::Evaluation>>>,
    /// The verifying key.
    pub verifying_key: MLEVerifyingKey<PCS>,
    /// The optional lookup proving key.
    pub lookup_proving_key: Option<MLELookupProvingKey<PCS>>,
    /// The PCS prover parameters.
    pub pcs_prover_params: <PCS::SRS as StructuredReferenceString>::ProverParam,
}

/// Verifying Key used for MLE Plonk proofs.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct MLEVerifyingKey<PCS: PolynomialCommitmentScheme> {
    /// Commitments to the selectors.
    pub selector_commitments: Vec<PCS::Commitment>,
    /// Commitments to the permutations.
    pub permutation_commitments: Vec<PCS::Commitment>,
    /// The optional lookup verifying key.
    pub lookup_verifying_key: Option<MLELookupVerifyingKey<PCS>>,
    /// The PCS verifier parameters.
    pub pcs_verifier_params: <PCS::SRS as StructuredReferenceString>::VerifierParam,
    /// The information that describes the gate equation.
    pub gate_info: GateInfo<PCS::Evaluation>,
    /// The number of public inputs to the proof.
    pub num_inputs: usize,
}

impl<PCS> VK<PCS> for MLEVerifyingKey<PCS>
where
    PCS: PolynomialCommitmentScheme,
    PCS::Commitment: AffineRepr,
    <PCS::Commitment as AffineRepr>::Config: HasTEForm,
    <PCS::Commitment as AffineRepr>::BaseField: PrimeField,
{
    fn domain_size(&self) -> usize {
        2
    }

    fn selector_comms(&self) -> &[<PCS as PolynomialCommitmentScheme>::Commitment] {
        &self.selector_commitments
    }

    fn sigma_comms(&self) -> &[<PCS as PolynomialCommitmentScheme>::Commitment] {
        &self.permutation_commitments
    }

    fn hash(&self) -> <PCS as PolynomialCommitmentScheme>::Evaluation {
        let mut hasher = Keccak256::new();

        let mut bytes = Vec::new();
        for com in self.sigma_comms().iter() {
            bytes.extend_from_slice(&to_bytes!(com).unwrap());
        }

        for com in self.selector_comms().iter() {
            bytes.extend_from_slice(&to_bytes!(com).unwrap());
        }

        if let Some(plookup_vk) = self.lookup_verifying_key.as_ref() {
            bytes.extend_from_slice(&to_bytes!(&plookup_vk.range_table_comm).unwrap());
            bytes.extend_from_slice(&to_bytes!(&plookup_vk.key_table_comm).unwrap());
            bytes.extend_from_slice(&to_bytes!(&plookup_vk.table_dom_sep_comm).unwrap());
            bytes.extend_from_slice(&to_bytes!(&plookup_vk.q_dom_sep_comm).unwrap());
        }
        // Find which of the two fields on the associated curve is smaller and make sure the
        // end result fits into both fields.
        let scalar_bit_size = <PCS::Commitment as AffineRepr>::ScalarField::MODULUS_BIT_SIZE;
        let scalar_bytes = (scalar_bit_size - 1) / 8;
        let base_bit_size = <PCS::Commitment as AffineRepr>::BaseField::MODULUS_BIT_SIZE;
        let base_bytes = (base_bit_size - 1) / 8;

        let bytes_to_take = ark_std::cmp::min(scalar_bytes, base_bytes);
        let bytes_to_take = ark_std::cmp::min(bytes_to_take, 32u32);

        hasher.update(&bytes);
        let buf = hasher.finalize();
        PCS::Evaluation::from_le_bytes_mod_order(&buf[..bytes_to_take as usize])
    }

    fn id(&self) -> Option<VerificationKeyId> {
        None
    }

    fn is_merged(&self) -> bool {
        false
    }

    fn k(&self) -> &[<PCS as PolynomialCommitmentScheme>::Evaluation] {
        &[]
    }

    fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    fn plookup_vk(&self) -> Option<&crate::nightfall::ipa_structs::PlookupVerifyingKey<PCS>> {
        None
    }
}

impl<PCS> TranscriptVisitor for MLEVerifyingKey<PCS>
where
    PCS: PolynomialCommitmentScheme,
    PCS::Commitment: AffineRepr,
    <PCS::Commitment as AffineRepr>::Config: HasTEForm,
    <PCS::Commitment as AffineRepr>::BaseField: PrimeField,
{
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) -> Result<(), PlonkError> {
        transcript.push_message(b"vk", &self.hash())?;
        Ok(())
    }
}
/// The proof returned by a [`MLEPlonk`] struct.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct MLEProof<PCS: PolynomialCommitmentScheme> {
    /// Commitments to the witness wires.
    pub wire_commitments: Vec<PCS::Commitment>,
    /// GKR Proof.
    pub gkr_proof: GKRProof<PCS::Evaluation>,
    /// SumCheck proof.
    pub sumcheck_proof: SumCheckProof<PCS::Evaluation, PolyOracle<PCS::Evaluation>>,
    /// Optional lookup proof.
    pub lookup_proof: Option<MLELookupProof<PCS>>,
    /// Claimed evaluations of the witness wires, selectors and permutation related polynomials.
    pub evals: MLEProofEvals<PCS>,
    /// Opening proof,
    pub opening_proof: PCS::Proof,
}

/// A proof used when we are producing an `MLEPlonk` proof for split accumulation.
/// This proof does not contain a batch opening and instead just stores the accumulator struct allowing us to avoid
/// running the batch opening protocol until we have accumulated all the commitments.
///
/// This should only be used in the context of recursive proving where we do not care about zero knowledge as
/// it requires us to pass around the witness polynomials.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct SAMLEProof<PCS: Accumulation> {
    /// Commitments to the witness wires.
    pub wire_commitments: Vec<PCS::Commitment>,
    /// GKR Proof.
    pub gkr_proof: GKRProof<PCS::Evaluation>,
    /// SumCheck proof.
    pub sumcheck_proof: SumCheckProof<PCS::Evaluation, PolyOracle<PCS::Evaluation>>,
    /// Optional lookup proof.
    pub lookup_proof: Option<MLELookupProof<PCS>>,
    /// Claimed evaluations of the witness wires, selectors and permutation related polynomials.
    pub evals: MLEProofEvals<PCS>,
    /// The point the polynomial is to be opened at.
    pub opening_point: PCS::Point,
    /// The combined polynomial witness.
    pub polynomial: PCS::Polynomial,
}

impl<PCS: Accumulation> SAMLEProof<PCS> {
    /// Converts this proof into a regular MLEProof.
    pub fn into_mle_proof(
        self,
        ck: &<PCS::SRS as StructuredReferenceString>::ProverParam,
    ) -> Result<MLEProof<PCS>, PlonkError> {
        let (proof, _) = PCS::open(ck, &self.polynomial, &self.opening_point)?;
        Ok(MLEProof {
            wire_commitments: self.wire_commitments,
            gkr_proof: self.gkr_proof,
            sumcheck_proof: self.sumcheck_proof,
            lookup_proof: self.lookup_proof,
            evals: self.evals,
            opening_proof: proof,
        })
    }
}

/// The claimed evaluations of the witness wires, selectors and permutation related polynomials.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct MLEProofEvals<PCS: PolynomialCommitmentScheme> {
    /// The claimed evaluations of the witness wires.
    pub wire_evals: Vec<PCS::Evaluation>,
    /// The claimed evaluations of the selector polynomials.
    pub selector_evals: Vec<PCS::Evaluation>,
    /// The claimed evaluations of the permutation polynomials.
    pub permutation_evals: Vec<PCS::Evaluation>,
}

/// Proof struct with items pertaining to lookup arguments.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct MLELookupProof<PCS: PolynomialCommitmentScheme> {
    /// Commitments to the multiplicity polynomial.
    pub m_poly_comm: PCS::Commitment,
    /// The claimed evaluations of the polynomials at the challenge point.
    pub lookup_evals: MLELookupEvals<PCS>,
}

/// The claimed evaluations of the lookup polynomials at the challenge point.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct MLELookupEvals<PCS: PolynomialCommitmentScheme> {
    /// Range table eval.
    pub range_table_eval: PCS::Evaluation,
    /// Key table eval.
    pub key_table_eval: PCS::Evaluation,
    /// Table domain separation eval.
    pub table_dom_sep_eval: PCS::Evaluation,
    /// Lookup domain separation selector eval.
    pub q_dom_sep_eval: PCS::Evaluation,
    /// Lookup selector eval.
    pub q_lookup_eval: PCS::Evaluation,
    /// The claimed evaluations of the multiplicity polynomial at the challenge point.
    pub m_poly_eval: PCS::Evaluation,
}

#[allow(dead_code)]
/// Struct used for proving MLE Plonk proofs that use lookups.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct MLELookupProvingKey<PCS: PolynomialCommitmentScheme> {
    /// Range table polynomial.
    pub(crate) range_table_mle: Arc<DenseMultilinearExtension<PCS::Evaluation>>,

    /// Key table polynomial.
    pub(crate) key_table_mle: Arc<DenseMultilinearExtension<PCS::Evaluation>>,

    /// Table domain separation polynomial.
    pub(crate) table_dom_sep_mle: Arc<DenseMultilinearExtension<PCS::Evaluation>>,

    /// Lookup domain separation selector polynomial.
    pub(crate) q_dom_sep_mle: Arc<DenseMultilinearExtension<PCS::Evaluation>>,

    /// Lookup selector polynomial.
    pub(crate) q_lookup_mle: Arc<DenseMultilinearExtension<PCS::Evaluation>>,
}

/// Preprocessed verifier parameters used to verify Plookup proofs for a certain
/// circuit.
#[derive(Debug, Clone, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct MLELookupVerifyingKey<PCS: PolynomialCommitmentScheme> {
    /// Range table polynomial commitment. The commitment is not hiding.
    pub(crate) range_table_comm: PCS::Commitment,

    /// Key table polynomial commitment. The commitment is not hiding.
    pub(crate) key_table_comm: PCS::Commitment,

    /// Table domain separation polynomial commitment. The commitment is not
    /// hiding.
    pub(crate) table_dom_sep_comm: PCS::Commitment,

    /// Lookup domain separation selector polynomial commitment. The commitment
    /// is not hiding.
    pub(crate) q_dom_sep_comm: PCS::Commitment,

    /// Lookup selector polynomial commitment. The commitment
    /// is not hiding.
    pub(crate) q_lookup_comm: PCS::Commitment,
}

/// Struct used to store challenges for the MLE Plonk protocol during verification.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub struct MLEChallenges<F: PrimeField> {
    /// Gamma challenge used in permutation polynomials.
    pub gamma: F,
    /// Alpha challenge used in ZeroCheck batching.
    pub alpha: F,
    /// Lookup challenge used in lookup arguments.
    pub tau: F,
    /// Beta challenge used for GKR domain separation.
    pub beta: F,
    /// The challenge used to combine all the commitments.
    pub delta: F,
    /// The challenge used for the batch ZeroCheck.
    pub epsilon: F,
}

impl<F: PrimeField> Default for MLEChallenges<F> {
    fn default() -> Self {
        MLEChallenges::<F> {
            gamma: F::zero(),
            alpha: F::zero(),
            tau: F::zero(),
            beta: F::zero(),
            delta: F::zero(),
            epsilon: F::zero(),
        }
    }
}

impl<F: PrimeField> MLEChallenges<F> {
    /// Create a new set of challenges from an MLEPlonk Proof.
    pub fn new<PCS, P, T>(
        proof: &MLEProof<PCS>,
        public_input: &[F],
        vk: &MLEVerifyingKey<PCS>,
        transcript: &mut T,
    ) -> Result<Self, PlonkError>
    where
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField + RescueParameter,
        F: EmulationConfig<P::BaseField>,
        PCS: PolynomialCommitmentScheme<Commitment = Affine<P>, Evaluation = P::ScalarField>,
        T: Transcript,
    {
        // Append Vk and public input to transcript.
        transcript.append_visitor(vk)?;

        for pi in public_input.iter() {
            transcript.push_message(b"public input", pi)?;
        }
        // We know that the commitments we are using will always be points on an SW curve.
        // We append wire commitments here.
        transcript.append_curve_points(b"wires", &proof.wire_commitments)?;
        let [gamma, alpha, tau]: [F; 3] = transcript
            .squeeze_scalar_challenges::<P>(b"gamma alpha tau", 3)?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters("Couldn't convert to fixed length array".to_string())
            })?;

        if let Some(lookup_proof) = proof.lookup_proof.as_ref() {
            transcript.append_curve_point(b"m_poly_comm", &lookup_proof.m_poly_comm)?;
        }

        let [beta, delta, epsilon]: [F; 3] = transcript
            .squeeze_scalar_challenges::<P>(b"beta, delta epsilon", 3)?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters("Couldn't convert to fixed length array".to_string())
            })?;

        Ok(Self {
            gamma,
            alpha,
            tau,
            beta,
            delta,
            epsilon,
        })
    }

    /// Create a new set of challenges from an SAMLEProof.
    pub fn new_recursion<PCS, P, T>(
        proof: &SAMLEProof<PCS>,
        public_input: &[F],
        vk: &MLEVerifyingKey<PCS>,
        transcript: &mut T,
    ) -> Result<Self, PlonkError>
    where
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField + RescueParameter,
        F: EmulationConfig<P::BaseField>,
        PCS: Accumulation<
            Commitment = Affine<P>,
            Evaluation = P::ScalarField,
            Point = Vec<P::ScalarField>,
            Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
        >,
        T: Transcript,
    {
        // Append Vk and public input to transcript.
        transcript.append_visitor(vk)?;

        transcript.push_message(b"public input", &public_input[0])?;

        // We know that the commitments we are using will always be points on an SW curve.
        // We append wire commitments here.
        transcript.append_curve_points(b"wires", &proof.wire_commitments)?;
        let [gamma, alpha, tau]: [F; 3] = transcript
            .squeeze_scalar_challenges::<P>(b"gamma alpha tau", 3)?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters("Couldn't convert to fixed length array".to_string())
            })?;

        if let Some(lookup_proof) = proof.lookup_proof.as_ref() {
            transcript.append_curve_point(b"m_poly_comm", &lookup_proof.m_poly_comm)?;
        }

        let [beta, delta, epsilon]: [F; 3] = transcript
            .squeeze_scalar_challenges::<P>(b"beta, delta epsilon", 3)?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters("Couldn't convert to fixed length array".to_string())
            })?;

        Ok(Self {
            gamma,
            alpha,
            tau,
            beta,
            delta,
            epsilon,
        })
    }
}

impl<F: PrimeField> TryFrom<Vec<F>> for MLEChallenges<F> {
    type Error = PlonkError;

    fn try_from(challenges: Vec<F>) -> Result<Self, Self::Error> {
        if challenges.len() != 6 {
            return Err(PlonkError::InvalidParameters(
                "MLEChallenges must have length 6".to_string(),
            ));
        }
        Ok(Self {
            gamma: challenges[0],
            alpha: challenges[1],
            tau: challenges[2],
            beta: challenges[3],
            delta: challenges[4],
            epsilon: challenges[5],
        })
    }
}

// Challenges always fit into either the base field or scalar field so we can write a From implementation for this struct.
impl<Fq, Fr> From<&MLEChallenges<Fq>> for MLEChallenges<Fr>
where
    Fr: PrimeField,
    Fq: PrimeField,
{
    fn from(challenges: &MLEChallenges<Fq>) -> Self {
        Self {
            gamma: Fr::from_le_bytes_mod_order(&challenges.gamma.into_bigint().to_bytes_le()),
            alpha: Fr::from_le_bytes_mod_order(&challenges.alpha.into_bigint().to_bytes_le()),
            tau: Fr::from_le_bytes_mod_order(&challenges.tau.into_bigint().to_bytes_le()),
            beta: Fr::from_le_bytes_mod_order(&challenges.beta.into_bigint().to_bytes_le()),
            delta: Fr::from_le_bytes_mod_order(&challenges.delta.into_bigint().to_bytes_le()),
            epsilon: Fr::from_le_bytes_mod_order(&challenges.epsilon.into_bigint().to_bytes_le()),
        }
    }
}
