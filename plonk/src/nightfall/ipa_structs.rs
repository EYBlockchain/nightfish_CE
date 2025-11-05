// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Data structures used in Plonk proof systems
use crate::{
    errors::{PlonkError, SnarkError::SnarkLookupUnsupported},
    proof_system::structs::{PlookupEvaluations, ProofEvaluations},
    transcript::{Transcript, TranscriptVisitor},
};
use ark_ec::{
    scalar_mul::variable_base::VariableBaseMSM,
    short_weierstrass::{Affine, SWCurveConfig},
    AffineRepr,
};
use ark_ff::{FftField, Field, PrimeField};
use ark_poly::univariate::DensePolynomial;
use ark_serialize::*;
use ark_std::{string::ToString, vec, vec::Vec};
use espresso_systems_common::jellyfish::tag;

use jf_primitives::pcs::{PolynomialCommitmentScheme, StructuredReferenceString};
use jf_relation::{
    constants::{compute_coset_representatives, GATE_WIDTH, N_TURBO_PLONK_SELECTORS},
    gadgets::ecc::HasTEForm,
};
use jf_utils::to_bytes;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use sha3::{Digest, Keccak256};
use tagged_base64::tagged;

use super::hops::srs::UnivariateUniversalIpaParams;

/// Universal StructuredReferenceString
pub type UniversalSrs<E> = UnivariateUniversalIpaParams<E>;
/// Commitment key
pub type CommitKey<E> = UnivariateUniversalIpaParams<E>;
/// Key for verifying PCS opening proof.
pub type OpenKey<E> = UnivariateUniversalIpaParams<E>;

/// A Plonk SNARK proof.
#[tagged(tag::PROOF)]
#[derive(Debug, Clone, Eq, CanonicalSerialize, CanonicalDeserialize, Derivative, Default)]
#[derivative(PartialEq, Hash)]
pub struct Proof<PCS>
where
    PCS: PolynomialCommitmentScheme,
{
    /// Wire witness polynomials commitments.
    pub(crate) wires_poly_comms: Vec<PCS::Commitment>,

    /// The polynomial commitment for the wire permutation argument.
    pub(crate) prod_perm_poly_comm: PCS::Commitment,

    /// Splitted quotient polynomial commitments.
    pub(crate) split_quot_poly_comms: Vec<PCS::Commitment>,

    /// (Aggregated) proof of evaluations at challenge point `v`.
    pub(crate) opening_proof: PCS::Proof,

    /// The commitment to the polynomial q(x) used in the multi-opening protocol.
    pub(crate) q_comm: PCS::Commitment,

    /// Polynomial evaluations.
    pub(crate) poly_evals: ProofEvaluations<PCS::Evaluation>,

    /// The partial proof for Plookup argument
    pub(crate) plookup_proof: Option<PlookupProof<PCS>>,
}

// helper function to convert a G1Affine or G2Affine into two base fields
#[allow(dead_code)]
fn group1_to_fields<P>(p: Affine<P>) -> Vec<P::BaseField>
where
    P: SWCurveConfig,
{
    // contains x, y, infinity_flag, only need the first 2 field elements
    vec![p.x, p.y]
}

/// A Plookup argument proof.
#[derive(Debug, Clone, Eq, CanonicalSerialize, CanonicalDeserialize, Derivative)]
#[derivative(PartialEq, Hash(bound = "PCS: PolynomialCommitmentScheme"))]
pub struct PlookupProof<PCS>
where
    PCS: PolynomialCommitmentScheme,
{
    /// The commitments for the polynomials that interpolate the sorted
    /// concatenation of the lookup table and the witnesses in the lookup gates.
    pub(crate) h_poly_comms: Vec<PCS::Commitment>,

    /// The product accumulation polynomial commitment for the Plookup argument
    pub(crate) prod_lookup_poly_comm: PCS::Commitment,

    /// Polynomial evaluations.
    pub(crate) poly_evals: PlookupEvaluations<PCS::Evaluation>,
}

/// Preprocessed prover parameters used to compute Plonk proofs for a certain
/// circuit.
#[derive(Debug, Clone, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProvingKey<PCS>
where
    PCS: PolynomialCommitmentScheme,
{
    /// Extended permutation (sigma) polynomials.
    pub(crate) sigmas: Vec<PCS::Polynomial>,

    /// Selector polynomials.
    pub(crate) selectors: Vec<PCS::Polynomial>,

    // KZG PCS committing key.
    pub(crate) commit_key: <PCS::SRS as StructuredReferenceString>::ProverParam,

    /// The verifying key. It is used by prover to initialize transcripts.
    pub vk: VerifyingKey<PCS>,

    /// Proving key for Plookup, None if not support lookup.
    pub(crate) plookup_pk: Option<PlookupProvingKey<PCS>>,
}

/// Preprocessed prover parameters used to compute Plookup proofs for a certain
/// circuit.
#[derive(Debug, Clone, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct PlookupProvingKey<PCS: PolynomialCommitmentScheme> {
    /// Range table polynomial.
    pub(crate) range_table_poly: PCS::Polynomial,

    /// Key table polynomial.
    pub(crate) key_table_poly: PCS::Polynomial,

    /// Table domain separation polynomial.
    pub(crate) table_dom_sep_poly: PCS::Polynomial,

    /// Lookup domain separation selector polynomial.
    pub(crate) q_dom_sep_poly: PCS::Polynomial,
}

impl<PCS> ProvingKey<PCS>
where
    PCS: PolynomialCommitmentScheme,
{
    /// The size of the evaluation domain. Should be a power of two.
    pub(crate) fn domain_size(&self) -> usize {
        self.vk.domain_size
    }
    /// The number of public inputs.
    #[allow(dead_code)]
    pub(crate) fn num_inputs(&self) -> usize {
        self.vk.num_inputs
    }
    /// The constants K0, ..., K4 that ensure wire subsets are disjoint.
    pub(crate) fn k(&self) -> &[PCS::Evaluation] {
        &self.vk.k
    }

    /// The lookup selector polynomial
    pub(crate) fn q_lookup_poly(&self) -> Result<&PCS::Polynomial, PlonkError> {
        if self.plookup_pk.is_none() {
            return Err(SnarkLookupUnsupported.into());
        }
        Ok(self.selectors.last().unwrap())
    }
}

/// Preprocessed verifier parameters used to verify Plonk proofs for a certain
/// circuit.
#[derive(Debug, Clone, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct VerifyingKey<PCS>
where
    PCS: PolynomialCommitmentScheme,
{
    /// The size of the evaluation domain. Should be a power of two.
    pub(crate) domain_size: usize,

    /// The number of public inputs.
    pub(crate) num_inputs: usize,

    /// The permutation polynomial commitments. The commitments are not hiding.
    pub(crate) sigma_comms: Vec<PCS::Commitment>,

    /// The selector polynomial commitments. The commitments are not hiding.
    pub(crate) selector_comms: Vec<PCS::Commitment>,

    /// The constants K0, ..., K_num_wire_types that ensure wire subsets are
    /// disjoint.
    pub(crate) k: Vec<PCS::Evaluation>,

    /// KZG PCS opening key.
    pub open_key: <PCS::SRS as StructuredReferenceString>::VerifierParam,

    /// A flag indicating whether the key is a merged key.
    pub(crate) is_merged: bool,

    /// Plookup verifying key, None if not support lookup.
    pub(crate) plookup_vk: Option<PlookupVerifyingKey<PCS>>,

    /// Used for client verification keys to distinguish between
    /// transfer/withdrawal and deposit.
    pub(crate) id: Option<VerificationKeyId>,
}
/// APIs for generic plonk verifying key.
pub trait VK<PCS: PolynomialCommitmentScheme> {
    /// The size of the evaluation domain. Should be a power of two.
    fn domain_size(&self) -> usize;

    /// The number of public inputs.
    fn num_inputs(&self) -> usize;

    /// The permutation polynomial commitments. The commitments are not hiding.
    fn sigma_comms(&self) -> &[PCS::Commitment];

    /// The selector polynomial commitments. The commitments are not hiding.
    fn selector_comms(&self) -> &[PCS::Commitment];

    /// The constants K0, ..., K_num_wire_types that ensure wire subsets are
    /// disjoint.
    fn k(&self) -> &[PCS::Evaluation];

    /// A flag indicating whether the key is a merged key.
    fn is_merged(&self) -> bool;

    /// Plookup verifying key, None if not support lookup.
    fn plookup_vk(&self) -> Option<&PlookupVerifyingKey<PCS>>;

    /// Get the id of the verifying key.
    fn id(&self) -> Option<VerificationKeyId>;

    /// Get the hash of the verifying key.
    fn hash(&self) -> PCS::Evaluation;
}

impl<PCS> VK<PCS> for VerifyingKey<PCS>
where
    PCS: PolynomialCommitmentScheme,
    <PCS::Commitment as AffineRepr>::BaseField: PrimeField,
{
    fn domain_size(&self) -> usize {
        self.domain_size
    }

    fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    fn selector_comms(&self) -> &[PCS::Commitment] {
        &self.selector_comms
    }

    fn sigma_comms(&self) -> &[PCS::Commitment] {
        &self.sigma_comms
    }

    fn k(&self) -> &[PCS::Evaluation] {
        &self.k
    }

    fn is_merged(&self) -> bool {
        self.is_merged
    }

    fn plookup_vk(&self) -> Option<&PlookupVerifyingKey<PCS>> {
        self.plookup_vk.as_ref()
    }

    fn id(&self) -> Option<VerificationKeyId> {
        self.id
    }

    fn hash(&self) -> <PCS as PolynomialCommitmentScheme>::Evaluation {
        let mut hasher = Keccak256::new();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&to_bytes!(&self.domain_size).unwrap());

        for com in self.sigma_comms.iter() {
            bytes.extend_from_slice(&to_bytes!(com).unwrap());
        }

        for com in self.selector_comms.iter() {
            bytes.extend_from_slice(&to_bytes!(com).unwrap());
        }

        for k in self.k.iter() {
            bytes.extend_from_slice(&to_bytes!(k).unwrap());
        }

        bytes.extend_from_slice(&to_bytes!(&self.open_key).unwrap());

        if let Some(plookup_vk) = self.plookup_vk.as_ref() {
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
}

impl<PCS> TranscriptVisitor for VerifyingKey<PCS>
where
    PCS: PolynomialCommitmentScheme,
    PCS::Commitment: AffineRepr,
    <PCS::Commitment as AffineRepr>::Config: HasTEForm,
    <PCS::Commitment as AffineRepr>::BaseField: PrimeField,
{
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) -> Result<(), PlonkError> {
        let id_field = if let Some(id) = self.id {
            Ok(PCS::Evaluation::from(id as u8))
        } else {
            Err(PlonkError::InvalidParameters(
                "Verifying key has no id".to_string(),
            ))
        }?;

        transcript.push_message(b"verifying key id", &id_field)?;
        Ok(())
    }
}

/// Preprocessed verifier parameters used to verify Plookup proofs for a certain
/// circuit.
#[derive(Debug, Clone, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct PlookupVerifyingKey<PCS>
where
    PCS: PolynomialCommitmentScheme,
{
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
}

/// An enum to identify different client verification keys.
/// Client means transfer or withdrawal.
/// These will not be used for merge verification keys.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, IntoPrimitive, TryFromPrimitive)]
pub enum VerificationKeyId {
    /// Transfer or Withdrawal
    Client = 0,
    /// Deposit
    Deposit = 1,
}

// --- Serialize the enum as exactly one byte ---
impl CanonicalSerialize for VerificationKeyId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let b: u8 = (*self).into();
        b.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        1
    }
}

impl CanonicalDeserialize for VerificationKeyId {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let b = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        VerificationKeyId::try_from(b).map_err(|_| SerializationError::InvalidData)
    }
}

impl Valid for VerificationKeyId {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<PCS> VerifyingKey<PCS>
where
    PCS: PolynomialCommitmentScheme,
{
    /// Create a dummy TurboPlonk verification key for a circuit with
    /// `num_inputs` public inputs and domain size `domain_size`.
    pub fn dummy(num_inputs: usize, domain_size: usize) -> Self {
        let num_wire_types = GATE_WIDTH + 1;
        Self {
            domain_size,
            num_inputs,
            sigma_comms: vec![PCS::Commitment::default(); num_wire_types],
            selector_comms: vec![PCS::Commitment::default(); N_TURBO_PLONK_SELECTORS],
            k: compute_coset_representatives(num_wire_types, Some(domain_size)),
            open_key: <PCS::SRS as StructuredReferenceString>::VerifierParam::default(),
            is_merged: false,
            plookup_vk: None,
            id: None,
        }
    }

    /// The lookup selector polynomial commitment
    pub(crate) fn q_lookup_comm(&self) -> Result<&PCS::Commitment, PlonkError> {
        if self.plookup_vk.is_none() {
            return Err(SnarkLookupUnsupported.into());
        }
        Ok(self.selector_comms.last().unwrap())
    }
}

/// Plonk IOP verifier challenges.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Challenges<F: Field> {
    pub(crate) tau: F,
    pub(crate) alpha: F,
    pub(crate) beta: F,
    pub(crate) gamma: F,
    pub(crate) zeta: F,
    pub(crate) v: F,
    pub(crate) u: F,
}

impl<F: Field> TryFrom<Vec<F>> for Challenges<F> {
    type Error = PlonkError;
    fn try_from(challenges: Vec<F>) -> Result<Self, Self::Error> {
        if challenges.len() != 7 {
            return Err(PlonkError::InvalidParameters(
                "Invalid number of challenges".to_string(),
            ));
        }
        Ok(Self {
            tau: challenges[0],
            alpha: challenges[1],
            beta: challenges[2],
            gamma: challenges[3],
            zeta: challenges[4],
            v: challenges[5],
            u: challenges[6],
        })
    }
}

/// Plonk IOP online polynomial oracles.
#[derive(Debug, Default, Clone)]
pub(crate) struct Oracles<F: FftField> {
    pub(crate) wire_polys: Vec<DensePolynomial<F>>,
    pub(crate) pub_inp_poly: DensePolynomial<F>,
    pub(crate) prod_perm_poly: DensePolynomial<F>,
    pub(crate) plookup_oracles: PlookupOracles<F>,
}

/// Plookup IOP online polynomial oracles.
#[derive(Debug, Default, Clone)]
pub(crate) struct PlookupOracles<F: FftField> {
    pub(crate) h_polys: Vec<DensePolynomial<F>>,
    pub(crate) prod_lookup_poly: DensePolynomial<F>,
}

/// The vector representation of bases and corresponding scalars.
#[derive(Debug, Clone)]
pub struct ScalarsAndBases<PCS: PolynomialCommitmentScheme> {
    pub(crate) bases: Vec<PCS::Commitment>,
    pub(crate) scalars: Vec<PCS::Evaluation>,
}

impl<PCS: PolynomialCommitmentScheme> ScalarsAndBases<PCS> {
    pub(crate) fn new() -> Self {
        Self {
            bases: Vec::new(),
            scalars: Vec::new(),
        }
    }
    /// Insert a base point and the corresponding scalar.
    pub(crate) fn push(
        &mut self,
        scalar: PCS::Evaluation,
        base: PCS::Commitment,
    ) -> Result<(), PlonkError> {
        self.bases.push(base);
        self.scalars.push(scalar);
        Ok(())
    }

    /// Add a list of scalars and bases into self, where each scalar is
    /// multiplied by a constant c.
    pub(crate) fn merge(
        &mut self,
        c: PCS::Evaluation,
        scalars_and_bases: &Self,
    ) -> Result<(), PlonkError> {
        for (base, scalar) in scalars_and_bases
            .bases
            .iter()
            .zip(scalars_and_bases.scalars.iter())
        {
            self.push(c * scalar, *base)?;
        }
        Ok(())
    }
    /// Compute the multi-scalar multiplication.
    pub(crate) fn multi_scalar_mul(&self) -> <PCS::Commitment as AffineRepr>::Group {
        let mut scalars = vec![];
        for scalar in self.scalars.iter() {
            scalars.push(scalar.into_bigint());
        }
        VariableBaseMSM::msm_bigint(&self.bases, &scalars)
    }

    #[allow(dead_code)]
    /// Returns the scalars as a slice.
    pub(crate) fn scalars(&self) -> &[PCS::Evaluation] {
        &self.scalars
    }

    #[allow(dead_code)]
    /// Returns the bases as a slice.
    pub(crate) fn bases(&self) -> &[PCS::Commitment] {
        &self.bases
    }
}

#[derive(Default, Debug, Clone)]
/// Used as the key in Key, Value pair in the BTreeMap used by the prover and verifier.
pub struct MapKey<PCS: PolynomialCommitmentScheme>(pub usize, pub PCS::Polynomial);

impl<PCS: PolynomialCommitmentScheme> PartialEq for MapKey<PCS> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<PCS: PolynomialCommitmentScheme> Eq for MapKey<PCS> {}

impl<PCS: PolynomialCommitmentScheme> Ord for MapKey<PCS> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<PCS: PolynomialCommitmentScheme> PartialOrd for MapKey<PCS> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// Utility function for computing merged table evaluations.
#[inline]
pub(crate) fn eval_merged_table<E: SWCurveConfig>(
    tau: E::ScalarField,
    range_eval: E::ScalarField,
    key_eval: E::ScalarField,
    q_lookup_eval: E::ScalarField,
    w3_eval: E::ScalarField,
    w4_eval: E::ScalarField,
    table_dom_sep_eval: E::ScalarField,
) -> E::ScalarField {
    range_eval
        + q_lookup_eval
            * tau
            * (table_dom_sep_eval + tau * (key_eval + tau * (w3_eval + tau * w4_eval)))
}

// Utility function for computing merged lookup witness evaluations.
#[inline]
pub(crate) fn eval_merged_lookup_witness<E: SWCurveConfig>(
    tau: E::ScalarField,
    w_range_eval: E::ScalarField,
    w_0_eval: E::ScalarField,
    w_1_eval: E::ScalarField,
    w_2_eval: E::ScalarField,
    q_lookup_eval: E::ScalarField,
    q_dom_sep_eval: E::ScalarField,
) -> E::ScalarField {
    w_range_eval
        + q_lookup_eval
            * tau
            * (q_dom_sep_eval + tau * (w_0_eval + tau * (w_1_eval + tau * w_2_eval)))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::nightfall::ipa_snark::test::gen_circuit_for_test;
    use crate::nightfall::FFTPlonk;
    use crate::proof_system::structs::UniversalSrs as KZGUniversalSrs;
    use crate::proof_system::UniversalSNARK;
    use crate::transcript::RescueTranscript;
    use crate::transcript::SolidityTranscript;
    use crate::transcript::StandardTranscript;
    use crate::PlonkType;
    use ark_bls12_377::{Bls12_377, Fq as Fq377};
    use ark_bls12_381::{Bls12_381, Fq as Fq381};
    use ark_bn254::{g1::Config, Bn254, Fq as Fq254};
    use ark_bw6_761::{Fq as Fq761, BW6_761};
    use ark_ec::pairing::Pairing;
    use ark_ec::short_weierstrass::Projective;
    use itertools::izip;
    use jf_primitives::pcs::prelude::UnivariateKzgPCS;
    use jf_primitives::rescue::RescueParameter;
    use jf_relation::gadgets::EmulationConfig;
    use jf_relation::Arithmetization;
    use jf_relation::Circuit;

    #[test]
    fn test_group_to_field() {
        let g1 = <Bn254 as Pairing>::G1Affine::generator();
        let f1: Vec<Fq254> = group1_to_fields::<Config>(g1);
        assert_eq!(f1.len(), 2);
    }
    #[test]
    fn test_serde_kzg() -> Result<(), PlonkError> {
        for (plonk_type, vk_id, blind) in izip!(
            [PlonkType::TurboPlonk, PlonkType::UltraPlonk],
            [
                None,
                Some(VerificationKeyId::Client),
                Some(VerificationKeyId::Deposit),
            ],
            [true, false],
        ) {
            // merlin transcripts
            test_serde_helper::<Bn254, Fq254, _, StandardTranscript>(plonk_type, vk_id, blind)?;
            test_serde_helper::<Bls12_377, Fq377, _, StandardTranscript>(plonk_type, vk_id, blind)?;
            test_serde_helper::<Bls12_381, Fq381, _, StandardTranscript>(plonk_type, vk_id, blind)?;
            test_serde_helper::<BW6_761, Fq761, _, StandardTranscript>(plonk_type, vk_id, blind)?;
            // rescue transcripts
            test_serde_helper::<Bls12_377, Fq377, _, RescueTranscript<Fq377>>(
                plonk_type, vk_id, blind,
            )?;
            // Solidity-friendly keccak256 transcript
            test_serde_helper::<Bls12_381, Fq381, _, SolidityTranscript>(plonk_type, vk_id, blind)?;
        }

        Ok(())
    }

    fn test_serde_helper<E, F, P, T>(
        plonk_type: PlonkType,
        vk_id: Option<VerificationKeyId>,
        blind: bool,
    ) -> Result<(), PlonkError>
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
        F: RescueParameter + PrimeField,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        P::ScalarField: EmulationConfig<F> + RescueParameter,
        T: Transcript,
        jf_primitives::pcs::prelude::UnivariateKzgPCS<E>: PartialEq,
    {
        let rng = &mut jf_utils::test_rng();
        let circuit = gen_circuit_for_test(3, 4, plonk_type, false)?;
        let srs_size = circuit.srs_size(blind).unwrap();
        let _max_degree = 80;
        let srs =
            FFTPlonk::<UnivariateKzgPCS<E>>::universal_setup_for_testing(srs_size, rng).unwrap();
        let (pk, vk) =
            FFTPlonk::<UnivariateKzgPCS<E>>::preprocess(&srs, vk_id, &circuit, blind).unwrap();
        let proof =
            FFTPlonk::<UnivariateKzgPCS<E>>::prove::<_, _, T>(rng, &circuit, &pk, None, blind)?;
        let public_inputs = circuit.public_input().unwrap();
        let public_inputs1 = public_inputs.as_slice();

        FFTPlonk::<UnivariateKzgPCS<E>>::verify::<T>(&vk, public_inputs1, &proof, None, blind)?;

        let mut ser_bytes = Vec::new();
        srs.serialize_compressed(&mut ser_bytes)?;
        let de = KZGUniversalSrs::<E>::deserialize_compressed(&ser_bytes[..])?;
        assert_eq!(de, srs);

        let mut ser_bytes = Vec::new();
        pk.serialize_compressed(&mut ser_bytes)?;

        let de = ProvingKey::<UnivariateKzgPCS<E>>::deserialize_compressed(&ser_bytes[..])?;
        assert_eq!(de, pk);

        let mut ser_bytes = Vec::new();
        vk.serialize_compressed(&mut ser_bytes)?;
        let de = VerifyingKey::<UnivariateKzgPCS<E>>::deserialize_compressed(&ser_bytes[..])?;
        assert_eq!(de, vk);

        let mut ser_bytes = Vec::new();
        proof.serialize_compressed(&mut ser_bytes)?;
        let de = Proof::<UnivariateKzgPCS<E>>::deserialize_compressed(&ser_bytes[..])?;
        assert_eq!(de, proof);

        Ok(())
    }
}
