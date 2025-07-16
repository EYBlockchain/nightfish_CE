// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Interfaces for Plonk-based proof systems
use ark_ec::AffineRepr;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    error::Error,
    fmt::Debug,
    rand::{CryptoRng, RngCore},
    vec::Vec,
};
use jf_primitives::pcs::{Accumulation, PolynomialCommitmentScheme};
use jf_relation::Arithmetization;

pub(crate) mod prover;

pub(crate) mod snark;
pub mod structs;
pub(crate) mod verifier;
use crate::transcript::Transcript;
pub use snark::PlonkKzgSnark;

// TODO: (alex) should we name it `PlonkishSNARK` instead? since we use
// `PlonkTranscript` on prove and verify.
/// An interface for SNARKs with universal setup.
pub trait UniversalSNARK<PCS: PolynomialCommitmentScheme>
where
    PCS::Commitment: AffineRepr,
    <PCS::Commitment as AffineRepr>::BaseField: PrimeField,
    <PCS::Commitment as AffineRepr>::ScalarField: PrimeField,
{
    /// The SNARK proof computed by the prover.
    type Proof: Clone + Send + Sync;

    /// The SNARK proof struct used in a recursive setting.
    type RecursiveProof: Default + Clone + Send + Sync;

    /// The parameters required by the prover to compute a proof for a specific
    /// circuit.
    type ProvingKey: Clone;

    /// The parameters required by the verifier to validate a proof for a
    /// specific circuit.
    type VerifyingKey: Clone;

    /// Universal Structured Reference String from `universal_setup`, used for
    /// all subsequent circuit-specific preprocessing
    type UniversalSRS: Clone + Debug;

    /// SNARK related error
    type Error: 'static + Error;

    /// Generate the universal SRS for the argument system.
    /// This setup is for trusted party to run, and mostly only used for
    /// testing purpose. In practice, a MPC flavor of the setup will be carried
    /// out to have higher assurance on the "toxic waste"/trapdoor being thrown
    /// away to ensure soundness of the argument system.
    fn universal_setup<R: RngCore + CryptoRng>(
        _max_degree: usize,
        _rng: &mut R,
    ) -> Result<PCS::SRS, Self::Error> {
        unimplemented!("Should load from files in practice.");
    }

    /// Same as `universal_setup`, but for testing and benchmarking code only.
    /// Insecure local generation for trusted setup! Don't use in production!
    #[cfg(any(test, feature = "test-srs"))]
    fn universal_setup_for_testing<R: RngCore + CryptoRng>(
        _max_degree: usize,
        _rng: &mut R,
    ) -> Result<PCS::SRS, Self::Error>;

    /// Circuit-specific preprocessing to compute the proving/verifying keys.
    fn preprocess<C: Arithmetization<<PCS::Commitment as AffineRepr>::ScalarField>>(
        srs: &Self::UniversalSRS,
        circuit: &C,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), Self::Error>;

    /// Compute a SNARK proof of a circuit `circuit`, using the corresponding
    /// proving key `prove_key`. The witness used to
    /// generate the proof can be obtained from `circuit`.
    ///
    /// `extra_transcript_init_msg` is the optional message to be
    /// appended to the transcript during its initialization before obtaining
    /// any challenges. This field allows application-specific data bound to the
    /// resulting proof without any check on the data. It does not incur any
    /// additional cost in proof size or prove time.
    fn prove<C, R, T>(
        rng: &mut R,
        circuit: &C,
        prove_key: &Self::ProvingKey,
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<Self::Proof, Self::Error>
    where
        C: Arithmetization<<PCS::Commitment as AffineRepr>::ScalarField>,
        R: CryptoRng + RngCore,
        T: Transcript;

    /// Compute a SNARK proof of a circuit `circuit` for use in a recursive proving system, using the corresponding
    /// proving key `prove_key`. The witness used to
    /// generate the proof can be obtained from `circuit`.
    /// This method is called with a transcript `T` that is used to generate the proof and
    /// will return the end transcript so that it can be used in a recursive context if applicable.
    ///
    /// `extra_transcript_init_msg` is the optional message to be
    /// appended to the transcript during its initialization before obtaining
    /// any challenges. This field allows application-specific data bound to the
    /// resulting proof without any check on the data. It does not incur any
    /// additional cost in proof size or prove time.
    fn recursive_prove<C, R, T>(
        rng: &mut R,
        circuit: &C,
        prove_key: &Self::ProvingKey,
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<RecursiveOutput<PCS, Self, T>, Self::Error>
    where
        Self: Sized,
        PCS: Accumulation,
        C: Arithmetization<<PCS::Commitment as AffineRepr>::ScalarField>,
        R: CryptoRng + RngCore,
        T: Transcript + CanonicalSerialize + CanonicalDeserialize,
        Self::RecursiveProof: CanonicalSerialize + CanonicalDeserialize,
        <PCS::Commitment as AffineRepr>::ScalarField: CanonicalSerialize + CanonicalDeserialize;

    /// Verify a SNARK proof `proof` of the circuit `circuit`, with respect to
    /// the public input `pub_input`.
    ///
    /// `extra_transcript_init_msg`: refer to documentation of `prove`
    fn verify<T: Transcript>(
        verify_key: &Self::VerifyingKey,
        public_input: &[<PCS::Commitment as AffineRepr>::ScalarField],
        proof: &Self::Proof,
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<(), Self::Error>;
}

/// This struct defines the output to the recursive prover.
#[derive(Debug, Clone, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct RecursiveOutput<PCS, Scheme, T>
where
    PCS: Accumulation,
    PCS::Commitment: AffineRepr,
    <PCS::Commitment as AffineRepr>::BaseField: PrimeField,
    <PCS::Commitment as AffineRepr>::ScalarField: PrimeField + CanonicalSerialize + CanonicalDeserialize,
    Scheme: UniversalSNARK<PCS>,
    Scheme::RecursiveProof: CanonicalSerialize + CanonicalDeserialize,
    T: Transcript + CanonicalSerialize + CanonicalDeserialize,
{
    /// The proof generated by the recursive prover.
    pub proof: Scheme::RecursiveProof,
    /// The hash of the public inputs to this proof.
    pub pi_hash: <PCS::Commitment as AffineRepr>::ScalarField,
    /// The transcript of the proof.
    pub transcript: T,
}

impl<PCS, Scheme, T> RecursiveOutput<PCS, Scheme, T>
where
    PCS: Accumulation,
    PCS::Commitment: AffineRepr,
    <PCS::Commitment as AffineRepr>::BaseField: PrimeField,
    <PCS::Commitment as AffineRepr>::ScalarField: PrimeField + CanonicalSerialize + CanonicalDeserialize,
    Scheme: UniversalSNARK<PCS>,
    Scheme::RecursiveProof: CanonicalSerialize + CanonicalDeserialize,
    T: Transcript + CanonicalSerialize + CanonicalDeserialize,
{
    /// Create a new recursive output.
    pub fn new(
        proof: Scheme::RecursiveProof,
        pi_hash: <PCS::Commitment as AffineRepr>::ScalarField,
        transcript: T,
    ) -> Self {
        Self {
            proof,
            pi_hash,
            transcript,
        }
    }
}
