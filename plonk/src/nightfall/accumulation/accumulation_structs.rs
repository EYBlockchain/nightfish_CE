//! Accumulation structs used by the accumulation prover and verifier circuit.
use ark_ec::AffineRepr;
use ark_ff::PrimeField;
use ark_poly::univariate::DensePolynomial;
use ark_std::{rand::Rng, vec::Vec};
use jf_primitives::{
    pcs::{Accumulation, PolynomialCommitmentScheme, StructuredReferenceString},
    rescue::RescueParameter,
};
use rand_chacha::rand_core::{CryptoRng, RngCore};

use crate::{
    errors::PlonkError,
    nightfall::mle::{
        mle_structs::PolynomialError,
        subroutines::{PolyOracle, SumCheckProof},
    },
    transcript::RescueTranscript,
};

/// A struct representing a split-accumulation instance.
/// This is a commitment to a polynomial 'p(x)', a point z, and
/// the value v = p(z).
#[derive(Debug, Clone)]
pub struct PCSInstance<PCS>
where
    PCS: Accumulation,
{
    /// The commitment to the witness polynomial.
    pub comm: PCS::Commitment,
    /// The claimed evaluation of the witness polynomial at 'point'.
    pub value: PCS::Evaluation,
    /// The point we evaluate at.
    pub point: PCS::Point,
}

impl<PCS: Accumulation> Default for PCSInstance<PCS> {
    fn default() -> Self {
        Self {
            comm: PCS::Commitment::default(),
            value: PCS::Evaluation::default(),
            point: PCS::Point::default(),
        }
    }
}

/// A struct representing an atomic-accumulation instance.
/// This is a commitment to a polynomial 'p(x)', a point z, and
/// the value v = p(z).
#[derive(Debug, Clone)]
pub struct AtomicInstance<PCS>
where
    PCS: Accumulation,
{
    /// The commitment to the witness polynomial.
    pub comm: PCS::Commitment,
    /// The claimed evaluation of the witness polynomial at 'point'.
    pub value: PCS::Evaluation,
    /// The point we evaluate at.
    pub point: PCS::Point,
    /// The opening proof that `comm` evaluates to `value` at `point`.
    pub opening_proof: PCS::Proof,
}

impl<PCS: Accumulation> Default for AtomicInstance<PCS> {
    fn default() -> Self {
        Self {
            comm: PCS::Commitment::default(),
            value: PCS::Evaluation::default(),
            point: PCS::Point::default(),
            opening_proof: PCS::Proof::default(),
        }
    }
}

/// A struct representing a split-accumulation witness.
/// This is a polynomial 'p(x)', a point z, and
/// the value v = p(z).
#[derive(Debug, Clone)]
pub struct PCSWitness<PCS>
where
    PCS: Accumulation,
{
    /// The witness polynomial.
    pub poly: PCS::Polynomial,
    /// The commitment to the witness polynomial.
    pub comm: PCS::Commitment,
    /// The claimed evaluation of the witness polynomial at 'point'.
    pub value: PCS::Evaluation,
    /// The point we evaluate at.
    pub point: PCS::Point,
}

impl<PCS: Accumulation> Default for PCSWitness<PCS> {
    fn default() -> Self {
        Self {
            poly: PCS::Polynomial::default(),
            comm: PCS::Commitment::default(),
            value: PCS::Evaluation::default(),
            point: PCS::Point::default(),
        }
    }
}

/// A struct representing an atomic accumulation proof.
#[derive(Clone)]
pub struct AtomicAccProof<PCS>
where
    PCS: Accumulation,
{
    /// The part of the proof that is accumulated with the commitments.
    pub s_beta_g: PCS::Commitment,
    /// The part of the proof that is accumulated with the opening proofs.
    pub s_g: PCS::Commitment,
}

/// A struct representing a split-acumulation proof
/// where the underlying commitment scheme uses ubivariate polynomials.
pub struct UVAccProof<PCS, F>
where
    PCS: Accumulation<Polynomial = DensePolynomial<F>, Evaluation = F>,
    F: PrimeField,
{
    /// The evaluations of the witness polynomials at the challenge point 'z_*'.
    pub y_i: Vec<PCS::Evaluation>,
    /// The evaluations of the proof polynomials 'W_i(x)' at the challenge point 'z_*'.
    pub y_i_prime: Vec<PCS::Evaluation>,
    /// Commitments to the proof polynomials 'w_i(x)'.
    pub w_i_comm: Vec<PCS::Commitment>,
}

/// In the multilinear case an accumulation proof is just a [`SumCheckProof`](crate::nightfall::mle::subroutines::SumCheckProof).
pub type MLEAccProof<PCS> = SumCheckProof<
    <PCS as PolynomialCommitmentScheme>::Evaluation,
    PolyOracle<<PCS as PolynomialCommitmentScheme>::Evaluation>,
>;

impl<PCS> PCSInstance<PCS>
where
    PCS: Accumulation,
{
    /// Creates a new PCSInstance from its constituent parts.
    pub fn new(comm: PCS::Commitment, value: PCS::Evaluation, point: PCS::Point) -> Self {
        Self { comm, value, point }
    }
}

impl<PCS> PCSWitness<PCS>
where
    PCS: Accumulation,
{
    /// Creates a new PCSInstance from its constituent parts.
    pub fn new(
        poly: PCS::Polynomial,
        comm: PCS::Commitment,
        value: PCS::Evaluation,
        point: PCS::Point,
    ) -> Self {
        Self {
            poly,
            comm,
            value,
            point,
        }
    }
}

impl<PCS> AtomicInstance<PCS>
where
    PCS: Accumulation,
{
    /// Creates a new AtomicInstance from its constituent parts.
    pub fn new(
        comm: PCS::Commitment,
        value: PCS::Evaluation,
        point: PCS::Point,
        opening_proof: PCS::Proof,
    ) -> Self {
        Self {
            comm,
            value,
            point,
            opening_proof,
        }
    }
}
/// Marker trait for things that can be accumulated.
pub trait Accumulator {}

/// Trait that provides interfaces for a split accumulator.
pub trait SplitAccumulator<PCS: Accumulation>: Accumulator {
    /// The type of the instance.
    type Instance: Clone;
    /// The type of the witness.
    type Witness: Clone;
    /// The type of the proof.
    type AccProof: Clone;
    /// The type of `prove_accumulation_with_challenges_and_scalars`'s return value.
    type WithChallengesOutput;

    /// Creates a new accumulator.
    fn new() -> Self;

    /// Returns the commitments in `self.instances` as a slice.
    fn commitments(&self) -> Vec<PCS::Commitment>;

    /// Returns the points in `self.instances` as a slice.
    fn points(&self) -> Vec<PCS::Point>;

    /// Returns the evaluations in `self.instances` as a slice.
    fn evaluations(&self) -> Vec<PCS::Evaluation>;

    /// Returns the witness polynomials in `self.witnesses` as a slice.
    fn polynomials(&self) -> Vec<PCS::Polynomial>;

    /// Given a polynomial, a commitment to the polynomial, a point, and a value, this function stores them for later use.
    fn push(
        &mut self,
        poly: PCS::Polynomial,
        comm: PCS::Commitment,
        point: PCS::Point,
        value: PCS::Evaluation,
    );

    /// Creates a new proof from a slice of instances and witnesses, also has an optional transcript input.
    fn prove_accumulation(
        &mut self,
        prover_param: &<PCS::SRS as StructuredReferenceString>::ProverParam,
        transcript: Option<&mut RescueTranscript<<PCS::Commitment as AffineRepr>::BaseField>>,
    ) -> Result<(Self::AccProof, Self), PolynomialError>
    where
        <PCS::Commitment as AffineRepr>::BaseField: PrimeField + RescueParameter,
        Self: Sized;

    /// Does the same as `prove_accumulation` but also returns the challenges and scalars
    fn prove_accumulation_with_challenges_and_scalars(
        &mut self,
        prover_param: &<PCS::SRS as StructuredReferenceString>::ProverParam,
        transcript: Option<&mut RescueTranscript<<PCS::Commitment as AffineRepr>::BaseField>>,
    ) -> Result<Self::WithChallengesOutput, PolynomialError>
    where
        <PCS::Commitment as AffineRepr>::BaseField: PrimeField + RescueParameter,
        Self: Sized;

    /// Given old instances and a proof, this verifies that `Self` is a valid accumulation of the old instances.
    fn verify_accumulation(
        &self,
        old_instances: &[Self::Instance],
        proof: &Self::AccProof,
        transcript: Option<&mut RescueTranscript<<PCS::Commitment as AffineRepr>::BaseField>>,
    ) -> Result<(), PolynomialError>
    where
        <PCS::Commitment as AffineRepr>::BaseField: PrimeField + RescueParameter;

    /// Performs a PCS opening on a given `Self::Witness` and returns the opening proof.
    fn open_witness(
        &self,
        prover_param: &<PCS::SRS as StructuredReferenceString>::ProverParam,
    ) -> Result<(PCS::Proof, PCS::Evaluation), PolynomialError>;

    /// Performs a batch opening. Mostly used if you are using the struct to store a list of polynomials in a clean manner.
    fn multi_open(
        &mut self,
        prover_param: &<PCS::SRS as StructuredReferenceString>::ProverParam,
    ) -> Result<PCS::BatchProof, PolynomialError>;

    /// Merges two accumulators into one.
    fn merge_accumulators(&mut self, other: &Self);
}

/// Trait that provides interfaces for an atomic accumulator.
pub trait AtomicAccumulator<PCS: Accumulation>: Accumulator {
    /// The instances to be accumulated.
    type Instance: Clone;
    /// Type of the proof.
    type AccProof: Clone;
    /// The type of `prove_accumulation_with_challenges_and_scalars`'s return value.
    type WithChallengesOutput;

    /// Creates a new accumulator.
    fn new() -> Self;

    /// Given a commitment, point, evaluation and an opening proof, this function stores them for later use.
    fn push(
        &mut self,
        comm: PCS::Commitment,
        point: PCS::Point,
        value: PCS::Evaluation,
        opening_proof: PCS::Proof,
    );

    /// Creates a new proof, also has an optional transcript input.
    fn prove_accumulation<R: CryptoRng + Rng + RngCore>(
        &self,
        rng: &mut R,
        prover_param: &<PCS::SRS as StructuredReferenceString>::ProverParam,
        transcript: Option<&mut RescueTranscript<<PCS::Commitment as AffineRepr>::BaseField>>,
    ) -> Result<(Self, Self::AccProof), PlonkError>
    where
        <PCS::Commitment as AffineRepr>::BaseField: PrimeField + RescueParameter,
        Self: Sized;

    /// Creates a new proof, also has an optional transcript input.
    fn prove_accumulation_with_challenges_and_scalars<R: CryptoRng + Rng + RngCore>(
        &self,
        rng: &mut R,
        prover_param: &<PCS::SRS as StructuredReferenceString>::ProverParam,
        transcript: Option<&mut RescueTranscript<<PCS::Commitment as AffineRepr>::BaseField>>,
    ) -> Result<Self::WithChallengesOutput, PlonkError>
    where
        <PCS::Commitment as AffineRepr>::BaseField: PrimeField + RescueParameter,
        Self: Sized;

    /// Given the new accumulator and a proof, this verifies that `Self` is a valid accumulation of the old instances and old accumulators.
    fn verify_accumulation(
        &self,
        old_accs: &[Self::AccProof],
        new_acc: &Self::AccProof,
        proof: &Self::AccProof,
        verifier_param: &<PCS::SRS as StructuredReferenceString>::VerifierParam,
        transcript: Option<&mut RescueTranscript<<PCS::Commitment as AffineRepr>::BaseField>>,
    ) -> Result<(), PlonkError>
    where
        <PCS::Commitment as AffineRepr>::BaseField: PrimeField + RescueParameter;

    /// Merges two accumulators into one.
    fn merge_accumulators(&mut self, other: &Self);
}
