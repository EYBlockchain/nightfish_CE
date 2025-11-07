//! This module contains the implementation of the Sumcheck protocol.

use crate::{
    errors::PlonkError,
    nightfall::mle::{mle_structs::PolynomialError, virtual_polynomial::PolynomialInfo},
    transcript::Transcript,
};

use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::vec::Vec;
use jf_relation::gadgets::{ecc::HasTEForm, EmulationConfig};

/// A trait that defines what it means to be an oracle to a polynomial.
pub trait Oracle<F: Field>:
    Sized + Clone + Sync + CanonicalDeserialize + CanonicalSerialize + Default
{
    /// The type of polynomial we are providing an oracle to.
    type Polynomial: PolynomialInfo<F>;
    /// The type of the point we are evaluating at (e.g. `F` or `Vec<F>`).
    type Point;
    /// Create a new oracle for a polynomial.
    fn from_poly(poly: &Self::Polynomial) -> Result<Self, PolynomialError>;
    /// Create a new oracle using pre-supplied weights and points.
    fn from_poly_and_info(
        poly: &Self::Polynomial,
        weights: &[F],
        points: &[Self::Point],
    ) -> Result<Self, PolynomialError>;
    /// A function that returns the evaluation of the polynomial at the point `point`
    fn evaluate(&self, point: &Self::Point) -> Result<F, PolynomialError>;
}

/// Struct used for proving a Sumcheck claim.
pub trait SumCheck<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    /// The type of the polynomial that is being proved, either multilinear or virtual.
    type Polynomial: PolynomialInfo<P::ScalarField>;
    /// The type of the final proof.
    type Proof;
    /// The type of the deferred check.
    type DeferredCheck;

    /// Create a new Sumcheck proof.
    fn prove<T: Transcript>(
        poly: &Self::Polynomial,
        transcript: &mut T,
    ) -> Result<Self::Proof, PlonkError>;

    /// Verify a Sumcheck proof.
    fn verify<T: Transcript>(
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<Self::DeferredCheck, PlonkError>;

    /// Recovers the sumcheck challenges from the proof and transcript.
    fn recover_sumcheck_challenges<T: Transcript>(
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<Vec<P::ScalarField>, PlonkError>;
}
