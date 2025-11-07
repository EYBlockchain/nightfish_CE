// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! This module implements three different types of transcripts that are
//! supported.

pub(crate) mod rescue;
pub(crate) mod solidity;
pub(crate) mod standard;

use ark_serialize::CanonicalSerialize;
use ark_std::vec::Vec;
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{CircuitPoint, HasTEForm, Point},
        EmulatedVariable, EmulationConfig,
    },
    PlonkCircuit, Variable,
};
pub use rescue::RescueTranscript;
pub use solidity::SolidityTranscript;
pub use standard::StandardTranscript;

use crate::errors::PlonkError;
use ark_ec::AffineRepr;
use ark_ff::PrimeField;

/// This trait defines base transcript APIs.
pub trait Transcript {
    /// Create a new plonk transcript.
    fn new_transcript(label: &'static [u8]) -> Self;

    /// Create a new plonk transcript with an initial message.
    fn new_with_initial_message<S, E>(msg: &S) -> Result<Self, PlonkError>
    where
        Self: Sized,
        S: CanonicalSerialize + ?Sized + 'static,
        E: HasTEForm,
        E::BaseField: PrimeField;

    /// Append a serializable message to the transcript.
    /// The message is serialized using `CanonicalSerialize`.
    fn push_message<S: CanonicalSerialize + ?Sized + 'static>(
        &mut self,
        label: &'static [u8],
        msg: &S,
    ) -> Result<(), PlonkError>;

    /// Append an elliptic curve point to the transcript.
    fn append_curve_point<E>(&mut self, label: &'static [u8], point: &E) -> Result<(), PlonkError>
    where
        E: AffineRepr,
        E::Config: HasTEForm,
        E::BaseField: PrimeField,
    {
        let nf_point = Point::<E::BaseField>::from(*point);
        for coord in nf_point.coords().iter() {
            self.push_message(label, coord)?;
        }
        Ok(())
    }

    /// Append a slice of elliptic curve points to the transcript.
    fn append_curve_points<E>(
        &mut self,
        label: &'static [u8],
        points: &[E],
    ) -> Result<(), PlonkError>
    where
        E: AffineRepr,
        E::Config: HasTEForm,
        E::BaseField: PrimeField,
    {
        for point in points.iter() {
            self.append_curve_point(label, point)?;
        }
        Ok(())
    }

    /// Generate a challenge in the scalar field of an elliptic curve
    /// whose base field is the same as the transcript field.
    ///
    /// i.e. ```F = E::BaseField```
    fn squeeze_scalar_challenge<E>(
        &mut self,
        label: &'static [u8],
    ) -> Result<E::ScalarField, PlonkError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField;

    /// Generate a list of challenges in the scalar field of an elliptic curve
    /// whose base field is the same as the transcript field.
    ///
    /// i.e. ```F = E::BaseField```
    fn squeeze_scalar_challenges<E>(
        &mut self,
        label: &'static [u8],
        number_of_challenges: usize,
    ) -> Result<Vec<E::ScalarField>, PlonkError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField;

    /// Append a struct that implements `TranscriptVisitor<Self, F>` to the transcript.
    fn append_visitor<T: TranscriptVisitor>(&mut self, visitor: &T) -> Result<(), PlonkError>
    where
        Self: Sized,
    {
        visitor.append_to_transcript(self)
    }

    /// Merge two transcripts together.
    fn merge(&mut self, other: &Self) -> Result<(), PlonkError>;
}

/// Defines transcript APIs for circuit based transcripts.
pub trait CircuitTranscript<F: PrimeField> {
    /// Create a new circuit transcript.
    fn new_transcript(circuit: &mut PlonkCircuit<F>) -> Self;

    /// Create a new plonk transcript with an initial message.
    fn new_with_initial_message<S, E>(
        msg: &S,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<Self, PlonkError>
    where
        Self: Sized,
        S: CanonicalSerialize + ?Sized + 'static,
        E: HasTEForm,
        E::BaseField: PrimeField;

    /// Append a serializable message to the transcript.
    /// The message is serialized using `CanonicalSerialize`.
    fn push_message<S: CanonicalSerialize + ?Sized + 'static, E>(
        &mut self,
        label: &'static [u8],
        msg: &S,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), PlonkError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField;

    /// Append a variable to the transcript
    fn push_variable(&mut self, var: &Variable) -> Result<(), CircuitError>;

    /// Append an [`EmulatedVariable`] to the transcript, we do this by converting the emulated variable to a list of variables and appending them.
    fn push_emulated_variable<E: EmulationConfig<F>>(
        &mut self,
        var: &EmulatedVariable<E>,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError> {
        let var = circuit.convert_for_transcript(var)?;
        self.push_variables(&var)?;
        Ok(())
    }

    /// Appends a list of variables to the transcript
    fn push_variables(&mut self, vars: &[Variable]) -> Result<(), CircuitError> {
        for var in vars.iter() {
            self.push_variable(var)?;
        }
        Ok(())
    }

    /// Appends a `PointVariable` to the transcript
    fn append_point_variable<P: CircuitPoint<F>>(
        &mut self,
        point: &P,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError> {
        let variables = point.prepare_for_transcript(circuit)?;
        self.push_variables(&variables)?;
        Ok(())
    }

    /// Appends a list of `PointVariable`s to the transcript
    fn append_point_variables<P: CircuitPoint<F>>(
        &mut self,
        points: &[P],
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError> {
        for point in points.iter() {
            self.append_point_variable(point, circuit)?;
        }
        Ok(())
    }

    /// Squeezes a challenge in the scalar field of an elliptic curve whose base field is the same as the transcript field.
    fn squeeze_scalar_challenge<E>(
        &mut self,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<Variable, CircuitError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField;

    /// Squeezes multiple scalar challenges in the scalar field of an elliptic curve whose base field is the same as the transcript field.
    fn squeeze_scalar_challenges<E>(
        &mut self,
        number_of_challenges: usize,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<Vec<Variable>, CircuitError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField;

    /// Append a struct that implements `CircuitTranscriptVisitor<Self, F>` to the transcript.
    fn append_visitor<T: CircuitTranscriptVisitor<Self, F>>(
        &mut self,
        visitor: &T,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError>
    where
        Self: Sized,
    {
        visitor.append_to_transcript(self, circuit)
    }

    /// Merge two transcripts together.
    fn merge(&mut self, other: &Self) -> Result<(), CircuitError>;
}

/// This trait is used to define a visitor pattern for the transcript `T`.
/// Implementing this trait on a struct `S` allows us to append it to the transcript `T`
/// meaning that we can define this on a per struct basis rather than on a per transcript basis.
pub trait TranscriptVisitor {
    /// Visit the transcript `T`.
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) -> Result<(), PlonkError>;
}

/// This trait is used to define a visitor pattern for the circuit transcript `T`.
/// Implementing this trait on a struct `S` allows us to append it to the circuit transcript `T`
/// meaning that we can define this on a per struct basis rather than on a per transcript basis.
pub trait CircuitTranscriptVisitor<T, F>
where
    F: PrimeField,
    T: CircuitTranscript<F>,
{
    /// Visit the circuit transcript `T`.
    fn append_to_transcript(
        &self,
        transcript: &mut T,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError>;
}
