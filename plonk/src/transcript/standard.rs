// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! This module is a wrapper of the Merlin transcript.
use super::Transcript as TranscriptTrait;
use crate::errors::PlonkError;

use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use ark_std::vec::Vec;
use jf_relation::gadgets::ecc::HasTEForm;
use jf_utils::to_bytes;
use merlin::Transcript;
/// A wrapper of `merlin::Transcript`.
pub struct StandardTranscript(Transcript);

impl TranscriptTrait for StandardTranscript {
    fn new_transcript(label: &'static [u8]) -> Self {
        Self(Transcript::new(label))
    }

    fn new_with_initial_message<S, E>(msg: &S) -> Result<Self, PlonkError>
    where
        Self: Sized,
        S: CanonicalSerialize + ?Sized + 'static,
        E: HasTEForm,
        E::BaseField: PrimeField,
    {
        let mut transcript = Self::new_transcript(b"");
        transcript.push_message(b"", msg)?;
        Ok(transcript)
    }

    fn push_message<S: CanonicalSerialize + ?Sized + 'static>(
        &mut self,
        label: &'static [u8],
        msg: &S,
    ) -> Result<(), PlonkError> {
        let mut writer = Vec::new();
        msg.serialize_uncompressed(&mut writer)?;
        self.0.append_message(label, writer.as_slice());

        Ok(())
    }

    fn squeeze_scalar_challenge<E>(
        &mut self,
        label: &'static [u8],
    ) -> Result<E::ScalarField, PlonkError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField,
    {
        let mut buf = [0u8; 64];
        self.0.challenge_bytes(label, &mut buf);
        let challenge = E::ScalarField::from_le_bytes_mod_order(&buf);
        self.0.append_message(label, &to_bytes!(&challenge)?);
        Ok(challenge)
    }

    fn squeeze_scalar_challenges<E>(
        &mut self,
        label: &'static [u8],
        number_of_challenges: usize,
    ) -> Result<Vec<E::ScalarField>, PlonkError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField,
    {
        let mut challenges = Vec::new();
        for _ in 0..number_of_challenges {
            challenges.push(self.squeeze_scalar_challenge::<E>(label)?);
        }
        Ok(challenges)
    }

    fn merge(&mut self, other: &Self) -> Result<(), PlonkError> {
        let mut other_clone = other.0.clone();
        let mut buf = [0u8; 64];
        Transcript::challenge_bytes(&mut other_clone, b"merge", &mut buf);
        self.0.append_message(b"merge", &buf);
        Ok(())
    }
}
