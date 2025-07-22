// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! This module implements solidity transcript.

use super::Transcript;
use crate::{constants::KECCAK256_STATE_SIZE, errors::PlonkError};

use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use ark_std::vec::Vec;
use jf_relation::gadgets::ecc::HasTEForm;
use sha3::{Digest, Keccak256};

/// Transcript with `keccak256` hash function.
///
/// It is almost identical to `RescueTranscript` except using Solidity's
/// `keccak256` for Solidity-friendly protocols.
///
/// It is currently implemented simply as
/// - an append only vector of field elements
/// - a state that is initialized with 0
///
/// We keep appending new elements to the transcript vector,
/// and when a challenge is to be generated,
/// we reset the state with the fresh challenge.
///
/// 1. state: \[F: STATE_SIZE\] = hash(state|transcript)
/// 2. challenge = state\[0\]
/// 3. transcript = vec!\[challenge\]
pub struct SolidityTranscript {
    transcript: Vec<u8>,
    state: [u8; KECCAK256_STATE_SIZE], // 64 bytes state size
}

impl Transcript for SolidityTranscript {
    fn new_transcript(_label: &'static [u8]) -> Self {
        SolidityTranscript {
            transcript: Vec::new(),
            state: [0u8; KECCAK256_STATE_SIZE],
        }
    }

    fn push_message<S: CanonicalSerialize + ?Sized + 'static>(
        &mut self,
        _label: &'static [u8],
        msg: &S,
    ) -> Result<(), PlonkError> {
        // We remove the labels for better efficiency
        let mut writer = Vec::new();
        msg.serialize_uncompressed(&mut writer)?;
        // Reverse the byte order for big-endian
        writer.reverse();
        self.transcript.extend_from_slice(writer.as_slice());
        Ok(())
    }

    fn squeeze_scalar_challenge<E>(
        &mut self,
        _label: &'static [u8],
    ) -> Result<E::ScalarField, PlonkError>
    where
        E: HasTEForm,
        E::BaseField: PrimeField,
    {
        ark_std::println!("squeeze_scalar_challenge in solidity");
        // Concatenate state and transcript with additional bytes
        let input = [self.state.as_ref(), self.transcript.as_ref()].concat();
        ark_std::println!(
            "input in squeeze_scalar_challenge solidity: {:?}",
            input.clone()
        );
        use ethers::types::Bytes;
        ark_std::println!(
            "input in squeeze_scalar_challenge solidity: {:?}",
            Bytes::from(input.clone())
        );
        // Hash the inputs using Keccak256
        let mut hasher = Keccak256::new();
        hasher.update(&input);
        let buf = hasher.finalize();

        // Copy the buffers into the state
        self.state.copy_from_slice(&buf);

        // Clear the transcript
        self.transcript = Vec::new();

        // Generate challenge from state bytes using little-endian order
        let challenge = E::ScalarField::from_be_bytes_mod_order(&buf);
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
        let mut challenges = Vec::with_capacity(number_of_challenges);
        for _ in 0..number_of_challenges {
            challenges.push(self.squeeze_scalar_challenge::<E>(label)?);
        }
        Ok(challenges)
    }

    fn merge(&mut self, other: &Self) -> Result<(), PlonkError> {
        self.transcript.extend_from_slice(other.state.as_slice());
        self.transcript
            .extend_from_slice(other.transcript.as_slice());
        Ok(())
    }
}

#[test]
fn test_solidity_keccak() {
    use hex::FromHex;
    use sha3::{Digest, Keccak256};
    let message = "the quick brown fox jumps over the lazy dog".as_bytes();

    let mut hasher = Keccak256::new();
    hasher.update(message);
    let output = hasher.finalize();

    // test example result yanked from smart contract execution
    assert_eq!(
        output[..],
        <[u8; 32]>::from_hex("865bf05cca7ba26fb8051e8366c6d19e21cadeebe3ee6bfa462b5c72275414ec")
            .unwrap()
    );
}
