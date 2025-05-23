// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Error types.

use crate::{poseidon::PoseidonError, rescue::errors::RescueError};
use ark_serialize::SerializationError;
use ark_std::string::String;
use displaydoc::Display;

/// A glorified [`bool`] that leverages compile lints to encourage the caller to
/// use the result.
///
/// Intended as the return type for verification of proofs, signatures, etc.
/// Recommended for use in the nested [`Result`] pattern: see <https://sled.rs/errors>.
pub type VerificationResult = Result<(), ()>;

/// A `enum` specifying the possible failure modes of the primitives.
#[derive(Debug, Display)]
pub enum PrimitivesError {
    /// Verify fail (proof, sig), {0} [DEPRACATED: use [`VerificationResult`]]
    VerificationError(String),
    /// Bad parameter in function call, {0}
    ParameterError(String),
    #[rustfmt::skip]
    /// ‼ ️Internal error! Please report to Crypto Team immediately!\nMessage: {0}
    InternalError(String),
    /// Deserialization failed: {0}
    DeserializationError(SerializationError),
    /// Decryption failed: {0}
    FailedDecryption(String),
    /// Rescue Error: {0}
    RescueError(RescueError),
    /// Inconsistent Structure error, {0}
    InconsistentStructureError(String),
    /// Poseidon Error: {0}
    PoseidonError(PoseidonError),
}

impl From<RescueError> for PrimitivesError {
    fn from(e: RescueError) -> Self {
        Self::RescueError(e)
    }
}

impl From<SerializationError> for PrimitivesError {
    fn from(e: SerializationError) -> Self {
        Self::DeserializationError(e)
    }
}

impl From<PoseidonError> for PrimitivesError {
    fn from(e: PoseidonError) -> Self {
        Self::PoseidonError(e)
    }
}

impl ark_std::error::Error for PrimitivesError {}
