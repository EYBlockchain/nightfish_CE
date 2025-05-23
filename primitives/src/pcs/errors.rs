// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Error module.

use super::transcript::TranscriptError;
use crate::errors::PrimitivesError;
use ark_serialize::SerializationError;
use ark_std::string::{String, ToString};
use displaydoc::Display;

/// A `enum` specifying the possible failure modes of the PCS.
#[derive(Display, Debug)]
pub enum PCSError {
    /// Invalid Prover: {0}
    InvalidProver(String),
    /// Invalid Verifier: {0}
    InvalidVerifier(String),
    /// Invalid Proof: {0}
    InvalidProof(String),
    /// Invalid parameters: {0}
    InvalidParameters(String),
    /// An error during (de)serialization: {0}
    SerializationError(SerializationError),
    /// Transcript error {0}
    TranscriptError(TranscriptError),
    /// Error from upstream dependencies: {0}
    UpstreamError(String),
    /// Invalid SRS
    InvalidSRS,
}

impl ark_std::error::Error for PCSError {}

impl From<SerializationError> for PCSError {
    fn from(e: ark_serialize::SerializationError) -> Self {
        Self::SerializationError(e)
    }
}

impl From<TranscriptError> for PCSError {
    fn from(e: TranscriptError) -> Self {
        Self::TranscriptError(e)
    }
}

impl From<PrimitivesError> for PCSError {
    fn from(e: PrimitivesError) -> Self {
        Self::UpstreamError(e.to_string())
    }
}

impl From<arithmetic::ArithErrors> for PCSError {
    fn from(e: arithmetic::ArithErrors) -> Self {
        Self::UpstreamError(e.to_string())
    }
}

impl From<transcript::TranscriptError> for PCSError {
    fn from(e: transcript::TranscriptError) -> Self {
        Self::UpstreamError(e.to_string())
    }
}
