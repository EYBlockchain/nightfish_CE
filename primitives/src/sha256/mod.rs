//! Code for performing a Keccak hash.
pub(crate) mod constants;
pub use self::constants::Sha256Params;

/// Error enum for the Sha256 hash function.
#[derive(Debug, Clone, PartialEq)]
pub enum Sha256Error {
    /// Thrown if the user attempts to input a vector whose length is
    /// greater than the predefined threshold.
    InvalidInputs,
}
