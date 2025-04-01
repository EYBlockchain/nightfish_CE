// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Interfaces for Plonk-based constraint systems

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
pub mod constants;
pub mod errors;
pub mod gadgets;
pub mod gates;

pub mod constraint_system;
use ark_ff::PrimeField;
pub use constraint_system::*;

/// Trait used to define the required methods for a hash to be used in a recursive setting.
pub trait RecursionHasher {
    /// The error type for this hasher.
    type Error;
    /// This function defines how public inputs will be hashed in a recursive setting.
    fn hash_public_inputs<F: PrimeField>(public_inputs: &[F]) -> Result<F, Self::Error>;
}
