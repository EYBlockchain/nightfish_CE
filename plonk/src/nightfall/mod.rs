// This Module contains all the additions implemented to run an IPA Plonk prover and verifier.

//! This module provides a Plonk IPA prover.

pub mod circuit;
/// Plonk IPA prover modules
pub(crate) mod hops;
pub(crate) mod ipa_prover;
pub(crate) mod ipa_snark;
pub mod ipa_structs;
pub(crate) mod ipa_verifier;
pub use hops::{srs::*, univariate_ipa::*};
pub use ipa_snark::PlonkIpaSnark;
pub mod accumulation;
pub mod mle;

pub use ipa_snark::FFTPlonk;
pub use ipa_verifier::reproduce_transcript;
