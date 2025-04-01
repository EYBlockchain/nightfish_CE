// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Prelude
pub use crate::pcs::{
    errors::PCSError,
    multilinear_kzg::{
        srs::{MultilinearProverParam, MultilinearUniversalParams, MultilinearVerifierParam},
        util::{get_batched_nv, merge_polynomials},
        MultilinearKzgBatchProof, MultilinearKzgPCS, MultilinearKzgProof, MLE,
    },
    univariate_kzg::{
        srs::{UnivariateProverParam, UnivariateUniversalParams, UnivariateVerifierParam},
        UnivariateKzgBatchProof, UnivariateKzgPCS, UnivariateKzgProof,
    },
    PolynomialCommitmentScheme, StructuredReferenceString,
};
use ark_ec::pairing::Pairing;

/// A commitment is an Affine point.
pub type Commitment<E> = <E as Pairing>::G1Affine;
