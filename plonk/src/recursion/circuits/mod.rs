//! This module contains the circuits that are common to all recursive provers.

use ark_bn254::Bn254;

use jf_primitives::pcs::prelude::UnivariateKzgPCS;

use nf_curves::grumpkin::Grumpkin;

use crate::nightfall::{mle::zeromorph::Zeromorph, UnivariateIpaPCS};

pub mod atomic_acc;
pub mod challenges;
pub mod emulated_mle_arithmetic;
pub mod fft_arithmetic;
pub mod mle_arithmetic;
pub mod split_acc;

/// Type alias for Zeromorph with Grumpkin
pub type Zmorph = Zeromorph<UnivariateIpaPCS<Grumpkin>>;

/// Type alias for Univariate KZG with Bn254.
pub type Kzg = UnivariateKzgPCS<Bn254>;
