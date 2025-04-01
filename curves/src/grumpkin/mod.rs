use self::dummy_pairing::{Grump, GrumpConfig};

pub mod dummy_pairing;
pub mod fields;
pub mod short_weierstrass;

use ark_ec::short_weierstrass::{Affine, Projective};
pub use fields::Fq;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config;

impl GrumpConfig for Config {
    type Fp = Fq;
    type G1Config = short_weierstrass::SWGrumpkin;
}

pub type Grumpkin = Grump<Config>;
pub type G1Affine = Affine<<Config as GrumpConfig>::G1Config>;
pub type G1Projective = Projective<<Config as GrumpConfig>::G1Config>;
pub type G2Affine = Affine<<Config as GrumpConfig>::G1Config>;
pub type G2Projective = Projective<<Config as GrumpConfig>::G1Config>;
