use ark_ec::{
    models::{short_weierstrass::SWCurveConfig, CurveConfig},
    short_weierstrass::{Affine, Projective},
};
use ark_ff::{Field, MontFp, Zero};

use super::fields::{Fq, Fr};

pub struct SWGrumpkin;

pub type GrumpAffine = Affine<SWGrumpkin>;
pub type GrumpProjective = Projective<SWGrumpkin>;

impl CurveConfig for SWGrumpkin {
    type BaseField = Fq;
    type ScalarField = Fr;

    /// COFACTOR = 1
    const COFACTOR: &'static [u64] = &[0x1];

    /// COFACTOR_INV = COFACTOR^{-1} mod r = 1
    const COFACTOR_INV: Fr = Fr::ONE;
}

impl SWCurveConfig for SWGrumpkin {
    /// COEFF_A = 0
    const COEFF_A: Fq = Fq::ZERO;

    /// COEFF_B = -17
    const COEFF_B: Fq = MontFp!("-17");

    /// AFFINE_GENERATOR_COEFFS = (G1_GENERATOR_X, G1_GENERATOR_Y)
    const GENERATOR: GrumpAffine = GrumpAffine::new_unchecked(GENERATOR_X, GENERATOR_Y);

    #[inline(always)]
    fn mul_by_a(_: Self::BaseField) -> Self::BaseField {
        Self::BaseField::zero()
    }
}

pub const GENERATOR_X: Fq = MontFp!("1");

pub const GENERATOR_Y: Fq = MontFp!("17631683881184975370165255887551781615748388533673675138860");

#[cfg(test)]
mod tests {
    use super::*;
    use ark_algebra_test_templates::*;

    test_group!(sw; GrumpProjective; sw);
}
