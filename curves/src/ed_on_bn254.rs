use ark_ec::{
    models::CurveConfig,
    short_weierstrass::{Affine as SWAffine, Projective as SWProjective, SWCurveConfig},
    twisted_edwards::{
        Affine as TEAffine, MontCurveConfig, Projective as TEProjective, TECurveConfig,
    },
};

pub use ark_ed_on_bn254::{Fq, Fr};
use ark_ff::{Field, MontFp};

pub type BJJTEAffine = TEAffine<BabyJubjub>;
pub type BJJTEProjective = TEProjective<BabyJubjub>;
pub type BJJSWAffine = SWAffine<BabyJubjub>;
pub type BJJSWProjective = SWProjective<BabyJubjub>;

#[derive(Clone, Default, PartialEq, Eq)]
pub struct BabyJubjub;

impl CurveConfig for BabyJubjub {
    type BaseField = Fq;
    type ScalarField = Fr;

    /// COFACTOR = 8
    const COFACTOR: &'static [u64] = &[8];

    /// COFACTOR^(-1) mod r =
    /// 2394026564107420727433200628387514462817212225638746351800188703329891451411
    const COFACTOR_INV: Fr =
        MontFp!("2394026564107420727433200628387514462817212225638746351800188703329891451411");
}

impl TECurveConfig for BabyJubjub {
    /// COEFF_A = 1
    const COEFF_A: Fq = Fq::ONE;

    #[inline(always)]
    fn mul_by_a(elem: Self::BaseField) -> Self::BaseField {
        elem
    }

    /// COEFF_D = 168696/168700 mod q
    ///         = 9706598848417545097372247223557719406784115219466060233080913168975159366771
    const COEFF_D: Fq =
        MontFp!("9706598848417545097372247223557719406784115219466060233080913168975159366771");

    /// AFFINE_GENERATOR_COEFFS = (GENERATOR_X, GENERATOR_Y)
    const GENERATOR: BJJTEAffine = TEAffine::new_unchecked(GENERATOR_X, GENERATOR_Y);

    type MontCurveConfig = BabyJubjub;
}

impl MontCurveConfig for BabyJubjub {
    /// COEFF_A = 168698
    const COEFF_A: Fq = MontFp!("168698");
    /// COEFF_B = 168700
    const COEFF_B: Fq = MontFp!("168700");

    type TECurveConfig = BabyJubjub;
}

impl SWCurveConfig for BabyJubjub {
    /// COEFF_A = 3915561033734670630843635270522714716872400990323396055797168613637673095919 mod q
    const COEFF_A: Self::BaseField =
        MontFp!("3915561033734670630843635270522714716872400990323396055797168613637673095919");
    /// COEFF_B = 4217185138631398382466346491768379401896178114478749112717062407767665636606 mod q
    const COEFF_B: Self::BaseField =
        MontFp!("4217185138631398382466346491768379401896178114478749112717062407767665636606");

    /// AFFINE_GENERATOR_COEFFS = (SW_GENERATOR_X, SW_GENERATOR_Y)
    const GENERATOR: BJJSWAffine = SWAffine::new_unchecked(SW_GENERATOR_X, SW_GENERATOR_Y);
}

/// GENERATOR_X =
/// 19698561148652590122159747500897617769866003486955115824547446575314762165298
pub const GENERATOR_X: Fq =
    MontFp!("19698561148652590122159747500897617769866003486955115824547446575314762165298");

/// GENERATOR_Y =
/// 19298250018296453272277890825869354524455968081175474282777126169995084727839
pub const GENERATOR_Y: Fq =
    MontFp!("19298250018296453272277890825869354524455968081175474282777126169995084727839");

/// SW_GENERATOR_X =
/// 4513000517330448244903653178865560289910339884906555605055646870021619219232
pub const SW_GENERATOR_X: Fq =
    MontFp!("4513000517330448244903653178865560289910339884906555605055646870021619219232");

/// SW_GENERATOR_Y =
/// 12354950672345577792670528317750261467336531611841695810091486319550864339243
pub const SW_GENERATOR_Y: Fq =
    MontFp!("12354950672345577792670528317750261467336531611841695810091486319550864339243");

#[cfg(test)]
mod tests {
    use super::*;
    use ark_algebra_test_templates::*;

    use ark_ec::AffineRepr;

    test_group!(te; BJJTEProjective; te);
    test_group!(sw; BJJSWProjective; sw);

    #[test]
    fn test_sw_is_on_curve() {
        assert!(BJJSWAffine::generator().is_on_curve());
    }
}
