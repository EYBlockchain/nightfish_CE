// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! this file implements the conversion logic for elliptic curve point between
//! - short Weierstrass form
//! - twisted Edwards form
//!
//! Note that the APIs below create no circuits.
//! An entity should either know both the SW and TE form of a
//! point; or know none of the two. There is no need to generate
//! a circuit for arguing secret knowledge of one form while
//! the other form is public. In practice a prover will convert all of the
//! points to the TE form and work on the TE form inside the circuits.

use ark_ec::{
    short_weierstrass::{Affine as SWAffine, Projective as SWProjective, SWCurveConfig as SWParam},
    twisted_edwards::{Affine, Projective, TECurveConfig as Config},
    AffineRepr, CurveGroup, Group,
};
use ark_ff::{BigInteger256, BigInteger384, BigInteger768, MontFp, PrimeField, Zero};
use ark_std::vec::Vec;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Signed;

use crate::gadgets::{from_emulated_field, EmulationConfig, SerializableEmulatedStruct};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// An elliptic curve point in twisted Edwards affine form (x, y).
pub struct TEPoint<F: PrimeField>(pub F, pub F);

impl<F: PrimeField> TEPoint<F> {
    /// Get the x coordinate of the point.
    pub fn get_x(&self) -> F {
        self.0
    }

    /// Get the y coordinate of the point.
    pub fn get_y(&self) -> F {
        self.1
    }

    /// The inverse point for the edward form.
    pub fn inverse(&self) -> Self {
        Self(-self.0, self.1)
    }
}

impl<F, P> From<Projective<P>> for TEPoint<F>
where
    F: PrimeField,
    P: Config<BaseField = F>,
{
    fn from(p: Projective<P>) -> Self {
        let affine_repr = p.into_affine();
        TEPoint(affine_repr.x, affine_repr.y)
    }
}

impl<F, P> From<TEPoint<F>> for Affine<P>
where
    F: PrimeField,
    P: Config<BaseField = F>,
{
    fn from(p: TEPoint<F>) -> Self {
        Self::new(p.0, p.1)
    }
}

impl<F, P> From<TEPoint<F>> for Projective<P>
where
    F: PrimeField,
    P: Config<BaseField = F>,
{
    fn from(p: TEPoint<F>) -> Self {
        let affine_point: Affine<P> = p.into();
        affine_point.into_group()
    }
}

impl<E, F> SerializableEmulatedStruct<F> for TEPoint<E>
where
    E: EmulationConfig<F>,
    F: PrimeField,
{
    fn serialize_to_native_elements(&self) -> Vec<F> {
        let mut result = from_emulated_field(self.0);
        result.extend(from_emulated_field(self.1));
        result
    }
}

impl<F, P> From<Affine<P>> for TEPoint<F>
where
    F: PrimeField,
    P: Config<BaseField = F>,
{
    fn from(p: Affine<P>) -> Self {
        if p.is_zero() {
            // separately treat point of infinity since maliciously constructed Affine
            // could be (0,0,true) which is still a valid infinity point but would result in
            // `Point(0, 0)` which might lead to problems as seen in precedence:
            // https://cryptosubtlety.medium.com/00-8d4adcf4d255
            let inf = Affine::<P>::zero();
            TEPoint(inf.x, inf.y)
        } else {
            TEPoint(p.x, p.y)
        }
    }
}

impl<F, P> From<SWAffine<P>> for TEPoint<F>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
{
    fn from(p: SWAffine<P>) -> Self {
        // this function is only correct for BLS12-377
        // (other curves does not impl an SW form)

        // if p is an infinity point
        // return infinity point
        if p.infinity {
            return Self(F::zero(), F::one());
        }

        // we need to firstly convert this point into
        // TE form, and then build the point

        // safe unwrap
        let s = P::BaseField::from(P::S);
        let neg_alpha = P::BaseField::from(P::NEG_ALPHA);
        let beta = P::BaseField::from(P::BETA);

        // we first transform the Weierstrass point (px, py) to Montgomery point (mx,
        // my) where mx = s * (px - alpha)
        // my = s * py
        let montgomery_x = s * (p.x + neg_alpha);
        let montgomery_y = s * p.y;
        // then we transform the Montgomery point (mx, my) to TE point (ex, ey) where
        // ex = beta * mx / my
        // ey = (mx - 1) / (mx + 1)
        let edwards_x = beta * montgomery_x / montgomery_y;
        let edwards_y = (montgomery_x - F::one()) / (montgomery_x + F::one());

        TEPoint(edwards_x, edwards_y)
    }
}

impl<F, P> From<TEPoint<F>> for SWAffine<P>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
{
    fn from(p: TEPoint<F>) -> Self {
        let x = p.get_x();
        let y = p.get_y();
        if x == F::zero() && y == F::one() {
            return Self {
                x: F::zero(),
                y: F::zero(),
                infinity: true,
            };
        }

        let s = P::BaseField::from(P::S);
        let neg_alpha = P::BaseField::from(P::NEG_ALPHA);
        let beta = P::BaseField::from(P::BETA);

        // Convert back into montgomery form point
        // montgomery_x = (1 + y) / (1 -y)
        // montgomery_y = (1 + y) * beta /(1 - y) * x
        let montgomery_x = (F::one() + y) / (F::one() - y);
        let montgomery_y = (montgomery_x * beta) / x;

        // Convert from Montgomery form to short Weierstrass form
        // sw_x = (mont_x / s) + alpha
        // sw_y = mont_y /s
        let sw_x = montgomery_x / s - neg_alpha;
        let sw_y = montgomery_y / s;

        Self {
            x: sw_x,
            y: sw_y,
            infinity: false,
        }
    }
}

/// A marker trait used to tell us if a curve has a Twisted Edwards form.
pub trait HasTEForm: SWParam
where
    Self::BaseField: PrimeField,
{
    /// Parameter S.
    const S: <Self::BaseField as PrimeField>::BigInt;
    /// Parameter 1/alpha.
    const NEG_ALPHA: <Self::BaseField as PrimeField>::BigInt;
    /// Parameter beta.
    const BETA: <Self::BaseField as PrimeField>::BigInt;

    /// Returns true if the curve has a twisted Edwards form.
    fn has_te_form() -> bool;

    /// Returns true if the curve implements GLVConfig.
    fn has_glv() -> bool;

    /// Constants that are used to calculate `phi(G) := lambda*G`.
    ///
    /// The coefficients of the endomorphism
    const ENDO_COEFFS: &'static [Self::BaseField];

    /// The eigenvalue corresponding to the endomorphism.
    const LAMBDA: Self::ScalarField;

    /// A 4-element vector representing a 2x2 matrix of coefficients the for scalar decomposition, s.t. k-th entry in the vector is at col i, row j in the matrix, with ij = BE binary decomposition of k.
    /// The entries are the LLL-reduced bases.
    /// The determinant of this matrix must equal `ScalarField::characteristic()`.
    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4];

    /// Decomposes a scalar s into k1, k2, s.t. s = k1 + lambda k2,
    fn scalar_decomposition(
        k: Self::ScalarField,
    ) -> ((bool, Self::ScalarField), (bool, Self::ScalarField)) {
        let scalar: BigInt = k.into_bigint().into().into();

        let coeff_bigints: [BigInt; 4] = Self::SCALAR_DECOMP_COEFFS
            .map(|x| BigInt::from_biguint(if x.0 { Sign::Plus } else { Sign::Minus }, x.1.into()));

        let [n11, n12, n21, n22] = coeff_bigints;
        let r = BigInt::from(Self::ScalarField::MODULUS.into());

        // beta = vector([k,0]) * self.curve.N_inv
        // The inverse of N is 1/r * Matrix([[n22, -n12], [-n21, n11]]).
        // so β = (k*n22, -k*n12)/r

        let beta_1 = &scalar * &n11 / &r;
        let beta_2 = &scalar * &n12 / &r;

        // b = vector([int(beta[0]), int(beta[1])]) * self.curve.N
        // b = (β1N11 + β2N21, β1N12 + β2N22) with the signs!
        //   = (b11   + b12  , b21   + b22)   with the signs!

        // b1
        let b11 = &beta_1 * &n11;
        let b12 = &beta_2 * &n21;
        let b1 = b11 + b12;

        // b2
        let b21 = &beta_1 * &n12;
        let b22 = &beta_2 * &n22;
        let b2 = b21 + b22;

        let k1 = &scalar - b1;
        let k1_abs = BigUint::try_from(k1.abs()).unwrap();
        let k1_sign = (k1.sign() == Sign::Plus) || k1.is_zero();
        // k2
        let k2 = -b2;
        let k2_abs = BigUint::try_from(k2.abs()).unwrap();
        let k2_sign = (k2.sign() == Sign::Plus) || k2.is_zero();

        (
            (k1_sign, Self::ScalarField::from(k1_abs)),
            (k2_sign, Self::ScalarField::from(k2_abs)),
        )
    }

    /// Applies the endomorphism to an affine point.
    fn endomorphism_affine(p: &SWAffine<Self>) -> SWAffine<Self>;

    /*/// Multiplies a projective point by a scalar using the GLV method.
    fn glv_mul_projective(p: SWProjective<Self>, k: Self::ScalarField) -> SWProjective<Self> {
        let ((sgn_k1, k1), (sgn_k2, k2)) = Self::scalar_decomposition(k);

        let mut b1 = p;
        let mut b2 = Self::endomorphism(&p);

        if !sgn_k1 {
            b1 = -b1;
        }
        if !sgn_k2 {
            b2 = -b2;
        }

        let b1b2 = b1 + b2;

        let iter_k1 = ark_ff::BitIteratorBE::new(k1.into_bigint());
        let iter_k2 = ark_ff::BitIteratorBE::new(k2.into_bigint());

        let mut res = SWProjective::<Self>::zero();
        let mut skip_zeros = true;
        for pair in iter_k1.zip(iter_k2) {
            if skip_zeros && pair == (false, false) {
                skip_zeros = false;
                continue;
            }
            res.double_in_place();
            match pair {
                (true, false) => res += b1,
                (false, true) => res += b2,
                (true, true) => res += b1b2,
                (false, false) => {},
            }
        }
        res
    }*/

    /*/// Multiplies a point by a scalar using the GLV method.
    fn glv_mul_affine(p: SWAffine<Self>, k: Self::ScalarField) -> SWAffine<Self> {
        let ((sgn_k1, k1), (sgn_k2, k2)) = Self::scalar_decomposition(k);

        let mut b1 = p;
        let mut b2 = Self::endomorphism_affine(&p);

        if !sgn_k1 {
            b1 = -b1;
        }
        if !sgn_k2 {
            b2 = -b2;
        }

        let b1b2 = b1 + b2;

        let iter_k1 = ark_ff::BitIteratorBE::new(k1.into_bigint());
        let iter_k2 = ark_ff::BitIteratorBE::new(k2.into_bigint());

        let mut res = SWProjective::<Self>::zero();
        let mut skip_zeros = true;
        for pair in iter_k1.zip(iter_k2) {
            if skip_zeros && pair == (false, false) {
                skip_zeros = false;
                continue;
            }
            res.double_in_place();
            match pair {
                (true, false) => res += b1,
                (false, true) => res += b2,
                (true, true) => res += b1b2,
                (false, false) => {},
            }
        }
        res.into_affine()
    }*/
}

impl HasTEForm for nf_curves::ed_on_bn254::BabyJubjub {
    // s = 10189023633222963290707194929886294091415157242906428298294512798502806398782149227503530278436336312243746741931
    const S: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([
        0x00000000000292fc,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
    ]);

    // alpha = -1
    const NEG_ALPHA: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([
        0x2e0ea0057b7951ed,
        0x31f7d845bb92ded4,
        0xd3b57417af3fbc64,
        0xc8dcfe60ab886ed,
    ]);

    // beta = 23560188534917577818843641916571445935985386319233886518929971599490231428764380923487987729215299304184915158756
    const BETA: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([
        0x0000000000000001,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
    ]);

    fn has_te_form() -> bool {
        true
    }

    fn has_glv() -> bool {
        false
    }

    // All zero constants as glv not enabled for this curve
    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!("0")];

    const LAMBDA: Self::ScalarField = MontFp!("0");

    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (true, ark_ff::BigInt!("0")),
    ];

    /*fn endomorphism(p: &SWProjective<Self>) -> SWProjective<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }*/
    fn endomorphism_affine(p: &SWAffine<Self>) -> SWAffine<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

impl HasTEForm for ark_bn254::g1::Config {
    const S: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([0, 0, 0, 0]);
    const NEG_ALPHA: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([0, 0, 0, 0]);
    const BETA: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([0, 0, 0, 0]);
    fn has_te_form() -> bool {
        false
    }
    fn has_glv() -> bool {
        true
    }

    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!(
        "21888242871839275220042445260109153167277707414472061641714758635765020556616"
    )];

    const LAMBDA: Self::ScalarField =
        MontFp!("21888242871839275217838484774961031246154997185409878258781734729429964517155");

    // The determinant of this is `-ScalarField::characteristic()` rather than `ScalarField::characteristic()`.
    // This guarantees `scalar_decomposition` outputs "small" `k1` and `k2`.
    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, ark_ff::BigInt!("9931322734385697763")),
        (
            false,
            ark_ff::BigInt!("147946756881789319010696353538189108491"),
        ),
        (
            false,
            ark_ff::BigInt!("147946756881789319000765030803803410728"),
        ),
        (true, ark_ff::BigInt!("9931322734385697763")),
    ];

    /*fn endomorphism(p: &SWProjective<Self>) -> SWProjective<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }*/
    fn endomorphism_affine(p: &SWAffine<Self>) -> SWAffine<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

impl HasTEForm for nf_curves::grumpkin::short_weierstrass::SWGrumpkin {
    const S: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([0, 0, 0, 0]);
    const NEG_ALPHA: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([0, 0, 0, 0]);
    const BETA: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([0, 0, 0, 0]);
    fn has_te_form() -> bool {
        false
    }
    fn has_glv() -> bool {
        true
    }
    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!(
        "21888242871839275217838484774961031246154997185409878258781734729429964517155"
    )];

    const LAMBDA: Self::ScalarField =
        MontFp!("21888242871839275220042445260109153167277707414472061641714758635765020556616");

    // The determinant of this is `-ScalarField::characteristic()` rather than `ScalarField::characteristic()`.
    // This guarantees `scalar_decomposition` outputs "small" `k1` and `k2`.
    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, ark_ff::BigInt!("9931322734385697762")),
        (
            false,
            ark_ff::BigInt!("147946756881789319010696353538189108491"),
        ),
        (
            false,
            ark_ff::BigInt!("147946756881789319000765030803803410729"),
        ),
        (true, ark_ff::BigInt!("9931322734385697762")),
    ];

    /*fn endomorphism(p: &SWProjective<Self>) -> SWProjective<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }*/
    fn endomorphism_affine(p: &SWAffine<Self>) -> SWAffine<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

impl HasTEForm for nf_curves::ed_on_bls_12_377::Ed377Config {
    // s = 10189023633222963290707194929886294091415157242906428298294512798502806398782149227503530278436336312243746741931
    const S: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([
        0x1c5a3725c8170aad,
        0xc6154c2682b8ac23,
        0x3a1fbcec6f5418e5,
        0x9d8f71eec83a44c,
    ]);

    // alpha = -1
    const NEG_ALPHA: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([
        0x58b07ffffffffe09,
        0x7338d254f0000000,
        0xcae6c45f74129000,
        0x63921ca3364371c,
    ]);

    // beta = 23560188534917577818843641916571445935985386319233886518929971599490231428764380923487987729215299304184915158756
    const BETA: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([
        0x0000000000000001,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
    ]);
    fn has_te_form() -> bool {
        true
    }
    fn has_glv() -> bool {
        false
    }

    // All zero constants as glv not enabled for this curve
    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!("0")];

    const LAMBDA: Self::ScalarField = MontFp!("0");

    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (true, ark_ff::BigInt!("0")),
    ];

    /*fn endomorphism(p: &SWProjective<Self>) -> SWProjective<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }*/
    fn endomorphism_affine(p: &SWAffine<Self>) -> SWAffine<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

impl HasTEForm for nf_curves::ed_on_bls_12_381_bandersnatch::BandersnatchConfig {
    // s = 10189023633222963290707194929886294091415157242906428298294512798502806398782149227503530278436336312243746741931
    const S: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([
        0x926c66eb6fa86d15,
        0xbd025b636bd74122,
        0x316b96e5c340cf6a,
        0x384d1c153c878eea,
    ]);

    // alpha = -1
    const NEG_ALPHA: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([
        0x376e57817be87130,
        0xa4ae256e16ae9167,
        0x1aab9359468ffedd,
        0x160d97955a941876,
    ]);

    // beta = 23560188534917577818843641916571445935985386319233886518929971599490231428764380923487987729215299304184915158756
    const BETA: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([
        0x0000000000000001,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
    ]);
    fn has_te_form() -> bool {
        true
    }
    fn has_glv() -> bool {
        false
    }

    // All zero constants as glv not enabled for this curve
    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!("0")];

    const LAMBDA: Self::ScalarField = MontFp!("0");

    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (true, ark_ff::BigInt!("0")),
    ];

    /*fn endomorphism(p: &SWProjective<Self>) -> SWProjective<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }*/
    fn endomorphism_affine(p: &SWAffine<Self>) -> SWAffine<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

impl HasTEForm for ark_bls12_377::g1::Config {
    // s = 10189023633222963290707194929886294091415157242906428298294512798502806398782149227503530278436336312243746741931
    const S: <Self::BaseField as PrimeField>::BigInt = BigInteger384::new([
        0x3401d618f0339eab,
        0x0f793b8504b428d4,
        0x0ff643cca95ccc0d,
        0xd7a504665d66cc8c,
        0x1dc07a44b1eeea84,
        0x10f272020f118a,
    ]);

    // alpha = -1
    const NEG_ALPHA: <Self::BaseField as PrimeField>::BigInt =
        BigInteger384::new([1, 0, 0, 0, 0, 0]);

    // beta = 23560188534917577818843641916571445935985386319233886518929971599490231428764380923487987729215299304184915158756
    const BETA: <Self::BaseField as PrimeField>::BigInt = BigInteger384::new([
        0x450ae9206343e6e4,
        0x7af39509df5027b6,
        0xab82b31405cf8a30,
        0x80d743e1f6c15c7c,
        0x0cec22e650360183,
        0x272fd56ac5c669,
    ]);
    fn has_te_form() -> bool {
        true
    }
    fn has_glv() -> bool {
        false
    }

    // All zero constants as glv not enabled for this curve
    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!("0")];

    const LAMBDA: Self::ScalarField = MontFp!("0");

    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (true, ark_ff::BigInt!("0")),
    ];

    /*fn endomorphism(p: &SWProjective<Self>) -> SWProjective<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }*/
    fn endomorphism_affine(p: &SWAffine<Self>) -> SWAffine<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

impl HasTEForm for ark_bls12_381::g1::Config {
    const S: <Self::BaseField as PrimeField>::BigInt = BigInteger384::new([0, 0, 0, 0, 0, 0]);
    const NEG_ALPHA: <Self::BaseField as PrimeField>::BigInt =
        BigInteger384::new([0, 0, 0, 0, 0, 0]);
    const BETA: <Self::BaseField as PrimeField>::BigInt = BigInteger384::new([0, 0, 0, 0, 0, 0]);
    fn has_te_form() -> bool {
        false
    }
    fn has_glv() -> bool {
        false
    }

    // All zero constants as glv not enabled for this curve
    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!("0")];

    const LAMBDA: Self::ScalarField = MontFp!("0");

    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (true, ark_ff::BigInt!("0")),
    ];

    /*fn endomorphism(p: &SWProjective<Self>) -> SWProjective<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }*/
    fn endomorphism_affine(p: &SWAffine<Self>) -> SWAffine<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

impl HasTEForm for ark_ed_on_bls12_381::EdwardsConfig {
    // s = 10189023633222963290707194929886294091415157242906428298294512798502806398782149227503530278436336312243746741931
    const S: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([
        0xfffffffeffff5ffd,
        0x53bda402fffe5bfe,
        0x3339d80809a1d805,
        0x73eda753299d7d48,
    ]);

    // alpha = -1
    const NEG_ALPHA: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([
        0xaa7ef00631a1f58e,
        0x30f6d81a76c5a323,
        0x4e7c4d040aa1a560,
        0x46309610e469f6f9,
    ]);

    // beta = 23560188534917577818843641916571445935985386319233886518929971599490231428764380923487987729215299304184915158756
    const BETA: <Self::BaseField as PrimeField>::BigInt = BigInteger256::new([
        0x0000000000000001,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
    ]);
    fn has_te_form() -> bool {
        true
    }
    fn has_glv() -> bool {
        false
    }

    // All zero constants as glv not enabled for this curve
    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!("0")];

    const LAMBDA: Self::ScalarField = MontFp!("0");

    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (true, ark_ff::BigInt!("0")),
    ];

    /*fn endomorphism(p: &SWProjective<Self>) -> SWProjective<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }*/
    fn endomorphism_affine(p: &SWAffine<Self>) -> SWAffine<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

impl HasTEForm for ark_bw6_761::g1::Config {
    const S: <Self::BaseField as PrimeField>::BigInt =
        BigInteger768::new([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const NEG_ALPHA: <Self::BaseField as PrimeField>::BigInt =
        BigInteger768::new([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const BETA: <Self::BaseField as PrimeField>::BigInt =
        BigInteger768::new([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    fn has_te_form() -> bool {
        false
    }
    fn has_glv() -> bool {
        false
    }

    // All zero constants as glv not enabled for this curve
    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!("0")];

    const LAMBDA: Self::ScalarField = MontFp!("0");

    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (false, ark_ff::BigInt!("0")),
        (true, ark_ff::BigInt!("0")),
    ];

    /*fn endomorphism(p: &SWProjective<Self>) -> SWProjective<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }*/
    fn endomorphism_affine(p: &SWAffine<Self>) -> SWAffine<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

#[cfg(test)]
mod test {
    use crate::gadgets::ecc::Point;

    use super::*;
    use ark_bls12_377::g1::Config as g1Config377;
    use ark_ec::{
        short_weierstrass::Projective as SWProjective,
        twisted_edwards::{Affine, Projective, TECurveConfig},
        AffineRepr, CurveConfig, CurveGroup,
    };
    use ark_ff::One;
    use ark_std::{UniformRand, Zero};
    use jf_utils::test_rng;
    use nf_curves::{
        ed_on_bls_12_377::Ed377Config, ed_on_bls_12_381_bandersnatch::BandersnatchConfig,
        ed_on_bn254::BabyJubjub,
    };

    // a helper function to check if a point is on the ed curve
    // of bls12-377 G1
    fn is_on_ed_curve<P: TECurveConfig + HasTEForm>(p: &Point<P::BaseField>) -> bool
    where
        <P as CurveConfig>::BaseField: PrimeField,
    {
        // Twisted Edwards curve 2: a * x² + y² = 1 + d * x² * y²
        let a = <P as TECurveConfig>::COEFF_A;
        let d = <P as TECurveConfig>::COEFF_D;

        let x2 = p.get_x() * p.get_x();
        let y2 = p.get_y() * p.get_y();

        let left = a * x2 + y2;
        let right = P::BaseField::one() + d * x2 * y2;

        left == right
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_sw_to_te_conversion() {
        test_sw_to_te_conversion_helper::<g1Config377>();
        test_sw_to_te_conversion_helper::<BabyJubjub>();
        test_sw_to_te_conversion_helper::<Ed377Config>();
        test_sw_to_te_conversion_helper::<BandersnatchConfig>();
    }

    #[test]
    fn test_te_to_sw_conversion() {
        test_te_to_sw_conversion_helper::<g1Config377>();
        test_te_to_sw_conversion_helper::<BabyJubjub>();
        test_te_to_sw_conversion_helper::<Ed377Config>();
        test_te_to_sw_conversion_helper::<BandersnatchConfig>();
    }

    #[allow(non_snake_case)]
    fn test_sw_to_te_conversion_helper<P: HasTEForm + TECurveConfig>()
    where
        <P as CurveConfig>::BaseField: PrimeField,
    {
        let mut rng = test_rng();

        // test generator
        let g1 = SWAffine::<P>::generator();
        let p: Point<P::BaseField> = g1.into();
        assert!(is_on_ed_curve::<P>(&p));

        // test zero point
        let g1 = SWAffine::<P>::zero();
        let p: Point<P::BaseField> = g1.into();
        assert_eq!(p.get_x(), P::BaseField::zero());
        assert_eq!(p.get_y(), P::BaseField::one());
        assert!(is_on_ed_curve::<P>(&p));

        // test a random group element
        let g1 = SWProjective::<P>::rand(&mut rng).into_affine();
        let p: Point<P::BaseField> = g1.into();
        assert!(is_on_ed_curve::<P>(&p));
    }

    fn test_te_to_sw_conversion_helper<P: TECurveConfig + HasTEForm>()
    where
        <P as CurveConfig>::BaseField: PrimeField,
    {
        let mut rng = test_rng();
        // test generator
        let p = Affine::<P>::generator();
        let p = Point::from(p);
        let g1: SWAffine<P> = p.into();
        assert!(g1.is_on_curve());

        // test zero point
        let p = Point::<P::BaseField>::TE(P::BaseField::zero(), P::BaseField::one());
        let g1: SWAffine<P> = p.into();
        assert!(g1.is_zero());

        // test a random group element
        let g1 = Projective::<P>::rand(&mut rng).into_affine();
        let p: Point<P::BaseField> = g1.into();
        let g1: SWAffine<P> = p.into();
        assert!(g1.is_on_curve());
    }
}
