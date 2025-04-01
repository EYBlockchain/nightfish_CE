// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Implementation of ECC related gates

use core::marker::PhantomData;

use crate::{
    constants::{GATE_WIDTH, N_MUL_SELECTORS},
    gadgets::ecc::HasTEForm,
    gates::Gate,
};
use ark_ec::short_weierstrass::SWCurveConfig as SWConfig;
use ark_ff::{Field, PrimeField};
use derivative::Derivative;
#[inline]
pub(crate) fn edwards_coeff_d<P>() -> P::BaseField
where
    P: HasTEForm,
    P::BaseField: PrimeField,
{
    let s = P::BaseField::from(P::S);
    let neg_alpha = P::BaseField::from(P::NEG_ALPHA);
    let beta = P::BaseField::from(P::BETA);
    (P::BaseField::from(3u64) * -neg_alpha * s - P::BaseField::from(2u64)) / (s * beta.square())
}

pub(crate) fn edwards_coeff_a<P>() -> P::BaseField
where
    P: HasTEForm,
    P::BaseField: PrimeField,
{
    let s = P::BaseField::from(P::S);
    let neg_alpha = P::BaseField::from(P::NEG_ALPHA);
    let beta = P::BaseField::from(P::BETA);
    (P::BaseField::from(3u64) * -neg_alpha * s + P::BaseField::from(2u64)) / (s * beta.square())
}

/// A gate for checking a point conforming the twisted Edwards curve equation
#[derive(Derivative)]
#[derivative(Clone(bound = "P: SWConfig"))]
pub struct EdwardsCurveEquationGate<P: SWConfig> {
    pub(crate) _phantom: PhantomData<P>,
}
#[derive(Derivative)]
#[derivative(Clone(bound = "P: SWConfig"))]
/// Gate that stores selector information for SW Curve equations.
pub struct SWCurveEquationGate<P: SWConfig> {
    pub(crate) _phantom: PhantomData<P>,
}

impl<F, P> Gate<F> for EdwardsCurveEquationGate<P>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
{
    fn name(&self) -> &'static str {
        "Curve Equation Gate"
    }
    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        // edwards equation: ax^2 + y^2 =1 + dx^2y^2
        // for ed_on_bn curves, we have a = 1
        // for ed_on_bls curves, we have a = -1
        let a = edwards_coeff_a::<P>();
        [-a, -F::one()]
    }
    fn q_c(&self) -> F {
        F::one()
    }
    fn q_ecc(&self) -> F {
        edwards_coeff_d::<P>()
    }
}
// NOTE: SW equation : y^2 = x^3 + ax + b
// But for bn-curves we have a = 0  and we can rewrite the curve a x^2y^2 = x^5 + bx^2
// Just to avoid more than one gate.

impl<F, P> Gate<F> for SWCurveEquationGate<P>
where
    F: PrimeField,
    P: SWConfig<BaseField = F>,
{
    fn name(&self) -> &'static str {
        "Curve Equation Gate"
    }
    fn q_hash(&self) -> [F; GATE_WIDTH] {
        [-F::one(), F::zero(), F::zero(), F::zero()]
    }
    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        [-P::COEFF_B, F::zero()]
    }
    fn q_ecc(&self) -> F {
        F::one()
    }
}

/// A gate for point addition on x-coordinate between two Curve Points
#[derive(Derivative)]
#[derivative(Clone(bound = "P: SWConfig"))]
pub struct CurvePointXAdditionGate<P: SWConfig> {
    pub(crate) _phantom: PhantomData<P>,
}

impl<F, P> Gate<F> for CurvePointXAdditionGate<P>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
{
    fn name(&self) -> &'static str {
        "Point Addition X-coordinate Gate"
    }
    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        [F::one(), F::one()]
    }
    fn q_o(&self) -> F {
        F::one()
    }
    fn q_ecc(&self) -> F {
        let d = edwards_coeff_d::<P>();
        -d
    }
}

/// A gate for point addition on y-coordinate between two Curve Points
#[derive(Derivative)]
#[derivative(Clone(bound = "P: SWConfig"))]
pub struct CurvePointYAdditionGate<P: SWConfig> {
    pub(crate) _phantom: PhantomData<P>,
}

impl<F, P> Gate<F> for CurvePointYAdditionGate<P>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
{
    fn name(&self) -> &'static str {
        "Point Addition Y-coordinate Gate"
    }
    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        [-edwards_coeff_a::<P>(), F::one()]
    }
    fn q_o(&self) -> F {
        F::one()
    }
    fn q_ecc(&self) -> F {
        edwards_coeff_d::<P>()
    }
}

/// A gate for point addition on x-coordinate between two SW Curve Points
#[derive(Derivative)]
#[derivative(Clone(bound = "P: SWConfig"))]
pub struct SWCurvePointXAdditionGate<P: SWConfig> {
    pub(crate) _phantom: PhantomData<P>,
}

impl<F, P> Gate<F> for SWCurvePointXAdditionGate<P>
where
    F: PrimeField,
    P: SWConfig<BaseField = F>,
{
    fn name(&self) -> &'static str {
        "Point Addition X-coordinate Gate"
    }
    fn q_x(&self) -> F {
        F::one()
    }
    fn q_x2(&self) -> F {
        -F::from(3u8) * P::COEFF_B
    }
    fn q_o(&self) -> F {
        F::one()
    }
}

/// A gate for point addition on x-coordinate between two SW Curve Points
#[derive(Derivative)]
#[derivative(Clone(bound = "P: SWConfig"))]
pub struct SWCurvePointYAdditionGateNew<P: SWConfig> {
    pub(crate) _phantom: PhantomData<P>,
}

impl<F, P> Gate<F> for SWCurvePointYAdditionGateNew<P>
where
    F: PrimeField,
    P: SWConfig<BaseField = F>,
{
    fn name(&self) -> &'static str {
        "New Point Addition Y-coordinate Gate"
    }
    fn q_x(&self) -> F {
        F::from(9u8) * P::COEFF_B
    }
    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        [-F::from(9u8) * P::COEFF_B * P::COEFF_B, F::zero()]
    }
    fn q_o(&self) -> F {
        F::one()
    }
}

/// A gate for point addition on x-coordinate between two SW Curve Points
#[derive(Derivative)]
#[derivative(Clone(bound = "P: SWConfig"))]
pub struct SWCurvePointYAdditionGateNew2<P: SWConfig> {
    pub(crate) _phantom: PhantomData<P>,
}

impl<F, P> Gate<F> for SWCurvePointYAdditionGateNew2<P>
where
    F: PrimeField,
    P: SWConfig<BaseField = F>,
{
    fn name(&self) -> &'static str {
        "New Point Addition Y-coordinate Gate 2"
    }
    fn q_y(&self) -> F {
        F::one()
    }
    fn q_lc(&self) -> [F; GATE_WIDTH] {
        [F::one(), F::zero(), F::zero(), F::zero()]
    }

    fn q_o(&self) -> F {
        F::one()
    }
}

/// A gate for point addition on x-coordinate between two SW Curve Points
#[derive(Derivative)]
#[derivative(Clone(bound = "P: SWConfig"))]
pub struct SWCurvePointYAdditionGate<P: SWConfig> {
    pub(crate) _phantom: PhantomData<P>,
}

impl<F, P> Gate<F> for SWCurvePointYAdditionGate<P>
where
    F: PrimeField,
    P: SWConfig<BaseField = F>,
{
    fn name(&self) -> &'static str {
        "Point Addition Y-coordinate Gate"
    }
    fn q_y(&self) -> F {
        F::one()
    }
    fn q_y2(&self) -> F {
        F::from(9u8) * P::COEFF_B
    }
    fn q_c(&self) -> F {
        -F::from(9u8) * P::COEFF_B * P::COEFF_B
    }
    fn q_o(&self) -> F {
        F::one()
    }
}

/// A point selection gate on x-coordinate for conditional selection among 4
/// point candidates
/// P0 is default neutral point, P1, P2, P3 are public constants
#[derive(Clone)]
pub struct QuaternaryPointSelectXGate<F: PrimeField> {
    pub(crate) x1: F,
    pub(crate) x2: F,
    pub(crate) x3: F,
}

impl<F> Gate<F> for QuaternaryPointSelectXGate<F>
where
    F: PrimeField,
{
    fn name(&self) -> &'static str {
        "4-ary Point Selection X-coordinate Gate"
    }
    fn q_lc(&self) -> [F; GATE_WIDTH] {
        [self.x1, self.x2, F::zero(), F::zero()]
    }
    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        [self.x3 - self.x2 - self.x1, F::zero()]
    }
    fn q_o(&self) -> F {
        F::one()
    }
}

/// A point selection gate on y-coordinate for conditional selection among 4
/// point candidates
/// P0 is default neutral point, P1, P2, P3 are public constants
#[derive(Clone)]
pub struct QuaternaryPointSelectYGate<F: PrimeField> {
    pub(crate) y1: F,
    pub(crate) y2: F,
    pub(crate) y3: F,
}

impl<F> Gate<F> for QuaternaryPointSelectYGate<F>
where
    F: PrimeField,
{
    fn name(&self) -> &'static str {
        "4-ary Point Selection Y-coordinate Gate"
    }
    fn q_lc(&self) -> [F; GATE_WIDTH] {
        [self.y1 - F::one(), self.y2 - F::one(), F::zero(), F::zero()]
    }
    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        [self.y3 - self.y2 - self.y1 + F::one(), F::zero()]
    }
    fn q_c(&self) -> F {
        F::one()
    }
    fn q_o(&self) -> F {
        F::one()
    }
}

/// A point selection gate on z-coordinate for conditional selection among 4
/// point candidates
/// P0 is default neutral point, P1, P2, P3 are public constants
#[derive(Clone)]
pub struct QuaternaryPointSelectZGate<F: PrimeField> {
    pub(crate) z1: F,
    pub(crate) z2: F,
    pub(crate) z3: F,
}

impl<F> Gate<F> for QuaternaryPointSelectZGate<F>
where
    F: PrimeField,
{
    fn name(&self) -> &'static str {
        "4-ary Point Selection Z-coordinate Gate"
    }
    fn q_lc(&self) -> [F; GATE_WIDTH] {
        [self.z1, self.z2, F::zero(), F::zero()]
    }
    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        [self.z3 - self.z2 - self.z1, F::zero()]
    }
    fn q_o(&self) -> F {
        F::one()
    }
}

/// Point doubling gate on x-coordinate
#[derive(Derivative)]
#[derivative(Clone(bound = "P: SWConfig"))]
pub struct PointDoubleXGate<P: SWConfig> {
    pub(crate) _phantom: PhantomData<P>,
}

impl<F, P> Gate<F> for PointDoubleXGate<P>
where
    F: PrimeField,
    P: SWConfig<BaseField = F>,
{
    fn name(&self) -> &'static str {
        "Point Doubling X-coordinate Gate"
    }
    fn q_lc(&self) -> [F; GATE_WIDTH] {
        [F::zero(), F::zero(), P::COEFF_B * F::from(9u8), F::zero()]
    }
    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        [F::from(4u8), -F::one()]
    }
    fn q_o(&self) -> F {
        F::zero()
    }
}

/// Point doubling gate on y-coordinate
#[derive(Derivative)]
#[derivative(Clone(bound = "P: SWConfig"))]
pub struct PointDoubleYGate<P: SWConfig> {
    pub(crate) _phantom: PhantomData<P>,
}

impl<F, P> Gate<F> for PointDoubleYGate<P>
where
    F: PrimeField,
    P: SWConfig<BaseField = F>,
{
    fn name(&self) -> &'static str {
        "Point Doubling Y-coordinate Gate"
    }
    fn q_lc(&self) -> [F; GATE_WIDTH] {
        [F::from(18u8) * P::COEFF_B, F::zero(), F::zero(), F::zero()]
    }
    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        [F::one(), -F::from(8u8)]
    }
    fn q_o(&self) -> F {
        F::one() + F::from(18u8) * P::COEFF_B
            - F::from(8u8)
            - F::from(27u8) * P::COEFF_B * P::COEFF_B
    }

    fn q_c(&self) -> F {
        -F::from(27u8) * P::COEFF_B * P::COEFF_B
    }
}
