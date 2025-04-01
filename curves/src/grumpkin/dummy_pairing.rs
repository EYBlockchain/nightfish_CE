//! This file implements a dummy Pairing trait so that the Grumpkin curve can be used
//! with the PCS traits and protocols.

use ark_bn254::Bn254;

use std::{fmt::Debug, marker::PhantomData};

use ark_ec::{
    pairing::{MillerLoopOutput, Pairing, PairingOutput},
    short_weierstrass::{Affine, Projective, SWCurveConfig},
    AffineRepr, CurveConfig, CurveGroup,
};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use derivative::Derivative;

#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(
    Clone(bound = "P: GrumpConfig"),
    Debug(bound = "P: GrumpConfig"),
    PartialEq(bound = "P: GrumpConfig"),
    Eq(bound = "P: GrumpConfig")
)]
pub struct G1Prepared<P: GrumpConfig>(pub Affine<P::G1Config>);

#[derive(Derivative, CanonicalSerialize, CanonicalDeserialize)]
#[derivative(
    Clone(bound = "P: GrumpConfig"),
    Debug(bound = "P: GrumpConfig"),
    PartialEq(bound = "P: GrumpConfig"),
    Eq(bound = "P: GrumpConfig")
)]
pub struct G2Prepared<P: GrumpConfig>(pub Affine<P::G1Config>);

impl<P: GrumpConfig> From<Affine<P::G1Config>> for G1Prepared<P> {
    fn from(item: Affine<P::G1Config>) -> Self {
        Self(item)
    }
}

impl<'a, P: GrumpConfig> From<&'a Affine<P::G1Config>> for G1Prepared<P> {
    fn from(other: &'a Affine<P::G1Config>) -> Self {
        G1Prepared(*other)
    }
}

impl<'a, P: GrumpConfig> From<&'a Projective<P::G1Config>> for G1Prepared<P> {
    fn from(other: &'a Projective<P::G1Config>) -> Self {
        other.into_affine().into()
    }
}

impl<P: GrumpConfig> Default for G1Prepared<P> {
    fn default() -> Self {
        G1Prepared(Affine::<P::G1Config>::generator())
    }
}

impl<P: GrumpConfig> From<Projective<P::G1Config>> for G1Prepared<P> {
    fn from(item: Projective<P::G1Config>) -> Self {
        Self(item.into())
    }
}

impl<P: GrumpConfig> From<Affine<P::G1Config>> for G2Prepared<P> {
    fn from(item: Affine<P::G1Config>) -> Self {
        Self(item)
    }
}

impl<P: GrumpConfig> From<G1Prepared<P>> for Affine<P::G1Config> {
    fn from(value: G1Prepared<P>) -> Self {
        value.0
    }
}

impl<P: GrumpConfig> From<G2Prepared<P>> for Affine<P::G1Config> {
    fn from(value: G2Prepared<P>) -> Self {
        value.0
    }
}

impl<P: GrumpConfig> From<Projective<P::G1Config>> for G2Prepared<P> {
    fn from(item: Projective<P::G1Config>) -> Self {
        Self(item.into())
    }
}

impl<'a, P: GrumpConfig> From<&'a Affine<P::G1Config>> for G2Prepared<P> {
    fn from(other: &'a Affine<P::G1Config>) -> Self {
        G2Prepared(*other)
    }
}

impl<'a, P: GrumpConfig> From<&'a Projective<P::G1Config>> for G2Prepared<P> {
    fn from(other: &'a Projective<P::G1Config>) -> Self {
        other.into_affine().into()
    }
}

impl<P: GrumpConfig> Default for G2Prepared<P> {
    fn default() -> Self {
        G2Prepared(Affine::<P::G1Config>::generator())
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Grump<P: GrumpConfig>(PhantomData<fn() -> P>);
pub trait GrumpConfig: 'static + Sized + Copy + Eq + Debug {
    type Fp: PrimeField + Into<<Self::Fp as PrimeField>::BigInt>;
    type G1Config: SWCurveConfig<BaseField = Self::Fp>;

    fn multi_miller_loop(
        _a: impl IntoIterator<Item = impl Into<G1Prepared<Self>>>,
        _b: impl IntoIterator<Item = impl Into<G2Prepared<Self>>>,
    ) -> MillerLoopOutput<Grump<Self>> {
        MillerLoopOutput::<Grump<Self>>(<Bn254 as Pairing>::TargetField::default())
    }

    fn final_exponentiation(
        _f: MillerLoopOutput<Grump<Self>>,
    ) -> Option<PairingOutput<Grump<Self>>> {
        Some(PairingOutput::<Grump<Self>>(
            <Bn254 as Pairing>::TargetField::default(),
        ))
    }
}

impl<P: GrumpConfig> Pairing for Grump<P> {
    type BaseField = <P::G1Config as CurveConfig>::BaseField;
    type ScalarField = <P::G1Config as CurveConfig>::ScalarField;
    type G1 = Projective<P::G1Config>;
    type G1Affine = Affine<P::G1Config>;
    type G1Prepared = G1Prepared<P>;
    type G2 = Projective<P::G1Config>;
    type G2Affine = Affine<P::G1Config>;
    type G2Prepared = G2Prepared<P>;
    type TargetField = <Bn254 as Pairing>::TargetField;

    fn multi_miller_loop(
        a: impl IntoIterator<Item = impl Into<Self::G1Prepared>>,
        b: impl IntoIterator<Item = impl Into<Self::G2Prepared>>,
    ) -> MillerLoopOutput<Self> {
        P::multi_miller_loop(a, b)
    }

    fn final_exponentiation(mlo: MillerLoopOutput<Self>) -> Option<PairingOutput<Self>> {
        P::final_exponentiation(mlo)
    }
}
