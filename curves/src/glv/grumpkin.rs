/*use super::super::grumpkin::short_weierstrass::SWGrumpkin;
use ark_ec::short_weierstrass::{Affine, Projective};
use ark_ff::{MontFp, PrimeField};

use super::GLVConfig;

impl GLVConfig for SWGrumpkin {
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

    fn endomorphism(p: &Projective<Self>) -> Projective<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
    fn endomorphism_affine(p: &Affine<Self>) -> Affine<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
    // The specific setup in this struct guarantees that the `k1` returned from `scalar_decomposition`
    // is always postive and the `k2` is "nearly" always positive.
}

#[cfg(test)]
/// return the highest non-zero bits of a bit string.
fn get_bits(a: &[bool]) -> u16 {
    let mut res = 256;
    for e in a.iter().rev() {
        if !e {
            res -= 1;
        } else {
            return res;
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq as Fr;
    use ark_ff::BigInteger;
    use ark_std::{test_rng, UniformRand};

    #[test]
    fn test_decomposition() {
        let mut rng = test_rng();
        let lambda: Fr = SWGrumpkin::LAMBDA;
        for _ in 0..100 {
            let scalar = Fr::rand(&mut rng);
            let ((k1_sign, k1), (k2_sign, k2)) = SWGrumpkin::scalar_decomposition(scalar);
            assert!(get_bits(&k1.into_bigint().to_bits_le()) <= 128);
            assert!(get_bits(&k2.into_bigint().to_bits_le()) <= 128);
            assert!(k1_sign);
            let k2 = if k2_sign { k2 } else { -k2 };
            assert_eq!(k1 + k2 * lambda, scalar);
        }
    }
}*/
