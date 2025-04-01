use ark_ec::{pairing::Pairing, short_weierstrass::SWCurveConfig, AffineRepr};
use ark_ff::PrimeField;
use ark_serialize::CanonicalDeserialize;
use digest::Digest;
use num_bigint::BigUint;
use sha2::Sha256;

/// An `enum` specifying the possible curve failures.
#[derive(Debug)]
pub enum CurveError {
    /// The BaseField is too large
    BaseFieldTooLarge,
}

/// Currently only supports BaseFields up to 256 bits
pub fn hash_to_curve<P>(msg: &[u8]) -> Result<P::G1Affine, CurveError>
where
    P: Pairing,
    <P::G1Affine as AffineRepr>::Config: SWCurveConfig,
{
    if P::BaseField::MODULUS.into() > BigUint::from(1_u8) << 256 {
        return Err(CurveError::BaseFieldTooLarge);
    }
    let mut attempt = 0_u32;
    loop {
        let mut hasher = Sha256::new();
        hasher.update(msg);
        hasher.update(attempt.to_le_bytes());
        let sha_bytes: [u8; 32] = hasher.finalize().into();

        match P::G1Affine::deserialize_compressed(&sha_bytes[..]) {
            Ok(aff_point) => match aff_point.is_zero() {
                false => return Ok(aff_point),
                true => attempt += 1,
            },
            Err(_) => attempt += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grumpkin::Grumpkin;
    use ark_bn254::Bn254;
    use ark_ec::short_weierstrass::Affine;
    use ark_std::{rand::Rng, test_rng};

    fn test_hash_to_curve_helper<P>() -> Result<(), CurveError>
    where
        P: Pairing,
        <P::G1Affine as AffineRepr>::Config: SWCurveConfig,
    {
        let mut rng = test_rng();
        for _ in 0..10 {
            let random_bytes: Vec<u8> = (0..10).map(|_| rng.gen()).collect();
            let point = hash_to_curve::<P>(&random_bytes)?;
            let aff_point = Affine::<<P::G1Affine as AffineRepr>::Config> {
                x: *point.x().unwrap(),
                y: *point.y().unwrap(),
                infinity: false,
            };
            assert!(aff_point.is_on_curve())
        }
        Ok(())
    }

    #[test]
    fn test_hash_to_curve() -> Result<(), CurveError> {
        test_hash_to_curve_helper::<Bn254>()?;
        test_hash_to_curve_helper::<Grumpkin>()?;
        Ok(())
    }
}
