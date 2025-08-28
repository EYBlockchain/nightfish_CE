#![allow(dead_code, unused_variables)]
use ark_ff::PrimeField;

use ark_ec::{
    pairing::Pairing,
    scalar_mul::fixed_base::FixedBase,
    short_weierstrass::{Affine, SWCurveConfig},
    AffineRepr, CurveGroup,
};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    fmt::Display, fs::File, io::Write, path::Path, string::ToString, vec::Vec, UniformRand,
};

use jf_primitives::pcs::{errors::PCSError, StructuredReferenceString};
use jf_relation::gadgets::ecc::HasTEForm;

use nf_curves::hash_to_curve::{hash_to_curve, CurveError};

use num_bigint::BigUint;

use crate::transcript::{Transcript, TranscriptVisitor};

/// Type alias for the curve used for G! point on a struct that implements [`Pairing`].
pub type BaseCurve<E> = <<E as Pairing>::G1Affine as AffineRepr>::Config;

/// `UniversalParams` are the universal parameters for the IPA scheme.
/// We require '<E::G1 as AffineRepr>::Config: WBConfig' so that
/// the actual curve that G1 points lie on implements hash-to-curve.
#[derive(Debug, Clone, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct UnivariateUniversalIpaParams<E>
where
    E: Pairing,
    <E::G1Affine as AffineRepr>::Config: SWCurveConfig,
{
    /// 'degree' different generators of the prime order group G.
    pub g_bases: Vec<E::G1Affine>,
    /// Separate generator used for blinding.
    pub h: E::G1Affine,
    /// The final generator used to create the random point U.
    pub u: E::G1Affine,
}

impl<E> UnivariateUniversalIpaParams<E>
where
    E: Pairing,
    <E::G1Affine as AffineRepr>::Config: SWCurveConfig,
{
    /// Returns the maximum supported degree
    pub fn max_degree(&self) -> usize {
        self.g_bases.len()
    }
}

impl<E> Display for UnivariateUniversalIpaParams<E>
where
    E: Pairing,
    <E::G1Affine as AffineRepr>::Config: SWCurveConfig,
{
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        write!(
            f,
            "g_bases: {:?}, h: {}, u: {}",
            self.g_bases, self.h, self.u
        )
    }
}

impl<E> Default for UnivariateUniversalIpaParams<E>
where
    E: Pairing,
    <E::G1Affine as AffineRepr>::Config: SWCurveConfig,
{
    fn default() -> Self {
        Self {
            g_bases: Vec::<E::G1Affine>::default(),
            h: E::G1Affine::default(),
            u: E::G1Affine::default(),
        }
    }
}

impl<E> StructuredReferenceString for UnivariateUniversalIpaParams<E>
where
    E: Pairing,
    <E::G1Affine as AffineRepr>::Config: SWCurveConfig,
{
    type ProverParam = UnivariateUniversalIpaParams<E>;
    type VerifierParam = UnivariateUniversalIpaParams<E>;
    type Item = E::G1Affine;

    fn extract_prover_param(&self, supported_size: usize) -> Self::ProverParam {
        Self::ProverParam {
            g_bases: self.g_bases.to_vec(),
            h: self.h,
            u: self.u,
        }
    }

    fn extract_verifier_param(&self, supported_size: usize) -> Self::VerifierParam {
        Self::VerifierParam {
            g_bases: self.g_bases.to_vec(),
            h: self.h,
            u: self.u,
        }
    }

    fn g(vp: &Self::VerifierParam) -> Self::Item {
        vp.g_bases[0]
    }

    fn gen_srs(mnemonic: &str, supported_degree: usize) -> Result<Self, CurveError> {
        let supported_degree = (supported_degree + 1).next_power_of_two();
        let mut big_uint = BigUint::from_bytes_be(mnemonic.as_bytes());
        let mut g_bases = Vec::<E::G1Affine>::new();
        for i in 0..supported_degree {
            let bytes = &big_uint.to_bytes_be();
            g_bases.push(hash_to_curve::<E>(bytes)?);
            big_uint += BigUint::from(1_u8);
        }
        let bytes = &big_uint.to_bytes_be();
        let h = hash_to_curve::<E>(bytes)?;
        big_uint += BigUint::from(1_u8);
        let bytes = &big_uint.to_bytes_be();
        let u = hash_to_curve::<E>(bytes)?;

        Ok(Self { g_bases, h, u })
    }

    fn load_srs_to_file(
        supported_degree: usize,
        file_name: &str,
        mnemonic: &str,
    ) -> Result<(), PCSError> {
        let srs = Self::gen_srs(mnemonic, supported_degree)
            .map_err(|_| PCSError::InvalidParameters("Failed to generate ipa srs".to_string()))?;

        let path = Path::new(file_name);
        let mut file = File::create(path)
            .map_err(|_| PCSError::InvalidParameters("Failed to create file path".to_string()))?;
        let mut compressed_bytes = Vec::new();
        srs.serialize_compressed(&mut compressed_bytes)
            .map_err(PCSError::SerializationError)?;
        file.write_all(&compressed_bytes)
            .map_err(|_| PCSError::InvalidParameters("Failed to write to file".to_string()))?;
        Ok(())
    }

    fn load_srs_from_file_for_testing(
        supported_degree: usize,
        _file: Option<&str>,
    ) -> Result<Self, jf_primitives::pcs::prelude::PCSError> {
        let supported_degree = (supported_degree + 1).next_power_of_two();
        if let Some(path) = _file {
            let mut srs = Self::deserialize_compressed(
                &*ark_std::fs::read(path).expect("Could not read ipa srs"),
            )?;
            if srs.g_bases.len() < supported_degree {
                return Err(PCSError::InvalidParameters(
                    "srs supports too low degree".to_string(),
                ));
            }
            srs.g_bases = srs.g_bases[..supported_degree].to_vec();
            Ok(srs)
        } else {
            let rng = &mut ark_std::test_rng();

            let powers_of_beta = (1..=supported_degree)
                .map(|i| E::ScalarField::from(i as u64))
                .collect::<Vec<E::ScalarField>>();
            let window_size = FixedBase::get_mul_window_size(supported_degree);
            let g = E::G1::rand(rng);
            let scalar_bits = E::ScalarField::MODULUS_BIT_SIZE as usize;

            let g_table = FixedBase::get_window_table(scalar_bits, window_size, g);
            let powers_of_g =
                FixedBase::msm::<E::G1>(scalar_bits, window_size, &g_table, &powers_of_beta);

            let g_bases = E::G1::normalize_batch(&powers_of_g);

            let list_len = g_bases.len();
            let h = (E::G1Affine::generator().into_group()
                * E::ScalarField::from((list_len + 3) as u64))
            .into();
            let u = (E::G1Affine::generator().into_group()
                * E::ScalarField::from((list_len + 4) as u64))
            .into();

            Ok(Self { g_bases, h, u })
        }
    }

    fn trim(
        &self,
        supported_degree: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), jf_primitives::pcs::prelude::PCSError>
    {
        let supported_degree = (supported_degree + 1).next_power_of_two();
        let trimmed = self.g_bases[..supported_degree].to_vec();
        Ok((
            Self {
                g_bases: trimmed.clone(),
                h: self.h,
                u: self.u,
            },
            Self {
                g_bases: trimmed,
                h: self.h,
                u: self.u,
            },
        ))
    }

    fn gen_srs_for_testing<
        R: rand_chacha::rand_core::RngCore + rand_chacha::rand_core::CryptoRng,
    >(
        rng: &mut R,
        supported_degree: usize,
    ) -> Result<Self, PCSError> {
        let rng = &mut ark_std::test_rng();
        let supported_degree = (supported_degree + 1).next_power_of_two();
        let powers_of_beta = (1..=supported_degree)
            .map(|i| E::ScalarField::from(i as u64))
            .collect::<Vec<E::ScalarField>>();
        let window_size = FixedBase::get_mul_window_size(supported_degree);
        let g = E::G1::rand(rng);
        let scalar_bits = E::ScalarField::MODULUS_BIT_SIZE as usize;

        let g_table = FixedBase::get_window_table(scalar_bits, window_size, g);
        let powers_of_g =
            FixedBase::msm::<E::G1>(scalar_bits, window_size, &g_table, &powers_of_beta);

        let g_bases = E::G1::normalize_batch(&powers_of_g);

        let list_len = g_bases.len();
        let h = (E::G1Affine::generator().into_group()
            * E::ScalarField::from((list_len + 3) as u64))
        .into();
        let u = (E::G1Affine::generator().into_group()
            * E::ScalarField::from((list_len + 4) as u64))
        .into();

        Ok(Self { g_bases, h, u })
    }
}

impl<E, P> TranscriptVisitor for UnivariateUniversalIpaParams<E>
where
    E: Pairing<BaseField = P::BaseField, G1Affine = Affine<P>>,
    P: HasTEForm,
    P::BaseField: PrimeField,
{
    fn append_to_transcript<T: Transcript>(
        &self,
        transcript: &mut T,
    ) -> Result<(), crate::errors::PlonkError> {
        transcript.append_curve_points(b"g_base", &self.g_bases)?;
        transcript.append_curve_point(b"h", &self.h)?;
        transcript.append_curve_point(b"u", &self.u)?;
        Ok(())
    }
}
