//! This module contains the implementation of Plonk using multlinear extensions.

pub mod mle_structs;
pub mod snark;
pub mod subroutines;
pub(crate) mod utils;
pub mod virtual_polynomial;
pub mod zeromorph;
use ark_ec::{short_weierstrass::Affine, AffineRepr};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::rand::{CryptoRng, RngCore};
use ark_std::{sync::Arc, vec::Vec};
#[cfg(any(test, feature = "test-srs"))]
use jf_primitives::pcs::StructuredReferenceString;
use jf_primitives::{
    pcs::{Accumulation, PolynomialCommitmentScheme},
    rescue::RescueParameter,
};
use jf_relation::{
    gadgets::{ecc::HasTEForm, EmulationConfig},
    Arithmetization,
};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
pub use snark::MLEPlonk;

use crate::proof_system::RecursiveOutput;
use crate::{errors::PlonkError, proof_system::UniversalSNARK, transcript::Transcript};

use self::mle_structs::SAMLEProof;
use self::{
    mle_structs::{MLEProof, MLEProvingKey, MLEVerifyingKey},
    zeromorph::Zeromorph,
};

use super::UnivariateIpaPCS;
/// Type for permorming MLEPlonk with the univariate IPA polynomial commitment scheme and Zeromorph.
pub type ZeromorphMLEPlonk<E> = MLEPlonk<Zeromorph<UnivariateIpaPCS<E>>>;

impl<PCS, P, F> UniversalSNARK<PCS> for MLEPlonk<PCS>
where
    F: PrimeField + RescueParameter,
    PCS: Accumulation<
        Evaluation = P::ScalarField,
        Point = Vec<P::ScalarField>,
        Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
        Commitment = Affine<P>,
    >,

    P: HasTEForm<BaseField = F>,
    P::ScalarField: EmulationConfig<F>,
{
    type Proof = MLEProof<PCS>;

    type RecursiveProof = SAMLEProof<PCS>;

    type ProvingKey = MLEProvingKey<PCS>;

    type VerifyingKey = MLEVerifyingKey<PCS>;

    type UniversalSRS = PCS::SRS;

    type Error = PlonkError;

    #[cfg(any(test, feature = "test-srs"))]
    fn universal_setup_for_testing<R: RngCore + CryptoRng>(
        max_degree: usize,
        rng: &mut R,
    ) -> Result<Self::UniversalSRS, Self::Error> {
        <PCS::SRS as StructuredReferenceString>::gen_srs_for_testing(rng, max_degree)
            .map_err(PlonkError::PCSError)
    }

    fn preprocess<C: Arithmetization<P::ScalarField>>(
        srs: &Self::UniversalSRS,
        circuit: &C,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), Self::Error> {
        Self::preprocess_helper(circuit, srs)
    }

    fn prove<C, R, T>(
        _rng: &mut R,
        circuit: &C,
        prove_key: &Self::ProvingKey,
        _extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<Self::Proof, Self::Error>
    where
        C: Arithmetization<
            <<PCS as PolynomialCommitmentScheme>::Commitment as AffineRepr>::ScalarField,
        >,
        R: ark_std::rand::prelude::CryptoRng + ark_std::rand::prelude::RngCore,
        T: Transcript,
    {
        Self::prove::<_, _, _, T>(circuit, prove_key)
    }

    fn recursive_prove<C, R, T>(
        _rng: &mut R,
        circuit: &C,
        prove_key: &Self::ProvingKey,
        _extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<crate::proof_system::RecursiveOutput<PCS, Self, T>, Self::Error>
    where
        Self: Sized,
        PCS: Accumulation,
        C: Arithmetization<
            <<PCS as PolynomialCommitmentScheme>::Commitment as AffineRepr>::ScalarField,
        >,
        R: CryptoRng + RngCore,
        T: Transcript,
    {
        let (proof, transcript) = Self::sa_prove::<_, _, _, T>(circuit, prove_key)?;
        let pi_hash = circuit.public_input()?[0];
        Ok(RecursiveOutput::new(proof, pi_hash, transcript))
    }

    fn verify<T: Transcript>(
        verify_key: &Self::VerifyingKey,
        public_input: &[<<PCS as PolynomialCommitmentScheme>::Commitment as AffineRepr>::ScalarField],
        proof: &Self::Proof,
        _extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<(), Self::Error> {
        // The rng is not actually used so we just use some test rng.
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        if !Self::verify::<_, _, _, T>(proof, verify_key, public_input, &mut rng)? {
            return Err(PlonkError::WrongProof);
        }
        Ok(())
    }
}
