//! This module contains the implementation of the ZeroCheck protocol.

use super::{sumcheck::SumCheck, DeferredCheck, PolyOracle, SumCheckProof, VPSumCheck};
use crate::{
    errors::PlonkError,
    nightfall::mle::{
        utils::{build_f_hat, eq_eval},
        virtual_polynomial::PolynomialInfo,
    },
    transcript::Transcript,
};

use ark_ff::PrimeField;

use jf_primitives::rescue::RescueParameter;
use jf_relation::gadgets::{ecc::HasTEForm, EmulationConfig};

/// Trait used for proving a ZeroCheck claim.
pub trait ZeroCheck<P>: SumCheck<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    /// Create a new ZeroCheck proof.
    fn prove<T: Transcript>(
        poly: &Self::Polynomial,
        transcript: &mut T,
    ) -> Result<Self::Proof, PlonkError>;

    /// Verify a ZeroCheck proof.
    fn verify<T: Transcript>(
        proof: &SumCheckProof<P::ScalarField, PolyOracle<P::ScalarField>>,
        transcript: &mut T,
    ) -> Result<<Self as SumCheck<P>>::DeferredCheck, PlonkError>;
}

impl<P> ZeroCheck<P> for VPSumCheck<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    fn prove<T: Transcript>(
        poly: &Self::Polynomial,
        transcript: &mut T,
    ) -> Result<Self::Proof, PlonkError> {
        let length = poly.num_vars();

        let r = transcript.squeeze_scalar_challenges::<P>(b"Zerocheck r", length)?;
        let f_hat = build_f_hat(poly, r.as_ref())?;
        <VPSumCheck<P> as SumCheck<P>>::prove(&f_hat, transcript)
    }

    fn verify<T: Transcript>(
        proof: &SumCheckProof<P::ScalarField, PolyOracle<P::ScalarField>>,
        transcript: &mut T,
    ) -> Result<DeferredCheck<P::ScalarField>, PlonkError> {
        let r = transcript.squeeze_scalar_challenges::<P>(b"Zerocheck r", proof.point.len())?;
        let mut deferred_check = <VPSumCheck<P> as SumCheck<P>>::verify(proof, transcript)?;
        let eq_a_r_eval = eq_eval(&deferred_check.point, &r)?;

        deferred_check.eval /= eq_a_r_eval;

        Ok(deferred_check)
    }
}

#[cfg(test)]
mod test {
    use super::{VPSumCheck, ZeroCheck};
    use crate::{
        errors::PlonkError,
        nightfall::mle::virtual_polynomial::VirtualPolynomial,
        transcript::{RescueTranscript, Transcript},
    };
    use nf_curves::grumpkin::{
        fields::{Fq, Fr},
        short_weierstrass::SWGrumpkin,
    };

    use ark_ff::{One, Zero};
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{sync::Arc, test_rng, vec, vec::Vec, UniformRand};
    #[test]
    fn test_zerocheck() -> Result<(), PlonkError> {
        let mut rng = test_rng();
        for _ in 0..10 {
            let num_vars = 1 + usize::rand(&mut rng) % 9;
            let mut mles = (0..2)
                .map(|_| Arc::new(DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng)))
                .collect::<Vec<_>>();

            let mle_3_evals = mles
                .chunks(2)
                .map(|chunk| {
                    chunk[0]
                        .to_evaluations()
                        .iter()
                        .zip(chunk[1].to_evaluations().iter())
                        .map(|(a, b)| a * b)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mle_3 = Arc::new(DenseMultilinearExtension::<Fr>::from_evaluations_slice(
                num_vars,
                &mle_3_evals[0],
            ));

            mles.push(mle_3);
            let products = vec![(Fr::one(), vec![0, 1]), (-Fr::one(), vec![2])];

            let virtual_polynomial =
                VirtualPolynomial::<Fr>::new(3, num_vars, mles.clone(), products.clone());

            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let proof = <VPSumCheck<SWGrumpkin> as ZeroCheck<SWGrumpkin>>::prove(
                &virtual_polynomial,
                &mut transcript,
            )?;
            assert_eq!(proof.eval, Fr::zero());
            let mut verify_transcript =
                <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let deferred_check = <VPSumCheck<SWGrumpkin> as ZeroCheck<SWGrumpkin>>::verify(
                &proof,
                &mut verify_transcript,
            )?;
            let expected_sum = virtual_polynomial.evaluate(&deferred_check.point)?;

            assert_eq!(expected_sum, deferred_check.eval);
        }

        Ok(())
    }
}
