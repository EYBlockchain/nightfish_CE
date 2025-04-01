//! This module contains the implementation of the PermutationCheck protocol.

use super::{
    gkr::{batch_prove_gkr, batch_verify_gkr, GKRDeferredCheck, GKRProof},
    VPSumCheck,
};
use crate::{
    errors::PlonkError, nightfall::mle::mle_structs::PolynomialError, transcript::Transcript,
};

use ark_poly::DenseMultilinearExtension;
use ark_std::{string::ToString, sync::Arc, vec::Vec};

use jf_primitives::rescue::RescueParameter;
use jf_relation::gadgets::{ecc::HasTEForm, EmulationConfig};

/// Struct used for proving a PermutationCheck claim.
pub trait PermutationCheck<P>
where
    P: HasTEForm,
    P::BaseField: RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    /// Create a new PermutationCheck proof.
    /// fxs is the permutation of gxs under the permutation given by perms
    fn prove<T: Transcript>(
        numerator: Arc<DenseMultilinearExtension<P::ScalarField>>,
        denominator: Arc<DenseMultilinearExtension<P::ScalarField>>,
        transcript: &mut T,
    ) -> Result<GKRProof<P::ScalarField>, PlonkError> {
        batch_prove_gkr::<P, T>(&[numerator], &[denominator], transcript)
    }

    /// Verify a PermutationCheck proof.
    fn verify<T: Transcript>(
        proof: &GKRProof<P::ScalarField>,
        transcript: &mut T,
    ) -> Result<GKRDeferredCheck<P::ScalarField>, PlonkError> {
        batch_verify_gkr::<P, T>(proof, transcript)
    }

    /// Reduce the PermutationCheck to a GKR check
    fn prep_for_gkr(
        pairs: &[[Vec<P::ScalarField>; 2]],
    ) -> Result<[Arc<DenseMultilinearExtension<P::ScalarField>>; 2], PlonkError> {
        if pairs.len() != 2 {
            return Err(PlonkError::PolynomialError(
                PolynomialError::ParameterError(
                    "Must have two pairs for permutation argument".to_string(),
                ),
            ));
        }
        // We can hard index here as we have already checked that pairs is non-empty.
        let num_vars = pairs[0][0].len().ilog2() as usize;
        for pair in pairs.iter() {
            if pair[0].len().ilog2() as usize != num_vars {
                return Err(PlonkError::PolynomialError(
                    PolynomialError::ParameterError(
                        "All polynomials must have the same number of variables".to_string(),
                    ),
                ));
            }
        }

        let p_poly = Arc::new(
            DenseMultilinearExtension::<P::ScalarField>::from_evaluations_vec(
                num_vars + 1,
                [pairs[0][0].as_slice(), pairs[1][0].as_slice()].concat(),
            ),
        );
        let q_poly = Arc::new(
            DenseMultilinearExtension::<P::ScalarField>::from_evaluations_vec(
                num_vars + 1,
                [pairs[0][1].as_slice(), pairs[1][1].as_slice()].concat(),
            ),
        );

        Ok([p_poly, q_poly])
    }
}

impl<P> PermutationCheck<P> for VPSumCheck<P>
where
    P: HasTEForm,
    P::BaseField: RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::nightfall::mle::snark::tests::gen_circuit_for_test;
    use crate::nightfall::mle::utils::build_eq_x_r_vec;
    use crate::transcript::{RescueTranscript, Transcript};
    use ark_bn254::{g1::Config, Fq, Fr};

    use ark_poly::MultilinearExtension;
    use ark_std::Zero;
    use jf_relation::Arithmetization;

    #[test]
    fn test_permutation_check() -> Result<(), PlonkError> {
        for nv in 1..10 {
            let plonk_type = if nv < 5 {
                jf_relation::PlonkType::TurboPlonk
            } else {
                jf_relation::PlonkType::UltraPlonk
            };
            let circuit = gen_circuit_for_test(5, nv, plonk_type, false)?;

            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");

            let gamma = transcript.squeeze_scalar_challenge::<Config>(b"gamma")?;
            let wire_polys = circuit.compute_wire_mles()?;
            let perm_mles = circuit.compute_extended_permutation_mles()?;
            let pairs = circuit.compute_prod_permutation_mles(&wire_polys, &perm_mles, &gamma)?;

            let num_vars = wire_polys[0].num_vars;
            let prepped = VPSumCheck::<Config>::prep_for_gkr(&pairs)?;

            let prove_result = <VPSumCheck<Config> as PermutationCheck<Config>>::prove(
                prepped[0].clone(),
                prepped[1].clone(),
                &mut transcript,
            )?;

            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");

            let _ = transcript.squeeze_scalar_challenge::<Config>(b"gamma")?;
            let gkr_deferred_check = <VPSumCheck<Config> as PermutationCheck<Config>>::verify(
                &prove_result,
                &mut transcript,
            )?;

            let gkr_eval = gkr_deferred_check.evals();
            let point = gkr_deferred_check.point();

            assert_eq!(gkr_eval.len(), 4);

            let gamma_wire_evals = wire_polys
                .iter()
                .map(|p| p.to_evaluations())
                .collect::<Vec<Vec<Fr>>>();
            let perm_evals = perm_mles
                .iter()
                .map(|p| p.to_evaluations())
                .collect::<Vec<Vec<Fr>>>();

            let eq_x_point = build_eq_x_r_vec(point);
            let evals = perm_evals
                .chunks(3)
                .zip(gamma_wire_evals.chunks(3))
                .map(|(perm_chunk, wire_chunk)| {
                    let length = perm_chunk.len();
                    let denominator_vec: Vec<Fr> =
                        wire_chunk
                            .iter()
                            .fold(eq_x_point.clone(), |acc, wire_evals| {
                                acc.iter()
                                    .zip(wire_evals.iter())
                                    .map(|(acc, wire_eval)| *acc * (gamma - *wire_eval))
                                    .collect()
                            });
                    let mut numerator = Vec::with_capacity(1 << num_vars);

                    for (i, denominator) in denominator_vec.iter().enumerate() {
                        let mut sum = Fr::zero();
                        for j in 0..length {
                            sum += perm_chunk[j][i] * *denominator / (gamma - wire_chunk[j][i]);
                        }
                        numerator.push(sum);
                    }

                    [
                        numerator.iter().sum::<Fr>(),
                        denominator_vec.iter().sum::<Fr>(),
                    ]
                })
                .collect::<Vec<[Fr; 2]>>();

            assert_eq!(evals[0][0], gkr_eval[0]);
            assert_eq!(evals[1][0], gkr_eval[1]);
            assert_eq!(evals[0][1], gkr_eval[2]);
            assert_eq!(evals[1][1], gkr_eval[3]);
        }
        Ok(())
    }
}
