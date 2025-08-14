//! GKR subroutines for the MLE protocol.
//!
//! This module contains the implementation of the GKR protocol for verifying polynomial evaluations
//! in the context of the MLE protocol. It includes structures for circuit layers, structured circuits,
//! GKR proofs, and deferred checks, as well as functions for proving and verifying GKR claims.
//!
//! The GKR protocol is a recursive proof system that allows for efficient verification of polynomial
//! evaluations by reducing the problem to a series of sum checks.
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    errors::PlonkError,
    nightfall::mle::{
        mle_structs::PolynomialError,
        subroutines::{sumcheck::SumCheck, VPSumCheck},
        utils::{build_eq_x_r, eq_eval},
        virtual_polynomial::VirtualPolynomial,
    },
    transcript::Transcript,
};

use ark_ff::{batch_inversion, Field, PrimeField};
use ark_poly::evaluations::multivariate::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{
    cfg_chunks, cfg_chunks_mut, cfg_iter, cfg_iter_mut, string::ToString, sync::Arc, vec, vec::Vec,
    Zero,
};

use itertools::Itertools;
use jf_primitives::rescue::RescueParameter;
use jf_relation::gadgets::{ecc::HasTEForm, EmulationConfig};
use rayon::prelude::*;

use super::{DeferredCheck, PolyOracle, SumCheckProof};

/// The DenseMultiLinearExtensions whose evaluations form a layer of the GKR circuit.
#[derive(Clone)]
pub struct CircuitLayer<F: Field> {
    pub(crate) p0: Arc<DenseMultilinearExtension<F>>,
    pub(crate) p1: Arc<DenseMultilinearExtension<F>>,
    pub(crate) q0: Arc<DenseMultilinearExtension<F>>,
    pub(crate) q1: Arc<DenseMultilinearExtension<F>>,
}

/// A GKR circuit.
pub struct StructuredCircuit<F: Field> {
    layers: Vec<CircuitLayer<F>>,
}

/// A GKR Proof
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct GKRProof<F: Field> {
    /// The sumcheck proofs for the GKR protocol.
    pub sumcheck_proofs: Vec<SumCheckProof<F, PolyOracle<F>>>,
    /// The evaluations of the polynomials at each layer.
    pub evals: Vec<Vec<F>>,
    /// The challenge point used in the GKR protocol.
    pub challenge_point: Vec<F>,
}

impl<F: Field> GKRProof<F> {
    /// Construct a new proof
    pub fn new(
        sumcheck_proofs: Vec<SumCheckProof<F, PolyOracle<F>>>,
        evals: Vec<Vec<F>>,
        challenge_point: Vec<F>,
    ) -> Self {
        Self {
            sumcheck_proofs,
            evals,
            challenge_point,
        }
    }

    /// Returns the sumcheck proofs as a slice.
    pub fn sumcheck_proofs(&self) -> &[SumCheckProof<F, PolyOracle<F>>] {
        &self.sumcheck_proofs
    }

    /// Returns the evaluations as a slice.
    pub fn evals(&self) -> &[Vec<F>] {
        &self.evals
    }

    /// Returns the challenge point as a slice.
    pub fn challenge_point(&self) -> &[F] {
        &self.challenge_point
    }
}

/// Struct that contains the deferred check from verifying a GKR proof.
/// This is the claimed evaluations of all the polynomials involved in the GKR proof
/// together with the point they are evaluated at.
#[derive(Clone, Debug, Default)]
pub struct GKRDeferredCheck<F: PrimeField> {
    pub(crate) evals: Vec<F>,
    pub(crate) point: Vec<F>,
}

impl<F: PrimeField> GKRDeferredCheck<F> {
    /// Constructs a new instance of the struct.
    pub fn new(evals: Vec<F>, point: Vec<F>) -> Self {
        Self { evals, point }
    }
    /// Returns the evaluations as a slice.
    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    /// Returns the point to evaluate at.
    pub fn point(&self) -> &[F] {
        &self.point
    }
}

impl<F: Field> CircuitLayer<F> {
    ///Returns the number of variables for the MLEs in the layer.
    pub fn num_vars(&self) -> usize {
        self.p0.num_vars()
    }

    /// Creates a new base layer for a Circuit.
    pub fn base(
        p: Arc<DenseMultilinearExtension<F>>,
        q: Arc<DenseMultilinearExtension<F>>,
    ) -> Result<Self, PolynomialError> {
        let num_vars = p.num_vars();
        if num_vars != q.num_vars() {
            return Err(PolynomialError::ParameterError(
                "MLEs must have the same number of variables".to_string(),
            ));
        };
        let p_evals = p.to_evaluations();
        let q_evals = q.to_evaluations();
        let length = 1 << (num_vars - 1);
        Ok(Self {
            p0: Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                num_vars - 1,
                &p_evals[..length],
            )),
            p1: Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                num_vars - 1,
                &p_evals[length..],
            )),
            q0: Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                num_vars - 1,
                &q_evals[..length],
            )),
            q1: Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                num_vars - 1,
                &q_evals[length..],
            )),
        })
    }

    /// Getter function that returns an array containing the MLEs in the layer.
    pub fn mles(&self) -> [&Arc<DenseMultilinearExtension<F>>; 4] {
        [&self.p0, &self.p1, &self.q0, &self.q1]
    }

    /// Moves up a layer in the circuit.
    pub fn next_layer(&self) -> Result<Self, PolynomialError> {
        let num_vars = self.num_vars();
        if num_vars == 0 {
            return Err(PolynomialError::ParameterError(
                "Cannot move up a layer as polynomial has no variables".to_string(),
            ));
        }

        let length = 1 << num_vars;
        let new_length = length >> 1;

        #[cfg(feature = "parallel")]
        let chunk_size = ((length / rayon::current_num_threads()) + 1).next_power_of_two();

        #[cfg(not(feature = "parallel"))]
        let chunk_size = length.next_power_of_two();

        let mles = self.mles();

        let (p0, p1, q0, q1) = (
            &mles[0].evaluations,
            &mles[1].evaluations,
            &mles[2].evaluations,
            &mles[3].evaluations,
        );
        let mut p_evals = vec![F::zero(); length];
        let mut q_evals = vec![F::zero(); length];
        cfg_chunks_mut!(p_evals, chunk_size)
            .zip(cfg_chunks_mut!(q_evals, chunk_size))
            .zip(cfg_chunks!(p0, chunk_size))
            .zip(cfg_chunks!(p1, chunk_size))
            .zip(cfg_chunks!(q0, chunk_size))
            .zip(cfg_chunks!(q1, chunk_size))
            .for_each(|(((((p_evals, q_evals), p0), p1), q0), q1)| {
                cfg_iter_mut!(p_evals)
                    .zip(cfg_iter_mut!(q_evals))
                    .zip(cfg_iter!(p0))
                    .zip(cfg_iter!(p1))
                    .zip(cfg_iter!(q0))
                    .zip(cfg_iter!(q1))
                    .for_each(|(((((p_eval, q_eval), p0), p1), q0), q1)| {
                        *p_eval = *p0 * *q1 + *p1 * *q0;
                        *q_eval = *q0 * *q1;
                    });
            });

        Ok(Self {
            p0: Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                num_vars - 1,
                &p_evals[..new_length],
            )),
            p1: Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                num_vars - 1,
                &p_evals[new_length..],
            )),
            q0: Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                num_vars - 1,
                &q_evals[..new_length],
            )),
            q1: Arc::new(DenseMultilinearExtension::from_evaluations_slice(
                num_vars - 1,
                &q_evals[new_length..],
            )),
        })
    }
}

impl<F: Field> StructuredCircuit<F> {
    /// Creates a new GKR circuit.
    pub fn new(
        p: Arc<DenseMultilinearExtension<F>>,
        q: Arc<DenseMultilinearExtension<F>>,
    ) -> Result<Self, PolynomialError> {
        if p.num_vars() != q.num_vars() {
            return Err(PolynomialError::ParameterError(
                "MLEs must have the same number of variables".to_string(),
            ));
        }

        Ok(Self {
            layers: ark_std::iter::successors(CircuitLayer::base(p, q).ok(), |layer| {
                layer.next_layer().ok()
            })
            .collect::<Vec<_>>(),
        })
    }

    /// Returns the number of variables in the circuit.
    pub fn num_vars(&self) -> usize {
        self.layers[0].num_vars()
    }

    /// Returns the number of layers in the circuit.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the output of the circuit.
    pub fn output(&self) -> Result<(F, F), PolynomialError> {
        let [p0, p1, q0, q1] = self
            .layers
            .last()
            .ok_or(PolynomialError::ParameterError(
                "Circuit has no layers".to_string(),
            ))?
            .mles();
        let p = p0[0] * q1[0] + p1[0] * q0[0];
        let q = q0[0] * q1[0];
        Ok((p, q))
    }

    /// Proves the circuit, returning the output of the circuit and the proof.
    pub fn prove<P, T>(&self, transcript: &mut T) -> Result<GKRProof<P::ScalarField>, PlonkError>
    where
        P: HasTEForm<ScalarField = F>,
        P::BaseField: RescueParameter,
        F: PrimeField + EmulationConfig<P::BaseField>,
        T: Transcript,
    {
        let mles = self
            .layers
            .first()
            .ok_or(PlonkError::PolynomialError(
                PolynomialError::ParameterError("Circuit has no layers".to_string()),
            ))?
            .mles();

        let p = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            mles[0].num_vars() + 1,
            [mles[0].to_evaluations(), mles[1].to_evaluations()].concat(),
        ));

        let q = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            mles[0].num_vars() + 1,
            [mles[2].to_evaluations(), mles[3].to_evaluations()].concat(),
        ));

        batch_prove_gkr::<P, T>(&[p], &[q], transcript)
    }

    /// Verifies a GKR proof for the circuit.
    pub fn verify<P, T>(
        proof: &GKRProof<P::ScalarField>,
        transcript: &mut T,
    ) -> Result<GKRDeferredCheck<F>, PlonkError>
    where
        P: HasTEForm<ScalarField = F>,
        P::BaseField: RescueParameter,
        F: EmulationConfig<P::BaseField>,
        T: Transcript,
    {
        batch_verify_gkr::<P, T>(proof, transcript)
    }
}

/// Prove multiple GKR claims simultaneously.
pub fn batch_prove_gkr<P, T>(
    ps: &[Arc<DenseMultilinearExtension<P::ScalarField>>],
    qs: &[Arc<DenseMultilinearExtension<P::ScalarField>>],
    transcript: &mut T,
) -> Result<GKRProof<P::ScalarField>, PlonkError>
where
    P: HasTEForm,
    P::BaseField: RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
    T: Transcript,
{
    if ps.is_empty() {
        return Err(PlonkError::PolynomialError(
            PolynomialError::ParameterError("No MLEs to prove".to_string()),
        ));
    }

    if ps.len() != qs.len() {
        return Err(PlonkError::PolynomialError(
            PolynomialError::ParameterError("Must have the same number of ps and qs".to_string()),
        ));
    }

    for mle in ps.iter().chain(qs.iter()) {
        if mle.num_vars() != ps[0].num_vars() {
            return Err(PlonkError::PolynomialError(
                PolynomialError::ParameterError(
                    "MLEs must have the same number of variables".to_string(),
                ),
            ));
        }
    }
    let base_layers = ps
        .iter()
        .zip(qs.iter())
        .map(|(p, q)| CircuitLayer::base(p.clone(), q.clone()))
        .collect::<Result<Vec<CircuitLayer<P::ScalarField>>, _>>()?;

    let layers: Vec<Vec<CircuitLayer<P::ScalarField>>> =
        ark_std::iter::successors(Some(base_layers), |layers| {
            layers
                .iter()
                .map(CircuitLayer::next_layer)
                .collect::<Result<Vec<_>, _>>()
                .ok()
        })
        .collect::<Vec<_>>();

    let batch_size = ps.len();
    // Unwrap here is safe as we have checked all the circuits have layers.
    let polys_vec = layers
        .last()
        .unwrap()
        .iter()
        .map(|layer| layer.mles())
        .collect::<Vec<_>>();

    // In every layer push the evaluations of the polys at that level to the proof.
    // Then when verifying the verifier checks that the correct combination is the starting evaluation used in the sumcheck.
    let mut evals = Vec::new();

    let mut first_evals = Vec::new();
    for [p0, p1, q0, q1] in polys_vec.iter() {
        first_evals.push(p0[0]);
        first_evals.push(p1[0]);
        first_evals.push(q0[0]);
        first_evals.push(q1[0]);
    }
    evals.push(first_evals);
    // Prove each layer of the circuit.
    // Start with the last layer.
    let r_0 = transcript.squeeze_scalar_challenge::<P>(b"r_0")?;
    ark_std::println!("r_0: {:?}", r_0);

    let mut challenge_point = vec![r_0];

    let mut sumcheck_proofs = Vec::new();

    // We have to exit out of the loop one layer early because we need the [`challenge_point`] from the second to last layer.
    for layer in layers.iter().rev().skip(1).take(layers.len() - 2) {
        let eq_x_r = Arc::new(build_eq_x_r(&challenge_point));
        let polys = layer
            .iter()
            .flat_map(|layer| layer.mles())
            .cloned()
            .chain([eq_x_r])
            .collect::<Vec<_>>();

        let lambda = transcript.squeeze_scalar_challenge::<P>(b"lambda")?;
        ark_std::println!("lambda: {:?}", lambda);
        let sumcheck_products = sum_check_products(batch_size, lambda);

        let num_vars = layer.first().unwrap().num_vars();

        let vp = VirtualPolynomial::new(3, num_vars, polys.clone(), sumcheck_products);

        let sumcheck_proof = VPSumCheck::<P>::prove(&vp, transcript)?;
        let r = transcript.squeeze_scalar_challenge::<P>(b"r")?;
        ark_std::println!("r: {:?}", r);

        evals.push(sumcheck_proof.poly_evals[..4 * batch_size].to_vec());

        challenge_point = [sumcheck_proof.point.as_slice(), &[r]].concat();

        sumcheck_proofs.push(sumcheck_proof);
    }
    let eq_x_r = Arc::new(build_eq_x_r(&challenge_point));
    let polys = layers
        .first()
        .unwrap()
        .iter()
        .flat_map(|layer| layer.mles())
        .cloned()
        .chain([eq_x_r])
        .collect::<Vec<_>>();

    let lambda = transcript.squeeze_scalar_challenge::<P>(b"lambda")?;
    let sumcheck_products = sum_check_products(batch_size, lambda);

    let num_vars = layers.first().unwrap()[0].num_vars();

    let vp = VirtualPolynomial::new(3, num_vars, polys.clone(), sumcheck_products);

    let sumcheck_proof = VPSumCheck::<P>::prove(&vp, transcript)?;

    let _ = transcript.squeeze_scalar_challenge::<P>(b"r")?;

    evals.push(sumcheck_proof.poly_evals[..4 * batch_size].to_vec());

    sumcheck_proofs.push(sumcheck_proof);

    Ok(GKRProof::new(sumcheck_proofs, evals, challenge_point))
}

/// Verify a batched instance of the GKR protocol.
pub fn batch_verify_gkr<P, T>(
    proof: &GKRProof<P::ScalarField>,
    transcript: &mut T,
) -> Result<GKRDeferredCheck<P::ScalarField>, PlonkError>
where
    P: HasTEForm,
    P::BaseField: RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
    T: Transcript,
{
    if proof.evals.is_empty() {
        return Err(PlonkError::InvalidParameters(
            "No evaluations to verify".to_string(),
        ));
    }

    // Unwrap is safe because we have checked that the proof has evaluations.
    let first_evals = proof.evals().first().unwrap();

    // We check that the sum of all (p0 * q1 + p1 * q0)/(q0 * q1) is zero as we have separated out unrelated checks using a domain separator.
    let mut p_claims = Vec::new();
    let mut q_claims = Vec::new();
    for eval_chunk in first_evals.chunks(4) {
        let p0 = eval_chunk[0];
        let p1 = eval_chunk[1];
        let q0 = eval_chunk[2];
        let q1 = eval_chunk[3];

        p_claims.push(p0 * q1 + p1 * q0);
        q_claims.push(q0 * q1);
    }

    batch_inversion(&mut q_claims);

    let sum = p_claims
        .into_iter()
        .zip(q_claims)
        .map(|(p, q)| p * q)
        .sum::<P::ScalarField>();

    if sum != P::ScalarField::zero() {
        return Err(PlonkError::InvalidParameters(
            "Sum of claims is not zero".to_string(),
        ));
    }

    let mut res = DeferredCheck::default();
    let mut lambda = P::ScalarField::zero();
    let mut r = transcript.squeeze_scalar_challenge::<P>(b"r_0")?;
    ark_std::println!("r_0: {:?}", r);
    let mut sc_eq_eval = P::ScalarField::zero();
    let mut challenge_point = vec![r];
    // Verify each sumcheck proof. We check that the out put of the previous sumcheck proof is consistent with the input to the next using the
    // supplied evaluations.
    for (i, (proof, evals)) in proof
        .sumcheck_proofs()
        .iter()
        .zip(proof.evals().iter())
        .enumerate()
    {
        // If its not the first round check that these evaluations line up with the expected evaluation from the previous round.
        if i != 0 {
            let expected_eval = sum_check_evaluation(evals, lambda) * sc_eq_eval;
            if expected_eval != res.eval {
                return Err(PlonkError::InvalidParameters(
                    "Sumcheck evaluation does not match expected value".to_string(),
                ));
            }
        }

        lambda = transcript.squeeze_scalar_challenge::<P>(b"lambda")?;
        ark_std::println!("lambda: {:?}", lambda);

        // Check that the initial evaluation of the sumcheck is correct.
        let initial_eval = sumcheck_intial_evaluation(evals, lambda, r);
        if proof.eval != initial_eval {
            return Err(PlonkError::InvalidParameters(
                "Initial sumcheck evaluation does not match expected value".to_string(),
            ));
        }

        let deferred_check = VPSumCheck::<P>::verify(proof, transcript)?;
        r = transcript.squeeze_scalar_challenge::<P>(b"r")?;
        ark_std::println!("r_0: {:?}", r);
        sc_eq_eval = eq_eval(&deferred_check.point, &challenge_point)?;
        challenge_point = [deferred_check.point.as_slice(), &[r]].concat();
        res = deferred_check;
    }
    let final_evals = proof.evals().last().unwrap();
    let expected_eval = sum_check_evaluation(final_evals, lambda) * sc_eq_eval;
    if expected_eval != res.eval {
        return Err(PlonkError::InvalidParameters(
            "Sumcheck evaluation does not match expected value".to_string(),
        ));
    }
    // Unwrap is safe because we checked the eval list was non-empty earlier
    Ok(GKRDeferredCheck::new(final_evals.to_vec(), res.point))
}

fn sum_check_products<F: PrimeField>(num_batching: usize, lambda: F) -> Vec<(F, Vec<usize>)> {
    let eq_index = 4 * num_batching;
    (0..4 * num_batching)
        .tuples::<(_, _, _, _)>()
        .enumerate()
        .flat_map(|(i, (p_l, p_r, q_l, q_r))| {
            let lambda_pow = lambda.pow([(2 * i) as u64]);
            [
                (lambda_pow, vec![p_l, q_r, eq_index]),
                (lambda_pow, vec![q_l, p_r, eq_index]),
                (lambda_pow * lambda, vec![q_l, q_r, eq_index]),
            ]
        })
        .collect_vec()
}

pub(crate) fn sum_check_evaluation<F: PrimeField>(evals: &[F], lambda: F) -> F {
    evals
        .chunks(4)
        .enumerate()
        .fold(F::zero(), |acc, (i, chunk)| {
            let p0 = chunk[0];
            let p1 = chunk[1];
            let q0 = chunk[2];
            let q1 = chunk[3];
            let lambda_pow = lambda.pow([(2 * i) as u64]);
            let p_eval = p0 * q1 + p1 * q0;
            let q_eval = q0 * q1;
            acc + p_eval * lambda_pow + q_eval * lambda * lambda_pow
        })
}

pub(crate) fn sumcheck_intial_evaluation<F: PrimeField>(evals: &[F], lambda: F, r: F) -> F {
    evals
        .chunks(4)
        .enumerate()
        .fold(F::zero(), |acc, (i, chunk)| {
            let p0 = chunk[0];
            let p1 = chunk[1];
            let q0 = chunk[2];
            let q1 = chunk[3];
            let lambda_pow = lambda.pow([(2 * i) as u64]);
            let p_eval = p0 + r * (p1 - p0);
            let q_eval = q0 + r * (q1 - q0);
            acc + p_eval * lambda_pow + q_eval * lambda * lambda_pow
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::RescueTranscript;
    #[allow(unused_imports)]
    use ark_bn254::{g1::Config as BnConfig, Fq, Fr};
    use ark_ff::{One, Zero};
    #[test]
    fn test_build_circuit() {
        let mut rng = ark_std::test_rng();
        let num_vars = 3;

        let p = Arc::new(DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng));
        let q = Arc::new(DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng));
        let circuit = StructuredCircuit::new(p.clone(), q.clone()).unwrap();
        let p_evals = p.to_evaluations();
        let q_evals = q.to_evaluations();

        let sum: Fr = p_evals.iter().zip(q_evals.iter()).map(|(a, b)| a / b).sum();
        let (p_out, q_out) = circuit.output().unwrap();

        assert_eq!(p_out / q_out, sum);
    }

    #[test]
    fn test_prove_and_verify() {
        let mut rng = ark_std::test_rng();
        for num_vars in 2usize..16 {
            let one_vec = vec![Fr::one(); 1 << (num_vars - 1)];
            let minus_one_vec = vec![-Fr::one(); 1 << (num_vars - 1)];
            let p = Arc::new(DenseMultilinearExtension::<Fr>::from_evaluations_vec(
                num_vars,
                [one_vec, minus_one_vec].concat(),
            ));
            let q_one =
                DenseMultilinearExtension::<Fr>::rand(num_vars - 1, &mut rng).to_evaluations();
            let mut q_two = q_one.clone();
            q_two.reverse();
            let q = Arc::new(DenseMultilinearExtension::<Fr>::from_evaluations_vec(
                num_vars,
                [q_one, q_two].concat(),
            ));
            let circuit = StructuredCircuit::new(p.clone(), q.clone()).unwrap();

            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let proof = circuit
                .prove::<BnConfig, RescueTranscript<Fq>>(&mut transcript)
                .unwrap();

            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let gkr_deferred_check = StructuredCircuit::verify::<BnConfig, RescueTranscript<Fq>>(
                &proof,
                &mut transcript,
            )
            .unwrap();

            let zero_check_point = [gkr_deferred_check.point(), &[Fr::zero()]].concat();
            let one_check_point = [gkr_deferred_check.point(), &[Fr::one()]].concat();
            let evals = gkr_deferred_check.evals();
            let p0 = p.evaluate(&zero_check_point).unwrap();
            let p1 = p.evaluate(&one_check_point).unwrap();
            let q0 = q.evaluate(&zero_check_point).unwrap();
            let q1 = q.evaluate(&one_check_point).unwrap();

            assert_eq!(p0, evals[0]);
            assert_eq!(p1, evals[1]);
            assert_eq!(q0, evals[2]);
            assert_eq!(q1, evals[3]);
        }
    }

    #[test]
    fn test_batch_prove_and_verify() {
        let mut rng = ark_std::test_rng();
        let batch_size = 3;
        for num_vars in 2usize..16 {
            let ps = (0..batch_size)
                .map(|_| {
                    let one_vec = vec![Fr::one(); 1 << (num_vars - 1)];
                    let minus_one_vec = vec![-Fr::one(); 1 << (num_vars - 1)];
                    Arc::new(DenseMultilinearExtension::<Fr>::from_evaluations_vec(
                        num_vars,
                        [one_vec, minus_one_vec].concat(),
                    ))
                })
                .collect::<Vec<_>>();
            let qs = (0..batch_size)
                .map(|_| {
                    let q_one = DenseMultilinearExtension::<Fr>::rand(num_vars - 1, &mut rng)
                        .to_evaluations();
                    let mut q_two = q_one.clone();
                    q_two.reverse();
                    Arc::new(DenseMultilinearExtension::<Fr>::from_evaluations_vec(
                        num_vars,
                        [q_one, q_two].concat(),
                    ))
                })
                .collect::<Vec<_>>();

            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");

            let proof =
                batch_prove_gkr::<BnConfig, RescueTranscript<Fq>>(&ps, &qs, &mut transcript)
                    .unwrap();

            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let gkr_deferred_check =
                batch_verify_gkr::<BnConfig, RescueTranscript<Fq>>(&proof, &mut transcript)
                    .unwrap();

            let zero_check_point = [gkr_deferred_check.point(), &[Fr::zero()]].concat();
            let one_check_point = [gkr_deferred_check.point(), &[Fr::one()]].concat();

            for ((p, q), evals) in ps
                .iter()
                .zip(qs.iter())
                .zip(gkr_deferred_check.evals().chunks(4))
            {
                let p0 = p.evaluate(&zero_check_point).unwrap();
                let p1 = p.evaluate(&one_check_point).unwrap();
                let q0 = q.evaluate(&zero_check_point).unwrap();
                let q1 = q.evaluate(&one_check_point).unwrap();
                assert_eq!(p0, evals[0]);
                assert_eq!(p1, evals[1]);
                assert_eq!(q0, evals[2]);
                assert_eq!(q1, evals[3]);
            }
        }
    }
}
