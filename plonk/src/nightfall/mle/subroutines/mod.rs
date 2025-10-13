//! This module containes the subroutines that are run during multilinear plonk proving, mostly
//! Sumcheck and its variations.

use crate::{
    errors::PlonkError,
    nightfall::mle::{
        mle_structs::PolynomialError,
        subroutines::sumcheck::Oracle,
        utils::compute_barycentric_weights,
        virtual_polynomial::{PolynomialInfo, VirtualPolynomial},
    },
    transcript::{Transcript, TranscriptVisitor},
};

use self::sumcheck::SumCheck;
use ark_ff::{Field, PrimeField};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, MultilinearExtension, Polynomial};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    cfg_into_iter, cfg_iter_mut, marker::PhantomData, string::ToString, vec, vec::Vec, One, Zero,
};
use itertools::Itertools;

use rayon::prelude::*;

use jf_primitives::rescue::RescueParameter;
use jf_relation::gadgets::{ecc::HasTEForm, EmulationConfig};

pub mod gkr;
pub mod lookupcheck;
pub mod permutationcheck;
pub mod sumcheck;
pub mod zerocheck;

type ExtrapolationAux<F> = (Vec<F>, Vec<F>);
/// Struct used to store internal state of the prover between Sumcheck rounds.
#[allow(dead_code)]
pub struct ProverState<'a, P>
where
    P: HasTEForm,
    P::BaseField: PrimeField,
{
    /// sampled randomness given by the verifier
    pub challenges: Vec<P::ScalarField>,
    /// The current evaluation of the virtual polynomial over the boolean hypercube (with challenges fixed).
    pub eval: P::ScalarField,
    /// The number of evaluations in the current round.
    pub mu: usize,
    /// The list of products from the virtual polynomial
    pub(crate) products: &'a [(P::ScalarField, Vec<usize>)],
    /// The evaluations of mle's used to make a virtual polynomial over the boolean hypercube.
    pub(crate) eval_tables: Vec<Vec<P::ScalarField>>,
    /// points with precomputed barycentric weights for extrapolating smaller
    /// degree uni-polys to `max_degree + 1` evaluations.
    pub(crate) extrapolation_aux: ExtrapolationAux<P::ScalarField>,
}

impl<'a, P> ProverState<'a, P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    /// Constructs a new instance of `ProverState<F>` from a virtual polynomial.
    pub fn new(poly: &'a VirtualPolynomial<P::ScalarField>) -> Result<Self, PolynomialError> {
        let max_degree = poly.max_degree();
        let num_vars: usize = poly.num_vars();
        let eval_tables = poly
            .polys
            .iter()
            .map(|poly| poly.to_evaluations())
            .collect::<Vec<Vec<P::ScalarField>>>();

        let points = (0..max_degree - 1)
            .map(|j| P::ScalarField::from(2u64 + j as u64))
            .collect::<Vec<P::ScalarField>>();

        let weights = compute_barycentric_weights(&points)?;

        let poly_products = &poly.products;

        let eval = cfg_into_iter!(0..1 << num_vars as u32)
            .map(|j| {
                poly_products
                    .iter()
                    .map(|(constant, product)| {
                        product.iter().fold(*constant, |acc, index| {
                            acc * eval_tables[*index][j as usize]
                        })
                    })
                    .sum::<P::ScalarField>()
            })
            .sum::<P::ScalarField>();

        Ok(Self {
            challenges: Vec::new(),
            eval,
            mu: 1usize << (num_vars - 1),
            products: &poly.products,
            eval_tables,
            extrapolation_aux: (points, weights),
        })
    }

    /// This function computes the next round of the Sumcheck protocol.
    pub fn compute_sumcheck_round<T: Transcript>(
        &mut self,
        transcript: &mut T,
    ) -> Result<(PolyOracle<P::ScalarField>, P::ScalarField), PolynomialError> {
        let res = self.compute_r_prime_poly_and_update_eval_tables::<T>(transcript)?;
        Ok(res)
    }

    /// function used to compute the r'_i(X) polynomial for the Sumcheck protocol and update the evaluation tables for the multilinear polynomials.
    pub fn compute_r_prime_poly_and_update_eval_tables<T: Transcript>(
        &mut self,
        transcript: &mut T,
    ) -> Result<(PolyOracle<P::ScalarField>, P::ScalarField), PolynomialError> {
        let mu = self.mu;

        let r_poly: DensePolynomial<P::ScalarField> = cfg_into_iter!((0..mu))
            .map(|j| {
                let j_index = j << 1;
                let eval_polys = self
                    .eval_tables
                    .iter()
                    .map(|table| {
                        DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                            table[j_index],
                            table[j_index + 1] - table[j_index],
                        ])
                    })
                    .collect::<Vec<DensePolynomial<P::ScalarField>>>();
                self.products
                    .iter()
                    .map(|(constant, product)| {
                        product.iter().fold(
                            DensePolynomial::from_coefficients_vec(vec![*constant]),
                            |acc, index| acc.naive_mul(&eval_polys[*index]),
                        )
                    })
                    .fold(DensePolynomial::zero(), |acc, poly| acc + poly)
            })
            .fold(DensePolynomial::zero, |acc, poly| acc + poly)
            .reduce(DensePolynomial::zero, |a, b| a + b);

        transcript
            .push_message(b"r_0_eval", &r_poly[0])
            .map_err(|_| {
                PolynomialError::ParameterError("Could not append to transcript".to_string())
            })?;

        let r_prime_poly = r_prime_poly::<P::ScalarField>(&r_poly);

        let oracle = PolyOracle::<P::ScalarField>::from_poly_and_info(
            &r_prime_poly,
            &self.extrapolation_aux.1,
            &self.extrapolation_aux.0,
        )?;

        oracle.append_to_transcript(transcript).map_err(|_| {
            PolynomialError::ParameterError("Could not append to transcript".to_string())
        })?;

        let challenge = transcript
            .squeeze_scalar_challenge::<P>(b"r_prime_challenge")
            .map_err(|_| PolynomialError::ParameterError("Could not get challenge".to_string()))?;
        self.challenges.push(challenge);

        self.eval = r_poly.evaluate(&challenge);

        cfg_iter_mut!(self.eval_tables).for_each(|eval_table| {
            for b in 0..mu {
                let left = eval_table[b << 1];
                let right = eval_table[(b << 1) + 1];
                eval_table[b] = left + challenge * (right - left);
            }
        });
        self.mu >>= 1;
        Ok((oracle, r_poly[0]))
    }
}

/// Function to compute the numerator of the r'(x) polynomial used in the modified Sumcheck protocol
/// found in `https://eprint.iacr.org/2022/1355.pdf`.
pub fn r_prime_poly<F: PrimeField>(r_poly: &DensePolynomial<F>) -> DensePolynomial<F> {
    if r_poly.degree() > 2 {
        let mut new_coeffs = r_poly.coeffs()[2..]
            .iter()
            .rev()
            .scan(F::zero(), |state, coeff| {
                *state -= coeff;
                Some(*state)
            })
            .collect::<Vec<F>>();
        new_coeffs.reverse();
        DensePolynomial::<F>::from_coefficients_vec(new_coeffs)
    } else {
        let r_1_eval = r_poly.evaluate(&F::one());
        let out_poly = r_poly
            - &DensePolynomial::<F>::from_coefficients_vec(vec![r_poly[0], r_1_eval - r_poly[0]]);

        DensePolynomial::<F>::from_coefficients_vec(vec![out_poly[1]])
    }
}

/// The oracle a prover sends to a verifier during Sumcheck. The set of points we lagrange interpolate over
/// will always be [2, 3, ..., degree + 2].
#[derive(Debug, Clone, CanonicalDeserialize, CanonicalSerialize, Eq, PartialEq, Default)]
pub struct PolyOracle<F: Field> {
    /// The degree + 1 evaluations of the polynomial
    pub evaluations: Vec<F>,
    /// The degree + 1 barycentric weights,
    pub weights: Vec<F>,
}

impl<F: Field> Oracle<F> for PolyOracle<F> {
    type Polynomial = DensePolynomial<F>;
    type Point = F;
    fn from_poly(poly: &Self::Polynomial) -> Result<Self, PolynomialError> {
        let size = poly.degree() + 1;
        let mut values = Vec::with_capacity(size);
        let mut points = Vec::with_capacity(size);
        for i in 2..2 + size {
            values.push(poly.evaluate(&F::from(i as u64)));
            points.push(F::from(i as u64));
        }

        let weights = compute_barycentric_weights(&points)?;
        Ok(Self {
            evaluations: values,
            weights,
        })
    }

    fn from_poly_and_info(
        poly: &Self::Polynomial,
        weights: &[F],
        points: &[F],
    ) -> Result<Self, PolynomialError> {
        let evaluations = points
            .iter()
            .map(|point| poly.evaluate(point))
            .collect::<Vec<F>>();

        Ok(Self {
            evaluations,
            weights: weights.to_vec(),
        })
    }

    fn evaluate(&self, point: &F) -> Result<F, PolynomialError> {
        let products = self
            .weights
            .iter()
            .enumerate()
            .map(|(i, weight)| {
                let i_point = F::from(i as u64 + 2);
                let denom = (*point - i_point).inverse();
                if let Some(val) = denom {
                    Ok(*weight * val)
                } else {
                    Err(PolynomialError::ParameterError(
                        "Denominator in PolyOracle evaluation is zero".to_string(),
                    ))
                }
            })
            .collect::<Result<Vec<F>, PolynomialError>>()?;

        let divisor = products
            .iter()
            .sum::<F>()
            .inverse()
            .ok_or(PolynomialError::UnreachableError)?;
        let numerator = products
            .into_iter()
            .zip(self.evaluations.iter())
            .map(|(a, b)| a * b)
            .sum::<F>();

        Ok(numerator * divisor)
    }
}

impl<SF: PrimeField> TranscriptVisitor for PolyOracle<SF> {
    fn append_to_transcript<T: Transcript>(
        &self,
        transcript: &mut T,
    ) -> Result<(), crate::errors::PlonkError> {
        for evaluation in self.evaluations.iter() {
            transcript.push_message(b"PolyOracle", evaluation)?;
        }
        Ok(())
    }
}

/// Struct used to manage internal state during Sumcheck verification.
pub struct VerifierState<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField,
{
    /// Sampled randomness for each round
    pub challenges: Vec<P::ScalarField>,
    /// Claimed starting evaluation of the polynomial over the boolean hypercube,
    pub eval: P::ScalarField,
}

impl<P> VerifierState<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    /// Initialise a new instance of `VerifierState<F, O>` from a Sumcheck proof.
    pub fn new() -> Self {
        Self {
            challenges: vec![],
            eval: P::ScalarField::default(),
        }
    }

    /// This function computes one round of Sumcheck verification.
    pub fn verify_sumcheck_round<T: Transcript, O>(
        &mut self,
        oracle: &O,
        r_0_eval: P::ScalarField,
        transcript: &mut T,
    ) -> Result<(), PolynomialError>
    where
        O: Oracle<P::ScalarField, Point = P::ScalarField> + TranscriptVisitor,
    {
        transcript
            .push_message(b"r_0_eval", &r_0_eval)
            .map_err(|_| {
                PolynomialError::ParameterError("Could not append oracle to transcript".to_string())
            })?;
        transcript.append_visitor(oracle).map_err(|_| {
            PolynomialError::ParameterError("Could not append oracle to transcript".to_string())
        })?;
        let challenge = transcript
            .squeeze_scalar_challenge::<P>(b"r_prime_challenge")
            .map_err(|_| PolynomialError::ParameterError("Could not get challenge".to_string()))?;
        self.challenges.push(challenge);
        let r_1_eval = self.eval - r_0_eval;
        let r_alpha_eval = oracle.evaluate(&challenge)?;
        self.eval = (r_alpha_eval * (P::ScalarField::one() - challenge) * challenge)
            + ((P::ScalarField::one() - challenge) * r_0_eval)
            + (challenge * r_1_eval);
        Ok(())
    }

    /// This function computes one round of Sumcheck verification,
    /// when the appropriate challenges has been pre-computed.
    pub fn verify_sumcheck_round_with_challenges<
        O: Oracle<P::ScalarField, Point = P::ScalarField>,
    >(
        &mut self,
        oracle: &O,
        r_0_eval: P::ScalarField,
        challenge: &P::ScalarField,
    ) -> Result<(), PolynomialError> {
        self.challenges.push(*challenge);
        let r_1_eval = self.eval - r_0_eval;
        let r_alpha_eval = oracle.evaluate(challenge)?;
        self.eval = (r_alpha_eval * (P::ScalarField::one() - challenge) * challenge)
            + ((P::ScalarField::one() - challenge) * r_0_eval)
            + (*challenge * r_1_eval);
        Ok(())
    }
}

impl<P> Default for VerifierState<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<P, O> From<&SumCheckProof<P::ScalarField, O>> for VerifierState<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
    O: Oracle<P::ScalarField>,
{
    fn from(proof: &SumCheckProof<P::ScalarField, O>) -> Self {
        Self {
            challenges: vec![],
            eval: proof.eval,
        }
    }
}

/// A struct that represents a Sumcheck proof minus the final evaluation check.
#[derive(Debug, Clone, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct SumCheckProof<F: Field, O: Oracle<F>> {
    /// Claimed initial evaluation of the polynomial over the boolean hypercube
    pub eval: F,
    /// Oracles to each of the univariate round polynomials
    pub oracles: Vec<O>,
    /// The evaluations of each r_i(x) at 0
    pub r_0_evals: Vec<F>,
    /// The evaluations of the individual polynomials at the final challenge point
    pub poly_evals: Vec<F>,
    /// The final challenge point
    pub point: Vec<F>,
}

impl<F: Field, O: Oracle<F>> SumCheckProof<F, O> {
    /// Returns a default proof using the supplied `num_vars`
    pub fn default_proof(num_vars: usize) -> Self {
        Self {
            eval: F::default(),
            oracles: vec![O::default(); num_vars],
            r_0_evals: vec![F::default(); num_vars],
            poly_evals: vec![F::default(); num_vars],
            point: vec![F::default(); num_vars],
        }
    }
}

impl<F: Field, O: Oracle<F>> Default for SumCheckProof<F, O> {
    fn default() -> Self {
        Self {
            eval: F::default(),
            oracles: vec![],
            r_0_evals: vec![],
            poly_evals: vec![],
            point: vec![],
        }
    }
}

/// A struct that represents the information we defer until later for a sumcheck.
#[derive(Debug, Clone, Default)]
pub struct DeferredCheck<F: PrimeField> {
    /// The point we evaluate the individual MLEs at.
    pub point: Vec<F>,
    /// The final evaluation of the SumCheck at `point`.
    pub eval: F,
}

impl<F: PrimeField> DeferredCheck<F> {
    /// Construct a new Deferred check from a point and evals.
    pub fn new(point: &[F], eval: F) -> Self {
        Self {
            point: point.to_vec(),
            eval,
        }
    }
}

/// A struct used for performing Sumcheck on virtual polynomials.
pub struct VPSumCheck<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField,
{
    /// Phantom data used so we can have an associated field to the prover.
    _phantom: PhantomData<P>,
}

impl<P> SumCheck<P> for VPSumCheck<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    type Polynomial = VirtualPolynomial<P::ScalarField>;
    type Proof = SumCheckProof<P::ScalarField, PolyOracle<P::ScalarField>>;
    type DeferredCheck = DeferredCheck<P::ScalarField>;

    fn prove<T: Transcript>(
        poly: &Self::Polynomial,
        transcript: &mut T,
    ) -> Result<Self::Proof, PlonkError> {
        if poly.max_degree() == 0 {
            return Err(PlonkError::InvalidParameters(
                "Cannot prove a degree 0 polynomial".to_string(),
            ));
        }
        if poly.num_vars() == 0 {
            return Err(PlonkError::InvalidParameters(
                "Cannot prove an polynomial with no variables".to_string(),
            ));
        }
        let mut prover_state = ProverState::<P>::new(poly)?;
        let eval = prover_state.eval;
        transcript.push_message(b"eval", &eval)?;

        let mut oracles = Vec::new();
        let mut r_0_evals = Vec::new();
        for _ in 0..poly.num_vars() {
            let (oracle, r_0_eval) = prover_state.compute_sumcheck_round(transcript)?;

            oracles.push(oracle);
            r_0_evals.push(r_0_eval);
        }

        Ok(SumCheckProof {
            eval,
            oracles,
            r_0_evals,
            poly_evals: prover_state
                .eval_tables
                .iter()
                .map(|table| table[0])
                .collect::<Vec<P::ScalarField>>(),
            point: prover_state.challenges,
        })
    }

    fn verify<T: Transcript>(
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<DeferredCheck<P::ScalarField>, PlonkError> {
        let mut verifier_state = VerifierState::<P>::from(proof);

        transcript.push_message(b"eval", &proof.eval)?;

        assert_eq!(proof.oracles.len(), proof.point.len());
        for (oracle, r_0_eval) in proof.oracles.iter().zip_eq(proof.r_0_evals.iter()) {
            let size = oracle.evaluations.len();
            // note that any affine transformation of the points does not change the result of the barycentric formula
            // however nonlinear transformations do change the result of the barycentric formula
            // the points are fixed (not chosen by the prover), so there is no need to include them in the transcript
            let points = (0..size)
                .map(|i| P::ScalarField::from(i as u64 + 2))
                .collect::<Vec<_>>();
            let weights = compute_barycentric_weights(&points)?;
            let oracle = PolyOracle::<P::ScalarField> {
                evaluations: oracle.evaluations.clone(),
                weights,
            };
            verifier_state.verify_sumcheck_round(&oracle, *r_0_eval, transcript)?;
        }

        let deferred_check =
            DeferredCheck::<P::ScalarField>::new(&verifier_state.challenges, verifier_state.eval);
        Ok(deferred_check)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use ark_poly::DenseMultilinearExtension;
    use ark_std::{
        rand::{distributions::Uniform, prelude::Distribution},
        sync::Arc,
        test_rng, UniformRand,
    };
    use nf_curves::grumpkin::{
        fields::{Fq, Fr},
        short_weierstrass::SWGrumpkin,
    };

    use crate::transcript::RescueTranscript;

    #[test]
    fn test_poly_oracle() {
        for _ in 0..100 {
            let mut rng = test_rng();
            let poly = DensePolynomial::<Fr>::rand(10, &mut rng);
            let oracle = PolyOracle::<Fr>::from_poly(&poly).unwrap();
            let point = Fr::rand(&mut rng);
            let eval = oracle.evaluate(&point).unwrap();
            assert_eq!(eval, poly.evaluate(&point));
        }
    }

    #[test]
    fn test_full_sumcheck() {
        let mut rng = test_rng();
        for _ in 0..10 {
            let max_degree = usize::rand(&mut rng) % 10 + 2;
            let num_vars = usize::rand(&mut rng) % 10 + 1;
            let mles = (0..20)
                .map(|_| Arc::new(DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng)))
                .collect::<Vec<_>>();

            let range = Uniform::new(0usize, 20);
            let products = (0..10)
                .map(|_| {
                    let mut product = Vec::new();
                    for _ in 0..max_degree {
                        product.push(range.sample(&mut rng));
                    }
                    (Fr::one(), product)
                })
                .collect::<Vec<_>>();
            let virtual_polynomial =
                VirtualPolynomial::<Fr>::new(max_degree, num_vars, mles.clone(), products.clone());
            let eval_tables = mles
                .iter()
                .map(|poly| poly.to_evaluations())
                .collect::<Vec<_>>();
            let eval = (0..2u64.pow(virtual_polynomial.num_vars() as u32))
                .map(|j| {
                    virtual_polynomial
                        .products
                        .iter()
                        .map(|(constant, product)| {
                            product
                                .iter()
                                .map(|index| eval_tables[*index][j as usize])
                                .product::<Fr>()
                                * constant
                        })
                        .sum::<Fr>()
                })
                .sum::<Fr>();
            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let proof =
                VPSumCheck::<SWGrumpkin>::prove(&virtual_polynomial, &mut transcript).unwrap();
            assert_eq!(proof.eval, eval);
            transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let deferred_check = VPSumCheck::<SWGrumpkin>::verify(&proof, &mut transcript).unwrap();

            let evals = mles
                .iter()
                .map(|poly| poly.evaluate(&deferred_check.point).unwrap())
                .collect::<Vec<Fr>>();

            let calc_eval = products
                .iter()
                .map(|(constant, product)| {
                    product.iter().map(|index| evals[*index]).product::<Fr>() * constant
                })
                .sum::<Fr>();
            assert_eq!(calc_eval, deferred_check.eval);
        }
    }
}
