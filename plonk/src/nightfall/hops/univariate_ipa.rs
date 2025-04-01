use ark_ec::{
    pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, AffineRepr, CurveGroup,
};
use jf_relation::gadgets::{ecc::HasTEForm, EmulationConfig};

use ark_ff::{Field, PrimeField};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    fmt::Debug,
    format,
    hash::Hash,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Mul, Neg},
    string::ToString,
    vec,
    vec::Vec,
    One, UniformRand, Zero,
};
use jf_primitives::{
    pcs::{
        prelude::{Commitment, PCSError},
        Accumulation, PolynomialCommitmentScheme, StructuredReferenceString,
    },
    rescue::RescueParameter,
};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use rayon::prelude::*;

use crate::transcript::{RescueTranscript, Transcript};

use super::srs::UnivariateUniversalIpaParams;

use itertools::izip;

/// IPA Polynomial Commitment Scheme using a univariate polynomial
#[derive(Clone, PartialEq, Debug)]
pub struct UnivariateIpaPCS<E: Pairing> {
    #[doc(hidden)]
    phantom: PhantomData<E>,
}

impl<E: Pairing> Default for UnivariateIpaPCS<E> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

#[derive(Debug, CanonicalDeserialize, CanonicalSerialize, Clone, PartialEq, Eq)]
/// proof of opening
pub struct UnivariateIpaProof<E: Pairing> {
    /// The left side of the inner product argument
    pub l_i: Vec<E::G1Affine>,
    /// The right side of the inner product argument
    pub r_i: Vec<E::G1Affine>,
    /// The synthetic blinding factor
    pub f: E::ScalarField,
    /// The collapsed coefficient vector of the committed polynomial.
    pub c: E::ScalarField,
}

impl<E: Pairing> Default for UnivariateIpaProof<E> {
    fn default() -> Self {
        Self {
            l_i: Vec::<E::G1Affine>::default(),
            r_i: Vec::<E::G1Affine>::default(),
            f: E::ScalarField::default(),
            c: E::ScalarField::default(),
        }
    }
}

impl<E: Pairing> Hash for UnivariateIpaProof<E> {
    fn hash<H: ark_std::hash::Hasher>(&self, state: &mut H) {
        for l in self.l_i.iter() {
            l.hash(state);
        }
        for r in self.r_i.iter() {
            r.hash(state);
        }
        self.f.hash(state);
        self.c.hash(state);
    }
}

/// Batch proof
impl<E> PolynomialCommitmentScheme for UnivariateIpaPCS<E>
where
    E: Pairing,
    E::G1Affine: AffineRepr<BaseField = E::BaseField>,
    <E::G1Affine as AffineRepr>::Config: HasTEForm<BaseField = E::BaseField>,
    <E as Pairing>::BaseField: RescueParameter,
    E::ScalarField: EmulationConfig<E::BaseField>,
{
    type SRS = UnivariateUniversalIpaParams<E>;
    type Polynomial = DensePolynomial<E::ScalarField>;
    type Point = E::ScalarField;
    type Evaluation = E::ScalarField;
    type Proof = UnivariateIpaProof<E>;
    type Commitment = Commitment<E>;
    type BatchProof = (Self::Commitment, Self::Proof);

    fn trim(
        srs: impl ark_std::borrow::Borrow<Self::SRS>,
        supported_degree: usize,
        _supported_num_vars: Option<usize>,
    ) -> Result<
        (
            <Self::SRS as jf_primitives::pcs::StructuredReferenceString>::ProverParam,
            <Self::SRS as jf_primitives::pcs::StructuredReferenceString>::VerifierParam,
        ),
        jf_primitives::pcs::prelude::PCSError,
    > {
        srs.borrow().trim(supported_degree)
    }

    fn commit(
        prover_param: impl ark_std::borrow::Borrow<
            <Self::SRS as StructuredReferenceString>::ProverParam,
        >,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let prover_param: &UnivariateUniversalIpaParams<E> = prover_param.borrow();

        if poly.degree() > prover_param.g_bases.len() - 1 {
            return Err(PCSError::InvalidParameters(format!(
                "poly degree {} is larger than allowed {}",
                poly.degree(),
                prover_param.g_bases.len()
            )));
        }

        let (number_leading_zeroes, plain_coeffs) = skip_leading_zeros_and_convert_to_bigints::<
            E::ScalarField,
            DensePolynomial<E::ScalarField>,
        >(poly);

        let commitment = E::G1::msm_bigint(
            &prover_param.g_bases[number_leading_zeroes..],
            &plain_coeffs,
        )
        .into_affine();

        Ok(commitment)
    }

    fn batch_commit(
        prover_param: impl ark_std::borrow::Borrow<
            <Self::SRS as StructuredReferenceString>::ProverParam,
        >,
        polys: &[Self::Polynomial],
    ) -> Result<Vec<Self::Commitment>, PCSError> {
        let prover_param: &UnivariateUniversalIpaParams<E> = prover_param.borrow();

        let res = polys
            .par_iter()
            .map(|poly| Self::commit(prover_param, poly))
            .collect::<Result<Vec<Commitment<E>>, PCSError>>()?;

        Ok(res)
    }

    fn open(
        prover_param: impl ark_std::borrow::Borrow<
            <Self::SRS as StructuredReferenceString>::ProverParam,
        >,
        polynomial: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let prover_param: &UnivariateUniversalIpaParams<E> = prover_param.borrow();
        let p_poly = polynomial;
        let mut rng = ChaCha20Rng::from_seed([0; 32]);
        let poly_commit = Self::commit(prover_param, p_poly)?;

        // We begin by initiating a new transcript that uses Jellyfish's sponge based rescue hash.
        let mut transcript: RescueTranscript<E::BaseField> =
            <RescueTranscript<E::BaseField> as Transcript>::new_transcript(&[]);

        // Append commitment to transcript
        transcript
            .append_curve_point(b"commitment", &poly_commit)
            .map_err(|_| {
                PCSError::InvalidParameters(
                    "Could not append poly_commit to transcript".to_string(),
                )
            })?;

        let evaluation = p_poly.evaluate(point);

        // Append the evaluation point to the transcript.
        transcript
            .push_message(b"evaluation point", point)
            .map_err(|_| {
                PCSError::InvalidParameters(
                    "Could not append evaluation point to transcript".to_string(),
                )
            })?;

        // Append the evaluation of the polynomial to the transcript.
        transcript
            .push_message(b"evaluation", &evaluation)
            .map_err(|_| {
                PCSError::InvalidParameters("Could not append evaluation to transcript".to_string())
            })?;

        // Generate a random polynomial of the same degree that has a zero at 'point'.
        let degree = prover_param.g_bases.len();

        // This challenge is used to generate our random 'U'.
        let alpha: E::ScalarField = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"label")
            .map_err(|_| {
                PCSError::InvalidParameters("Couldn't squeeze a challenge scalar".to_string())
            })?;

        // Now we calculate P' = P - evaluation * G_{0} + xi * S.
        let p_prime_poly =
            p_poly - &DensePolynomial::<E::ScalarField>::from_coefficients_vec(vec![evaluation]);

        // This variable accumulates the synthetic blinding factor.
        let mut f = E::ScalarField::ZERO;

        // Initialise a vector 'b' that is the powers of the evaluation point.
        let mut b = Vec::<E::ScalarField>::new();

        let k = degree.ilog2();

        let mut cur = E::ScalarField::ONE;
        for _ in 0..(1 << k) {
            b.push(cur);
            cur *= point;
        }

        // Initialise the vector p_prime as the coefficient vector of p_prime.
        let mut p_prime = p_prime_poly.coeffs;
        p_prime.resize(degree, E::ScalarField::ZERO);

        //Initialise the vector "G'" from the SRS.
        let mut g_prime = prover_param.g_bases.clone();

        // Initialise the vectors that will store the l_j and r_j points and the challenges.
        let mut l_j_vec = Vec::<E::G1Affine>::new();
        let mut r_j_vec = Vec::<E::G1Affine>::new();

        let mut l_j_challenges_vec = Vec::<E::ScalarField>::new();
        let mut r_j_challenges_vec = Vec::<E::ScalarField>::new();

        // Perform the inner product argument, round by round.
        for j in 0..k {
            let half = 1 << (k - j - 1); // half the length of `p_prime`, `b`, `G'`

            let p_prime_big_ints: Vec<<E::ScalarField as PrimeField>::BigInt> = p_prime
                .par_iter()
                .map(|coeff| coeff.into_bigint())
                .collect::<Vec<_>>();

            let value_l_j = compute_inner_product(&p_prime[half..], &b[0..half]);
            let value_r_j = compute_inner_product(&p_prime[0..half], &b[half..]);

            let l_j_randomness = E::ScalarField::rand(&mut rng);
            let r_j_randomness = E::ScalarField::rand(&mut rng);

            let l_j_bases = &[&g_prime[0..half], &[prover_param.u], &[prover_param.h]].concat();
            let r_j_bases = &[&g_prime[half..], &[prover_param.u], &[prover_param.h]].concat();

            let l_j_scalars = &[
                &p_prime_big_ints[half..],
                &[(value_l_j * alpha).into_bigint()],
                &[l_j_randomness.into_bigint()],
            ]
            .concat();
            let r_j_scalars = &[
                &p_prime_big_ints[0..half],
                &[(value_r_j * alpha).into_bigint()],
                &[r_j_randomness.into_bigint()],
            ]
            .concat();

            // Compute L, R
            let l_j = E::G1::msm_bigint(l_j_bases, l_j_scalars).into_affine();
            let r_j = E::G1::msm_bigint(r_j_bases, r_j_scalars).into_affine();

            transcript
                .append_curve_points(b"label", &[l_j, r_j])
                .map_err(|_| {
                    PCSError::InvalidParameters("could not append curve points".to_string())
                })?;

            l_j_vec.push(l_j);
            r_j_vec.push(r_j);

            l_j_challenges_vec.push(l_j_randomness);
            r_j_challenges_vec.push(r_j_randomness);

            let u_j: E::ScalarField = transcript
                .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"u j")
                .map_err(|_| {
                    PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
                })?;

            let u_j_inv = u_j.inverse().ok_or(PCSError::InvalidParameters(
                "Could not invert u_j challenge".to_string(),
            ))?;

            // Now we collapse the vectors down.
            p_prime = p_prime
                .par_iter()
                .take(half)
                .zip(p_prime.par_iter().skip(half))
                .map(|(a, b)| *a + *b * u_j_inv)
                .collect::<Vec<_>>();

            b = b
                .par_iter()
                .take(half)
                .zip(b.par_iter().skip(half))
                .map(|(a, b)| *a + *b * u_j)
                .collect::<Vec<_>>();

            g_prime = g_prime
                .par_iter()
                .take(half)
                .zip(g_prime.par_iter().skip(half))
                .map(|(g_1, g_2)| (*g_1 + g_2.mul(u_j).into_affine()).into_affine())
                .collect::<Vec<_>>();

            // Add the randmoness to the synthetic blinding factor
            f += (l_j_randomness * u_j_inv) + (r_j_randomness * u_j);
        }

        let c = p_prime[0];

        Ok((
            UnivariateIpaProof::<E> {
                l_i: l_j_vec,
                r_i: r_j_vec,
                f,
                c,
            },
            evaluation,
        ))
    }

    fn batch_open(
        prover_param: impl ark_std::borrow::Borrow<
            <Self::SRS as StructuredReferenceString>::ProverParam,
        >,
        batch_commitment: &[Self::Commitment],
        polynomials: &[Self::Polynomial],
        points: &[Self::Point],
    ) -> Result<(Self::BatchProof, Vec<Self::Evaluation>), PCSError> {
        if polynomials.len() != points.len() {
            return Err(PCSError::InvalidParameters(format!(
                "poly length {} is different from points length {}",
                polynomials.len(),
                points.len()
            )));
        }

        let prover_param: &UnivariateUniversalIpaParams<E> = prover_param.borrow();

        // We begin by initiating a new transcript that uses Jellyfish's sponge based rescue hash.
        let mut transcript: RescueTranscript<E::BaseField> =
            <RescueTranscript<E::BaseField> as Transcript>::new_transcript(&[]);

        // Append commitments to the transcript.
        transcript
            .append_curve_points(b"commitments", batch_commitment)
            .map_err(|_| {
                PCSError::InvalidParameters(
                    "Could not append commitments to transcript".to_string(),
                )
            })?;

        let mut evals = Vec::<Self::Evaluation>::new();
        for (poly, point) in polynomials.iter().zip(points.iter()) {
            let evaluation = poly.evaluate(point);
            evals.push(evaluation);

            // Append the evaluation point to the transcript.
            transcript
                .push_message(b"evaluation point", point)
                .map_err(|_| {
                    PCSError::InvalidParameters(
                        "Could not append evaluation point to transcript".to_string(),
                    )
                })?;

            // Append the evaluation of the polynomial to the transcript.
            transcript
                .push_message(b"evaluation", &evaluation)
                .map_err(|_| {
                    PCSError::InvalidParameters(
                        "Could not append evaluation to transcript".to_string(),
                    )
                })?;
        }

        // This challenge is used to generate our polynomial 'q'.
        let rho: E::ScalarField = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"rho")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        // Construct the polynomial 'q'.
        let mut q_poly = Self::Polynomial::zero();
        let mut multiplier = E::ScalarField::one();
        for (poly, point, value) in izip!(polynomials, points, &evals) {
            let z_over_z_i = Self::Polynomial::from_coefficients_vec(vec![
                E::ScalarField::zero() - point,
                E::ScalarField::one(),
            ]);
            let t_i_poly = Self::Polynomial::from_coefficients_vec(vec![
                *value - point,
                E::ScalarField::one(),
            ]);
            q_poly.add_assign(
                &(((poly.clone()).add(t_i_poly.neg())).div(&z_over_z_i)).mul(multiplier),
            );
            multiplier *= rho;
        }

        let q_commitment = Self::commit(prover_param, &q_poly)?;

        // Append the commitment to q to the transcript.
        transcript
            .append_curve_point(b"q commitment", &q_commitment)
            .map_err(|_| {
                PCSError::InvalidParameters(
                    "Could not append q commitment to transcript".to_string(),
                )
            })?;

        // This challenge is used to generate our polynomial 'g'.
        let r: E::ScalarField = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"r")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        let mut z_coeff = E::ScalarField::one();
        for point in points {
            z_coeff *= r - point;
        }

        // Construct the polynomial 'g'.
        let mut g_poly: Self::Polynomial = Self::Polynomial::zero();
        let mut multiplier = E::ScalarField::one();
        for (poly, point, value) in izip!(polynomials, points, &evals) {
            let z_i_coeff = z_coeff / (r - point);
            let t_i_poly = Self::Polynomial::from_coefficients_vec(vec![
                *value - point,
                E::ScalarField::one(),
            ]);
            g_poly.add_assign(&(poly.add(&t_i_poly.neg())).mul(multiplier * z_i_coeff));
            multiplier *= rho;
        }

        g_poly.add_assign(&q_poly.mul(z_coeff).neg());

        let (proof, actual_eval) = Self::open(prover_param, &g_poly, &r)?;
        if actual_eval != E::ScalarField::zero() {
            return Err(PCSError::InvalidParameters(
                "actual_eval is not zero".to_string(),
            ));
        }

        Ok(((q_commitment, proof), evals))
    }

    fn verify(
        verifier_param: &<Self::SRS as StructuredReferenceString>::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &Self::Evaluation,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        // We begin by initiating a new transcript that uses Jellyfish's sponge based rescue hash.
        let mut transcript: RescueTranscript<E::BaseField> =
            <RescueTranscript<E::BaseField> as Transcript>::new_transcript(&[]);
        transcript
            .append_curve_point(b"commitment", commitment)
            .map_err(|_| {
                PCSError::InvalidParameters("Could not append commitment to transcript".to_string())
            })?;

        // Append the evaluation point to the transcript.
        transcript
            .push_message(b"evaluation point", point)
            .map_err(|_| {
                PCSError::InvalidParameters(
                    "Could not append evaluation point to transcript".to_string(),
                )
            })?;

        // Append the evaluation of the polynomial to the transcript.
        transcript.push_message(b"evaluation", value).map_err(|_| {
            PCSError::InvalidParameters("Could not append evaluation to transcript".to_string())
        })?;

        // This challenge is used to generate our random 'U'.
        let alpha: E::ScalarField = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"label")
            .map_err(|_| {
                PCSError::InvalidParameters("Couldn't squeeze a challenge scalar".to_string())
            })?;

        let l_j = proof.l_i.clone();
        let r_j = proof.r_i.clone();

        let mut u_j_vec = Vec::new();
        let mut u_j_inv_vec = Vec::new();

        for (l, r) in l_j.iter().zip(r_j.iter()) {
            transcript
                .append_curve_points(b"label", &[*l, *r])
                .map_err(|_| {
                    PCSError::InvalidParameters(
                        "Couldn't append curve points to transcript".to_string(),
                    )
                })?;
            let u_j: E::ScalarField = transcript
                .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"u j")
                .map_err(|_| {
                    PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
                })?;

            let u_j_inv = u_j.inverse().ok_or(PCSError::InvalidParameters(
                "Could not invert u_j challenge".to_string(),
            ))?;

            u_j_vec.push(u_j);
            u_j_inv_vec.push(u_j_inv);
        }

        let scalars = &[u_j_inv_vec.as_slice(), u_j_vec.as_slice()].concat();
        let big_ints = scalars
            .par_iter()
            .map(|scalar| scalar.into_bigint())
            .collect::<Vec<_>>();

        let bases = &[l_j.as_slice(), r_j.as_slice()].concat();

        let sum_no_p_prime = E::G1::msm_bigint(bases, &big_ints).into_affine();

        let p_prime = (commitment.into_group() - verifier_param.g_bases[0] * value).into_affine();

        let lhs: E::G1Affine = (sum_no_p_prime + p_prime).into();
        //Initialise the vector "G'" from the SRS.
        let mut g_prime = verifier_param.g_bases.clone();
        let degree = g_prime.len();
        let mut b = Vec::new();

        let k = degree.ilog2();

        let mut cur = E::ScalarField::ONE;
        for _ in 0..(1 << k) {
            b.push(cur);
            cur *= point;
        }

        let k = degree.ilog2();

        // This is used when a default proof is constructed so that we don't panic when we index into the vecs.
        if u_j_vec.is_empty() {
            u_j_vec.resize(k as usize, E::ScalarField::one());
            u_j_inv_vec.resize(k as usize, E::ScalarField::one());
        }

        for j in 0..k {
            let half = 1 << (k - j - 1); // half the length of `p_prime`, `b`, `G'`
            let u_j = u_j_vec[j as usize];
            g_prime = g_prime
                .par_iter()
                .take(half)
                .zip(g_prime.par_iter().skip(half))
                .map(|(g_1, g_2)| (*g_1 + g_2.mul(u_j).into_affine()).into_affine())
                .collect::<Vec<_>>();

            b = b
                .par_iter()
                .take(half)
                .zip(b.par_iter().skip(half))
                .map(|(a, b)| *a + *b * u_j)
                .collect::<Vec<_>>();
        }

        let g_prime_0 = g_prime[0];
        let b_0 = b[0];

        let u_scalar = proof.c * b_0 * alpha;

        let rhs = E::G1::msm_bigint(
            &[g_prime_0, verifier_param.u, verifier_param.h],
            &[
                proof.c.into_bigint(),
                u_scalar.into_bigint(),
                proof.f.into_bigint(),
            ],
        )
        .into_affine();

        let result = lhs == rhs;
        Ok(result)
    }

    fn batch_verify<R: rand_chacha::rand_core::RngCore + rand_chacha::rand_core::CryptoRng>(
        verifier_param: &<Self::SRS as StructuredReferenceString>::VerifierParam,
        multi_commitment: &[Self::Commitment],
        points: &[Self::Point],
        values: &[Self::Evaluation],
        batch_proof: &Self::BatchProof,
        _rng: &mut R,
    ) -> Result<bool, PCSError> {
        // We begin by initiating a new transcript that uses Jellyfish's sponge based rescue hash.
        let mut transcript: RescueTranscript<E::BaseField> =
            <RescueTranscript<E::BaseField> as Transcript>::new_transcript(&[]);

        // Append commitments to the transcript.
        transcript
            .append_curve_points(b"commitments", multi_commitment)
            .map_err(|_| {
                PCSError::InvalidParameters(
                    "Could not append commitments to transcript".to_string(),
                )
            })?;

        for (point, value) in points.iter().zip(values.iter()) {
            // Append the evaluation point to the transcript.
            transcript
                .push_message(b"evaluation point", point)
                .map_err(|_| {
                    PCSError::InvalidParameters(
                        "Could not append evaluation point to transcript".to_string(),
                    )
                })?;

            // Append the evaluation of the polynomial to the transcript.
            transcript.push_message(b"evaluation", value).map_err(|_| {
                PCSError::InvalidParameters("Could not append evaluation to transcript".to_string())
            })?;
        }

        // This is the challenge 'rho'.
        let rho: E::ScalarField = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"rho")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        // Append the commitment to q to the transcript.
        transcript
            .append_curve_point(b"q commitment", &batch_proof.0)
            .map_err(|_| {
                PCSError::InvalidParameters(
                    "Could not append q commitment to transcript".to_string(),
                )
            })?;

        // This is the challenge 'r'.
        let r: E::ScalarField = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"r")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        let mut z_coeff = E::ScalarField::one();
        for point in points {
            z_coeff *= r - point;
        }

        // Construct the commitment to the polynomial 'g' given the commitment to 'q'.
        let mut g_commitment_point = -batch_proof.0.into_group() * z_coeff;
        let mut multiplier = E::ScalarField::one();

        for (commitment, point, value) in izip!(multi_commitment, points, values) {
            let z_i_coeff = z_coeff / (r - point);
            let t_i_commitment = Self::commit(
                verifier_param,
                &Self::Polynomial::from_coefficients_vec(vec![
                    *value - point,
                    E::ScalarField::one(),
                ]),
            )?;
            g_commitment_point +=
                (commitment.into_group() - t_i_commitment) * multiplier * z_i_coeff;
            multiplier *= rho;
        }

        Self::verify(
            verifier_param,
            &g_commitment_point.into_affine(),
            &r,
            &E::ScalarField::zero(),
            &batch_proof.1,
        )
    }

    #[allow(unused_variables)]
    fn multi_open(
        prover_param: impl ark_std::borrow::Borrow<
            <Self::SRS as StructuredReferenceString>::ProverParam,
        >,
        polynomial: &Self::Polynomial,
        points: &[Self::Point],
    ) -> Result<(Vec<Self::Proof>, Vec<Self::Evaluation>), PCSError> {
        todo!()
    }
}

impl<E: Pairing> Accumulation for UnivariateIpaPCS<E>
where
    E: Pairing,
    E::G1Affine: AffineRepr<BaseField = E::BaseField>,
    <E::G1Affine as AffineRepr>::Config: HasTEForm<BaseField = E::BaseField>,
    <E as Pairing>::BaseField: RescueParameter,
    E::ScalarField: EmulationConfig<E::BaseField>,
{
}

fn skip_leading_zeros_and_convert_to_bigints<F: PrimeField, P: DenseUVPolynomial<F>>(
    p: &P,
) -> (usize, Vec<F::BigInt>) {
    let mut num_leading_zeros = 0;
    while num_leading_zeros < p.coeffs().len() && p.coeffs()[num_leading_zeros].is_zero() {
        num_leading_zeros += 1;
    }
    let coeffs = convert_to_bigints(&p.coeffs()[num_leading_zeros..]);
    (num_leading_zeros, coeffs)
}

fn convert_to_bigints<F: PrimeField>(p: &[F]) -> Vec<F::BigInt> {
    let coeffs = p.iter().map(|s| s.into_bigint()).collect::<Vec<_>>();

    coeffs
}

fn compute_inner_product<F: PrimeField>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip(b.par_iter())
        .fold(|| F::zero(), |acc, (a, b)| acc + (*a * b))
        .reduce(|| F::zero(), |a, b| a + b)
}
/// Adjusts the degree of a polynomial
pub fn polynomial_adjust_degree<F: PrimeField>(
    poly: &DensePolynomial<F>,
    degree_shift: &usize,
) -> DensePolynomial<F> {
    let mut poly_vec = vec![F::zero(); *degree_shift];
    let mut poly_coeffs: Vec<F> = poly.coeffs().to_vec();
    poly_vec.append(&mut poly_coeffs);
    DensePolynomial::<F>::from_coefficients_vec(poly_vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ark_bls12_377::Bls12_377;
    use ark_bn254::Bn254;
    use ark_poly::univariate::DensePolynomial;
    use ark_std::{
        rand::{distributions::Alphanumeric, Rng},
        string::String,
        UniformRand,
    };
    use jf_primitives::pcs::PolynomialCommitmentScheme;
    use jf_utils::test_rng;
    use nf_curves::grumpkin::Grumpkin;
    use std::{env, fs};

    fn end_to_end_test_template<E>() -> Result<(), PCSError>
    where
        E: Pairing,
        <E::G1Affine as AffineRepr>::Config: HasTEForm<BaseField = E::BaseField>,
        E::G1Affine: AffineRepr<BaseField = E::BaseField>,
        <E as Pairing>::BaseField: RescueParameter,
        E::ScalarField: EmulationConfig<E::BaseField>,
    {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let mut degree = 0;
            while degree <= 1 {
                degree = usize::rand(rng) % 20;
            }
            let pp = <UnivariateIpaPCS<E> as PolynomialCommitmentScheme>::load_srs_from_file(
                degree, None,
            )?;
            let (ck, vk) = pp.trim(degree)?;
            let p = <DensePolynomial<E::ScalarField> as DenseUVPolynomial<E::ScalarField>>::rand(
                degree, rng,
            );
            let comm = <UnivariateIpaPCS<E> as PolynomialCommitmentScheme>::commit(&ck, &p)?;
            let point = E::ScalarField::rand(rng);
            let (proof, value) = UnivariateIpaPCS::<E>::open(&ck, &p, &point)?;
            assert!(
                <UnivariateIpaPCS<E> as PolynomialCommitmentScheme>::verify(
                    &vk, &comm, &point, &value, &proof
                )?,
                "proof was incorrect for max_degree = {}, polynomial_degree = {}",
                degree,
                p.degree(),
            );
        }
        Ok(())
    }

    fn linear_polynomial_test_template<E>() -> Result<(), PCSError>
    where
        E: Pairing,
        <E::G1Affine as AffineRepr>::Config: HasTEForm<BaseField = E::BaseField>,
        E::G1Affine: AffineRepr<BaseField = E::BaseField>,
        <E as Pairing>::BaseField: RescueParameter,
        E::ScalarField: EmulationConfig<E::BaseField>,
    {
        let rng = &mut test_rng();
        for _ in 0..100 {
            let degree = 50;

            let pp = UnivariateIpaPCS::<E>::load_srs_from_file(degree, None)?;
            let (ck, vk) = pp.trim(degree)?;
            let commit_in = <DensePolynomial<E::ScalarField> as DenseUVPolynomial<
                E::ScalarField,
            >>::rand(degree, rng);
            let comm = UnivariateIpaPCS::<E>::commit(&ck, &commit_in)?;
            let point = E::ScalarField::rand(rng);
            let (proof, value) = UnivariateIpaPCS::<E>::open(&ck, &commit_in, &point)?;
            assert!(
                UnivariateIpaPCS::<E>::verify(&vk, &comm, &point, &value, &proof)?,
                "proof was incorrect for max_degree = {}, polynomial_degree = {}",
                degree,
                commit_in.degree(),
            );
        }
        Ok(())
    }

    fn polynomial_test_with_srs_gen_template<E>() -> Result<(), PCSError>
    where
        E: Pairing,
        <E::G1Affine as AffineRepr>::Config: HasTEForm<BaseField = E::BaseField>,
        E::G1Affine: AffineRepr<BaseField = E::BaseField>,
        <E as Pairing>::BaseField: RescueParameter,
        E::ScalarField: EmulationConfig<E::BaseField>,
    {
        let rng = &mut test_rng();
        env::set_current_dir("../").unwrap();
        for _ in 0..5 {
            // Use fold to create a random string by adding one character at a time
            let mnemonic = (0..10).fold(String::new(), |mut acc, _| {
                acc.push(rng.sample(Alphanumeric) as char);
                acc
            });
            let path = "srs";
            <UnivariateIpaPCS<E> as PolynomialCommitmentScheme>::SRS::load_srs_to_file(
                20, path, &mnemonic,
            )?;
            let rng = &mut test_rng();
            for _ in 0..20 {
                let mut degree = 0;
                while degree <= 1 {
                    degree = usize::rand(rng) % 20;
                }

                let pp = UnivariateIpaPCS::<E>::load_srs_from_file(degree, Some(path))?;
                let (ck, vk) = pp.trim(degree)?;
                let commit_in = <DensePolynomial<E::ScalarField> as DenseUVPolynomial<
                    E::ScalarField,
                >>::rand(degree, rng);
                let comm = UnivariateIpaPCS::<E>::commit(&ck, &commit_in)?;
                let point = E::ScalarField::rand(rng);
                let (proof, value) = UnivariateIpaPCS::<E>::open(&ck, &commit_in, &point)?;
                assert!(
                    UnivariateIpaPCS::<E>::verify(&vk, &comm, &point, &value, &proof)?,
                    "proof was incorrect for max_degree = {}, polynomial_degree = {}",
                    degree,
                    commit_in.degree(),
                );
            }
            fs::remove_file(path).unwrap();
        }
        Ok(())
    }

    fn batch_check_test_template<E>() -> Result<(), PCSError>
    where
        E: Pairing,
        <E::G1Affine as AffineRepr>::Config: HasTEForm<BaseField = E::BaseField>,
        E::G1Affine: AffineRepr<BaseField = E::BaseField>,
        <E as Pairing>::BaseField: RescueParameter,
        E::ScalarField: EmulationConfig<E::BaseField>,
    {
        let rng = &mut test_rng();
        let max_degree = 30;
        for _ in 0..10 {
            let pp = UnivariateIpaPCS::<E>::load_srs_from_file(max_degree, None)?;
            let (ck, vk) = UnivariateIpaPCS::<E>::trim(&pp, max_degree, None)?;
            let mut polys = Vec::new();
            let mut points = Vec::new();
            for _ in 0..10 {
                let degree = (usize::rand(rng) % (max_degree - 10)) + 10;
                let poly =
                    <DensePolynomial<E::ScalarField> as DenseUVPolynomial<E::ScalarField>>::rand(
                        degree, rng,
                    );
                let point = E::ScalarField::rand(rng);

                polys.push(poly);
                points.push(point);
            }
            let batch_commitment = UnivariateIpaPCS::<E>::batch_commit(&ck, &polys)?;
            let (batch_proof, values) =
                UnivariateIpaPCS::<E>::batch_open(&ck, &batch_commitment, &polys, &points)?;

            assert!(
                UnivariateIpaPCS::<E>::batch_verify(
                    &vk,
                    &batch_commitment,
                    &points,
                    &values,
                    &batch_proof,
                    rng
                )?,
                "batch proof was incorrect for max_degree = {}",
                max_degree,
            );
        }
        Ok(())
    }

    #[test]
    fn end_to_end_test() {
        end_to_end_test_template::<Bls12_377>().expect("test failed for bls12-381");
    }

    #[test]
    fn linear_polynomial_test() {
        linear_polynomial_test_template::<Bls12_377>().expect("test failed for bls12-381");
    }

    #[test]
    fn linear_polynomial_and_srs_gen_test() {
        polynomial_test_with_srs_gen_template::<Bn254>().expect("test failed for bn254");
        polynomial_test_with_srs_gen_template::<Grumpkin>().expect("test failed for grumpkin");
    }

    #[test]
    fn batch_check_test() {
        batch_check_test_template::<Bls12_377>().expect("test failed for bls12-381");
    }
}
