//! This module contains the implementation of the Zeromorph protocol
use crate::nightfall::hops::univariate_ipa::polynomial_adjust_degree;

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use jf_primitives::pcs::Accumulation;
use jf_relation::gadgets::ecc::HasTEForm;

use ark_poly::{
    evaluations::multivariate::multilinear::DenseMultilinearExtension, univariate::DensePolynomial,
    DenseUVPolynomial, MultilinearExtension,
};
#[cfg(any(test, feature = "test-srs"))]
use ark_std::rand::{CryptoRng, RngCore};

use ark_std::{
    cfg_iter, format,
    hash::Hash,
    marker::PhantomData,
    ops::{AddAssign, Mul},
    string::ToString,
    sync::Arc,
    vec::Vec,
    One, Zero,
};
use jf_relation::gadgets::EmulationConfig;
use rayon::prelude::*;

use super::super::super::hops::{
    srs::UnivariateUniversalIpaParams, univariate_ipa::UnivariateIpaPCS,
};

use crate::{
    nightfall::mle::{
        subroutines::{PolyOracle, SumCheckProof},
        utils::{add_extra_variables, mv_batch_open, mv_batch_verify},
    },
    transcript::{RescueTranscript, Transcript},
};

use jf_primitives::{
    pcs::{
        prelude::{Commitment, PCSError},
        PolynomialCommitmentScheme, StructuredReferenceString,
    },
    rescue::RescueParameter,
};

use super::ZeromorphHelper;

#[derive(Debug, CanonicalDeserialize, CanonicalSerialize, Clone, PartialEq, Eq)]
/// Zeromorph proof when the underlying univariate PCS is Ipa
pub struct ZeromorphIpaProof<E>
where
    E: Pairing,
    E::G1Affine: AffineRepr<BaseField = E::BaseField>,
    <E::G1Affine as AffineRepr>::Config: HasTEForm<BaseField = E::BaseField>,
    <E as Pairing>::BaseField: RescueParameter,
    E::ScalarField: EmulationConfig<E::BaseField>,
{
    /// Commitments to all the `U(q_k)`s, `U(f)` and `q_hat`.
    pub commitments: Vec<<UnivariateIpaPCS<E> as PolynomialCommitmentScheme>::Commitment>,
    /// Proof associated with the degree check part of the zeromorph proof.
    pub degree_check_proof: <UnivariateIpaPCS<E> as PolynomialCommitmentScheme>::Proof,
}

impl<E: Pairing> Default for ZeromorphIpaProof<E>
where
    E: Pairing,
    E::G1Affine: AffineRepr<BaseField = E::BaseField>,
    <E::G1Affine as AffineRepr>::Config: HasTEForm<BaseField = E::BaseField>,
    <E as Pairing>::BaseField: RescueParameter,
    E::ScalarField: EmulationConfig<E::BaseField>,
{
    fn default() -> Self {
        Self {
            commitments:
                Vec::<<UnivariateIpaPCS<E> as PolynomialCommitmentScheme>::Commitment>::default(),
            degree_check_proof: <UnivariateIpaPCS<E> as PolynomialCommitmentScheme>::Proof::default(
            ),
        }
    }
}

impl<E: Pairing> Hash for ZeromorphIpaProof<E>
where
    E: Pairing,
    E::G1Affine: AffineRepr<BaseField = E::BaseField>,
    <E::G1Affine as AffineRepr>::Config: HasTEForm<BaseField = E::BaseField>,
    <E as Pairing>::BaseField: RescueParameter,
    E::ScalarField: EmulationConfig<E::BaseField>,
{
    fn hash<H: ark_std::hash::Hasher>(&self, state: &mut H) {
        for commit in self.commitments.iter() {
            Hash::hash(&commit, state);
        }
        Hash::hash(&self.degree_check_proof, state);
        Hash::hash(&self.degree_check_proof, state);
    }
}

#[derive(Clone, PartialEq, Eq)]
/// Zeromorph batch proof when the underlying univariate PCS is Ipa
pub struct ZeromorphIpaBatchProof<E>
where
    E: Pairing,
    E::G1Affine: AffineRepr<BaseField = E::BaseField>,
    <E::G1Affine as AffineRepr>::Config: HasTEForm<BaseField = E::BaseField>,
    <E as Pairing>::BaseField: RescueParameter,
    E::ScalarField: EmulationConfig<E::BaseField>,
{
    /// Sum check proof associated to a batch mle proof.
    pub sum_check_proof: SumCheckProof<E::ScalarField, PolyOracle<E::ScalarField>>,
    /// Proof for the polynomial g_prime.
    pub final_proof: ZeromorphIpaProof<E>,
}

impl<E> Default for ZeromorphIpaBatchProof<E>
where
    E: Pairing,
    E::G1Affine: AffineRepr<BaseField = E::BaseField>,
    <E::G1Affine as AffineRepr>::Config: HasTEForm<BaseField = E::BaseField>,
    <E as Pairing>::BaseField: RescueParameter,
    E::ScalarField: EmulationConfig<E::BaseField>,
{
    fn default() -> Self {
        Self {
            sum_check_proof: SumCheckProof::<E::ScalarField, PolyOracle<E::ScalarField>>::default(),
            final_proof: ZeromorphIpaProof::<E>::default(),
        }
    }
}

/// Zeromorph PCS for multilinear polynomials
#[derive(Clone, Debug)]
pub struct Zeromorph<PCS: PolynomialCommitmentScheme> {
    phantom: PhantomData<PCS>,
}

impl<PCS: PolynomialCommitmentScheme> Default for Zeromorph<PCS> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<E> PolynomialCommitmentScheme for Zeromorph<UnivariateIpaPCS<E>>
where
    E: Pairing,
    E::ScalarField: EmulationConfig<E::BaseField>,
    E::G1Affine: AffineRepr<BaseField = E::BaseField, ScalarField = E::ScalarField>,
    <E::G1Affine as AffineRepr>::Config:
        HasTEForm<BaseField = E::BaseField, ScalarField = E::ScalarField>,
    <E as Pairing>::BaseField: RescueParameter,
{
    type SRS = UnivariateUniversalIpaParams<E>;
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Proof = ZeromorphIpaProof<E>;
    type Commitment = Commitment<E>;
    type BatchProof = ZeromorphIpaBatchProof<E>;

    fn trim(
        srs: impl ark_std::borrow::Borrow<Self::SRS>,
        supported_degree: usize,
        supported_num_vars: Option<usize>,
    ) -> Result<
        (
            <Self::SRS as jf_primitives::pcs::StructuredReferenceString>::ProverParam,
            <Self::SRS as jf_primitives::pcs::StructuredReferenceString>::VerifierParam,
        ),
        jf_primitives::pcs::prelude::PCSError,
    > {
        if let Some(num) = supported_num_vars {
            return srs.borrow().trim((1 << num) - 1);
        }
        srs.borrow().trim(supported_degree)
    }

    #[cfg(any(test, feature = "test-srs"))]
    fn gen_srs_for_testing<R: RngCore + CryptoRng>(
        rng: &mut R,
        supported_degree: usize,
    ) -> Result<Self::SRS, PCSError> {
        Self::SRS::gen_srs_for_testing(rng, (1 << supported_degree) - 1)
    }

    fn commit(
        prover_param: impl ark_std::borrow::Borrow<
            <Self::SRS as StructuredReferenceString>::ProverParam,
        >,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let prover_param: &UnivariateUniversalIpaParams<E> = prover_param.borrow();

        if poly.evaluations.len() > prover_param.g_bases.len() {
            return Err(PCSError::InvalidParameters(format!(
                "Number of evaluations {} is larger than allowed {}",
                poly.evaluations.len(),
                prover_param.g_bases.len()
            )));
        }
        let mut num_leading_zeroes = 0;
        while num_leading_zeroes < poly.evaluations.len()
            && poly.evaluations[num_leading_zeroes].is_zero()
        {
            num_leading_zeroes += 1;
        }

        let plain_coeffs = poly.evaluations[num_leading_zeroes..]
            .iter()
            .map(|x| x.into_bigint())
            .collect::<Vec<_>>();

        let commitment =
            E::G1::msm_bigint(&prover_param.g_bases[num_leading_zeroes..], &plain_coeffs)
                .into_affine();

        Ok(commitment)
        // UnivariateIpaPCS::<E>::commit(prover_param, &ZeromorphHelper::isom(poly))
    }

    fn batch_commit(
        prover_param: impl ark_std::borrow::Borrow<
            <Self::SRS as StructuredReferenceString>::ProverParam,
        >,
        polys: &[Self::Polynomial],
    ) -> Result<Vec<Self::Commitment>, PCSError> {
        let prover_param: &UnivariateUniversalIpaParams<E> = prover_param.borrow();

        let max_num_vars =
            polys
                .iter()
                .map(|x| x.num_vars)
                .max()
                .ok_or(PCSError::InvalidParameters(
                    "Couldn't calculate max_num_vars".to_string(),
                ))?;

        let res = polys
            .par_iter()
            .map(|poly| {
                Self::commit(
                    prover_param,
                    &add_extra_variables(poly, &max_num_vars).map_err(|_| {
                        PCSError::InvalidParameters("Could not add extra variables".to_string())
                    })?,
                )
            })
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

        // We compute the q_k polynomials.
        let qks = ZeromorphHelper::compute_qks(polynomial, point).ok_or(
            PCSError::InvalidParameters("Could not append compute q_k polymonials".to_string()),
        )?;
        // Act U_n on each q_k polynomial.
        let mut u_vec = qks
            .iter()
            .map(ZeromorphHelper::isom)
            .collect::<Vec<DensePolynomial<E::ScalarField>>>();
        // We also need to degree-check the polynomial itself.
        u_vec.push(ZeromorphHelper::isom(polynomial));

        let num_vars = polynomial.num_vars;
        let max_degree = prover_param.max_degree() - 1;

        let mut u_commitments = UnivariateIpaPCS::<E>::batch_commit(prover_param, &u_vec)?;
        // We initiate a new transcript that uses Jellyfish's sponge based rescue hash.
        let mut transcript: RescueTranscript<E::BaseField> =
            <RescueTranscript<E::BaseField> as Transcript>::new_transcript(&[]);

        // We append the commitments to the transcript
        for commitment in u_commitments.iter() {
            transcript
                .append_curve_point(b"commitment", commitment)
                .map_err(|_| {
                    PCSError::InvalidParameters(
                        "Could not append commitment to transcript".to_string(),
                    )
                })?;
        }

        // This challenge y is used for the batched degree check on u_vec.
        let y = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"y")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        // We construct the `q_hat` poly and push its commitment to the transcript.
        let mut q_hat = DensePolynomial::<E::ScalarField>::zero();
        let mut y_pow = E::ScalarField::one();
        for (k, qk_hat) in u_vec.iter().enumerate().take(num_vars) {
            q_hat.add_assign(
                &polynomial_adjust_degree(qk_hat, &(max_degree - (1 << k) + 1)).mul(y_pow),
            );
            y_pow *= y;
        }

        let q_hat_comm = UnivariateIpaPCS::<E>::commit(prover_param, &q_hat)?;
        u_commitments.push(q_hat_comm);
        transcript
            .append_curve_point(b"commitment", &q_hat_comm)
            .map_err(|_| {
                PCSError::InvalidParameters("Could not append commitment to transcript".to_string())
            })?;

        // This challenge x is used to construct our Z_x polynomial.
        let x = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"x")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        // This challenge z is used to batch our two degree checks.
        let z = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"z")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        // We construct the `zeta` and `Z_x` polys.
        let mut zeta = q_hat;
        y_pow = E::ScalarField::one();
        for (k, qk_hat) in u_vec.iter().enumerate().take(num_vars) {
            zeta.add_assign(&qk_hat.mul(-y_pow * x.pow([(max_degree as u64 - (1 << k) + 1)])));
            y_pow *= y;
        }

        let z_poly = ZeromorphHelper::z_poly(polynomial, point, &x, &qks).ok_or(
            PCSError::InvalidParameters("Could not append compute q_k polymonial".to_string()),
        )?;

        let (degree_check_proof, zero_eval) =
            UnivariateIpaPCS::<E>::open(prover_param, &(zeta + z_poly.mul(z)), &x)?;

        // We check that the `zeta + z * Z_x` polynomial evaluates to zero at x.
        if zero_eval != E::ScalarField::zero() {
            return Err(PCSError::InvalidParameters(
                "Z_x polynomial does not evaluate to zero".to_string(),
            ));
        }

        let v = polynomial
            .evaluate(point)
            .ok_or(PCSError::InvalidParameters(
                "Could not evaluate polynomial".to_string(),
            ))?;

        Ok((
            Self::Proof {
                commitments: u_commitments,
                degree_check_proof,
            },
            v,
        ))
    }

    #[allow(unused_variables)]
    fn batch_open(
        prover_param: impl ark_std::borrow::Borrow<
            <Self::SRS as StructuredReferenceString>::ProverParam,
        >,
        batch_commitment: &[Self::Commitment],
        polynomials: &[Self::Polynomial],
        points: &[Self::Point],
    ) -> Result<(Self::BatchProof, Vec<Self::Evaluation>), PCSError> {
        let prover_param: &UnivariateUniversalIpaParams<E> = prover_param.borrow();
        // We begin by initiating a new transcript that uses Jellyfish's sponge based rescue hash.
        let mut transcript: RescueTranscript<E::BaseField> =
            <RescueTranscript<E::BaseField> as Transcript>::new_transcript(&[]);

        // We append each commitment to the transcript.
        for commitment in batch_commitment.iter() {
            transcript
                .append_curve_point(b"commitment", commitment)
                .map_err(|_| {
                    PCSError::InvalidParameters(
                        "Could not append commitment to transcript".to_string(),
                    )
                })?;
        }
        // We evaluate all the polys at the relevant points
        let evals = cfg_iter!(polynomials)
            .zip(cfg_iter!(points))
            .map(|(poly, point)| {
                poly.evaluate(point).ok_or(PCSError::InvalidParameters(
                    "Couldn't evaluate polynomial".to_string(),
                ))
            })
            .collect::<Result<Vec<E::ScalarField>, PCSError>>()?;
        // We batch the polynomials into a Checksum proof.
        let (sum_check_proof, _, mle, a_2) = mv_batch_open::<<E::G1Affine as AffineRepr>::Config>(
            polynomials,
            points,
            &mut transcript,
        )
        .map_err(|e| {
            PCSError::InvalidParameters(
                e.to_string(), // "Couldn't batch open multilinear polynomials".to_string(),
            )
        })?;

        // We open the relevant mle polynomial.
        let (mle_proof, _mle_eval) = Self::open(prover_param, &mle, &a_2)?;

        Ok((
            Self::BatchProof {
                sum_check_proof,
                final_proof: mle_proof,
            },
            evals,
        ))
    }

    fn verify(
        verifier_param: &<Self::SRS as StructuredReferenceString>::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &Self::Evaluation,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        if proof.commitments.len() < 3 {
            return Err(PCSError::InvalidParameters(
                "Not enough commitments in proof".to_string(),
            ));
        }
        let commitments = &proof.commitments;

        let degree_check_proof = &proof.degree_check_proof;

        let q_hat_comm = &commitments[commitments.len() - 1];
        let f_hat_comm = &commitments[commitments.len() - 2];

        // We check that the commitment is the penultimate commitment in the batch commitment.
        if *commitment != *f_hat_comm {
            return Err(PCSError::InvalidParameters(
                "Polynomial commitment does not match penultimate commitment in proof".to_string(),
            ));
        }

        let num_vars = commitments.len() - 2;
        let max_degree = verifier_param.max_degree() - 1;

        // We initiate a new transcript that uses a sponge based rescue hash.
        let mut transcript: RescueTranscript<E::BaseField> =
            <RescueTranscript<E::BaseField> as Transcript>::new_transcript(&[]);

        // We push the commitments to the `q_k`s and `f` to the transcript.
        for commitment in commitments.iter().take(num_vars + 1) {
            transcript
                .append_curve_point(b"commitment", commitment)
                .map_err(|_| {
                    PCSError::InvalidParameters(
                        "Could not append commitment to transcript".to_string(),
                    )
                })?;
        }

        // This challenge y is used for the batched degree check on all the `U(q_k)`s.
        let y = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"y")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        // We push the commitment to `q_hat` to the transcript.
        transcript
            .append_curve_point(b"commitment", q_hat_comm)
            .map_err(|_| {
                PCSError::InvalidParameters("Could not append commitment to transcript".to_string())
            })?;

        // This challenge x is used to construct our Z_x polynomial.
        let x = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"x")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        // This challenge z is used to batch our two degree checks.
        let z = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"z")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        // We construct the commitment to the poly `zeta`.
        let mut zeta_comm = q_hat_comm.into_group();
        let mut y_pow = E::ScalarField::one();
        for (k, qk_hat_comm) in commitments.iter().take(num_vars).enumerate() {
            zeta_comm -=
                qk_hat_comm.into_group() * (y_pow * x.pow([(max_degree as u64 - (1 << k) + 1)]));
            y_pow *= y;
        }

        // We construct the commitment to the poly `Z_x`.
        let const_value = ZeromorphHelper::eval_phi_poly(num_vars, &x).ok_or(
            PCSError::InvalidParameters("could evaluate phi_poly".to_string()),
        )? * value;

        let const_commitment = UnivariateIpaPCS::<E>::commit(
            verifier_param,
            &DensePolynomial::<E::ScalarField>::from_coefficients_vec([const_value].to_vec()),
        )?;
        let mut z_comm = f_hat_comm.into_group();
        z_comm -= const_commitment.into_group();

        let mut x_pow = x;
        for k in 0..num_vars {
            let coeff = x_pow
                * ZeromorphHelper::eval_phi_poly(num_vars - k - 1, &(x_pow * x_pow)).ok_or(
                    PCSError::InvalidParameters("could evaluate phi_poly".to_string()),
                )?
                - point[k]
                    * ZeromorphHelper::eval_phi_poly(num_vars - k, &x_pow).ok_or(
                        PCSError::InvalidParameters("could evaluate phi_poly".to_string()),
                    )?;
            z_comm -= commitments[k].into_group() * coeff;
            x_pow *= x_pow;
        }

        // We verify our batched commitment evaluates to zero.
        let batch_comm = zeta_comm + z_comm * z;

        UnivariateIpaPCS::<E>::verify(
            verifier_param,
            &Commitment::<E>::from(batch_comm),
            &x,
            &E::ScalarField::zero(),
            degree_check_proof,
        )
    }

    fn batch_verify<R: rand_chacha::rand_core::RngCore + rand_chacha::rand_core::CryptoRng>(
        verifier_param: &<Self::SRS as StructuredReferenceString>::VerifierParam,
        multi_commitment: &[Self::Commitment],
        points: &[Self::Point],
        values: &[Self::Evaluation],
        batch_proof: &Self::BatchProof,
        _rng: &mut R,
    ) -> Result<bool, PCSError> {
        let sum_check_proof = &batch_proof.sum_check_proof;
        let mle_proof = &batch_proof.final_proof;

        // We initiate a new transcript that uses Jellyfish's sponge based rescue hash.
        let mut transcript: RescueTranscript<E::BaseField> =
            <RescueTranscript<E::BaseField> as Transcript>::new_transcript(&[]);

        // We append each commitment to the transcript.
        for commitment in multi_commitment.iter() {
            transcript
                .append_curve_point(b"commitment", commitment)
                .map_err(|_| {
                    PCSError::InvalidParameters(
                        "Could not append commitment to transcript".to_string(),
                    )
                })?;
        }

        // eq_tilde evaluated at (a_1, a_2) and the coefficients of the f_i's used to construct g_prime
        let (g_tilde_eval, f_i_coeff, a_2) =
            mv_batch_verify::<<E::G1Affine as AffineRepr>::Config>(
                points,
                values,
                sum_check_proof,
                &mut transcript,
            )
            .map_err(|_| {
                PCSError::InvalidParameters(
                    "Couldn't batch verify multilinear polynomials".to_string(),
                )
            })?;

        // Construct the commitment to the mle 'g_prime'.
        let mut g_prime_commitment_point = E::G1Affine::zero().into_group();
        for (commitment, coeff) in multi_commitment.iter().zip(f_i_coeff.iter()) {
            g_prime_commitment_point += *commitment * *coeff;
        }
        let g_prime_commitment = Commitment::<E>::from(g_prime_commitment_point);
        Self::verify(
            verifier_param,
            &g_prime_commitment,
            &a_2,
            &g_tilde_eval,
            mle_proof,
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

impl<E: Pairing> Zeromorph<UnivariateIpaPCS<E>>
where
    E: Pairing,
    E::ScalarField: EmulationConfig<E::BaseField>,
    E::G1Affine: AffineRepr<BaseField = E::BaseField, ScalarField = E::ScalarField>,
    <E::G1Affine as AffineRepr>::Config:
        HasTEForm<BaseField = E::BaseField, ScalarField = E::ScalarField>,
    <E as Pairing>::BaseField: RescueParameter,
{
    /// This function performs the opening but also return the polynomial created
    #[allow(clippy::type_complexity)]
    pub fn open_with_poly(
        prover_param: impl ark_std::borrow::Borrow<
            <<Self as PolynomialCommitmentScheme>::SRS as StructuredReferenceString>::ProverParam,
        >,
        polynomial: &<Self as PolynomialCommitmentScheme>::Polynomial,
        point: &<Self as PolynomialCommitmentScheme>::Point,
    ) -> Result<
        (
            <Self as PolynomialCommitmentScheme>::Proof,
            <Self as PolynomialCommitmentScheme>::Evaluation,
            DensePolynomial<E::ScalarField>,
        ),
        PCSError,
    > {
        let prover_param: &UnivariateUniversalIpaParams<E> = prover_param.borrow();

        // We compute the q_k polynomials.
        let qks = ZeromorphHelper::compute_qks(polynomial, point).ok_or(
            PCSError::InvalidParameters("Could not append compute q_k polymonials".to_string()),
        )?;
        // Act U_n on each q_k polynomial.
        let mut u_vec = qks
            .iter()
            .map(ZeromorphHelper::isom)
            .collect::<Vec<DensePolynomial<E::ScalarField>>>();
        // We also need to degree-check the polynomial itself.
        u_vec.push(ZeromorphHelper::isom(polynomial));

        let num_vars = polynomial.num_vars;
        let max_degree = prover_param.max_degree() - 1;

        let mut u_commitments = UnivariateIpaPCS::<E>::batch_commit(prover_param, &u_vec)?;
        // We initiate a new transcript that uses Jellyfish's sponge based rescue hash.
        let mut transcript: RescueTranscript<E::BaseField> =
            <RescueTranscript<E::BaseField> as Transcript>::new_transcript(&[]);

        // We append the commitments to the transcript
        for commitment in u_commitments.iter() {
            transcript
                .append_curve_point(b"commitment", commitment)
                .map_err(|_| {
                    PCSError::InvalidParameters(
                        "Could not append commitment to transcript".to_string(),
                    )
                })?;
        }

        // This challenge y is used for the batched degree check on u_vec.
        let y = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"y")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        // We construct the `q_hat` poly and push its commitment to the transcript.
        let mut q_hat = DensePolynomial::<E::ScalarField>::zero();
        let mut y_pow = E::ScalarField::one();
        for (k, qk_hat) in u_vec.iter().enumerate().take(num_vars) {
            q_hat.add_assign(
                &polynomial_adjust_degree(qk_hat, &(max_degree - (1 << k) + 1)).mul(y_pow),
            );
            y_pow *= y;
        }

        let q_hat_comm = UnivariateIpaPCS::<E>::commit(prover_param, &q_hat)?;
        u_commitments.push(q_hat_comm);
        transcript
            .append_curve_point(b"commitment", &q_hat_comm)
            .map_err(|_| {
                PCSError::InvalidParameters("Could not append commitment to transcript".to_string())
            })?;

        // This challenge x is used to construct our Z_x polynomial.
        let x = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"x")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        // This challenge z is used to batch our two degree checks.
        let z = transcript
            .squeeze_scalar_challenge::<<E::G1Affine as AffineRepr>::Config>(b"z")
            .map_err(|_| {
                PCSError::InvalidParameters("could not squeeze challenge scalar".to_string())
            })?;

        // We construct the `zeta` and `Z_x` polys.
        let mut zeta = q_hat;
        y_pow = E::ScalarField::one();
        for (k, qk_hat) in u_vec.iter().enumerate().take(num_vars) {
            zeta.add_assign(&qk_hat.mul(-y_pow * x.pow([(max_degree as u64 - (1 << k) + 1)])));
            y_pow *= y;
        }

        let z_poly = ZeromorphHelper::z_poly(polynomial, point, &x, &qks).ok_or(
            PCSError::InvalidParameters("Could not append compute q_k polymonial".to_string()),
        )?;
        let final_poly = zeta + z_poly.mul(z);
        let (degree_check_proof, zero_eval) =
            UnivariateIpaPCS::<E>::open(prover_param, &final_poly, &x)?;

        // We check that the `zeta + z * Z_x` polynomial evaluates to zero at x.
        if zero_eval != E::ScalarField::zero() {
            return Err(PCSError::InvalidParameters(
                "Z_x polynomial does not evaluate to zero".to_string(),
            ));
        }

        let v = polynomial
            .evaluate(point)
            .ok_or(PCSError::InvalidParameters(
                "Could not evaluate polynomial".to_string(),
            ))?;

        Ok((
            ZeromorphIpaProof::<E> {
                commitments: u_commitments,
                degree_check_proof,
            },
            v,
            final_poly,
        ))
    }
}

impl<E: Pairing> Accumulation for Zeromorph<UnivariateIpaPCS<E>>
where
    E: Pairing,
    E::ScalarField: EmulationConfig<E::BaseField>,
    E::G1Affine: AffineRepr<BaseField = E::BaseField, ScalarField = E::ScalarField>,
    <E::G1Affine as AffineRepr>::Config:
        HasTEForm<BaseField = E::BaseField, ScalarField = E::ScalarField>,
    <E as Pairing>::BaseField: RescueParameter,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    use ark_bls12_377::Bls12_377;
    use ark_std::UniformRand;
    use jf_utils::test_rng;
    use nf_curves::grumpkin::Grumpkin;

    fn zeromorph_test_template<E>() -> Result<(), PCSError>
    where
        E: Pairing,
        E::ScalarField: EmulationConfig<E::BaseField>,
        E::G1Affine: AffineRepr<BaseField = E::BaseField, ScalarField = E::ScalarField>,
        <E::G1Affine as AffineRepr>::Config:
            HasTEForm<BaseField = E::BaseField, ScalarField = E::ScalarField>,
        <E as Pairing>::BaseField: RescueParameter,
    {
        let rng = &mut test_rng();
        let max_num_vars = 12;
        for _ in 0..10 {
            let num_vars = (usize::rand(rng) % (max_num_vars - 5)) + 5;
            let pp = UnivariateIpaPCS::<E>::load_srs_from_file((1 << num_vars) - 1, None)?;
            let (ck, vk) = Zeromorph::<UnivariateIpaPCS<E>>::trim(&pp, 0, Some(num_vars))?;

            let mle = Arc::new(DenseMultilinearExtension::<E::ScalarField>::rand(
                num_vars, rng,
            ));
            let commitment = Zeromorph::<UnivariateIpaPCS<E>>::commit(&ck, &mle)?;
            let point = (0..num_vars)
                .map(|_| E::ScalarField::rand(rng))
                .collect::<Vec<E::ScalarField>>();
            let (proof, eval) = Zeromorph::<UnivariateIpaPCS<E>>::open(&ck, &mle, &point)?;
            assert!(
                Zeromorph::<UnivariateIpaPCS<E>>::verify(&vk, &commitment, &point, &eval, &proof)?,
                "proof was incorrect for num_vars = {}",
                num_vars
            );
        }
        Ok(())
    }

    fn batch_zeromorph_test_template<E>() -> Result<(), PCSError>
    where
        E: Pairing,
        E::ScalarField: EmulationConfig<E::BaseField>,
        E::G1Affine: AffineRepr<BaseField = E::BaseField, ScalarField = E::ScalarField>,
        <E::G1Affine as AffineRepr>::Config:
            HasTEForm<BaseField = E::BaseField, ScalarField = E::ScalarField>,
        <E as Pairing>::BaseField: RescueParameter,
    {
        let rng = &mut test_rng();
        let max_num_vars = 10;
        for _ in 0..10 {
            let pp = UnivariateIpaPCS::<E>::load_srs_from_file((1 << max_num_vars) - 1, None)?;
            let (ck, vk) = Zeromorph::<UnivariateIpaPCS<E>>::trim(&pp, 0, Some(max_num_vars))?;
            let mut polys = Vec::new();
            let mut points = Vec::new();
            let num_vars = (usize::rand(rng) % (max_num_vars - 5)) + 5;
            for _ in 0..5 {
                let poly = Arc::new(DenseMultilinearExtension::<E::ScalarField>::rand(
                    num_vars, rng,
                ));
                let point = (0..num_vars)
                    .map(|_| E::ScalarField::rand(rng))
                    .collect::<Vec<E::ScalarField>>();

                polys.push(poly);
                points.push(point);
            }
            let batch_commitment = Zeromorph::<UnivariateIpaPCS<E>>::batch_commit(&ck, &polys)?;
            let (batch_proof, values) = Zeromorph::<UnivariateIpaPCS<E>>::batch_open(
                &ck,
                &batch_commitment,
                &polys,
                &points,
            )?;

            assert!(
                Zeromorph::<UnivariateIpaPCS<E>>::batch_verify(
                    &vk,
                    &batch_commitment,
                    &points,
                    &values,
                    &batch_proof,
                    rng
                )?,
                "proof was incorrect for num_vars = {}",
                max_num_vars
            );
        }
        Ok(())
    }

    #[test]
    fn zeromorph_test() {
        zeromorph_test_template::<Bls12_377>().expect("test failed for bls12-381");
    }

    #[test]
    fn batch_zeromorph_test() {
        batch_zeromorph_test_template::<Bls12_377>().expect("test failed for bls12-381");
        batch_zeromorph_test_template::<Grumpkin>().expect("test failed for Grumpkin");
    }
}
