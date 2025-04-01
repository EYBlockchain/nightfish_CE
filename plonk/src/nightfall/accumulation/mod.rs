//! This module provides functionality to perform split accumulation.

use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, Projective},
    CurveGroup, VariableBaseMSM,
};
use ark_ff::{Field, PrimeField};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{
    cfg_into_iter, cfg_iter, ops::Neg, rand::Rng, string::ToString, sync::Arc, vec, vec::Vec,
    UniformRand, Zero,
};
use jf_primitives::{
    pcs::{
        prelude::{
            UnivariateKzgPCS, UnivariateKzgProof, UnivariateProverParam, UnivariateVerifierParam,
        },
        Accumulation, StructuredReferenceString,
    },
    rescue::RescueParameter,
};
use jf_relation::gadgets::{ecc::HasTEForm, EmulationConfig};
use rand_chacha::rand_core::{CryptoRng, RngCore};

use crate::{
    errors::PlonkError,
    transcript::{RescueTranscript, Transcript},
};

use rayon::prelude::*;

use self::accumulation_structs::{
    Accumulator, AtomicAccProof, AtomicAccumulator, AtomicInstance, MLEAccProof, PCSInstance,
    PCSWitness, SplitAccumulator,
};
use super::mle::{
    mle_structs::PolynomialError,
    utils::{challenges_and_scalars_from_mv_batch_open, mv_batch_verify},
};

pub mod accumulation_structs;
pub mod circuit;

/// Struct used to accumulate univariate PCS openings into a single opening proof.
/// This struct uses atomic accumulation.
#[derive(Debug, Clone)]
pub struct UVAtomicAccumulator<PCS: Accumulation> {
    /// List of instances to be accumulated.
    pub(crate) instances: Vec<AtomicInstance<PCS>>,
}
impl<PCS: Accumulation> Accumulator for UVAtomicAccumulator<PCS> {}
/// Struct used to accumulate multilinear PCS openings into a single opening proof.
/// Currently unoptimized and does so in a naive way.
#[derive(Debug, Clone)]
pub struct MLAccumulator<PCS: Accumulation> {
    /// List of instances to be accumulated.
    pub(crate) instances: Vec<PCSInstance<PCS>>,
    /// List of witnesses to be accumulated.
    pub(crate) witnesses: Vec<PCSWitness<PCS>>,
}
impl<PCS: Accumulation> Accumulator for MLAccumulator<PCS> {}
impl<PCS, P> SplitAccumulator<PCS> for MLAccumulator<PCS>
where
    P: HasTEForm,
    PCS: Accumulation<
        Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
        Commitment = Affine<P>,
        Evaluation = P::ScalarField,
        Point = Vec<P::ScalarField>,
    >,
    P::BaseField: PrimeField + RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    type Instance = PCSInstance<PCS>;
    type Witness = PCSWitness<PCS>;
    type AccProof = MLEAccProof<PCS>;
    type WithChallengesOutput = (
        AccumulationChallengesAndScalars<PCS::Evaluation>,
        Self::AccProof,
        Self,
    );

    fn new() -> Self {
        Self {
            instances: Vec::new(),
            witnesses: Vec::new(),
        }
    }

    fn commitments(&self) -> Vec<<PCS>::Commitment> {
        self.instances.iter().map(|i| i.comm).collect::<Vec<_>>()
    }

    fn polynomials(&self) -> Vec<<PCS>::Polynomial> {
        self.witnesses
            .iter()
            .map(|w| w.poly.clone())
            .collect::<Vec<_>>()
    }

    fn points(&self) -> Vec<<PCS>::Point> {
        self.instances
            .iter()
            .map(|i| i.point.clone())
            .collect::<Vec<_>>()
    }

    fn evaluations(&self) -> Vec<<PCS>::Evaluation> {
        self.witnesses.iter().map(|w| w.value).collect::<Vec<_>>()
    }

    fn push(
        &mut self,
        poly: <PCS>::Polynomial,
        comm: <PCS>::Commitment,
        point: <PCS>::Point,
        value: <PCS>::Evaluation,
    ) {
        self.instances
            .push(PCSInstance::new(comm, value, point.clone()));
        self.witnesses
            .push(PCSWitness::new(poly, comm, value, point));
    }

    fn prove_accumulation(
        &mut self,
        prover_param: &<PCS::SRS as StructuredReferenceString>::ProverParam,
        transcript: Option<&mut RescueTranscript<P::BaseField>>,
    ) -> Result<(Self::AccProof, Self), PolynomialError> {
        let (_, sumcheck_proof, new_accumulator) =
            self.prove_accumulation_with_challenges_and_scalars(prover_param, transcript)?;

        Ok((sumcheck_proof, new_accumulator))
    }

    fn prove_accumulation_with_challenges_and_scalars(
        &mut self,
        prover_param: &<PCS::SRS as StructuredReferenceString>::ProverParam,
        transcript: Option<&mut RescueTranscript<P::BaseField>>,
    ) -> Result<Self::WithChallengesOutput, PolynomialError> {
        if self.instances.len() != self.witnesses.len() {
            return Err(PolynomialError::UnreachableError);
        }

        // We have to have a power of two number of instances for the batch opening to work.
        let new_length = self.instances.len().next_power_of_two();
        let num_vars = self.witnesses[0].poly.num_vars();
        let default_instance = PCSInstance::<PCS> {
            comm: Affine::<P>::default(),
            value: P::ScalarField::zero(),
            point: vec![P::ScalarField::zero(); num_vars],
        };
        let default_witness = PCSWitness::<PCS> {
            poly: Arc::new(
                DenseMultilinearExtension::<P::ScalarField>::from_evaluations_vec(
                    num_vars,
                    vec![P::ScalarField::zero(); 1 << num_vars],
                ),
            ),
            comm: Affine::<P>::default(),
            value: P::ScalarField::zero(),
            point: vec![P::ScalarField::zero(); num_vars],
        };
        self.instances.resize(new_length, default_instance);
        self.witnesses.resize(new_length, default_witness);

        let mut new_transcript =
            <RescueTranscript<P::BaseField> as Transcript>::new_transcript(b"accumulation");
        let transcript = if let Some(transcript) = transcript {
            transcript
        } else {
            &mut new_transcript
        };
        let polys = self.polynomials();
        let points = self.points();

        let (sumcheck_proof, eval, acc_witness_poly, accumulation_challenges_and_scalars) =
            challenges_and_scalars_from_mv_batch_open::<P>(&polys, &points, transcript)?;

        let comm = PCS::commit(prover_param, &acc_witness_poly)?;
        let mut new_accumulator = Self::new();
        let acc_point = accumulation_challenges_and_scalars.a_2.clone();
        new_accumulator.push(acc_witness_poly, comm, acc_point, eval);

        Ok((
            accumulation_challenges_and_scalars,
            sumcheck_proof,
            new_accumulator,
        ))
    }

    fn verify_accumulation(
        &self,
        old_instances: &[Self::Instance],
        proof: &Self::AccProof,
        transcript: Option<&mut RescueTranscript<P::BaseField>>,
    ) -> Result<(), PolynomialError> {
        if self.instances.len() != 1 {
            return Err(PolynomialError::ParameterError(
                "Can only verify a single accumulator is correct".to_string(),
            ));
        };
        let mut new_transcript =
            <RescueTranscript<P::BaseField> as Transcript>::new_transcript(b"accumulation");
        let transcript = if let Some(transcript) = transcript {
            transcript
        } else {
            &mut new_transcript
        };
        let points = old_instances
            .iter()
            .map(|i| i.point.clone())
            .collect::<Vec<_>>();
        let values = old_instances.iter().map(|i| i.value).collect::<Vec<_>>();
        let (eval, f_scalars, point) = mv_batch_verify::<P>(&points, &values, proof, transcript)?;
        let f_scalars_bigints = f_scalars
            .iter()
            .map(|f: &P::ScalarField| f.into_bigint())
            .collect::<Vec<_>>();
        let comms = old_instances.iter().map(|i| i.comm).collect::<Vec<_>>();
        let calc_comm = Projective::<P>::msm_bigint(&comms, &f_scalars_bigints).into_affine();
        if calc_comm != self.instances[0].comm
            || point != self.instances[0].point
            || eval != self.instances[0].value
        {
            return Err(PolynomialError::ParameterError(
                "Accumulator is incorrect".to_string(),
            ));
        }

        Ok(())
    }

    fn merge_accumulators(&mut self, other: &Self) {
        self.instances.extend(other.instances.clone());
        self.witnesses.extend(other.witnesses.clone());
    }

    fn open_witness(
        &self,
        prover_param: &<<PCS>::SRS as StructuredReferenceString>::ProverParam,
    ) -> Result<(<PCS>::Proof, P::ScalarField), PolynomialError> {
        if self.witnesses.len() != 1 {
            return Err(PolynomialError::ParameterError(
                "Can only open an accumulator consisting of a single witness".to_string(),
            ));
        }
        PCS::open(
            prover_param,
            &self.witnesses[0].poly,
            &self.witnesses[0].point,
        )
        .map_err(|_| PolynomialError::ParameterError("Inner PCS error with opening".to_string()))
    }

    /// Provide a single opening proof for all the polynomials in the accumulator.
    fn multi_open(
        &mut self,
        prover_param: &<PCS::SRS as StructuredReferenceString>::ProverParam,
    ) -> Result<PCS::BatchProof, PolynomialError> {
        let (proof, _) = PCS::batch_open(
            prover_param,
            &self.commitments(),
            &self.polynomials(),
            &self.points(),
        )
        .map_err(|_| PolynomialError::ParameterError("batch open failed".to_string()))?;
        Ok(proof)
    }
}

impl<E, P> AtomicAccumulator<UnivariateKzgPCS<E>> for UVAtomicAccumulator<UnivariateKzgPCS<E>>
where
    E: Pairing<G1Affine = Affine<P>>,
    P: HasTEForm<BaseField = E::BaseField, ScalarField = E::ScalarField>,
    P::BaseField: PrimeField + RescueParameter,
    E::ScalarField: EmulationConfig<E::BaseField>,
{
    type Instance = AtomicInstance<UnivariateKzgPCS<E>>;
    type AccProof = AtomicAccProof<UnivariateKzgPCS<E>>;
    type WithChallengesOutput = (
        AtomicAccumulationChallengesAndScalars<E::ScalarField>,
        Self,
        Self::AccProof,
    );

    fn new() -> Self {
        Self {
            instances: Vec::new(),
        }
    }

    fn push(
        &mut self,
        comm: E::G1Affine,
        point: E::ScalarField,
        value: E::ScalarField,
        opening_proof: UnivariateKzgProof<E>,
    ) {
        self.instances
            .push(AtomicInstance::new(comm, value, point, opening_proof));
    }

    fn merge_accumulators(&mut self, other: &Self) {
        self.instances.extend(other.instances.clone());
    }

    fn prove_accumulation<R: CryptoRng + Rng + RngCore>(
        &self,
        rng: &mut R,
        prover_param: &UnivariateProverParam<E>,
        transcript: Option<&mut RescueTranscript<P::BaseField>>,
    ) -> Result<(Self, Self::AccProof), PlonkError>
    where
        Self: Sized,
    {
        let (_, new_acc, acc_proof) = self.prove_accumulation_with_challenges_and_scalars::<R>(
            rng,
            prover_param,
            transcript,
        )?;
        Ok((new_acc, acc_proof))
    }

    fn prove_accumulation_with_challenges_and_scalars<R: CryptoRng + Rng + RngCore>(
        &self,
        rng: &mut R,
        prover_param: &UnivariateProverParam<E>,
        transcript: Option<&mut RescueTranscript<P::BaseField>>,
    ) -> Result<Self::WithChallengesOutput, PlonkError>
    where
        Self: Sized,
    {
        let mut new_transcript =
            <RescueTranscript<P::BaseField> as Transcript>::new_transcript(b"accumulation");
        let transcript = if let Some(transcript) = transcript {
            transcript
        } else {
            &mut new_transcript
        };

        let g = prover_param.powers_of_g[0];
        let beta_g = prover_param.powers_of_g[1];

        // Randomly generate an s for the accumulation proof.
        let s = E::ScalarField::rand(rng);
        let s_beta = (beta_g * s).into_affine();
        let s_g = (g * s).into_affine();
        for instance in self.instances.iter() {
            transcript.append_curve_point(b"commitment", &instance.comm)?;
            transcript.push_message(b"point", &instance.point)?;
            transcript.push_message(b"value", &instance.value)?;
            transcript.append_curve_point(b"opening_proof", &instance.opening_proof.proof)?;
        }

        transcript.append_curve_point(b"s_beta_g", &s_beta)?;
        transcript.append_curve_point(b"s_g", &s_g)?;

        let r = transcript.squeeze_scalar_challenge::<P>(b"r")?;
        let r_powers = cfg_into_iter!(0..=self.instances.len())
            .map(|i| r.pow([i as u64]))
            .collect::<Vec<_>>();

        let r_powers_bigints = cfg_iter!(r_powers)
            .map(|r_power| r_power.into_bigint())
            .collect::<Vec<_>>();

        // Produce a vec containing r^i * z_i.
        let z_r_powers = cfg_iter!(self.instances)
            .zip(cfg_iter!(r_powers))
            .map(|(instance, r_power)| instance.point * *r_power)
            .collect::<Vec<_>>();

        let z_r_powers_big_ints = cfg_iter!(z_r_powers)
            .map(|z_r_power| z_r_power.into_bigint())
            .collect::<Vec<_>>();

        // Calculate the sum of the evaluations multiplied by the relevant power of r.
        let minus_v_r_powers = cfg_iter!(self.instances)
            .zip(cfg_iter!(r_powers))
            .map(|(instance, r_power)| (instance.value * *r_power).neg())
            .sum::<P::ScalarField>();

        let comms = cfg_iter!(self.instances)
            .map(|instance| instance.comm)
            .collect::<Vec<_>>();
        // Extract the proof commitments from each instance.
        let mut proofs = cfg_iter!(self.instances)
            .map(|instance| instance.opening_proof.proof)
            .collect::<Vec<_>>();
        // Produce the list of scalars to be used in the multi-scalar multiplication for the new instance commitment.
        let scalars_comms = r_powers_bigints
            .iter()
            .copied()
            .take(comms.len())
            .chain([minus_v_r_powers.into_bigint()])
            .chain(z_r_powers_big_ints)
            .chain(r_powers_bigints.iter().skip(comms.len()).copied())
            .collect::<Vec<_>>();
        // Produce the list of bases to be used in the multi-scalar multiplication for the new instance commitment.
        let mut msm_comms = comms
            .iter()
            .copied()
            .chain([g])
            .chain(proofs.iter().copied())
            .collect::<Vec<_>>();
        msm_comms.push(s_beta);
        // Produce the list of bases to be used in the multi-scalar multiplication for the new proof.

        proofs.push(s_g);

        let out_comm = Projective::<P>::msm_bigint(&msm_comms, &scalars_comms).into_affine();
        let out_proof = Projective::<P>::msm_bigint(&proofs, &r_powers_bigints).into_affine();

        let mut new_acc = UVAtomicAccumulator::<UnivariateKzgPCS<E>>::new();
        new_acc.push(
            out_comm,
            P::ScalarField::zero(),
            P::ScalarField::zero(),
            UnivariateKzgProof::<E> { proof: out_proof },
        );

        let acc_proof = AtomicAccProof {
            s_beta_g: s_beta,
            s_g,
        };
        Ok((
            AtomicAccumulationChallengesAndScalars {
                r,
                r_powers,
                z_r_powers,
                minus_v_r_powers,
            },
            new_acc,
            acc_proof,
        ))
    }

    fn verify_accumulation(
        &self,
        old_accs: &[Self::AccProof],
        new_acc: &Self::AccProof,
        proof: &Self::AccProof,
        verifier_param: &UnivariateVerifierParam<E>,
        transcript: Option<&mut RescueTranscript<P::BaseField>>,
    ) -> Result<(), PlonkError>
    where
        P::BaseField: PrimeField + RescueParameter,
    {
        let mut new_transcript =
            <RescueTranscript<P::BaseField> as Transcript>::new_transcript(b"accumulation");
        let transcript = if let Some(transcript) = transcript {
            transcript
        } else {
            &mut new_transcript
        };

        // Add items to the transcript in the same order as the prover.
        let s_beta = proof.s_beta_g;
        let s_g = proof.s_g;
        for instance in self.instances.iter() {
            transcript.append_curve_point(b"commitment", &instance.comm)?;
            transcript.push_message(b"point", &instance.point)?;
            transcript.push_message(b"value", &instance.value)?;
            transcript.append_curve_point(b"opening_proof", &instance.opening_proof.proof)?;
        }

        for old_acc in old_accs.iter() {
            transcript.append_curve_point(b"commitment", &old_acc.s_beta_g)?;
            transcript.append_curve_point(b"opening_proof", &old_acc.s_g)?;
        }

        transcript.append_curve_point(b"s_beta_g", &s_beta)?;
        transcript.append_curve_point(b"s_g", &s_g)?;

        let r = transcript.squeeze_scalar_challenge::<P>(b"r")?;
        let r_powers = cfg_into_iter!(0..=self.instances.len() + old_accs.len())
            .map(|i| r.pow([i as u64]))
            .collect::<Vec<_>>();

        let r_powers_bigints = cfg_iter!(r_powers)
            .map(|r_power| r_power.into_bigint())
            .collect::<Vec<_>>();
        // Calculate sum_i -v_i * r^i.
        let minus_v_r_powers_bigints = cfg_iter!(self.instances)
            .zip(cfg_iter!(r_powers))
            .map(|(instance, r_power)| (-instance.value * *r_power))
            .sum::<P::ScalarField>()
            .into_bigint();

        // Produce a vec containing r^i * z_i.
        let z_r_powers_big_ints = cfg_iter!(self.instances)
            .zip(cfg_iter!(r_powers))
            .map(|(instance, r_power)| (instance.point * *r_power).into_bigint())
            .collect::<Vec<_>>();

        // Extract the commitments from the old instances
        let comms = cfg_iter!(self.instances)
            .map(|instance| instance.comm)
            .collect::<Vec<_>>();
        // Extract the commitments related to instances from the old accumulators.
        let acc_comms = cfg_iter!(old_accs)
            .map(|old_acc| old_acc.s_beta_g)
            .collect::<Vec<_>>();
        // Extract the proof commitments from each instance.
        let proofs = cfg_iter!(self.instances)
            .map(|instance| instance.opening_proof.proof)
            .collect::<Vec<_>>();
        // Extract the proof commitment from each of the old accumulators.
        let acc_proofs = cfg_iter!(old_accs)
            .map(|old_acc| old_acc.s_g)
            .collect::<Vec<_>>();
        // Produce the list of scalars to be used in the multi-scalar multiplication for the new instance commitment.
        let scalars_comms = r_powers_bigints
            .iter()
            .copied()
            .take(comms.len())
            .chain([minus_v_r_powers_bigints])
            .chain(z_r_powers_big_ints.iter().copied())
            .chain(r_powers_bigints.iter().skip(comms.len()).copied())
            .collect::<Vec<_>>();
        // Produce the list of bases to be used in the multi-scalar multiplication for the new instance commitment.
        let mut msm_comms = comms
            .iter()
            .copied()
            .chain(vec![verifier_param.g])
            .chain(proofs.iter().copied())
            .chain(acc_comms.iter().copied())
            .collect::<Vec<_>>();
        msm_comms.push(s_beta);
        // Produce the list of bases to be used in the multi-scalar multiplication for the new proof.
        let mut msm_proofs = proofs
            .iter()
            .copied()
            .chain(acc_proofs.iter().copied())
            .collect::<Vec<_>>();
        msm_proofs.push(s_g);

        let out_comm = Projective::<P>::msm_bigint(&msm_comms, &scalars_comms).into_affine();
        let out_proof = Projective::<P>::msm_bigint(&msm_proofs, &r_powers_bigints).into_affine();

        if out_comm != new_acc.s_beta_g || out_proof != new_acc.s_g {
            return Err(PlonkError::InvalidParameters(
                "Accumulator is incorrect".to_string(),
            ));
        }

        Ok(())
    }
}

/// Struct used to store the challenges and scalars generated during the
/// accumulation process. They are ultimately the t, a_1, a_2 challenges and
/// the coefficient of the f_i(X) from the multivariate polynomial batch
/// prove/verify protocol. See S3.8 of https://eprint.iacr.org/2022/1355.pdf.
#[derive(Default)]
#[allow(dead_code)]
pub struct AccumulationChallengesAndScalars<F: PrimeField> {
    /// List of t challenges.
    pub(crate) t: Vec<F>,
    /// SumCheck challenges.
    pub(crate) a_1: Vec<F>,
    pub(crate) a_2: Vec<F>,
    /// Coefficients of the f_i(X).
    pub(crate) coeffs: Vec<F>,
}

impl<F: PrimeField> AccumulationChallengesAndScalars<F> {
    /// Returns the number of challenges and scalars.
    pub fn len(&self) -> usize {
        self.t.len() + self.a_1.len() + self.a_2.len() + self.coeffs.len()
    }

    /// Getter for the t challenges.
    pub fn t(&self) -> &[F] {
        &self.t
    }

    /// Getter for the a_1 challenges.
    pub fn a_1(&self) -> &[F] {
        &self.a_1
    }

    /// Getter for the a_2 challenges.
    pub fn a_2(&self) -> &[F] {
        &self.a_2
    }

    /// Getter for the coeffs.
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    /// Returns a boolean indicating if the struct is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Struct used to store the challenges and scalars generated during the atomic
/// accumulation process.
#[derive(Default)]
#[allow(dead_code)]
pub struct AtomicAccumulationChallengesAndScalars<F: PrimeField> {
    pub(crate) r: F,
    pub(crate) r_powers: Vec<F>,
    pub(crate) z_r_powers: Vec<F>,
    pub(crate) minus_v_r_powers: F,
}

impl<F: PrimeField> AtomicAccumulationChallengesAndScalars<F> {
    /// Returns the number of challenges and scalars.
    pub fn len(&self) -> usize {
        self.r_powers.len() + self.z_r_powers.len() + 2
    }

    /// Getter for the r_powers.
    pub fn r_powers(&self) -> &[F] {
        &self.r_powers
    }

    /// Getter for the z_r_powers.
    pub fn z_r_powers(&self) -> &[F] {
        &self.z_r_powers
    }

    /// Getter for the minus_v_r_powers.
    pub fn minus_v_r_powers(&self) -> F {
        self.minus_v_r_powers
    }

    /// Getter for the r.
    pub fn r(&self) -> F {
        self.r
    }

    /// Returns a boolean indicating if the struct is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nightfall::{hops::univariate_ipa::UnivariateIpaPCS, mle::zeromorph::Zeromorph},
        transcript::RescueTranscript,
    };
    use ark_bls12_377::Bls12_377;
    use ark_ec::pairing::Pairing;
    use ark_poly::{
        evaluations::multivariate::MultilinearExtension, univariate::DensePolynomial,
        DenseUVPolynomial,
    };
    use ark_std::{vec, vec::Vec, One, UniformRand};
    use jf_primitives::pcs::{prelude::UnivariateKzgPCS, PolynomialCommitmentScheme};
    use jf_utils::test_rng;

    #[test]
    fn test_prover() -> Result<(), PlonkError> {
        test_prover_output_helper_zeromorph::<Bls12_377, _, _>()?;
        test_prover_output_helper_kzg::<Bls12_377, _>()?;
        Ok(())
    }

    fn test_prover_output_helper_zeromorph<E, P, F>() -> Result<(), PolynomialError>
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>>,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        E::ScalarField: EmulationConfig<F>,
        F: PrimeField + RescueParameter,
    {
        let rng = &mut test_rng();

        let num_vars = usize::rand(rng) % 5 + 3;

        let pp = Zeromorph::<UnivariateIpaPCS<E>>::gen_srs_for_testing(rng, num_vars).unwrap();
        let (ck, vk) = Zeromorph::<UnivariateIpaPCS<E>>::trim(pp, 0, Some(num_vars))?;
        let mut zeromorph_accumulator: MLAccumulator<Zeromorph<UnivariateIpaPCS<E>>> =
            MLAccumulator::<Zeromorph<UnivariateIpaPCS<E>>>::new();
        for _ in 0..100 {
            let poly = Arc::new(DenseMultilinearExtension::<E::ScalarField>::rand(
                num_vars, rng,
            ));
            let comm = Zeromorph::commit(&ck, &poly).unwrap();
            let point = vec![E::ScalarField::rand(rng); num_vars];
            let value = poly.evaluate(&point).unwrap();
            zeromorph_accumulator.push(poly, comm, point, value);
        }
        let mut transcript =
            <RescueTranscript<E::BaseField> as Transcript>::new_transcript(b"test");
        let (_proof, new_acc) =
            zeromorph_accumulator.prove_accumulation(&ck, Some(&mut transcript))?;

        let (ipa_proof, _evaluation) = Zeromorph::<UnivariateIpaPCS<E>>::open(
            &ck,
            &new_acc.witnesses[0].poly,
            &new_acc.witnesses[0].point,
        )
        .unwrap();

        assert!(Zeromorph::<UnivariateIpaPCS<E>>::verify(
            &vk,
            &new_acc.instances[0].comm,
            &new_acc.instances[0].point,
            &new_acc.instances[0].value,
            &ipa_proof
        )
        .unwrap());
        Ok(())
    }

    fn test_prover_output_helper_kzg<E, P>() -> Result<(), PlonkError>
    where
        E: Pairing<G1Affine = Affine<P>>,
        P: HasTEForm<BaseField = E::BaseField, ScalarField = E::ScalarField>,
        P::BaseField: RescueParameter + PrimeField,
        E::ScalarField: EmulationConfig<P::BaseField>,
    {
        let rng = &mut test_rng();

        let degree = 50;
        let mut kzg_atomic_accumulator = UVAtomicAccumulator::<UnivariateKzgPCS<E>>::new();
        let pp = UnivariateKzgPCS::<E>::gen_srs_for_testing(rng, degree).unwrap();
        let (ck, vk) = pp.trim(degree)?;

        for _ in 0..100 {
            let commit_in = <DensePolynomial<E::ScalarField> as DenseUVPolynomial<
                E::ScalarField,
            >>::rand(degree, rng);
            let comm = UnivariateKzgPCS::commit(&ck, &commit_in).unwrap();
            let point = E::ScalarField::rand(rng);
            let (opening_proof, value) = UnivariateKzgPCS::open(&ck, &commit_in, &point).unwrap();
            kzg_atomic_accumulator.push(comm, point, value, opening_proof);
        }
        let mut transcript =
            <RescueTranscript<P::BaseField> as Transcript>::new_transcript(b"test");
        let (accumulation, _witness) =
            kzg_atomic_accumulator.prove_accumulation(rng, &ck, Some(&mut transcript))?;

        let pairing_inputs_l: Vec<E::G1Prepared> = vec![
            accumulation.instances[0].comm.into(),
            accumulation.instances[0].opening_proof.proof.neg().into(),
        ];
        let pairing_inputs_r: Vec<E::G2Prepared> = vec![vk.h.into(), vk.beta_h.into()];

        let res = E::multi_pairing(pairing_inputs_l, pairing_inputs_r)
            .0
            .is_one();
        assert!(res);
        Ok(())
    }
}
