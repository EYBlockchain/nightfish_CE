//! This module contains code for verifying GKR proofs in the Plonk protocol.
use crate::{
    nightfall::{
        circuit::plonk_partial_verifier::{emulated_eq_x_r_eval_circuit, eq_x_r_eval_circuit},
        mle::subroutines::gkr::GKRProof,
    },
    transcript::{rescue::RescueTranscriptVar, CircuitTranscript},
};
use ark_ff::PrimeField;
use ark_std::{string::ToString, vec, vec::Vec};
use jf_primitives::rescue::RescueParameter;
use jf_relation::{
    errors::CircuitError,
    gadgets::{ecc::HasTEForm, EmulatedVariable, EmulationConfig},
    Circuit, PlonkCircuit, Variable,
};

use itertools::izip;

use super::{
    structs::{EmulatedSumCheckProofVar, SumCheckProofVar},
    sumcheck::SumCheckGadget,
};

/// Struct that represents a GKRProof in a circuit where the scalars used are all in the correct field.
pub struct GKRProofVar {
    /// The SumCheck proof variables.
    pub sumcheck_proof_vars: Vec<SumCheckProofVar>,
    /// The output evaluations of the polynomials after each round.
    pub evals: Vec<Vec<Variable>>,
    /// The final challenge point
    pub challenges: Vec<Variable>,
}

impl GKRProofVar {
    /// Construct a new GKRProofVar instance.
    pub fn new(
        sumcheck_proof_vars: Vec<SumCheckProofVar>,
        evals: Vec<Vec<Variable>>,
        challenges: Vec<Variable>,
    ) -> Self {
        Self {
            sumcheck_proof_vars,
            evals,
            challenges,
        }
    }

    /// Construct a new [`GKRProofVar`] instance from a [`GKRProof`].
    pub fn from_struct<F: PrimeField + RescueParameter>(
        proof: &GKRProof<F>,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<GKRProofVar, CircuitError> {
        let sumcheck_proof_vars = proof
            .sumcheck_proofs
            .iter()
            .map(|proof| circuit.sum_check_proof_to_var(proof))
            .collect::<Result<Vec<_>, _>>()?;
        let evals = proof
            .evals
            .iter()
            .map(|eval| {
                eval.iter()
                    .map(|eval| circuit.create_variable(*eval))
                    .collect::<Result<Vec<Variable>, CircuitError>>()
            })
            .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()?;

        let challenges = proof
            .challenge_point
            .iter()
            .map(|challenge| circuit.create_variable(*challenge))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        Ok(Self::new(sumcheck_proof_vars, evals, challenges))
    }

    /// Getter for the evals variable.
    pub fn evals(&self) -> &[Vec<Variable>] {
        &self.evals
    }

    /// Getter for the sumcheck_proof_vars variable.
    pub fn sumcheck_proofs(&self) -> &[SumCheckProofVar] {
        &self.sumcheck_proof_vars
    }

    /// Runs the verify procedure on a [`GKRProofVar`] struct without calculating any challenges.
    /// The output of this function is the evaluation returned by the deferred check, we assume lambda has been calculated elsewhere.
    pub fn verify_gkr_proof_with_challenges<F>(
        &self,
        circuit: &mut PlonkCircuit<F>,
        lambdas: &[Variable],
        r_challenges: &[Variable],
        sumcheck_challenges: &[Vec<Variable>],
    ) -> Result<Vec<Variable>, CircuitError>
    where
        F: PrimeField + RescueParameter,
    {
        if self.evals().is_empty() || r_challenges.is_empty() || lambdas.is_empty() {
            return Err(CircuitError::ParameterError(
                "No evaluations to verify".to_string(),
            ));
        }

        // Unwrap is safe because we have checked that the proof has evaluations.
        let first_evals = self.evals().first().unwrap();

        for eval_chunk in first_evals.chunks(4) {
            let p0 = eval_chunk[0];
            let p1 = eval_chunk[1];
            let q0 = eval_chunk[2];
            let q1 = eval_chunk[3];

            circuit.mul_add_gate(&[p0, q1, p1, q0, circuit.zero()], &[F::one(), F::one()])?;
            let q_claim = circuit.mul(q0, q1)?;
            circuit.non_zero_gate(q_claim)?;
        }

        let mut res = circuit.zero();
        let mut lambda;
        let mut lambda_powers = vec![];
        let mut r = *r_challenges.first().unwrap();
        let mut sc_eq_eval = circuit.zero();
        let mut challenge_point = vec![r];
        // Verify each sumcheck proof. We check that the output of the previous sumcheck proof is consistent with the input to the next using the
        // supplied evaluations.
        for (i, (proof, evals, l, r_challenge, sumcheck_challenges)) in izip!(
            self.sumcheck_proofs().iter(),
            self.evals().iter(),
            lambdas.iter(),
            r_challenges[1..].iter(),
            sumcheck_challenges.iter()
        )
        .enumerate()
        {
            // If its not the first round check that these evaluations line up with the expected evaluation from the previous round.
            if i != 0 {
                let sumcheck_eval = sum_check_evaluation(evals, &lambda_powers, circuit)?;
                circuit.mul_gate(sumcheck_eval, sc_eq_eval, res)?;
            }

            lambda = *l;
            let mut tmp_lambda_powers = vec![circuit.one()];
            for i in 0..first_evals.len() >> 1 {
                tmp_lambda_powers.push(circuit.mul(tmp_lambda_powers[i], lambda)?);
            }

            lambda_powers = tmp_lambda_powers;

            // Check that the initial evaluation of the sumcheck is correct.
            let calc_initial_eval = sumcheck_initial_evaluation(evals, &lambda_powers, r, circuit)?;
            circuit.enforce_equal(calc_initial_eval, proof.eval_var)?;

            res = circuit.verify_sum_check_with_challenges(proof)?;
            r = *r_challenge;
            sc_eq_eval = eq_x_r_eval_circuit(circuit, sumcheck_challenges, &challenge_point)?;
            challenge_point = [sumcheck_challenges.as_slice(), &[r]].concat();
        }

        let final_evals = self.evals().last().unwrap();
        let last_eval = sum_check_evaluation(final_evals, &lambda_powers, circuit)?;
        circuit.mul_gate(last_eval, sc_eq_eval, res)?;

        Ok(final_evals.to_vec())
    }

    /// Runs the verify procedure on a [`GKRProofVar`] struct.
    /// The output of this function is the evaluation returned by the deferred check, we assume lambda has been calculated elsewhere.
    pub fn verify_gkr_proof<P, F, C>(
        &self,
        circuit: &mut PlonkCircuit<F>,
        transcript: &mut C,
    ) -> Result<Vec<Variable>, CircuitError>
    where
        F: PrimeField + RescueParameter,
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField,
        C: CircuitTranscript<F>,
    {
        if self.evals().is_empty() {
            return Err(CircuitError::ParameterError(
                "No evaluations to verify".to_string(),
            ));
        }

        // Unwrap is safe because we have checked that the proof has evaluations.
        let first_evals = self.evals().first().unwrap();

        for eval_chunk in first_evals.chunks(4) {
            let p0 = eval_chunk[0];
            let p1 = eval_chunk[1];
            let q0 = eval_chunk[2];
            let q1 = eval_chunk[3];

            circuit.mul_add_gate(&[p0, q1, p1, q0, circuit.zero()], &[F::one(), F::one()])?;
            let q_claim = circuit.mul(q0, q1)?;
            circuit.non_zero_gate(q_claim)?;
        }

        let mut res = circuit.zero();
        let mut lambda;
        let mut lambda_powers = vec![];
        let mut r = transcript.squeeze_scalar_challenge::<P>(circuit)?;
        let mut sc_eq_eval = circuit.zero();
        let mut challenge_point = vec![r];
        // Verify each sumcheck proof. We check that the out put of the previous sumcheck proof is consistent with the input to the next using the
        // supplied evaluations.
        for (i, (proof, evals)) in
            izip!(self.sumcheck_proofs().iter(), self.evals().iter(),).enumerate()
        {
            // If its not the first round check that these evaluations line up with the expected evaluation from the previous round.
            if i != 0 {
                let sumcheck_eval = sum_check_evaluation(evals, &lambda_powers, circuit)?;
                circuit.mul_gate(sumcheck_eval, sc_eq_eval, res)?;
            }

            lambda = transcript.squeeze_scalar_challenge::<P>(circuit)?;
            let mut tmp_lambda_powers = vec![circuit.one()];
            for i in 0..first_evals.len() >> 1 {
                tmp_lambda_powers.push(circuit.mul(tmp_lambda_powers[i], lambda)?);
            }

            lambda_powers = tmp_lambda_powers;

            // Check that the initial evaluation of the sumcheck is correct.
            let init_eval = sumcheck_initial_evaluation(evals, &lambda_powers, r, circuit)?;
            circuit.enforce_equal(init_eval, proof.eval_var)?;

            res = circuit.verify_sum_check::<P, C>(proof, transcript)?;
            r = transcript.squeeze_scalar_challenge::<P>(circuit)?;
            sc_eq_eval = eq_x_r_eval_circuit(circuit, &proof.point_var, &challenge_point)?;
            challenge_point = [proof.point_var.as_slice(), &[r]].concat();
        }

        let final_evals = self.evals().last().unwrap();
        let last_eval = sum_check_evaluation(final_evals, &lambda_powers, circuit)?;
        circuit.mul_gate(last_eval, sc_eq_eval, res)?;

        Ok(final_evals.to_vec())
    }
}

fn sum_check_evaluation<F: PrimeField>(
    evals: &[Variable],
    lambda_powers: &[Variable],
    circuit: &mut PlonkCircuit<F>,
) -> Result<Variable, CircuitError> {
    let p0 = evals[0];
    let p1 = evals[1];
    let q0 = evals[2];
    let q1 = evals[3];
    let p_lambda = lambda_powers[0];
    let q_lambda = lambda_powers[1];
    let p_eval = circuit.mul_add(&[p0, q1, p1, q0], &[F::one(), F::one()])?;
    let q_eval = circuit.mul(q0, q1)?;
    let init_acc_var =
        circuit.mul_add(&[p_eval, p_lambda, q_eval, q_lambda], &[F::one(), F::one()])?;

    evals
        .chunks(4)
        .zip(lambda_powers.chunks(2))
        .skip(1)
        .try_fold(init_acc_var, |acc, (evals, lambdas)| {
            let p0 = evals[0];
            let p1 = evals[1];
            let q0 = evals[2];
            let q1 = evals[3];
            let p_lambda = lambdas[0];
            let q_lambda = lambdas[1];
            let p_eval = circuit.mul_add(&[p0, q1, p1, q0], &[F::one(), F::one()])?;
            let q_eval = circuit.mul(q0, q1)?;
            let tmp =
                circuit.mul_add(&[p_eval, p_lambda, q_eval, q_lambda], &[F::one(), F::one()])?;
            circuit.add(acc, tmp)
        })
}

fn sumcheck_initial_evaluation<F: PrimeField>(
    evals: &[Variable],
    lambda_powers: &[Variable],
    r: Variable,
    circuit: &mut PlonkCircuit<F>,
) -> Result<Variable, CircuitError> {
    let p0 = evals[0];
    let p1 = evals[1];
    let q0 = evals[2];
    let q1 = evals[3];
    let p_lambda = lambda_powers[0];
    let q_lambda = lambda_powers[1];
    let p_eval = circuit.gen_quad_poly(
        &[p0, r, p1, r],
        &[F::one(), F::zero(), F::zero(), F::zero()],
        &[-F::one(), F::one()],
        F::zero(),
    )?;
    let q_eval = circuit.gen_quad_poly(
        &[q0, r, q1, r],
        &[F::one(), F::zero(), F::zero(), F::zero()],
        &[-F::one(), F::one()],
        F::zero(),
    )?;
    let init_acc_var =
        circuit.mul_add(&[p_eval, p_lambda, q_eval, q_lambda], &[F::one(), F::one()])?;

    evals
        .chunks(4)
        .zip(lambda_powers.chunks(2))
        .skip(1)
        .try_fold(init_acc_var, |acc, (evals, lambdas)| {
            let p0 = evals[0];
            let p1 = evals[1];
            let q0 = evals[2];
            let q1 = evals[3];
            let p_lambda = lambdas[0];
            let q_lambda = lambdas[1];
            let p_eval = circuit.gen_quad_poly(
                &[p0, r, p1, r],
                &[F::one(), F::zero(), F::zero(), F::zero()],
                &[-F::one(), F::one()],
                F::zero(),
            )?;
            let q_eval = circuit.gen_quad_poly(
                &[q0, r, q1, r],
                &[F::one(), F::zero(), F::zero(), F::zero()],
                &[-F::one(), F::one()],
                F::zero(),
            )?;
            let tmp =
                circuit.mul_add(&[p_eval, p_lambda, q_eval, q_lambda], &[F::one(), F::one()])?;
            circuit.add(acc, tmp)
        })
}

/// Used to verify a GKR proof in a circuit where the scalars used are all in the wrong field.
pub struct EmulatedGKRProofVar<E: PrimeField> {
    /// The SumCheck proof variables.
    pub sumcheck_proof_vars: Vec<EmulatedSumCheckProofVar<E>>,
    /// The output evaluations of the polynomials after each round.
    pub evals: Vec<Vec<EmulatedVariable<E>>>,
    /// The final challenge point
    pub challenges: Vec<EmulatedVariable<E>>,
}

impl<E: PrimeField> EmulatedGKRProofVar<E> {
    /// Construct a new EmulatedGKRProofVar instance.
    pub fn new(
        sumcheck_proof_vars: Vec<EmulatedSumCheckProofVar<E>>,
        evals: Vec<Vec<EmulatedVariable<E>>>,
        challenges: Vec<EmulatedVariable<E>>,
    ) -> Self {
        Self {
            sumcheck_proof_vars,
            evals,
            challenges,
        }
    }

    /// Construct a new [`EmulatedGKRProofVar`] instance from a [`GKRProof`].
    pub fn from_proof<P>(
        circuit: &mut PlonkCircuit<P::BaseField>,
        proof: &GKRProof<P::ScalarField>,
    ) -> Result<Self, CircuitError>
    where
        P: HasTEForm<ScalarField = E>,
        E: PrimeField + EmulationConfig<P::BaseField> + RescueParameter,
        P::BaseField: PrimeField + RescueParameter + EmulationConfig<E>,
    {
        let sumcheck_proof_vars = proof
            .sumcheck_proofs
            .iter()
            .map(|proof| circuit.proof_to_emulated_var::<P>(proof))
            .collect::<Result<Vec<_>, _>>()?;
        let evals = proof
            .evals
            .iter()
            .map(|eval| {
                eval.iter()
                    .map(|eval| circuit.create_emulated_variable(*eval))
                    .collect::<Result<Vec<EmulatedVariable<P::ScalarField>>, CircuitError>>()
            })
            .collect::<Result<Vec<Vec<EmulatedVariable<P::ScalarField>>>, CircuitError>>()?;

        let challenges = proof
            .challenge_point
            .iter()
            .map(|challenge| circuit.create_emulated_variable(*challenge))
            .collect::<Result<Vec<EmulatedVariable<P::ScalarField>>, CircuitError>>()?;
        Ok(Self::new(sumcheck_proof_vars, evals, challenges))
    }

    /// Function to extract the challenges from a [`EmulatedGKRProofVar`] struct so they can be forwarded to another circuit for verifying arithmetic.
    /// The challenges are returned in the order, r_challenges, lambda_challenges, sumcheck_challenges.
    /// If the polynomials corresponding to the proof have n variables then r_challenges has length n + 1, lambda challenges has length n
    /// and sumcheck challenges has length 1/2 * n(n+1).
    pub fn extract_challenges<P, F, C>(
        &self,
        circuit: &mut PlonkCircuit<F>,
        transcript: &mut C,
    ) -> Result<Vec<Variable>, CircuitError>
    where
        F: PrimeField + RescueParameter,
        E: EmulationConfig<F> + RescueParameter,
        P: HasTEForm<BaseField = F, ScalarField = E>,
        C: CircuitTranscript<F>,
    {
        // First we extract the initial r challenge.
        let r_0 = transcript.squeeze_scalar_challenge::<P>(circuit)?;

        // Now we create vecs to store the r, lambda and sumcheck challenges.
        let mut r_challenges = vec![r_0];
        let mut lambda_challenges = vec![];
        let mut sumcheck_challenges = vec![];

        for sumcheck_proof in self.sumcheck_proof_vars.iter() {
            let lambda_round = transcript.squeeze_scalar_challenge::<P>(circuit)?;
            let sumcheck_challenges_round =
                circuit.recover_sumcheck_challenges::<P, C>(sumcheck_proof, transcript)?;
            let r_round = transcript.squeeze_scalar_challenge::<P>(circuit)?;

            lambda_challenges.push(lambda_round);
            sumcheck_challenges.push(sumcheck_challenges_round);
            r_challenges.push(r_round);
        }

        // Now we flatten into a single vec and return.
        Ok([
            r_challenges,
            lambda_challenges,
            sumcheck_challenges.concat(),
        ]
        .concat())
    }

    /// Getter function for the sumcheck proofs.
    pub fn sumcheck_proofs(&self) -> &[EmulatedSumCheckProofVar<E>] {
        &self.sumcheck_proof_vars
    }

    /// getter function for the claimed polynomial evaluations.
    pub fn evals(&self) -> &[Vec<EmulatedVariable<E>>] {
        &self.evals
    }

    /// Verifies a GKRProof in a circuit where the scalars used are all in the wrong field.
    pub fn verify_gkr_proof_emulated<P>(
        &self,
        circuit: &mut PlonkCircuit<P::BaseField>,
        transcript: &mut RescueTranscriptVar<P::BaseField>,
    ) -> Result<Vec<EmulatedVariable<E>>, CircuitError>
    where
        P: HasTEForm<ScalarField = E>,
        E: PrimeField + EmulationConfig<P::BaseField> + RescueParameter,
        P::BaseField: PrimeField + RescueParameter + EmulationConfig<E>,
    {
        if self.evals().is_empty() {
            return Err(CircuitError::ParameterError(
                "No evaluations to verify".to_string(),
            ));
        }

        // Unwrap is safe because we have checked that the proof has evaluations.
        let first_evals = self.evals().first().unwrap();
        let zero_var = circuit.emulated_zero();
        for eval_chunk in first_evals.chunks(4) {
            let p0 = &eval_chunk[0];
            let p1 = &eval_chunk[1];
            let q0 = &eval_chunk[2];
            let q1 = &eval_chunk[3];

            let p_term_one = circuit.emulated_mul(p0, q1)?;
            let p_term_two = circuit.emulated_mul(p1, q0)?;
            circuit.emulated_add_gate(&p_term_one, &p_term_two, &zero_var)?;

            let q_eval = circuit.emulated_mul(q0, q1)?;
            let zero_check = circuit.is_emulated_var_zero(&q_eval)?;
            circuit.enforce_false(zero_check.into())?;
        }

        let one_var = circuit.emulated_one();
        let mut res = zero_var.clone();
        let mut lambda;
        let mut lambda_powers = vec![];
        let r_native = transcript.squeeze_scalar_challenge::<P>(circuit)?;
        let mut r = circuit.to_emulated_variable(r_native)?;
        let mut sc_eq_eval = zero_var.clone();
        let mut challenge_point = vec![r.clone()];
        // Verify each sumcheck proof. We check that the out put of the previous sumcheck proof is consistent with the input to the next using the
        // supplied evaluations.
        for (i, (proof, evals)) in
            izip!(self.sumcheck_proofs().iter(), self.evals().iter(),).enumerate()
        {
            // If its not the first round check that these evaluations line up with the expected evaluation from the previous round.
            if i != 0 {
                let sumcheck_eval = sum_check_emulated_evaluation(evals, &lambda_powers, circuit)?;
                circuit.emulated_mul_gate(&sumcheck_eval, &sc_eq_eval, &res)?;
            }
            let lambda_native = transcript.squeeze_scalar_challenge::<P>(circuit)?;
            lambda = circuit.to_emulated_variable(lambda_native)?;

            let mut tmp_lambda_powers = vec![one_var.clone()];
            for i in 0..first_evals.len() >> 1 {
                tmp_lambda_powers.push(circuit.emulated_mul(&tmp_lambda_powers[i], &lambda)?);
            }

            lambda_powers = tmp_lambda_powers;

            // Check that the initial evaluation of the sumcheck is correct.
            let init_eval =
                sumcheck_emulated_initial_evaluation(evals, &lambda_powers, &r, circuit)?;
            circuit.enforce_emulated_var_equal(&init_eval, &proof.eval_var)?;

            res = circuit.verify_emulated_proof::<P>(proof, transcript)?;
            let r_native = transcript.squeeze_scalar_challenge::<P>(circuit)?;
            r = circuit.to_emulated_variable(r_native)?;
            sc_eq_eval = emulated_eq_x_r_eval_circuit(circuit, &proof.point_var, &challenge_point)?;
            challenge_point = [proof.point_var.as_slice(), &[r.clone()]].concat();
        }

        let final_evals = self.evals().last().unwrap();
        let last_eval = sum_check_emulated_evaluation(final_evals, &lambda_powers, circuit)?;
        circuit.emulated_mul_gate(&last_eval, &sc_eq_eval, &res)?;

        Ok(final_evals.to_vec())
    }
}

fn sum_check_emulated_evaluation<F: PrimeField, E: EmulationConfig<F>>(
    evals: &[EmulatedVariable<E>],
    lambda_powers: &[EmulatedVariable<E>],
    circuit: &mut PlonkCircuit<F>,
) -> Result<EmulatedVariable<E>, CircuitError> {
    let p0 = &evals[0];
    let p1 = &evals[1];
    let q0 = &evals[2];
    let q1 = &evals[3];
    let p_lambda = &lambda_powers[0];
    let q_lambda = &lambda_powers[1];

    let p_term_one = circuit.emulated_mul(p0, q1)?;
    let p_term_two = circuit.emulated_mul(p1, q0)?;
    let p_eval = circuit.emulated_add(&p_term_one, &p_term_two)?;

    let q_eval = circuit.emulated_mul(q0, q1)?;

    let acc_term_one = circuit.emulated_mul(&p_eval, p_lambda)?;
    let acc_term_two = circuit.emulated_mul(&q_eval, q_lambda)?;

    let init_acc_var = circuit.emulated_add(&acc_term_one, &acc_term_two)?;

    evals
        .chunks(4)
        .zip(lambda_powers.chunks(2))
        .skip(1)
        .try_fold(init_acc_var, |acc, (evals, lambdas)| {
            let p0 = &evals[0];
            let p1 = &evals[1];
            let q0 = &evals[2];
            let q1 = &evals[3];
            let p_lambda = &lambdas[0];
            let q_lambda = &lambdas[1];

            let p_term_one = circuit.emulated_mul(p0, q1)?;
            let p_term_two = circuit.emulated_mul(p1, q0)?;
            let p_eval = circuit.emulated_add(&p_term_one, &p_term_two)?;

            let q_eval = circuit.emulated_mul(q0, q1)?;

            let acc_term_one = circuit.emulated_mul(&p_eval, p_lambda)?;
            let acc_term_two = circuit.emulated_mul(&q_eval, q_lambda)?;

            let add_term = circuit.emulated_add(&acc_term_one, &acc_term_two)?;
            circuit.emulated_add(&acc, &add_term)
        })
}

fn sumcheck_emulated_initial_evaluation<F: PrimeField, E: EmulationConfig<F>>(
    evals: &[EmulatedVariable<E>],
    lambda_powers: &[EmulatedVariable<E>],
    r: &EmulatedVariable<E>,
    circuit: &mut PlonkCircuit<F>,
) -> Result<EmulatedVariable<E>, CircuitError> {
    let p0 = &evals[0];
    let p1 = &evals[1];
    let q0 = &evals[2];
    let q1 = &evals[3];
    let p_lambda = &lambda_powers[0];
    let q_lambda = &lambda_powers[1];

    let p1_m_p0 = circuit.emulated_sub(p1, p0)?;
    let r_p1p0 = circuit.emulated_mul(r, &p1_m_p0)?;
    let p_eval = circuit.emulated_add(p0, &r_p1p0)?;

    let q1_m_q0 = circuit.emulated_sub(q1, q0)?;
    let r_q1q0 = circuit.emulated_mul(r, &q1_m_q0)?;
    let q_eval = circuit.emulated_add(q0, &r_q1q0)?;

    let lambda_p_eval = circuit.emulated_mul(&p_eval, p_lambda)?;
    let lambda_q_eval = circuit.emulated_mul(&q_eval, q_lambda)?;

    let init_acc_var = circuit.emulated_add(&lambda_p_eval, &lambda_q_eval)?;

    evals
        .chunks(4)
        .zip(lambda_powers.chunks(2))
        .skip(1)
        .try_fold(init_acc_var, |acc, (evals, lambdas)| {
            let p0 = &evals[0];
            let p1 = &evals[1];
            let q0 = &evals[2];
            let q1 = &evals[3];
            let p_lambda = &lambdas[0];
            let q_lambda = &lambdas[1];

            let p1_m_p0 = circuit.emulated_sub(p1, p0)?;
            let r_p1p0 = circuit.emulated_mul(r, &p1_m_p0)?;
            let p_eval = circuit.emulated_add(p0, &r_p1p0)?;

            let q1_m_q0 = circuit.emulated_sub(q1, q0)?;
            let r_q1q0 = circuit.emulated_mul(r, &q1_m_q0)?;
            let q_eval = circuit.emulated_add(q0, &r_q1q0)?;

            let lambda_p_eval = circuit.emulated_mul(&p_eval, p_lambda)?;
            let lambda_q_eval = circuit.emulated_mul(&q_eval, q_lambda)?;

            let add_term = circuit.emulated_add(&lambda_p_eval, &lambda_q_eval)?;
            circuit.emulated_add(&acc, &add_term)
        })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        errors::PlonkError,
        nightfall::mle::{
            subroutines::{
                gkr::{
                    sum_check_evaluation as sc_eval, sumcheck_intial_evaluation, StructuredCircuit,
                },
                sumcheck::SumCheck,
                DeferredCheck, VPSumCheck,
            },
            utils::eq_eval,
        },
        transcript::{rescue::RescueTranscript, Transcript},
    };

    use ark_bn254::{g1::Config as BnConfig, Fq, Fr};
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{sync::Arc, One, Zero};

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

            // Extract challenges to pass in to the circuit.
            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let (lambdas, r_challenges) =
                extract_gkr_challenges::<BnConfig, RescueTranscript<Fq>>(&proof, &mut transcript)
                    .unwrap();
            let gkr_sumcheck_challenges = proof
                .sumcheck_proofs()
                .iter()
                .map(|p| p.point.clone())
                .collect::<Vec<Vec<Fr>>>();
            // Get the deferred check eval outside of the circuit.
            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let gkr_deferred_check = StructuredCircuit::verify::<BnConfig, RescueTranscript<Fq>>(
                &proof,
                &mut transcript,
            )
            .unwrap();
            let gkr_evals = gkr_deferred_check.evals();

            let mut circuit = PlonkCircuit::<Fr>::new_turbo_plonk();

            let lambda_vars = lambdas
                .iter()
                .map(|l| circuit.create_variable(*l))
                .collect::<Result<Vec<Variable>, CircuitError>>()
                .unwrap();
            let r_challenges = r_challenges
                .iter()
                .map(|r| circuit.create_variable(*r))
                .collect::<Result<Vec<Variable>, CircuitError>>()
                .unwrap();
            let gkr_sumcheck_challenges = gkr_sumcheck_challenges
                .iter()
                .map(|p| {
                    p.iter()
                        .map(|j| circuit.create_variable(*j))
                        .collect::<Result<Vec<Variable>, CircuitError>>()
                })
                .collect::<Result<Vec<Vec<Variable>>, CircuitError>>()
                .unwrap();

            let proof_var = GKRProofVar::from_struct(&proof, &mut circuit).unwrap();
            let circuit_gkr_evals = proof_var
                .verify_gkr_proof_with_challenges(
                    &mut circuit,
                    &lambda_vars,
                    &r_challenges,
                    &gkr_sumcheck_challenges,
                )
                .unwrap();

            circuit.check_circuit_satisfiability(&[]).unwrap();

            for (circuit_eval, eval) in circuit_gkr_evals.iter().zip(gkr_evals.iter()) {
                assert_eq!(circuit.witness(*circuit_eval).unwrap(), *eval);
            }
        }
    }

    pub type GKRChallenges<F> = (Vec<F>, Vec<F>);

    /// Extract the GKR challenges needed for circuit verification without hashing.
    pub fn extract_gkr_challenges<P, T>(
        proof: &GKRProof<P::ScalarField>,
        transcript: &mut T,
    ) -> Result<GKRChallenges<P::ScalarField>, PlonkError>
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

        for eval_chunk in first_evals.chunks(4) {
            let p0 = eval_chunk[0];
            let p1 = eval_chunk[1];
            let q0 = eval_chunk[2];
            let q1 = eval_chunk[3];

            let p_claim = P::ScalarField::zero() == (p0 * q1 + p1 * q0);
            let q_claim = P::ScalarField::zero() != (q0 * q1);

            let both_claims = p_claim && q_claim;
            // Return an error if the constant values `claimed_p` and `claimed_q` do not
            // match the the values we have used for proving.
            if !both_claims {
                return Err(PlonkError::InvalidParameters(
                    "Claimed values do not match the provided evaluations".to_string(),
                ));
            }
        }

        let mut res = DeferredCheck::default();
        let mut lambda = P::ScalarField::zero();
        let mut lambdas = Vec::<P::ScalarField>::with_capacity(proof.sumcheck_proofs().len());
        let mut r = transcript.squeeze_scalar_challenge::<P>(b"r_0")?;
        let mut sc_eq_eval = P::ScalarField::zero();
        let mut challenge_point = vec![r];
        let mut r_challenges =
            Vec::<P::ScalarField>::with_capacity(proof.sumcheck_proofs().len() + 1);
        r_challenges.push(r);
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
                let expected_eval = sc_eval(evals, lambda) * sc_eq_eval;
                if expected_eval != res.eval {
                    return Err(PlonkError::InvalidParameters(
                        "Sumcheck evaluation does not match expected value".to_string(),
                    ));
                }
            }

            lambda = transcript.squeeze_scalar_challenge::<P>(b"lambda")?;
            lambdas.push(lambda);
            // Check that the initial evaluation of the sumcheck is correct.
            let initial_eval = sumcheck_intial_evaluation(evals, lambda, r);
            if proof.eval != initial_eval {
                return Err(PlonkError::InvalidParameters(
                    "Initial sumcheck evaluation does not match expected value".to_string(),
                ));
            }

            let deferred_check = VPSumCheck::<P>::verify(proof, transcript)?;
            r = transcript.squeeze_scalar_challenge::<P>(b"r")?;
            r_challenges.push(r);
            sc_eq_eval = eq_eval(&deferred_check.point, &challenge_point)?;
            challenge_point = [deferred_check.point.as_slice(), &[r]].concat();
            res = deferred_check;
        }
        let final_evals = proof.evals().last().unwrap();
        let expected_eval = sc_eval(final_evals, lambda) * sc_eq_eval;
        if expected_eval != res.eval {
            return Err(PlonkError::InvalidParameters(
                "Sumcheck evaluation does not match expected value".to_string(),
            ));
        }
        // Unwrap is safe because we checked the eval list was non-empty earlier
        Ok((lambdas, r_challenges))
    }

    #[test]
    fn test_prove_and_verify_emulated() {
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

            // Get the deferred check eval outside of the circuit.
            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let gkr_deferred_check = StructuredCircuit::verify::<BnConfig, RescueTranscript<Fq>>(
                &proof,
                &mut transcript,
            )
            .unwrap();
            let gkr_evals = gkr_deferred_check.evals();

            let mut circuit = PlonkCircuit::<Fq>::new_ultra_plonk(8);

            let proof_var =
                EmulatedGKRProofVar::from_proof::<BnConfig>(&mut circuit, &proof).unwrap();
            let mut transcript = RescueTranscriptVar::<Fq>::new_transcript(&mut circuit);
            let circuit_gkr_evals = proof_var
                .verify_gkr_proof_emulated::<BnConfig>(&mut circuit, &mut transcript)
                .unwrap();

            circuit.check_circuit_satisfiability(&[]).unwrap();

            for (circuit_eval, eval) in circuit_gkr_evals.iter().zip(gkr_evals.iter()) {
                assert_eq!(circuit.emulated_witness(circuit_eval).unwrap(), *eval);
            }
            ark_std::println!("constraints: {}", circuit.num_gates());
        }
    }
}
