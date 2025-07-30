//! The construction of the sumcheck circuit
use super::structs::{
    EmulatedPolyOracleVar, EmulatedSumCheckProofVar, PolyOracleVar, SumCheckProofVar,
};
use crate::{
    nightfall::mle::subroutines::{PolyOracle, SumCheckProof},
    transcript::{rescue::RescueTranscriptVar, CircuitTranscript},
};

use ark_ff::{Field, PrimeField};
use ark_std::{string::ToString, vec, vec::Vec};
use jf_primitives::rescue::RescueParameter;
use jf_relation::{
    errors::CircuitError,
    gadgets::{ecc::HasTEForm, EmulatedVariable, EmulationConfig},
    Circuit, PlonkCircuit, Variable,
};

/// A trait with methods that are required for verifying a SumCheck proof in a circuit context.
pub trait SumCheckGadget<F: PrimeField + RescueParameter> {
    /// Takes in a polynomial oracle,
    /// returns the variable of the oracle.
    fn poly_oracle_to_var(
        &mut self,
        poly_oracle: &PolyOracle<F>,
    ) -> Result<PolyOracleVar, CircuitError>;
    /// Circuit to evaluate a polynomial oracle at a given point,
    /// returns the variable of the evaluation.
    fn evaluate_poly_oracle(
        &mut self,
        poly_oracle_var: &PolyOracleVar,
        point_var: &Variable,
    ) -> Result<Variable, CircuitError>;
    /// Takes in a sum check proof,
    /// returns the variable of the proof.
    fn sum_check_proof_to_var(
        &mut self,
        sum_check_proof: &SumCheckProof<F, PolyOracle<F>>,
    ) -> Result<SumCheckProofVar, CircuitError>;
    /// Circuit to verify a sum check proof,
    /// returns the variable of the final evaluation of the SumCheck at `point`.
    fn verify_sum_check<P, C>(
        &mut self,
        sum_check_proof_var: &SumCheckProofVar,
        transcript: &mut C,
    ) -> Result<Variable, CircuitError>
    where
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField,
        C: CircuitTranscript<F>;

    /// Circuit to verify a sum check proof where the challenges have been pre-calculated,
    /// returns the variable of the final evaluation of the SumCheck at `point`.
    fn verify_sum_check_with_challenges(
        &mut self,
        sum_check_proof_var: &SumCheckProofVar,
    ) -> Result<Variable, CircuitError>;

    /// Creates a [`EmulatedSumCheckProofVar`] from a [`SumCheckProof`]. Used to fully verify the proof in one circuit.
    fn proof_to_emulated_var<P>(
        &mut self,
        proof: &SumCheckProof<P::ScalarField, PolyOracle<P::ScalarField>>,
    ) -> Result<EmulatedSumCheckProofVar<P::ScalarField>, CircuitError>
    where
        P: HasTEForm<BaseField = F>,
        P::ScalarField: EmulationConfig<F> + RescueParameter;

    /// Verifies an [`EmulatedSumCheckProofVar`] in a circuit context.
    fn verify_emulated_proof<P>(
        &mut self,
        proof: &EmulatedSumCheckProofVar<P::ScalarField>,
        transcript: &mut RescueTranscriptVar<F>,
    ) -> Result<EmulatedVariable<P::ScalarField>, CircuitError>
    where
        P: HasTEForm<BaseField = F>,
        P::ScalarField: EmulationConfig<F> + RescueParameter;

    /// Makes a [`EmulatedPolyOracleVar<E>`] from a [`PolyOracle<E>`].
    /// This is used when verifying transcript hashes in a circuit context and `E` is field that implements
    /// [`EmulationConfig<F>`].
    fn poly_oracle_to_emulated_var<E>(
        &mut self,
        poly_oracle: &PolyOracle<E>,
    ) -> Result<EmulatedPolyOracleVar<E>, CircuitError>
    where
        E: PrimeField + EmulationConfig<F>;

    /// Circuit gadget to verify the challenges in a sumcheck proof.
    fn verify_challenges<E>(
        &mut self,
        poly_oracles: &[EmulatedPolyOracleVar<E::ScalarField>],
        challenges: &[EmulatedVariable<E::ScalarField>],
        transcript: &mut RescueTranscriptVar<F>,
    ) -> Result<(), CircuitError>
    where
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
        F: PrimeField + RescueParameter;

    /// Circuit gadget to verify the challenges in a sumcheck proof without enforcing equality on the final point.
    /// Used when verifying GKR proofs.
    fn verify_challenges_no_enforce<E>(
        &mut self,
        poly_oracles: &[EmulatedPolyOracleVar<E::ScalarField>],
        transcript: &mut RescueTranscriptVar<F>,
    ) -> Result<(), CircuitError>
    where
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
        F: PrimeField + RescueParameter;

    /// Circuit gadget used to recover the challenges from a SumCheck proof defined over a non-native field.
    fn recover_sumcheck_challenges<E, C>(
        &mut self,
        proof: &EmulatedSumCheckProofVar<E::ScalarField>,
        transcript: &mut C,
    ) -> Result<Vec<Variable>, CircuitError>
    where
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
        C: CircuitTranscript<F>;

    /// Evaluates an [`EmulatedPolyOracleVar`] in a circuit context.
    fn evaluate_emulated_poly_oracle<E>(
        &mut self,
        poly_oracle_var: &EmulatedPolyOracleVar<E::ScalarField>,
        point_var: &EmulatedVariable<E::ScalarField>,
    ) -> Result<EmulatedVariable<E::ScalarField>, CircuitError>
    where
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F> + RescueParameter;
}

impl<F> SumCheckGadget<F> for PlonkCircuit<F>
where
    F: PrimeField + RescueParameter,
{
    fn poly_oracle_to_var(
        &mut self,
        poly_oracle: &PolyOracle<F>,
    ) -> Result<PolyOracleVar, CircuitError> {
        let mut poly_oracle_evaluations_var: Vec<Variable> = Vec::new();
        for eval in poly_oracle.evaluations.iter() {
            poly_oracle_evaluations_var.push(self.create_variable(*eval)?);
        }
        let mut poly_oracle_weights_var: Vec<Variable> = Vec::new();
        for weight in poly_oracle.weights.iter() {
            poly_oracle_weights_var.push(self.create_variable(*weight)?);
        }

        Ok(PolyOracleVar {
            evaluations_var: poly_oracle_evaluations_var,
            weights_var: poly_oracle_weights_var,
        })
    }

    fn evaluate_poly_oracle(
        &mut self,
        poly_oracle_var: &PolyOracleVar,
        point_var: &Variable,
    ) -> Result<Variable, CircuitError> {
        let degree = poly_oracle_var.evaluations_var.len() - 1;
        let mut products_var: Vec<Variable> = Vec::new();

        // products vector
        for i in 0..=degree {
            let x_minus_i_inv = (self.witness(*point_var)? - F::from(i as u64 + 2))
                .inverse()
                .ok_or(CircuitError::ParameterError("Inverse Failed".to_string()))?;
            let x_minus_i_inv_var = self.create_variable(x_minus_i_inv)?;
            // We constrain `(point - (i+2)) * x_minus_i_inv == 1`.
            self.mul_add_gate(
                &[
                    *point_var,
                    x_minus_i_inv_var,
                    self.one(),
                    x_minus_i_inv_var,
                    self.one(),
                ],
                &[F::one(), -F::from(i as u64 + 2)],
            )?;
            products_var.push(self.mul(x_minus_i_inv_var, poly_oracle_var.weights_var[i])?);
        }

        // numerator
        let numerator_var = if degree == 0 {
            self.mul(products_var[0], poly_oracle_var.evaluations_var[0])?
        } else {
            let mut numerator_var = self.mul_add(
                &[
                    products_var[0],
                    poly_oracle_var.evaluations_var[0],
                    products_var[1],
                    poly_oracle_var.evaluations_var[1],
                ],
                &[F::one(), F::one()],
            )?;
            for (product_var, evaluation_var) in products_var
                .iter()
                .zip(poly_oracle_var.evaluations_var.iter())
                .skip(2)
            {
                numerator_var = self.mul_add(
                    &[*product_var, *evaluation_var, numerator_var, self.one()],
                    &[F::one(), F::one()],
                )?;
            }
            numerator_var
        };

        // denominator
        let denominator_var = self.lin_comb(
            &vec![F::one(); products_var.len()],
            &F::zero(),
            &products_var,
        )?;

        let denominator_inv = (self.witness(denominator_var)?)
            .inverse()
            .ok_or(CircuitError::ParameterError("Inverse Failed".to_string()))?;
        let denominator_inv_var = self.create_variable(denominator_inv)?;
        self.mul_gate(denominator_var, denominator_inv_var, self.one())?;

        self.mul(numerator_var, denominator_inv_var)
    }

    fn sum_check_proof_to_var(
        &mut self,
        sum_check_proof: &SumCheckProof<F, PolyOracle<F>>,
    ) -> Result<SumCheckProofVar, CircuitError> {
        let eval_var = self.create_variable(sum_check_proof.eval)?;
        let mut oracles_var: Vec<PolyOracleVar> = Vec::new();
        for oracle in sum_check_proof.oracles.iter() {
            oracles_var.push(self.poly_oracle_to_var(oracle)?);
        }
        let mut r_0_evals_var: Vec<Variable> = Vec::new();
        for r_0_eval in sum_check_proof.r_0_evals.iter() {
            r_0_evals_var.push(self.create_variable(*r_0_eval)?);
        }
        let mut point_var: Vec<Variable> = Vec::new();
        for challenge in sum_check_proof.point.iter() {
            point_var.push(self.create_variable(*challenge)?);
        }

        Ok(SumCheckProofVar {
            eval_var,
            oracles_var,
            r_0_evals_var,
            point_var,
        })
    }

    fn proof_to_emulated_var<P>(
        &mut self,
        proof: &SumCheckProof<P::ScalarField, PolyOracle<P::ScalarField>>,
    ) -> Result<EmulatedSumCheckProofVar<P::ScalarField>, CircuitError>
    where
        P: HasTEForm<BaseField = F>,
        P::ScalarField: EmulationConfig<F> + RescueParameter,
    {
        let eval_var = self.create_emulated_variable(proof.eval)?;
        let mut oracles_var: Vec<EmulatedPolyOracleVar<P::ScalarField>> = Vec::new();
        for oracle in proof.oracles.iter() {
            oracles_var.push(self.poly_oracle_to_emulated_var(oracle)?);
        }
        let mut r_0_evals_var: Vec<EmulatedVariable<P::ScalarField>> = Vec::new();
        for r_0_eval in proof.r_0_evals.iter() {
            r_0_evals_var.push(self.create_emulated_variable(*r_0_eval)?);
        }
        let mut point_var: Vec<EmulatedVariable<P::ScalarField>> = Vec::new();
        for challenge in proof.point.iter() {
            point_var.push(self.create_emulated_variable(*challenge)?);
        }
        Ok(EmulatedSumCheckProofVar {
            eval_var,
            oracles_var,
            r_0_evals_var,
            point_var,
        })
    }

    fn evaluate_emulated_poly_oracle<E>(
        &mut self,
        poly_oracle_var: &EmulatedPolyOracleVar<E::ScalarField>,
        point_var: &EmulatedVariable<E::ScalarField>,
    ) -> Result<EmulatedVariable<E::ScalarField>, CircuitError>
    where
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
    {
        let degree = poly_oracle_var.evaluations_var.len() - 1;
        let mut products_var: Vec<EmulatedVariable<E::ScalarField>> = Vec::new();

        // products vector
        for i in 0..=degree {
            let x_minus_i_var =
                self.emulated_sub_constant(point_var, E::ScalarField::from(i as u64 + 2))?;
            let x_minus_i_inv = (self.emulated_witness(&x_minus_i_var)?)
                .inverse()
                .ok_or(CircuitError::ParameterError("Inverse Failed".to_string()))?;
            let x_minus_i_inv_var = self.create_emulated_variable(x_minus_i_inv)?;
            self.emulated_mul_gate(&x_minus_i_var, &x_minus_i_inv_var, &self.emulated_one())?;
            products_var
                .push(self.emulated_mul(&x_minus_i_inv_var, &poly_oracle_var.weights_var[i])?);
        }

        // numerator
        let mut numerator_var =
            self.emulated_mul(&products_var[0], &poly_oracle_var.evaluations_var[0])?;
        for (product_var, evaluation_var) in products_var
            .iter()
            .zip(poly_oracle_var.evaluations_var.iter())
            .skip(1)
        {
            let tmp = self.emulated_mul(product_var, evaluation_var)?;
            numerator_var = self.emulated_add(&tmp, &numerator_var)?;
        }

        // denominator
        let mut denominator_var = products_var[0].clone();
        for product_var in products_var.iter().skip(1) {
            denominator_var = self.emulated_add(&denominator_var, product_var)?;
        }

        let denominator_inv = (self.emulated_witness(&denominator_var)?)
            .inverse()
            .ok_or(CircuitError::ParameterError("Inverse Failed".to_string()))?;
        let denominator_inv_var =
            self.create_emulated_variable::<E::ScalarField>(denominator_inv)?;
        self.emulated_mul_gate(&denominator_var, &denominator_inv_var, &self.emulated_one())?;
        let result_var = self.emulated_mul(&numerator_var, &denominator_inv_var)?;

        Ok(result_var)
    }

    fn verify_emulated_proof<P>(
        &mut self,
        proof: &EmulatedSumCheckProofVar<P::ScalarField>,
        transcript: &mut RescueTranscriptVar<F>,
    ) -> Result<EmulatedVariable<P::ScalarField>, CircuitError>
    where
        P: HasTEForm<BaseField = F>,
        P::ScalarField: EmulationConfig<F> + RescueParameter,
    {
        let num_rounds = proof.point_var.len();
        let mut eval_var = proof.eval_var.clone();
        for i in 0..num_rounds {
            for evaluation in proof.oracles_var[i].evaluations_var.iter() {
                transcript.push_emulated_variable(evaluation, self)?;
            }
            let challenge_var = transcript.squeeze_scalar_challenge::<P>(self)?;
            let challenge = self.to_emulated_variable(challenge_var)?;
            let r_1_eval_var = self.emulated_sub(&eval_var, &proof.r_0_evals_var[i])?;
            let r_alpha_eval_var =
                self.evaluate_emulated_poly_oracle::<P>(&proof.oracles_var[i], &challenge)?;

            let r_alpha_eval_times_alpha_var =
                self.emulated_mul(&r_alpha_eval_var, &proof.point_var[i])?;
            let tmp1 = self.emulated_mul(&challenge, &r_alpha_eval_times_alpha_var)?;
            let eval_var_1 = self.emulated_sub(&r_alpha_eval_times_alpha_var, &tmp1)?;

            let tmp2 = self.emulated_mul(&proof.r_0_evals_var[i], &challenge)?;
            let eval_var_2 = self.emulated_sub(&proof.r_0_evals_var[i], &tmp2)?;

            let eval_var_3 = self.emulated_mul(&challenge, &r_1_eval_var)?;
            let tmp3 = self.emulated_add(&eval_var_1, &eval_var_2)?;
            eval_var = self.emulated_add(&tmp3, &eval_var_3)?;
        }
        Ok(eval_var)
    }

    /// This function verifies Sumcheck over a native field, it is mostly for completeness.
    fn verify_sum_check<P, C>(
        &mut self,
        sum_check_proof_var: &SumCheckProofVar,
        transcript: &mut C,
    ) -> Result<Variable, CircuitError>
    where
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField,
        C: CircuitTranscript<F>,
    {
        let num_rounds = sum_check_proof_var.point_var.len();
        let mut eval_var = sum_check_proof_var.eval_var;
        for i in 0..num_rounds {
            for evaluation in sum_check_proof_var.oracles_var[i].evaluations_var.iter() {
                transcript.push_variable(evaluation)?;
            }
            let challenge = transcript.squeeze_scalar_challenge::<P>(self)?;

            let r_alpha_eval_var =
                self.evaluate_poly_oracle(&sum_check_proof_var.oracles_var[i], &challenge)?;

            let r_alpha_eval_times_alpha_var =
                self.mul(r_alpha_eval_var, sum_check_proof_var.point_var[i])?;
            let eval_var_1 = self.mul_add(
                &[
                    self.one(),
                    r_alpha_eval_times_alpha_var,
                    challenge,
                    r_alpha_eval_times_alpha_var,
                ],
                &[F::one(), -F::one()],
            )?;

            let eval_var_2 = self.mul_add(
                &[
                    self.one(),
                    sum_check_proof_var.r_0_evals_var[i],
                    challenge,
                    sum_check_proof_var.r_0_evals_var[i],
                ],
                &[F::one(), -F::one()],
            )?;

            let eval_var_3 = self.mul_add(
                &[
                    challenge,
                    eval_var,
                    challenge,
                    sum_check_proof_var.r_0_evals_var[i],
                ],
                &[F::one(), -F::one()],
            )?;

            eval_var = self.sum(&[eval_var_1, eval_var_2, eval_var_3])?;
        }
        Ok(eval_var)
    }

    fn verify_sum_check_with_challenges(
        &mut self,
        sum_check_proof_var: &SumCheckProofVar,
    ) -> Result<Variable, CircuitError> {
        let mut eval_var = sum_check_proof_var.eval_var;
        let challenges = &sum_check_proof_var.point_var;
        for (i, challenge) in challenges.iter().enumerate() {
            let r_alpha_eval_var =
                self.evaluate_poly_oracle(&sum_check_proof_var.oracles_var[i], challenge)?;

            let r_alpha_eval_times_alpha_var = self.mul(r_alpha_eval_var, *challenge)?;
            let eval_var_1 = self.mul_add(
                &[
                    self.one(),
                    r_alpha_eval_times_alpha_var,
                    *challenge,
                    r_alpha_eval_times_alpha_var,
                ],
                &[F::one(), -F::one()],
            )?;

            let eval_var_2 = self.mul_add(
                &[
                    self.one(),
                    sum_check_proof_var.r_0_evals_var[i],
                    *challenge,
                    sum_check_proof_var.r_0_evals_var[i],
                ],
                &[F::one(), -F::one()],
            )?;

            let eval_var_3 = self.mul_add(
                &[
                    *challenge,
                    eval_var,
                    *challenge,
                    sum_check_proof_var.r_0_evals_var[i],
                ],
                &[F::one(), -F::one()],
            )?;

            eval_var = self.sum(&[eval_var_1, eval_var_2, eval_var_3])?;
        }
        Ok(eval_var)
    }

    fn poly_oracle_to_emulated_var<E>(
        &mut self,
        poly_oracle: &PolyOracle<E>,
    ) -> Result<EmulatedPolyOracleVar<E>, CircuitError>
    where
        E: PrimeField + EmulationConfig<F>,
    {
        let mut poly_oracle_evaluations_var: Vec<EmulatedVariable<E>> = Vec::new();
        for eval in poly_oracle.evaluations.iter() {
            poly_oracle_evaluations_var.push(self.create_emulated_variable(*eval)?);
        }
        let mut poly_oracle_weights_var: Vec<EmulatedVariable<E>> = Vec::new();
        for weight in poly_oracle.weights.iter() {
            poly_oracle_weights_var.push(self.create_emulated_variable(*weight)?);
        }

        Ok(EmulatedPolyOracleVar {
            evaluations_var: poly_oracle_evaluations_var,
            weights_var: poly_oracle_weights_var,
        })
    }

    fn recover_sumcheck_challenges<E, C>(
        &mut self,
        proof: &EmulatedSumCheckProofVar<E::ScalarField>,
        transcript: &mut C,
    ) -> Result<Vec<Variable>, CircuitError>
    where
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
        C: CircuitTranscript<F>,
    {
        let mut calc_challenges = vec![];

        for evaluations in proof
            .oracles_var
            .iter()
            .map(|oracle| &oracle.evaluations_var)
        {
            for evaluation in evaluations {
                transcript.push_emulated_variable(evaluation, self)?;
            }
            let tmp = transcript.squeeze_scalar_challenge::<E>(self)?;
            calc_challenges.push(tmp);
        }

        Ok(calc_challenges)
    }

    fn verify_challenges<E>(
        &mut self,
        poly_oracles: &[EmulatedPolyOracleVar<E::ScalarField>],
        challenges: &[EmulatedVariable<E::ScalarField>],
        transcript: &mut RescueTranscriptVar<F>,
    ) -> Result<(), CircuitError>
    where
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
        F: PrimeField + RescueParameter,
    {
        let mut calc_challenges = vec![];

        for evaluations in poly_oracles.iter().map(|oracle| &oracle.evaluations_var) {
            for evaluation in evaluations {
                transcript.push_emulated_variable(evaluation, self)?;
            }
            let tmp = transcript.squeeze_scalar_challenge::<E>(self)?;
            calc_challenges.push(tmp);
        }

        for (calc_challenge, challenge) in calc_challenges.iter().zip(challenges) {
            let calc_challenge = self.to_emulated_variable(*calc_challenge)?;
            self.enforce_emulated_var_equal(&calc_challenge, challenge)?;
        }

        Ok(())
    }

    fn verify_challenges_no_enforce<E>(
        &mut self,
        poly_oracles: &[EmulatedPolyOracleVar<E::ScalarField>],
        transcript: &mut RescueTranscriptVar<F>,
    ) -> Result<(), CircuitError>
    where
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
        F: PrimeField + RescueParameter,
    {
        for evaluations in poly_oracles.iter().map(|oracle| &oracle.evaluations_var) {
            for evaluation in evaluations {
                transcript.push_emulated_variable(evaluation, self)?;
            }
            let _ = transcript.squeeze_scalar_challenge::<E>(self)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        nightfall::mle::{
            subroutines::{
                sumcheck::{Oracle, SumCheck},
                VPSumCheck,
            },
            virtual_polynomial::VirtualPolynomial,
        },
        transcript::{RescueTranscript, Transcript},
    };
    use ark_bls12_377::Bls12_377;

    use ark_ec::pairing::Pairing;
    use ark_poly::{
        univariate::DensePolynomial, DenseMultilinearExtension, DenseUVPolynomial,
        MultilinearExtension, Polynomial,
    };
    use ark_std::{
        rand::distributions::{Distribution, Uniform},
        string::ToString,
        sync::Arc,
        test_rng,
        vec::Vec,
        UniformRand,
    };
    use jf_primitives::rescue::RescueParameter;
    use jf_relation::gadgets::ecc::HasTEForm;
    use nf_curves::grumpkin::short_weierstrass::SWGrumpkin;

    fn test_poly_oracle_circuit_template<E>() -> Result<(), CircuitError>
    where
        E: Pairing,
        E::ScalarField: PrimeField + RescueParameter,
    {
        let mut rng = test_rng();
        for _ in 0..10 {
            let degree = (usize::rand(&mut rng) % (5)) + 5;
            let poly = <DensePolynomial<E::ScalarField> as DenseUVPolynomial<E::ScalarField>>::rand(
                degree, &mut rng,
            );
            let point = E::ScalarField::rand(&mut rng);
            let poly_oracle = PolyOracle::<E::ScalarField>::from_poly(&poly).map_err(|_| {
                CircuitError::ParameterError("PolyOracle Creation Failed".to_string())
            })?;
            let evaluation = poly.evaluate(&point);
            let mut circuit = PlonkCircuit::<E::ScalarField>::new_turbo_plonk();
            let point_var = circuit.create_variable(point)?;
            let poly_oracle_var = circuit.poly_oracle_to_var(&poly_oracle)?;
            let result_var = circuit.evaluate_poly_oracle(&poly_oracle_var, &point_var)?;
            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
            assert_eq!(circuit.witness(result_var)?, evaluation);
        }
        Ok(())
    }

    fn test_sum_check_circuit_template<E, Fr, Fq>() -> Result<(), CircuitError>
    where
        E: HasTEForm<ScalarField = Fr, BaseField = Fq>,
        Fr: PrimeField + RescueParameter + EmulationConfig<Fq>,
        Fq: PrimeField + RescueParameter,
    {
        let mut rng = test_rng();
        for _ in 0..10 {
            let max_degree = usize::rand(&mut rng) % 4 + 2;
            let num_vars = usize::rand(&mut rng) % 10;
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
            let mut transcript = <RescueTranscript<Fr> as Transcript>::new_transcript(b"test");
            let sum_check_proof =
                VPSumCheck::<E>::prove(&virtual_polynomial, &mut transcript).unwrap();
            let evaluation = virtual_polynomial
                .evaluate(&sum_check_proof.point)
                .map_err(|_| CircuitError::ParameterError("Evaluation Failed".to_string()))?;
            let mut circuit = PlonkCircuit::<Fr>::new_turbo_plonk();
            let sum_check_proof_var = circuit.sum_check_proof_to_var(&sum_check_proof)?;
            let mut transcript_var = RescueTranscriptVar::<Fr>::new_transcript(&mut circuit);
            let result_var = circuit.verify_sum_check::<E, RescueTranscriptVar<Fr>>(
                &sum_check_proof_var,
                &mut transcript_var,
            )?;
            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
            assert_eq!(circuit.witness(result_var)?, evaluation);
        }
        Ok(())
    }

    fn test_sum_check_challenge_verifier_template<E, Fr, Fq>() -> Result<(), CircuitError>
    where
        E: HasTEForm<ScalarField = Fr, BaseField = Fq>,
        Fr: PrimeField + RescueParameter + EmulationConfig<Fq>,
        Fq: PrimeField + RescueParameter,
    {
        let mut rng = test_rng();
        for _ in 0..10 {
            let max_degree = usize::rand(&mut rng) % 6 + 2;
            let num_vars = usize::rand(&mut rng) % 10;
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
            let mut transcript: RescueTranscript<Fq> =
                <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let sum_check_proof =
                VPSumCheck::<E>::prove(&virtual_polynomial, &mut transcript).unwrap();
            let oracles = sum_check_proof.oracles.clone();
            let challenges = sum_check_proof.point.clone();
            let mut circuit = PlonkCircuit::<Fq>::new_ultra_plonk(8);

            let emul_oracles_var = oracles
                .iter()
                .map(|oracle| circuit.poly_oracle_to_emulated_var(oracle))
                .collect::<Result<Vec<_>, _>>()?;
            let emul_challenges_var = challenges
                .iter()
                .map(|challenge| circuit.create_emulated_variable(*challenge))
                .collect::<Result<Vec<_>, _>>()?;
            let mut transcript = RescueTranscriptVar::<Fq>::new_transcript(&mut circuit);
            circuit.verify_challenges::<E>(
                &emul_oracles_var,
                &emul_challenges_var,
                &mut transcript,
            )?;
            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
            // ark_std::println!("number of constraint: {}", circuit.num_gates());
        }
        Ok(())
    }

    fn test_emulated_sumcheck_circuit_template<P, Fr, Fq>() -> Result<(), CircuitError>
    where
        P: HasTEForm<ScalarField = Fr, BaseField = Fq>,

        Fr: PrimeField + RescueParameter + EmulationConfig<Fq>,
        Fq: PrimeField + RescueParameter + EmulationConfig<Fr>,
    {
        let mut rng = test_rng();
        for _ in 0..10 {
            let num_vars = usize::rand(&mut rng) % 10;
            let mles = (0..20)
                .map(|_| Arc::new(DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng)))
                .collect::<Vec<_>>();

            let range = Uniform::new(0usize, 20);
            let products = (0..10)
                .map(|_| {
                    let mut product = Vec::new();
                    for _ in 0..2 {
                        product.push(range.sample(&mut rng));
                    }
                    (Fr::one(), product)
                })
                .collect::<Vec<_>>();
            let virtual_polynomial =
                VirtualPolynomial::<Fr>::new(2, num_vars, mles.clone(), products.clone());
            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let sum_check_proof =
                VPSumCheck::<P>::prove(&virtual_polynomial, &mut transcript).unwrap();
            let evaluation = virtual_polynomial
                .evaluate(&sum_check_proof.point)
                .map_err(|_| CircuitError::ParameterError("Evaluation Failed".to_string()))?;
            let mut circuit = PlonkCircuit::<Fq>::new_ultra_plonk(8);
            let sum_check_proof_var = circuit.proof_to_emulated_var::<P>(&sum_check_proof)?;
            let mut transcript = RescueTranscriptVar::<Fq>::new_transcript(&mut circuit);
            let result_var =
                circuit.verify_emulated_proof::<P>(&sum_check_proof_var, &mut transcript)?;
            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
            assert_eq!(circuit.emulated_witness(&result_var)?, evaluation);
            // ark_std::println!(
            //     "number of constraint emulated proof verify: {}",
            //     circuit.num_gates()
            // );
        }
        Ok(())
    }

    #[test]
    fn test_poly_oracle_circuit() {
        test_poly_oracle_circuit_template::<Bls12_377>().expect("test failed for bls12-381");
    }

    #[test]
    fn test_sum_check_circuit() {
        test_sum_check_circuit_template::<SWGrumpkin, _, _>().unwrap();
    }

    #[test]
    fn test_sumcheck_challenge_verifier() {
        test_sum_check_challenge_verifier_template::<SWGrumpkin, _, _>()
            .expect("test failed for Grumpkin");
    }

    #[test]
    fn test_emulated_sumcheck_circuit() {
        test_emulated_sumcheck_circuit_template::<SWGrumpkin, _, _>()
            .expect("test failed for Grumpkin");
    }
}
