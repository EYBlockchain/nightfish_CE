//! Circuit gadgets for verifying split-accumulation, both all in one circuit and also if we are using a cycle of curves by
//! splitting it over two circuits to minimise wrong field arithmetic.

use ark_ff::PrimeField;
use ark_std::{vec, vec::Vec, One, Zero};

use itertools::izip;
use jf_primitives::rescue::RescueParameter;
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{EmulMultiScalarMultiplicationCircuit, HasTEForm, PointVariable},
        EmulatedVariable, EmulationConfig,
    },
    Circuit, PlonkCircuit, Variable,
};

use crate::{
    nightfall::circuit::subroutine_verifiers::{
        structs::{EmulatedPolyOracleVar, EmulatedSumCheckProofVar},
        sumcheck::SumCheckGadget,
    },
    transcript::{rescue::RescueTranscriptVar, CircuitTranscript},
};

use super::structs::{EmulatedPCSInstanceVar, MVSplitProofVar, PCSInstanceVar};

/// Trait for performing split-accumulation verification in a circuit context.
pub trait MVSplitAccumulatorGadget<E: HasTEForm>
where
    E::BaseField: PrimeField + RescueParameter,
    E::ScalarField: EmulationConfig<E::BaseField>,
{
    /// This method is used to fully verify split accumulation in a single circuit.
    fn verify_split_accumulation(
        &mut self,
        acc: &EmulatedPCSInstanceVar<E>,
        old_instances: &[EmulatedPCSInstanceVar<E>],
        proof: &EmulatedSumCheckProofVar<E::ScalarField>,
    ) -> Result<(), CircuitError>;
    /// This method is used when proving recursively over a cycle of curves. It verifies all transcript related hashes and MSMs but no scalar field arithmetic.
    #[allow(clippy::too_many_arguments)]
    fn verify_split_level_one(
        &mut self,
        acc: &EmulatedPCSInstanceVar<E>,
        old_instances: &[EmulatedPCSInstanceVar<E>],
        t: &[EmulatedVariable<E::ScalarField>],
        oracles: &[EmulatedPolyOracleVar<E::ScalarField>],
        eval: &EmulatedVariable<E::ScalarField>,
        r_0_evals: &[EmulatedVariable<E::ScalarField>],
        poly_coeffs: &[EmulatedVariable<E::ScalarField>],
        challenges: &[EmulatedVariable<E::ScalarField>],
        transcript: &mut RescueTranscriptVar<E::BaseField>,
    ) -> Result<(), CircuitError>;
    /// This method is used when proving recursively over a cycle of curves. It verifies all transcript related hashes and MSMs but no scalar field arithmetic.
    #[allow(clippy::too_many_arguments)]
    fn verify_split_level_one_recursion(
        &mut self,
        acc: &EmulatedPCSInstanceVar<E>,
        old_instances: &[EmulatedPCSInstanceVar<E>],
        t: &[EmulatedVariable<E::ScalarField>],
        oracles: &[EmulatedPolyOracleVar<E::ScalarField>],
        eval: &EmulatedVariable<E::ScalarField>,
        r_0_evals: &[EmulatedVariable<E::ScalarField>],
        poly_coeffs: &[EmulatedVariable<E::ScalarField>],
        challenges: &[EmulatedVariable<E::ScalarField>],
        transcript: &mut RescueTranscriptVar<E::BaseField>,
    ) -> Result<(), CircuitError>;
    /// This method verifies all scalar arithmetic and is used when proving recursively over a cycle of curves.
    fn verify_split_level_two(
        &mut self,
        acc: &PCSInstanceVar,
        t: &[Variable],
        old_instances: &[PCSInstanceVar],
        poly_coeffs: &[Variable],
        proof: &MVSplitProofVar,
    ) -> Result<(), CircuitError>;
}

impl<E> MVSplitAccumulatorGadget<E> for PlonkCircuit<E::BaseField>
where
    E: HasTEForm,
    E::BaseField: PrimeField + RescueParameter,
    E::ScalarField: EmulationConfig<E::BaseField> + RescueParameter,
{
    fn verify_split_accumulation(
        &mut self,
        acc: &EmulatedPCSInstanceVar<E>,
        old_instances: &[EmulatedPCSInstanceVar<E>],
        proof: &EmulatedSumCheckProofVar<<E>::ScalarField>,
    ) -> Result<(), CircuitError> {
        let mut transcript = RescueTranscriptVar::<E::BaseField>::new_transcript(self);

        let num_polys = old_instances.len();

        let l = num_polys.next_power_of_two().ilog2() as usize;

        let t = transcript.squeeze_scalar_challenges::<E>(l, self)?;
        let t = t
            .iter()
            .map(|t| self.to_emulated_variable(*t))
            .collect::<Result<Vec<_>, _>>()?;
        let deferred_check = self.verify_emulated_proof::<E>(proof, &mut transcript)?;

        // We will repeatedly need the zero and one of the scalar field.
        let zero_var = self.emulated_zero();
        let one_var = self.emulated_one();

        let a_1 = &proof.point_var[0..l];
        let a2 = &proof.point_var[l..];
        let mut eq_i_a2_evals = vec![];
        for instance in old_instances.iter() {
            let mut prod = one_var.clone();
            for (z_i_j, a2_j) in instance.point.iter().zip(a2.iter()) {
                let tmp1 = self.emulated_mul(z_i_j, a2_j)?;
                let tmp2 = self.emulated_add(&tmp1, &tmp1)?;
                let tmp3 = self.emulated_sub(&tmp2, a2_j)?;
                let tmp4 = self.emulated_sub(&tmp3, z_i_j)?;
                let sum = self.emulated_add_constant(&tmp4, E::ScalarField::one())?;

                prod = self.emulated_mul(&prod, &sum)?;
            }
            eq_i_a2_evals.push(prod);
        }
        let mut default_value = one_var.clone();
        for a in a2.iter() {
            let tmp1 = self.emulated_sub(&one_var, a)?;
            default_value = self.emulated_mul(&default_value, &tmp1)?;
        }
        eq_i_a2_evals.resize(1 << l, default_value);

        // We decompose the values in [0,2^l - 1] into their binary decomposition.

        let field_indices = (0..(1 << l))
            .map(|j| {
                let mut bits = vec![];
                for i in 0..l {
                    let bit = (j >> i) & 1;
                    bits.push(E::ScalarField::from(bit as u8));
                }
                bits
            })
            .collect::<Vec<Vec<E::ScalarField>>>();

        let mut coeffs = vec![];
        let mut eq_a1_a2_eval = zero_var.clone();
        for (j, field_index) in field_indices.iter().enumerate() {
            let mut eq_a1_j = one_var.clone();
            let mut acc = one_var.clone();
            for (bit, t, a1) in izip!(field_index, t.iter(), a_1.iter()) {
                if j < num_polys {
                    if *bit == E::ScalarField::one() {
                        let tmp1 = self.emulated_mul(a1, t)?;
                        acc = self.emulated_mul(&acc, &tmp1)?;
                    } else {
                        let tmp1 = self.emulated_sub_constant(a1, E::ScalarField::one())?;
                        let tmp2 = self.emulated_sub_constant(t, E::ScalarField::one())?;
                        let tmp3 = self.emulated_mul(&tmp1, &tmp2)?;
                        acc = self.emulated_mul(&acc, &tmp3)?;
                    }
                }
                if *bit == E::ScalarField::one() {
                    eq_a1_j = self.emulated_mul(&eq_a1_j, a1)?;
                } else {
                    let tmp1 = self.emulated_sub(&one_var, a1)?;
                    eq_a1_j = self.emulated_mul(&eq_a1_j, &tmp1)?;
                }
            }
            if j < num_polys {
                coeffs.push(acc);
            }
            let tmp = self.emulated_mul(&eq_a1_j, &eq_i_a2_evals[j])?;
            eq_a1_a2_eval = self.emulated_add(&eq_a1_a2_eval, &tmp)?;
        }

        let bases = old_instances
            .iter()
            .map(|instance| instance.comm)
            .collect::<Vec<PointVariable>>();

        let commitment =
            EmulMultiScalarMultiplicationCircuit::<E::BaseField, E>::msm(self, &bases, &coeffs)?;
        self.enforce_point_equal(&commitment, &acc.comm)?;
        self.emulated_mul_gate(&acc.value, &eq_a1_a2_eval, &deferred_check)?;

        Ok(())
    }
    fn verify_split_level_one(
        &mut self,
        acc: &EmulatedPCSInstanceVar<E>,
        old_instances: &[EmulatedPCSInstanceVar<E>],
        t: &[EmulatedVariable<E::ScalarField>],
        oracles: &[EmulatedPolyOracleVar<E::ScalarField>],
        eval: &EmulatedVariable<E::ScalarField>,
        r_0_evals: &[EmulatedVariable<E::ScalarField>],
        poly_coeffs: &[EmulatedVariable<E::ScalarField>],
        challenges: &[EmulatedVariable<E::ScalarField>],
        transcript: &mut RescueTranscriptVar<E::BaseField>,
    ) -> Result<(), CircuitError> {
        let l = old_instances.len().next_power_of_two().ilog2() as usize;

        let calc_t = transcript.squeeze_scalar_challenges::<E>(l, self)?;

        for (calc_ts, ts) in calc_t.iter().zip(t.iter()) {
            let ts = self.mod_to_native_field(ts)?;
            self.enforce_equal(*calc_ts, ts)?;
        }

        <Self as SumCheckGadget<E::BaseField>>::verify_challenges::<E>(
            self, oracles, eval, r_0_evals, challenges, transcript,
        )?;

        let comms = old_instances
            .iter()
            .map(|instance| instance.comm)
            .collect::<Vec<_>>();

        let calc_point = <Self as EmulMultiScalarMultiplicationCircuit<_, E>>::msm(
            self,
            &comms,
            &poly_coeffs[..old_instances.len()],
        )?;

        self.enforce_point_equal(&calc_point, &acc.comm)
    }

    fn verify_split_level_one_recursion(
        &mut self,
        acc: &EmulatedPCSInstanceVar<E>,
        old_instances: &[EmulatedPCSInstanceVar<E>],
        t: &[EmulatedVariable<E::ScalarField>],
        oracles: &[EmulatedPolyOracleVar<E::ScalarField>],
        eval: &EmulatedVariable<E::ScalarField>,
        r_0_evals: &[EmulatedVariable<E::ScalarField>],
        poly_coeffs: &[EmulatedVariable<E::ScalarField>],
        challenges: &[EmulatedVariable<E::ScalarField>],
        transcript: &mut RescueTranscriptVar<E::BaseField>,
    ) -> Result<(), CircuitError> {
        let l = old_instances.len().next_power_of_two().ilog2() as usize;

        let calc_t = transcript.squeeze_scalar_challenges::<E>(l, self)?;

        for (calc_ts, ts) in calc_t.iter().zip(t.iter()) {
            let ts = self.mod_to_native_field(ts)?;
            self.enforce_equal(*calc_ts, ts)?;
        }

        <Self as SumCheckGadget<E::BaseField>>::verify_challenges::<E>(
            self, oracles, eval, r_0_evals, challenges, transcript,
        )?;

        let comms = old_instances
            .iter()
            .map(|instance| instance.comm)
            .collect::<Vec<_>>();

        let comms_until_prod = &comms[0..29];
        let prod_one_comm = comms[29];
        let frac_one_comm = comms[33];
        let next_part_one = &comms[36..42];
        let next_one = &comms[43..49];
        let prod_two_comm = comms[72];
        let frac_two_comm = comms[76];
        let next_part_two = &comms[79..85];
        let rest_of_it = &comms[86..];

        let final_comms = [
            comms_until_prod,
            &[prod_one_comm, frac_one_comm],
            next_part_one,
            next_one,
            &[prod_two_comm, frac_two_comm],
            next_part_two,
            rest_of_it,
        ]
        .concat();

        let raw_poly_values = poly_coeffs
            .iter()
            .map(|poly_coeff| self.emulated_witness(poly_coeff))
            .collect::<Result<Vec<_>, CircuitError>>()?;

        let (proof_one_scalars, rest) = raw_poly_values.split_at(43);
        let proof_two_scalars = rest.split_at(43).0;
        let out_scalars = combine_two_proof_scalars(proof_one_scalars, proof_two_scalars)
            .iter()
            .map(|f| self.create_emulated_variable::<E::ScalarField>(*f))
            .collect::<Result<Vec<_>, CircuitError>>()?;
        let final_scalars = [out_scalars.as_slice(), poly_coeffs[86..90].as_ref()].concat();

        let calc_point = <Self as EmulMultiScalarMultiplicationCircuit<_, E>>::msm(
            self,
            &final_comms,
            &final_scalars[..final_comms.len()],
        )?;

        self.enforce_point_equal(&calc_point, &acc.comm)
    }

    fn verify_split_level_two(
        &mut self,
        acc: &PCSInstanceVar,
        t: &[Variable],
        old_instances: &[PCSInstanceVar],
        poly_coeffs: &[Variable],
        proof: &MVSplitProofVar,
    ) -> Result<(), CircuitError> {
        let l = old_instances.len().next_power_of_two().ilog2() as usize;
        let eval =
            <Self as SumCheckGadget<E::BaseField>>::verify_sum_check_with_challenges(self, proof)?;

        let a1 = &proof.point_var[..l];
        let a2 = &proof.point_var[l..];

        // New way of computing eq(<i>, a1) * eq(t, <i>)

        // Now we need to calculate \tilde{eq}(a1, a2).
        // First step is to calculate {\tilde{eq}(<i>, a2) = eq(a2, z_i)}_{i \in [2^l]}.
        let mut eq_i_a2_evals = vec![];
        for instance in old_instances.iter() {
            let mut prod = self.one();
            for (&z_i_j, &a2_j) in instance.point.iter().zip(a2.iter()) {
                let wires_in = [z_i_j, a2_j, self.zero(), self.zero()];
                let coeffs = [
                    -E::BaseField::one(),
                    -E::BaseField::one(),
                    E::BaseField::zero(),
                    E::BaseField::zero(),
                ];
                let q_muls = [E::BaseField::from(2u8), E::BaseField::zero()];
                let sum = self.gen_quad_poly(&wires_in, &coeffs, &q_muls, E::BaseField::one())?;

                prod = self.mul(prod, sum)?;
            }
            eq_i_a2_evals.push(prod);
        }
        let mut default_value = self.one();
        for a in a2.iter() {
            default_value = self.mul_add(
                &[default_value, self.one(), default_value, *a],
                &[E::BaseField::one(), -E::BaseField::one()],
            )?;
        }
        eq_i_a2_evals.resize(1 << l, default_value);

        // We decompose the values in [0,2^l - 1] into their binary decomposition.

        let field_indices = (0..(1 << l))
            .map(|j| {
                let mut bits = vec![];
                for i in 0..l {
                    let bit = (j >> i) & 1;
                    bits.push(E::BaseField::from(bit as u8));
                }
                bits
            })
            .collect::<Vec<Vec<E::BaseField>>>();

        let mut coeffs = vec![];
        let mut eq_a1_a2_eval = self.zero();

        for (j, field_index) in field_indices.iter().enumerate() {
            let mut eq_a1_j = self.one();
            let mut acc = self.one();
            for (bit, t, a1) in izip!(field_index, t.iter(), a1.iter()) {
                if j < old_instances.len() {
                    let tmp_selector = *bit - E::BaseField::one();
                    let tmp = self.gen_quad_poly(
                        &[*a1, *t, self.zero(), self.zero()],
                        &[
                            tmp_selector,
                            tmp_selector,
                            E::BaseField::zero(),
                            E::BaseField::zero(),
                        ],
                        &[E::BaseField::one(), E::BaseField::zero()],
                        E::BaseField::one() - *bit,
                    )?;
                    acc = self.mul(acc, tmp)?;
                }
                if *bit == E::BaseField::one() {
                    eq_a1_j = self.mul(eq_a1_j, *a1)?;
                } else {
                    eq_a1_j = self.mul_add(
                        &[eq_a1_j, self.one(), eq_a1_j, *a1],
                        &[E::BaseField::one(), -E::BaseField::one()],
                    )?;
                }
            }
            if j < old_instances.len() {
                coeffs.push(acc);
            }
            eq_a1_a2_eval = self.mul_add(
                &[eq_a1_a2_eval, self.one(), eq_a1_j, eq_i_a2_evals[j]],
                &[E::BaseField::one(), E::BaseField::one()],
            )?;
        }

        // We enforce the supplied coefficients are the same as the calculated ones.
        for (coeff, poly_coeff) in coeffs.iter().zip(poly_coeffs.iter()) {
            self.enforce_equal(*coeff, *poly_coeff)?;
        }

        let value = acc.value;
        // constrains that the value in the new accumulator is the ne coming from the proof.
        self.mul_gate(value, eq_a1_a2_eval, eval)
    }
}

fn combine_two_proof_scalars<F: PrimeField>(
    proof_one_scalars: &[F],
    proof_two_scalars: &[F],
) -> Vec<F> {
    let selector_scalars = proof_one_scalars[6..29]
        .iter()
        .zip(proof_two_scalars[6..29].iter())
        .map(|(a, b)| *a + *b);
    let product_one_scalar = proof_one_scalars[29..33].iter().sum();
    let product_two_scalar = proof_two_scalars[29..33].iter().sum();
    let frac_one_scalar = proof_one_scalars[33..36].iter().sum();
    let frac_two_scalar = proof_two_scalars[33..36].iter().sum();
    let range_wire_one = proof_one_scalars[5] + proof_one_scalars[42];
    let range_wire_two = proof_two_scalars[5] + proof_two_scalars[42];
    proof_one_scalars
        .iter()
        .take(5)
        .copied()
        .chain(vec![range_wire_one])
        .chain(selector_scalars)
        .chain(vec![product_one_scalar, frac_one_scalar])
        .chain(proof_one_scalars.iter().skip(36).take(6).copied())
        .chain(proof_two_scalars.iter().take(5).copied())
        .chain(vec![range_wire_two])
        .chain(vec![product_two_scalar, frac_two_scalar])
        .chain(proof_two_scalars.iter().skip(36).take(6).copied())
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use crate::{
        nightfall::{
            accumulation::{accumulation_structs::SplitAccumulator, MLAccumulator},
            hops::univariate_ipa::UnivariateIpaPCS,
            mle::{utils::mv_batch_verify, zeromorph::Zeromorph},
        },
        transcript::{RescueTranscript, Transcript},
    };
    use ark_bn254::g1::Config as BnConfig;

    use ark_poly::{evaluations::multivariate::DenseMultilinearExtension, MultilinearExtension};

    use nf_curves::grumpkin::Grumpkin;

    use ark_ec::{pairing::Pairing, short_weierstrass::Affine};

    use ark_std::{sync::Arc, vec, vec::Vec, UniformRand};
    use jf_primitives::pcs::{Accumulation, PolynomialCommitmentScheme, StructuredReferenceString};
    use jf_utils::test_rng;

    use super::*;

    #[test]
    fn test_prover_output_helper() {
        test_prover_output_helper_zeromorph::<Grumpkin, _, BnConfig, _>().unwrap();
    }

    fn test_prover_output_helper_zeromorph<E, P, Q, F>() -> Result<(), CircuitError>
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>>,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        Q: HasTEForm<BaseField = E::ScalarField, ScalarField = P::BaseField>,
        F: PrimeField + RescueParameter + EmulationConfig<E::ScalarField>,
        E::ScalarField: EmulationConfig<F> + RescueParameter + PrimeField,
    {
        let rng = &mut test_rng();

        let num_vars = usize::rand(rng) % 5 + 3;

        let pp = Zeromorph::<UnivariateIpaPCS<E>>::gen_srs_for_testing(rng, num_vars).unwrap();
        let (ck, _) = Zeromorph::<UnivariateIpaPCS<E>>::trim(pp, 0, Some(num_vars)).unwrap();
        for _ in 0..10 {
            let mut zeromorph_accumulator: MLAccumulator<Zeromorph<UnivariateIpaPCS<E>>> =
                MLAccumulator::<Zeromorph<UnivariateIpaPCS<E>>>::new();
            let num_polys = usize::rand(rng) % 20 + 2;
            let num_polys = num_polys.next_power_of_two();
            for _ in 0..num_polys {
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
            let (proof, new_acc) = zeromorph_accumulator
                .prove_accumulation(&ck, Some(&mut transcript))
                .unwrap();

            // First we test that the level one verification passes.
            let mut circuit = PlonkCircuit::<E::BaseField>::new_ultra_plonk(8);
            // Make emulated variables for the instances.
            let mut instance_vars = vec![];
            for instance in zeromorph_accumulator.instances.iter() {
                let instance_var =
                    EmulatedPCSInstanceVar::<P>::from_instance(instance, &mut circuit).unwrap();
                instance_vars.push(instance_var);
            }

            // Make new_acc variable.
            let new_acc_var =
                EmulatedPCSInstanceVar::<P>::from_instance(&new_acc.instances[0], &mut circuit)
                    .unwrap();

            // Make emulated variables for the challenges.
            let challenges = proof.point.clone();
            let mut challenge_vars = vec![];
            for challenge in challenges.iter() {
                let challenge_var = circuit
                    .create_emulated_variable::<E::ScalarField>(*challenge)
                    .unwrap();
                challenge_vars.push(challenge_var);
            }

            let oracles = proof.oracles.clone();
            let mut oracle_vars = vec![];
            for oracle in oracles.iter() {
                let oracle_var = circuit.poly_oracle_to_emulated_var(oracle).unwrap();
                oracle_vars.push(oracle_var);
            }

            let eval = circuit
                .create_emulated_variable::<E::ScalarField>(proof.eval)
                .unwrap();

            let r_0_evals_vars = proof
                .r_0_evals
                .iter()
                .map(|e| circuit.create_emulated_variable(*e))
                .collect::<Result<Vec<_>, CircuitError>>()?;

            let points = zeromorph_accumulator.points();

            let values = zeromorph_accumulator.evaluations();

            let mut transcript =
                <RescueTranscript<E::BaseField> as Transcript>::new_transcript(b"test");
            let (_g_tilde_eval, poly_coeffs, _a2) =
                mv_batch_verify::<P>(&points, &values, &proof, &mut transcript).unwrap();

            let emulated_poly_coeffs = poly_coeffs
                .iter()
                .map(|coeff| {
                    circuit
                        .create_emulated_variable::<E::ScalarField>(*coeff)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let t = recover_t_values(&mut zeromorph_accumulator, &ck);
            let mut t_vars = vec![];
            for t in t.iter() {
                let t_var = circuit
                    .create_emulated_variable::<E::ScalarField>(*t)
                    .unwrap();
                t_vars.push(t_var);
            }

            let mut transcript_var =
                RescueTranscriptVar::<E::BaseField>::new_transcript(&mut circuit);

            circuit.verify_split_level_one(
                &new_acc_var,
                &instance_vars,
                &t_vars,
                &oracle_vars,
                &eval,
                &r_0_evals_vars,
                &emulated_poly_coeffs,
                &challenge_vars,
                &mut transcript_var,
            )?;

            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

            // Now we test that level two verification passes.

            let mut circuit = PlonkCircuit::<Q::BaseField>::new_ultra_plonk(8);
            // We make all the standard variables in the correct field for the circuit this time.
            let mut instance_vars = vec![];
            for instance in zeromorph_accumulator.instances.iter() {
                let mut point = vec![];
                for val in instance.point.iter() {
                    let val_var = circuit.create_variable(*val).unwrap();
                    point.push(val_var);
                }
                let value = circuit.create_variable(instance.value).unwrap();
                let instance_var = PCSInstanceVar { value, point };
                instance_vars.push(instance_var);
            }

            let new_point_var = new_acc.instances[0]
                .point
                .iter()
                .map(|val| circuit.create_variable(*val).unwrap())
                .collect::<Vec<_>>();
            let new_value_var = circuit.create_variable(new_acc.instances[0].value).unwrap();
            let new_acc = PCSInstanceVar {
                value: new_value_var,
                point: new_point_var,
            };

            let t_vec = t
                .iter()
                .map(|t| circuit.create_variable(*t).unwrap())
                .collect::<Vec<_>>();
            let poly_coeffs = poly_coeffs
                .iter()
                .map(|coeff| circuit.create_variable(*coeff).unwrap())
                .collect::<Vec<_>>();
            let proof_var = circuit.sum_check_proof_to_var(&proof).unwrap();
            <PlonkCircuit<Q::BaseField> as MVSplitAccumulatorGadget<Q>>::verify_split_level_two(
                &mut circuit,
                &new_acc,
                &t_vec,
                &instance_vars,
                &poly_coeffs,
                &proof_var,
            )?;

            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        }

        Ok(())
    }

    #[test]
    fn test_emulated_verifier() {
        test_emulated_verifier_helper::<Grumpkin, _, _>().unwrap();
    }

    fn test_emulated_verifier_helper<E, P, F>() -> Result<(), CircuitError>
    where
        E: Pairing<G1Affine = Affine<P>, BaseField = F>,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        F: PrimeField + RescueParameter + EmulationConfig<E::ScalarField>,
        E::ScalarField: EmulationConfig<F> + RescueParameter + PrimeField,
    {
        let rng = &mut test_rng();

        let num_vars = usize::rand(rng) % 5 + 3;

        let pp = Zeromorph::<UnivariateIpaPCS<E>>::gen_srs_for_testing(rng, num_vars).unwrap();
        let (ck, _) = Zeromorph::<UnivariateIpaPCS<E>>::trim(pp, 0, Some(num_vars)).unwrap();
        for _ in 0..10 {
            let mut zeromorph_accumulator: MLAccumulator<Zeromorph<UnivariateIpaPCS<E>>> =
                MLAccumulator::<Zeromorph<UnivariateIpaPCS<E>>>::new();
            let num_polys = usize::rand(rng) % 20 + 2;
            let num_polys = num_polys.next_power_of_two();
            for _ in 0..num_polys {
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
            let (proof, new_acc) = zeromorph_accumulator
                .prove_accumulation(&ck, Some(&mut transcript))
                .unwrap();

            let mut circuit = PlonkCircuit::<E::BaseField>::new_ultra_plonk(8);
            // Make emulated variables for the instances.
            let mut instance_vars = vec![];
            for instance in zeromorph_accumulator.instances.iter() {
                let instance_var =
                    EmulatedPCSInstanceVar::<P>::from_instance(instance, &mut circuit).unwrap();
                instance_vars.push(instance_var);
            }

            // Make new_acc variable.
            let new_acc_var =
                EmulatedPCSInstanceVar::<P>::from_instance(&new_acc.instances[0], &mut circuit)
                    .unwrap();

            let emulated_proof = circuit.proof_to_emulated_var::<P>(&proof).unwrap();
            circuit.verify_split_accumulation(&new_acc_var, &instance_vars, &emulated_proof)?;

            circuit.check_circuit_satisfiability(&[]).unwrap();
            ark_std::println!("Constraints: {}", circuit.num_gates());
        }
        Ok(())
    }

    fn recover_t_values<PCS, P>(
        acc: &mut MLAccumulator<PCS>,
        ck: &<PCS::SRS as StructuredReferenceString>::ProverParam,
    ) -> Vec<PCS::Evaluation>
    where
        P: HasTEForm,
        P::BaseField: PrimeField + RescueParameter,
        P::ScalarField: EmulationConfig<P::BaseField>,
        PCS: Accumulation<
            Commitment = Affine<P>,
            Polynomial = Arc<DenseMultilinearExtension<P::ScalarField>>,
            Evaluation = P::ScalarField,
            Point = Vec<P::ScalarField>,
        >,
    {
        let mut transcript =
            <RescueTranscript<P::BaseField> as Transcript>::new_transcript(b"test");
        let (_, _) = acc.prove_accumulation(ck, Some(&mut transcript)).unwrap();

        let l = acc.instances.len().next_power_of_two().ilog2() as usize;
        let mut transcript =
            <RescueTranscript<P::BaseField> as Transcript>::new_transcript(b"test");
        transcript.squeeze_scalar_challenges::<P>(b"t", l).unwrap()
    }
}
