//! Code for fully verifying a zeromorph proof in a circuit.

use ark_ec::{pairing::Pairing, short_weierstrass::Affine};
use ark_ff::{Field, PrimeField};

use ark_std::{string::ToString, vec, vec::Vec, One};
use jf_primitives::{pcs::StructuredReferenceString, rescue::RescueParameter};
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{EmulMultiScalarMultiplicationCircuit, HasTEForm, Point, PointVariable},
        EmulatedVariable, EmulationConfig,
    },
    BoolVar, Circuit, PlonkCircuit,
};

use crate::{
    nightfall::{
        hops::srs::UnivariateUniversalIpaParams,
        mle::zeromorph::zeromorph_protocol::ZeromorphIpaProof,
    },
    transcript::{rescue::RescueTranscriptVar, CircuitTranscript},
};

use super::verify_ipa::verify_ipa_circuit;

/// Circuit gadget to perform a full IPA verification.
pub fn verify_zeromorph_circuit<E, F, P>(
    circuit: &mut PlonkCircuit<F>,
    verifier_param: &<UnivariateUniversalIpaParams<E> as StructuredReferenceString>::VerifierParam,
    commitment: &PointVariable,
    evaluation_point: &[EmulatedVariable<E::ScalarField>],
    evaluation: &EmulatedVariable<E::ScalarField>,
    proof: ZeromorphIpaProof<E>,
) -> Result<(), CircuitError>
where
    E: Pairing<BaseField = F, G1Affine = Affine<P>>,
    <E as Pairing>::ScalarField: EmulationConfig<F> + PrimeField + RescueParameter,
    F: RescueParameter + PrimeField,
    P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
{
    let commitments = &proof.commitments;
    let num_vars = commitments.len() - 2;
    let commitment_vars = commitments
        .iter()
        .take(num_vars)
        .map(|c| circuit.create_point_variable(&Point::<F>::from(*c)))
        .collect::<Result<Vec<PointVariable>, CircuitError>>()?;

    let q_hat_comm = &commitments[commitments.len() - 1];
    let q_hat_var = circuit.create_point_variable(&Point::<F>::from(*q_hat_comm))?;
    let f_hat_comm = &commitments[commitments.len() - 2];
    let f_hat_var = circuit.create_point_variable(&Point::<F>::from(*f_hat_comm))?;

    // We check that the commitment is the penultimate commitment in the batch commitment.
    circuit.enforce_point_equal(commitment, &f_hat_var)?;

    let max_degree = verifier_param.max_degree() - 1;

    let log_max_degree = max_degree.ilog2();

    let pow_two_diff = if log_max_degree as usize >= num_vars {
        log_max_degree as usize - num_vars
    } else if log_max_degree as usize + 1 == num_vars {
        0
    } else {
        return Err(CircuitError::ParameterError(
            "log_max_degree + 1 < num_vars".to_string(),
        ));
    };

    let difference = if log_max_degree as usize + 1 == num_vars {
        0
    } else {
        max_degree - (1 << log_max_degree)
    };
    // We initiate a new transcript that uses Jellyfish's sponge based rescue hash.
    let mut transcript = RescueTranscriptVar::<E::BaseField>::new_transcript(circuit);

    // We push the commitments to the `q_k`s and `f` to the transcript.

    transcript.append_point_variables(&commitment_vars, circuit)?;
    transcript.append_point_variable(&f_hat_var, circuit)?;

    // This challenge y is used for the batched degree check on all the `U(q_k)`s.
    let y = transcript.squeeze_scalar_challenge::<P>(circuit)?;
    let y = circuit.to_emulated_variable(y)?;

    // We push the commitment to `q_hat` to the transcript.
    transcript.append_point_variable(&q_hat_var, circuit)?;

    // This challenge x is used to construct our Z_x polynomial.
    let x = transcript.squeeze_scalar_challenge::<P>(circuit)?;
    let x = circuit.to_emulated_variable(x)?;

    // This challenge z is used to batch our two degree checks.
    let z = transcript.squeeze_scalar_challenge::<P>(circuit)?;
    let z = circuit.to_emulated_variable(z)?;

    let minus_one = circuit.create_constant_emulated_variable(-E::ScalarField::one())?;
    let one_var = circuit.emulated_one();
    let zero_var = circuit.emulated_zero();

    let x_minus_one = circuit.emulated_add(&x, &minus_one)?;
    let x_minus_one_val = circuit.emulated_witness(&x_minus_one)?;
    let x_minus_one_inv = x_minus_one_val
        .inverse()
        .ok_or(CircuitError::ParameterError(
            "could not invert x_minus_one".to_string(),
        ))?;
    let x_minus_one_inv_var = circuit.create_emulated_variable(x_minus_one_inv)?;
    circuit.emulated_mul_gate(&x_minus_one, &x_minus_one_inv_var, &one_var)?;

    let mut x_powers = vec![x.clone()];

    let x_val = circuit.emulated_witness(&x)?;
    let x_inv = x_val.inverse().ok_or(CircuitError::ParameterError(
        "could not invert x".to_string(),
    ))?;
    let x_inv_var = circuit.create_emulated_variable(x_inv)?;
    circuit.emulated_mul_gate(&x, &x_inv_var, &one_var)?;

    let mut inverse_x_powers = vec![x_inv_var.clone()];
    for i in 0..num_vars {
        let tmp = circuit.emulated_mul(&x_powers[i], &x_powers[i])?;
        let tmp_val = circuit.emulated_witness(&tmp)?;
        let tmp_inv = tmp_val.inverse().ok_or(CircuitError::ParameterError(
            "could not invert tmp".to_string(),
        ))?;
        let tmp_inv_var = circuit.create_emulated_variable(tmp_inv)?;
        circuit.emulated_mul_gate(&tmp, &tmp_inv_var, &one_var)?;
        x_powers.push(tmp);
        inverse_x_powers.push(tmp_inv_var);
    }

    let mut x_max = x_powers[num_vars].clone();
    for _ in 0..pow_two_diff {
        x_max = circuit.emulated_mul(&x_max, &x_max)?;
    }

    for _ in 0..difference {
        x_max = circuit.emulated_mul(&x_max, &x)?;
    }
    let selector_boolean = num_vars == log_max_degree as usize + 1;
    let bool_var = circuit.create_boolean_variable(selector_boolean)?;
    let x_max_plus_one_option = circuit.emulated_mul(&x_max, &x)?;
    let x_max_plus_one =
        circuit.conditional_select_emulated(bool_var, &x_max_plus_one_option, &x_max)?;
    let mut y_powers = vec![minus_one.clone()];
    for (i, x_inv_var) in inverse_x_powers.iter().enumerate().take(num_vars) {
        let degree_shift = circuit.emulated_mul(x_inv_var, &x_max_plus_one)?;
        if i != num_vars - 1 {
            let tmp = circuit.emulated_mul(&y_powers[i], &y)?;
            y_powers.push(tmp);
        }
        y_powers[i] = circuit.emulated_mul(&y_powers[i], &degree_shift)?;
    }

    // We construct the commitment to the poly `Z_x`.

    let phi_n_k = circuit.emulated_add(&x_powers[num_vars], &minus_one)?;
    let phi_denom = circuit.emulated_mul(&x_minus_one_inv_var, &phi_n_k)?;
    let value_var = circuit.emulated_sub(&zero_var, evaluation)?;
    let phi_value = circuit.emulated_mul(&phi_denom, &value_var)?;
    let z_phi_value = circuit.emulated_mul(&phi_value, &z)?;
    let z_phi_n_k = circuit.emulated_mul(&z, &phi_n_k)?;

    let mut q_k_scalars = vec![];

    for (x_pow, u_k) in x_powers.iter().zip(evaluation_point.iter()) {
        let mut denom_1 = circuit.emulated_mul(x_pow, x_pow)?;
        denom_1 = circuit.emulated_add(&denom_1, &minus_one)?;
        let denom_1_val = circuit.emulated_witness(&denom_1)?;
        let denom_1_inv = denom_1_val.inverse().ok_or(CircuitError::ParameterError(
            "could not invert denom_1".to_string(),
        ))?;
        let denom_1_inv_var = circuit.create_emulated_variable(denom_1_inv)?;
        circuit.emulated_mul_gate(&denom_1, &denom_1_inv_var, &one_var)?;

        let denom_2 = circuit.emulated_add(x_pow, &minus_one)?;
        let denom_2_val = circuit.emulated_witness(&denom_2)?;
        let denom_2_inv = denom_2_val.inverse().ok_or(CircuitError::ParameterError(
            "could not invert denom_2".to_string(),
        ))?;
        let denom_2_inv_var = circuit.create_emulated_variable(denom_2_inv)?;
        circuit.emulated_mul_gate(&denom_2, &denom_2_inv_var, &one_var)?;

        let tmp0 = circuit.emulated_mul(&denom_1_inv_var, &z_phi_n_k)?;
        let tmp1 = circuit.emulated_mul(&tmp0, x_pow)?;
        let tmp2 = circuit.emulated_mul(u_k, &z_phi_n_k)?;
        let tmp3 = circuit.emulated_mul(&denom_2_inv_var, &tmp2)?;
        let tmp4 = circuit.emulated_sub(&tmp1, &tmp3)?;
        let out = circuit.emulated_mul(&tmp4, &minus_one)?;
        q_k_scalars.push(out);
    }

    let scalars = [
        &[one_var, z, z_phi_value],
        q_k_scalars.as_slice(),
        y_powers.as_slice(),
    ]
    .concat();
    let g_base = circuit.create_point_variable(&Point::from(verifier_param.g_bases[0]))?;
    let bases = [
        &[q_hat_var, f_hat_var, g_base],
        commitment_vars.as_slice(),
        commitment_vars.as_slice(),
    ]
    .concat();

    let mut num_gates = circuit.num_gates();

    let batch_comm = EmulMultiScalarMultiplicationCircuit::<F, P>::msm(circuit, &bases, &scalars)?;

    ark_std::println!(
        "Zeromorph IPA verification MSM used {} gates",
        circuit.num_gates() - num_gates
    );

    let proof_l_i: Vec<PointVariable> = proof
        .degree_check_proof
        .l_i
        .iter()
        .copied()
        .map(|p| circuit.create_point_variable(&Point::<F>::from(p)))
        .collect::<Result<Vec<PointVariable>, _>>()?;

    let proof_r_i: Vec<PointVariable> = proof
        .degree_check_proof
        .r_i
        .iter()
        .copied()
        .map(|p| circuit.create_point_variable(&Point::<F>::from(p)))
        .collect::<Result<Vec<PointVariable>, _>>()?;

    num_gates = circuit.num_gates();

    for point_var in proof_l_i.iter().chain(proof_r_i.iter()) {
        let is_neutral: BoolVar = circuit.is_neutral_point::<P>(point_var)?;
        circuit.enforce_false(is_neutral.into())?;
        circuit.enforce_on_curve::<P>(point_var)?;
    }

    ark_std::println!(
        "Zeromorph IPA verification point checks used {} gates",
        circuit.num_gates() - num_gates
    );

    verify_ipa_circuit(
        circuit,
        verifier_param,
        &batch_comm,
        x,
        zero_var,
        proof_l_i,
        proof_r_i,
        proof.degree_check_proof.c,
        proof.degree_check_proof.f,
    )
}

#[cfg(test)]
mod tests {
    use crate::nightfall::{mle::zeromorph::Zeromorph, UnivariateIpaPCS};

    use super::*;
    use ark_poly::evaluations::multivariate::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{sync::Arc, UniformRand};
    use jf_primitives::pcs::PolynomialCommitmentScheme;
    use jf_relation::Circuit;
    use nf_curves::grumpkin::Grumpkin;

    #[test]
    fn test_zeromorph_circuit() -> Result<(), CircuitError> {
        test_zeromorph_circuit_helper::<Grumpkin, _, _>()
    }

    fn test_zeromorph_circuit_helper<E, P, F>() -> Result<(), CircuitError>
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>>,
        <E as Pairing>::ScalarField: EmulationConfig<F> + PrimeField + RescueParameter,
        F: RescueParameter + PrimeField,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
    {
        let rng = &mut ark_std::test_rng();
        let max_num_vars = 10;

        let pp =
            UnivariateIpaPCS::<E>::load_srs_from_file_for_testing((1 << max_num_vars) - 1, None)
                .unwrap();
        let (ck, vk) = Zeromorph::<UnivariateIpaPCS<E>>::trim(&pp, 0, Some(max_num_vars)).unwrap();

        let num_vars = (usize::rand(rng) % (max_num_vars - 5)) + 5;

        let poly = Arc::new(DenseMultilinearExtension::<E::ScalarField>::rand(
            num_vars, rng,
        ));
        let point = (0..num_vars)
            .map(|_| E::ScalarField::rand(rng))
            .collect::<Vec<E::ScalarField>>();

        let batch_commitment = Zeromorph::<UnivariateIpaPCS<E>>::commit(&ck, &poly).unwrap();
        let (batch_proof, value, _dense_poly) =
            Zeromorph::<UnivariateIpaPCS<E>>::open_with_poly(&ck, &poly, &point).unwrap();

        let mut circuit = PlonkCircuit::<F>::new_ultra_plonk(16);
        let commit_var = circuit.create_point_variable(&Point::<F>::from(batch_commitment))?;
        let point_var = point
            .iter()
            .map(|p| circuit.create_emulated_variable(*p).unwrap())
            .collect::<Vec<EmulatedVariable<E::ScalarField>>>();
        let eval_var = circuit.create_emulated_variable(value)?;
        verify_zeromorph_circuit(
            &mut circuit,
            &vk,
            &commit_var,
            &point_var,
            &eval_var,
            batch_proof.clone(),
        )?;

        assert!(Zeromorph::<UnivariateIpaPCS<E>>::verify(
            &vk,
            &batch_commitment,
            &point,
            &value,
            &batch_proof,
        )
        .unwrap());

        circuit.check_circuit_satisfiability(&[]).unwrap();
        ark_std::println!(" constraint count: {}", circuit.num_gates());

        Ok(())
    }
}
