//! The circuit gadget used to verify IPA in a Plonk Circuit.

use ark_ec::{pairing::Pairing, short_weierstrass::Affine};
use ark_ff::{Field, PrimeField};
use ark_std::{string::ToString, vec, vec::Vec};

use super::plonk_partial_verifier::UnivariateUniversalIpaParamsVar;
use crate::{
    nightfall::hops::srs::UnivariateUniversalIpaParams,
    transcript::{rescue::RescueTranscriptVar, CircuitTranscript},
};
use jf_primitives::{pcs::StructuredReferenceString, rescue::RescueParameter};
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{EmulMultiScalarMultiplicationCircuit, HasTEForm, PointVariable},
        EmulatedVariable, EmulationConfig,
    },
    Circuit, PlonkCircuit,
};

/// Circuit gadget to perform a full IPA verification.
#[allow(clippy::too_many_arguments)]
pub fn verify_ipa_circuit<E, F, P>(
    circuit: &mut PlonkCircuit<F>,
    verifier_param: &<UnivariateUniversalIpaParams<E> as StructuredReferenceString>::VerifierParam,
    commitment: &PointVariable,
    eval_point_var: EmulatedVariable<E::ScalarField>,
    evaluation: E::ScalarField,
    proof_l_i: Vec<PointVariable>,
    proof_r_i: Vec<PointVariable>,
    proof_c: E::ScalarField,
    proof_f: E::ScalarField,
) -> Result<(), CircuitError>
where
    E: Pairing<BaseField = F, G1Affine = Affine<P>>,
    <E as Pairing>::ScalarField: EmulationConfig<F> + PrimeField + RescueParameter,
    F: RescueParameter + PrimeField,
    P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
{
    let mut transcript_var: RescueTranscriptVar<F> = RescueTranscriptVar::new_transcript(circuit);
    let verifier_param_var = UnivariateUniversalIpaParamsVar::<F>::new(verifier_param, circuit)?;

    // Append the commitment
    transcript_var.append_point_variable(commitment, circuit)?;

    // Append evaluation points
    transcript_var.push_emulated_variable(&eval_point_var, circuit)?;

    // Append Poly Evaluation
    let poly_eval_var = circuit.create_emulated_variable(evaluation)?;
    transcript_var.push_emulated_variable(&poly_eval_var, circuit)?;

    let alpha = transcript_var.squeeze_scalar_challenge::<P>(circuit)?;
    let alpha = circuit.to_emulated_variable(alpha)?;

    // Loop appending commits
    let mut u_j_vec = Vec::<EmulatedVariable<E::ScalarField>>::new();
    let mut u_j_value_vec = Vec::new();
    let mut u_j_inv_vec = Vec::new();
    let one_emul_var = circuit.emulated_one();

    if proof_l_i.len() != proof_r_i.len() {
        return Err(CircuitError::ParameterError(
            "Length of proof.l_i and proof.r_i must be equal".to_string(),
        ));
    }
    if proof_l_i.len() != verifier_param.g_bases.len().ilog2() as usize {
        return Err(CircuitError::ParameterError(
            ark_std::format!("Length of proof.l_i and proof.r_i, {}, must be equal to the log2 of the number of bases, {}",
            proof_l_i.len(),
            verifier_param.g_bases.len().ilog2() as usize),
        ));
    }

    for (l, r) in proof_l_i.iter().zip(proof_r_i.iter()) {
        let commit_vars = vec![*l, *r];

        transcript_var.append_point_variables(&commit_vars, circuit)?;
        let u_j_idx = transcript_var.squeeze_scalar_challenge::<P>(circuit)?;
        let u_j_idx = circuit.to_emulated_variable::<E::ScalarField>(u_j_idx)?;

        let u_j = circuit.emulated_witness(&u_j_idx)?;

        let u_j_inv = u_j
            .inverse()
            .ok_or(CircuitError::ParameterError("Inverse Failed".to_string()))?;
        let u_j_inv_idx = circuit.create_emulated_variable(u_j_inv)?;

        circuit.emulated_mul_gate(&u_j_idx, &u_j_inv_idx, &one_emul_var)?;
        u_j_vec.push(u_j_idx);
        u_j_value_vec.push(u_j);
        u_j_inv_vec.push(u_j_inv_idx);
    }

    // Calc LHS and MSM
    let mut scalar_vars = [u_j_inv_vec.as_slice(), u_j_vec.as_slice()].concat();
    let mut bases_vars: Vec<PointVariable> = proof_l_i.into_iter().chain(proof_r_i).collect();

    let verifier_point_var = verifier_param_var.g_bases[0];
    let zero_var = circuit.emulated_zero();
    let additional_scalar = circuit.emulated_sub(&zero_var, &poly_eval_var)?;
    scalar_vars.push(additional_scalar);
    bases_vars.push(verifier_point_var);

    let degree = verifier_param.g_bases.len();

    let k = degree.ilog2();
    let g_prime = &verifier_param.g_bases;

    let mut b_powers = vec![eval_point_var.clone()];
    let mut current_power = eval_point_var;
    for _ in 0..k - 1 {
        current_power = circuit.emulated_mul(&current_power, &current_power)?;
        b_powers.push(current_power.clone());
    }

    let mut b_0 = one_emul_var.clone();
    for (b_power, u_j) in b_powers.iter().zip(u_j_vec.iter().rev()) {
        let tmp = circuit.emulated_mul_add(b_power, u_j, &one_emul_var)?;
        b_0 = circuit.emulated_mul(&b_0, &tmp)?;
    }
    let c_var = circuit.create_emulated_variable(proof_c)?;
    let c_var = circuit.emulated_sub(&zero_var, &c_var)?;
    let mut msm_scalars = vec![c_var.clone()];
    for u_j in u_j_vec.iter().rev() {
        let mut temporary_vec = vec![];
        for scalar in msm_scalars.iter() {
            let tmp = circuit.emulated_mul(scalar, u_j)?;
            temporary_vec.push(tmp);
        }
        msm_scalars.extend(temporary_vec);
    }

    let c_b = circuit.emulated_mul(&c_var, &b_0)?;

    let u_scalar = circuit.emulated_mul(&c_b, &alpha)?;

    let f_var = circuit.create_emulated_variable(proof_f)?;
    let f_var = circuit.emulated_sub(&zero_var, &f_var)?;

    let verifier_u_var = verifier_param_var.u;
    let verifier_h_var = verifier_param_var.h;
    let bases_vars = [bases_vars.as_slice(), &[verifier_u_var, verifier_h_var]].concat();
    let scalars = [scalar_vars.as_slice(), &[u_scalar.clone(), f_var.clone()]].concat();

    let g_base = EmulMultiScalarMultiplicationCircuit::<_, P>::fixed_base_msm(
        circuit,
        g_prime,
        &msm_scalars,
    )?;
    let intermediate =
        EmulMultiScalarMultiplicationCircuit::<_, P>::msm(circuit, &bases_vars, &scalars)?;

    let out = circuit.ecc_add::<P>(commitment, &intermediate)?;
    let out = circuit.ecc_add::<P>(&out, &g_base)?;

    circuit.neutral_point_gate::<P>(&out, circuit.true_var())?;

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        nightfall::{ipa_verifier::FFTVerifier, PlonkIpaSnark, UnivariateIpaPCS},
        proof_system::{PlonkKzgSnark, UniversalSNARK},
        transcript::{RescueTranscript, StandardTranscript},
    };

    use super::*;
    use ark_bls12_377::{g1::Config as Param377, Bls12_377, Fq, Fr};
    use ark_bw6_761::BW6_761;
    use ark_ec::{AffineRepr, CurveGroup};

    use ark_std::{rand::SeedableRng, UniformRand, Zero};
    use jf_primitives::pcs::prelude::UnivariateKzgPCS;
    use jf_relation::{
        gadgets::ecc::{MultiScalarMultiplicationCircuit, Point},
        BoolVar,
    };
    use jf_relation::{Arithmetization, Circuit};
    use jf_utils::fr_to_fq;
    use nf_curves::ed_on_bls_12_377::{
        Ed377Config as TE_377_Config, EdwardsAffine as TE_377, Fr as Fr_TE_377,
    };
    use rand_chacha::ChaCha20Rng;

    #[test]
    #[allow(non_snake_case)]
    #[ignore = "Very long test"]
    fn test_verify_ipa() -> Result<(), CircuitError> {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);

        let x_ipa = Fr_TE_377::rand(&mut rng);
        let G_ipa = TE_377::generator();
        let X_ipa = (G_ipa * x_ipa).into_affine();
        let mut circuit = PlonkCircuit::<Fr>::new_turbo_plonk();
        let x_fq = fr_to_fq::<_, TE_377_Config>(&x_ipa);
        let x_var = circuit.create_variable(x_fq)?;
        let G_jf: Point<Fr> = G_ipa.into();
        let G_var = circuit.create_constant_point_variable(&G_jf)?;
        let X_jf: Point<Fr> = X_ipa.into();
        let X_var = circuit.create_public_point_variable(&X_jf)?;

        let X_var_computed = MultiScalarMultiplicationCircuit::<_, TE_377_Config>::msm(
            &mut circuit,
            &[G_var],
            &[x_var],
        )?;
        circuit.enforce_point_equal(&X_var_computed, &X_var)?;
        circuit.check_circuit_satisfiability(&[X_jf.get_x(), X_jf.get_y()])?;
        circuit.finalize_for_arithmetization()?;

        let srs_size = circuit.srs_size()?;
        let ipa_srs =
            <PlonkIpaSnark<Bls12_377> as UniversalSNARK<UnivariateIpaPCS<Bls12_377>>>::universal_setup_for_testing(
                srs_size, &mut rng,
            )?;
        let (ipa_pk, ipa_vk) = PlonkIpaSnark::<Bls12_377>::preprocess(&ipa_srs, &circuit)?;

        let ipa_proof = PlonkIpaSnark::<Bls12_377>::prove::<_, _, RescueTranscript<Fq>>(
            &mut rng, &circuit, &ipa_pk, None,
        )?;

        let public_inputs = circuit.public_input().unwrap();

        assert!(PlonkIpaSnark::<Bls12_377>::verify::<RescueTranscript<Fq>>(
            &ipa_vk,
            &public_inputs,
            &ipa_proof,
            None,
        )
        .is_ok());

        let verifier = FFTVerifier::<UnivariateIpaPCS<Bls12_377>>::new(ipa_vk.domain_size)?;
        let pcs_info = verifier.prepare_pcs_info::<RescueTranscript<Fq>>(
            &ipa_vk,
            &public_inputs[..],
            &ipa_proof,
            &None,
        )?;

        let g_comm = pcs_info
            .comm_scalars_and_bases
            .multi_scalar_mul()
            .into_affine();

        let open_key = ipa_vk.open_key;
        let mut circuit = PlonkCircuit::new_ultra_plonk(8);

        let g_comm_var = circuit.create_point_variable(&Point::from(g_comm))?;

        let proof_l_i: Vec<PointVariable> = pcs_info
            .opening_proof
            .l_i
            .iter()
            .copied()
            .map(|p| circuit.create_point_variable(&Point::<Fq>::from(p)))
            .collect::<Result<Vec<PointVariable>, _>>()?;

        let proof_r_i: Vec<PointVariable> = pcs_info
            .opening_proof
            .r_i
            .iter()
            .copied()
            .map(|p| circuit.create_point_variable(&Point::<Fq>::from(p)))
            .collect::<Result<Vec<PointVariable>, _>>()?;

        for point_var in proof_l_i.iter().chain(proof_r_i.iter()) {
            let is_neutral: BoolVar = circuit.is_neutral_point::<Param377>(point_var)?;
            circuit.enforce_false(is_neutral.into())?;
            circuit.enforce_on_curve::<Param377>(point_var)?;
        }

        let eval_point_var = circuit.create_emulated_variable(pcs_info.u)?;

        verify_ipa_circuit::<Bls12_377, _, Param377>(
            &mut circuit,
            &open_key,
            &g_comm_var,
            eval_point_var,
            Fr::zero(),
            proof_l_i,
            proof_r_i,
            pcs_info.opening_proof.c,
            pcs_info.opening_proof.f,
        )
        .unwrap();
        let g_comm_te: Point<Fq> = g_comm.into();

        circuit.check_circuit_satisfiability(&[]).unwrap();
        ark_std::println!("Circuit size before finalize: {}", circuit.num_gates());
        circuit.finalize_for_arithmetization().unwrap();
        ark_std::println!("IPA Verification Circuit Finalised");
        let srs_size = circuit.srs_size().unwrap();
        ark_std::println!("IPA Circuit size: {}", srs_size);

        let srs = <PlonkKzgSnark<BW6_761> as UniversalSNARK<UnivariateKzgPCS<BW6_761>>>::universal_setup_for_testing(
            srs_size, &mut rng,
        )
        .unwrap();
        ark_std::println!("KZG SRS Generated");
        let (pk, vk) = PlonkKzgSnark::<BW6_761>::preprocess(&srs, &circuit).unwrap();
        ark_std::println!("KZG Proof Generated");
        let now = ark_std::time::Instant::now();
        let proof = PlonkKzgSnark::<BW6_761>::prove::<_, _, StandardTranscript>(
            &mut rng, &circuit, &pk, None,
        )
        .unwrap();
        ark_std::println!("KZG Proof time: {:?}", now.elapsed());
        PlonkKzgSnark::verify::<StandardTranscript>(
            &vk,
            &[g_comm_te.get_x(), g_comm_te.get_y()],
            &proof,
            None,
        )
        .unwrap();
        ark_std::println!("KZG Proof Verified");

        Ok(())
    }
}
