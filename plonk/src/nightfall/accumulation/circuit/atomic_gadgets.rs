//! Circuit gadgets for verifying atomic-accumulation, both all in one circuit and also if we are using a cycle of curves by
//! splitting it over two circuits to minimise wrong field arithmetic.

use ark_ff::PrimeField;
use ark_std::{vec, vec::Vec, One};

use jf_primitives::rescue::RescueParameter;
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{EmulMultiScalarMultiplicationCircuit, HasTEForm, PointVariable},
        EmulatedVariable, EmulationConfig,
    },
    Circuit, PlonkCircuit, Variable,
};

use crate::transcript::{rescue::RescueTranscriptVar, CircuitTranscript};

use super::structs::AtomicInstanceVar;

/// A trait for circuits to verify atomic-acumulation.
pub trait AtomicAccumulatorGadget<E>
where
    E: HasTEForm,
    E::ScalarField: EmulationConfig<E::BaseField> + RescueParameter,
    E::BaseField: RescueParameter + PrimeField,
{
    /// Verifies atomic accumulation when given the old instances, old accumulators and the accumulation proof.
    fn verify_atomic_accumulation(
        &mut self,
        old_instances: &[AtomicInstanceVar<E>],
        new_accumulator: &AtomicInstanceVar<E>,
        srs_base_point: &PointVariable,
    ) -> Result<(), CircuitError>;

    /// Only verifies the point msm part of the atomic accumulation and the transcript hashing. That is it doesn't perform non-native multiplication.
    #[allow(clippy::too_many_arguments)]
    fn verify_atomic_level_one(
        &mut self,
        old_instances: &[AtomicInstanceVar<E>],
        new_accumulator: &AtomicInstanceVar<E>,
        srs_base_point: &PointVariable,
        r_challenge: &EmulatedVariable<E::ScalarField>,
        instance_scalars: &[EmulatedVariable<E::ScalarField>],
        proof_scalars: &[EmulatedVariable<E::ScalarField>],
    ) -> Result<(), CircuitError>;

    /// This gadget is the counter part to the above gadget, it verifies the scalars used in the msm have been calculated correctly.
    /// If `P` is an elliptic curve then the [`AtomicAccumulatorGadget::verify_atomic_level_one`] would be called on a circuit defined over
    /// `P::BaseField` and this gadget would be called on a circuit defined over `P::ScalarField`.
    fn verify_atomic_level_two(
        &mut self,
        r_challenge: Variable,
        points: &[Variable],
        evaluations: &[Variable],
        instance_scalars: &[Variable],
        proof_scalars: &[Variable],
    ) -> Result<(), CircuitError>;
}

impl<E: HasTEForm> AtomicAccumulatorGadget<E> for PlonkCircuit<E::BaseField>
where
    E::ScalarField: EmulationConfig<E::BaseField> + RescueParameter,
    E::BaseField: RescueParameter + PrimeField,
{
    /// This method is exclusively used in tests.
    fn verify_atomic_accumulation(
        &mut self,
        old_instances: &[AtomicInstanceVar<E>],
        new_accumulator: &AtomicInstanceVar<E>,
        srs_base_point: &PointVariable,
    ) -> Result<(), CircuitError> {
        let mut transcript = RescueTranscriptVar::new_transcript(self);

        for instance in old_instances.iter() {
            transcript.append_point_variable(&instance.comm, self)?;
            transcript.push_emulated_variable(&instance.point, self)?;
            transcript.push_emulated_variable(&instance.value, self)?;
            transcript.append_point_variable(&instance.opening_proof, self)?;
        }

        let r = transcript.squeeze_scalar_challenge::<E>(self)?;
        let r = self.to_emulated_variable(r)?;
        let zero = self.emulated_zero();
        let mut r_power = self.emulated_one();
        let mut r_powers = vec![r_power.clone()];
        for _ in 0..(old_instances.len() - 1) {
            r_power = self.emulated_mul(&r_power, &r)?;
            r_powers.push(r_power.clone());
        }

        let z_r_powers = r_powers
            .iter()
            .zip(old_instances.iter().map(|instance| &instance.point))
            .map(|(r_power, point)| self.emulated_mul(r_power, point))
            .collect::<Result<Vec<EmulatedVariable<E::ScalarField>>, _>>()?;
        let minus_v_r_powers = r_powers
            .iter()
            .zip(old_instances.iter().map(|instance| &instance.value))
            .map(|(r_power, value)| self.emulated_mul(r_power, value))
            .collect::<Result<Vec<EmulatedVariable<E::ScalarField>>, _>>()?;
        let minus_v_r_powers = minus_v_r_powers
            .iter()
            .map(|value| self.emulated_sub(&zero, value))
            .collect::<Result<Vec<EmulatedVariable<E::ScalarField>>, _>>()?;
        let g_vec = vec![*srs_base_point; old_instances.len()];
        let instance_bases = old_instances
            .iter()
            .map(|instance| instance.comm)
            .chain(g_vec)
            .chain(old_instances.iter().map(|instance| instance.opening_proof))
            .collect::<Vec<PointVariable>>();

        let instance_scalars = r_powers
            .iter()
            .take(old_instances.len())
            .chain(minus_v_r_powers.iter())
            .chain(z_r_powers.iter())
            .cloned()
            .collect::<Vec<_>>();

        let out_instance = <Self as EmulMultiScalarMultiplicationCircuit<_, E>>::msm(
            self,
            &instance_bases,
            &instance_scalars,
        )?;
        let proof_bases = old_instances
            .iter()
            .map(|instance| instance.opening_proof)
            .collect::<Vec<PointVariable>>();
        let out_proof = <Self as EmulMultiScalarMultiplicationCircuit<_, E>>::msm(
            self,
            &proof_bases,
            &r_powers,
        )?;

        self.enforce_point_equal(&out_instance, &new_accumulator.comm)?;
        self.enforce_point_equal(&out_proof, &new_accumulator.opening_proof)?;

        Ok(())
    }

    /// This method is exclusively used in tests.
    fn verify_atomic_level_one(
        &mut self,
        old_instances: &[AtomicInstanceVar<E>],
        new_accumulator: &AtomicInstanceVar<E>,
        srs_base_point: &PointVariable,
        r_challenge: &EmulatedVariable<E::ScalarField>,
        instance_scalars: &[EmulatedVariable<E::ScalarField>],
        proof_scalars: &[EmulatedVariable<E::ScalarField>],
    ) -> Result<(), CircuitError> {
        let mut transcript = RescueTranscriptVar::new_transcript(self);

        for instance in old_instances.iter() {
            transcript.append_point_variable(&instance.comm, self)?;
            transcript.push_emulated_variable(&instance.point, self)?;
            transcript.push_emulated_variable(&instance.value, self)?;
            transcript.append_point_variable(&instance.opening_proof, self)?;
        }

        let r = transcript.squeeze_scalar_challenge::<E>(self)?;

        let r_challenge = self.mod_to_native_field(r_challenge)?;
        self.enforce_equal(r_challenge, r)?;

        let g_vec = vec![*srs_base_point; 1];
        let instance_bases = old_instances
            .iter()
            .map(|instance| instance.comm)
            .chain(g_vec)
            .chain(old_instances.iter().map(|instance| instance.opening_proof))
            .collect::<Vec<PointVariable>>();

        let out_instance = <Self as EmulMultiScalarMultiplicationCircuit<_, E>>::msm(
            self,
            &instance_bases,
            instance_scalars,
        )?;

        let proof_bases = old_instances
            .iter()
            .map(|instance| instance.opening_proof)
            .collect::<Vec<PointVariable>>();
        let out_proof = <Self as EmulMultiScalarMultiplicationCircuit<_, E>>::msm(
            self,
            &proof_bases,
            proof_scalars,
        )?;

        self.enforce_point_equal(&out_instance, &new_accumulator.comm)?;
        self.enforce_point_equal(&out_proof, &new_accumulator.opening_proof)?;

        Ok(())
    }

    /// This method is exclusively used in tests.
    fn verify_atomic_level_two(
        &mut self,
        r_challenge: Variable,
        points: &[Variable],
        evaluations: &[Variable],
        instance_scalars: &[Variable],
        proof_scalars: &[Variable],
    ) -> Result<(), CircuitError> {
        let mut r_powers = vec![self.one()];
        let mut r_power = self.one();
        for _ in 0..proof_scalars.len() {
            r_power = self.mul(r_power, r_challenge)?;
            r_powers.push(r_power);
        }

        let z_r_scalars = r_powers
            .iter()
            .zip(points.iter())
            .map(|(r_power, point)| self.mul(*r_power, *point))
            .collect::<Result<Vec<Variable>, _>>()?;

        let minus_v_r_scalars = r_powers.iter().zip(evaluations.iter()).try_fold(
            self.zero(),
            |acc, (&r_power, &evaluation)| {
                let wires_in = [r_power, evaluation, acc, self.one()];
                let q_muls = [E::BaseField::one(), E::BaseField::one()];
                self.mul_add(&wires_in, &q_muls)
            },
        )?;
        let minus_v_r_scalars = self.sub(self.zero(), minus_v_r_scalars)?;

        let calc_instance_scalars = r_powers
            .iter()
            .take(points.len())
            .chain(vec![&minus_v_r_scalars])
            .chain(z_r_scalars.iter())
            .chain(r_powers.iter().skip(points.len()))
            .copied()
            .collect::<Vec<_>>();

        for (calc_instance_scalar, instance_scalar) in
            calc_instance_scalars.iter().zip(instance_scalars.iter())
        {
            self.enforce_equal(*calc_instance_scalar, *instance_scalar)?;
        }

        for (r_power, proof_scalar) in r_powers.iter().zip(proof_scalars.iter()) {
            self.enforce_equal(*r_power, *proof_scalar)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        nightfall::{
            accumulation::{accumulation_structs::AtomicAccumulator, UVAtomicAccumulator},
            mle::{zeromorph::Zeromorph, MLEPlonk},
            UnivariateIpaPCS,
        },
        proof_system::UniversalSNARK,
        transcript::{RescueTranscript, Transcript},
    };

    use super::*;

    use ark_bn254::{g1::Config as G1Config, Bn254, Fq as FqBn254, Fr as FrBn254};
    use ark_ff::Field;
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
    use ark_std::{cfg_into_iter, cfg_iter, UniformRand};
    use jf_primitives::{pcs::prelude::*, rescue::sponge::RescueCRHF};
    use jf_relation::{gadgets::ecc::Point, Circuit};
    use nf_curves::grumpkin::{short_weierstrass::SWGrumpkin, Grumpkin};
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
    use rayon::prelude::*;

    #[test]
    fn test_verify_atomic_accumulation() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let degree = 50;
        let mut kzg_atomic_accumulator = UVAtomicAccumulator::<UnivariateKzgPCS<Bn254>>::new();
        let pp = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(&mut rng, degree).unwrap();
        let (ck, _) = pp.trim(degree).unwrap();

        for _ in 0..20 {
            let commit_in =
                <DensePolynomial<FrBn254> as DenseUVPolynomial<FrBn254>>::rand(degree, &mut rng);
            let comm = UnivariateKzgPCS::commit(&ck, &commit_in).unwrap();
            let point = FrBn254::rand(&mut rng);
            let (opening_proof, value) = UnivariateKzgPCS::open(&ck, &commit_in, &point).unwrap();
            kzg_atomic_accumulator.push(comm, point, value, opening_proof);
        }
        let mut transcript = <RescueTranscript<FqBn254> as Transcript>::new_transcript(b"test");
        let new_acc = kzg_atomic_accumulator
            .prove_accumulation(&ck, Some(&mut transcript))
            .unwrap();

        let mut circuit = PlonkCircuit::<FqBn254>::new_ultra_plonk(16);
        let mut old_instances = Vec::<AtomicInstanceVar<G1Config>>::new();

        for instance in kzg_atomic_accumulator.instances.iter() {
            let comm_point = Point::from(instance.comm);
            let comm = circuit.create_point_variable(&comm_point).unwrap();
            let opening_pf_point = Point::from(instance.opening_proof.proof);
            let point = circuit.create_emulated_variable(instance.point).unwrap();
            let value = circuit.create_emulated_variable(instance.value).unwrap();
            let opening_proof = circuit.create_point_variable(&opening_pf_point).unwrap();

            let tmp_instance = AtomicInstanceVar::new(comm, value, point, opening_proof);
            old_instances.push(tmp_instance);
        }

        let new_acc_var =
            AtomicInstanceVar::from_struct::<Bn254>(&new_acc.instances[0], &mut circuit).unwrap();

        let srs_base_point = circuit
            .create_point_variable(&Point::from(ck.powers_of_g[0]))
            .unwrap();

        circuit
            .verify_atomic_accumulation(&old_instances, &new_acc_var, &srs_base_point)
            .unwrap();

        circuit
            .finalize_for_recursive_mle_arithmetization::<RescueCRHF<FqBn254>>()
            .unwrap();
        let pi = circuit.public_input().unwrap()[0];
        circuit.check_circuit_satisfiability(&[pi]).unwrap();
    }

    #[test]
    fn verify_atomic_level_one_test() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let degree = 50;
        let mut kzg_atomic_accumulator = UVAtomicAccumulator::<UnivariateKzgPCS<Bn254>>::new();
        let pp = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(&mut rng, degree).unwrap();
        let (ck, _) = pp.trim(degree).unwrap();

        for _ in 0..20 {
            let commit_in =
                <DensePolynomial<FrBn254> as DenseUVPolynomial<FrBn254>>::rand(degree, &mut rng);
            let comm = UnivariateKzgPCS::commit(&ck, &commit_in).unwrap();
            let point = FrBn254::rand(&mut rng);
            let (opening_proof, value) = UnivariateKzgPCS::open(&ck, &commit_in, &point).unwrap();
            kzg_atomic_accumulator.push(comm, point, value, opening_proof);
        }
        let mut transcript = <RescueTranscript<FqBn254> as Transcript>::new_transcript(b"test");
        let new_acc = kzg_atomic_accumulator
            .prove_accumulation(&ck, Some(&mut transcript))
            .unwrap();

        let mut transcript = <RescueTranscript<FqBn254> as Transcript>::new_transcript(b"test");

        for instance in kzg_atomic_accumulator.instances.iter() {
            <RescueTranscript<FqBn254> as Transcript>::append_curve_point(
                &mut transcript,
                b"commitment",
                &instance.comm,
            )
            .unwrap();
            <RescueTranscript<FqBn254> as Transcript>::push_message(
                &mut transcript,
                b"point",
                &instance.point,
            )
            .unwrap();
            <RescueTranscript<FqBn254> as Transcript>::push_message(
                &mut transcript,
                b"value",
                &instance.value,
            )
            .unwrap();
            <RescueTranscript<FqBn254> as Transcript>::append_curve_point(
                &mut transcript,
                b"opening_proof",
                &instance.opening_proof.proof,
            )
            .unwrap();
        }

        let r = transcript
            .squeeze_scalar_challenge::<G1Config>(b"r")
            .unwrap();
        let r_powers = cfg_into_iter!(0..kzg_atomic_accumulator.instances.len())
            .map(|i| r.pow([i as u64]))
            .collect::<Vec<_>>();

        let minus_v_r_powers_bigints = cfg_iter!(kzg_atomic_accumulator.instances)
            .zip(cfg_iter!(r_powers))
            .map(|(instance, r_power)| (-instance.value * *r_power))
            .sum::<FrBn254>();

        let z_r_powers_big_ints = cfg_iter!(kzg_atomic_accumulator.instances)
            .zip(cfg_iter!(r_powers))
            .map(|(instance, r_power)| (instance.point * *r_power))
            .collect::<Vec<_>>();

        let scalars_comms = r_powers
            .iter()
            .copied()
            .take(kzg_atomic_accumulator.instances.len())
            .chain(vec![minus_v_r_powers_bigints])
            .chain(z_r_powers_big_ints.iter().copied())
            .collect::<Vec<_>>();

        let mut circuit = PlonkCircuit::<FqBn254>::new_ultra_plonk(16);
        let mut old_instances = Vec::<AtomicInstanceVar<G1Config>>::new();

        for instance in kzg_atomic_accumulator.instances.iter() {
            let comm_point = Point::from(instance.comm);
            let comm = circuit.create_point_variable(&comm_point).unwrap();
            let opening_pf_point = Point::from(instance.opening_proof.proof);
            let point = circuit.create_emulated_variable(instance.point).unwrap();
            let value = circuit.create_emulated_variable(instance.value).unwrap();
            let opening_proof = circuit.create_point_variable(&opening_pf_point).unwrap();

            let tmp_instance = AtomicInstanceVar::new(comm, value, point, opening_proof);
            old_instances.push(tmp_instance);
        }

        let new_acc_var =
            AtomicInstanceVar::from_struct::<Bn254>(&new_acc.instances[0], &mut circuit).unwrap();

        let srs_base_point = circuit
            .create_point_variable(&Point::from(ck.powers_of_g[0]))
            .unwrap();

        let r_challenge = circuit.create_emulated_variable(r).unwrap();

        let instance_scalars = scalars_comms
            .iter()
            .map(|scalar| circuit.create_emulated_variable(*scalar))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let proof_scalars = instance_scalars
            .iter()
            .take(kzg_atomic_accumulator.instances.len())
            .cloned()
            .collect::<Vec<_>>();
        circuit
            .verify_atomic_level_one(
                &old_instances,
                &new_acc_var,
                &srs_base_point,
                &r_challenge,
                &instance_scalars,
                &proof_scalars,
            )
            .unwrap();

        circuit
            .finalize_for_recursive_mle_arithmetization::<RescueCRHF<FrBn254>>()
            .unwrap();
        let pi = circuit.public_input().unwrap()[0];
        circuit.check_circuit_satisfiability(&[pi]).unwrap();
        let srs_size = circuit.num_gates().ilog2() as usize;
        let srs = Zeromorph::<UnivariateIpaPCS<Grumpkin>>::gen_srs_for_testing(&mut rng, srs_size)
            .unwrap();
        let (pk, _vk) =
            MLEPlonk::<Zeromorph<UnivariateIpaPCS<Grumpkin>>>::preprocess(&srs, None, &circuit, false)
                .unwrap();

        let _proof = MLEPlonk::<Zeromorph<UnivariateIpaPCS<Grumpkin>>>::sa_prove::<
            _,
            _,
            _,
            RescueTranscript<FrBn254>,
        >(&circuit, &pk)
        .unwrap()
        .0;
    }

    #[test]
    fn verify_atomic_level_two_test() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let degree = 50;
        let mut kzg_atomic_accumulator = UVAtomicAccumulator::<UnivariateKzgPCS<Bn254>>::new();
        let pp = UnivariateKzgPCS::<Bn254>::gen_srs_for_testing(&mut rng, degree).unwrap();
        let (ck, _) = pp.trim(degree).unwrap();

        for _ in 0..20 {
            let commit_in =
                <DensePolynomial<FrBn254> as DenseUVPolynomial<FrBn254>>::rand(degree, &mut rng);
            let comm = UnivariateKzgPCS::commit(&ck, &commit_in).unwrap();
            let point = FrBn254::rand(&mut rng);
            let (opening_proof, value) = UnivariateKzgPCS::open(&ck, &commit_in, &point).unwrap();
            kzg_atomic_accumulator.push(comm, point, value, opening_proof);
        }

        let mut transcript = <RescueTranscript<FqBn254> as Transcript>::new_transcript(b"test");
        let _ = kzg_atomic_accumulator
            .prove_accumulation(&ck, Some(&mut transcript))
            .unwrap();

        let mut transcript = <RescueTranscript<FqBn254> as Transcript>::new_transcript(b"test");

        for instance in kzg_atomic_accumulator.instances.iter() {
            <RescueTranscript<FqBn254> as Transcript>::append_curve_point(
                &mut transcript,
                b"commitment",
                &instance.comm,
            )
            .unwrap();
            <RescueTranscript<FqBn254> as Transcript>::push_message(
                &mut transcript,
                b"point",
                &instance.point,
            )
            .unwrap();
            <RescueTranscript<FqBn254> as Transcript>::push_message(
                &mut transcript,
                b"value",
                &instance.value,
            )
            .unwrap();
            <RescueTranscript<FqBn254> as Transcript>::append_curve_point(
                &mut transcript,
                b"opening_proof",
                &instance.opening_proof.proof,
            )
            .unwrap();
        }

        let r = transcript
            .squeeze_scalar_challenge::<G1Config>(b"r")
            .unwrap();
        let r_powers = cfg_into_iter!(0..=kzg_atomic_accumulator.instances.len())
            .map(|i| r.pow([i as u64]))
            .collect::<Vec<_>>();

        let minus_v_r_powers_bigints = cfg_iter!(kzg_atomic_accumulator.instances)
            .zip(cfg_iter!(r_powers))
            .map(|(instance, r_power)| (-instance.value * *r_power))
            .sum::<FrBn254>();

        let z_r_powers_big_ints = cfg_iter!(kzg_atomic_accumulator.instances)
            .zip(cfg_iter!(r_powers))
            .map(|(instance, r_power)| (instance.point * *r_power))
            .collect::<Vec<_>>();

        let scalars_comms = r_powers
            .iter()
            .copied()
            .take(kzg_atomic_accumulator.instances.len())
            .chain(vec![minus_v_r_powers_bigints])
            .chain(z_r_powers_big_ints.iter().copied())
            .chain(
                r_powers
                    .iter()
                    .skip(kzg_atomic_accumulator.instances.len())
                    .copied(),
            )
            .collect::<Vec<_>>();

        let mut circuit = PlonkCircuit::<FrBn254>::new_ultra_plonk(16);

        let mut points = Vec::<Variable>::new();
        let mut evaluations = Vec::<Variable>::new();

        for instance in kzg_atomic_accumulator.instances.iter() {
            let point = circuit.create_variable(instance.point).unwrap();
            let value = circuit.create_variable(instance.value).unwrap();
            points.push(point);
            evaluations.push(value);
        }

        let instance_scalars = scalars_comms
            .iter()
            .map(|scalar| circuit.create_variable(*scalar))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let proof_scalars = r_powers
            .iter()
            .map(|r_power| circuit.create_variable(*r_power))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let r_var = circuit.create_variable(r).unwrap();
        <PlonkCircuit<FrBn254> as AtomicAccumulatorGadget<SWGrumpkin>>::verify_atomic_level_two(
            &mut circuit,
            r_var,
            &points,
            &evaluations,
            &instance_scalars,
            &proof_scalars,
        )
        .unwrap();

        circuit.finalize_for_arithmetization().unwrap();

        circuit.check_circuit_satisfiability(&[]).unwrap();
    }
}
