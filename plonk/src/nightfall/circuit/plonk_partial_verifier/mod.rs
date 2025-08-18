// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Circuits for Plonk verifiers.
use crate::{
    nightfall::ipa_structs::{PlookupVerifyingKey, VK},
    transcript::*,
};
use ark_ec::{short_weierstrass::Affine, AffineRepr};
use ark_ff::PrimeField;
use ark_poly::univariate::DensePolynomial;
use ark_std::{vec, vec::Vec};
use jf_primitives::{
    pcs::{PolynomialCommitmentScheme, StructuredReferenceString},
    rescue::RescueParameter,
};
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{HasTEForm, Point, PointVariable},
        EmulatedVariable, EmulationConfig,
    },
    PlonkCircuit, Variable,
};

mod gadgets;
mod poly;
mod proof_to_var;
mod structs;

pub use gadgets::*;
pub use poly::*;
pub use proof_to_var::*;
pub use structs::*;

#[derive(Debug, Clone, Eq, PartialEq)]
/// Represent variable of a Plonk verifying key.
pub struct VerifyingKeyVar<PCS: PolynomialCommitmentScheme> {
    /// The variables for the permutation polynomial commitments.
    pub(crate) sigma_comms: Vec<PointVariable>,
    /// The variables for the selector polynomial commitments.
    pub(crate) selector_comms: Vec<PointVariable>,
    /// A flag indicating whether the key is a merged key.
    is_merged: bool,
    /// Plookup verifying key variable.
    pub(crate) plookup_vk: Option<PlookupVerifyingKeyVar>,

    /// The size of the evaluation domain. Should be a power of two.
    domain_size: usize,

    /// The number of public inputs.
    num_inputs: usize,

    /// The constants K0, ..., K_num_wire_types that ensure wire subsets are
    /// disjoint.
    k: Vec<PCS::Evaluation>,

    /// The hash of the verification key.
    hash: EmulatedVariable<PCS::Evaluation>,
}

impl<T, F, PCS> CircuitTranscriptVisitor<T, F> for VerifyingKeyVar<PCS>
where
    T: CircuitTranscript<F>,
    F: PrimeField,
    PCS: PolynomialCommitmentScheme,
    PCS::Commitment: AffineRepr<BaseField = F>,
    PCS::Evaluation: EmulationConfig<F>,
{
    fn append_to_transcript(
        &self,
        transcript: &mut T,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError> {
        transcript.push_emulated_variable(&self.hash, circuit)
    }
}

/// A struct used to represent a [`PlookupVerifyingKey`] as a variable.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PlookupVerifyingKeyVar {
    /// Range table polynomial commitment. The commitment is not hiding.
    pub(crate) range_table_comm: PointVariable,

    /// Key table polynomial commitment. The commitment is not hiding.
    pub(crate) key_table_comm: PointVariable,

    /// Table domain separation polynomial commitment. The commitment is not
    /// hiding.
    pub(crate) table_dom_sep_comm: PointVariable,

    /// Lookup domain separation selector polynomial commitment. The commitment
    /// is not hiding.
    pub(crate) q_dom_sep_comm: PointVariable,
}

impl PlookupVerifyingKeyVar {
    /// Create a new [`PlookupVerifyingKeyVar`].
    pub fn new(
        range_table_comm: PointVariable,
        key_table_comm: PointVariable,
        table_dom_sep_comm: PointVariable,
        q_dom_sep_comm: PointVariable,
    ) -> Self {
        Self {
            range_table_comm,
            key_table_comm,
            table_dom_sep_comm,
            q_dom_sep_comm,
        }
    }

    /// Create a new [`PlookupVerifyingKeyVar`] from a [`PlookupVerifyingKey`].
    pub fn from_struct<PCS, P>(
        vk: &PlookupVerifyingKey<PCS>,
        circuit: &mut PlonkCircuit<P::BaseField>,
    ) -> Result<Self, CircuitError>
    where
        PCS: PolynomialCommitmentScheme<Commitment = Affine<P>>,
        P: HasTEForm,
        P::BaseField: PrimeField + RescueParameter,
    {
        let range_table_comm = circuit.create_point_variable(&Point::from(vk.range_table_comm))?;
        let key_table_comm = circuit.create_point_variable(&Point::from(vk.key_table_comm))?;
        let table_dom_sep_comm =
            circuit.create_point_variable(&Point::from(vk.table_dom_sep_comm))?;
        let q_dom_sep_comm = circuit.create_point_variable(&Point::from(vk.q_dom_sep_comm))?;
        Ok(Self::new(
            range_table_comm,
            key_table_comm,
            table_dom_sep_comm,
            q_dom_sep_comm,
        ))
    }
}

impl<T, F> CircuitTranscriptVisitor<T, F> for PlookupVerifyingKeyVar
where
    T: CircuitTranscript<F>,
    F: PrimeField,
{
    fn append_to_transcript(
        &self,
        transcript: &mut T,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError> {
        transcript.append_point_variable(&self.range_table_comm, circuit)?;
        transcript.append_point_variable(&self.key_table_comm, circuit)?;
        transcript.append_point_variable(&self.table_dom_sep_comm, circuit)?;
        transcript.append_point_variable(&self.q_dom_sep_comm, circuit)?;
        Ok(())
    }
}

impl<PCS, P, F> VerifyingKeyVar<PCS>
where
    PCS: PolynomialCommitmentScheme<
        Commitment = Affine<P>,
        Evaluation = P::ScalarField,
        Polynomial = DensePolynomial<P::ScalarField>,
        Point = P::ScalarField,
    >,
    P: HasTEForm<BaseField = F>,
    P::ScalarField: PrimeField + RescueParameter + EmulationConfig<F>,
    F: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
    PCS::Proof: ProofToVar<P>,
    PCS::SRS: StructuredReferenceString<Item = Affine<P>>,
{
    /// Create a variable for a Plonk verifying key.
    pub fn new<VerifyingKey>(
        circuit: &mut PlonkCircuit<F>,
        verify_key: &VerifyingKey,
    ) -> Result<Self, CircuitError>
    where
        VerifyingKey: VK<PCS>,
    {
        let sigma_comms = verify_key
            .sigma_comms()
            .iter()
            .map(|comm| circuit.create_point_variable(&Point::from(*comm)))
            .collect::<Result<Vec<_>, CircuitError>>()?;
        let selector_comms = verify_key
            .selector_comms()
            .iter()
            .map(|comm| circuit.create_point_variable(&Point::from(*comm)))
            .collect::<Result<Vec<_>, CircuitError>>()?;
        let plookup_vk = if let Some(plookup_vk) = verify_key.plookup_vk() {
            Some(PlookupVerifyingKeyVar::from_struct(plookup_vk, circuit)?)
        } else {
            None
        };

        let hash = circuit.create_emulated_variable(verify_key.hash())?;
        Ok(Self {
            sigma_comms,
            selector_comms,
            plookup_vk,
            is_merged: verify_key.is_merged(),
            domain_size: verify_key.domain_size(),
            num_inputs: verify_key.num_inputs(),
            k: verify_key.k().to_vec(),
            hash,
        })
    }

    /// Convert to a list of variables.
    pub fn to_vec(&self) -> Vec<Variable> {
        let mut res = vec![];
        for sigma_comm in self.sigma_comms.iter() {
            res.push(sigma_comm.get_x());
            res.push(sigma_comm.get_y());
        }
        for selector_comm in self.selector_comms.iter() {
            res.push(selector_comm.get_x());
            res.push(selector_comm.get_y());
        }

        if self.plookup_vk.is_some() {
            let plookup_vk = self.plookup_vk.as_ref().unwrap();
            res.push(plookup_vk.range_table_comm.get_x());
            res.push(plookup_vk.range_table_comm.get_y());
            res.push(plookup_vk.key_table_comm.get_x());
            res.push(plookup_vk.key_table_comm.get_y());
            res.push(plookup_vk.table_dom_sep_comm.get_x());
            res.push(plookup_vk.table_dom_sep_comm.get_y());
            res.push(plookup_vk.q_dom_sep_comm.get_x());
            res.push(plookup_vk.q_dom_sep_comm.get_y());
        }
        res.extend_from_slice(&self.hash.to_vec());
        res
    }

    /// Returns a vector of [`PointVariable`]s for that agrees with
    /// the `comms()` method of a [`VerifyingKey<PCS>`].
    pub fn comms(&self) -> Vec<PointVariable> {
        let mut res = vec![];
        res.extend_from_slice(&self.sigma_comms);
        res.extend_from_slice(&self.selector_comms);
        if let Some(plookup_vk) = &self.plookup_vk {
            res.push(plookup_vk.range_table_comm);
            res.push(plookup_vk.key_table_comm);
            res.push(plookup_vk.table_dom_sep_comm);
            res.push(plookup_vk.q_dom_sep_comm);
        }
        res
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        errors::PlonkError,
        nightfall::{ipa_verifier::FFTVerifier, FFTPlonk},
        proof_system::UniversalSNARK,
        transcript::RescueTranscript,
    };
    use ark_bn254::{g1::Config as BnConfig, Bn254, Fq as Fq254, Fr as Fr254};
    use ark_ec::{short_weierstrass::Projective, CurveGroup, VariableBaseMSM};
    use ark_std::UniformRand;
    use jf_primitives::{
        pcs::{prelude::*, Accumulation},
        rescue::{sponge::RescueCRHF, RescueParameter},
    };
    use jf_relation::{
        gadgets::ecc::{EmulMultiScalarMultiplicationCircuit, Point},
        Arithmetization, Circuit,
    };
    use jf_utils::test_rng;
    use nf_curves::grumpkin::short_weierstrass::SWGrumpkin;

    pub(crate) fn new_circuit_for_test_ultra<F: PrimeField>(
        public_input: F,
        i: usize,
    ) -> Result<PlonkCircuit<F>, PlonkError> {
        let mut circuit = PlonkCircuit::new_ultra_plonk(8);
        let shared_pub_var = circuit.create_public_variable(public_input)?;
        let mut var = shared_pub_var;
        for _ in 0..i {
            var = circuit.add(var, shared_pub_var)?;
        }
        let range_var = circuit.create_variable(-F::from(1u8) - F::from(233456u64))?;
        circuit.enforce_in_range(range_var, F::MODULUS_BIT_SIZE as usize)?;
        Ok(circuit)
    }

    #[test]
    fn test_compute_scalars_native() -> Result<(), CircuitError> {
        compute_scalars_native_helper::<UnivariateKzgPCS<Bn254>, Fq254, BnConfig, SWGrumpkin>()
    }
    fn compute_scalars_native_helper<PCS, F, P, E>() -> Result<(), CircuitError>
    where
        PCS: Accumulation<
            Commitment = Affine<P>,
            Evaluation = P::ScalarField,
            Polynomial = DensePolynomial<P::ScalarField>,
            Point = P::ScalarField,
        >,
        F: RescueParameter + PrimeField + EmulationConfig<P::ScalarField>,
        P: HasTEForm<BaseField = F>,
        E: HasTEForm<BaseField = P::ScalarField, ScalarField = F>,
        P::ScalarField: PrimeField + RescueParameter + EmulationConfig<F>,
        PCS::Proof: ProofToVar<P>,
        PCS::SRS: StructuredReferenceString<Item = Affine<P>>,
    {
        let rng = &mut test_rng();

        for _ in 8..12 {
            // =======================================
            // setup
            // =======================================

            // 1. Simulate universal setup

            let public_input = P::ScalarField::rand(rng);

            let grump_gen = E::GENERATOR;
            let point = Point::<P::ScalarField>::from(grump_gen);
            let scalar = E::ScalarField::rand(rng);

            // 2. Create circuit
            let mut circuit =
                new_circuit_for_test_ultra::<P::ScalarField>(public_input, usize::rand(rng) % 50)
                    .unwrap();
            let emulated_scalar = circuit.create_emulated_variable(scalar)?;
            let point_var = circuit.create_point_variable(&point)?;
            let _ = EmulMultiScalarMultiplicationCircuit::<P::ScalarField, E>::msm(
                &mut circuit,
                &[point_var],
                &[emulated_scalar],
            )?;
            circuit
                .finalize_for_recursive_arithmetization::<RescueCRHF<P::ScalarField>>()
                .unwrap();
            let public_inputs = circuit.public_input().unwrap();
            let max_degree = circuit.srs_size()?;
            let srs = FFTPlonk::<PCS>::universal_setup_for_testing(max_degree, rng).unwrap();

            // 3. Create proof
            let (pk, vk) = FFTPlonk::<PCS>::preprocess(&srs, &circuit).unwrap();
            let proof = FFTPlonk::<PCS>::recursive_prove::<_, _, RescueTranscript<P::BaseField>>(
                rng, &circuit, &pk, None,
            )
            .unwrap();

            // 4. Verification

            let verifier = FFTVerifier::<PCS>::new(vk.domain_size).unwrap();

            let pcs_info = verifier
                .prepare_pcs_info::<RescueTranscript<P::BaseField>>(
                    &vk,
                    &public_inputs,
                    &proof.proof,
                    &None,
                )
                .unwrap();

            // Compute commitment to g(x).
            let g_comm = pcs_info
                .comm_scalars_and_bases
                .multi_scalar_mul()
                .into_affine();

            let challenges = FFTVerifier::<PCS>::compute_challenges::<
                RescueTranscript<P::BaseField>,
            >(&vk, &public_inputs, &proof.proof, &None)?;

            let mut circuit = PlonkCircuit::<P::ScalarField>::new_turbo_plonk();
            let tau = circuit.create_variable(challenges.tau)?;
            let alpha = circuit.create_variable(challenges.alpha)?;
            let alpha_squared = circuit.mul(alpha, alpha)?;
            let alpha_cubed = circuit.mul(alpha_squared, alpha)?;
            let beta = circuit.create_variable(challenges.beta)?;
            let gamma = circuit.create_variable(challenges.gamma)?;
            let zeta = circuit.create_variable(challenges.zeta)?;
            let v = circuit.create_variable(challenges.v)?;
            let u = circuit.create_variable(challenges.u)?;

            let challenges_var = ChallengesVar {
                tau,
                alphas: [alpha, alpha_squared, alpha_cubed],
                beta,
                gamma,
                zeta,
                v,
                u,
            };

            let proof_evals =
                ProofEvalsVarNative::from_struct(&mut circuit, &proof.proof.poly_evals)?;
            let lookup_evals = PlookupEvalsVarNative::from_struct(
                &mut circuit,
                &proof.proof.plookup_proof.as_ref().unwrap().poly_evals,
            )?;
            let vk_k = vk
                .k()
                .iter()
                .map(|k| circuit.create_variable(*k))
                .collect::<Result<Vec<_>, CircuitError>>()?;

            //let pi_var = circuit.create_variable(pi)?;
            let mut pi_vars = vec![];
            for pi in public_inputs {
                pi_vars.push(circuit.create_variable(pi).unwrap());
            }

            let scalars = compute_scalars_for_native_field::<P::ScalarField>(
                &mut circuit,
                pi_vars,
                &challenges_var,
                &proof_evals,
                Some(lookup_evals),
                &vk_k,
                verifier.domain.size as usize,
            )?;

            let real_scalars = pcs_info.comm_scalars_and_bases.scalars.clone();
            let mut real_scalars_clone = real_scalars.clone();
            real_scalars_clone[10] += real_scalars[22];
            real_scalars_clone[11] += real_scalars[48];
            real_scalars_clone[12] += real_scalars[47];

            let _ = real_scalars_clone.remove(22);
            let mut comms = pcs_info.comm_scalars_and_bases.bases()[..47].to_vec();

            let _ = comms.remove(22);
            let scalars = scalars
                .iter()
                .map(|s| circuit.witness(*s))
                .collect::<Result<Vec<_>, _>>()?;

            let scalars_bigints = scalars.iter().map(|s| s.into_bigint()).collect::<Vec<_>>();

            let computed_g_comm =
                Projective::<P>::msm_bigint(&comms, &scalars_bigints).into_affine();

            assert_eq!(g_comm, computed_g_comm);
        }
        Ok(())
    }

    #[test]
    fn test_compute_scalars_zero() {
        let mut circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(8);

        let challenges_var = ChallengesVar {
            tau: circuit.zero(),
            alphas: [circuit.zero(), circuit.zero(), circuit.zero()],
            beta: circuit.zero(),
            gamma: circuit.zero(),
            zeta: circuit.zero(),
            v: circuit.zero(),
            u: circuit.zero(),
        };

        let proof_evals = ProofEvalsVarNative {
            wires_evals: vec![circuit.zero(); 6],
            wire_sigma_evals: vec![circuit.zero(); 5],
            perm_next_eval: circuit.zero(),
        };

        let lookup_evals = PlookupEvalsVarNative {
            range_table_eval: circuit.zero(),
            key_table_eval: circuit.zero(),
            table_dom_sep_eval: circuit.zero(),
            q_dom_sep_eval: circuit.zero(),
            h_1_eval: circuit.zero(),
            h_2_next_eval: circuit.zero(),
            prod_next_eval: circuit.zero(),
            q_lookup_eval: circuit.zero(),
            range_table_next_eval: circuit.zero(),
            key_table_next_eval: circuit.zero(),
            q_lookup_next_eval: circuit.zero(),
            h_1_next_eval: circuit.zero(),
            w_3_next_eval: circuit.zero(),
            w_4_next_eval: circuit.zero(),
            table_dom_sep_next_eval: circuit.zero(),
        };

        let vk_k = vec![circuit.zero(); 6];
        let scalars = compute_scalars_for_native_field::<Fr254>(
            &mut circuit,
            vec![0],
            &challenges_var,
            &proof_evals,
            Some(lookup_evals),
            &vk_k,
            1 << 15,
        )
        .unwrap();

        for var in scalars.iter() {
            ark_std::println!("scalar value: {}", circuit.witness(*var).unwrap());
        }
    }
}
