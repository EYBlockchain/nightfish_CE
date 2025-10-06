// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

use ark_bn254::Fr as Fr254;
use ark_ec::{pairing::Pairing, short_weierstrass::Affine, AffineRepr};
use ark_ff::PrimeField;

use ark_std::{marker::PhantomData, string::ToString, vec, vec::Vec};
use jf_primitives::{
    pcs::{prelude::UnivariateKzgPCS, Accumulation, PolynomialCommitmentScheme},
    rescue::RescueParameter,
};
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{CircuitPoint, EmulatedPointVariable, HasTEForm, Point, PointVariable},
        EmulatedVariable, EmulationConfig,
    },
    Circuit, PlonkCircuit, Variable,
};

use crate::{
    nightfall::{
        circuit::subroutine_verifiers::{
            gkr::{EmulatedGKRProofVar, GKRProofVar},
            structs::{EmulatedSumCheckProofVar, SumCheckProofVar},
            sumcheck::SumCheckGadget,
        },
        hops::srs::UnivariateUniversalIpaParams,
        ipa_structs::{Challenges as PCSChallenges, PlookupProof, Proof as PCSProof, VK},
        mle::mle_structs::{
            MLEChallenges, MLELookupEvals, MLELookupProof, MLEProofEvals, MLEVerifyingKey,
            SAMLEProof,
        },
    },
    proof_system::structs::{PlookupEvaluations, ProofEvaluations},
    recursion::merge_functions::Bn254Output,
    transcript::{rescue::RescueTranscriptVar, CircuitTranscript, CircuitTranscriptVisitor},
};

use super::ProofToVar;

/// Plonk IOP verifier challenges using [`Variable`]s.
#[derive(Debug)]
#[allow(dead_code)]
pub struct ChallengesVar {
    pub(crate) tau: Variable,
    pub(crate) alphas: [Variable; 3],
    pub(crate) beta: Variable,
    pub(crate) gamma: Variable,
    pub(crate) zeta: Variable,
    pub(crate) v: Variable,
    pub(crate) u: Variable,
}

impl ChallengesVar {
    /// Create a new [`ChallengesVar`].
    pub fn new(
        tau: Variable,
        alphas: [Variable; 3],
        beta: Variable,
        gamma: Variable,
        zeta: Variable,
        v: Variable,
        u: Variable,
    ) -> Self {
        Self {
            tau,
            alphas,
            beta,
            gamma,
            zeta,
            v,
            u,
        }
    }

    /// Creates a new [`ChallengesVar`] from a reference to a [`PCSChallenges`].
    pub fn from_struct<F: PrimeField>(
        circuit: &mut PlonkCircuit<F>,
        challenges: &PCSChallenges<F>,
    ) -> Result<Self, CircuitError> {
        let tau = circuit.create_variable(challenges.tau)?;
        let alpha = circuit.create_variable(challenges.alpha)?;
        let alpha_2 = circuit.mul(alpha, alpha)?;
        let alpha_3 = circuit.mul(alpha_2, alpha)?;
        let beta = circuit.create_variable(challenges.beta)?;
        let gamma = circuit.create_variable(challenges.gamma)?;
        let zeta = circuit.create_variable(challenges.zeta)?;
        let v = circuit.create_variable(challenges.v)?;
        let u = circuit.create_variable(challenges.u)?;

        Ok(Self::new(
            tau,
            [alpha, alpha_2, alpha_3],
            beta,
            gamma,
            zeta,
            v,
            u,
        ))
    }

    /// Computes challenges from a proof.
    pub fn compute_challenges<PCS, P, F, C>(
        circuit: &mut PlonkCircuit<F>,
        vk_id: Option<Variable>,
        pi: &Variable,
        proof: &ProofVarNative<P>,
        transcript: &mut C,
    ) -> Result<Self, CircuitError>
    where
        PCS: Accumulation<Commitment = Affine<P>, Evaluation = P::ScalarField>,
        PCS::Proof: ProofToVar<P>,
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField + EmulationConfig<F> + RescueParameter,
        F: PrimeField + EmulationConfig<P::BaseField> + RescueParameter,
        C: CircuitTranscript<F>,
    {
        if let Some(id) = vk_id {
            transcript.push_variable(&id)?;
        }
        transcript.push_variable(pi)?;

        transcript.append_point_variables(&proof.wire_commitments, circuit)?;

        let tau = transcript.squeeze_scalar_challenge::<P>(circuit)?;

        if let Some(proof_lkup) = proof.plookup_proof.as_ref() {
            transcript.append_point_variables(&proof_lkup.h_poly_comms, circuit)?;
        }

        let [beta, gamma]: [Variable; 2] = transcript
            .squeeze_scalar_challenges::<P>(2, circuit)?
            .try_into()
            .map_err(|_| {
                CircuitError::ParameterError("Couldn't convert to fixed length array".to_string())
            })?;

        transcript.append_point_variable(&proof.prod_perm_poly_comm, circuit)?;

        if let Some(proof_lkup) = proof.plookup_proof.as_ref() {
            transcript.append_point_variable(&proof_lkup.prod_lookup_poly_comm, circuit)?;
        }

        let alpha = transcript.squeeze_scalar_challenge::<P>(circuit)?;
        let alpha_sq = circuit.mul(alpha, alpha)?;
        let alpha_cube = circuit.mul(alpha_sq, alpha)?;
        let alphas = [alpha, alpha_sq, alpha_cube];
        transcript.append_point_variables(&proof.split_quot_poly_comms, circuit)?;
        let zeta = transcript.squeeze_scalar_challenge::<P>(circuit)?;

        transcript.append_visitor(&proof.poly_evals, circuit)?;

        if let Some(proof_lkup) = proof.plookup_proof.as_ref() {
            transcript.append_visitor(&proof_lkup.poly_evals, circuit)?;
        }

        let v = transcript.squeeze_scalar_challenge::<P>(circuit)?;

        transcript.append_point_variable(&proof.q_comm, circuit)?;

        let u = transcript.squeeze_scalar_challenge::<P>(circuit)?;
        Ok(Self::new(tau, alphas, beta, gamma, zeta, v, u))
    }
}

/// Struct used to represent [`MLEChallenges`] as [`EmulatedVariable`]s.
#[allow(dead_code)]
pub struct EmulatedMLEChallenges<E: PrimeField> {
    pub(crate) gamma: EmulatedVariable<E>,
    pub(crate) alpha: EmulatedVariable<E>,
    pub(crate) tau: EmulatedVariable<E>,
    pub(crate) beta: EmulatedVariable<E>,
    pub(crate) delta: EmulatedVariable<E>,
    pub(crate) epsilon: EmulatedVariable<E>,
}

impl<E: PrimeField> EmulatedMLEChallenges<E> {
    /// Create a new [`EmulatedMLEChallenges`].
    pub fn new(
        gamma: EmulatedVariable<E>,
        alpha: EmulatedVariable<E>,
        tau: EmulatedVariable<E>,
        beta: EmulatedVariable<E>,
        delta: EmulatedVariable<E>,
        epsilon: EmulatedVariable<E>,
    ) -> Self {
        Self {
            gamma,
            alpha,
            tau,
            beta,
            delta,
            epsilon,
        }
    }

    /// Create a new [`EmulatedMLEChallenges`] variable from a reference to a [`MLEChallenges`].
    pub fn from_struct<P>(
        circuit: &mut PlonkCircuit<P::BaseField>,
        challenges: &MLEChallenges<P::ScalarField>,
    ) -> Result<Self, CircuitError>
    where
        P: HasTEForm<ScalarField = E>,
        P::BaseField: PrimeField + EmulationConfig<P::ScalarField> + RescueParameter,
        P::ScalarField: PrimeField + EmulationConfig<P::BaseField> + RescueParameter,
    {
        let gamma = circuit.create_emulated_variable(challenges.gamma)?;
        let alpha = circuit.create_emulated_variable(challenges.alpha)?;
        let tau = circuit.create_emulated_variable(challenges.tau)?;
        let beta = circuit.create_emulated_variable(challenges.beta)?;
        let delta = circuit.create_emulated_variable(challenges.delta)?;
        let epsilon = circuit.create_emulated_variable(challenges.epsilon)?;

        Ok(Self::new(gamma, alpha, tau, beta, delta, epsilon))
    }

    /// Computes challenges from a proof.
    pub fn compute_challenges_vars<PCS, P>(
        circuit: &mut PlonkCircuit<P::BaseField>,
        vk_var: &MLEVerifyingKeyVar<PCS>,
        pi: &EmulatedVariable<P::ScalarField>,
        proof_var: &SAMLEProofVar<PCS>,
        transcript_var: &mut RescueTranscriptVar<P::BaseField>,
    ) -> Result<EmulatedMLEChallenges<E>, CircuitError>
    where
        PCS: PolynomialCommitmentScheme<Commitment = Affine<P>, Evaluation = P::ScalarField>,
        P: HasTEForm<ScalarField = E>,
        P::BaseField: PrimeField + EmulationConfig<P::ScalarField> + RescueParameter,
        P::ScalarField: PrimeField + EmulationConfig<P::BaseField> + RescueParameter,
    {
        transcript_var.append_visitor(vk_var, circuit)?;
        transcript_var.push_emulated_variable(pi, circuit)?;
        transcript_var.append_point_variables(&proof_var.wire_commitments_var, circuit)?;

        let [gamma, alpha, tau]: [usize; 3] = transcript_var
            .squeeze_scalar_challenges::<P>(3, circuit)?
            .try_into()
            .map_err(|_| {
                CircuitError::ParameterError("Could not convert to fixed length array".to_string())
            })?;

        let gamma_var = circuit.to_emulated_variable(gamma)?;

        let alpha_var = circuit.to_emulated_variable(alpha)?;

        let tau_var = circuit.to_emulated_variable(tau)?;

        if let Some(lookup_proof_var) = proof_var.lookup_proof_var.as_ref() {
            transcript_var.append_point_variable(&lookup_proof_var.m_poly_comm_var, circuit)?;
        }

        let [beta, delta, epsilon]: [usize; 3] = transcript_var
            .squeeze_scalar_challenges::<P>(3, circuit)?
            .try_into()
            .map_err(|_| {
                CircuitError::ParameterError("Could not convert to fixed length array".to_string())
            })?;
        let beta_var = circuit.to_emulated_variable(beta)?;
        let delta_var = circuit.to_emulated_variable(delta)?;
        let epsilon_var = circuit.to_emulated_variable(epsilon)?;
        Ok(Self::new(
            gamma_var,
            alpha_var,
            tau_var,
            beta_var,
            delta_var,
            epsilon_var,
        ))
    }
}

/// Struct use to represent [`MLEChallenges`] as [`Variable`]s over the native field.
#[derive(Debug, Clone, Copy)]
pub struct MLEChallengesVar {
    pub(crate) gamma: Variable,
    pub(crate) alpha: Variable,
    pub(crate) tau: Variable,
    pub(crate) beta: Variable,
    pub(crate) delta: Variable,
    pub(crate) epsilon: Variable,
}

impl MLEChallengesVar {
    /// Create a new [`MLEChallengesVar`].
    pub fn new(
        gamma: Variable,
        alpha: Variable,
        tau: Variable,
        beta: Variable,
        delta: Variable,
        epsilon: Variable,
    ) -> Self {
        Self {
            gamma,
            alpha,
            tau,
            beta,
            delta,
            epsilon,
        }
    }

    /// Create a new [`MLEChallengesNative`] variable from a reference to a [`MLEChallenges`].
    pub fn from_struct<F>(
        circuit: &mut PlonkCircuit<F>,
        challenges: &MLEChallenges<F>,
    ) -> Result<Self, CircuitError>
    where
        F: PrimeField,
    {
        let gamma = circuit.create_variable(challenges.gamma)?;
        let alpha = circuit.create_variable(challenges.alpha)?;
        let tau = circuit.create_variable(challenges.tau)?;
        let beta = circuit.create_variable(challenges.beta)?;
        let delta = circuit.create_variable(challenges.delta)?;
        let epsilon = circuit.create_variable(challenges.epsilon)?;

        Ok(Self::new(gamma, alpha, tau, beta, delta, epsilon))
    }

    /// Computes challenges from a proof.
    pub fn compute_challenges<PCS, P, F, C>(
        circuit: &mut PlonkCircuit<F>,
        vk_var: &MLEVerifyingKeyVar<PCS>,
        pi_hash: &EmulatedVariable<P::ScalarField>,
        proof_var: &SAMLEProofVar<PCS>,
        transcript_var: &mut C,
    ) -> Result<Self, CircuitError>
    where
        PCS: PolynomialCommitmentScheme<Commitment = Affine<P>, Evaluation = P::ScalarField>,
        P: HasTEForm,
        P::BaseField: PrimeField + RescueParameter,
        P::ScalarField: PrimeField + RescueParameter + EmulationConfig<F>,
        F: PrimeField,
        C: CircuitTranscript<F>,
    {
        transcript_var.append_visitor(vk_var, circuit)?;
        transcript_var.push_emulated_variable(pi_hash, circuit)?;
        transcript_var.append_point_variables(&proof_var.wire_commitments_var, circuit)?;

        let [gamma, alpha, tau]: [usize; 3] = transcript_var
            .squeeze_scalar_challenges::<P>(3, circuit)?
            .try_into()
            .map_err(|_| {
                CircuitError::ParameterError("Could not convert to fixed length array".to_string())
            })?;

        if let Some(lookup_proof_var) = proof_var.lookup_proof_var.as_ref() {
            transcript_var.append_point_variable(&lookup_proof_var.m_poly_comm_var, circuit)?;
        }

        let [beta, delta, epsilon]: [usize; 3] = transcript_var
            .squeeze_scalar_challenges::<P>(3, circuit)?
            .try_into()
            .map_err(|_| {
                CircuitError::ParameterError("Could not convert to fixed length array".to_string())
            })?;

        Ok(Self::new(gamma, alpha, tau, beta, delta, epsilon))
    }

    /// Exposes the challenges as public inputs to the circuit.
    pub fn set_public<F>(&self, circuit: &mut PlonkCircuit<F>) -> Result<(), CircuitError>
    where
        F: PrimeField,
    {
        circuit.set_variable_public(self.gamma)?;
        circuit.set_variable_public(self.alpha)?;
        circuit.set_variable_public(self.tau)?;
        circuit.set_variable_public(self.beta)?;
        circuit.set_variable_public(self.delta)?;
        circuit.set_variable_public(self.epsilon)
    }

    /// Converts the challenges to field elements.
    pub fn to_field<F>(
        &self,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<MLEChallenges<F>, CircuitError>
    where
        F: PrimeField,
    {
        Ok(MLEChallenges {
            gamma: circuit.witness(self.gamma)?,
            alpha: circuit.witness(self.alpha)?,
            tau: circuit.witness(self.tau)?,
            beta: circuit.witness(self.beta)?,
            delta: circuit.witness(self.delta)?,
            epsilon: circuit.witness(self.epsilon)?,
        })
    }
}

/// A struct that stores the scalars used in the partial verification of a plonk proof.
#[derive(Debug, Clone)]
pub struct PVScalarsVar {
    /// Scalar corresponding with the generator point.
    pub g_scalar: Variable,
    /// Scalars corresponding with the wire polynomials.
    pub wire_scalars: Vec<Variable>,
    /// Scalars corresponding with the permutation polynomials.
    pub sigma_scalars: Vec<Variable>,
    /// Scalar corresponding to the permutation product polynomial.
    pub prod_perm_scalar: Variable,
    /// Scalars corresponding with the quotient polynomials.
    pub quot_scalars: Vec<Variable>,
    /// Optional lookup scalars.
    pub lookup_scalars: Option<PVLookupScalarsVar>,
    /// Scalar for the q_commitment.
    pub q_commitment_scalar: Variable,
    /// Scalars corresponding to selectors.
    pub selector_scalars: Vec<Variable>,
}

/// A struct that stores the scalars relating to lookups used in the partial verification of a plonk proof.
#[derive(Clone, Debug, Copy)]
pub struct PVLookupScalarsVar {
    /// Scalar for q_dom_sep.
    pub q_dom_sep_scalar: Variable,
    /// Scalars corresponding to h_2_commitment.
    pub h_2_scalar: Variable,
    /// Scalar corresponding to the lookup product commitment.
    pub prod_lookup_scalar: Variable,
    /// Scalars corresponding with the range table polynomial.
    pub range_table_scalar: Variable,
    /// Scalars corresponding with the key table polynomial.
    pub key_table_scalar: Variable,
    /// Scalars corresponding with the first sorted vector polynomial.
    pub h_1_scalar: Variable,
    /// Scalars corresponding with the domain separation selector polynomial.
    pub q_lookup_scalar: Variable,
    /// Scalars corresponding with the table domain separation polynomial.
    pub table_dom_sep_scalar: Variable,
}

/// Convert a [`PVScalarsVar`] to a [`Vec`] of [`Variable`]s.
impl From<PVScalarsVar> for Vec<Variable> {
    fn from(scalars: PVScalarsVar) -> Self {
        // List in circuit is g_scalars, wires, prod_perm, split_quotient, lookup h polys, lookup prod poly, q_comm, sigmas, selectors, plookup_vk,
        let mut scalars_vec = vec![scalars.g_scalar];
        scalars_vec.extend_from_slice(&scalars.wire_scalars);

        scalars_vec.push(scalars.prod_perm_scalar);
        scalars_vec.extend_from_slice(&scalars.quot_scalars);
        if let Some(lookup_scalars) = scalars.lookup_scalars {
            scalars_vec.push(lookup_scalars.h_1_scalar);
            scalars_vec.push(lookup_scalars.h_2_scalar);
            scalars_vec.push(lookup_scalars.prod_lookup_scalar);
        }
        scalars_vec.push(scalars.q_commitment_scalar);
        scalars_vec.extend_from_slice(&scalars.sigma_scalars);

        scalars_vec.extend_from_slice(&scalars.selector_scalars);
        if let Some(lookup_scalars) = &scalars.lookup_scalars {
            scalars_vec.push(lookup_scalars.q_lookup_scalar);
            scalars_vec.push(lookup_scalars.range_table_scalar);
            scalars_vec.push(lookup_scalars.key_table_scalar);
            scalars_vec.push(lookup_scalars.table_dom_sep_scalar);
            scalars_vec.push(lookup_scalars.q_dom_sep_scalar);
        }
        scalars_vec
    }
}

/// A variable used for storing an Ipa opening proof.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct IpaProofVar<E>
where
    E: Pairing,
    <E as Pairing>::ScalarField: EmulationConfig<E::BaseField> + PrimeField,
    <E as Pairing>::BaseField: PrimeField + RescueParameter,
{
    pub(crate) l_i: Vec<PointVariable>,
    pub(crate) r_i: Vec<PointVariable>,
    pub(crate) f: EmulatedVariable<E::ScalarField>,
    pub(crate) c: EmulatedVariable<E::ScalarField>,
}

/// Struct used so that we may pass a [`PCSProof`] into a circuit that is defined over the scalar field of the commitment curve
/// (so the commitments have to be emulated variables).
///
/// For the time being this can only be used with KZG based proofs.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ProofVarNative<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + EmulationConfig<P::ScalarField>,
{
    /// The wire commitments.
    pub(crate) wire_commitments: Vec<EmulatedPointVariable<P::BaseField>>,
    /// The polynomial commitment for the wire permutation argument.
    pub(crate) prod_perm_poly_comm: EmulatedPointVariable<P::BaseField>,

    /// Splitted quotient polynomial commitments.
    pub(crate) split_quot_poly_comms: Vec<EmulatedPointVariable<P::BaseField>>,

    /// Optional PlookupProof.
    pub(crate) plookup_proof: Option<PlookupProofVarNative<P>>,

    /// (Aggregated) proof of evaluations at challenge point `v`.
    pub(crate) opening_proof: EmulatedPointVariable<P::BaseField>,

    /// The commitment to the polynomial q(x) used in the multi-opening protocol.
    pub(crate) q_comm: EmulatedPointVariable<P::BaseField>,

    /// Polynomial evaluations.
    pub(crate) poly_evals: ProofEvalsVarNative,
}

impl<P> ProofVarNative<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + EmulationConfig<P::ScalarField>,
{
    /// Creates a new instance of [`ProofVarNative`].
    pub fn new(
        wire_commitments: Vec<EmulatedPointVariable<P::BaseField>>,
        prod_perm_poly_comm: EmulatedPointVariable<P::BaseField>,
        split_quot_poly_comms: Vec<EmulatedPointVariable<P::BaseField>>,
        plookup_proof: Option<PlookupProofVarNative<P>>,
        opening_proof: EmulatedPointVariable<P::BaseField>,
        q_comm: EmulatedPointVariable<P::BaseField>,
        poly_evals: ProofEvalsVarNative,
    ) -> Self {
        Self {
            wire_commitments,
            prod_perm_poly_comm,
            split_quot_poly_comms,
            plookup_proof,
            opening_proof,
            q_comm,
            poly_evals,
        }
    }

    /// Creates a new [`ProofVarNative`] variable from a reference to a [`PCSProof`].
    pub fn from_struct<E, F>(
        proof: &PCSProof<UnivariateKzgPCS<E>>,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<Self, CircuitError>
    where
        E: Pairing<G1Affine = Affine<P>, BaseField = P::BaseField, ScalarField = F>,
        F: PrimeField + RescueParameter,
        P: HasTEForm<ScalarField = F>,
    {
        let wire_commitments = proof
            .wires_poly_comms
            .iter()
            .map(|comm| {
                let point =
                    circuit.create_emulated_point_variable(&Point::<P::BaseField>::from(*comm))?;

                circuit.enforce_emulated_on_curve::<P, P::BaseField>(&point)?;

                let is_neutral = circuit.is_emulated_neutral_point::<P, P::BaseField>(&point)?;
                circuit.enforce_false(is_neutral.into())?;
                Ok(point)
            })
            .collect::<Result<Vec<EmulatedPointVariable<P::BaseField>>, CircuitError>>()?;

        let prod_perm_poly_comm = circuit.create_emulated_point_variable(
            &Point::<P::BaseField>::from(proof.prod_perm_poly_comm),
        )?;
        circuit.enforce_emulated_on_curve::<P, P::BaseField>(&prod_perm_poly_comm)?;

        let is_neutral =
            circuit.is_emulated_neutral_point::<P, P::BaseField>(&prod_perm_poly_comm)?;
        circuit.enforce_false(is_neutral.into())?;

        let split_quot_poly_comms = proof
            .split_quot_poly_comms
            .iter()
            .map(|comm| {
                let point =
                    circuit.create_emulated_point_variable(&Point::<P::BaseField>::from(*comm))?;

                circuit.enforce_emulated_on_curve::<P, P::BaseField>(&point)?;

                let is_neutral = circuit.is_emulated_neutral_point::<P, P::BaseField>(&point)?;
                circuit.enforce_false(is_neutral.into())?;
                Ok(point)
            })
            .collect::<Result<Vec<EmulatedPointVariable<P::BaseField>>, CircuitError>>()?;

        let plookup_proof = if let Some(plookup_proof) = &proof.plookup_proof {
            Some(PlookupProofVarNative::from_struct(circuit, plookup_proof)?)
        } else {
            None
        };

        let opening_proof = circuit.create_emulated_point_variable(
            &Point::<P::BaseField>::from(proof.opening_proof.proof),
        )?;

        circuit.enforce_emulated_on_curve::<P, P::BaseField>(&opening_proof)?;

        let is_neutral = circuit.is_emulated_neutral_point::<P, P::BaseField>(&opening_proof)?;
        circuit.enforce_false(is_neutral.into())?;

        let q_comm =
            circuit.create_emulated_point_variable(&Point::<P::BaseField>::from(proof.q_comm))?;

        circuit.enforce_emulated_on_curve::<P, P::BaseField>(&q_comm)?;

        let is_neutral = circuit.is_emulated_neutral_point::<P, P::BaseField>(&q_comm)?;
        circuit.enforce_false(is_neutral.into())?;

        let poly_evals = ProofEvalsVarNative::from_struct(circuit, &proof.poly_evals)?;
        Ok(Self::new(
            wire_commitments,
            prod_perm_poly_comm,
            split_quot_poly_comms,
            plookup_proof,
            opening_proof,
            q_comm,
            poly_evals,
        ))
    }

    /// Converts the `EmulatedPointVariable`s into a vector of `Variable`s that can be absorbed into a transcript.
    pub fn convert_to_vec_for_transcript<E, F>(
        &self,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<Vec<Variable>, CircuitError>
    where
        E: Pairing<G1Affine = Affine<P>, BaseField = P::BaseField, ScalarField = F>,
        F: PrimeField + RescueParameter,
        P: HasTEForm<ScalarField = F>,
    {
        let mut vars = vec![];
        for wire_comm in &self.wire_commitments {
            vars.extend_from_slice(&circuit.convert_for_transcript(&wire_comm.get_x())?);
            vars.extend_from_slice(&circuit.convert_for_transcript(&wire_comm.get_y())?);
        }

        vars.extend_from_slice(&circuit.convert_for_transcript(&self.prod_perm_poly_comm.get_x())?);
        vars.extend_from_slice(&circuit.convert_for_transcript(&self.prod_perm_poly_comm.get_y())?);

        for quot_poly_comm in &self.split_quot_poly_comms {
            vars.extend_from_slice(&circuit.convert_for_transcript(&quot_poly_comm.get_x())?);
            vars.extend_from_slice(&circuit.convert_for_transcript(&quot_poly_comm.get_y())?);
        }

        vars.extend_from_slice(&circuit.convert_for_transcript(&self.opening_proof.get_x())?);
        vars.extend_from_slice(&circuit.convert_for_transcript(&self.opening_proof.get_y())?);

        vars.extend_from_slice(&circuit.convert_for_transcript(&self.q_comm.get_x())?);
        vars.extend_from_slice(&circuit.convert_for_transcript(&self.q_comm.get_y())?);

        if let Some(plookup_proof) = &self.plookup_proof {
            for h_poly_comm in &plookup_proof.h_poly_comms {
                vars.extend_from_slice(&circuit.convert_for_transcript(&h_poly_comm.get_x())?);
                vars.extend_from_slice(&circuit.convert_for_transcript(&h_poly_comm.get_y())?);
            }

            vars.extend_from_slice(
                &circuit.convert_for_transcript(&plookup_proof.prod_lookup_poly_comm.get_x())?,
            );
            vars.extend_from_slice(
                &circuit.convert_for_transcript(&plookup_proof.prod_lookup_poly_comm.get_y())?,
            );
        }
        Ok(vars)
    }
}

/// A Plookup argument proof to be passed into a circuit defined over the scalar field of the commitment curve.
#[derive(Debug, Clone)]
pub struct PlookupProofVarNative<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + EmulationConfig<P::ScalarField>,
{
    /// The commitments for the polynomials that interpolate the sorted
    /// concatenation of the lookup table and the witnesses in the lookup gates.
    pub(crate) h_poly_comms: Vec<EmulatedPointVariable<P::BaseField>>,

    /// The product accumulation polynomial commitment for the Plookup argument
    pub(crate) prod_lookup_poly_comm: EmulatedPointVariable<P::BaseField>,

    /// Polynomial evaluations.
    pub(crate) poly_evals: PlookupEvalsVarNative,
}

impl<P> PlookupProofVarNative<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + EmulationConfig<P::ScalarField>,
{
    /// Create a new instance of [`PlookupProofVarNative`].
    pub fn new(
        h_poly_comms: Vec<EmulatedPointVariable<P::BaseField>>,
        prod_lookup_poly_comm: EmulatedPointVariable<P::BaseField>,
        poly_evals: PlookupEvalsVarNative,
    ) -> Self {
        Self {
            h_poly_comms,
            prod_lookup_poly_comm,
            poly_evals,
        }
    }

    /// Create a new [`PlookupProofVarNative`] variable from a reference to a [`PlookupProof`].
    pub fn from_struct<E, F>(
        circuit: &mut PlonkCircuit<F>,
        proof: &PlookupProof<UnivariateKzgPCS<E>>,
    ) -> Result<Self, CircuitError>
    where
        E: Pairing<G1Affine = Affine<P>, BaseField = P::BaseField, ScalarField = F>,
        F: PrimeField + RescueParameter,
        P: HasTEForm<ScalarField = F>,
    {
        let h_poly_comms = proof
            .h_poly_comms
            .iter()
            .map(|comm| {
                let point =
                    circuit.create_emulated_point_variable(&Point::<P::BaseField>::from(*comm))?;

                circuit.enforce_emulated_on_curve::<P, P::BaseField>(&point)?;

                let is_neutral = circuit.is_emulated_neutral_point::<P, P::BaseField>(&point)?;
                circuit.enforce_false(is_neutral.into())?;
                Ok(point)
            })
            .collect::<Result<Vec<EmulatedPointVariable<P::BaseField>>, CircuitError>>()?;

        let prod_lookup_poly_comm = circuit.create_emulated_point_variable(
            &Point::<P::BaseField>::from(proof.prod_lookup_poly_comm),
        )?;

        circuit.enforce_emulated_on_curve::<P, P::BaseField>(&prod_lookup_poly_comm)?;
        let is_neutral =
            circuit.is_emulated_neutral_point::<P, P::BaseField>(&prod_lookup_poly_comm)?;
        circuit.enforce_false(is_neutral.into())?;

        let poly_evals = PlookupEvalsVarNative::from_struct(circuit, &proof.poly_evals)?;
        Ok(Self::new(h_poly_comms, prod_lookup_poly_comm, poly_evals))
    }
}

/// Represent variables for a struct that stores the polynomial evaluations in a
/// Plonk proof.
#[derive(Debug, Clone)]
pub struct ProofEvalsVarNative {
    /// Wire witness polynomials evaluations at point `zeta`.
    pub(crate) wires_evals: Vec<Variable>,

    /// Extended permutation (sigma) polynomials evaluations at point `zeta`.
    /// We do not include the last sigma polynomial evaluation.
    pub(crate) wire_sigma_evals: Vec<Variable>,

    /// Permutation product polynomial evaluation at point `zeta * g`.
    pub(crate) perm_next_eval: Variable,
}

impl ProofEvalsVarNative {
    /// Create a new instance of [`ProofEvalsVarNative`].
    pub fn new(
        wires_evals: Vec<Variable>,
        wire_sigma_evals: Vec<Variable>,
        perm_next_eval: Variable,
    ) -> Self {
        Self {
            wires_evals,
            wire_sigma_evals,
            perm_next_eval,
        }
    }

    /// Create a new [`ProofEvalsVarNative`] variable from a reference to a [`ProofEvaluations`]
    pub fn from_struct<E>(
        circuit: &mut PlonkCircuit<E>,
        evals: &ProofEvaluations<E>,
    ) -> Result<Self, CircuitError>
    where
        E: PrimeField + RescueParameter,
    {
        let wires_evals = evals
            .wires_evals
            .iter()
            .map(|eval| circuit.create_variable(*eval))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        let wire_sigma_evals = evals
            .wire_sigma_evals
            .iter()
            .map(|eval| circuit.create_variable(*eval))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        let perm_next_eval = circuit.create_variable(evals.perm_next_eval)?;

        Ok(Self::new(wires_evals, wire_sigma_evals, perm_next_eval))
    }
}

/// Represent variables for a struct that stores the scalars in a
/// Plonk proof.
#[derive(Debug, Clone)]
pub struct ProofScalarsVarNative {
    pub(crate) evals: ProofEvalsVarNative,
    pub(crate) lookup_evals: Option<PlookupEvalsVarNative>,
    pub(crate) pi_hash: Variable,
}

impl ProofScalarsVarNative {
    /// Create a new instance of [`ProofScalarVarNative`].
    pub fn new(
        evals: ProofEvalsVarNative,
        lookup_evals: Option<PlookupEvalsVarNative>,
        pi_hash: Variable,
    ) -> Self {
        Self {
            evals,
            lookup_evals,
            pi_hash,
        }
    }

    /// Create a new [`ProofScalarVarNative`] variable from a reference to a [`ProofEvaluations`] and a pi_hash.
    pub fn from_struct(
        bn254_output: &Bn254Output,
        circuit: &mut PlonkCircuit<Fr254>,
    ) -> Result<Self, CircuitError> {
        let evals = ProofEvalsVarNative::from_struct(circuit, &bn254_output.proof.poly_evals)?;
        let lookup_evals = if let Some(plookup_proof) = &bn254_output.proof.plookup_proof {
            Some(PlookupEvalsVarNative::from_struct(
                circuit,
                &plookup_proof.poly_evals,
            )?)
        } else {
            None
        };
        let pi_hash = circuit.create_variable(bn254_output.pi_hash)?;

        Ok(Self::new(evals, lookup_evals, pi_hash))
    }
}

impl<C, F> CircuitTranscriptVisitor<C, F> for ProofEvalsVarNative
where
    C: CircuitTranscript<F>,
    F: PrimeField + RescueParameter,
{
    fn append_to_transcript(
        &self,
        transcript_var: &mut C,
        _circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError> {
        for eval in self.wires_evals.iter() {
            transcript_var.push_variable(eval)?;
        }

        for eval in self.wire_sigma_evals.iter() {
            transcript_var.push_variable(eval)?;
        }

        transcript_var.push_variable(&self.perm_next_eval)?;

        Ok(())
    }
}

/// A struct that stores the polynomial evaluations in a Plookup argument proof in variables.
#[derive(Debug, Clone)]
pub struct PlookupEvalsVarNative {
    /// Range table polynomial evaluation at point `zeta`.
    pub(crate) range_table_eval: Variable,

    /// Key table polynomial evaluation at point `zeta`.
    pub(crate) key_table_eval: Variable,

    /// Table domain separation polynomial evaluation at point `zeta`.
    pub(crate) table_dom_sep_eval: Variable,

    /// Domain separation selector polynomial evaluation at point `zeta`.
    pub(crate) q_dom_sep_eval: Variable,

    /// The first sorted vector polynomial evaluation at point `zeta`.
    pub(crate) h_1_eval: Variable,

    /// The lookup selector polynomial evaluation at point `zeta`.
    pub(crate) q_lookup_eval: Variable,

    /// Lookup product polynomial evaluation at point `zeta * g`.
    pub(crate) prod_next_eval: Variable,

    /// Range table polynomial evaluation at point `zeta * g`.
    pub(crate) range_table_next_eval: Variable,

    /// Key table polynomial evaluation at point `zeta * g`.
    pub(crate) key_table_next_eval: Variable,

    /// Table domain separation polynomial evaluation at point `zeta * g`.
    pub(crate) table_dom_sep_next_eval: Variable,

    /// The first sorted vector polynomial evaluation at point `zeta * g`.
    pub(crate) h_1_next_eval: Variable,

    /// The second sorted vector polynomial evaluation at point `zeta * g`.
    pub(crate) h_2_next_eval: Variable,

    /// The lookup selector polynomial evaluation at point `zeta * g`.
    pub(crate) q_lookup_next_eval: Variable,

    /// The 4th witness polynomial evaluation at point `zeta * g`.
    pub(crate) w_3_next_eval: Variable,

    /// The 5th witness polynomial evaluation at point `zeta * g`.
    pub(crate) w_4_next_eval: Variable,
}

impl PlookupEvalsVarNative {
    /// Create a new instance of [`PlookupEvalsVarNative`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        range_table_eval: Variable,
        key_table_eval: Variable,
        table_dom_sep_eval: Variable,
        q_dom_sep_eval: Variable,
        h_1_eval: Variable,
        q_lookup_eval: Variable,
        prod_next_eval: Variable,
        range_table_next_eval: Variable,
        key_table_next_eval: Variable,
        table_dom_sep_next_eval: Variable,
        h_1_next_eval: Variable,
        h_2_next_eval: Variable,
        q_lookup_next_eval: Variable,
        w_3_next_eval: Variable,
        w_4_next_eval: Variable,
    ) -> Self {
        Self {
            range_table_eval,
            key_table_eval,
            table_dom_sep_eval,
            q_dom_sep_eval,
            h_1_eval,
            q_lookup_eval,
            prod_next_eval,
            range_table_next_eval,
            key_table_next_eval,
            table_dom_sep_next_eval,
            h_1_next_eval,
            h_2_next_eval,
            q_lookup_next_eval,
            w_3_next_eval,
            w_4_next_eval,
        }
    }

    /// Create a new [`PlookupEvalsVarNative`] variable from a reference to a [`PlookupEvaluations`]
    pub fn from_struct<E>(
        circuit: &mut PlonkCircuit<E>,
        evals: &PlookupEvaluations<E>,
    ) -> Result<Self, CircuitError>
    where
        E: PrimeField + RescueParameter,
    {
        let range_table_eval = circuit.create_variable(evals.range_table_eval)?;
        let key_table_eval = circuit.create_variable(evals.key_table_eval)?;
        let table_dom_sep_eval = circuit.create_variable(evals.table_dom_sep_eval)?;
        let q_dom_sep_eval = circuit.create_variable(evals.q_dom_sep_eval)?;
        let h_1_eval = circuit.create_variable(evals.h_1_eval)?;
        let q_lookup_eval = circuit.create_variable(evals.q_lookup_eval)?;
        let prod_next_eval = circuit.create_variable(evals.prod_next_eval)?;
        let range_table_next_eval = circuit.create_variable(evals.range_table_next_eval)?;
        let key_table_next_eval = circuit.create_variable(evals.key_table_next_eval)?;
        let table_dom_sep_next_eval = circuit.create_variable(evals.table_dom_sep_next_eval)?;
        let h_1_next_eval = circuit.create_variable(evals.h_1_next_eval)?;
        let h_2_next_eval = circuit.create_variable(evals.h_2_next_eval)?;
        let q_lookup_next_eval = circuit.create_variable(evals.q_lookup_next_eval)?;
        let w_3_next_eval = circuit.create_variable(evals.w_3_next_eval)?;
        let w_4_next_eval = circuit.create_variable(evals.w_4_next_eval)?;

        Ok(Self::new(
            range_table_eval,
            key_table_eval,
            table_dom_sep_eval,
            q_dom_sep_eval,
            h_1_eval,
            q_lookup_eval,
            prod_next_eval,
            range_table_next_eval,
            key_table_next_eval,
            table_dom_sep_next_eval,
            h_1_next_eval,
            h_2_next_eval,
            q_lookup_next_eval,
            w_3_next_eval,
            w_4_next_eval,
        ))
    }
}

impl<F, C> CircuitTranscriptVisitor<C, F> for PlookupEvalsVarNative
where
    F: PrimeField + RescueParameter,
    C: CircuitTranscript<F>,
{
    fn append_to_transcript(
        &self,
        transcript: &mut C,
        _circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError> {
        transcript.push_variable(&self.key_table_eval)?;
        transcript.push_variable(&self.table_dom_sep_eval)?;
        transcript.push_variable(&self.range_table_eval)?;
        transcript.push_variable(&self.q_dom_sep_eval)?;
        transcript.push_variable(&self.h_1_eval)?;
        transcript.push_variable(&self.q_lookup_eval)?;
        transcript.push_variable(&self.prod_next_eval)?;
        transcript.push_variable(&self.range_table_next_eval)?;
        transcript.push_variable(&self.key_table_next_eval)?;
        transcript.push_variable(&self.table_dom_sep_next_eval)?;
        transcript.push_variable(&self.h_1_next_eval)?;
        transcript.push_variable(&self.h_2_next_eval)?;
        transcript.push_variable(&self.q_lookup_next_eval)?;
        transcript.push_variable(&self.w_3_next_eval)?;
        transcript.push_variable(&self.w_4_next_eval)?;
        Ok(())
    }
}

/// A struct containing the bases associated with a Plookup argument proof to be passed
/// into a circuit defined over the base field of the commitment curve.
#[derive(Debug, Clone)]
pub struct PlookupProofScalarsAndBasesVar<PCS>
where
    PCS: PolynomialCommitmentScheme,
{
    /// The commitments for the polynomials that interpolate the sorted
    /// concatenation of the lookup table and the witnesses in the lookup gates.
    pub(crate) h_poly_comms: Vec<PointVariable>,

    /// The product accumulation polynomial commitment for the Plookup argument
    pub(crate) prod_lookup_poly_comm: PointVariable,

    /// Polynomial evaluations that we store in the clear.
    pub(crate) poly_evals: PlookupEvaluations<PCS::Evaluation>,
}

impl<PCS: PolynomialCommitmentScheme> PlookupProofScalarsAndBasesVar<PCS> {
    /// Convert to vector of point variables.
    pub fn to_vec(&self) -> Vec<PointVariable> {
        let mut bases = self.h_poly_comms.clone();
        bases.push(self.prod_lookup_poly_comm);
        bases
    }
}

/// A struct used to represent Ipa proving/verifying params in a circuit.
#[derive(Debug, Clone)]
pub struct UnivariateUniversalIpaParamsVar<F: PrimeField> {
    /// 'degree' different generators of the prime order group G.
    pub g_bases: Vec<PointVariable>,
    /// Separate generator used for blinding.
    pub h: PointVariable,
    /// The final generator used to create the random point U.
    pub u: PointVariable,
    _phantom: PhantomData<F>,
}

impl<F: PrimeField> UnivariateUniversalIpaParamsVar<F> {
    /// Creates a new UnivariateUniversalIpaParamsVar variable from a reference.
    pub fn new<E>(
        params: &UnivariateUniversalIpaParams<E>,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<Self, CircuitError>
    where
        E: Pairing<BaseField = F>,
        <E::G1Affine as AffineRepr>::Config: HasTEForm<BaseField = F>,
        E::G1Affine: AffineRepr<BaseField = F, ScalarField = E::ScalarField>,
    {
        let g_bases = params
            .g_bases
            .iter()
            .map(|base| {
                let point = Point::<F>::from(*base);
                circuit.create_point_variable(&point)
            })
            .collect::<Result<Vec<PointVariable>, CircuitError>>()?;
        let h = {
            let point = Point::<F>::from(params.h);
            circuit.create_point_variable(&point)?
        };
        let u = {
            let point = Point::<F>::from(params.u);
            circuit.create_point_variable(&point)?
        };

        Ok(Self {
            g_bases,
            h,
            u,
            _phantom: PhantomData::<F>,
        })
    }
}

impl<T, F> CircuitTranscriptVisitor<T, F> for UnivariateUniversalIpaParamsVar<F>
where
    T: CircuitTranscript<F>,
    F: PrimeField,
{
    fn append_to_transcript(
        &self,
        transcript: &mut T,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError> {
        transcript.append_point_variables(&self.g_bases, circuit)?;
        transcript.append_point_variable(&self.h, circuit)?;
        transcript.append_point_variable(&self.u, circuit)?;
        Ok(())
    }
}

/// Struct used to represent an MLELookupProof in a circuit.
/// The commitments we are using will always be points on an SW curve.
pub struct MLELookupProofVar<PCS: PolynomialCommitmentScheme> {
    /// Representing commitments to the multiplicity polynomial.
    pub m_poly_comm_var: PointVariable,
    /// Evaluations of the polynomials.
    pub poly_evals_var: MLELookupEvaluationsVar<PCS::Evaluation>,
}

#[allow(dead_code)]
impl<PCS: PolynomialCommitmentScheme> MLELookupProofVar<PCS> {
    /// Create a new [`MLELookupProofVar`] variable.
    pub fn new(
        m_poly_comm_var: PointVariable,
        poly_evals_var: MLELookupEvaluationsVar<PCS::Evaluation>,
    ) -> Self {
        Self {
            m_poly_comm_var,
            poly_evals_var,
        }
    }

    /// Create a new [`MLELookupProofVar`] variable from a reference to a [`MLELookupProof`].
    pub fn from_struct<P>(
        circuit: &mut PlonkCircuit<P::BaseField>,
        proof: &MLELookupProof<PCS>,
    ) -> Result<Self, CircuitError>
    where
        PCS: PolynomialCommitmentScheme<Commitment = Affine<P>, Evaluation = P::ScalarField>,
        P: HasTEForm,
        P::BaseField: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
        P::ScalarField: PrimeField + RescueParameter + EmulationConfig<P::BaseField>,
    {
        let m_poly_comm_point = Point::<P::BaseField>::from(proof.m_poly_comm);
        let m_poly_comm_var = circuit.create_point_variable(&m_poly_comm_point)?;

        let poly_evals_var =
            MLELookupEvaluationsVar::<PCS::Evaluation>::from_struct(circuit, &proof.lookup_evals)?;

        Ok(Self::new(m_poly_comm_var, poly_evals_var))
    }

    /// Getter for the poly evals
    pub fn poly_evals(&self) -> &MLELookupEvaluationsVar<PCS::Evaluation> {
        &self.poly_evals_var
    }
}

/// Struct use to represent a [`MLELookupProof`] in a native circuit (so no [`PointVariable`]s).
pub struct MLELookupProofNative {
    /// The polynomial evaluations,
    pub poly_evals: MLELookupEvaluationsNativeVar,
}

impl MLELookupProofNative {
    /// Create a new instance of [`MLELookupProofNative`].
    pub fn new(poly_evals: MLELookupEvaluationsNativeVar) -> Self {
        Self { poly_evals }
    }

    /// Create a new [`MLELookupProofNative`] variable from a reference to a [`MLELookupProof`]
    pub fn from_struct<PCS>(
        circuit: &mut PlonkCircuit<PCS::Evaluation>,
        proof: &MLELookupProof<PCS>,
    ) -> Result<Self, CircuitError>
    where
        PCS: Accumulation,
        PCS::Evaluation: PrimeField + RescueParameter,
    {
        let poly_evals = MLELookupEvaluationsNativeVar::from_struct(circuit, &proof.lookup_evals)?;

        Ok(Self::new(poly_evals))
    }

    /// Getter for the polynomial evaluations.
    pub fn poly_evals(&self) -> &MLELookupEvaluationsNativeVar {
        &self.poly_evals
    }
}

/// Struct used to represent the polynomial evaluations in an MLELookupProof in a circuit.
pub struct MLELookupEvaluationsVar<F: PrimeField> {
    /// Range table eval.
    pub range_table_eval: EmulatedVariable<F>,
    /// Key table eval.
    pub key_table_eval: EmulatedVariable<F>,
    /// Table domain separation eval.
    pub table_dom_sep_eval: EmulatedVariable<F>,
    /// Lookup domain separation selector eval.
    pub q_dom_sep_eval: EmulatedVariable<F>,
    /// Lookup selector eval.
    pub q_lookup_eval: EmulatedVariable<F>,
    /// The claimed evaluations of the multiplicity polynomial at the challenge point.
    pub m_poly_eval: EmulatedVariable<F>,
}

impl<F: PrimeField> MLELookupEvaluationsVar<F> {
    /// Create a new instance of [`MLELookupEvaluationsVar`].
    pub fn new(
        range_table_eval: EmulatedVariable<F>,
        key_table_eval: EmulatedVariable<F>,
        table_dom_sep_eval: EmulatedVariable<F>,
        q_dom_sep_eval: EmulatedVariable<F>,
        q_lookup_eval: EmulatedVariable<F>,
        m_poly_eval: EmulatedVariable<F>,
    ) -> Self {
        Self {
            range_table_eval,
            key_table_eval,
            table_dom_sep_eval,
            q_dom_sep_eval,
            q_lookup_eval,
            m_poly_eval,
        }
    }

    /// Create a new [`MLELookupEvaluationsVar`] variable from a reference to a [`MLELookupEvaluations`]
    pub fn from_struct<PCS, P>(
        circuit: &mut PlonkCircuit<P::BaseField>,
        evals: &MLELookupEvals<PCS>,
    ) -> Result<Self, CircuitError>
    where
        PCS: PolynomialCommitmentScheme<Commitment = Affine<P>, Evaluation = P::ScalarField>,
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
        F: PrimeField + RescueParameter + EmulationConfig<P::BaseField>,
    {
        let range_table_eval = circuit.create_emulated_variable(evals.range_table_eval)?;
        let key_table_eval = circuit.create_emulated_variable(evals.key_table_eval)?;
        let table_dom_sep_eval = circuit.create_emulated_variable(evals.table_dom_sep_eval)?;
        let q_dom_sep_eval = circuit.create_emulated_variable(evals.q_dom_sep_eval)?;
        let q_lookup_eval = circuit.create_emulated_variable(evals.q_lookup_eval)?;
        let m_poly_eval = circuit.create_emulated_variable(evals.m_poly_eval)?;

        Ok(Self::new(
            range_table_eval,
            key_table_eval,
            table_dom_sep_eval,
            q_dom_sep_eval,
            q_lookup_eval,
            m_poly_eval,
        ))
    }
}

/// Struct used to represent an MLEProof in a circuit.
/// The commitments we are using will always be points on an SW curve.
#[allow(dead_code)]
pub struct SAMLEProofVar<PCS: PolynomialCommitmentScheme>
where
    PCS: PolynomialCommitmentScheme,
    <PCS::Commitment as AffineRepr>::Config: HasTEForm,
    <PCS::Commitment as AffineRepr>::BaseField: PrimeField,
{
    /// Representing commitments to the witness wires.
    pub wire_commitments_var: Vec<PointVariable>,
    /// The GKR proof for the permutation argument and lookup argument if present.
    pub gkr_proof: EmulatedGKRProofVar<PCS::Evaluation>,
    /// The final SumCheck proof for the gate equation and for checking the output evals of the GKR proof.
    pub sumcheck_proof: EmulatedSumCheckProofVar<PCS::Evaluation>,
    /// Representing optional lookup proof.
    pub lookup_proof_var: Option<MLELookupProofVar<PCS>>,
    /// Evaluations of the polynomials.
    pub poly_evals_var: MLEProofEvaluationsVar<PCS::Evaluation>,
    /// The point used in the polynomial opening.
    pub opening_point_var: Vec<EmulatedVariable<PCS::Evaluation>>,
}

#[allow(dead_code)]
impl<PCS> SAMLEProofVar<PCS>
where
    PCS: PolynomialCommitmentScheme,
    <PCS::Commitment as AffineRepr>::Config: HasTEForm,
    <PCS::Commitment as AffineRepr>::BaseField: PrimeField,
{
    /// Create a new [`MLEProofVar`] variable from a reference to a [`MLEProof`].
    pub fn from_struct<P>(
        circuit: &mut PlonkCircuit<P::BaseField>,
        proof: &SAMLEProof<PCS>,
    ) -> Result<Self, CircuitError>
    where
        PCS: Accumulation<
            Commitment = Affine<P>,
            Evaluation = P::ScalarField,
            Point = Vec<P::ScalarField>,
        >,
        P: HasTEForm,
        P::BaseField: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
        P::ScalarField: PrimeField + RescueParameter + EmulationConfig<P::BaseField>,
    {
        let mut wire_commitments_var = Vec::<PointVariable>::new();
        for wire_commitment in proof.wire_commitments.iter() {
            let comm_point = Point::<P::BaseField>::from(*wire_commitment);
            let comm_var = circuit.create_point_variable(&comm_point)?;
            wire_commitments_var.push(comm_var);
        }

        let lookup_proof_var = match &proof.lookup_proof {
            Some(lookup_proof) => Some(MLELookupProofVar::from_struct::<P>(circuit, lookup_proof)?),
            None => None,
        };
        let poly_evals_var =
            MLEProofEvaluationsVar::<P::ScalarField>::from_struct::<PCS, P>(circuit, &proof.evals)?;

        let gkr_proof = EmulatedGKRProofVar::from_proof::<P>(circuit, &proof.gkr_proof)?;
        let sumcheck_proof = circuit.proof_to_emulated_var::<P>(&proof.sumcheck_proof)?;
        let opening_point_var = proof
            .opening_point
            .iter()
            .map(|point| circuit.create_emulated_variable(*point))
            .collect::<Result<Vec<EmulatedVariable<P::ScalarField>>, CircuitError>>()?;

        Ok(Self {
            wire_commitments_var,
            gkr_proof,
            sumcheck_proof,
            lookup_proof_var,
            poly_evals_var,
            opening_point_var,
        })
    }
}

/// Used to represent a [`SAMLEProof`] over a native circuit (so no [`PointVariable`]s).
pub struct SAMLEProofNative {
    /// The native GKR proof.
    pub gkr_proof: GKRProofVar,
    /// The native SumCheck proof.
    pub sumcheck_proof: SumCheckProofVar,
    /// The polynomial evaluations.
    pub poly_evals: MLEProofEvalsNativeVar,
    /// The native lookup proof (if lookup is used).
    pub lookup_proof: Option<MLELookupProofNative>,
}

impl SAMLEProofNative {
    /// Create a new instance of [`SAMLEProofNative`].
    pub fn new(
        gkr_proof: GKRProofVar,
        sumcheck_proof: SumCheckProofVar,
        poly_evals: MLEProofEvalsNativeVar,
        lookup_proof: Option<MLELookupProofNative>,
    ) -> Self {
        Self {
            gkr_proof,
            sumcheck_proof,
            poly_evals,
            lookup_proof,
        }
    }

    /// Create a new [`SAMLEProofNative`] variable from a reference to a [`SAMLEProof`]
    pub fn from_struct<PCS>(
        circuit: &mut PlonkCircuit<PCS::Evaluation>,
        proof: &SAMLEProof<PCS>,
    ) -> Result<Self, CircuitError>
    where
        PCS: Accumulation,
        PCS::Evaluation: PrimeField + RescueParameter,
    {
        let gkr_proof = GKRProofVar::from_struct(&proof.gkr_proof, circuit)?;
        let sumcheck_proof = circuit.sum_check_proof_to_var(&proof.sumcheck_proof)?;
        let poly_evals = MLEProofEvalsNativeVar::from_struct(circuit, &proof.evals)?;
        let lookup_proof = match &proof.lookup_proof {
            Some(lookup_proof) => Some(MLELookupProofNative::from_struct(circuit, lookup_proof)?),
            None => None,
        };
        Ok(Self::new(
            gkr_proof,
            sumcheck_proof,
            poly_evals,
            lookup_proof,
        ))
    }

    /// Getter for the GKR proof.
    pub fn gkr_proof(&self) -> &GKRProofVar {
        &self.gkr_proof
    }
    /// Getter for the SumCheck proof.
    pub fn sumcheck_proof(&self) -> &SumCheckProofVar {
        &self.sumcheck_proof
    }
    /// Getter for the polynomial evaluations.
    pub fn poly_evals(&self) -> &MLEProofEvalsNativeVar {
        &self.poly_evals
    }
    /// Getter for the lookup proof.
    pub fn lookup_proof(&self) -> Option<&MLELookupProofNative> {
        self.lookup_proof.as_ref()
    }
}

/// Struct used to represent the polynomial evaluations in an MLEProof in a circuit.
pub struct MLEProofEvaluationsVar<F: PrimeField> {
    /// The claimed evaluations of the witness wires.
    pub wire_evals: Vec<EmulatedVariable<F>>,
    /// The claimed evaluations of the selector polynomials.
    pub selector_evals: Vec<EmulatedVariable<F>>,
    /// The claimed evaluations of the permutation polynomials.
    pub permutation_evals: Vec<EmulatedVariable<F>>,
}

impl<F: PrimeField> MLEProofEvaluationsVar<F> {
    /// Creates a new instance of [`MLEProofEvaluationsVar`].
    pub fn new(
        wire_evals: Vec<EmulatedVariable<F>>,
        selector_evals: Vec<EmulatedVariable<F>>,
        permutation_evals: Vec<EmulatedVariable<F>>,
    ) -> Self {
        Self {
            wire_evals,
            selector_evals,
            permutation_evals,
        }
    }

    /// Create a new [`MLEProofEvaluationsVar`] variable from a reference to a [`MLEProofEvaluations`]
    pub fn from_struct<PCS, P>(
        circuit: &mut PlonkCircuit<P::BaseField>,
        evals: &MLEProofEvals<PCS>,
    ) -> Result<Self, CircuitError>
    where
        PCS: PolynomialCommitmentScheme<Commitment = Affine<P>, Evaluation = P::ScalarField>,
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
        F: PrimeField + RescueParameter + EmulationConfig<P::BaseField>,
    {
        let wire_evals = evals
            .wire_evals
            .iter()
            .map(|eval| circuit.create_emulated_variable(*eval))
            .collect::<Result<Vec<EmulatedVariable<F>>, CircuitError>>()?;
        let selector_evals = evals
            .selector_evals
            .iter()
            .map(|eval| circuit.create_emulated_variable(*eval))
            .collect::<Result<Vec<EmulatedVariable<F>>, CircuitError>>()?;
        let permutation_evals = evals
            .permutation_evals
            .iter()
            .map(|eval| circuit.create_emulated_variable(*eval))
            .collect::<Result<Vec<EmulatedVariable<F>>, CircuitError>>()?;

        Ok(Self::new(wire_evals, selector_evals, permutation_evals))
    }
}

/// Struct used to represent the polynomial evaluations in an MLELookupProof in a circuit over the correct field.
pub struct MLEProofEvalsNativeVar {
    /// The claimed evaluations of the witness wires.
    pub wire_evals: Vec<Variable>,
    /// The claimed evaluations of the selector polynomials.
    pub selector_evals: Vec<Variable>,
    /// The claimed evaluations of the permutation polynomials.
    pub permutation_evals: Vec<Variable>,
}

impl MLEProofEvalsNativeVar {
    /// Creates a new instance of [`MLEProofEvalsNativeVar`].
    pub fn new(
        wire_evals: Vec<Variable>,
        selector_evals: Vec<Variable>,
        permutation_evals: Vec<Variable>,
    ) -> Self {
        Self {
            wire_evals,
            selector_evals,
            permutation_evals,
        }
    }

    /// Creates a new [`MLEProofEvalsNativeVar`] variable from a reference to a [`MLEProofEvals`]
    pub fn from_struct<PCS>(
        circuit: &mut PlonkCircuit<PCS::Evaluation>,
        evals: &MLEProofEvals<PCS>,
    ) -> Result<Self, CircuitError>
    where
        PCS: Accumulation,
        PCS::Evaluation: PrimeField + RescueParameter,
    {
        let wire_evals = evals
            .wire_evals
            .iter()
            .map(|eval| circuit.create_variable(*eval))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        let selector_evals = evals
            .selector_evals
            .iter()
            .map(|eval| circuit.create_variable(*eval))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        let permutation_evals = evals
            .permutation_evals
            .iter()
            .map(|eval| circuit.create_variable(*eval))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;

        Ok(Self::new(wire_evals, selector_evals, permutation_evals))
    }
}

/// Struct used to represent the polynomial evaluations in an MLELookupProof in a circuit over the correct field.
pub struct MLELookupEvaluationsNativeVar {
    /// Range table eval.
    pub range_table_eval: Variable,
    /// Key table eval.
    pub key_table_eval: Variable,
    /// Table domain separation eval.
    pub table_dom_sep_eval: Variable,
    /// Lookup domain separation selector eval.
    pub q_dom_sep_eval: Variable,
    /// Lookup selector eval.
    pub q_lookup_eval: Variable,
    /// The claimed evaluations of the multiplicity polynomial at the challenge point.
    pub m_poly_eval: Variable,
}

impl MLELookupEvaluationsNativeVar {
    /// Creates a new instance of [`MLELookupEvaluationsNativeVar`].
    pub fn new(
        range_table_eval: Variable,
        key_table_eval: Variable,
        table_dom_sep_eval: Variable,
        q_dom_sep_eval: Variable,
        q_lookup_eval: Variable,
        m_poly_eval: Variable,
    ) -> Self {
        Self {
            range_table_eval,
            key_table_eval,
            table_dom_sep_eval,
            q_dom_sep_eval,
            q_lookup_eval,
            m_poly_eval,
        }
    }

    /// Create a new [`MLELookupEvaluationsNativeVar`] variable from a reference to a [`MLELookupEvals`]
    pub fn from_struct<PCS>(
        circuit: &mut PlonkCircuit<PCS::Evaluation>,
        evals: &MLELookupEvals<PCS>,
    ) -> Result<Self, CircuitError>
    where
        PCS: PolynomialCommitmentScheme,
    {
        let range_table_eval = circuit.create_variable(evals.range_table_eval)?;
        let key_table_eval = circuit.create_variable(evals.key_table_eval)?;
        let table_dom_sep_eval = circuit.create_variable(evals.table_dom_sep_eval)?;
        let q_dom_sep_eval = circuit.create_variable(evals.q_dom_sep_eval)?;
        let q_lookup_eval = circuit.create_variable(evals.q_lookup_eval)?;
        let m_poly_eval = circuit.create_variable(evals.m_poly_eval)?;

        Ok(Self::new(
            range_table_eval,
            key_table_eval,
            table_dom_sep_eval,
            q_dom_sep_eval,
            q_lookup_eval,
            m_poly_eval,
        ))
    }
}

/// Struct used to represent the verifying key for an MLE Plonk protocol.
/// The commitments we are using will always be points on an SW curve.
#[allow(dead_code)]
pub struct MLEVerifyingKeyVar<PCS>
where
    PCS: PolynomialCommitmentScheme,
{
    /// Representing commitments to the selectors.
    pub selector_commitments_var: Vec<PointVariable>,
    /// Representing commitments to the permutations.
    pub permutation_commitments_var: Vec<PointVariable>,
    /// The hash of the verifying key.
    pub hash: EmulatedVariable<PCS::Evaluation>,
}

#[allow(dead_code)]
impl<PCS: PolynomialCommitmentScheme> MLEVerifyingKeyVar<PCS> {
    pub(crate) fn new<F, P>(
        circuit: &mut PlonkCircuit<F>,
        vk: &MLEVerifyingKey<PCS>,
    ) -> Result<Self, CircuitError>
    where
        PCS: PolynomialCommitmentScheme<Commitment = Affine<P>>,
        P: HasTEForm<BaseField = F>,
        F: PrimeField,
        PCS::Evaluation: EmulationConfig<F>,
    {
        let mut selector_commitments_var = Vec::<PointVariable>::new();
        for selector_commitment in vk.selector_commitments.iter() {
            let comm_point = Point::<F>::from(*selector_commitment);
            let comm_var = circuit.create_point_variable(&comm_point)?;
            selector_commitments_var.push(comm_var);
        }
        let mut permutation_commitments_var = Vec::<PointVariable>::new();
        for permutation_commitment in vk.permutation_commitments.iter() {
            let comm_point = Point::<F>::from(*permutation_commitment);
            let comm_var = circuit.create_point_variable(&comm_point)?;
            permutation_commitments_var.push(comm_var);
        }

        let hash = circuit.create_emulated_variable(vk.hash())?;

        Ok(Self {
            selector_commitments_var,
            permutation_commitments_var,
            hash,
        })
    }
}

impl<T, F, PCS> CircuitTranscriptVisitor<T, F> for MLEVerifyingKeyVar<PCS>
where
    T: CircuitTranscript<F>,
    F: PrimeField,
    PCS: PolynomialCommitmentScheme,
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
