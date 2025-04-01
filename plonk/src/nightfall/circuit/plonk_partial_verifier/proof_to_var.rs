//! Code for converting PCS::Proof to variable form.
use ark_ec::{pairing::Pairing, short_weierstrass::Affine};
use ark_ff::PrimeField;
use ark_std::{fmt::Debug, vec::Vec};
use jf_primitives::{pcs::prelude::UnivariateKzgProof, rescue::RescueParameter};
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{HasTEForm, Point, PointVariable},
        EmulatedVariable, EmulationConfig,
    },
    PlonkCircuit,
};

use crate::nightfall::hops::univariate_ipa::UnivariateIpaProof;

/// Trait that tells a circuit how to convert a specific PCS proof into a circuit variable.
pub trait ProofToVar<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
    P::ScalarField: PrimeField + RescueParameter + EmulationConfig<P::BaseField>,
{
    /// The variable form of the proof.
    type ProofVar: Debug + Clone;
    /// Create a new variable from a reference to a proof.
    fn create_variables(
        &self,
        circuit: &mut PlonkCircuit<P::BaseField>,
    ) -> Result<Self::ProofVar, CircuitError>;
}

/// Struct representing an univariate IPA proof in variable form.
#[derive(Debug, Clone)]
pub struct UnivariateIpaProofVar<F: PrimeField> {
    /// The left side of the proof.
    pub l_i: Vec<PointVariable>,
    /// The right side of the proof.
    pub r_i: Vec<PointVariable>,
    /// The synthetic blinding factor
    pub f: EmulatedVariable<F>,
    /// The collapsed coefficient vector of the committed polynomial.
    pub c: EmulatedVariable<F>,
}

impl<E, P> ProofToVar<P> for UnivariateIpaProof<E>
where
    E: Pairing<BaseField = P::BaseField, ScalarField = P::ScalarField, G1Affine = Affine<P>>,
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
    P::ScalarField: PrimeField + RescueParameter + EmulationConfig<P::BaseField>,
{
    type ProofVar = UnivariateIpaProofVar<E::ScalarField>;

    fn create_variables(
        &self,
        circuit: &mut PlonkCircuit<P::BaseField>,
    ) -> Result<Self::ProofVar, CircuitError> {
        let l_i = self
            .l_i
            .iter()
            .map(|p| circuit.create_point_variable(&Point::<P::BaseField>::from(*p)))
            .collect::<Result<Vec<PointVariable>, CircuitError>>()?;
        let r_i = self
            .r_i
            .iter()
            .map(|p| circuit.create_point_variable(&Point::<P::BaseField>::from(*p)))
            .collect::<Result<Vec<PointVariable>, CircuitError>>()?;
        let f = circuit.create_emulated_variable(self.f)?;
        let c = circuit.create_emulated_variable(self.c)?;
        Ok(UnivariateIpaProofVar { l_i, r_i, f, c })
    }
}

impl<E, P> ProofToVar<P> for UnivariateKzgProof<E>
where
    E: Pairing<BaseField = P::BaseField, ScalarField = P::ScalarField, G1Affine = Affine<P>>,
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
    P::ScalarField: PrimeField + RescueParameter + EmulationConfig<P::BaseField>,
{
    type ProofVar = PointVariable;

    fn create_variables(
        &self,
        circuit: &mut PlonkCircuit<P::BaseField>,
    ) -> Result<Self::ProofVar, CircuitError> {
        circuit.create_point_variable(&Point::<P::BaseField>::from(self.proof))
    }
}
