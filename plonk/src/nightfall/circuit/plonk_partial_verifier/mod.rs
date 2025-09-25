// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Circuits for Plonk verifiers.
use crate::{
    nightfall::ipa_structs::{PlookupVerifyingKey, VerificationKeyId, VerifyingKey, VK},
    recursion::circuits::Kzg,
    transcript::*,
};
use ark_bn254::{Fq as Fq254, Fr as Fr254};
use ark_ec::{short_weierstrass::Affine, AffineRepr};
use ark_ff::PrimeField;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, Radix2EvaluationDomain};
use ark_std::{marker::PhantomData, string::ToString, vec, vec::Vec};
use itertools::izip;
use jf_primitives::{
    pcs::{PolynomialCommitmentScheme, StructuredReferenceString},
    rescue::RescueParameter,
};
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{HasTEForm, Point, PointVariable},
        EmulationConfig,
    },
    Circuit, PlonkCircuit, Variable,
};

mod gadgets;
mod poly;
mod proof_to_var;
mod scalars_and_bases;
mod structs;

pub use gadgets::*;
pub use poly::*;
pub use proof_to_var::*;
pub use scalars_and_bases::*;
pub use structs::*;

/// Represent variable of a Plonk verifying key.
#[derive(Debug, Clone, Eq, PartialEq)]
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

    /// Used for client verification keys to identify distinguish between
    /// transfer/withdrawal and deposit.
    pub(crate) id: Option<Variable>,
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
        _circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError> {
        let id = if let Some(id) = self.id {
            Ok(id)
        } else {
            Err(CircuitError::ParameterError(
                "Verifying key has no id".to_string(),
            ))
        }?;
        transcript.push_variable(&id)
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

    /// Create a new constant [`PlookupVerifyingKeyVar`] from a [`PlookupVerifyingKey`].
    pub fn constant_from_struct<PCS, P>(
        vk: &PlookupVerifyingKey<PCS>,
        circuit: &mut PlonkCircuit<P::BaseField>,
    ) -> Result<Self, CircuitError>
    where
        PCS: PolynomialCommitmentScheme<Commitment = Affine<P>>,
        P: HasTEForm,
        P::BaseField: PrimeField + RescueParameter,
    {
        let range_table_comm =
            circuit.create_constant_point_variable(&Point::from(vk.range_table_comm))?;
        let key_table_comm =
            circuit.create_constant_point_variable(&Point::from(vk.key_table_comm))?;
        let table_dom_sep_comm =
            circuit.create_constant_point_variable(&Point::from(vk.table_dom_sep_comm))?;
        let q_dom_sep_comm =
            circuit.create_constant_point_variable(&Point::from(vk.q_dom_sep_comm))?;
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
        let id = if let Some(id) = verify_key.id() {
            Some(circuit.create_variable(F::from(id as u8))?)
        } else {
            None
        };

        Ok(Self {
            sigma_comms,
            selector_comms,
            plookup_vk,
            is_merged: verify_key.is_merged(),
            domain_size: verify_key.domain_size(),
            num_inputs: verify_key.num_inputs(),
            k: verify_key.k().to_vec(),
            id,
        })
    }

    /// Create a constant variable for a Plonk verifying key.
    pub fn new_constant<VerifyingKey>(
        circuit: &mut PlonkCircuit<F>,
        verify_key: &VerifyingKey,
    ) -> Result<Self, CircuitError>
    where
        VerifyingKey: VK<PCS>,
    {
        if verify_key.id().is_some() {
            return Err(CircuitError::ParameterError(
                "Constant VerifyingKeyVar should not have an ID".to_string(),
            ));
        }
        let sigma_comms = verify_key
            .sigma_comms()
            .iter()
            .map(|comm| circuit.create_constant_point_variable(&Point::from(*comm)))
            .collect::<Result<Vec<_>, CircuitError>>()?;
        let selector_comms = verify_key
            .selector_comms()
            .iter()
            .map(|comm| circuit.create_constant_point_variable(&Point::from(*comm)))
            .collect::<Result<Vec<_>, CircuitError>>()?;
        let plookup_vk = if let Some(plookup_vk) = verify_key.plookup_vk() {
            Some(PlookupVerifyingKeyVar::constant_from_struct(
                plookup_vk, circuit,
            )?)
        } else {
            None
        };

        Ok(Self {
            sigma_comms,
            selector_comms,
            plookup_vk,
            is_merged: verify_key.is_merged(),
            domain_size: verify_key.domain_size(),
            num_inputs: verify_key.num_inputs(),
            k: verify_key.k().to_vec(),
            id: None,
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
        if self.id.is_some() {
            res.push(self.id.unwrap());
        }
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

#[derive(Debug, Clone, Eq, PartialEq)]
/// Represents the scalars in a verification key variable.
/// Note, unlike `VerifyingKeyVar` and `BasesVerifyingKeyVar`, `ScalarsVerifyingKeyVar` is used in a circuit defined over PCS::Evaluation.
pub struct VerifyingKeyScalarsVar<PCS: PolynomialCommitmentScheme> {
    /// The log of the evaluation domain size.
    log_domain_size: Variable,

    /// The constants K0, ..., K_num_wire_types that ensure wire subsets are
    /// disjoint.
    k: Vec<Variable>,

    /// Used for client verification keys to identify distinguish between
    /// transfer/withdrawal and deposit.
    pub(crate) id: Variable,
    _phantom: PhantomData<PCS>,
}

impl<PCS, F> VerifyingKeyScalarsVar<PCS>
where
    PCS: PolynomialCommitmentScheme<Evaluation = F>,
    F: PrimeField,
{
    /// Create a variable representing a reduced Plonk verifying key.
    pub fn new<VerifyingKey>(
        circuit: &mut PlonkCircuit<F>,
        verify_key: &VerifyingKey,
    ) -> Result<Self, CircuitError>
    where
        VerifyingKey: VK<PCS>,
    {
        let id_var = if let Some(id) = verify_key.id() {
            circuit.create_variable(F::from(id as u8))
        } else {
            return Err(CircuitError::ParameterError(
                "ScalarsVerifyingKeyVar are only created from VerifyingKeys with an ID".to_string(),
            ));
        }?;

        let log_domain = verify_key.domain_size().ilog2();
        let log_domain_var = circuit.create_variable(F::from(log_domain))?;

        let k_var = verify_key
            .k()
            .iter()
            .map(|k| circuit.create_variable(*k))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;

        Ok(Self {
            log_domain_size: log_domain_var,
            k: k_var,
            id: id_var,
            _phantom: PhantomData,
        })
    }
}

/// A struct containing the scalars and bases associated with a verification key.
/// The bases are stored as `Variable`s to be passed into a circuit defined over the
/// base field of the commitment curve.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct VerifyingKeyScalarsAndBasesVar<PCS: PolynomialCommitmentScheme> {
    /// The size of the evaluation domain. Should be a power of two.
    /// This is stored in the clear.
    pub(crate) domain_size: usize,
    /// The number of public inputs. This is stored in the clear.
    pub(crate) num_inputs: usize,
    /// The variables for the permutation polynomial commitments.
    pub(crate) sigma_comms: Vec<PointVariable>,
    /// The variables for the selector polynomial commitments.
    pub(crate) selector_comms: Vec<PointVariable>,
    /// The constants K0, ..., K_num_wire_types that ensure wire subsets are
    /// disjoint. These are stored in the clear.
    pub(crate) k: Vec<PCS::Evaluation>,
    /// The base point of the KZG PCS opening key.
    pub(crate) g: PointVariable,
    /// A flag indicating whether the key is a merged key.
    /// This is stored in the clear.
    pub(crate) is_merged: bool,
    /// Plookup verifying key variable.
    pub(crate) plookup_vk: Option<PlookupVerifyingKeyVar>,
    /// Used for client verification keys to identify distinguish between
    /// transfer/withdrawal and deposit.
    pub(crate) id: Option<Variable>,
}

impl VerifyingKeyScalarsAndBasesVar<Kzg> {
    /// Create a variable for a Plonk verifying key.
    pub fn new(
        circuit: &mut PlonkCircuit<Fq254>,
        verify_key: &VerifyingKey<Kzg>,
    ) -> Result<Self, CircuitError> {
        let domain_size = verify_key.domain_size();
        let num_inputs = verify_key.num_inputs();
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
        let k = verify_key.k().to_vec();
        // We constrain g to be constant, as our KZG SRS is fixed across nightfall.
        let g = circuit.create_constant_point_variable(&Point::from(verify_key.open_key.g))?;
        let is_merged = verify_key.is_merged();
        let plookup_vk = if let Some(plookup_vk) = verify_key.plookup_vk() {
            Some(PlookupVerifyingKeyVar::from_struct(plookup_vk, circuit)?)
        } else {
            None
        };
        let id = if let Some(id) = verify_key.id() {
            Some(circuit.create_variable(Fq254::from(id as u8))?)
        } else {
            None
        };

        Ok(Self {
            domain_size,
            num_inputs,
            sigma_comms,
            selector_comms,
            k,
            g,
            is_merged,
            plookup_vk,
            id,
        })
    }

    /// Create a constant variable for a Plonk verifying key.
    pub fn new_constant(
        circuit: &mut PlonkCircuit<Fq254>,
        verify_key: &VerifyingKey<Kzg>,
    ) -> Result<Self, CircuitError> {
        let domain_size = verify_key.domain_size();
        let num_inputs = verify_key.num_inputs();
        let sigma_comms = verify_key
            .sigma_comms()
            .iter()
            .map(|comm| circuit.create_constant_point_variable(&Point::from(*comm)))
            .collect::<Result<Vec<_>, CircuitError>>()?;
        let selector_comms = verify_key
            .selector_comms()
            .iter()
            .map(|comm| circuit.create_constant_point_variable(&Point::from(*comm)))
            .collect::<Result<Vec<_>, CircuitError>>()?;
        let k = verify_key.k().to_vec();
        // We constrain g to be constant, as our KZG SRS is fixed across nightfall.
        let g = circuit.create_constant_point_variable(&Point::from(verify_key.open_key.g))?;
        let is_merged = verify_key.is_merged();
        let plookup_vk = if let Some(plookup_vk) = verify_key.plookup_vk() {
            Some(PlookupVerifyingKeyVar::constant_from_struct(
                plookup_vk, circuit,
            )?)
        } else {
            None
        };
        if verify_key.id().is_some() {
            return Err(CircuitError::ParameterError(
                "Constant VerifyingKeyBasesVar should not have an ID".to_string(),
            ));
        }

        Ok(Self {
            domain_size,
            num_inputs,
            sigma_comms,
            selector_comms,
            k,
            g,
            is_merged,
            plookup_vk,
            id: None,
        })
    }

    /// Constrain a `VerifyingKeyBasesVar` to agree with the bases of one of two `VerifyingKey`s.
    pub fn cond_select_equal_bases(
        &self,
        circuit: &mut PlonkCircuit<Fq254>,
        vks: &[VerifyingKey<Kzg>],
    ) -> Result<(), CircuitError> {
        if vks.len() != 2 {
            return Err(CircuitError::ParameterError(
                "Currently, cond_select_equal_bases only supports two verifying keys".to_string(),
            ));
        }
        let id_var = self.id.ok_or(CircuitError::ParameterError(
            "cond_select_equal_bases requires VerifyingKeyBasesVar to have an ID".to_string(),
        ))?;
        let id = circuit.witness(id_var)?;
        let client_ids = vks
            .iter()
            .map(|vk| {
                vk.id().ok_or(CircuitError::ParameterError(
                    "cond_select_equal_bases requires all verifying keys to have an ID".to_string(),
                ))
            })
            .collect::<Result<Vec<VerificationKeyId>, CircuitError>>()?;

        // We determine the index of the verifying key that matches the ID.
        let idx = if let Some(idx) = client_ids.iter().position(|&x| Fq254::from(x as u8) == id) {
            Ok(idx)
        } else {
            Err(CircuitError::ParameterError(
                "VerifyingKeyBasesVar ID does not match any of the provided verifying keys"
                    .to_string(),
            ))
        }?;
        // The remainder of the function assumes only two verifying keys
        // We will change this when we introduce more possible client keys
        let cond_sel_bool = circuit.create_boolean_variable(idx == 1)?;
        circuit.const_conditional_select_gate(
            cond_sel_bool,
            id_var,
            Fq254::from(client_ids[0] as u8),
            Fq254::from(client_ids[1] as u8),
        )?;

        if self.sigma_comms.len() != vks[0].sigma_comms().len()
            || self.sigma_comms.len() != vks[1].sigma_comms().len()
        {
            return Err(CircuitError::ParameterError(
                "VerifyingKeyBasesVar and VerifyingKeys have different number of sigma commitments"
                    .to_string(),
            ));
        }
        for (sigma_comm_var, sigma_comm_0, sigma_comm_1) in izip!(
            self.sigma_comms.iter(),
            vks[0].sigma_comms().iter(),
            vks[1].sigma_comms().iter()
        ) {
            circuit.const_conditional_select_gate(
                cond_sel_bool,
                sigma_comm_var.get_x(),
                sigma_comm_0.x,
                sigma_comm_1.x,
            )?;
            circuit.const_conditional_select_gate(
                cond_sel_bool,
                sigma_comm_var.get_y(),
                sigma_comm_0.y,
                sigma_comm_1.y,
            )?;
        }

        if self.selector_comms.len() != vks[0].selector_comms().len()
            || self.selector_comms.len() != vks[1].selector_comms().len()
        {
            return Err(CircuitError::ParameterError(
                "VerifyingKeyBasesVar and VerifyingKeys have different number of selector commitments".to_string(),
            ));
        }
        for (selector_comm_var, selector_comm_0, selector_comm_1) in izip!(
            self.selector_comms.iter(),
            vks[0].selector_comms().iter(),
            vks[1].selector_comms().iter()
        ) {
            circuit.const_conditional_select_gate(
                cond_sel_bool,
                selector_comm_var.get_x(),
                selector_comm_0.x,
                selector_comm_1.x,
            )?;
            circuit.const_conditional_select_gate(
                cond_sel_bool,
                selector_comm_var.get_y(),
                selector_comm_0.y,
                selector_comm_1.y,
            )?;
        }

        if let Some(plookup_vk_var) = &self.plookup_vk {
            let plookup_vk_0 = vks[0].plookup_vk().ok_or(CircuitError::ParameterError(
                "VerifyingKeyBasesVar has a PlookupVerifyingKeyVar but the first verifying key does not have a PlookupVerifyingKey".to_string(),
            ))?;
            let plookup_vk_1 = vks[1].plookup_vk().ok_or(CircuitError::ParameterError(
                "VerifyingKeyBasesVar has a PlookupVerifyingKeyVar but the second verifying key does not have a PlookupVerifyingKey".to_string(),
            ))?;

            circuit.const_conditional_select_gate(
                cond_sel_bool,
                plookup_vk_var.range_table_comm.get_x(),
                plookup_vk_0.range_table_comm.x,
                plookup_vk_1.range_table_comm.x,
            )?;
            circuit.const_conditional_select_gate(
                cond_sel_bool,
                plookup_vk_var.range_table_comm.get_y(),
                plookup_vk_0.range_table_comm.y,
                plookup_vk_1.range_table_comm.y,
            )?;

            circuit.const_conditional_select_gate(
                cond_sel_bool,
                plookup_vk_var.key_table_comm.get_x(),
                plookup_vk_0.key_table_comm.x,
                plookup_vk_1.key_table_comm.x,
            )?;
            circuit.const_conditional_select_gate(
                cond_sel_bool,
                plookup_vk_var.key_table_comm.get_y(),
                plookup_vk_0.key_table_comm.y,
                plookup_vk_1.key_table_comm.y,
            )?;

            circuit.const_conditional_select_gate(
                cond_sel_bool,
                plookup_vk_var.table_dom_sep_comm.get_x(),
                plookup_vk_0.table_dom_sep_comm.x,
                plookup_vk_1.table_dom_sep_comm.x,
            )?;
            circuit.const_conditional_select_gate(
                cond_sel_bool,
                plookup_vk_var.table_dom_sep_comm.get_y(),
                plookup_vk_0.table_dom_sep_comm.y,
                plookup_vk_1.table_dom_sep_comm.y,
            )?;

            circuit.const_conditional_select_gate(
                cond_sel_bool,
                plookup_vk_var.q_dom_sep_comm.get_x(),
                plookup_vk_0.q_dom_sep_comm.x,
                plookup_vk_1.q_dom_sep_comm.x,
            )?;
            circuit.const_conditional_select_gate(
                cond_sel_bool,
                plookup_vk_var.q_dom_sep_comm.get_y(),
                plookup_vk_0.q_dom_sep_comm.y,
                plookup_vk_1.q_dom_sep_comm.y,
            )?;
        } else if vks[0].plookup_vk().is_some() || vks[1].plookup_vk().is_some() {
            return Err(CircuitError::ParameterError(
                "VerifyingKeyBasesVar has no PlookupVerifyingKeyVar but one of the verifying keys has a PlookupVerifyingKey".to_string(),
            ));
        }

        Ok(())
    }

    /// The lookup selector polynomial commitment
    pub(crate) fn q_lookup_comm(&self) -> Result<&PointVariable, CircuitError> {
        if self.plookup_vk.is_none() {
            return Err(CircuitError::ParameterError(
                "This verifying key does not have a Plookup verifying key".to_string(),
            ));
        }
        Ok(self.selector_comms.last().unwrap())
    }
}

#[derive(Clone, Default)]
/// Stores the domain size and generator variables for a Plonk circuit.
pub(crate) struct DomainVar {
    /// The size of the evaluation domain. Should be a power of two.
    pub(crate) domain_size: Variable,
    /// The generator of the evaluation domain.
    pub(crate) gen: Variable,
}

#[derive(Clone, Default)]
/// A struct used to represent the native field scalars in a verifying key.
pub(crate) struct VerifyingKeyNativeScalarsVar {
    /// The size of the evaluation domain. Should be a power of two.
    pub(crate) domain: DomainVar,
    /// The constants K0, ..., K_num_wire_types that ensure wire subsets are
    /// disjoint.
    pub(crate) k: Vec<Variable>,
    /// Used for client verification keys to identify distinguish between
    /// transfer/withdrawal and deposit.
    pub(crate) id: Variable,
}

impl VerifyingKeyNativeScalarsVar {
    /// Create a variable representing a the scalars in a verifying key.
    pub fn new<PCS, VerifyingKey, F>(
        circuit: &mut PlonkCircuit<F>,
        verify_key: &VerifyingKey,
    ) -> Result<Self, CircuitError>
    where
        PCS: PolynomialCommitmentScheme<Evaluation = F>,
        VerifyingKey: VK<PCS>,
        F: PrimeField,
    {
        let id_var = if let Some(id) = verify_key.id() {
            circuit.create_variable(F::from(id as u8))
        } else {
            return Err(CircuitError::ParameterError(
                "NativeScalarsVerifyingKeyVar are only created from VerifyingKeys with an ID"
                    .to_string(),
            ));
        }?;

        let domain = Radix2EvaluationDomain::<F>::new(verify_key.domain_size()).ok_or(
            CircuitError::ParameterError("Could not create vk domain".to_string()),
        )?;
        let domain_size_var = circuit.create_variable(F::from(verify_key.domain_size() as u64))?;
        let gen = domain.group_gen;
        let gen_var = circuit.create_variable(gen)?;

        let k_var = verify_key
            .k()
            .iter()
            .map(|k| circuit.create_variable(*k))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;

        Ok(Self {
            domain: DomainVar {
                domain_size: domain_size_var,
                gen: gen_var,
            },
            k: k_var,
            id: id_var,
        })
    }

    /// Constrain a `VerifyingKeyNativeScalarsVar` to agree with the scalars of one of two `VerifyingKey`s.
    pub fn cond_select_equal_scalars(
        &self,
        circuit: &mut PlonkCircuit<Fr254>,
        vks: &[VerifyingKey<Kzg>],
    ) -> Result<(), CircuitError> {
        if vks.len() != 2 {
            return Err(CircuitError::ParameterError(
                "Currently, cond_select_equal_scalars only supports two verifying keys".to_string(),
            ));
        }
        let id = circuit.witness(self.id)?;
        let client_ids = vks
            .iter()
            .map(|vk| {
                vk.id().ok_or(CircuitError::ParameterError(
                    "cond_select_equal_scalars requires all verifying keys to have an ID"
                        .to_string(),
                ))
            })
            .collect::<Result<Vec<VerificationKeyId>, CircuitError>>()?;

        // We determine the index of the verifying key that matches the ID.
        let idx = if let Some(idx) = client_ids.iter().position(|&x| Fr254::from(x as u8) == id) {
            Ok(idx)
        } else {
            Err(CircuitError::ParameterError(
                "VerifyingKeyNativeScalarsVar ID does not match any of the provided verifying keys"
                    .to_string(),
            ))
        }?;
        // The remainder of the function assumes only two verifying keys
        // We will change this when we introduce more possible client keys
        let cond_sel_bool = circuit.create_boolean_variable(idx == 1)?;
        circuit.const_conditional_select_gate(
            cond_sel_bool,
            self.id,
            Fr254::from(client_ids[0] as u8),
            Fr254::from(client_ids[1] as u8),
        )?;

        let vk_gens = vks
            .iter()
            .map(|vk| {
                let domain = Radix2EvaluationDomain::<Fr254>::new(vk.domain_size()).ok_or(
                    CircuitError::ParameterError("Could not create vk domain".to_string()),
                )?;
                Ok(domain.group_gen)
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;

        circuit.const_conditional_select_gate(
            cond_sel_bool,
            self.domain.domain_size,
            Fr254::from(vks[0].domain_size() as u64),
            Fr254::from(vks[1].domain_size() as u64),
        )?;

        circuit.const_conditional_select_gate(
            cond_sel_bool,
            self.domain.gen,
            vk_gens[0],
            vk_gens[1],
        )?;

        if self.k.len() != vks[0].k().len() || self.k.len() != vks[1].k().len() {
            return Err(CircuitError::ParameterError(
                "VerifyingKeyNativeScalarsVar and VerifyingKeys have different number of k constants"
                    .to_string(),
            ));
        }
        for (k_var, k_0, k_1) in izip!(self.k.iter(), vks[0].k().iter(), vks[1].k().iter()) {
            circuit.const_conditional_select_gate(cond_sel_bool, *k_var, *k_0, *k_1)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        errors::PlonkError,
        nightfall::{ipa_structs::VerificationKeyId, ipa_verifier::FFTVerifier, FFTPlonk},
        proof_system::UniversalSNARK,
        transcript::RescueTranscript,
    };
    use ark_bn254::{g1::Config as BnConfig, Bn254, Fq as Fq254, Fr as Fr254};
    use ark_ec::{short_weierstrass::Projective, CurveGroup, VariableBaseMSM};
    use ark_std::{UniformRand, Zero};
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
        for vk_id in [
            None,
            Some(VerificationKeyId::Client),
            Some(VerificationKeyId::Deposit),
        ]
        .iter()
        {
            compute_scalars_native_helper::<UnivariateKzgPCS<Bn254>, Fq254, BnConfig, SWGrumpkin>(
                *vk_id,
            )?;
        }
        Ok(())
    }
    fn compute_scalars_native_helper<PCS, F, P, E>(
        vk_id: Option<VerificationKeyId>,
    ) -> Result<(), CircuitError>
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
            let pi = circuit.public_input().unwrap()[0];
            let max_degree = circuit.srs_size()?;
            let srs = FFTPlonk::<PCS>::universal_setup_for_testing(max_degree, rng).unwrap();

            // 3. Create proof
            let (pk, vk) = FFTPlonk::<PCS>::preprocess(&srs, vk_id, &circuit).unwrap();
            let proof = FFTPlonk::<PCS>::recursive_prove::<_, _, RescueTranscript<P::BaseField>>(
                rng, &circuit, &pk, None,
            )
            .unwrap();

            // 4. Verification

            let verifier = FFTVerifier::<PCS>::new(vk.domain_size).unwrap();

            let pcs_info = verifier
                .prepare_pcs_info::<RescueTranscript<P::BaseField>>(&vk, &[pi], &proof.proof, &None)
                .unwrap();

            // Compute commitment to g(x).
            let g_comm = pcs_info
                .comm_scalars_and_bases
                .multi_scalar_mul()
                .into_affine();

            let challenges = FFTVerifier::<PCS>::compute_challenges::<
                RescueTranscript<P::BaseField>,
            >(&vk, &[pi], &proof.proof, &None)?;

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

            let pi_var = circuit.create_variable(pi)?;

            let scalars = compute_scalars_for_native_field::<P::ScalarField>(
                &mut circuit,
                &pi_var,
                &challenges_var,
                &proof_evals,
                &Some(lookup_evals),
                &vk.k,
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

        let rng = &mut test_rng();
        //
        let zeta = circuit.create_variable(Fr254::rand(rng)).unwrap();

        let challenges_var = ChallengesVar {
            tau: circuit.zero(),
            alphas: [circuit.zero(), circuit.zero(), circuit.zero()],
            beta: circuit.zero(),
            gamma: circuit.zero(),
            zeta,
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

        let vk_k = vec![Fr254::zero(); 6];
        let scalars = compute_scalars_for_native_field::<Fr254>(
            &mut circuit,
            &0,
            &challenges_var,
            &proof_evals,
            &Some(lookup_evals),
            &vk_k,
            1 << 15,
        )
        .unwrap();

        for var in scalars.iter() {
            ark_std::println!("scalar value: {}", circuit.witness(*var).unwrap());
        }
    }
}
