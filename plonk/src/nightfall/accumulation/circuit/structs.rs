//! Circuit variable forms of the accumulation structs.

use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, SWCurveConfig},
    AffineRepr,
};

use ark_ff::{BigInteger, PrimeField};
use ark_std::{vec, vec::Vec};

use jf_primitives::{
    pcs::{prelude::UnivariateKzgPCS, Accumulation},
    rescue::RescueParameter,
};
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{HasTEForm, Point, PointVariable},
        EmulatedVariable, EmulationConfig,
    },
    Circuit, PlonkCircuit, Variable,
};

use crate::nightfall::{
    accumulation::{
        accumulation_structs::{AtomicAccProof, AtomicInstance, PCSInstance},
        AtomicAccumulationChallengesAndScalars,
    },
    circuit::subroutine_verifiers::structs::SumCheckProofVar,
};

/// Here the generic 'E' will be the pairing curve the previous proof used.
/// This struct is a circuit representation of the instance part of a Pedersen
/// polynomial commitment predicate as described in https://eprint.iacr.org/2020/1618.pdf.
#[derive(Debug)]
pub struct EmulatedPCSInstanceVar<E>
where
    E: HasTEForm,
    E::BaseField: PrimeField,
{
    /// PointVariable representing the commitment in 'q.x'
    pub comm: PointVariable,
    /// The ScalarField value (from the previous curve's ScalarField)
    /// that the polynomial stored in 'comm' evaluates to at 'point'.
    pub value: EmulatedVariable<E::ScalarField>,
    /// The point we evaluate the polynomial stored in 'comm' at.
    pub point: Vec<EmulatedVariable<E::ScalarField>>,
}

impl<E> Clone for EmulatedPCSInstanceVar<E>
where
    E: HasTEForm,
    E::BaseField: PrimeField,
{
    fn clone(&self) -> Self {
        Self {
            comm: self.comm,
            value: self.value.clone(),
            point: self.point.clone(),
        }
    }
}

/// This struct is used in circuit where commitments in a [`PCSInstance`] are defined over the base field
/// of a curve but the circuit is defined over the scalar field.
pub struct PCSInstanceVar {
    /// The value that the polynomial evaluates to at `point`.
    pub value: Variable,
    /// The point we evaluate the polynomial at.
    pub point: Vec<Variable>,
}

impl PCSInstanceVar {
    /// Create a new instance from the associated variables.
    pub fn new(value: Variable, point: Vec<Variable>) -> Self {
        Self { value, point }
    }

    /// Creates the appropriate variable from a [`PCSInstance`].
    pub fn from_struct<PCS, F>(
        instance: &PCSInstance<PCS>,
        circuit: &mut PlonkCircuit<PCS::Evaluation>,
    ) -> Result<Self, CircuitError>
    where
        PCS: Accumulation<Point = Vec<F>, Evaluation = F>,
        F: PrimeField,
    {
        let value = circuit.create_variable(instance.value)?;
        let point = instance
            .point
            .iter()
            .map(|point| circuit.create_variable(*point))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        Ok(PCSInstanceVar::new(value, point))
    }
}

impl<E> EmulatedPCSInstanceVar<E>
where
    E: HasTEForm,
    E::BaseField: PrimeField,
{
    /// Create a new instance from the associated variables.
    pub fn new(
        comm: PointVariable,
        value: EmulatedVariable<E::ScalarField>,
        point: Vec<EmulatedVariable<E::ScalarField>>,
    ) -> Self {
        Self { comm, value, point }
    }

    /// Creates the appropriate variable from a [`PCSInstance<PCS>`].
    pub fn from_instance<PCS>(
        instance: &PCSInstance<PCS>,
        circuit: &mut PlonkCircuit<E::BaseField>,
    ) -> Result<Self, CircuitError>
    where
        PCS: Accumulation<
            Commitment = Affine<E>,
            Evaluation = E::ScalarField,
            Point = Vec<E::ScalarField>,
        >,
        E::ScalarField: EmulationConfig<E::BaseField>,
    {
        let comm_point = Point::from(instance.comm);
        let comm_var = circuit.create_point_variable(&comm_point)?;
        let eval_var = circuit.create_emulated_variable::<E::ScalarField>(instance.value)?;
        let mut point_vars = vec![];
        for point in instance.point.iter() {
            let point_var = circuit.create_emulated_variable::<E::ScalarField>(*point)?;
            point_vars.push(point_var);
        }

        Ok(EmulatedPCSInstanceVar::<E>::new(
            comm_var, eval_var, point_vars,
        ))
    }

    /// Creates the appropriate public variable from a [`PCSInstance<PCS>`].
    pub fn from_instance_public<PCS>(
        instance: &PCSInstance<PCS>,
        circuit: &mut PlonkCircuit<E::BaseField>,
    ) -> Result<Self, CircuitError>
    where
        PCS: Accumulation<
            Commitment = Affine<E>,
            Evaluation = E::ScalarField,
            Point = Vec<E::ScalarField>,
        >,
        E::ScalarField: EmulationConfig<E::BaseField>,
    {
        let comm_point = Point::from(instance.comm);
        let comm_var = circuit.create_public_point_variable(&comm_point)?;
        let eval_var = circuit.create_public_emulated_variable::<E::ScalarField>(instance.value)?;
        let mut point_vars = vec![];
        for point in instance.point.iter() {
            let point_var = circuit.create_emulated_variable::<E::ScalarField>(*point)?;
            let point_native = circuit.mod_to_native_field(&point_var)?;
            circuit.set_variable_public(point_native)?;
            point_vars.push(point_var);
        }

        Ok(EmulatedPCSInstanceVar::<E>::new(
            comm_var, eval_var, point_vars,
        ))
    }

    /// Creates a new instance from a [`PCSInstance`] with the relevant field public inputs for recursion over a cycle of curves.
    pub fn from_instance_recursion<PCS>(
        instance: &PCSInstance<PCS>,
        circuit: &mut PlonkCircuit<E::BaseField>,
    ) -> Result<Self, CircuitError>
    where
        PCS: Accumulation<
            Commitment = Affine<E>,
            Evaluation = E::ScalarField,
            Point = Vec<E::ScalarField>,
        >,
        E::ScalarField: EmulationConfig<E::BaseField>,
    {
        let comm_point = Point::from(instance.comm);
        let comm_var = circuit.create_point_variable(&comm_point)?;
        let eval_var = circuit.create_public_emulated_variable::<E::ScalarField>(instance.value)?;
        let mut point_vars = vec![];
        for point in instance.point.iter() {
            let point_var = circuit.create_emulated_variable::<E::ScalarField>(*point)?;
            let point_native = circuit.mod_to_native_field(&point_var)?;
            circuit.set_variable_public(point_native)?;
            point_vars.push(point_var);
        }

        Ok(EmulatedPCSInstanceVar::<E>::new(
            comm_var, eval_var, point_vars,
        ))
    }
}

/// Variable form of an [`AtomicInstance`].
pub struct AtomicInstanceVar<E: SWCurveConfig> {
    /// PointVariable representing the commitment in 'q.x'
    pub comm: PointVariable,
    /// The ScalarField value (from the previous curve's ScalarField)
    /// that the polynomial stored in 'comm' evaluates to at 'point'.
    pub value: EmulatedVariable<E::ScalarField>,
    /// The point we evaluate the polynomial stored in 'comm' at.
    pub point: EmulatedVariable<E::ScalarField>,
    /// The opening proof that `comm` evaluates to `value` at `point`.
    pub opening_proof: PointVariable,
}

/// Variable version of [`AtomicAccProof`]. This can be used as either a proof of accumulation or
/// to represent an atomic accumulator.
pub struct AtomicAccProofVar {
    /// The part of the proof that is accumulated with the commitments.
    pub instance: PointVariable,
    /// The part of the proof that is accumulated with the witness.
    pub proof: PointVariable,
}

impl<E> AtomicInstanceVar<E>
where
    E: HasTEForm,
    E::BaseField: PrimeField + RescueParameter,
{
    /// Creates a new [`AtomicInstanceVar`] from its constituent variables.
    pub fn new(
        comm: PointVariable,
        value: EmulatedVariable<E::ScalarField>,
        point: EmulatedVariable<E::ScalarField>,
        opening_proof: PointVariable,
    ) -> Self {
        Self {
            comm,
            value,
            point,
            opening_proof,
        }
    }

    /// Creates a new variable from an [`AtomicInstance`].
    pub fn from_struct<P>(
        instance: &AtomicInstance<UnivariateKzgPCS<P>>,
        circuit: &mut PlonkCircuit<E::BaseField>,
    ) -> Result<Self, CircuitError>
    where
        P: Pairing<BaseField = E::BaseField, G1Affine = Affine<E>, ScalarField = E::ScalarField>,
        E::ScalarField: PrimeField + EmulationConfig<E::BaseField>,
    {
        let comm_point = Point::from(instance.comm);
        let comm_var = circuit.create_point_variable(&comm_point)?;
        let eval_var = circuit.create_emulated_variable::<E::ScalarField>(instance.value)?;
        let point_var = circuit.create_emulated_variable::<E::ScalarField>(instance.point)?;
        let proof_point = Point::from(instance.opening_proof.proof);
        let proof_var = circuit.create_point_variable(&proof_point)?;
        Ok(AtomicInstanceVar::new(
            comm_var, eval_var, point_var, proof_var,
        ))
    }

    /// Creates a new public variable from an [`AtomicInstance`].
    pub fn from_struct_public<P>(
        instance: &AtomicInstance<UnivariateKzgPCS<P>>,
        circuit: &mut PlonkCircuit<E::BaseField>,
    ) -> Result<Self, CircuitError>
    where
        P: Pairing<BaseField = E::BaseField, G1Affine = Affine<E>, ScalarField = E::ScalarField>,
        E::ScalarField: PrimeField + EmulationConfig<E::BaseField>,
    {
        let comm_point = Point::from(instance.comm);
        let comm_var = circuit.create_public_point_variable(&comm_point)?;
        let eval_var = circuit.create_public_emulated_variable::<E::ScalarField>(instance.value)?;
        let point_var =
            circuit.create_public_emulated_variable::<E::ScalarField>(instance.point)?;
        let proof_point = Point::from(instance.opening_proof.proof);
        let proof_var = circuit.create_public_point_variable(&proof_point)?;
        Ok(AtomicInstanceVar::new(
            comm_var, eval_var, point_var, proof_var,
        ))
    }

    /// Creates a new variable from an [`AtomicInstance`] with the relevant field public inputs for recursion over a cycle of curves.
    pub fn from_struct_recursion<P>(
        instance: &AtomicInstance<UnivariateKzgPCS<P>>,
        circuit: &mut PlonkCircuit<E::BaseField>,
    ) -> Result<Self, CircuitError>
    where
        P: Pairing<BaseField = E::BaseField, G1Affine = Affine<E>, ScalarField = E::ScalarField>,
        E::ScalarField: PrimeField + EmulationConfig<E::BaseField>,
    {
        let comm_point = Point::from(instance.comm);
        let comm_var = circuit.create_point_variable(&comm_point)?;
        let eval_var = circuit.create_public_emulated_variable::<E::ScalarField>(instance.value)?;
        let point_var =
            circuit.create_public_emulated_variable::<E::ScalarField>(instance.point)?;
        let proof_point = Point::from(instance.opening_proof.proof);
        let proof_var = circuit.create_point_variable(&proof_point)?;
        Ok(AtomicInstanceVar::new(
            comm_var, eval_var, point_var, proof_var,
        ))
    }
}

impl AtomicAccProofVar {
    /// Creates a new [`AtomicAccProofVar`] from its constituent variables.
    pub fn new(instance: PointVariable, proof: PointVariable) -> Self {
        Self { instance, proof }
    }

    /// Creates a new variable from an [`AtomicAccProof`].
    pub fn from_proof<PCS, F>(
        proof: &AtomicAccProof<PCS>,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<Self, CircuitError>
    where
        PCS: Accumulation,
        F: PrimeField + RescueParameter,
        PCS::Commitment: AffineRepr<BaseField = F>,
        <PCS::Commitment as AffineRepr>::Config: HasTEForm<BaseField = F>,
    {
        let instance_point = Point::from(proof.s_beta_g);
        let instance_var = circuit.create_point_variable(&instance_point)?;
        let proof_point = Point::from(proof.s_g);
        let proof_var = circuit.create_point_variable(&proof_point)?;
        Ok(AtomicAccProofVar::new(instance_var, proof_var))
    }
}

/// Here the generic 'E' will be the pairing curve the previous proof used.
/// This is a circuit representation of the the accumulation proof for Pedersen polynomial
/// commitments as described in https://eprint.iacr.org/2020/1618.pdf.
pub struct UVSplitProofVar<E: SWCurveConfig> {
    /// The evaluation of 'p_i(x)' at the point 'z_*'.
    pub y_i: Vec<EmulatedVariable<E::ScalarField>>,
    /// The evaluation of 'w_i(x)' at the point 'z_*'.
    pub y_i_prime: Vec<EmulatedVariable<E::ScalarField>>,
    /// Commitment to the polynomial 'w_i(x)'
    pub w_i_comm: Vec<PointVariable>,
}

impl<E: SWCurveConfig> UVSplitProofVar<E> {
    /// Construct a new proof from its constituent variables.
    pub fn new(
        y_i: Vec<EmulatedVariable<E::ScalarField>>,
        y_i_prime: Vec<EmulatedVariable<E::ScalarField>>,
        w_i_comm: Vec<PointVariable>,
    ) -> Self {
        Self {
            y_i,
            y_i_prime,
            w_i_comm,
        }
    }
}

/// Type that represent the variable version of a multilinear split-accumulation proof.
pub type MVSplitProofVar = SumCheckProofVar;

/// Struct used to store the challenges and scalars generated during the atomic
/// accumulation process as variables.
#[derive(Debug, Clone)]
pub struct AtomicAccumulationChallengesAndScalarsVar<F: PrimeField> {
    pub(crate) r: EmulatedVariable<F>,
    pub(crate) r_powers: Vec<EmulatedVariable<F>>,
    pub(crate) z_r_powers: Vec<EmulatedVariable<F>>,
    pub(crate) minus_v_r_powers: EmulatedVariable<F>,
}

impl<F: PrimeField> AtomicAccumulationChallengesAndScalarsVar<F> {
    /// Create a new instance from the associated variables.
    pub fn new(
        r: EmulatedVariable<F>,
        r_powers: Vec<EmulatedVariable<F>>,
        z_r_powers: Vec<EmulatedVariable<F>>,
        minus_v_r_powers: EmulatedVariable<F>,
    ) -> Self {
        Self {
            r,
            r_powers,
            z_r_powers,
            minus_v_r_powers,
        }
    }

    /// Creates the appropriate variable from a [`AtomicAccumulationChallengesAndScalars`].
    pub fn from_struct<P>(
        challenges_and_scalars: &AtomicAccumulationChallengesAndScalars<P::ScalarField>,
        circuit: &mut PlonkCircuit<P::BaseField>,
    ) -> Result<Self, CircuitError>
    where
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
        P::ScalarField: PrimeField + RescueParameter + EmulationConfig<P::BaseField>,
    {
        let r = circuit.create_emulated_variable(challenges_and_scalars.r)?;
        let r_powers = challenges_and_scalars
            .r_powers
            .iter()
            .map(|r_power| circuit.create_emulated_variable(*r_power))
            .collect::<Result<Vec<EmulatedVariable<F>>, CircuitError>>()?;
        let z_r_powers = challenges_and_scalars
            .z_r_powers
            .iter()
            .map(|z_r_power| circuit.create_emulated_variable(*z_r_power))
            .collect::<Result<Vec<EmulatedVariable<F>>, CircuitError>>()?;
        let minus_v_r_powers =
            circuit.create_emulated_variable(challenges_and_scalars.minus_v_r_powers)?;
        Ok(AtomicAccumulationChallengesAndScalarsVar::new(
            r,
            r_powers,
            z_r_powers,
            minus_v_r_powers,
        ))
    }

    /// Creates the appropriate public variable from a [`AtomicAccumulationChallengesAndScalars`].
    pub fn from_struct_public<P>(
        challenges_and_scalars: &AtomicAccumulationChallengesAndScalars<P::ScalarField>,
        circuit: &mut PlonkCircuit<P::BaseField>,
    ) -> Result<Self, CircuitError>
    where
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
        P::ScalarField: PrimeField + RescueParameter + EmulationConfig<P::BaseField>,
    {
        let r = circuit.create_public_emulated_variable(challenges_and_scalars.r)?;
        let r_powers = challenges_and_scalars
            .r_powers
            .iter()
            .map(|r_power| circuit.create_public_emulated_variable(*r_power))
            .collect::<Result<Vec<EmulatedVariable<F>>, CircuitError>>()?;
        let z_r_powers = challenges_and_scalars
            .z_r_powers
            .iter()
            .map(|z_r_power| circuit.create_public_emulated_variable(*z_r_power))
            .collect::<Result<Vec<EmulatedVariable<F>>, CircuitError>>()?;
        let minus_v_r_powers =
            circuit.create_public_emulated_variable(challenges_and_scalars.minus_v_r_powers)?;
        Ok(AtomicAccumulationChallengesAndScalarsVar::new(
            r,
            r_powers,
            z_r_powers,
            minus_v_r_powers,
        ))
    }

    /// Returns the instance scalars as a vector of variables.
    pub fn instance_scalars(&self) -> Vec<EmulatedVariable<F>> {
        self.r_powers
            .iter()
            .take(self.r_powers.len() - 1)
            .cloned()
            .chain(vec![self.minus_v_r_powers.clone()])
            .chain(self.z_r_powers.clone())
            .chain(self.r_powers.iter().skip(self.r_powers.len() - 1).cloned())
            .collect::<Vec<_>>()
    }

    /// Returns the proof scalars as a vector of variables.
    pub fn proof_scalars(&self) -> Vec<EmulatedVariable<F>> {
        self.r_powers.clone()
    }

    /// Returns a reference to the challenge `r`
    pub fn r(&self) -> &EmulatedVariable<F> {
        &self.r
    }
}

/// Struct used to store the challenges and scalars generated during the atomic
/// accumulation process as variables when the scalar field is small enough.
#[derive(Debug, Clone)]
pub struct AtomicAccumulationChallengesAndScalarsNativeVar<F: PrimeField> {
    pub(crate) r: EmulatedVariable<F>,
    pub(crate) r_powers: Vec<Variable>,
    pub(crate) z_r_powers: Vec<Variable>,
    pub(crate) minus_v_r_powers: Variable,
}

impl<F: PrimeField> AtomicAccumulationChallengesAndScalarsNativeVar<F> {
    /// Create a new instance from the associated variables.
    pub fn new(
        r: EmulatedVariable<F>,
        r_powers: Vec<Variable>,
        z_r_powers: Vec<Variable>,
        minus_v_r_powers: Variable,
    ) -> Self {
        Self {
            r,
            r_powers,
            z_r_powers,
            minus_v_r_powers,
        }
    }

    /// Creates the appropriate variable from a [`AtomicAccumulationChallengesAndScalars`].
    pub fn from_struct<P>(
        challenges_and_scalars: &AtomicAccumulationChallengesAndScalars<P::ScalarField>,
        circuit: &mut PlonkCircuit<P::BaseField>,
    ) -> Result<Self, CircuitError>
    where
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
        P::ScalarField: PrimeField + RescueParameter + EmulationConfig<P::BaseField>,
    {
        let r = circuit.create_emulated_variable(challenges_and_scalars.r)?;
        let r_powers = challenges_and_scalars
            .r_powers
            .iter()
            .map(|r_power| {
                let r_power_base =
                    P::BaseField::from_le_bytes_mod_order(&r_power.into_bigint().to_bytes_le());
                circuit.create_variable(r_power_base)
            })
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        let z_r_powers = challenges_and_scalars
            .z_r_powers
            .iter()
            .map(|z_r_power| {
                let z_r_power_base =
                    P::BaseField::from_le_bytes_mod_order(&z_r_power.into_bigint().to_bytes_le());
                circuit.create_variable(z_r_power_base)
            })
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        let minus_base = P::BaseField::from_le_bytes_mod_order(
            &challenges_and_scalars
                .minus_v_r_powers
                .into_bigint()
                .to_bytes_le(),
        );
        let minus_v_r_powers = circuit.create_variable(minus_base)?;
        Ok(AtomicAccumulationChallengesAndScalarsNativeVar::new(
            r,
            r_powers,
            z_r_powers,
            minus_v_r_powers,
        ))
    }

    /// Creates the appropriate public variable from a [`AtomicAccumulationChallengesAndScalars`].
    pub fn from_struct_public<P>(
        challenges_and_scalars: &AtomicAccumulationChallengesAndScalars<P::ScalarField>,
        circuit: &mut PlonkCircuit<P::BaseField>,
    ) -> Result<Self, CircuitError>
    where
        P: HasTEForm<ScalarField = F>,
        P::BaseField: PrimeField + RescueParameter + EmulationConfig<P::ScalarField>,
        P::ScalarField: PrimeField + RescueParameter + EmulationConfig<P::BaseField>,
    {
        let r = circuit.create_public_emulated_variable(challenges_and_scalars.r)?;
        let r_powers = challenges_and_scalars
            .r_powers
            .iter()
            .map(|r_power| {
                let r_power_base =
                    P::BaseField::from_le_bytes_mod_order(&r_power.into_bigint().to_bytes_le());
                circuit.create_public_variable(r_power_base)
            })
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        let z_r_powers = challenges_and_scalars
            .z_r_powers
            .iter()
            .map(|z_r_power| {
                let z_r_power_base =
                    P::BaseField::from_le_bytes_mod_order(&z_r_power.into_bigint().to_bytes_le());
                circuit.create_public_variable(z_r_power_base)
            })
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        let minus_base = P::BaseField::from_le_bytes_mod_order(
            &challenges_and_scalars
                .minus_v_r_powers
                .into_bigint()
                .to_bytes_le(),
        );
        let minus_v_r_powers = circuit.create_public_variable(minus_base)?;
        Ok(AtomicAccumulationChallengesAndScalarsNativeVar::new(
            r,
            r_powers,
            z_r_powers,
            minus_v_r_powers,
        ))
    }

    /// Returns the instance scalars as a vector of variables.
    pub fn instance_scalars(&self) -> Vec<Variable> {
        self.r_powers
            .iter()
            .take(self.r_powers.len() - 1)
            .cloned()
            .chain(vec![self.minus_v_r_powers])
            .chain(self.z_r_powers.clone())
            .chain(self.r_powers.iter().skip(self.r_powers.len() - 1).cloned())
            .collect::<Vec<_>>()
    }

    /// Returns the proof scalars as a vector of variables.
    pub fn proof_scalars(&self) -> Vec<Variable> {
        self.r_powers.clone()
    }

    /// Returns a reference to the challenge `r`
    pub fn r(&self) -> &EmulatedVariable<F> {
        &self.r
    }
}
