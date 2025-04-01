//! This file contains the code for verifying the accumulation of proofs using atomic accumulation.

use ark_bn254::{g1::Config as BnConfig, Fq as Fq254, Fr as Fr254};
use ark_ec::short_weierstrass::Affine;

use ark_std::vec::Vec;
use jf_relation::{
    errors::CircuitError,
    gadgets::ecc::{MultiScalarMultiplicationCircuit, Point, PointVariable},
    Circuit, PlonkCircuit, Variable,
};
use jf_utils::fr_to_fq;

/// Function to perform atomic accumulation in a Grumpkin circuit.
pub fn atomic_accumulate(
    circuit: &mut PlonkCircuit<Fq254>,
    instance_scalars: &[Fr254],
    instance_bases: &[Affine<BnConfig>],
    proof_scalars: &[Fr254],
    proof_bases: &[Affine<BnConfig>],
) -> Result<(PointVariable, PointVariable), CircuitError> {
    let instance_scalar_vars = instance_scalars
        .iter()
        .map(|s| circuit.create_variable(fr_to_fq::<Fq254, BnConfig>(s)))
        .collect::<Result<Vec<Variable>, CircuitError>>()?;
    let proof_scalar_vars = proof_scalars
        .iter()
        .map(|s| circuit.create_variable(fr_to_fq::<Fq254, BnConfig>(s)))
        .collect::<Result<Vec<Variable>, CircuitError>>()?;

    let instance_base_vars = instance_bases
        .iter()
        .map(|base| circuit.create_point_variable(&Point::<Fq254>::from(*base)))
        .collect::<Result<Vec<PointVariable>, CircuitError>>()?;
    let proof_base_vars = proof_bases
        .iter()
        .map(|base| circuit.create_point_variable(&Point::<Fq254>::from(*base)))
        .collect::<Result<Vec<PointVariable>, CircuitError>>()?;

    let acc_instance = MultiScalarMultiplicationCircuit::<Fq254, BnConfig>::msm(
        circuit,
        &instance_base_vars,
        &instance_scalar_vars,
    )?;

    let acc_proof = MultiScalarMultiplicationCircuit::<Fq254, BnConfig>::msm(
        circuit,
        &proof_base_vars[1..],
        &proof_scalar_vars[1..],
    )?;

    let acc_proof = circuit.ecc_add::<BnConfig>(&acc_proof, &proof_base_vars[0])?;
    Ok((acc_instance, acc_proof))
}
