//! Structs used in verifying SumCheck proof scalar field arithmetic in a circuit context.
use ark_ff::PrimeField;
use ark_std::vec::Vec;

use jf_relation::{gadgets::EmulatedVariable, Variable};

/// A struct used to put polynomial oracles into circuits
pub struct PolyOracleVar {
    pub(crate) evaluations_var: Vec<Variable>,
    pub(crate) weights_var: Vec<Variable>,
}

/// A struct used to put sum check proofs into circuits
pub struct SumCheckProofVar {
    pub(crate) eval_var: Variable,
    pub(crate) oracles_var: Vec<PolyOracleVar>,
    pub(crate) r_0_evals_var: Vec<Variable>,
    pub(crate) point_var: Vec<Variable>,
}

/// Struct used for representing a SumCheck proof using [`Emulatedvariable`]s so that the whole thing can be verified in a circuit.
pub struct EmulatedSumCheckProofVar<E: PrimeField> {
    pub(crate) eval_var: EmulatedVariable<E>,
    pub(crate) oracles_var: Vec<EmulatedPolyOracleVar<E>>,
    pub(crate) r_0_evals_var: Vec<EmulatedVariable<E>>,
    pub(crate) point_var: Vec<EmulatedVariable<E>>,
}

/// Struct used to represent a `PolyOracle` using emulated variables so that hashes can be verified.
pub struct EmulatedPolyOracleVar<E: PrimeField> {
    pub(crate) evaluations_var: Vec<EmulatedVariable<E>>,
    pub(crate) weights_var: Vec<EmulatedVariable<E>>,
}
