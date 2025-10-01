//! Code for converting PCS::Proof and Scheme::RecursiveProof to variable form.
use crate::nightfall::{
    circuit::plonk_partial_verifier::PlookupProofScalarsAndBasesVar,
    hops::univariate_ipa::UnivariateIpaProof,
    ipa_structs::{PlookupProof, Proof},
    FFTPlonk,
};
use crate::proof_system::{structs::ProofEvaluations, UniversalRecursiveSNARK};
use crate::recursion::{circuits::Kzg, merge_functions::Bn254Output};
use crate::transcript::{RescueTranscript, Transcript};
use ark_bn254::{Fq as Fq254, Fr as Fr254};
use ark_ec::{pairing::Pairing, short_weierstrass::Affine, AffineRepr};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{fmt::Debug, vec, vec::Vec};
use jf_primitives::{
    pcs::{prelude::UnivariateKzgProof, Accumulation, PolynomialCommitmentScheme},
    rescue::RescueParameter,
};
use jf_relation::{
    errors::CircuitError,
    gadgets::{
        ecc::{HasTEForm, Point, PointVariable},
        EmulatedVariable, EmulationConfig,
    },
    PlonkCircuit,
};

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

/// Trait that tells a circuit how to convert a Plookup proof into circuit variables.
pub trait PlookupProofToVar<PCS>
where
    PCS: PolynomialCommitmentScheme,
    <<PCS as PolynomialCommitmentScheme>::Commitment as AffineRepr>::BaseField: PrimeField,
{
    /// The variable form of the Plookup proof bases.
    type PlookupProofVar: Debug + Clone;
    /// Create a new variable from a reference to a Plookup proof bases.
    fn create_variables(
        &self,
        circuit: &mut PlonkCircuit<<PCS::Commitment as AffineRepr>::BaseField>,
    ) -> Result<Self::PlookupProofVar, CircuitError>;
}

impl<PCS, P> PlookupProofToVar<PCS> for PlookupProof<PCS>
where
    PCS: PolynomialCommitmentScheme<Commitment = Affine<P>> + Debug,
    P: HasTEForm,
    P::BaseField: PrimeField,
{
    type PlookupProofVar = PlookupProofScalarsAndBasesVar<PCS>;
    fn create_variables(
        &self,
        circuit: &mut PlonkCircuit<<PCS::Commitment as AffineRepr>::BaseField>,
    ) -> Result<Self::PlookupProofVar, CircuitError> {
        let h_poly_comms = self
            .h_poly_comms
            .iter()
            .map(|p| circuit.create_point_variable(&Point::from(*p)))
            .collect::<Result<Vec<PointVariable>, CircuitError>>()?;
        let prod_lookup_poly_comm = circuit
            .create_point_variable(&Point::<P::BaseField>::from(self.prod_lookup_poly_comm))?;
        Ok(Self::PlookupProofVar {
            h_poly_comms,
            prod_lookup_poly_comm,
            poly_evals: self.poly_evals.clone(),
        })
    }
}

/// Trait that tells a circuit how to convert a recursive proof into circuit variables.
pub trait RecursiveProofToScalarsAndBasesVar<PCS>
where
    PCS: PolynomialCommitmentScheme,
    <<PCS as PolynomialCommitmentScheme>::Commitment as AffineRepr>::BaseField: PrimeField,
{
    /// The variable form of the recursive proof.
    type RecursiveProofVar: Debug + Clone;
    /// Create a new variable from a reference to a recursive proof.
    fn create_variables(
        &self,
        circuit: &mut PlonkCircuit<<PCS::Commitment as AffineRepr>::BaseField>,
    ) -> Result<Self::RecursiveProofVar, CircuitError>;
}

/// A struct containing the scalars and bases associated with a Plookup argument proof.
/// The bases are stored as `Variable`s to be passed into a circuit defined over the
/// base field of the commitment curve.
#[derive(Debug, Clone)]
pub struct Bn254ProofScalarsandBasesVar {
    /// Wire witness polynomials commitments.
    pub wires_poly_comms: Vec<PointVariable>,

    /// The polynomial commitment for the wire permutation argument.
    pub prod_perm_poly_comm: PointVariable,

    /// Splitted quotient polynomial commitments.
    pub split_quot_poly_comms: Vec<PointVariable>,

    /// (Aggregated) proof of evaluations at challenge point `v`.
    pub opening_proof: PointVariable,

    /// The commitment to the polynomial q(x) used in the multi-opening protocol.
    pub q_comm: PointVariable,

    /// Polynomial evaluations that we store in the clear
    pub(crate) poly_evals: ProofEvaluations<Fr254>,

    /// The bases associated to the partial proof for Plookup argument
    pub plookup_proof: Option<PlookupProofScalarsAndBasesVar<Kzg>>,
}

impl Bn254ProofScalarsandBasesVar {
    /// Convert to vector of point variables.
    pub fn to_vec(&self) -> Vec<PointVariable> {
        let mut vars = vec![];
        vars.extend_from_slice(&self.wires_poly_comms);
        vars.push(self.prod_perm_poly_comm);
        vars.extend_from_slice(&self.split_quot_poly_comms);
        vars.push(self.opening_proof);
        vars.push(self.q_comm);
        if let Some(plookup_proof) = &self.plookup_proof {
            vars.extend_from_slice(&plookup_proof.h_poly_comms);
            vars.push(plookup_proof.prod_lookup_poly_comm);
        }
        vars
    }
}

impl RecursiveProofToScalarsAndBasesVar<Kzg> for Proof<Kzg> {
    type RecursiveProofVar = Bn254ProofScalarsandBasesVar;
    fn create_variables(
        &self,
        circuit: &mut PlonkCircuit<Fq254>,
    ) -> Result<Self::RecursiveProofVar, CircuitError> {
        let wires_poly_comms = self
            .wires_poly_comms
            .iter()
            .map(|p| circuit.create_point_variable(&Point::<Fq254>::from(*p)))
            .collect::<Result<Vec<PointVariable>, CircuitError>>()?;

        let prod_perm_poly_comm =
            circuit.create_point_variable(&Point::<Fq254>::from(self.prod_perm_poly_comm))?;

        let split_quot_poly_comms = self
            .split_quot_poly_comms
            .iter()
            .map(|p| circuit.create_point_variable(&Point::<Fq254>::from(*p)))
            .collect::<Result<Vec<PointVariable>, CircuitError>>()?;

        let opening_proof =
            circuit.create_point_variable(&Point::<Fq254>::from(self.opening_proof.proof))?;

        let q_comm = circuit.create_point_variable(&Point::<Fq254>::from(self.q_comm))?;

        let poly_evals = self.poly_evals.clone();

        let plookup_proof = if let Some(plookup_proof) = &self.plookup_proof {
            plookup_proof.create_variables(circuit).map(Some)?
        } else {
            None
        };

        Ok(Self::RecursiveProofVar {
            wires_poly_comms,
            prod_perm_poly_comm,
            split_quot_poly_comms,
            opening_proof,
            q_comm,
            poly_evals,
            plookup_proof,
        })
    }
}

/// Trait that tells a circuit how to convert a recursive output into circuit variables.
pub trait RecursiveOutputToScalarsAndBasesVar<PCS, Scheme, T>
where
    PCS: Accumulation,
    PCS::Commitment: AffineRepr,
    <PCS::Commitment as AffineRepr>::BaseField: PrimeField,
    <PCS::Commitment as AffineRepr>::ScalarField:
        PrimeField + CanonicalSerialize + CanonicalDeserialize,
    Scheme: UniversalRecursiveSNARK<PCS>,
    Scheme::RecursiveProof: CanonicalSerialize + CanonicalDeserialize,
    T: Transcript + CanonicalSerialize + CanonicalDeserialize,
    PCS: PolynomialCommitmentScheme,
    <<PCS as PolynomialCommitmentScheme>::Commitment as AffineRepr>::BaseField: PrimeField,
{
    /// The variable form of the recursive proof.
    type RecursiveOutputVar: Debug + Clone;
    /// Create a new variable from a reference to a recursive proof.
    fn create_variables(
        &self,
        circuit: &mut PlonkCircuit<<PCS::Commitment as AffineRepr>::BaseField>,
    ) -> Result<Self::RecursiveOutputVar, CircuitError>;
}

/// A struct containing the scalars and bases associated with a `Bn254Output`.
/// The bases are stored as `Variable`s to be passed into a circuit defined over the
/// base field of the commitment curve.
#[derive(Debug, Clone)]
pub struct Bn254OutputScalarsAndBasesVar {
    /// The proof generated by the recursive prover.
    pub proof: Bn254ProofScalarsandBasesVar,
    /// The hash of the public inputs to this proof stored in the clear.
    pub pi_hash: Fr254,
    /// The transcript of the proof stored in the clear.
    pub transcript: RescueTranscript<Fr254>,
}

impl Bn254OutputScalarsAndBasesVar {
    /// Convert to vector of point variables.
    pub fn to_vec(&self) -> Vec<PointVariable> {
        self.proof.to_vec()
    }
}

impl RecursiveOutputToScalarsAndBasesVar<Kzg, FFTPlonk<Kzg>, RescueTranscript<Fr254>>
    for Bn254Output
{
    type RecursiveOutputVar = Bn254OutputScalarsAndBasesVar;
    fn create_variables(
        &self,
        circuit: &mut PlonkCircuit<Fq254>,
    ) -> Result<Self::RecursiveOutputVar, CircuitError> {
        let proof = self.proof.create_variables(circuit)?;
        let pi_hash = self.pi_hash;
        let transcript = self.transcript.clone();
        Ok(Self::RecursiveOutputVar {
            proof,
            pi_hash,
            transcript,
        })
    }
}
