// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Instantiations of Plonk-based proof systems

use crate::{
    constants::EXTRA_TRANSCRIPT_MSG_LABEL,
    errors::PlonkError,
    nightfall::{
        hops::univariate_ipa::UnivariateIpaPCS,
        ipa_structs::{
            Challenges, Oracles, PlookupProof, PlookupProvingKey, PlookupVerifyingKey, Proof,
            ProvingKey, VerificationKeyId, VerifyingKey,
        },
        ipa_verifier::FFTVerifier,
    },
    proof_system::{RecursiveOutput, UniversalRecursiveSNARK, UniversalSNARK},
    transcript::*,
};

use ark_ec::{pairing::Pairing, short_weierstrass::Affine};
use ark_ff::PrimeField;

use ark_poly::{univariate::DensePolynomial, Polynomial};
use ark_std::{
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    string::ToString,
    vec::Vec,
    One,
};

use super::ipa_prover::FFTProver;
use jf_primitives::{
    pcs::{
        prelude::UnivariateKzgProof, Accumulation, PolynomialCommitmentScheme,
        StructuredReferenceString,
    },
    rescue::RescueParameter,
};
use jf_relation::{
    constants::compute_coset_representatives,
    gadgets::{ecc::HasTEForm, EmulationConfig},
    Arithmetization,
};
use jf_utils::par_utils::parallelizable_slice_iter;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// We need to be able to push `UnivariateKzgProof`s to the transcript
impl<E, P> TranscriptVisitor for UnivariateKzgProof<E>
where
    E: Pairing<BaseField = P::BaseField, G1Affine = Affine<P>>,
    P: HasTEForm,
    P::BaseField: PrimeField,
{
    fn append_to_transcript<T: Transcript>(
        &self,
        transcript: &mut T,
    ) -> Result<(), crate::errors::PlonkError> {
        transcript.append_curve_point(b"kzg_proof", &self.proof)?;
        Ok(())
    }
}

/// A struct for making Plonk proofs with FFTs using the IPA PCS.
pub type PlonkIpaSnark<E> = FFTPlonk<UnivariateIpaPCS<E>>;

/// A struct for making Plonk proofs with FFTs using a PCS that has additively homomorphic commitments.
#[derive(Debug, Default, Clone)]
pub struct FFTPlonk<PCS: PolynomialCommitmentScheme>(PhantomData<PCS>);

impl<PCS, F, P> FFTPlonk<PCS>
where
    PCS: PolynomialCommitmentScheme<
        Evaluation = P::ScalarField,
        Polynomial = DensePolynomial<P::ScalarField>,
        Point = P::ScalarField,
        Commitment = Affine<P>,
    >,
    F: RescueParameter,
    P: HasTEForm<BaseField = F>,
    P::ScalarField: EmulationConfig<F>,
    PCS::SRS: StructuredReferenceString<Item = PCS::Commitment>,
{
    #[allow(clippy::new_without_default, dead_code)]
    /// A new FFTPlonk SNARK
    pub fn new() -> Self {
        Self(PhantomData)
    }

    #[allow(dead_code)]
    /// Generate a Plonk proof.
    pub fn prove<C, R, T>(
        prng: &mut R,
        circuits: &C,
        prove_keys: &ProvingKey<PCS>,
        extra_transcript_init_msg: Option<Vec<u8>>,
        blind: bool,
    ) -> Result<Proof<PCS>, PlonkError>
    where
        C: Arithmetization<P::ScalarField>,
        R: CryptoRng + RngCore,
        T: Transcript,
    {
        let (proof, ..) = Self::prove_internal::<_, _, T>(
            prng,
            circuits,
            prove_keys,
            extra_transcript_init_msg,
            blind,
        )?;
        Ok(proof)
    }

    /// Verify a single aggregated Plonk proof.
    pub fn verify_proof<T>(
        verify_key: &VerifyingKey<PCS>,
        public_inputs: &[P::ScalarField],
        proof: &Proof<PCS>,
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<(), PlonkError>
    where
        T: Transcript,
    {
        let verifier = FFTVerifier::<PCS>::new(verify_key.domain_size)?;
        let pcs_info = verifier.prepare_pcs_info::<T>(
            verify_key,
            public_inputs,
            proof,
            &extra_transcript_init_msg,
        )?;
        if !FFTVerifier::verify_opening_proofs(
            &verify_key.open_key, // all open_key are the same
            &pcs_info,
        )? {
            return Err(PlonkError::WrongProof);
        }
        Ok(())
    }

    /// Verify a single Plonk proof that has been produced for recursive purposes.
    pub fn verify_recursive_proof<T>(
        verify_key: &VerifyingKey<PCS>,
        proof: &RecursiveOutput<PCS, Self, T>,
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<(), PlonkError>
    where
        PCS: Accumulation<Proof: TranscriptVisitor>,
        T: Transcript + ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize,
    {
        let verifier = FFTVerifier::<PCS>::new(verify_key.domain_size)?;
        let pcs_info = verifier.prepare_pcs_info::<T>(
            verify_key,
            &[proof.pi_hash],
            &proof.proof,
            &extra_transcript_init_msg,
        )?;
        if !FFTVerifier::verify_opening_proofs(
            &verify_key.open_key, // all open_key are the same
            &pcs_info,
        )? {
            return Err(PlonkError::WrongProof);
        }
        Ok(())
    }

    /// An internal private API for ease of testing
    ///
    /// Compute a Plonk IPA proof. Return the
    /// proof and the corresponding online polynomial oracles and
    /// challenges. Refer to Sec 8.4 of https://eprint.iacr.org/2019/953.pdf
    #[allow(clippy::type_complexity)]
    fn prove_internal<C, R, T>(
        prng: &mut R,
        circuits: &C,
        prove_keys: &ProvingKey<PCS>,
        extra_transcript_init_msg: Option<Vec<u8>>,
        blind: bool,
    ) -> Result<
        (
            Proof<PCS>,
            Oracles<P::ScalarField>,
            Challenges<P::ScalarField>,
        ),
        PlonkError,
    >
    where
        C: Arithmetization<P::ScalarField>,
        R: CryptoRng + RngCore,
        T: Transcript,
    {
        let n = circuits.eval_domain_size()?;
        let num_wire_types = circuits.num_wire_types();

        // Initialize transcript
        let mut transcript = T::new_transcript(b"PlonkProof");
        if let Some(msg) = extra_transcript_init_msg {
            transcript.push_message(EXTRA_TRANSCRIPT_MSG_LABEL, &msg)?;
        }

        // For FFTPlonk we only add the vk ID in the non-merged case.
        if prove_keys.vk.id.is_some() {
            transcript.append_visitor(&prove_keys.vk)?;
        }

        for pub_in in circuits.public_input()?.iter() {
            transcript.push_message(b"public_input", pub_in)?;
        }

        // Initialize verifier challenges and online polynomial oracles.
        let mut challenges = Challenges::default();
        let mut online_oracles = Oracles::default();
        let prover = FFTProver::<PCS>::new(n, num_wire_types)?;

        // Round 1

        let ((wires_poly_comms, wire_polys), pi_poly) =
            prover.run_1st_round(prng, &prove_keys.commit_key, circuits, blind)?;
        online_oracles.wire_polys = wire_polys;
        online_oracles.pub_inp_poly = pi_poly;

        transcript.append_curve_points(b"witness_poly_comms", &wires_poly_comms)?;

        // Round 1.5
        // Plookup: compute and interpolate the sorted concatenation of the (merged)
        // lookup table and the (merged) witness values
        challenges.tau = transcript.squeeze_scalar_challenge::<P>(b"tau")?;

        let (sorted_vec, h_poly_comms, merged_table) = if circuits.support_lookup() {
            let ((h_poly_comms, h_polys), sorted_vec, merged_table) = prover
                .run_plookup_1st_round(
                    prng,
                    &prove_keys.commit_key,
                    circuits,
                    challenges.tau,
                    blind,
                )?;
            online_oracles.plookup_oracles.h_polys = h_polys;

            transcript.append_curve_points(b"h_poly_comms", &h_poly_comms)?;
            (Some(sorted_vec), Some(h_poly_comms), Some(merged_table))
        } else {
            (None, None, None)
        };

        // Round 2

        let [beta, gamma]: [P::ScalarField; 2] = transcript
            .squeeze_scalar_challenges::<P>(b"beta gamma", 2)?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters("Couldn't convert to fixed length array".to_string())
            })?;

        challenges.beta = beta;
        challenges.gamma = gamma;
        let (prod_perm_poly_comm, prod_perm_poly) =
            prover.run_2nd_round(prng, &prove_keys.commit_key, circuits, &challenges, blind)?;
        online_oracles.prod_perm_poly = prod_perm_poly;
        transcript.append_curve_point(b"perm_poly_comms", &prod_perm_poly_comm)?;

        // Round 2.5
        // Plookup: compute Plookup product accumulation polynomial

        let prod_lookup_poly_comm = if circuits.support_lookup() {
            let (prod_lookup_poly_comm, prod_lookup_poly) = prover.run_plookup_2nd_round(
                prng,
                &prove_keys.commit_key,
                circuits,
                &challenges,
                merged_table.as_ref(),
                sorted_vec.as_ref(),
                blind,
            )?;
            online_oracles.plookup_oracles.prod_lookup_poly = prod_lookup_poly;
            transcript.append_curve_point(b"plookup_poly_comms", &prod_lookup_poly_comm)?;
            Some(prod_lookup_poly_comm)
        } else {
            None
        };

        // Round 3
        challenges.alpha = transcript.squeeze_scalar_challenge::<P>(b"alpha")?;
        let (split_quot_poly_comms, split_quot_polys) = prover.run_3rd_round(
            prng,
            &prove_keys.commit_key,
            prove_keys,
            &challenges,
            &online_oracles,
            num_wire_types,
            blind,
        )?;

        transcript.append_curve_points(b"quot_poly_comms", &split_quot_poly_comms)?;

        // Round 4
        challenges.zeta = transcript.squeeze_scalar_challenge::<P>(b"zeta")?;

        let poly_evals =
            prover.compute_evaluations(prove_keys, &challenges, &online_oracles, num_wire_types);
        transcript.append_visitor(&poly_evals)?;

        // Round 4.5
        // Plookup: compute evaluations on Plookup-related polynomials

        let plookup_evals = if circuits.support_lookup() {
            let evals =
                prover.compute_plookup_evaluations(prove_keys, &challenges, &online_oracles)?;
            transcript.append_visitor(&evals)?;
            Some(evals)
        } else {
            None
        };

        let mut lin_poly = FFTProver::<PCS>::compute_quotient_component_for_lin_poly(
            n,
            challenges.zeta,
            &split_quot_polys,
        )?;

        lin_poly = lin_poly
            + prover.compute_non_quotient_component_for_lin_poly(
                P::ScalarField::one(),
                prove_keys,
                &challenges,
                &online_oracles,
                &poly_evals,
                plookup_evals.as_ref(),
            )?;

        let lin_poly_const = lin_poly.evaluate(&challenges.zeta);

        lin_poly[0] -= lin_poly_const;

        // Round 5

        challenges.v = transcript.squeeze_scalar_challenge::<P>(b"v")?;

        let (polys_and_eval_sets, optional_lagrange_polys) = prover.create_polys_and_eval_sets(
            prove_keys,
            &challenges.zeta,
            &poly_evals,
            &online_oracles,
            &lin_poly,
        )?;

        let (q_comm, q_poly) = prover.compute_q_comm_and_eval_polys(
            &prove_keys.commit_key,
            &polys_and_eval_sets,
            &challenges.v,
        )?;

        transcript.append_curve_point(b"q_comm", &q_comm)?;
        challenges.u = transcript.squeeze_scalar_challenge::<P>(b"u")?;

        let g_poly = prover.compute_g_poly_and_comm(
            &polys_and_eval_sets,
            optional_lagrange_polys,
            &challenges.v,
            &challenges.u,
            &challenges.zeta,
            &q_poly,
        )?;

        let (opening_proof, _) = PCS::open(&prove_keys.commit_key, &g_poly, &challenges.u)?;

        // Plookup: build Plookup argument

        let plookup_proof = if circuits.support_lookup() {
            Some(PlookupProof {
                h_poly_comms: h_poly_comms.unwrap(),
                prod_lookup_poly_comm: prod_lookup_poly_comm.unwrap(),
                poly_evals: plookup_evals.unwrap(),
            })
        } else {
            None
        };
        drop(polys_and_eval_sets);
        Ok((
            Proof {
                wires_poly_comms,
                prod_perm_poly_comm,
                poly_evals,
                plookup_proof,
                split_quot_poly_comms,
                opening_proof,
                q_comm,
            },
            online_oracles,
            challenges,
        ))
    }

    /// An internal private API for ease of testing
    ///
    /// Compute a Plonk proof for use in recursion, so we commit to the public input polynomial. Return the
    /// proof and the corresponding online polynomial oracles and
    /// challenges. Refer to Sec 8.4 of https://eprint.iacr.org/2019/953.pdf
    #[allow(clippy::type_complexity)]
    fn recursive_prove_internal<C, R, T>(
        prng: &mut R,
        circuits: &C,
        prove_keys: &ProvingKey<PCS>,
        extra_transcript_init_msg: Option<Vec<u8>>,
        blind: bool,
    ) -> Result<InternalRecursionOutput<PCS, T>, PlonkError>
    where
        PCS: PolynomialCommitmentScheme<Proof: TranscriptVisitor>,
        C: Arithmetization<P::ScalarField>,
        R: CryptoRng + RngCore,
        T: Transcript,
    {
        let n = circuits.eval_domain_size()?;
        let num_wire_types = circuits.num_wire_types();

        if !circuits.is_recursive() {
            return Err(PlonkError::InvalidParameters(
                "Circuit is not recursive".to_string(),
            ));
        }

        // Initialize transcript
        let mut transcript = T::new_transcript(b"PlonkProof");
        if let Some(msg) = extra_transcript_init_msg {
            transcript.push_message(EXTRA_TRANSCRIPT_MSG_LABEL, &msg)?;
        }

        // Initialize verifier challenges and online polynomial oracles.
        let mut challenges = Challenges::default();
        let mut online_oracles = Oracles::default();
        let prover = FFTProver::<PCS>::new(n, num_wire_types)?;

        // For FFTPlonk we add the ID to the transcript, if the vk has one.
        if prove_keys.vk.id.is_some() {
            transcript.append_visitor(&prove_keys.vk)?;
        }
        // In the recursive setting we know that the public inputs have length 1.
        transcript.push_message(b"public_input", &circuits.public_input()?[0])?;

        // Round 1
        let ((wires_poly_comms, wire_polys), pi_poly) =
            prover.run_1st_round(prng, &prove_keys.commit_key, circuits, blind)?;
        for poly in wire_polys.clone() {
            ark_std::println!("wire poly degree: {}", poly.degree());
        }
        online_oracles.wire_polys = wire_polys;
        online_oracles.pub_inp_poly = pi_poly;

        // Append the wire poly comms.
        transcript.append_curve_points(b"witness_poly_comms", &wires_poly_comms)?;

        // Round 1.5
        // Plookup: compute and interpolate the sorted concatenation of the (merged)
        // lookup table and the (merged) witness values
        challenges.tau = transcript.squeeze_scalar_challenge::<P>(b"tau")?;

        let (sorted_vec, h_poly_comms, merged_table) = if circuits.support_lookup() {
            let ((h_poly_comms, h_polys), sorted_vec, merged_table) = prover
                .run_plookup_1st_round(
                    prng,
                    &prove_keys.commit_key,
                    circuits,
                    challenges.tau,
                    blind,
                )?;
            online_oracles.plookup_oracles.h_polys = h_polys;

            transcript.append_curve_points(b"h_poly_comms", &h_poly_comms)?;
            (Some(sorted_vec), Some(h_poly_comms), Some(merged_table))
        } else {
            (None, None, None)
        };

        // Round 2

        let [beta, gamma]: [P::ScalarField; 2] = transcript
            .squeeze_scalar_challenges::<P>(b"beta gamma", 2)?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters("Couldn't convert to fixed length array".to_string())
            })?;

        challenges.beta = beta;
        challenges.gamma = gamma;
        let (prod_perm_poly_comm, prod_perm_poly) =
            prover.run_2nd_round(prng, &prove_keys.commit_key, circuits, &challenges, blind)?;
        online_oracles.prod_perm_poly = prod_perm_poly;
        transcript.append_curve_point(b"perm_poly_comms", &prod_perm_poly_comm)?;

        // Round 2.5
        // Plookup: compute Plookup product accumulation polynomial

        let prod_lookup_poly_comm = if circuits.support_lookup() {
            let (prod_lookup_poly_comm, prod_lookup_poly) = prover.run_plookup_2nd_round(
                prng,
                &prove_keys.commit_key,
                circuits,
                &challenges,
                merged_table.as_ref(),
                sorted_vec.as_ref(),
                blind,
            )?;
            online_oracles.plookup_oracles.prod_lookup_poly = prod_lookup_poly;
            transcript.append_curve_point(b"plookup_poly_comms", &prod_lookup_poly_comm)?;
            Some(prod_lookup_poly_comm)
        } else {
            None
        };

        // Round 3
        challenges.alpha = transcript.squeeze_scalar_challenge::<P>(b"alpha")?;
        let (split_quot_poly_comms, split_quot_polys) = prover.run_3rd_round(
            prng,
            &prove_keys.commit_key,
            prove_keys,
            &challenges,
            &online_oracles,
            num_wire_types,
            blind,
        )?;

        transcript.append_curve_points(b"quot_poly_comms", &split_quot_poly_comms)?;

        // Round 4
        challenges.zeta = transcript.squeeze_scalar_challenge::<P>(b"zeta")?;

        let poly_evals =
            prover.compute_evaluations(prove_keys, &challenges, &online_oracles, num_wire_types);
        transcript.append_visitor(&poly_evals)?;

        // Round 4.5
        // Plookup: compute evaluations on Plookup-related polynomials

        let plookup_evals = if circuits.support_lookup() {
            let evals =
                prover.compute_plookup_evaluations(prove_keys, &challenges, &online_oracles)?;
            transcript.append_visitor(&evals)?;
            Some(evals)
        } else {
            None
        };

        let mut lin_poly = FFTProver::<PCS>::compute_quotient_component_for_lin_poly(
            n,
            challenges.zeta,
            &split_quot_polys,
        )?;

        lin_poly = lin_poly
            + prover.compute_non_quotient_component_for_lin_poly(
                P::ScalarField::one(),
                prove_keys,
                &challenges,
                &online_oracles,
                &poly_evals,
                plookup_evals.as_ref(),
            )?;

        let lin_poly_const = lin_poly.evaluate(&challenges.zeta);

        lin_poly[0] -= lin_poly_const;

        // Round 5

        challenges.v = transcript.squeeze_scalar_challenge::<P>(b"v")?;

        let (polys_and_eval_sets, optional_lagrange_polys) = prover.create_polys_and_eval_sets(
            prove_keys,
            &challenges.zeta,
            &poly_evals,
            &online_oracles,
            &lin_poly,
        )?;

        let (q_comm, q_poly) = prover.compute_q_comm_and_eval_polys(
            &prove_keys.commit_key,
            &polys_and_eval_sets,
            &challenges.v,
        )?;

        transcript.append_curve_point(b"q_comm", &q_comm)?;
        challenges.u = transcript.squeeze_scalar_challenge::<P>(b"u")?;

        let g_poly = prover.compute_g_poly_and_comm(
            &polys_and_eval_sets,
            optional_lagrange_polys,
            &challenges.v,
            &challenges.u,
            &challenges.zeta,
            &q_poly,
        )?;

        let (opening_proof, _) = PCS::open(&prove_keys.commit_key, &g_poly, &challenges.u)?;
        // As we continue to use the transcript in the recursive setting,
        // we need to append the opening proof to the transcript.
        transcript.append_visitor(&opening_proof)?;

        // Plookup: build Plookup argument

        let plookup_proof = if circuits.support_lookup() {
            Some(PlookupProof {
                h_poly_comms: h_poly_comms.unwrap(),
                prod_lookup_poly_comm: prod_lookup_poly_comm.unwrap(),
                poly_evals: plookup_evals.unwrap(),
            })
        } else {
            None
        };
        drop(polys_and_eval_sets);

        let proof = Proof {
            wires_poly_comms,
            prod_perm_poly_comm,
            poly_evals,
            plookup_proof,
            split_quot_poly_comms,
            opening_proof,
            q_comm,
        };

        Ok((proof, transcript))
    }
}

/// Type for simplifying recursive proving outputs.
type InternalRecursionOutput<PCS, T> = (Proof<PCS>, T);

impl<PCS, F, P> UniversalSNARK<PCS> for FFTPlonk<PCS>
where
    F: RescueParameter,
    P: HasTEForm<BaseField = F>,
    P::ScalarField: EmulationConfig<F>,
    PCS: PolynomialCommitmentScheme<
        Evaluation = P::ScalarField,
        Polynomial = DensePolynomial<P::ScalarField>,
        Point = P::ScalarField,
        Commitment = Affine<P>,
    >,
    PCS::SRS: StructuredReferenceString<Item = PCS::Commitment>,
{
    type Proof = Proof<PCS>;
    type ProvingKey = ProvingKey<PCS>;
    type VerifyingKey = VerifyingKey<PCS>;
    type UniversalSRS = PCS::SRS;
    type Error = PlonkError;

    // FIXME: (alex) see <https://github.com/EspressoSystems/jellyfish/issues/249>
    #[cfg(any(test, feature = "test-srs"))]
    fn universal_setup_for_testing<R: RngCore + CryptoRng>(
        max_degree: usize,
        rng: &mut R,
    ) -> Result<Self::UniversalSRS, Self::Error> {
        <PCS::SRS as StructuredReferenceString>::gen_srs_for_testing(rng, max_degree)
            .map_err(PlonkError::PCSError)
    }

    /// Input a circuit and the SRS, precompute the proving key and verification
    /// key.
    fn preprocess<C: Arithmetization<P::ScalarField>>(
        srs: &Self::UniversalSRS,
        vk_id: Option<VerificationKeyId>,
        circuit: &C,
        blind: bool,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), Self::Error> {
        // Make sure the SRS can support the circuit (with possible hiding degree of 2 for zk)
        let domain_size = circuit.eval_domain_size()?;
        let srs_size = circuit.srs_size(blind)?;
        let num_inputs = circuit.num_inputs();

        // 1. Compute selector and permutation polynomials.
        let selectors_polys = circuit.compute_selector_polynomials()?;
        let sigma_polys = circuit.compute_extended_permutation_polynomials()?;

        // Compute Plookup proving key if support lookup.
        let plookup_pk = if circuit.support_lookup() {
            let range_table_poly = circuit.compute_range_table_polynomial()?;
            let key_table_poly = circuit.compute_key_table_polynomial()?;
            let table_dom_sep_poly = circuit.compute_table_dom_sep_polynomial()?;
            let q_dom_sep_poly = circuit.compute_q_dom_sep_polynomial()?;
            Some(PlookupProvingKey {
                range_table_poly,
                key_table_poly,
                table_dom_sep_poly,
                q_dom_sep_poly,
            })
        } else {
            None
        };

        // 2. Compute VerifyingKey
        let (commit_key, open_key) = srs.trim(srs_size)?;

        let selector_comms = parallelizable_slice_iter(&selectors_polys)
            .map(|poly| PCS::commit(&commit_key, poly).map_err(PlonkError::PCSError))
            .collect::<Result<Vec<_>, PlonkError>>()?
            .into_iter()
            .collect();
        let sigma_comms = parallelizable_slice_iter(&sigma_polys)
            .map(|poly| PCS::commit(&commit_key, poly).map_err(PlonkError::PCSError))
            .collect::<Result<Vec<_>, PlonkError>>()?
            .into_iter()
            .collect();

        // Compute Plookup verifying key if support lookup.
        let plookup_vk = match circuit.support_lookup() {
            false => None,
            true => Some(PlookupVerifyingKey {
                range_table_comm: PCS::commit(
                    &commit_key,
                    &plookup_pk.as_ref().unwrap().range_table_poly,
                )?,
                key_table_comm: PCS::commit(
                    &commit_key,
                    &plookup_pk.as_ref().unwrap().key_table_poly,
                )?,
                table_dom_sep_comm: PCS::commit(
                    &commit_key,
                    &plookup_pk.as_ref().unwrap().table_dom_sep_poly,
                )?,
                q_dom_sep_comm: PCS::commit(
                    &commit_key,
                    &plookup_pk.as_ref().unwrap().q_dom_sep_poly,
                )?,
            }),
        };

        let vk = VerifyingKey {
            domain_size,
            num_inputs,
            selector_comms,
            sigma_comms,
            k: compute_coset_representatives(circuit.num_wire_types(), Some(domain_size)),
            open_key,
            plookup_vk,
            is_merged: false,
            id: vk_id,
        };

        // Compute ProvingKey (which includes the VerifyingKey)
        let pk = ProvingKey {
            sigmas: sigma_polys,
            selectors: selectors_polys,
            commit_key,
            vk: vk.clone(),
            plookup_pk,
        };

        Ok((pk, vk))
    }

    /// Compute a Plonk proof.
    /// Refer to Sec 8.4 of <https://eprint.iacr.org/2019/953.pdf>
    fn prove<C, R, T>(
        rng: &mut R,
        circuit: &C,
        prove_key: &Self::ProvingKey,
        extra_transcript_init_msg: Option<Vec<u8>>,
        blind: bool,
    ) -> Result<Self::Proof, Self::Error>
    where
        C: Arithmetization<P::ScalarField>,
        R: CryptoRng + RngCore,
        T: Transcript,
    {
        let (proof, _, _) = Self::prove_internal::<_, _, T>(
            rng,
            circuit,
            prove_key,
            extra_transcript_init_msg,
            blind,
        )?;
        Ok(Proof {
            wires_poly_comms: proof.wires_poly_comms.clone(),
            prod_perm_poly_comm: proof.prod_perm_poly_comm,
            split_quot_poly_comms: proof.split_quot_poly_comms,
            opening_proof: proof.opening_proof,
            q_comm: proof.q_comm,
            poly_evals: proof.poly_evals.clone(),
            plookup_proof: proof.plookup_proof,
        })
    }

    fn verify<T>(
        verify_key: &Self::VerifyingKey,
        public_input: &[P::ScalarField],
        proof: &Self::Proof,
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<(), Self::Error>
    where
        T: Transcript,
    {
        Self::verify_proof::<T>(verify_key, public_input, proof, extra_transcript_init_msg)
    }
}

impl<PCS, F, P> UniversalRecursiveSNARK<PCS> for FFTPlonk<PCS>
where
    F: RescueParameter,
    P: HasTEForm<BaseField = F>,
    P::ScalarField: EmulationConfig<F>,
    PCS: PolynomialCommitmentScheme<
        Evaluation = P::ScalarField,
        Polynomial = DensePolynomial<P::ScalarField>,
        Point = P::ScalarField,
        Commitment = Affine<P>,
        Proof: TranscriptVisitor,
    >,
    PCS::SRS: StructuredReferenceString<Item = PCS::Commitment>,
{
    type RecursiveProof = Proof<PCS>;

    fn recursive_prove<C, R, T>(
        rng: &mut R,
        circuit: &C,
        prove_key: &Self::ProvingKey,
        extra_transcript_init_msg: Option<Vec<u8>>,
        blind: bool,
    ) -> Result<RecursiveOutput<PCS, Self, T>, Self::Error>
    where
        Self: Sized,
        PCS: jf_primitives::pcs::Accumulation,
        C: Arithmetization<
            <<PCS as PolynomialCommitmentScheme>::Commitment as ark_ec::AffineRepr>::ScalarField,
        >,
        R: CryptoRng + RngCore,
        T: Transcript + ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize,
    {
        let (proof, transcript) = Self::recursive_prove_internal::<_, _, T>(
            rng,
            circuit,
            prove_key,
            extra_transcript_init_msg,
            blind,
        )?;

        let pi_hash = circuit.public_input()?[0];

        Ok(RecursiveOutput::new(proof, pi_hash, transcript))
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::{
        errors::PlonkError,
        nightfall::{
            hops::univariate_ipa::UnivariateIpaPCS,
            ipa_snark::FFTPlonk,
            ipa_structs::{
                eval_merged_lookup_witness, eval_merged_table, Challenges, Oracles, ProvingKey,
            },
        },
        proof_system::UniversalSNARK,
        transcript::{rescue::RescueTranscript, standard::StandardTranscript, Transcript},
        PlonkType,
    };
    use ark_bls12_377::{g1::Config as Config377, Bls12_377, Fq as Fq377};
    use ark_bls12_381::{Bls12_381, Fq as Fq381};
    use ark_bn254::{g1::Config as BnConfig, Bn254, Fq as Fq254};
    use ark_bw6_761::{Fq as Fq761, BW6_761};
    use ark_ec::{
        short_weierstrass::{Affine, SWCurveConfig},
        AffineRepr,
    };
    use ark_ff::{One, PrimeField, Zero};
    use ark_poly::{
        univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Polynomial,
        Radix2EvaluationDomain,
    };

    use ark_std::{format, string::ToString, vec, vec::Vec};
    use itertools::izip;
    use core::ops::{Mul, Neg};
    use jf_primitives::{
        pcs::{prelude::UnivariateKzgPCS, PolynomialCommitmentScheme},
        rescue::{sponge::RescueCRHF, RescueParameter},
    };
    use jf_relation::{
        constants::GATE_WIDTH, gadgets::ecc::HasTEForm, Arithmetization, Circuit, PlonkCircuit,
    };
    use jf_utils::test_rng;

    // Different `m`s lead to different circuits.
    // Different `a0`s lead to different witness values.
    // For UltraPlonk circuits, `a0` should be less than or equal to `m+1`
    pub(crate) fn gen_circuit_for_test<F: RescueParameter>(
        m: usize,
        a0: usize,
        plonk_type: PlonkType,
        is_recursive: bool,
    ) -> Result<PlonkCircuit<F>, PlonkError> {
        let range_bit_len = 5;
        let mut cs: PlonkCircuit<F> = match plonk_type {
            PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
            PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(range_bit_len),
        };
        // Create variables
        let mut a = vec![];
        for i in a0..(a0 + 4 * m) {
            a.push(cs.create_variable(F::from(i as u64))?);
        }
        let b = [
            cs.create_public_variable(F::from(m as u64 * 2))?,
            cs.create_public_variable(F::from(a0 as u64 * 2 + m as u64 * 4 - 1))?,
        ];
        let c = cs.create_public_variable(
            (cs.witness(b[1])? + cs.witness(a[0])?) * (cs.witness(b[1])? - cs.witness(a[0])?),
        )?;

        // Create gates:
        // 1. a0 + ... + a_{4*m-1} = b0 * b1
        // 2. (b1 + a0) * (b1 - a0) = c
        // 3. b0 = 2 * m
        let mut acc = cs.zero();
        a.iter().for_each(|&elem| acc = cs.add(acc, elem).unwrap());
        let b_mul = cs.mul(b[0], b[1])?;
        cs.enforce_equal(acc, b_mul)?;
        let b1_plus_a0 = cs.add(b[1], a[0])?;
        let b1_minus_a0 = cs.sub(b[1], a[0])?;
        cs.mul_gate(b1_plus_a0, b1_minus_a0, c)?;
        cs.enforce_constant(b[0], F::from(m as u64 * 2))?;

        if plonk_type == PlonkType::UltraPlonk {
            // Create range gates
            // 1. range_table = {0, 1, ..., 31}
            // 2. a_i \in range_table for i = 0..m-1
            // 3. b0 \in range_table
            for &var in a.iter().take(m) {
                cs.add_range_check_variable(var)?;
            }
            cs.add_range_check_variable(b[0])?;

            // Create variable table lookup gates
            // 1. table = [(a0, a2), (a1, a3), (b0, a0)]
            let table_vars = [(a[0], a[2]), (a[1], a[3]), (b[0], a[0])];
            // 2. lookup_witness = [(1, a0+1, a0+3), (2, 2m, a0)]
            let key0 = cs.one();
            let key1 = cs.create_variable(F::from(2u8))?;
            let two_m = cs.create_public_variable(F::from(m as u64 * 2))?;
            let a1 = cs.add_constant(a[0], &F::one())?;
            let a3 = cs.add_constant(a[0], &F::from(3u8))?;
            let lookup_vars = [(key0, a1, a3), (key1, two_m, a[0])];
            cs.create_table_and_lookup_variables(&lookup_vars, &table_vars)?;
        }

        // Finalize the circuit.
        if is_recursive {
            cs.finalize_for_recursive_arithmetization::<RescueCRHF<F>>()?;
        } else {
            cs.finalize_for_arithmetization()?;
        }
        Ok(cs)
    }

    #[test]
    fn test_preprocessing() -> Result<(), PlonkError> {
        for (plonk_type, vk_id) in [PlonkType::TurboPlonk, PlonkType::UltraPlonk].iter().zip(
            [
                None,
                Some(VerificationKeyId::Client),
                Some(VerificationKeyId::Deposit),
            ]
            .iter(),
        ) {
            test_preprocessing_helper::<UnivariateKzgPCS<Bn254>, Fq254, _>(*plonk_type, *vk_id)?;
            test_preprocessing_helper::<UnivariateKzgPCS<Bls12_377>, Fq377, _>(
                *plonk_type,
                *vk_id,
            )?;
            test_preprocessing_helper::<UnivariateKzgPCS<Bls12_381>, Fq381, _>(
                *plonk_type,
                *vk_id,
            )?;
            test_preprocessing_helper::<UnivariateKzgPCS<BW6_761>, Fq761, _>(*plonk_type, *vk_id)?;
        }
        Ok(())
    }
    fn test_preprocessing_helper<PCS, F, E>(
        plonk_type: PlonkType,
        vk_id: Option<VerificationKeyId>,
    ) -> Result<(), PlonkError>
    where
        PCS: PolynomialCommitmentScheme<
            Evaluation = E::ScalarField,
            Polynomial = DensePolynomial<E::ScalarField>,
            Point = E::ScalarField,
            Commitment = Affine<E>,
        >,
        PCS::SRS: StructuredReferenceString<Item = PCS::Commitment>,
        F: RescueParameter,
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
    {
        let rng = &mut jf_utils::test_rng();
        let circuit = gen_circuit_for_test(5, 6, plonk_type, false)?;
        let domain_size = circuit.eval_domain_size()?;
        let num_inputs = circuit.num_inputs();
        let selectors = circuit.compute_selector_polynomials()?;
        let sigmas = circuit.compute_extended_permutation_polynomials()?;

        let max_degree = 64;
        let srs = FFTPlonk::<PCS>::universal_setup_for_testing(max_degree, rng)?;
        let (pk, vk) = FFTPlonk::<PCS>::preprocess(&srs, vk_id, &circuit, false)?;

        // check proving key
        assert_eq!(pk.selectors, selectors);
        assert_eq!(pk.sigmas, sigmas);
        assert_eq!(pk.domain_size(), domain_size);
        assert_eq!(pk.num_inputs(), num_inputs);
        let num_wire_types = GATE_WIDTH
            + 1
            + match plonk_type {
                PlonkType::TurboPlonk => 0,
                PlonkType::UltraPlonk => 1,
            };
        assert_eq!(pk.sigmas.len(), num_wire_types);
        // check plookup proving key
        if plonk_type == PlonkType::UltraPlonk {
            let range_table_poly = circuit.compute_range_table_polynomial()?;
            assert_eq!(
                pk.plookup_pk.as_ref().unwrap().range_table_poly,
                range_table_poly
            );

            let key_table_poly = circuit.compute_key_table_polynomial()?;
            assert_eq!(
                pk.plookup_pk.as_ref().unwrap().key_table_poly,
                key_table_poly
            );
        }

        // check verifying key
        assert_eq!(vk.domain_size, domain_size);
        assert_eq!(vk.num_inputs, num_inputs);
        assert_eq!(vk.selector_comms.len(), selectors.len());
        assert_eq!(vk.sigma_comms.len(), sigmas.len());
        assert_eq!(vk.sigma_comms.len(), num_wire_types);
        selectors
            .iter()
            .zip(vk.selector_comms.iter())
            .for_each(|(p, &p_comm)| {
                let expected_comm = PCS::commit(&pk.commit_key, p).unwrap();
                assert_eq!(expected_comm, p_comm);
            });
        sigmas
            .iter()
            .zip(vk.sigma_comms.iter())
            .for_each(|(p, &p_comm)| {
                let expected_comm = PCS::commit(&pk.commit_key, p).unwrap();
                assert_eq!(expected_comm, p_comm);
            });
        // check plookup verification key
        if plonk_type == PlonkType::UltraPlonk {
            let expected_comm = PCS::commit(
                &pk.commit_key,
                &pk.plookup_pk.as_ref().unwrap().range_table_poly,
            )
            .unwrap();
            assert_eq!(
                expected_comm,
                vk.plookup_vk.as_ref().unwrap().range_table_comm
            );

            let expected_comm = PCS::commit(
                &pk.commit_key,
                &pk.plookup_pk.as_ref().unwrap().key_table_poly,
            )
            .unwrap();
            assert_eq!(
                expected_comm,
                vk.plookup_vk.as_ref().unwrap().key_table_comm
            );
        }

        Ok(())
    }

    #[test]
    fn test_plonk_proof_system() -> Result<(), PlonkError> {
        for (plonk_type, vk_id) in [PlonkType::TurboPlonk, PlonkType::UltraPlonk].iter().zip(
            [
                None,
                Some(VerificationKeyId::Client),
                Some(VerificationKeyId::Deposit),
            ]
            .iter(),
        ) {
            // merlin transcripts
            test_plonk_proof_system_helper::<
                BnConfig,
                Fq254,
                UnivariateKzgPCS<Bn254>,
                StandardTranscript,
            >(*plonk_type, *vk_id)?;
            test_plonk_proof_system_helper::<
                Config377,
                Fq377,
                UnivariateIpaPCS<Bls12_377>,
                StandardTranscript,
            >(*plonk_type, *vk_id)?;
            // rescue transcripts
            test_plonk_proof_system_helper::<
                BnConfig,
                Fq254,
                UnivariateKzgPCS<Bn254>,
                RescueTranscript<Fq254>,
            >(*plonk_type, *vk_id)?;
            test_plonk_proof_system_helper::<
                Config377,
                Fq377,
                UnivariateIpaPCS<Bls12_377>,
                RescueTranscript<Fq377>,
            >(*plonk_type, *vk_id)?;

            test_recursive_plonk_proof_system_helper::<
                BnConfig,
                Fq254,
                UnivariateKzgPCS<Bn254>,
                RescueTranscript<Fq254>,
            >(*plonk_type, *vk_id)?;
        }

        Ok(())
    }

    fn test_plonk_proof_system_helper<E, F, PCS, T>(
        plonk_type: PlonkType,
        vk_id: Option<VerificationKeyId>,
    ) -> Result<(), PlonkError>
    where
        PCS: PolynomialCommitmentScheme<
            Evaluation = E::ScalarField,
            Polynomial = DensePolynomial<E::ScalarField>,
            Point = E::ScalarField,
            Commitment = Affine<E>,
        >,
        PCS::SRS: StructuredReferenceString<Item = PCS::Commitment>,
        F: RescueParameter,
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
        T: Transcript,
    {
        for blind in [true, false] {
            // 1. Simulate universal setup
            let rng = &mut test_rng();
            let n = 64;
            let max_degree = n + 2 * blind as usize;
            let srs = FFTPlonk::<PCS>::universal_setup_for_testing(max_degree, rng)?;

            // 2. Create circuits
            let circuits = (0..6)
                .map(|i| {
                    let m = 2 + i / 3;
                    let a0 = 1 + i % 3;
                    gen_circuit_for_test(m, a0, plonk_type, false)
                })
                .collect::<Result<Vec<_>, PlonkError>>()?;
            // 3. Preprocessing
            let (pk1, vk1) = <FFTPlonk<PCS> as UniversalSNARK<PCS>>::preprocess(
                &srs,
                vk_id,
                &circuits[0],
                blind,
            )?;
            let (pk2, vk2) = FFTPlonk::<PCS>::preprocess(&srs, vk_id, &circuits[3], blind)?;
            // 4. Proving
            let mut proofs = vec![];
            let mut extra_msgs = vec![];
            for (i, cs) in circuits.iter().enumerate() {
                let pk_ref = if i < 3 { &pk1 } else { &pk2 };
                let extra_msg = if i % 2 == 0 {
                    None
                } else {
                    Some(format!("extra message: {}", i).into_bytes())
                };
                proofs.push(
                    FFTPlonk::<PCS>::prove::<_, _, T>(rng, cs, pk_ref, extra_msg.clone(), blind)
                        .unwrap(),
                );
                extra_msgs.push(extra_msg);
            }

            // 5. Verification
            let public_inputs: Vec<Vec<E::ScalarField>> = circuits
                .iter()
                .map(|cs| cs.public_input())
                .collect::<Result<Vec<Vec<E::ScalarField>>, _>>()?;
            for (i, proof) in proofs.iter().enumerate() {
                let vk_ref = if i < 3 { &vk1 } else { &vk2 };
                assert!(FFTPlonk::<PCS>::verify::<T>(
                    vk_ref,
                    &public_inputs[i],
                    proof,
                    extra_msgs[i].clone(),
                )
                .is_ok());
                // Inconsistent proof should fail the verification.
                let mut bad_pub_input = public_inputs[i].clone();
                bad_pub_input[0] = E::ScalarField::from(0u8);
                assert!(FFTPlonk::<PCS>::verify::<T>(
                    vk_ref,
                    &bad_pub_input,
                    proof,
                    extra_msgs[i].clone(),
                )
                .is_err());
                // Incorrect extra transcript message should fail
                assert!(FFTPlonk::<PCS>::verify::<T>(
                    vk_ref,
                    &bad_pub_input,
                    proof,
                    Some("wrong message".to_string().into_bytes()),
                )
                .is_err());

                // Incorrect proof [W_z] = 0, [W_z*g] = 0
                // attack against some vulnerable implementation described in:
                // https://cryptosubtlety.medium.com/00-8d4adcf4d255
                let mut bad_proof = proof.clone();
                bad_proof.opening_proof = PCS::Proof::default();

                assert!(FFTPlonk::<PCS>::verify::<T>(
                    vk_ref,
                    &public_inputs[i],
                    &bad_proof,
                    extra_msgs[i].clone(),
                )
                .is_err());
            }
        }
        Ok(())
    }

    fn test_recursive_plonk_proof_system_helper<E, F, PCS, T>(
        plonk_type: PlonkType,
        vk_id: Option<VerificationKeyId>,
    ) -> Result<(), PlonkError>
    where
        PCS: Accumulation<
            Evaluation = E::ScalarField,
            Polynomial = DensePolynomial<E::ScalarField>,
            Point = E::ScalarField,
            Commitment = Affine<E>,
            Proof: TranscriptVisitor,
        >,
        PCS::SRS: StructuredReferenceString<Item = PCS::Commitment>,
        F: RescueParameter,
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
        T: Transcript + ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize,
    {
        for blind in [true, false] {
            // 1. Simulate universal setup
            let rng = &mut test_rng();
            let n = 64;
            let max_degree = n + 2 * blind as usize;
            let srs = FFTPlonk::<PCS>::universal_setup_for_testing(max_degree, rng)?;

            // 2. Create circuits
            let circuits = (0..6)
                .map(|i| {
                    let m = 2 + i / 3;
                    let a0 = 1 + i % 3;
                    gen_circuit_for_test(m, a0, plonk_type, true)
                })
                .collect::<Result<Vec<_>, PlonkError>>()?;
            // 3. Preprocessing
            let (pk1, vk1) = <FFTPlonk<PCS> as UniversalSNARK<PCS>>::preprocess(
                &srs,
                vk_id,
                &circuits[0],
                blind,
            )?;
            let (pk2, vk2) = FFTPlonk::<PCS>::preprocess(&srs, vk_id, &circuits[3], blind)?;
            // 4. Proving
            let mut proofs = vec![];
            let mut extra_msgs = vec![];
            let mut public_inputs = vec![];
            for (i, cs) in circuits.iter().enumerate() {
                let pk_ref = if i < 3 { &pk1 } else { &pk2 };
                let extra_msg = if i % 2 == 0 {
                    None
                } else {
                    Some(format!("extra message: {}", i).into_bytes())
                };
                public_inputs.push(cs.public_input().unwrap());
                proofs.push(
                    FFTPlonk::<PCS>::recursive_prove::<_, _, T>(
                        rng,
                        cs,
                        pk_ref,
                        extra_msg.clone(),
                        blind,
                    )
                    .unwrap(),
                );
                extra_msgs.push(extra_msg);
            }

            // 5. Verification
            for (i, proof) in proofs.iter().enumerate() {
                let vk_ref = if i < 3 { &vk1 } else { &vk2 };
                assert!(FFTPlonk::<PCS>::verify_recursive_proof::<T>(
                    vk_ref,
                    proof,
                    extra_msgs[i].clone(),
                )
                .is_ok());
                // Inconsistent proof should fail the verification.
                let bad_proof = RecursiveOutput {
                    proof: proof.proof.clone(),
                    pi_hash: E::ScalarField::zero(),
                    transcript: T::new_transcript(b"bad_transcript"),
                };

                assert!(FFTPlonk::<PCS>::verify_recursive_proof::<T>(
                    vk_ref,
                    &bad_proof,
                    extra_msgs[i].clone(),
                )
                .is_err());

                // Incorrect proof [W_z] = 0, [W_z*g] = 0
                // attack against some vulnerable implementation described in:
                // https://cryptosubtlety.medium.com/00-8d4adcf4d255
                let mut bad_proof = proof.proof.clone();
                bad_proof.opening_proof = PCS::Proof::default();
                let bad_proof = RecursiveOutput::new(
                    bad_proof,
                    proof.pi_hash,
                    T::new_transcript(b"bad_transcript"),
                );

                assert!(FFTPlonk::<PCS>::verify_recursive_proof::<T>(
                    vk_ref,
                    &bad_proof,
                    extra_msgs[i].clone(),
                )
                .is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn test_inconsistent_pub_input_len() -> Result<(), PlonkError> {
        for (plonk_type, vk_id) in [PlonkType::TurboPlonk, PlonkType::UltraPlonk].iter().zip(
            [
                None,
                Some(VerificationKeyId::Client),
                Some(VerificationKeyId::Deposit),
            ]
            .iter(),
        ) {
            // merlin transcripts
            test_inconsistent_pub_input_len_helper::<
                Config377,
                Fq377,
                UnivariateIpaPCS<Bls12_377>,
                StandardTranscript,
            >(*plonk_type, *vk_id)?;
            // rescue transcripts
            test_inconsistent_pub_input_len_helper::<
                Config377,
                Fq377,
                UnivariateIpaPCS<Bls12_377>,
                RescueTranscript<Fq377>,
            >(*plonk_type, *vk_id)?;
        }

        Ok(())
    }

    fn test_inconsistent_pub_input_len_helper<E, F, PCS, T>(
        plonk_type: PlonkType,
        vk_id: Option<VerificationKeyId>,
    ) -> Result<(), PlonkError>
    where
        PCS: PolynomialCommitmentScheme<
            Evaluation = E::ScalarField,
            Polynomial = DensePolynomial<E::ScalarField>,
            Point = E::ScalarField,
            Commitment = Affine<E>,
        >,
        PCS::SRS: StructuredReferenceString<Item = PCS::Commitment>,
        F: RescueParameter,
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F>,
        T: Transcript,
    {
        for blind in [true, false] {
            // 1. Simulate universal setup
            let rng = &mut test_rng();

            // 2. Create circuits
            let mut cs1: PlonkCircuit<E::ScalarField> = match plonk_type {
                PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
                PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(2),
            };
            let var = cs1.create_variable(E::ScalarField::from(1u8))?;
            cs1.enforce_constant(var, E::ScalarField::from(1u8))?;
            ark_std::println!(
                "Number of gates in cs1 pre-finalization: {}",
                cs1.num_gates()
            );
            cs1.finalize_for_arithmetization()?;
            ark_std::println!(
                "Number of gates in cs1 post-finalization: {}",
                cs1.num_gates()
            );
            ark_std::println!("cs1 eval_domain_size: {}", cs1.eval_domain_size()?);
            let mut cs2: PlonkCircuit<E::ScalarField> = match plonk_type {
                PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
                PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(2),
            };
            cs2.create_public_variable(E::ScalarField::from(1u8))?;
            ark_std::println!(
                "Number of gates in cs2 pre-finalization: {}",
                cs2.num_gates()
            );
            cs2.finalize_for_arithmetization()?;
            ark_std::println!(
                "Number of gates in cs2 post-finalization: {}",
                cs2.num_gates()
            );
            ark_std::println!("cs2 eval_domain_size: {}", cs2.eval_domain_size()?);

            // 3. Preprocessing
            let size_one = cs1.srs_size(blind)?;
            let size_two = cs2.srs_size(blind)?;
            let size = ark_std::cmp::max(size_one, size_two);
            let srs = FFTPlonk::<PCS>::universal_setup_for_testing(size, rng)?;
            let (pk1, vk1) = FFTPlonk::<PCS>::preprocess(&srs, vk_id, &cs1, blind)?;
            let (pk2, vk2) = FFTPlonk::<PCS>::preprocess(&srs, vk_id, &cs2, blind)?;

            // 4. Proving
            assert!(FFTPlonk::<PCS>::prove::<_, _, T>(rng, &cs2, &pk1, None, blind).is_err());
            let proof2 = FFTPlonk::<PCS>::prove::<_, _, T>(rng, &cs2, &pk2, None, blind)?;

            // 5. Verification
            assert!(FFTPlonk::<PCS>::verify::<T>(
                &vk2,
                &[E::ScalarField::from(1u8)],
                &proof2,
                None,
            )
            .is_ok());
            // wrong verification key
            assert!(FFTPlonk::<PCS>::verify::<T>(
                &vk1,
                &[E::ScalarField::from(1u8)],
                &proof2,
                None,
            )
            .is_err());
            // wrong public input
            assert!(FFTPlonk::<PCS>::verify::<T>(&vk2, &[], &proof2, None).is_err());
        }
        Ok(())
    }

    #[test]
    fn test_plonk_prover_polynomials() -> Result<(), PlonkError> {
        for (plonk_type, vk_id, blind) in izip!(
            [PlonkType::TurboPlonk, PlonkType::UltraPlonk],
            [
                None,
                Some(VerificationKeyId::Client),
                Some(VerificationKeyId::Deposit),
            ],
            [true, false],
        ) {
            // merlin transcripts
            test_plonk_prover_polynomials_helper::<
                Config377,
                Fq377,
                UnivariateIpaPCS<Bls12_377>,
                StandardTranscript,
            >(plonk_type, vk_id, blind)?;

            // rescue transcripts
            test_plonk_prover_polynomials_helper::<
                Config377,
                Fq377,
                UnivariateIpaPCS<Bls12_377>,
                RescueTranscript<Fq377>,
            >(plonk_type, vk_id, blind)?;
        }

        Ok(())
    }

    fn test_plonk_prover_polynomials_helper<E, F, PCS, T>(
        plonk_type: PlonkType,
        vk_id: Option<VerificationKeyId>,
        blind: bool,
    ) -> Result<(), PlonkError>
    where
        PCS: PolynomialCommitmentScheme<
            Evaluation = E::ScalarField,
            Polynomial = DensePolynomial<E::ScalarField>,
            Point = E::ScalarField,
            Commitment = Affine<E>,
        >,
        PCS::SRS: StructuredReferenceString<Item = PCS::Commitment>,
        F: RescueParameter,
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F> + RescueParameter,
        T: Transcript,
    {
        // 1. Simulate universal setup
        let rng = &mut test_rng();
        let n = 64;
        let max_degree = n + 2 * blind as usize;
        let srs = <FFTPlonk<PCS> as UniversalSNARK<PCS>>::universal_setup_for_testing(
            max_degree, rng,
        )?;

        // 2. Create the circuit
        let circuit = gen_circuit_for_test(10, 3, plonk_type, false)?;
        assert!(circuit.num_gates() <= n);

        // 3. Preprocessing
        let (pk, _) = FFTPlonk::<PCS>::preprocess(&srs, vk_id, &circuit, blind)?;

        // 4. Proving
        let (_, oracles, challenges) =
            FFTPlonk::<PCS>::prove_internal::<_, _, T>(rng, &circuit, &pk, None, blind)?;

        // 5. Check that the targeted polynomials evaluate to zero on the vanishing set.
        check_plonk_prover_polynomials(plonk_type, &oracles, &pk, &challenges)?;
        Ok(())
    }

    fn check_plonk_prover_polynomials<PCS, E>(
        plonk_type: PlonkType,
        oracles: &Oracles<E::ScalarField>,
        pk: &ProvingKey<PCS>,
        challenges: &Challenges<E::ScalarField>,
    ) -> Result<(), PlonkError>
    where
        PCS: PolynomialCommitmentScheme<
            Evaluation = E::ScalarField,
            Polynomial = DensePolynomial<E::ScalarField>,
            Point = E::ScalarField,
        >,
        PCS::Commitment:
            AffineRepr<Config = E, BaseField = E::BaseField, ScalarField = E::ScalarField>,
        E: SWCurveConfig,
        E::BaseField: PrimeField,
        E::ScalarField: PrimeField + EmulationConfig<E::BaseField>,
    {
        check_circuit_polynomial_on_vanishing_set(oracles, pk)?;
        check_perm_polynomials_on_vanishing_set(oracles, pk, challenges)?;
        if plonk_type == PlonkType::UltraPlonk {
            check_lookup_polynomials_on_vanishing_set(oracles, pk, challenges)?;
        }

        Ok(())
    }

    fn check_circuit_polynomial_on_vanishing_set<PCS, E>(
        oracles: &Oracles<E::ScalarField>,
        pk: &ProvingKey<PCS>,
    ) -> Result<(), PlonkError>
    where
        PCS: PolynomialCommitmentScheme<
            Evaluation = E::ScalarField,
            Polynomial = DensePolynomial<E::ScalarField>,
            Point = E::ScalarField,
        >,
        PCS::Commitment:
            AffineRepr<Config = E, BaseField = E::BaseField, ScalarField = E::ScalarField>,
        E: SWCurveConfig,
    {
        let q_lc: Vec<&DensePolynomial<E::ScalarField>> =
            (0..GATE_WIDTH).map(|j| &pk.selectors[j]).collect();
        let q_mul: Vec<&DensePolynomial<E::ScalarField>> = (GATE_WIDTH..GATE_WIDTH + 2)
            .map(|j| &pk.selectors[j])
            .collect();
        let q_hash: Vec<&DensePolynomial<E::ScalarField>> = (GATE_WIDTH + 2..2 * GATE_WIDTH + 2)
            .map(|j| &pk.selectors[j])
            .collect();
        let q_o = &pk.selectors[2 * GATE_WIDTH + 2];
        let q_c = &pk.selectors[2 * GATE_WIDTH + 3];
        let q_ecc = &pk.selectors[2 * GATE_WIDTH + 4];
        let circuit_poly = q_c
            + &oracles.pub_inp_poly
            + oracles.wire_polys[0].mul(q_lc[0])
            + oracles.wire_polys[1].mul(q_lc[1])
            + oracles.wire_polys[2].mul(q_lc[2])
            + oracles.wire_polys[3].mul(q_lc[3])
            + oracles.wire_polys[0]
                .mul(&oracles.wire_polys[1])
                .mul(q_mul[0])
            + oracles.wire_polys[2]
                .mul(&oracles.wire_polys[3])
                .mul(q_mul[1])
            + oracles.wire_polys[0]
                .mul(&oracles.wire_polys[1])
                .mul(&oracles.wire_polys[2])
                .mul(&oracles.wire_polys[3])
                .mul(&oracles.wire_polys[4])
                .mul(q_ecc)
            + oracles.wire_polys[0]
                .mul(&oracles.wire_polys[0])
                .mul(&oracles.wire_polys[0])
                .mul(&oracles.wire_polys[0])
                .mul(&oracles.wire_polys[0])
                .mul(q_hash[0])
            + oracles.wire_polys[1]
                .mul(&oracles.wire_polys[1])
                .mul(&oracles.wire_polys[1])
                .mul(&oracles.wire_polys[1])
                .mul(&oracles.wire_polys[1])
                .mul(q_hash[1])
            + oracles.wire_polys[2]
                .mul(&oracles.wire_polys[2])
                .mul(&oracles.wire_polys[2])
                .mul(&oracles.wire_polys[2])
                .mul(&oracles.wire_polys[2])
                .mul(q_hash[2])
            + oracles.wire_polys[3]
                .mul(&oracles.wire_polys[3])
                .mul(&oracles.wire_polys[3])
                .mul(&oracles.wire_polys[3])
                .mul(&oracles.wire_polys[3])
                .mul(q_hash[3])
            + oracles.wire_polys[4].mul(q_o).neg();

        // check that the polynomial evaluates to zero on the vanishing set
        let domain = Radix2EvaluationDomain::<E::ScalarField>::new(pk.domain_size())
            .ok_or(PlonkError::DomainCreationError)?;
        for i in 0..domain.size() {
            assert_eq!(
                circuit_poly.evaluate(&domain.element(i)),
                E::ScalarField::zero()
            );
        }

        Ok(())
    }

    fn check_perm_polynomials_on_vanishing_set<PCS, E>(
        oracles: &Oracles<E::ScalarField>,
        pk: &ProvingKey<PCS>,
        challenges: &Challenges<E::ScalarField>,
    ) -> Result<(), PlonkError>
    where
        PCS: PolynomialCommitmentScheme<
            Evaluation = E::ScalarField,
            Polynomial = DensePolynomial<E::ScalarField>,
            Point = E::ScalarField,
        >,
        PCS::Commitment:
            AffineRepr<Config = E, BaseField = E::BaseField, ScalarField = E::ScalarField>,
        E: SWCurveConfig,
        E::BaseField: PrimeField,
        E::ScalarField: PrimeField + EmulationConfig<E::BaseField>,
    {
        let beta = challenges.beta;
        let gamma = challenges.gamma;

        // check that \prod_i [w_i(X) + beta * k_i * X + gamma] * z(X) = \prod_i [w_i(X)
        // + beta * sigma_i(X) + gamma] * z(wX) on the vanishing set
        let one_poly = DensePolynomial::from_coefficients_vec(vec![E::ScalarField::one()]);
        let poly_1 = oracles
            .wire_polys
            .iter()
            .enumerate()
            .fold(one_poly.clone(), |acc, (j, w)| {
                let poly =
                    &DensePolynomial::from_coefficients_vec(vec![gamma, beta * pk.k()[j]]) + w;
                acc.mul(&poly)
            });
        let poly_2 =
            oracles
                .wire_polys
                .iter()
                .zip(pk.sigmas.iter())
                .fold(one_poly, |acc, (w, sigma)| {
                    let poly = w.clone()
                        + sigma.mul(beta)
                        + DensePolynomial::from_coefficients_vec(vec![gamma]);
                    acc.mul(&poly)
                });

        let domain = Radix2EvaluationDomain::<E::ScalarField>::new(pk.domain_size())
            .ok_or(PlonkError::DomainCreationError)?;
        for i in 0..domain.size() {
            let point = domain.element(i);
            let eval_1 = poly_1.evaluate(&point) * oracles.prod_perm_poly.evaluate(&point);
            let eval_2 = poly_2.evaluate(&point)
                * oracles.prod_perm_poly.evaluate(&(point * domain.group_gen));
            assert_eq!(eval_1, eval_2);
        }

        // check z(X) = 1 at point 1
        assert_eq!(
            oracles.prod_perm_poly.evaluate(&domain.element(0)),
            E::ScalarField::from(1u64)
        );

        Ok(())
    }

    fn check_lookup_polynomials_on_vanishing_set<PCS, E>(
        oracles: &Oracles<E::ScalarField>,
        pk: &ProvingKey<PCS>,
        challenges: &Challenges<E::ScalarField>,
    ) -> Result<(), PlonkError>
    where
        PCS: PolynomialCommitmentScheme<
            Evaluation = E::ScalarField,
            Polynomial = DensePolynomial<E::ScalarField>,
            Point = E::ScalarField,
        >,
        PCS::Commitment:
            AffineRepr<Config = E, BaseField = E::BaseField, ScalarField = E::ScalarField>,
        E: SWCurveConfig,
        E::BaseField: PrimeField,
        E::ScalarField: PrimeField + EmulationConfig<E::BaseField>,
    {
        let beta = challenges.beta;
        let gamma = challenges.gamma;
        let n = pk.domain_size();
        let domain = Radix2EvaluationDomain::<E::ScalarField>::new(n)
            .ok_or(PlonkError::DomainCreationError)?;
        let prod_poly = &oracles.plookup_oracles.prod_lookup_poly;
        let h_polys = &oracles.plookup_oracles.h_polys;

        // check z(X) = 1 at point 1
        assert_eq!(
            prod_poly.evaluate(&domain.element(0)),
            E::ScalarField::one()
        );

        // check z(X) = 1 at point w^{n-1}
        assert_eq!(
            prod_poly.evaluate(&domain.element(n - 1)),
            E::ScalarField::one()
        );

        // check h1(X) = h2(w * X) at point w^{n-1}
        assert_eq!(
            h_polys[0].evaluate(&domain.element(n - 1)),
            h_polys[1].evaluate(&domain.element(0))
        );

        // check z(X) *
        //      (1+beta) * (gamma + merged_lookup_wire(X)) *
        //      (gamma(1+beta) + merged_table(X) + beta * merged_table(Xw))
        //     = z(Xw) *
        //      (gamma(1+beta) + h1(X) + beta * h1(Xw)) *
        //      (gamma(1+beta) + h2(x) + beta * h2(Xw))
        // on the vanishing set excluding point w^{n-1}
        let beta_plus_one = E::ScalarField::one() + beta;
        let gamma_mul_beta_plus_one = gamma * beta_plus_one;

        let range_table_poly_ref = &pk.plookup_pk.as_ref().unwrap().range_table_poly;
        let key_table_poly_ref = &pk.plookup_pk.as_ref().unwrap().key_table_poly;
        let table_dom_sep_poly_ref = &pk.plookup_pk.as_ref().unwrap().table_dom_sep_poly;
        let q_dom_sep_poly_ref = &pk.plookup_pk.as_ref().unwrap().q_dom_sep_poly;

        for i in 0..domain.size() - 1 {
            let point = domain.element(i);
            let next_point = point * domain.group_gen;
            let merged_lookup_wire_eval = eval_merged_lookup_witness::<E>(
                challenges.tau,
                oracles.wire_polys[5].evaluate(&point),
                oracles.wire_polys[0].evaluate(&point),
                oracles.wire_polys[1].evaluate(&point),
                oracles.wire_polys[2].evaluate(&point),
                pk.q_lookup_poly()?.evaluate(&point),
                q_dom_sep_poly_ref.evaluate(&point),
            );
            let merged_table_eval = eval_merged_table::<E>(
                challenges.tau,
                range_table_poly_ref.evaluate(&point),
                key_table_poly_ref.evaluate(&point),
                pk.q_lookup_poly()?.evaluate(&point),
                oracles.wire_polys[3].evaluate(&point),
                oracles.wire_polys[4].evaluate(&point),
                table_dom_sep_poly_ref.evaluate(&point),
            );
            let merged_table_next_eval = eval_merged_table::<E>(
                challenges.tau,
                range_table_poly_ref.evaluate(&next_point),
                key_table_poly_ref.evaluate(&next_point),
                pk.q_lookup_poly()?.evaluate(&next_point),
                oracles.wire_polys[3].evaluate(&next_point),
                oracles.wire_polys[4].evaluate(&next_point),
                table_dom_sep_poly_ref.evaluate(&next_point),
            );

            let eval_1 = prod_poly.evaluate(&point)
                * beta_plus_one
                * (gamma + merged_lookup_wire_eval)
                * (gamma_mul_beta_plus_one + merged_table_eval + beta * merged_table_next_eval);
            let eval_2 = prod_poly.evaluate(&next_point)
                * (gamma_mul_beta_plus_one
                    + h_polys[0].evaluate(&point)
                    + beta * h_polys[0].evaluate(&next_point))
                * (gamma_mul_beta_plus_one
                    + h_polys[1].evaluate(&point)
                    + beta * h_polys[1].evaluate(&next_point));
            assert_eq!(eval_1, eval_2, "i={}, domain_size={}", i, domain.size());
        }

        Ok(())
    }
}
