// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Instantiations of Plonk-based proof systems
use super::{
    prover::Prover,
    structs::{
        BatchProof, Challenges, Oracles, PlookupProof, PlookupProvingKey, PlookupVerifyingKey,
        Proof, ProvingKey, VerifyingKey,
    },
    verifier::Verifier,
    UniversalSNARK,
};
use crate::{
    constants::EXTRA_TRANSCRIPT_MSG_LABEL,
    errors::{PlonkError, SnarkError::ParameterError},
    proof_system::structs::UniversalSrs,
    transcript::*,
};
use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, Projective},
};
use ark_ff::{Field, One, PrimeField};
use ark_std::{
    format,
    marker::PhantomData,
    rand::{CryptoRng, RngCore},
    string::ToString,
    vec,
    vec::Vec,
};
use jf_primitives::{
    pcs::{prelude::UnivariateKzgPCS, PolynomialCommitmentScheme, StructuredReferenceString},
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
/// A Plonk instantiated with KZG PCS
pub struct PlonkKzgSnark<E: Pairing>(PhantomData<E>);

impl<E, F, P> PlonkKzgSnark<E>
where
    E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
    F: RescueParameter + PrimeField,
    P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
    E::ScalarField: EmulationConfig<F>,
{
    #[allow(clippy::new_without_default)]
    /// A new Plonk KZG SNARK
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Generate an aggregated Plonk proof for multiple instances.
    pub fn batch_prove<C, R, T>(
        prng: &mut R,
        circuits: &[&C],
        prove_keys: &[&ProvingKey<E>],
    ) -> Result<BatchProof<E>, PlonkError>
    where
        C: Arithmetization<E::ScalarField>,
        R: CryptoRng + RngCore,
        T: Transcript,
    {
        let (batch_proof, ..) =
            Self::batch_prove_internal::<_, _, T>(prng, circuits, prove_keys, None)?;
        Ok(batch_proof)
    }

    /// Verify a single aggregated Plonk proof.
    pub fn verify_batch_proof<T>(
        verify_keys: &[&VerifyingKey<E>],
        public_inputs: &[&[E::ScalarField]],
        batch_proof: &BatchProof<E>,
    ) -> Result<(), PlonkError>
    where
        T: Transcript,
    {
        if verify_keys.is_empty() {
            return Err(ParameterError("empty verification keys".to_string()).into());
        }
        let verifier = Verifier::new(verify_keys[0].domain_size)?;
        let pcs_info =
            verifier.prepare_pcs_info::<T>(verify_keys, public_inputs, batch_proof, &None)?;
        if !Verifier::batch_verify_opening_proofs::<T>(
            &verify_keys[0].open_key, // all open_key are the same
            &[pcs_info],
        )? {
            return Err(PlonkError::WrongProof);
        }
        Ok(())
    }

    /// Batch verify multiple SNARK proofs (w.r.t. different verifying keys).
    pub fn batch_verify<T>(
        verify_keys: &[&VerifyingKey<E>],
        public_inputs: &[&[E::ScalarField]],
        proofs: &[&Proof<E>],
        extra_transcript_init_msgs: &[Option<Vec<u8>>],
    ) -> Result<(), PlonkError>
    where
        T: Transcript,
    {
        if public_inputs.len() != proofs.len()
            || verify_keys.len() != proofs.len()
            || extra_transcript_init_msgs.len() != proofs.len()
        {
            return Err(ParameterError(format!(
                "verify_keys.len: {}, public_inputs.len: {}, proofs.len: {}, \
                 extra_transcript_msg.len: {}",
                verify_keys.len(),
                public_inputs.len(),
                proofs.len(),
                extra_transcript_init_msgs.len()
            ))
            .into());
        }
        if verify_keys.is_empty() {
            return Err(
                ParameterError("the number of instances cannot be zero".to_string()).into(),
            );
        }

        let pcs_infos = parallelizable_slice_iter(verify_keys)
            .zip(parallelizable_slice_iter(proofs))
            .zip(parallelizable_slice_iter(public_inputs))
            .zip(parallelizable_slice_iter(extra_transcript_init_msgs))
            .map(|(((&vk, &proof), &pub_input), extra_msg)| {
                let verifier = Verifier::new(vk.domain_size)?;
                verifier.prepare_pcs_info::<T>(
                    &[vk],
                    &[pub_input],
                    &(*proof).clone().into(),
                    extra_msg,
                )
            })
            .collect::<Result<Vec<_>, PlonkError>>()?;

        match Verifier::batch_verify_opening_proofs::<T>(
            &verify_keys[0].open_key, // all open_key are the same
            &pcs_infos,
        ) {
            Ok(false) => Err(PlonkError::WrongProof),
            Err(e) => Err(e),
            Ok(true) => Ok(()),
        }
    }

    /// An internal private API for ease of testing
    ///
    /// Batchly compute a Plonk proof for multiple instances. Return the batch
    /// proof and the corresponding online polynomial oracles and
    /// challenges. Refer to Sec 8.4 of https://eprint.iacr.org/2019/953.pdf
    ///
    /// `circuit` and `prove_key` has to be consistent (with the same evaluation
    /// domain etc.), otherwise return error.
    #[allow(clippy::type_complexity)]
    fn batch_prove_internal<C, R, T>(
        prng: &mut R,
        circuits: &[&C],
        prove_keys: &[&ProvingKey<E>],
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<
        (
            BatchProof<E>,
            Vec<Oracles<E::ScalarField>>,
            Challenges<E::ScalarField>,
            T,
        ),
        PlonkError,
    >
    where
        C: Arithmetization<E::ScalarField>,
        R: CryptoRng + RngCore,
        T: Transcript,
    {
        if circuits.is_empty() {
            return Err(ParameterError("zero number of circuits/proving keys".to_string()).into());
        }
        if circuits.len() != prove_keys.len() {
            return Err(ParameterError(format!(
                "the number of circuits {} != the number of proving keys {}",
                circuits.len(),
                prove_keys.len()
            ))
            .into());
        }
        let n = circuits[0].eval_domain_size()?;
        let num_wire_types = circuits[0].num_wire_types();
        for (circuit, pk) in circuits.iter().zip(prove_keys.iter()) {
            if circuit.eval_domain_size()? != n {
                return Err(ParameterError(format!(
                    "circuit domain size {} != expected domain size {}",
                    circuit.eval_domain_size()?,
                    n
                ))
                .into());
            }
            if pk.domain_size() != n {
                return Err(ParameterError(format!(
                    "proving key domain size {} != expected domain size {}",
                    pk.domain_size(),
                    n
                ))
                .into());
            }
            if circuit.num_inputs() != pk.vk.num_inputs {
                return Err(ParameterError(format!(
                    "circuit.num_inputs {} != prove_key.num_inputs {}",
                    circuit.num_inputs(),
                    pk.vk.num_inputs
                ))
                .into());
            }
            if circuit.support_lookup() != pk.plookup_pk.is_some() {
                return Err(ParameterError(
                    "Mismatched Plonk types between the proving key and the circuit".to_string(),
                )
                .into());
            }
            if circuit.num_wire_types() != num_wire_types {
                return Err(ParameterError("inconsistent plonk circuit types".to_string()).into());
            }
        }

        // Initialize transcript
        let mut transcript = T::new_transcript(b"PlonkProof");
        if let Some(msg) = extra_transcript_init_msg {
            transcript.push_message(EXTRA_TRANSCRIPT_MSG_LABEL, &msg)?;
        }
        for (pk, circuit) in prove_keys.iter().zip(circuits.iter()) {
            transcript.append_visitor(&pk.vk)?;
            for pub_input in circuit.public_input()? {
                transcript.push_message(b"public_input", &pub_input)?;
            }
        }
        // Initialize verifier challenges and online polynomial oracles.
        let mut challenges = Challenges::default();
        let mut online_oracles = vec![Oracles::default(); circuits.len()];
        let prover = Prover::new(n, num_wire_types)?;

        // Round 1
        let mut wires_poly_comms_vec = vec![];
        for i in 0..circuits.len() {
            let ((wires_poly_comms, wire_polys), pi_poly) =
                prover.run_1st_round(prng, &prove_keys[i].commit_key, circuits[i])?;
            online_oracles[i].wire_polys = wire_polys;
            online_oracles[i].pub_inp_poly = pi_poly;
            for wire_poly_comm in wires_poly_comms.iter() {
                transcript.append_curve_point(b"wire_poly_comms", wire_poly_comm)?;
            }
            wires_poly_comms_vec.push(wires_poly_comms);
        }

        // Round 1.5
        // Plookup: compute and interpolate the sorted concatenation of the (merged)
        // lookup table and the (merged) witness values
        challenges.tau = transcript.squeeze_scalar_challenge::<P>(b"tau")?;
        let mut h_poly_comms_vec = vec![];
        let mut sorted_vec_list = vec![];
        let mut merged_table_list = vec![];
        for i in 0..circuits.len() {
            let (sorted_vec, h_poly_comms, merged_table) = if circuits[i].support_lookup() {
                let ((h_poly_comms, h_polys), sorted_vec, merged_table) = prover
                    .run_plookup_1st_round(
                        prng,
                        &prove_keys[i].commit_key,
                        circuits[i],
                        challenges.tau,
                    )?;
                online_oracles[i].plookup_oracles.h_polys = h_polys;
                for h_poly_comm in h_poly_comms.iter() {
                    transcript.append_curve_point(b"h_poly_comms", h_poly_comm)?;
                }

                (Some(sorted_vec), Some(h_poly_comms), Some(merged_table))
            } else {
                (None, None, None)
            };
            h_poly_comms_vec.push(h_poly_comms);
            sorted_vec_list.push(sorted_vec);
            merged_table_list.push(merged_table);
        }

        // Round 2
        challenges.beta = transcript.squeeze_scalar_challenge::<P>(b"beta")?;
        challenges.gamma = transcript.squeeze_scalar_challenge::<P>(b"gamma")?;
        let mut prod_perm_poly_comms_vec = vec![];
        for i in 0..circuits.len() {
            let (prod_perm_poly_comm, prod_perm_poly) =
                prover.run_2nd_round(prng, &prove_keys[i].commit_key, circuits[i], &challenges)?;
            online_oracles[i].prod_perm_poly = prod_perm_poly;
            transcript.append_curve_point(b"perm_poly_comms", &prod_perm_poly_comm)?;
            prod_perm_poly_comms_vec.push(prod_perm_poly_comm);
        }

        // Round 2.5
        // Plookup: compute Plookup product accumulation polynomial
        let mut prod_lookup_poly_comms_vec = vec![];
        for i in 0..circuits.len() {
            let prod_lookup_poly_comm = if circuits[i].support_lookup() {
                let (prod_lookup_poly_comm, prod_lookup_poly) = prover.run_plookup_2nd_round(
                    prng,
                    &prove_keys[i].commit_key,
                    circuits[i],
                    &challenges,
                    merged_table_list[i].as_ref(),
                    sorted_vec_list[i].as_ref(),
                )?;
                online_oracles[i].plookup_oracles.prod_lookup_poly = prod_lookup_poly;
                transcript.append_curve_point(b"plookup_poly_comms", &prod_lookup_poly_comm)?;
                Some(prod_lookup_poly_comm)
            } else {
                None
            };
            prod_lookup_poly_comms_vec.push(prod_lookup_poly_comm);
        }

        // Round 3
        challenges.alpha = transcript.squeeze_scalar_challenge::<P>(b"alpha")?;
        let (split_quot_poly_comms, split_quot_polys) = prover.run_3rd_round(
            prng,
            &prove_keys[0].commit_key,
            prove_keys,
            &challenges,
            &online_oracles,
            num_wire_types,
        )?;

        for split_quot_poly_comm in split_quot_poly_comms.iter() {
            transcript.append_curve_point(b"quot_poly_comms", split_quot_poly_comm)?;
        }

        // Round 4
        challenges.zeta = transcript.squeeze_scalar_challenge::<P>(b"zeta")?;
        let mut poly_evals_vec = vec![];
        for i in 0..circuits.len() {
            let poly_evals = prover.compute_evaluations(
                prove_keys[i],
                &challenges,
                &online_oracles[i],
                num_wire_types,
            );
            transcript.append_visitor(&poly_evals)?;
            poly_evals_vec.push(poly_evals);
        }

        // Round 4.5
        // Plookup: compute evaluations on Plookup-related polynomials
        let mut plookup_evals_vec = vec![];
        for i in 0..circuits.len() {
            let plookup_evals = if circuits[i].support_lookup() {
                let evals = prover.compute_plookup_evaluations(
                    prove_keys[i],
                    &challenges,
                    &online_oracles[i],
                )?;
                transcript.append_visitor(&evals)?;
                Some(evals)
            } else {
                None
            };
            plookup_evals_vec.push(plookup_evals);
        }

        let mut lin_poly = Prover::<E>::compute_quotient_component_for_lin_poly(
            n,
            challenges.zeta,
            &split_quot_polys,
        )?;
        let mut alpha_base = E::ScalarField::one();
        let alpha_3 = challenges.alpha.square() * challenges.alpha;
        let alpha_7 = alpha_3.square() * challenges.alpha;
        for i in 0..circuits.len() {
            lin_poly = lin_poly
                + prover.compute_non_quotient_component_for_lin_poly(
                    alpha_base,
                    prove_keys[i],
                    &challenges,
                    &online_oracles[i],
                    &poly_evals_vec[i],
                    plookup_evals_vec[i].as_ref(),
                )?;
            // update the alpha power term (i.e. the random combiner that aggregates
            // multiple instances)
            if plookup_evals_vec[i].is_some() {
                alpha_base *= alpha_7;
            } else {
                alpha_base *= alpha_3;
            }
        }

        // Round 5
        challenges.v = transcript.squeeze_scalar_challenge::<P>(b"v")?;
        let (opening_proof, shifted_opening_proof) = prover.compute_opening_proofs(
            &prove_keys[0].commit_key,
            prove_keys,
            &challenges.zeta,
            &challenges.v,
            &online_oracles,
            &lin_poly,
        )?;

        // Plookup: build Plookup argument
        let mut plookup_proofs_vec = vec![];
        for i in 0..circuits.len() {
            let plookup_proof = if circuits[i].support_lookup() {
                Some(PlookupProof {
                    h_poly_comms: h_poly_comms_vec[i].clone().unwrap(),
                    prod_lookup_poly_comm: prod_lookup_poly_comms_vec[i].unwrap(),
                    poly_evals: plookup_evals_vec[i].clone().unwrap(),
                })
            } else {
                None
            };
            plookup_proofs_vec.push(plookup_proof);
        }

        Ok((
            BatchProof {
                wires_poly_comms_vec,
                prod_perm_poly_comms_vec,
                poly_evals_vec,
                plookup_proofs_vec,
                split_quot_poly_comms,
                opening_proof,
                shifted_opening_proof,
            },
            online_oracles,
            challenges,
            transcript,
        ))
    }
}

impl<E, F, P> UniversalSNARK<UnivariateKzgPCS<E>> for PlonkKzgSnark<E>
where
    E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
    F: RescueParameter + PrimeField,
    P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
    E::ScalarField: EmulationConfig<F>,
{
    type Proof = Proof<E>;
    type RecursiveProof = ();
    type ProvingKey = ProvingKey<E>;
    type VerifyingKey = VerifyingKey<E>;
    type UniversalSRS = UniversalSrs<E>;
    type Error = PlonkError;

    // FIXME: (alex) see <https://github.com/EspressoSystems/jellyfish/issues/249>
    #[cfg(any(test, feature = "test-srs"))]
    fn universal_setup_for_testing<R: RngCore + CryptoRng>(
        max_degree: usize,
        rng: &mut R,
    ) -> Result<Self::UniversalSRS, Self::Error> {
        use ark_ec::{scalar_mul::fixed_base::FixedBase, CurveGroup};
        use ark_std::{end_timer, start_timer, UniformRand};

        let setup_time = start_timer!(|| format!("KZG10::Setup with degree {}", max_degree));
        let beta = E::ScalarField::rand(rng);
        let g = E::G1::rand(rng);
        let h = E::G2::rand(rng);

        let mut powers_of_beta = vec![E::ScalarField::one()];

        let mut cur = beta;
        for _ in 0..max_degree {
            powers_of_beta.push(cur);
            cur *= &beta;
        }

        let window_size = FixedBase::get_mul_window_size(max_degree + 1);

        let scalar_bits = E::ScalarField::MODULUS_BIT_SIZE as usize;
        let g_time = start_timer!(|| "Generating powers of G");
        // TODO: parallelization
        let g_table = FixedBase::get_window_table(scalar_bits, window_size, g);
        let powers_of_g = FixedBase::msm::<E::G1>(
            scalar_bits,
            window_size,
            &g_table,
            powers_of_beta.as_slice(),
        );
        end_timer!(g_time);

        let powers_of_g = E::G1::normalize_batch(&powers_of_g);

        let h = h.into_affine();
        let beta_h = (h * beta).into_affine();

        let pp = UniversalSrs {
            powers_of_g,
            h,
            beta_h,
        };
        end_timer!(setup_time);
        Ok(pp)
    }

    /// Input a circuit and the SRS, precompute the proving key and verification
    /// key.
    fn preprocess<C: Arithmetization<<E as Pairing>::ScalarField>>(
        srs: &Self::UniversalSRS,
        circuit: &C,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), Self::Error> {
        // Make sure the SRS can support the circuit (with hiding degree of 2 for zk)
        let domain_size = circuit.eval_domain_size()?;
        let srs_size = circuit.srs_size()?;
        let num_inputs = circuit.num_inputs();
        if srs.max_degree() < circuit.srs_size()? {
            return Err(PlonkError::IndexTooLarge);
        }
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
            .map(|poly| UnivariateKzgPCS::commit(&commit_key, poly).map_err(PlonkError::PCSError))
            .collect::<Result<Vec<_>, PlonkError>>()?
            .into_iter()
            .collect();
        let sigma_comms = parallelizable_slice_iter(&sigma_polys)
            .map(|poly| UnivariateKzgPCS::commit(&commit_key, poly).map_err(PlonkError::PCSError))
            .collect::<Result<Vec<_>, PlonkError>>()?
            .into_iter()
            .collect();

        // Compute Plookup verifying key if support lookup.
        let plookup_vk = match circuit.support_lookup() {
            false => None,
            true => Some(PlookupVerifyingKey {
                range_table_comm: UnivariateKzgPCS::commit(
                    &commit_key,
                    &plookup_pk.as_ref().unwrap().range_table_poly,
                )?,
                key_table_comm: UnivariateKzgPCS::commit(
                    &commit_key,
                    &plookup_pk.as_ref().unwrap().key_table_poly,
                )?,
                table_dom_sep_comm: UnivariateKzgPCS::commit(
                    &commit_key,
                    &plookup_pk.as_ref().unwrap().table_dom_sep_poly,
                )?,
                q_dom_sep_comm: UnivariateKzgPCS::commit(
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
    ///
    /// `circuit` and `prove_key` has to be consistent (with the same evaluation
    /// domain etc.), otherwise return error.
    fn prove<C, R, T>(
        rng: &mut R,
        circuit: &C,
        prove_key: &Self::ProvingKey,
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<Self::Proof, Self::Error>
    where
        Self: Sized,
        C: Arithmetization<<<UnivariateKzgPCS<E> as PolynomialCommitmentScheme>::Commitment as ark_ec::AffineRepr>::ScalarField>,
        R: CryptoRng + RngCore,
    T: Transcript{
        let (batch_proof, _, _, _) = Self::batch_prove_internal::<_, _, T>(
            rng,
            &[circuit],
            &[prove_key],
            extra_transcript_init_msg,
        )?;
        Ok(Proof {
            wires_poly_comms: batch_proof.wires_poly_comms_vec[0].clone(),
            prod_perm_poly_comm: batch_proof.prod_perm_poly_comms_vec[0],
            split_quot_poly_comms: batch_proof.split_quot_poly_comms,
            opening_proof: batch_proof.opening_proof,
            shifted_opening_proof: batch_proof.shifted_opening_proof,
            poly_evals: batch_proof.poly_evals_vec[0].clone(),
            plookup_proof: batch_proof.plookup_proofs_vec[0].clone(),
        })
    }


    fn recursive_prove<C, R, T>(
            _rng: &mut R,
            _circuit: &C,
            _prove_key: &Self::ProvingKey,
            _extra_transcript_init_msg: Option<Vec<u8>>,
        ) -> Result<super::RecursiveOutput<UnivariateKzgPCS<E>, Self, T>, Self::Error>
        where
            Self: Sized,
            C: Arithmetization<<<UnivariateKzgPCS<E> as PolynomialCommitmentScheme>::Commitment as ark_ec::AffineRepr>::ScalarField>,
            R: CryptoRng + RngCore,
    T: Transcript + ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize{
        unimplemented!("Recursive proving is not supported for PlonkKzgSnark")
    }

    fn verify<T>(
        verify_key: &Self::VerifyingKey,
        public_input: &[E::ScalarField],
        proof: &Self::Proof,
        extra_transcript_init_msg: Option<Vec<u8>>,
    ) -> Result<(), Self::Error>
    where
        T: Transcript,
    {
        Self::batch_verify::<T>(
            &[verify_key],
            &[public_input],
            &[proof],
            &[extra_transcript_init_msg],
        )
    }
}

#[cfg(test)]
pub mod test {
    use crate::{
        errors::PlonkError,
        nightfall::FFTPlonk,
        proof_system::{
            structs::{
                eval_merged_lookup_witness, eval_merged_table, Challenges, Oracles, Proof,
                ProvingKey, UniversalSrs, VerifyingKey,
            },
            PlonkKzgSnark, UniversalSNARK,
        },
        transcript::{
            rescue::RescueTranscript, solidity::SolidityTranscript, standard::StandardTranscript,
            Transcript,
        },
        PlonkType,
    };
    use ark_bls12_377::{Bls12_377, Fq as Fq377};
    use ark_bls12_381::{Bls12_381, Fq as Fq381};
    use ark_bn254::{Bn254, Fq as Fq254, Fr as Fr254};
    use ark_bw6_761::{Fq as Fq761, BW6_761};
    use ark_ec::{
        pairing::Pairing,
        short_weierstrass::{Affine, Projective},
    };
    use ark_ff::{One, PrimeField, Zero};
    use ark_poly::{
        univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Polynomial,
        Radix2EvaluationDomain,
    };
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use ark_std::path::PathBuf;
    use ark_std::{
        boxed::Box, convert::TryInto, format, fs::remove_file, path::Path, rand::SeedableRng,
        string::ToString, vec, vec::Vec, UniformRand,
    };
    use core::ops::{Mul, Neg};
    use jf_primitives::{
        pcs::{prelude::UnivariateKzgPCS, PolynomialCommitmentScheme},
        rescue::RescueParameter,
    };
    use jf_relation::{
        constants::GATE_WIDTH,
        gadgets::{ecc::HasTEForm, EmulationConfig},
        Arithmetization, Circuit, PlonkCircuit,
    };
    use jf_utils::test_rng;
    use rand_chacha::ChaCha20Rng;
    // Different `m`s lead to different circuits.
    // Different `a0`s lead to different witness values.
    // For UltraPlonk circuits, `a0` should be less than or equal to `m+1`
    pub(crate) fn gen_circuit_for_test<F: PrimeField>(
        m: usize,
        a0: usize,
        plonk_type: PlonkType,
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
        cs.finalize_for_arithmetization()?;
        Ok(cs)
    }
    fn download_ptau_file(url: &str, destination: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut response = reqwest::blocking::get(url)?;
        let mut file = std::fs::File::create(destination)?;
        std::io::copy(&mut response, &mut file)?;
        Ok(())
    }
    fn generate_simple_circuit(
        a: Fr254,
        b: Fr254,
        c: Fr254,
    ) -> Result<PlonkCircuit<Fr254>, PlonkError> {
        let mut circuit = PlonkCircuit::<Fr254>::new_turbo_plonk();
        let a_var = circuit.create_variable(a)?;
        let b_var = circuit.create_variable(b)?;
        let c_var = circuit.create_public_variable(c)?;
        circuit.mul_gate(a_var, b_var, c_var)?;
        circuit.finalize_for_arithmetization()?;
        Ok(circuit)
    }
    #[test]
    fn test_ptau_srs() {
        let srs_size = 6;
        let ptau_file_url = "https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_10.ptau";
        let ptau_file_path = "/tmp/test.ptau";
        download_ptau_file(ptau_file_url, ptau_file_path)
            .expect("Failed to download the PTAU file");
        let srs = UnivariateKzgPCS::<Bn254>::universal_setup_bn254(
            &PathBuf::from(ptau_file_path),
            srs_size + 1,
        )
        .expect("Failed to generate SRS from PTAU file");

        let a = Fr254::from(3u64);
        let b = Fr254::from(5u64);
        let c = a * b;
        let circuit = generate_simple_circuit(a, b, c).unwrap();

        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let pub_inputs = circuit.public_input().unwrap();
        let (test_pk_plonk_kzg_snark, test_vk_plonk_kzg_snark) =
            PlonkKzgSnark::<Bn254>::preprocess(&srs, &circuit)
                .expect("Failed to preprocess the test circuit");
        let (test_pk_fftplonk, test_vk_fftplonk) =
            FFTPlonk::<UnivariateKzgPCS<Bn254>>::preprocess(&srs, &circuit)
                .expect("Failed to preprocess the test circuit");

        let proof_plonk_kzg_snark = PlonkKzgSnark::<Bn254>::prove::<_, _, StandardTranscript>(
            &mut rng,
            &circuit,
            &test_pk_plonk_kzg_snark,
            None,
        )
        .unwrap();
        let proof_fftplonk =
            FFTPlonk::<UnivariateKzgPCS<Bn254>>::prove::<_, _, StandardTranscript>(
                &mut rng,
                &circuit,
                &test_pk_fftplonk,
                None,
            )
            .unwrap();
        assert!(
            PlonkKzgSnark::<Bn254>::verify::<StandardTranscript>(
                &test_vk_plonk_kzg_snark,
                &pub_inputs,
                &proof_plonk_kzg_snark,
                None
            )
            .is_ok(),
            "Proof_PlonkKzgSnark verification failed"
        );
        assert!(
            FFTPlonk::<UnivariateKzgPCS<Bn254>>::verify::<StandardTranscript>(
                &test_vk_fftplonk,
                &pub_inputs,
                &proof_fftplonk,
                None
            )
            .is_ok(),
            "Proof_FFTPlonk verification failed"
        );

        // Loop through more inputs, using the same circuit structure but different inputs
        for _ in 0..10 {
            // Generate random inputs for the circuit
            let new_a = Fr254::rand(&mut rng);
            let new_b = Fr254::rand(&mut rng);
            let new_c = new_a * new_b;

            let circuit = generate_simple_circuit(new_a, new_b, new_c).unwrap();
            // Generate a new proving key and verifying key for the circuit with new inputs
            let (new_pk_plonk_kzg_snark, new_vk_plonk_kzg_snark) =
                PlonkKzgSnark::<Bn254>::preprocess(&srs, &circuit)
                    .expect("Failed to preprocess the circuit with new inputs");

            // Compare the new PK and VK with the initial ones
            assert_eq!(
                test_pk_plonk_kzg_snark, new_pk_plonk_kzg_snark,
                "Proving keys do not match"
            );
            assert_eq!(
                test_vk_plonk_kzg_snark, new_vk_plonk_kzg_snark,
                "Verifying keys do not match"
            );
            let (new_pk_fftplonk, new_vk_fftplonk) =
                FFTPlonk::<UnivariateKzgPCS<Bn254>>::preprocess(&srs, &circuit).unwrap();
            assert_eq!(
                test_pk_fftplonk, new_pk_fftplonk,
                "Proving keys do not match"
            );
            assert_eq!(
                test_vk_fftplonk, new_vk_fftplonk,
                "Verifying keys do not match"
            );
        }
        // Clean up: Delete the downloaded PTAU file
        if Path::new(ptau_file_path).exists() {
            remove_file(ptau_file_path).expect("Failed to delete PTAU file");
        }
    }

    #[test]
    fn test_preprocessing() -> Result<(), PlonkError> {
        test_preprocessing_helper::<Bn254, Fq254, _>(PlonkType::TurboPlonk)?;
        test_preprocessing_helper::<Bn254, Fq254, _>(PlonkType::UltraPlonk)?;
        test_preprocessing_helper::<Bls12_377, Fq377, _>(PlonkType::TurboPlonk)?;
        test_preprocessing_helper::<Bls12_377, Fq377, _>(PlonkType::UltraPlonk)?;
        test_preprocessing_helper::<Bls12_381, Fq381, _>(PlonkType::TurboPlonk)?;
        test_preprocessing_helper::<Bls12_381, Fq381, _>(PlonkType::UltraPlonk)?;
        test_preprocessing_helper::<BW6_761, Fq761, _>(PlonkType::TurboPlonk)?;
        test_preprocessing_helper::<BW6_761, Fq761, _>(PlonkType::UltraPlonk)
    }
    fn test_preprocessing_helper<E, F, P>(plonk_type: PlonkType) -> Result<(), PlonkError>
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
        F: RescueParameter + PrimeField,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        E::ScalarField: EmulationConfig<F>,
    {
        let rng = &mut jf_utils::test_rng();
        let circuit = gen_circuit_for_test(5, 6, plonk_type)?;
        let domain_size = circuit.eval_domain_size()?;
        let num_inputs = circuit.num_inputs();
        let selectors = circuit.compute_selector_polynomials()?;
        let sigmas = circuit.compute_extended_permutation_polynomials()?;

        let max_degree = 64 + 2;
        let srs = PlonkKzgSnark::<E>::universal_setup_for_testing(max_degree, rng)?;
        let (pk, vk) = PlonkKzgSnark::<E>::preprocess(&srs, &circuit)?;

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
                let expected_comm = UnivariateKzgPCS::commit(&pk.commit_key, p).unwrap();
                assert_eq!(expected_comm, p_comm);
            });
        sigmas
            .iter()
            .zip(vk.sigma_comms.iter())
            .for_each(|(p, &p_comm)| {
                let expected_comm = UnivariateKzgPCS::commit(&pk.commit_key, p).unwrap();
                assert_eq!(expected_comm, p_comm);
            });
        // check plookup verification key
        if plonk_type == PlonkType::UltraPlonk {
            let expected_comm = UnivariateKzgPCS::commit(
                &pk.commit_key,
                &pk.plookup_pk.as_ref().unwrap().range_table_poly,
            )
            .unwrap();
            assert_eq!(
                expected_comm,
                vk.plookup_vk.as_ref().unwrap().range_table_comm
            );

            let expected_comm = UnivariateKzgPCS::commit(
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
        // merlin transcripts
        test_plonk_proof_system_helper::<Bn254, Fq254, _, StandardTranscript>(
            PlonkType::TurboPlonk,
        )?;
        test_plonk_proof_system_helper::<Bn254, Fq254, _, StandardTranscript>(
            PlonkType::UltraPlonk,
        )?;
        test_plonk_proof_system_helper::<Bls12_377, Fq377, _, StandardTranscript>(
            PlonkType::TurboPlonk,
        )?;
        test_plonk_proof_system_helper::<Bls12_377, Fq377, _, StandardTranscript>(
            PlonkType::UltraPlonk,
        )?;
        test_plonk_proof_system_helper::<Bls12_381, Fq381, _, StandardTranscript>(
            PlonkType::TurboPlonk,
        )?;
        test_plonk_proof_system_helper::<Bls12_381, Fq381, _, StandardTranscript>(
            PlonkType::UltraPlonk,
        )?;
        test_plonk_proof_system_helper::<BW6_761, Fq761, _, StandardTranscript>(
            PlonkType::TurboPlonk,
        )?;
        test_plonk_proof_system_helper::<BW6_761, Fq761, _, StandardTranscript>(
            PlonkType::UltraPlonk,
        )?;

        // rescue transcripts
        // currently only available for bls12-377
        test_plonk_proof_system_helper::<Bls12_377, Fq377, _, RescueTranscript<Fq377>>(
            PlonkType::TurboPlonk,
        )?;
        test_plonk_proof_system_helper::<Bls12_377, Fq377, _, RescueTranscript<Fq377>>(
            PlonkType::UltraPlonk,
        )?;

        // solidity-friendly keccak256 transcripts
        // currently only needed for CAPE using bls12-381
        test_plonk_proof_system_helper::<Bls12_381, Fq381, _, SolidityTranscript>(
            PlonkType::TurboPlonk,
        )?;
        Ok(())
    }

    fn test_plonk_proof_system_helper<E, F, P, T>(plonk_type: PlonkType) -> Result<(), PlonkError>
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
        E::ScalarField: EmulationConfig<F>,
        F: RescueParameter + PrimeField,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        T: Transcript,
    {
        // 1. Simulate universal setup
        let rng = &mut test_rng();
        let n = 64;
        let max_degree = n + 2;
        let srs = PlonkKzgSnark::<E>::universal_setup_for_testing(max_degree, rng)?;

        // 2. Create circuits
        let circuits = (0..6)
            .map(|i| {
                let m = 2 + i / 3;
                let a0 = 1 + i % 3;
                gen_circuit_for_test(m, a0, plonk_type)
            })
            .collect::<Result<Vec<_>, PlonkError>>()?;
        // 3. Preprocessing
        let (pk1, vk1) = PlonkKzgSnark::<E>::preprocess(&srs, &circuits[0])?;
        let (pk2, vk2) = PlonkKzgSnark::<E>::preprocess(&srs, &circuits[3])?;
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
                PlonkKzgSnark::<E>::prove::<_, _, T>(rng, cs, pk_ref, extra_msg.clone()).unwrap(),
            );
            extra_msgs.push(extra_msg);
        }

        // 5. Verification
        let public_inputs: Vec<Vec<E::ScalarField>> = circuits
            .iter()
            .map(|cs| cs.public_input())
            .collect::<Result<Vec<Vec<E::ScalarField>>, _>>(
        )?;
        for (i, proof) in proofs.iter().enumerate() {
            let vk_ref = if i < 3 { &vk1 } else { &vk2 };
            assert!(PlonkKzgSnark::<E>::verify::<T>(
                vk_ref,
                &public_inputs[i],
                proof,
                extra_msgs[i].clone(),
            )
            .is_ok());
            // Inconsistent proof should fail the verification.
            let mut bad_pub_input = public_inputs[i].clone();
            bad_pub_input[0] = E::ScalarField::from(0u8);
            assert!(PlonkKzgSnark::<E>::verify::<T>(
                vk_ref,
                &bad_pub_input,
                proof,
                extra_msgs[i].clone(),
            )
            .is_err());
            // Incorrect extra transcript message should fail
            assert!(PlonkKzgSnark::<E>::verify::<T>(
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
            bad_proof.opening_proof = E::G1Affine::default();
            bad_proof.shifted_opening_proof = E::G1Affine::default();
            assert!(PlonkKzgSnark::<E>::verify::<T>(
                vk_ref,
                &public_inputs[i],
                &bad_proof,
                extra_msgs[i].clone(),
            )
            .is_err());
        }

        // 6. Batch verification
        let vks = vec![&vk1, &vk1, &vk1, &vk2, &vk2, &vk2];
        let mut public_inputs_ref: Vec<&[E::ScalarField]> = public_inputs
            .iter()
            .map(|pub_input| &pub_input[..])
            .collect();
        let mut proofs_ref: Vec<&Proof<E>> = proofs.iter().collect();
        assert!(PlonkKzgSnark::<E>::batch_verify::<T>(
            &vks,
            &public_inputs_ref,
            &proofs_ref,
            &extra_msgs,
        )
        .is_ok());

        // Inconsistent params
        assert!(PlonkKzgSnark::<E>::batch_verify::<T>(
            &vks[..5],
            &public_inputs_ref,
            &proofs_ref,
            &extra_msgs,
        )
        .is_err());

        assert!(PlonkKzgSnark::<E>::batch_verify::<T>(
            &vks,
            &public_inputs_ref[..5],
            &proofs_ref,
            &extra_msgs,
        )
        .is_err());

        assert!(PlonkKzgSnark::<E>::batch_verify::<T>(
            &vks,
            &public_inputs_ref,
            &proofs_ref[..5],
            &extra_msgs,
        )
        .is_err());

        assert!(PlonkKzgSnark::<E>::batch_verify::<T>(
            &vks,
            &public_inputs_ref,
            &proofs_ref,
            &vec![None; vks.len()],
        )
        .is_err());

        assert!(
            PlonkKzgSnark::<E>::batch_verify::<T>(&vks, &public_inputs_ref, &proofs_ref, &[],)
                .is_err()
        );

        // Empty params
        assert!(PlonkKzgSnark::<E>::batch_verify::<T>(&[], &[], &[], &[],).is_err());

        // Error paths
        let tmp_pi_ref = public_inputs_ref[0];
        public_inputs_ref[0] = public_inputs_ref[1];
        assert!(PlonkKzgSnark::<E>::batch_verify::<T>(
            &vks,
            &public_inputs_ref,
            &proofs_ref,
            &extra_msgs,
        )
        .is_err());
        public_inputs_ref[0] = tmp_pi_ref;

        proofs_ref[0] = proofs_ref[1];
        assert!(PlonkKzgSnark::<E>::batch_verify::<T>(
            &vks,
            &public_inputs_ref,
            &proofs_ref,
            &extra_msgs,
        )
        .is_err());

        Ok(())
    }

    #[test]
    fn test_inconsistent_pub_input_len() -> Result<(), PlonkError> {
        // merlin transcripts
        test_inconsistent_pub_input_len_helper::<Bn254, Fq254, _, StandardTranscript>(
            PlonkType::TurboPlonk,
        )?;
        test_inconsistent_pub_input_len_helper::<Bn254, Fq254, _, StandardTranscript>(
            PlonkType::UltraPlonk,
        )?;
        test_inconsistent_pub_input_len_helper::<Bls12_377, Fq377, _, StandardTranscript>(
            PlonkType::TurboPlonk,
        )?;
        test_inconsistent_pub_input_len_helper::<Bls12_377, Fq377, _, StandardTranscript>(
            PlonkType::UltraPlonk,
        )?;
        test_inconsistent_pub_input_len_helper::<Bls12_381, Fq381, _, StandardTranscript>(
            PlonkType::TurboPlonk,
        )?;
        test_inconsistent_pub_input_len_helper::<Bls12_381, Fq381, _, StandardTranscript>(
            PlonkType::UltraPlonk,
        )?;
        test_inconsistent_pub_input_len_helper::<BW6_761, Fq761, _, StandardTranscript>(
            PlonkType::TurboPlonk,
        )?;
        test_inconsistent_pub_input_len_helper::<BW6_761, Fq761, _, StandardTranscript>(
            PlonkType::UltraPlonk,
        )?;

        // rescue transcripts
        // currently only available for bls12-377
        test_inconsistent_pub_input_len_helper::<Bls12_377, Fq377, _, RescueTranscript<Fq377>>(
            PlonkType::TurboPlonk,
        )?;
        test_inconsistent_pub_input_len_helper::<Bls12_377, Fq377, _, RescueTranscript<Fq377>>(
            PlonkType::UltraPlonk,
        )?;

        // Solidity-friendly keccak256 transcript
        test_inconsistent_pub_input_len_helper::<Bls12_381, Fq381, _, SolidityTranscript>(
            PlonkType::TurboPlonk,
        )?;

        Ok(())
    }

    fn test_inconsistent_pub_input_len_helper<E, F, P, T>(
        plonk_type: PlonkType,
    ) -> Result<(), PlonkError>
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
        F: RescueParameter + PrimeField,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        T: Transcript,
        E::ScalarField: EmulationConfig<F>,
    {
        // 1. Simulate universal setup
        let rng = &mut test_rng();

        // 2. Create circuits
        let mut cs1: PlonkCircuit<E::ScalarField> = match plonk_type {
            PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
            PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(2),
        };
        let var = cs1.create_variable(E::ScalarField::from(1u8))?;
        cs1.enforce_constant(var, E::ScalarField::from(1u8))?;
        cs1.finalize_for_arithmetization()?;
        let mut cs2: PlonkCircuit<E::ScalarField> = match plonk_type {
            PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
            PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(2),
        };
        cs2.create_public_variable(E::ScalarField::from(1u8))?;
        cs2.finalize_for_arithmetization()?;

        // 3. Preprocessing
        let size_one = cs1.srs_size()?;
        let size_two = cs2.srs_size()?;
        let size = ark_std::cmp::max(size_one, size_two);
        let srs = PlonkKzgSnark::<E>::universal_setup_for_testing(size, rng)?;
        let (pk1, vk1) = PlonkKzgSnark::<E>::preprocess(&srs, &cs1)?;
        let (pk2, vk2) = PlonkKzgSnark::<E>::preprocess(&srs, &cs2)?;

        // 4. Proving
        assert!(PlonkKzgSnark::<E>::prove::<_, _, T>(rng, &cs2, &pk1, None).is_err());
        let proof2 = PlonkKzgSnark::<E>::prove::<_, _, T>(rng, &cs2, &pk2, None)?;

        // 5. Verification
        assert!(
            PlonkKzgSnark::<E>::verify::<T>(&vk2, &[E::ScalarField::from(1u8)], &proof2, None,)
                .is_ok()
        );
        // wrong verification key
        assert!(
            PlonkKzgSnark::<E>::verify::<T>(&vk1, &[E::ScalarField::from(1u8)], &proof2, None,)
                .is_err()
        );
        // wrong public input
        assert!(PlonkKzgSnark::<E>::verify::<T>(&vk2, &[], &proof2, None).is_err());

        Ok(())
    }

    #[test]
    fn test_plonk_prover_polynomials() -> Result<(), PlonkError> {
        // merlin transcripts
        test_plonk_prover_polynomials_helper::<Bn254, Fq254, _, StandardTranscript>(
            PlonkType::TurboPlonk,
        )?;
        test_plonk_prover_polynomials_helper::<Bls12_377, Fq377, _, StandardTranscript>(
            PlonkType::TurboPlonk,
        )?;
        test_plonk_prover_polynomials_helper::<Bls12_381, Fq381, _, StandardTranscript>(
            PlonkType::TurboPlonk,
        )?;
        test_plonk_prover_polynomials_helper::<BW6_761, Fq761, _, StandardTranscript>(
            PlonkType::TurboPlonk,
        )?;
        test_plonk_prover_polynomials_helper::<Bn254, Fq254, _, StandardTranscript>(
            PlonkType::UltraPlonk,
        )?;
        test_plonk_prover_polynomials_helper::<Bls12_377, Fq377, _, StandardTranscript>(
            PlonkType::UltraPlonk,
        )?;
        test_plonk_prover_polynomials_helper::<Bls12_381, Fq381, _, StandardTranscript>(
            PlonkType::UltraPlonk,
        )?;
        test_plonk_prover_polynomials_helper::<BW6_761, Fq761, _, StandardTranscript>(
            PlonkType::UltraPlonk,
        )?;

        // rescue transcripts
        // currently only available for bls12-377
        test_plonk_prover_polynomials_helper::<Bls12_377, Fq377, _, RescueTranscript<Fq377>>(
            PlonkType::TurboPlonk,
        )?;
        test_plonk_prover_polynomials_helper::<Bls12_377, Fq377, _, RescueTranscript<Fq377>>(
            PlonkType::UltraPlonk,
        )?;

        // Solidity-friendly keccak256 transcript
        test_plonk_prover_polynomials_helper::<Bls12_381, Fq381, _, SolidityTranscript>(
            PlonkType::TurboPlonk,
        )?;

        Ok(())
    }

    fn test_plonk_prover_polynomials_helper<E, F, P, T>(
        plonk_type: PlonkType,
    ) -> Result<(), PlonkError>
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
        F: RescueParameter + PrimeField,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        T: Transcript,
        E::ScalarField: EmulationConfig<F>,
    {
        // 1. Simulate universal setup
        let rng = &mut test_rng();
        let n = 64;
        let max_degree = n + 2;
        let srs = PlonkKzgSnark::<E>::universal_setup_for_testing(max_degree, rng)?;

        // 2. Create the circuit
        let circuit = gen_circuit_for_test(10, 3, plonk_type)?;
        assert!(circuit.num_gates() <= n);

        // 3. Preprocessing
        let (pk, _) = PlonkKzgSnark::<E>::preprocess(&srs, &circuit)?;

        // 4. Proving
        let (_, oracles, challenges, _) =
            PlonkKzgSnark::<E>::batch_prove_internal::<_, _, T>(rng, &[&circuit], &[&pk], None)?;

        // 5. Check that the targeted polynomials evaluate to zero on the vanishing set.
        check_plonk_prover_polynomials(plonk_type, &oracles[0], &pk, &challenges)?;

        Ok(())
    }

    fn check_plonk_prover_polynomials<E: Pairing>(
        plonk_type: PlonkType,
        oracles: &Oracles<E::ScalarField>,
        pk: &ProvingKey<E>,
        challenges: &Challenges<E::ScalarField>,
    ) -> Result<(), PlonkError> {
        check_circuit_polynomial_on_vanishing_set(oracles, pk)?;
        check_perm_polynomials_on_vanishing_set(oracles, pk, challenges)?;
        if plonk_type == PlonkType::UltraPlonk {
            check_lookup_polynomials_on_vanishing_set(oracles, pk, challenges)?;
        }

        Ok(())
    }

    fn check_circuit_polynomial_on_vanishing_set<E: Pairing>(
        oracles: &Oracles<E::ScalarField>,
        pk: &ProvingKey<E>,
    ) -> Result<(), PlonkError> {
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

    fn check_perm_polynomials_on_vanishing_set<E: Pairing>(
        oracles: &Oracles<E::ScalarField>,
        pk: &ProvingKey<E>,
        challenges: &Challenges<E::ScalarField>,
    ) -> Result<(), PlonkError> {
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
            E::ScalarField::one()
        );

        Ok(())
    }

    fn check_lookup_polynomials_on_vanishing_set<E: Pairing>(
        oracles: &Oracles<E::ScalarField>,
        pk: &ProvingKey<E>,
        challenges: &Challenges<E::ScalarField>,
    ) -> Result<(), PlonkError> {
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

    #[test]
    fn test_proof_from_to_fields() -> Result<(), PlonkError> {
        test_proof_from_to_fields_helper::<Bn254, _>()?;
        test_proof_from_to_fields_helper::<Bls12_381, _>()?;
        test_proof_from_to_fields_helper::<Bls12_377, _>()?;
        test_proof_from_to_fields_helper::<BW6_761, _>()?;
        Ok(())
    }

    fn test_proof_from_to_fields_helper<E, P>() -> Result<(), PlonkError>
    where
        E: Pairing<G1Affine = Affine<P>, G1 = Projective<P>>,
        E::BaseField: RescueParameter + PrimeField,
        P: HasTEForm<BaseField = E::BaseField, ScalarField = E::ScalarField>,
        E::ScalarField: EmulationConfig<E::BaseField>,
    {
        let rng = &mut jf_utils::test_rng();
        let circuit = gen_circuit_for_test(3, 4, PlonkType::TurboPlonk)?;
        let max_degree = 80;
        let srs = PlonkKzgSnark::<E>::universal_setup_for_testing(max_degree, rng)?;

        let (pk, _) = PlonkKzgSnark::<E>::preprocess(&srs, &circuit)?;
        let proof =
            PlonkKzgSnark::<E>::prove::<_, _, StandardTranscript>(rng, &circuit, &pk, None)?;

        let base_fields: Vec<E::BaseField> = proof.clone().into();
        let res: Proof<E> = base_fields.try_into()?;
        assert_eq!(res, proof);

        Ok(())
    }

    #[test]
    fn test_serde() -> Result<(), PlonkError> {
        // merlin transcripts
        test_serde_helper::<Bn254, Fq254, _, StandardTranscript>(PlonkType::TurboPlonk)?;
        test_serde_helper::<Bn254, Fq254, _, StandardTranscript>(PlonkType::UltraPlonk)?;
        test_serde_helper::<Bls12_377, Fq377, _, StandardTranscript>(PlonkType::TurboPlonk)?;
        test_serde_helper::<Bls12_377, Fq377, _, StandardTranscript>(PlonkType::UltraPlonk)?;
        test_serde_helper::<Bls12_381, Fq381, _, StandardTranscript>(PlonkType::TurboPlonk)?;
        test_serde_helper::<Bls12_381, Fq381, _, StandardTranscript>(PlonkType::UltraPlonk)?;
        test_serde_helper::<BW6_761, Fq761, _, StandardTranscript>(PlonkType::TurboPlonk)?;
        test_serde_helper::<BW6_761, Fq761, _, StandardTranscript>(PlonkType::UltraPlonk)?;

        // rescue transcripts
        // currently only available for bls12-377
        test_serde_helper::<Bls12_377, Fq377, _, RescueTranscript<Fq377>>(PlonkType::TurboPlonk)?;
        test_serde_helper::<Bls12_377, Fq377, _, RescueTranscript<Fq377>>(PlonkType::UltraPlonk)?;

        // Solidity-friendly keccak256 transcript
        test_serde_helper::<Bls12_381, Fq381, _, SolidityTranscript>(PlonkType::TurboPlonk)?;

        Ok(())
    }

    fn test_serde_helper<E, F, P, T>(plonk_type: PlonkType) -> Result<(), PlonkError>
    where
        E: Pairing<BaseField = F, G1Affine = Affine<P>, G1 = Projective<P>>,
        F: RescueParameter + PrimeField,
        P: HasTEForm<BaseField = F, ScalarField = E::ScalarField>,
        T: Transcript,
        E::ScalarField: EmulationConfig<E::BaseField>,
    {
        let rng = &mut jf_utils::test_rng();
        let circuit = gen_circuit_for_test(3, 4, plonk_type)?;
        let max_degree = 80;
        let srs = PlonkKzgSnark::<E>::universal_setup_for_testing(max_degree, rng)?;

        let (pk, vk) = PlonkKzgSnark::<E>::preprocess(&srs, &circuit)?;
        let proof = PlonkKzgSnark::<E>::prove::<_, _, T>(rng, &circuit, &pk, None)?;

        let mut ser_bytes = Vec::new();
        srs.serialize_compressed(&mut ser_bytes)?;
        let de = UniversalSrs::<E>::deserialize_compressed(&ser_bytes[..])?;
        assert_eq!(de, srs);

        let mut ser_bytes = Vec::new();
        pk.serialize_compressed(&mut ser_bytes)?;
        let de = ProvingKey::<E>::deserialize_compressed(&ser_bytes[..])?;
        assert_eq!(de, pk);

        let mut ser_bytes = Vec::new();
        vk.serialize_compressed(&mut ser_bytes)?;
        let de = VerifyingKey::<E>::deserialize_compressed(&ser_bytes[..])?;
        assert_eq!(de, vk);

        let mut ser_bytes = Vec::new();
        proof.serialize_compressed(&mut ser_bytes)?;
        let de = Proof::<E>::deserialize_compressed(&ser_bytes[..])?;
        assert_eq!(de, proof);

        Ok(())
    }
}
