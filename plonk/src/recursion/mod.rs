//! Code for recursively proving plonk proofs over a cycle of curves.

use ark_bn254::{Bn254, Fq as Fq254, Fr as Fr254};
use ark_poly::DenseMultilinearExtension;
use ark_std::time::Instant;
use ark_std::{
    cfg_chunks, cfg_into_iter, format, rand::SeedableRng, string::ToString, sync::Arc, vec,
    vec::Vec, One, Zero,
};

use itertools::Itertools;
use rand_chacha::ChaCha20Rng;

use rayon::prelude::*;

use jf_relation::Circuit;

use circuits::{Kzg, Zmorph};
use jf_primitives::{
    pcs::prelude::{UnivariateKzgProof, UnivariateUniversalParams},
    rescue::sponge::RescueCRHF,
};
use jf_relation::{errors::CircuitError, PlonkCircuit, Variable};
use merge_functions::{
    decider_circuit, prove_bn254_accumulation, prove_grumpkin_accumulation, Bn254CircuitOutput,
    Bn254Output, Bn254RecursiveInfo, GrumpkinCircuitOutput, GrumpkinOutput, GrumpkinRecursiveInfo,
};
use nf_curves::grumpkin::Grumpkin;

use crate::{
    errors::PlonkError,
    nightfall::{
        accumulation::accumulation_structs::{AtomicInstance, PCSWitness},
        ipa_structs::{ProvingKey, VerifyingKey, VK},
        mle::{
            mle_structs::{GateInfo, MLEProvingKey, MLEVerifyingKey},
            MLEPlonk,
        },
        FFTPlonk, UnivariateUniversalIpaParams,
    },
    proof_system::{
        structs::{Proof, ProvingKey as JFProvingKey},
        PlonkKzgSnark, UniversalSNARK,
    },
    transcript::SolidityTranscript,
};

pub mod circuits;
pub mod merge_functions;

// There are 5 main circuits/proofs this trait uses to do this, two "base circuits", one defined over Bn254 and one defined over Grumpkin,
// two "compress circuit", one for each of the curves and a final circuit that out puts a single Bn254 KZG proof.
//
// Throughout public inputs need to be forwarded a certain number of layers down the recursion. In order to achieve this public inputs will be sorted
// into standard blocks. each of the compress circuits takes in public inputs from the four proofs two levels down and the two proofs one level down as private inputs.
// These will be vectors of field elements ordered to be "specific implementation public input, msm scalars, forwarded accumulators, output accumulators, old pi hashes".
// We use UltraPlonk proofs throughout.
//
// There are also some inputs that are taken from the base proofs, these are used to link the base proofs to the final proof.

type GrumpkinOut = (PlonkCircuit<Fq254>, GrumpkinCircuitOutput);
type Bn254Out = (PlonkCircuit<Fr254>, Bn254CircuitOutput);

/// Trait used to define a recursive proving engine working over the Bn254 and Grumpkin curves. The base input proofs are assumed to be Bn254 proofs.
pub trait RecursiveProver {
    /// This function applies implementation specific checks to in the first Bn254 circuit in the recursive prover
    fn base_bn254_checks(
        specific_pis: &[Vec<Variable>],
        circuit: &mut PlonkCircuit<Fr254>,
    ) -> Result<Vec<Variable>, CircuitError>;
    /// This function is for any extra checks that require more information than those supplied in the base proofs.
    fn base_bn254_extra_checks(
        specific_pis: &[Variable],
        circuit: &mut PlonkCircuit<Fr254>,
    ) -> Result<Vec<Variable>, CircuitError>;
    /// This function applies implementation specific checks in the first Grumpkin circuit in the recursive prover.
    fn base_grumpkin_checks(
        specific_pis: &[Vec<Variable>],
        circuit: &mut PlonkCircuit<Fq254>,
    ) -> Result<Vec<Variable>, CircuitError>;
    /// This function applies implementation specific checks for the Bn254 merge circuits (i.e. every Bn254 circuit apart from the first).
    fn bn254_merge_circuit_checks(
        specific_pis: &[Vec<Variable>],
        circuit: &mut PlonkCircuit<Fr254>,
    ) -> Result<Vec<Variable>, CircuitError>;
    /// This function applies implementation specific checks for the Grumpkin merge circuits (i.e. every Grumpkin circuit apart from the first).
    fn grumpkin_merge_circuit_checks(
        specific_pis: &[Vec<Variable>],
        circuit: &mut PlonkCircuit<Fq254>,
    ) -> Result<Vec<Variable>, CircuitError>;
    /// This function applies the final implementation specific checks that are run in the decider circuit.
    fn decider_circuit_checks(
        specific_pis: &[Vec<Variable>],
        circuit: &mut PlonkCircuit<Fr254>,
    ) -> Result<Vec<Variable>, CircuitError>;
    /// Retrieve the list of acceptable verification key hashes
    fn get_vk_hash_list() -> Vec<Fr254>;
    /// Retrieves the base Grumpkin proving key.
    fn get_base_grumpkin_pk() -> MLEProvingKey<Zmorph>;
    /// Retrieves the base Bn254 proving key.
    fn get_base_bn254_pk() -> ProvingKey<Kzg>;
    /// Retrieves the merge Grumpkin proving key.
    fn get_merge_grumpkin_pk() -> MLEProvingKey<Zmorph>;
    /// Retrieves the merge Bn254 proving key.
    fn get_merge_bn254_pk_4() -> ProvingKey<Kzg>;
    /// Retrieves the merge Bn254 proving key.
    fn get_merge_bn254_pk_16() -> ProvingKey<Kzg>;
    /// Retrieves the final proving key.
    fn get_decider_pk() -> JFProvingKey<Bn254>;
    /// Stores the base Grumpkin proving key.
    fn store_base_grumpkin_pk(pk: MLEProvingKey<Zmorph>) -> Option<()>;
    /// Stores the base Bn254 proving key.
    fn store_base_bn254_pk(pk: ProvingKey<Kzg>) -> Option<()>;
    /// Stores the merge Grumpkin proving key.
    fn store_merge_grumpkin_pk(pk: MLEProvingKey<Zmorph>) -> Option<()>;
    /// Stores the merge Bn254 proving key.
    fn store_merge_bn254_pk(pk: ProvingKey<Kzg>) -> Option<()>;
    /// Stores the decider proving key.
    fn store_decider_pk(pk: JFProvingKey<Bn254>) -> Option<()>;
    /// This function takes in the input proofs and outputs [`GrumpkinCircuitOutput`] to be taken in by the next function.
    fn base_grumpkin_circuit(
        outputs: &[Bn254Output; 2],
        specific_pi: &[Vec<Fr254>; 2],
        input_vks: &[VerifyingKey<Kzg>; 2],
        kzg_srs: &UnivariateUniversalParams<Bn254>,
    ) -> Result<GrumpkinOut, PlonkError> {
        ark_std::println!("intermediate structure: base_grumpkin_circuit");
        // record how long it takes to run this function
        let start  = Instant::now();
        let mut circuit = PlonkCircuit::<Fq254>::new_ultra_plonk(12);
        let bn254info = initial_bn254_info(outputs, specific_pi, kzg_srs);
        // We make a dummy VK at this level because it won't be used in the circuit here.
        let vk_grumpkin = MLEVerifyingKey::<Zmorph> {
            selector_commitments: vec![],
            permutation_commitments: vec![],
            lookup_verifying_key: None,
            pcs_verifier_params: UnivariateUniversalIpaParams::default(),
            gate_info: GateInfo::<Fq254> {
                max_degree: 0,
                products: vec![],
            },
            num_inputs: 0,
        };

        let circuit_output = prove_bn254_accumulation::<true>(
            &bn254info,
            input_vks,
            &vk_grumpkin,
            Self::base_grumpkin_checks,
            &mut circuit,
        )?;

        #[cfg(test)]
        {
            ark_std::println!(
                "base grumpkin circuit size pre-finalize: {}",
                circuit.num_gates()
            );
        }

        circuit.finalize_for_recursive_mle_arithmetization::<RescueCRHF<Fr254>>()?;

        #[cfg(test)]
        {
            let pi = circuit.public_input()?;
            circuit.check_circuit_satisfiability(&pi)?;
            ark_std::println!("base grumpkin circuit size: {}", circuit.num_gates());
        }
        let elapsed = start.elapsed();
        ark_std::println!(
            "base_grumpkin_circuit took: {} seconds",
            elapsed.as_secs_f64()
        );
        Ok((circuit, circuit_output))
    }
    /// This function takes in [`GrumpkinOut`] types, proves the circuits and then produces another circuit proving their correct accumulation.
    fn base_bn254_circuit(
        outputs: [GrumpkinOut; 2],
        base_grumpkin_pk: &MLEProvingKey<Zmorph>,
        input_vks: &[VerifyingKey<Kzg>; 4],
        extra_base_info: &[Fr254],
    ) -> Result<Bn254Out, PlonkError> {
        ark_std::println!("intermediate structure: base_bn254_circuit");
        let start  = Instant::now();
        let (circuits, grumpkin_circuit_outs_vec): (
            Vec<PlonkCircuit<Fq254>>,
            Vec<GrumpkinCircuitOutput>,
        ) = outputs.into_iter().unzip();

        let circuit_outputs: [GrumpkinCircuitOutput; 2] =
            grumpkin_circuit_outs_vec.try_into().map_err(|_| {
                PlonkError::InvalidParameters(
                    "Could not create an array of length 2 for GrumpkinCircuitOutput".to_string(),
                )
            })?;
        let grumpkin_outputs: [GrumpkinOutput; 2] = cfg_into_iter!(circuits)
            .map(|circuit| {
                let rng = &mut jf_utils::test_rng();
                MLEPlonk::<Zmorph>::recursive_prove(rng, &circuit, base_grumpkin_pk, None)
            })
            .collect::<Result<Vec<GrumpkinOutput>, PlonkError>>()?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters(
                    "Could not map to a fixed length array of size 2 for GrumpkinOutput"
                        .to_string(),
                )
            })?;

        let num_vars = grumpkin_outputs[0].proof.polynomial.num_vars;
        let mut grumpkin_info =
            GrumpkinRecursiveInfo::from_parts(grumpkin_outputs, circuit_outputs);

        // We have to change the old grumpkin accumulators to contain polynomials of the correct size.

        let poly_evals = [vec![Fq254::one()], vec![Fq254::zero(); (1 << num_vars) - 1]].concat();
        let poly = Arc::new(DenseMultilinearExtension::<Fq254>::from_evaluations_vec(
            num_vars, poly_evals,
        ));
        let comm = base_grumpkin_pk.pcs_prover_params.g_bases[0];
        let point = vec![Fq254::zero(); num_vars];
        let grumpkin_pcs_witness = PCSWitness::new(poly, comm, Fq254::one(), point);

        let old_accumulators = [
            grumpkin_pcs_witness.clone(),
            grumpkin_pcs_witness.clone(),
            grumpkin_pcs_witness.clone(),
            grumpkin_pcs_witness,
        ];

        grumpkin_info.old_accumulators = old_accumulators;
        let mut circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(12);

        let hash_list = Self::get_vk_hash_list();
        for vk in input_vks.iter() {
            Self::generate_vk_check_constraint(vk.hash(), &hash_list, &mut circuit)?;
        }
        // Perform any extra checks that only happen at base level.
        let extra_checks_pi = extra_base_info
            .iter()
            .map(|ei| circuit.create_variable(*ei))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        let extra_checks_pi_out = Self::base_bn254_extra_checks(&extra_checks_pi, &mut circuit)?;
        extra_checks_pi_out
            .iter()
            .try_for_each(|pi| circuit.set_variable_public(*pi))?;

        let extra_checks_pi_field = extra_checks_pi_out
            .into_iter()
            .map(|pi| circuit.witness(pi))
            .collect::<Result<Vec<Fr254>, CircuitError>>()?;
        let mut bn254_circuit_out = prove_grumpkin_accumulation::<true>(
            &grumpkin_info,
            input_vks,
            base_grumpkin_pk,
            Self::base_bn254_checks,
            &mut circuit,
        )?;

        bn254_circuit_out.specific_pi =
            [extra_checks_pi_field, bn254_circuit_out.specific_pi].concat();

        #[cfg(test)]
        {
            ark_std::println!(
                "base bn254 circuit size pre-finalize: {}",
                circuit.num_gates()
            );
        }

        circuit.finalize_for_recursive_arithmetization::<RescueCRHF<Fq254>>()?;

        // Run the following code only when testing
        #[cfg(test)]
        {
            let pi = circuit.public_input()?;
            circuit.check_circuit_satisfiability(&pi)?;
            ark_std::println!("base bn254 circuit size: {}", circuit.num_gates());
        }
        let elapsed = start.elapsed();
        ark_std::println!(
            "base_bn254_circuit took: {} seconds",
            elapsed.as_secs_f64()
        );
        Ok((circuit, bn254_circuit_out))
    }
    /// This function takes in [`Bn254Out`] types, proves the circuits and then produces another circuit proving their correct accumulation.
    fn merge_grumpkin_circuit(
        outputs: [Bn254Out; 2],
        base_bn254_pk: &ProvingKey<Kzg>,
        base_grumpkin_pk: &MLEProvingKey<Zmorph>,
    ) -> Result<GrumpkinOut, PlonkError> {
        ark_std::println!("intermediate structure: merge_grumpkin_circuit");
        let start  = Instant::now();
        let (circuits, bn254_circuit_outs_vec): (
            Vec<PlonkCircuit<Fr254>>,
            Vec<Bn254CircuitOutput>,
        ) = outputs.into_iter().unzip();

        let circuit_outputs: [Bn254CircuitOutput; 2] =
            bn254_circuit_outs_vec.try_into().map_err(|_| {
                PlonkError::InvalidParameters(
                    "Could not create an array of length 2 for Bn254CircuitOutput".to_string(),
                )
            })?;

        let bn254_outputs: [Bn254Output; 2] = circuits
            .into_iter()
            .map(|circuit| {
                let rng = &mut jf_utils::test_rng();
                FFTPlonk::<Kzg>::recursive_prove(rng, &circuit, base_bn254_pk, None)
            })
            .collect::<Result<Vec<Bn254Output>, PlonkError>>()?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters(
                    "Could not map to a fixed length array of size 2 for Bn254Output".to_string(),
                )
            })?;

        let bn254info = Bn254RecursiveInfo::from_parts(bn254_outputs, circuit_outputs);

        let mut circuit = PlonkCircuit::<Fq254>::new_ultra_plonk(12);

        ark_std::println!(
            "JJ: am about to call prove_bn254_accumulation"
        );

        let grumpkin_circuit_out = prove_bn254_accumulation::<false>(
            &bn254info,
            &[base_bn254_pk.vk.clone(), base_bn254_pk.vk.clone()],
            &base_grumpkin_pk.verifying_key,
            Self::grumpkin_merge_circuit_checks,
            &mut circuit,
        )?;

        #[cfg(test)]
        {
            ark_std::println!(
                "merge grumpkin circuit size pre-finalize: {}",
                circuit.num_gates()
            );
        }

        circuit.finalize_for_recursive_mle_arithmetization::<RescueCRHF<Fr254>>()?;

        // Run the following code only when testing
        #[cfg(test)]
        {
            let pi = circuit.public_input()?;
            circuit.check_circuit_satisfiability(&pi)?;
            ark_std::println!("merge grumpkin circuit size: {}", circuit.num_gates());
        }
        let elapsed = start.elapsed();
        ark_std::println!(
            "merge grumpkin circuit took: {} seconds",
            elapsed.as_secs_f64()
        );

        Ok((circuit, grumpkin_circuit_out))
    }
    /// This function takes in [`GrumpkinOut`] types, proves the circuits and then produces another circuit proving their correct accumulation.
    fn merge_bn254_circuit(
        outputs: [GrumpkinOut; 2],
        merge_grumpkin_pk: &MLEProvingKey<Zmorph>,
        bn254_pk: &ProvingKey<Kzg>,
    ) -> Result<Bn254Out, PlonkError> {
        ark_std::println!("intermediate structure: merge_bn254_circuit");
        let start  = Instant::now();
        let (circuits, grumpkin_circuit_outs_vec): (
            Vec<PlonkCircuit<Fq254>>,
            Vec<GrumpkinCircuitOutput>,
        ) = outputs.into_iter().unzip();

        let circuit_outputs: [GrumpkinCircuitOutput; 2] =
            grumpkin_circuit_outs_vec.try_into().map_err(|_| {
                PlonkError::InvalidParameters(
                    "Could not create an array of length 2 for GrumpkinCircuitOutput".to_string(),
                )
            })?;
        let grumpkin_outputs: [GrumpkinOutput; 2] = cfg_into_iter!(circuits)
            .map(|circuit| {
                let rng = &mut jf_utils::test_rng();
                MLEPlonk::<Zmorph>::recursive_prove(rng, &circuit, merge_grumpkin_pk, None)
            })
            .collect::<Result<Vec<GrumpkinOutput>, PlonkError>>()?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters(
                    "Could not map to a fixed length array of size 2 for GrumpkinOutput"
                        .to_string(),
                )
            })?;

        let grumpkin_info = GrumpkinRecursiveInfo::from_parts(grumpkin_outputs, circuit_outputs);

        let mut circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(12);

        let bn254_circuit_out = prove_grumpkin_accumulation::<false>(
            &grumpkin_info,
            &[
                bn254_pk.vk.clone(),
                bn254_pk.vk.clone(),
                bn254_pk.vk.clone(),
                bn254_pk.vk.clone(),
            ],
            merge_grumpkin_pk,
            Self::bn254_merge_circuit_checks,
            &mut circuit,
        )?;

        circuit.finalize_for_recursive_arithmetization::<RescueCRHF<Fq254>>()?;

        // Run the following code only when testing
        #[cfg(test)]
        {
            let pi = circuit.public_input()?;
            circuit.check_circuit_satisfiability(&pi)?;
            ark_std::println!("merge bn254 size: {}", circuit.num_gates());
        }
        let elapsed = start.elapsed();
        ark_std::println!(
            "merge_bn254_circuit took: {} seconds",
            elapsed.as_secs_f64()
        );
        Ok((circuit, bn254_circuit_out))
    }
    /// The decider circuit that fully verifies the Grumpkin accumulator along with the final grumpkin proof.
    fn decider_circuit(
        grumpkin_outputs: [GrumpkinOut; 2],
        extra_decider_info: &[Fr254],
        merge_grumpkin_pk: &MLEProvingKey<Zmorph>,
        merge_bn254_pk: &ProvingKey<Kzg>,
    ) -> Result<DeciderOut, PlonkError> {
        ark_std::println!("intermediate structure: decider_circuit");
        let start  = Instant::now();
        let (circuits, grumpkin_circuit_outs_vec): (
            Vec<PlonkCircuit<Fq254>>,
            Vec<GrumpkinCircuitOutput>,
        ) = grumpkin_outputs.into_iter().unzip();

        let circuit_outputs: [GrumpkinCircuitOutput; 2] =
            grumpkin_circuit_outs_vec.try_into().map_err(|_| {
                PlonkError::InvalidParameters(
                    "Could not create an array of length 2 for GrumpkinCircuitOutput".to_string(),
                )
            })?;
        let grumpkin_outputs: [GrumpkinOutput; 2] = cfg_into_iter!(circuits)
            .map(|circuit| {
                let rng = &mut jf_utils::test_rng();
                MLEPlonk::<Zmorph>::recursive_prove(rng, &circuit, merge_grumpkin_pk, None)
            })
            .collect::<Result<Vec<GrumpkinOutput>, PlonkError>>()?
            .try_into()
            .map_err(|_| {
                PlonkError::InvalidParameters(
                    "Could not map to a fixed length array of size 2 for GrumpkinOutput"
                        .to_string(),
                )
            })?;

        let grumpkin_info = GrumpkinRecursiveInfo::from_parts(grumpkin_outputs, circuit_outputs);

        let mut circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(12);

        let pi_out = decider_circuit(
            &grumpkin_info,
            &merge_bn254_pk.vk,
            merge_grumpkin_pk,
            extra_decider_info,
            Self::decider_circuit_checks,
            &mut circuit,
        )?;

        // #[cfg(test)]
        // {
        ark_std::println!(
            "decider circuit circuit size pre-finalize: {}",
            circuit.num_gates()
        );
        // }

        circuit.finalize_for_arithmetization()?;
        ark_std::println!(
            "decider circuit circuit size after finalize: {}",
            circuit.num_gates()
        );

        // Run the following code only when testing
        #[cfg(test)]
        {
            let pi = circuit.public_input()?;
            circuit.check_circuit_satisfiability(&pi)?;
            ark_std::println!("decider circuit size: {}", circuit.num_gates());
        }

        let GrumpkinRecursiveInfo {
            forwarded_accumulators,
            ..
        } = grumpkin_info;

        let elapsed = start.elapsed();
        ark_std::println!(
            "decider_circuit took: {} seconds",
            elapsed.as_secs_f64()
        );
        Ok(DeciderOut::new(circuit, pi_out, forwarded_accumulators))
    }

    /// The function for preprocessing the circuits and storing the keys produced
    fn preprocess(
        outputs: &[(Bn254Output, VerifyingKey<Kzg>)],
        specific_pi: &[Vec<Fr254>],
        extra_base_info: &[Vec<Fr254>],
        extra_decider_info: &[Fr254],
        ipa_srs: &UnivariateUniversalIpaParams<Grumpkin>,
        kzg_srs: &UnivariateUniversalParams<Bn254>,
    ) -> Result<(), PlonkError> {
        // First check that we have the same number of outputs and pi's and that they are also non-zero in length
        if outputs.len() != specific_pi.len() {
            return Err(PlonkError::InvalidParameters(format!(
                "The number of outputs: {} does not equal the number of public input lists: {}",
                outputs.len(),
                specific_pi.len()
            )));
        }

        if outputs.is_empty() {
            return Err(PlonkError::InvalidParameters(
                "Need a non-zero number of proofs".to_string(),
            ));
        }

        if outputs.len().next_power_of_two() != outputs.len() {
            return Err(PlonkError::InvalidParameters(
                "Outputs length is not a power of two".to_string(),
            ));
        }

        let (outputs, vks): (Vec<Bn254Output>, Vec<VerifyingKey<Kzg>>) =
            outputs.iter().map(|(o, c)| (o.clone(), c.clone())).unzip();

        let base_grumpkin_out = cfg_chunks!(outputs, 2)
            .zip(cfg_chunks!(specific_pi, 2))
            .zip(cfg_chunks!(vks, 2))
            .map(|((chunk_one, chunk_two), chunk_three)| {
                let out_slice: &[Bn254Output; 2] = chunk_one.try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length slice".to_string(),
                    )
                })?;
                let pi_slice: &[Vec<Fr254>; 2] = chunk_two.try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length slice".to_string(),
                    )
                })?;
                let vk_slice: &[VerifyingKey<Kzg>; 2] = chunk_three.try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length slice".to_string(),
                    )
                })?;
                Self::base_grumpkin_circuit(out_slice, pi_slice, vk_slice, kzg_srs)
            })
            .collect::<Result<Vec<GrumpkinOut>, PlonkError>>()?;

        // We know the outputs is non-zero so we can safely unwrap here
        let base_grumpkin_circuit = &base_grumpkin_out[0].0;

        let (base_grumpkin_pk, _) = MLEPlonk::<Zmorph>::preprocess(ipa_srs, base_grumpkin_circuit)?;

        // Produce and store the base Bn254 proving key
        let base_grumpkin_chunks: Vec<[GrumpkinOut; 2]> = base_grumpkin_out
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|chunk| {
                chunk.collect::<Vec<GrumpkinOut>>().try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<[GrumpkinOut; 2]>, PlonkError>>()?;
        let vk_chunks = vks
            .into_iter()
            .chunks(4)
            .into_iter()
            .map(|chunk| {
                chunk
                    .collect::<Vec<VerifyingKey<Kzg>>>()
                    .try_into()
                    .map_err(|_| {
                        PlonkError::InvalidParameters(
                            "Could not convert to fixed length array".to_string(),
                        )
                    })
            })
            .collect::<Result<Vec<[VerifyingKey<Kzg>; 4]>, PlonkError>>()?;
        let base_bn254_out = cfg_into_iter!(base_grumpkin_chunks)
            .zip(cfg_into_iter!(vk_chunks))
            .zip(cfg_into_iter!(extra_base_info))
            .map(|((chunk, vk_chunk), extra_info)| {
                Self::base_bn254_circuit(chunk, &base_grumpkin_pk, &vk_chunk, extra_info)
            })
            .collect::<Result<Vec<Bn254Out>, PlonkError>>()?;
        let base_bn254_circuit = &base_bn254_out[0].0;

        let (base_bn254_pk, _) = FFTPlonk::<Kzg>::preprocess(kzg_srs, base_bn254_circuit)?;

        // Produce the Grumpkin merge proving key
        let base_bn254_chunks: Vec<[Bn254Out; 2]> = base_bn254_out
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|chunk| {
                chunk.collect::<Vec<Bn254Out>>().try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<[Bn254Out; 2]>, PlonkError>>()?;
        let merge_grumpkin_out = base_bn254_chunks
            .into_iter()
            .map(|chunk| Self::merge_grumpkin_circuit(chunk, &base_bn254_pk, &base_grumpkin_pk))
            .collect::<Result<Vec<GrumpkinOut>, PlonkError>>()?;

        let merge_grumpkin_circuit = &merge_grumpkin_out[0].0;

        let (merge_grumpkin_pk, _) =
            MLEPlonk::<Zmorph>::preprocess(ipa_srs, merge_grumpkin_circuit)?;

        // Produce the Bn254 merge proving key
        let merge_grumpkin_chunks: Vec<[GrumpkinOut; 2]> = merge_grumpkin_out
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|chunk| {
                chunk.collect::<Vec<GrumpkinOut>>().try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<[GrumpkinOut; 2]>, PlonkError>>()?;
        let merge_bn254_out = cfg_into_iter!(merge_grumpkin_chunks)
            .map(|chunk| Self::merge_bn254_circuit(chunk, &merge_grumpkin_pk, &base_bn254_pk))
            .collect::<Result<Vec<Bn254Out>, PlonkError>>()?;

        let merge_bn254_circuit = &merge_bn254_out[0].0;

        let (merge_bn254_pk, _) = FFTPlonk::<Kzg>::preprocess(kzg_srs, merge_bn254_circuit)?;

        // Now we need to run merge grumpkin one more time
        let merge_bn254_chunks: Vec<[Bn254Out; 2]> = merge_bn254_out
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|chunk| {
                chunk.collect::<Vec<Bn254Out>>().try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<[Bn254Out; 2]>, PlonkError>>()?;
        let decider_input = cfg_into_iter!(merge_bn254_chunks)
            .map(|chunk| Self::merge_grumpkin_circuit(chunk, &merge_bn254_pk, &merge_grumpkin_pk))
            .collect::<Result<Vec<GrumpkinOut>, PlonkError>>()?;

        // Check the length is exactly 2
        if decider_input.len() != 2 {
            return Err(PlonkError::InvalidParameters(format!(
                "Decider function input should be length 2, it has length: {}",
                decider_input.len()
            )));
        }
        let decider_input_exact: [GrumpkinOut; 2] =
            [decider_input[0].clone(), decider_input[1].clone()];

        let decider_out = Self::decider_circuit(
            decider_input_exact,
            extra_decider_info,
            &merge_grumpkin_pk,
            &merge_bn254_pk,
        )?;

        let (decider_pk, _) = PlonkKzgSnark::<Bn254>::preprocess(kzg_srs, &decider_out.circuit)?;

        Self::store_base_grumpkin_pk(base_grumpkin_pk).ok_or(PlonkError::InvalidParameters(
            "Could not store base Grumpkin proving key".to_string(),
        ))?;

        Self::store_base_bn254_pk(base_bn254_pk).ok_or(PlonkError::InvalidParameters(
            "Could not store base Bn254 proving key".to_string(),
        ))?;

        Self::store_merge_grumpkin_pk(merge_grumpkin_pk).ok_or(PlonkError::InvalidParameters(
            "Could not store merge Grumpkin proving key".to_string(),
        ))?;

        Self::store_merge_bn254_pk(merge_bn254_pk).ok_or(PlonkError::InvalidParameters(
            "Could not store merge Bn254 proving key".to_string(),
        ))?;

        Self::store_decider_pk(decider_pk).ok_or(PlonkError::InvalidParameters(
            "Could not store decider proving key".to_string(),
        ))
    }

    /// Creates a recursive proof.
    fn prove(
        outputs_and_circuit_type: &[(Bn254Output, VerifyingKey<Kzg>)],
        specific_pi: &[Vec<Fr254>],
        extra_decider_info: &[Fr254],
        extra_base_info: &[Vec<Fr254>],
    ) -> Result<RecursiveProof, PlonkError> {
        ark_std::println!("JJ: am provingg recursive proof");
        // First check that we have the same number of outputs and pi's and that they are also non-zero in length
        if outputs_and_circuit_type.len() != specific_pi.len() {
            return Err(PlonkError::InvalidParameters(format!(
                "The number of outputs: {} does not equal the number of public input lists: {}",
                outputs_and_circuit_type.len(),
                specific_pi.len()
            )));
        }

        if outputs_and_circuit_type.is_empty() {
            return Err(PlonkError::InvalidParameters(
                "Need a non-zero number of proofs".to_string(),
            ));
        }

        if (outputs_and_circuit_type.len().next_power_of_two() != outputs_and_circuit_type.len())
            && (outputs_and_circuit_type.len().ilog2() % 2 != 0)
        {
            return Err(PlonkError::InvalidParameters(
                "Outputs length is not a power of four".to_string(),
            ));
        }

        let (outputs, circuit_indices): (Vec<Bn254Output>, Vec<VerifyingKey<Kzg>>) =
            outputs_and_circuit_type
                .iter()
                .map(|(o, c)| (o.clone(), c.clone()))
                .unzip();

        let base_grumpkin_pk_128 = Self::get_base_grumpkin_pk();
        let base_bn254_pk_64 = Self::get_base_bn254_pk();

        let merge_grumpkin_pk_32 = Self::get_merge_grumpkin_pk();
        let merge_grumpkin_pk_8 = Self::get_merge_grumpkin_pk();
        let merge_grumpkin_pk_2 = Self::get_merge_grumpkin_pk();

        let merge_bn254_pk_16 = Self::get_merge_bn254_pk_16();
        let merge_bn254_pk_4 = Self::get_merge_bn254_pk_4();

        let decider_pk = Self::get_decider_pk();

        let kzg_srs = UnivariateUniversalParams::<Bn254> {
            powers_of_g: decider_pk.commit_key.powers_of_g.clone(),
            h: decider_pk.vk.open_key.h,
            beta_h: decider_pk.vk.open_key.beta_h,
        };
        // Chunking Inputs:
        // The code takes the list of base proof outputs (outputs), their public inputs (specific_pi), and their verifying keys (circuit_indices), and splits each into chunks of 2.
        // This is because the recursion tree aggregates proofs in pairs at each layer.
        // Zipping Chunks:

        // It zips together the corresponding chunks from each list, so each iteration gets a tuple of:
        // 2 proof outputs,
        // 2 sets of public inputs,
        // 2 verifying keys.
        // Type Conversion:

        // Each chunk is converted into a fixed-size array of length 2 (required by the circuit interface).
        // Calling the Base Grumpkin Circuit:

        // For each pair, it calls Self::base_grumpkin_circuit, which creates a new circuit that verifies both proofs and their public inputs, and accumulates them for the next recursion layer.
        // Collecting Results:

        // The results (one per pair) are collected into a vector. Each result is a GrumpkinOut, which contains the new circuit and its output.
        let base_grumpkin_out_128 = cfg_chunks!(outputs, 2)
            .zip(cfg_chunks!(specific_pi, 2))
            .zip(cfg_chunks!(circuit_indices, 2))
            .map(|((chunk_one, chunk_two), chunk_three)| {
                let out_slice: &[Bn254Output; 2] = chunk_one.try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length slice".to_string(),
                    )
                })?;
                let pi_slice: &[Vec<Fr254>; 2] = chunk_two.try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length slice".to_string(),
                    )
                })?;
                let vk_indices: &[VerifyingKey<Kzg>; 2] = chunk_three.try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length slice".to_string(),
                    )
                })?;
                Self::base_grumpkin_circuit(out_slice, pi_slice, vk_indices, &kzg_srs)
            })
            .collect::<Result<Vec<GrumpkinOut>, PlonkError>>()?;
        // Start with base_grumpkin_out:

        // This is a vector of GrumpkinOut (each is a tuple: a circuit and its output), produced by the previous step where base proofs were aggregated in pairs.
        // Chunking:

        // .chunks(2) splits the vector into groups of 2. This is because the recursion tree aggregates proofs in pairs at each layer.
        // Collecting and Converting:

        // For each chunk (which is an iterator over 2 GrumpkinOut), it collects them into a Vec<GrumpkinOut>, then tries to convert that into a fixed-size array [GrumpkinOut; 2].
        // If the chunk is not exactly 2 elements, it returns an error.
        // Result:

        // We get a Vec<[GrumpkinOut; 2]>, i.e., a vector of arrays, each array containing exactly 2 GrumpkinOut.
        // This is the format needed for the next layer of recursion, which expects pairs of proofs/circuits as input.
        let base_grumpkin_chunks_64: Vec<[GrumpkinOut; 2]> = base_grumpkin_out_128
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|chunk| {
                chunk.collect::<Vec<GrumpkinOut>>().try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<[GrumpkinOut; 2]>, PlonkError>>()?;
        // Iterate over all verifying keys (circuit_indices).
        // Chunk them into groups of 4 using .chunks(4).
        // For each chunk:
        // Collect the chunk into a Vec<VerifyingKey<Kzg>>.
        // Try to convert that vector into a fixed-size array [VerifyingKey<Kzg>; 4].
        // If the chunk is not exactly 4 elements, return an error.
        // Collect all these arrays into a vector: Vec<[VerifyingKey<Kzg>; 4]>.
        let vk_chunks = circuit_indices
            .into_iter()
            .chunks(4)
            .into_iter()
            .map(|chunk| {
                chunk
                    .collect::<Vec<VerifyingKey<Kzg>>>()
                    .try_into()
                    .map_err(|_| {
                        PlonkError::InvalidParameters(
                            "Could not convert to fixed length array".to_string(),
                        )
                    })
            })
            .collect::<Result<Vec<[VerifyingKey<Kzg>; 4]>, PlonkError>>()?;

        let base_bn254_out_64 = cfg_into_iter!(base_grumpkin_chunks_64)
            .zip(cfg_into_iter!(vk_chunks))
            .zip(cfg_into_iter!(extra_base_info))
            .map(|((chunk, vk_chunk), extra_info)| {
                Self::base_bn254_circuit(chunk, &base_grumpkin_pk_128, &vk_chunk, extra_info)
            })
            .collect::<Result<Vec<Bn254Out>, PlonkError>>()?;

        let base_bn254_chunks_32: Vec<[Bn254Out; 2]> = base_bn254_out_64
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|chunk| {
                chunk.collect::<Vec<Bn254Out>>().try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<[Bn254Out; 2]>, PlonkError>>()?;
        let mut merge_grumpkin_out_32 = cfg_into_iter!(base_bn254_chunks_32)
            .map(|chunk| Self::merge_grumpkin_circuit(chunk, &base_bn254_pk_64, &base_grumpkin_pk_128))
            .collect::<Result<Vec<GrumpkinOut>, PlonkError>>()?;

        let merge_grumpkin_chunks_16: Vec<[GrumpkinOut; 2]> = merge_grumpkin_out_32
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|chunk| {
                chunk.collect::<Vec<GrumpkinOut>>().try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<[GrumpkinOut; 2]>, PlonkError>>()?;

        let merge_bn254_out_16 = cfg_into_iter!(merge_grumpkin_chunks_16)
            .map(|chunk| Self::merge_bn254_circuit(chunk, &merge_grumpkin_pk_32, &base_bn254_pk_64))
            .collect::<Result<Vec<Bn254Out>, PlonkError>>()?;

        let merge_bn254_chunks_8: Vec<[Bn254Out; 2]> = merge_bn254_out_16
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|chunk| {
                chunk.collect::<Vec<Bn254Out>>().try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<[Bn254Out; 2]>, PlonkError>>()?;

         let merge_grumpkin_out_8 = merge_bn254_chunks_8
            .into_iter()
            .map(|chunk| Self::merge_grumpkin_circuit(chunk, &merge_bn254_pk_16, &merge_grumpkin_pk_32))
            .collect::<Result<Vec<GrumpkinOut>, PlonkError>>()?;


        let merge_grumpkin_chunks_4: Vec<[GrumpkinOut; 2]> = merge_grumpkin_out_8
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|chunk| {
                chunk.collect::<Vec<GrumpkinOut>>().try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<[GrumpkinOut; 2]>, PlonkError>>()?;

        let merge_bn254_out_4 = cfg_into_iter!(merge_grumpkin_chunks_4)
            .map(|chunk| Self::merge_bn254_circuit(chunk, &merge_grumpkin_pk_8, &merge_bn254_pk_16))
            .collect::<Result<Vec<Bn254Out>, PlonkError>>()?;

        let merge_bn254_chunks_2: Vec<[Bn254Out; 2]> = merge_bn254_out_4
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|chunk| {
                chunk.collect::<Vec<Bn254Out>>().try_into().map_err(|_| {
                    PlonkError::InvalidParameters(
                        "Could not convert to fixed length array".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<[Bn254Out; 2]>, PlonkError>>()?;

        let decider_input = cfg_into_iter!(merge_bn254_chunks_2)
            .map(|chunk| Self::merge_grumpkin_circuit(chunk, &merge_bn254_pk_4, &merge_grumpkin_pk_8))
            .collect::<Result<Vec<GrumpkinOut>, PlonkError>>()?;

         let decider_input_exact: [GrumpkinOut; 2] =
            [decider_input[0].clone(), decider_input[1].clone()];

        let DeciderOut {
            circuit,
            specific_pi,
            accumulators,
        } = Self::decider_circuit(
            decider_input_exact,
            extra_decider_info,
            &merge_grumpkin_pk_2,
            &merge_bn254_pk_4,
        )?;
        // is it a good idea to use system time as a seed?
        let seed = ark_std::time::SystemTime::now()
            .duration_since(ark_std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let proof = PlonkKzgSnark::<Bn254>::prove::<_, _, SolidityTranscript>(
            &mut rng,
            &circuit,
            &decider_pk,
            None,
        )?;
        ark_std::println!("Recursive proof: {:?}", proof);
        // get the public inputs
        let public_inputs = circuit.public_input().unwrap();
        ark_std::println!("Public Inputs of this Recursive proof: {:?}", public_inputs);
        ark_std::println!("specific_pi as return: {:?}", specific_pi);
        ark_std::println!("accumulators as return: {:?}", accumulators);
        //verify this proof
        ark_std::println!("JJ: Verifying recursive proof with decider vk");
        PlonkKzgSnark::<Bn254>::verify::<SolidityTranscript>(
            &decider_pk.vk,
            &public_inputs,
            &proof,
            None,
        )
        .unwrap();
        ark_std::println!("JJ: Verifying recursive proof with decider vk is done");

        Ok(RecursiveProof {
            proof,
            pi: specific_pi,
            accumulators,
        })
    }
    /// Function that takes a list of acceptable verifying key hashes and generates constraints to enforce that one of them is used in the circuits.
    fn generate_vk_check_constraint(
        check_hash: Fr254,
        vk_hashes: &[Fr254],
        circuit: &mut PlonkCircuit<Fr254>,
    ) -> Result<(), CircuitError> {
        let constant_vars = vk_hashes
            .iter()
            .map(|hash| circuit.create_constant_variable(*hash))
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        let check_var = circuit.create_variable(check_hash)?;
        let prod = constant_vars
            .iter()
            .try_fold(circuit.one(), |acc, &const_var| {
                circuit.gen_quad_poly(
                    &[acc, check_var, acc, const_var],
                    &[Fr254::zero(); 4],
                    &[Fr254::one(), -Fr254::one()],
                    Fr254::zero(),
                )
            })?;
        circuit.enforce_equal(prod, circuit.zero())
    }
}

/// Function that produces the initial [`Bn254RecursiveInfo`] from a pair of proofs.
fn initial_bn254_info(
    outputs: &[Bn254Output; 2],
    specific_pi: &[Vec<Fr254>; 2],
    kzg_srs: &UnivariateUniversalParams<Bn254>,
) -> Bn254RecursiveInfo {
    let g = kzg_srs.powers_of_g[0];
    let beta_g = kzg_srs.powers_of_g[1];

    let old_acc = AtomicInstance::<Kzg>::new(
        beta_g,
        Fr254::zero(),
        Fr254::zero(),
        UnivariateKzgProof::<Bn254> { proof: g },
    );

    Bn254RecursiveInfo {
        bn254_outputs: outputs.clone(),
        specific_pi: specific_pi.clone(),
        old_accumulators: [old_acc.clone(), old_acc.clone(), old_acc.clone(), old_acc],
        ..Default::default()
    }
}

/// The output of the final decider circuit function
#[derive(Debug, Clone)]
pub struct DeciderOut {
    /// The Bn254 circuit
    pub circuit: PlonkCircuit<Fr254>,
    /// The specific implementation pi
    pub specific_pi: Vec<Fr254>,
    /// The Bn254 accumulator
    pub accumulators: [AtomicInstance<Kzg>; 2],
}

impl DeciderOut {
    /// Create a new instance of the struct
    pub fn new(
        circuit: PlonkCircuit<Fr254>,
        specific_pi: Vec<Fr254>,
        accumulators: [AtomicInstance<Kzg>; 2],
    ) -> Self {
        Self {
            circuit,
            specific_pi,
            accumulators,
        }
    }
}

/// Struct holding all the data needed to verify a recursive proof
#[derive(Clone, Debug)]
pub struct RecursiveProof {
    /// The actual Plonk proof
    pub proof: Proof<Bn254>,
    /// The public input
    pub pi: Vec<Fr254>,
    /// The two Bn254 accumulators that also need to be verified
    pub accumulators: [AtomicInstance<Kzg>; 2],
}

// #[cfg(test)]
// mod tests {

//     use std::sync::{OnceLock, RwLock};

//     use crate::{proof_system::UniversalSNARK, transcript::RescueTranscript};

//     use super::*;
//     use ark_ec::{short_weierstrass::Affine, AffineRepr};
//     use ark_ff::{BigInteger, PrimeField};
//     use ark_std::{
//         cfg_iter,
//         collections::HashMap,
//         rand::SeedableRng,
//         string::{String, ToString},
//         UniformRand,
//     };
//     use jf_primitives::{
//         circuit::{poseidon::PoseidonHashGadget, rescue::RescueNativeGadget},
//         pcs::PolynomialCommitmentScheme,
//         poseidon::{FieldHasher, Poseidon},
//     };

//     use jf_relation::gadgets::ecc::{EmulMultiScalarMultiplicationCircuit, Point};
//     use nf_curves::{ed_on_bn254::BabyJubjub, grumpkin::short_weierstrass::SWGrumpkin};
//     use rand_chacha::ChaCha20Rng;
//     use sha3::{Digest, Keccak256};

//     #[derive(Debug, Clone)]
//     #[allow(clippy::upper_case_acronyms)]
//     pub(crate) enum Key {
//         FFT(ProvingKey<Kzg>),
//         MLE(MLEProvingKey<Zmorph>),
//         Decider(JFProvingKey<Bn254>),
//     }

//     impl Key {
//         fn get_fft_pk(&self) -> ProvingKey<Kzg> {
//             match self {
//                 Key::FFT(k) => k.clone(),
//                 _ => panic!(),
//             }
//         }

//         fn get_mle_pk(&self) -> MLEProvingKey<Zmorph> {
//             match self {
//                 Key::MLE(k) => k.clone(),
//                 _ => panic!(),
//             }
//         }

//         fn get_decider_pk(&self) -> JFProvingKey<Bn254> {
//             match self {
//                 Key::Decider(k) => k.clone(),
//                 _ => panic!(),
//             }
//         }
//     }

//     fn base_proof_circuit_generator(
//         scalar: Fr254,
//         hash: Fr254,
//     ) -> Result<PlonkCircuit<Fr254>, PlonkError> {
//         let mut circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(8);
//         let scalar_var = circuit.create_variable(scalar)?;
//         let calc_hash = circuit.poseidon_hash(&[scalar_var])?;
//         let vars = (0..10u8)
//             .map(|i| circuit.create_variable(Fr254::from(i)))
//             .collect::<Result<Vec<Variable>, CircuitError>>()?;
//         let _ = circuit.gen_quad_poly(
//             &[vars[0], vars[1], vars[2], vars[3]],
//             &[
//                 Fr254::from(1u8),
//                 Fr254::from(2u8),
//                 Fr254::from(3u8),
//                 Fr254::from(4u8),
//             ],
//             &[Fr254::from(6u8), -Fr254::from(7u8)],
//             Fr254::from(16u8),
//         )?;
//         let pi_hash = circuit.create_public_variable(hash)?;
//         let _ = (0u8..100)
//             .map(|j| circuit.create_public_variable(Fr254::from(j)))
//             .collect::<Result<Vec<Variable>, CircuitError>>()?;
//         circuit.enforce_equal(calc_hash, pi_hash)?;
//         let _has_two = circuit.poseidon_hash(&[vars[6], vars[7], vars[8]])?;
//         let bjj_generator = Affine::<BabyJubjub>::generator();
//         let generator = Affine::<SWGrumpkin>::generator();
//         let bjj_point = Point::<Fr254>::from(bjj_generator);
//         let point = Point::<Fr254>::from(generator);
//         let point_var = circuit.create_point_variable(&point)?;
//         let bjj_point_var = circuit.create_point_variable(&bjj_point)?;
//         circuit.ecc_double::<SWGrumpkin>(&point_var)?;
//         circuit.ecc_double::<BabyJubjub>(&bjj_point_var)?;
//         circuit.is_lt(vars[4], vars[5])?;
//         let emulated_scalar = circuit.create_emulated_variable(Fq254::from(7u8))?;
//         let emulated_scalar_two = circuit.create_emulated_variable(Fq254::from(8u8))?;
//         EmulMultiScalarMultiplicationCircuit::<Fr254, SWGrumpkin>::msm(
//             &mut circuit,
//             &[point_var],
//             &[emulated_scalar],
//         )?;
//         EmulMultiScalarMultiplicationCircuit::<Fr254, SWGrumpkin>::msm(
//             &mut circuit,
//             &[point_var],
//             &[emulated_scalar_two],
//         )?;
//         Ok(circuit)
//     }

//     fn base_proof_circuit_generator_two(
//         scalar: Fr254,
//         hash: Fr254,
//     ) -> Result<PlonkCircuit<Fr254>, PlonkError> {
//         let mut circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(8);
//         let scalar_var = circuit.create_variable(scalar)?;
//         let calc_hash = circuit.poseidon_hash(&[scalar_var])?;
//         let vars = (0..10u8)
//             .map(|i| circuit.create_variable(Fr254::from(i)))
//             .collect::<Result<Vec<Variable>, CircuitError>>()?;
//         let _ = circuit.gen_quad_poly(
//             &[vars[0], vars[1], vars[2], vars[3]],
//             &[
//                 Fr254::from(1u8),
//                 Fr254::from(2u8),
//                 Fr254::from(3u8),
//                 Fr254::from(4u8),
//             ],
//             &[Fr254::from(6u8), -Fr254::from(7u8)],
//             Fr254::from(16u8),
//         )?;
//         let pi_hash = circuit.create_public_variable(hash)?;
//         let _ = (0u8..100)
//             .map(|j| circuit.create_public_variable(Fr254::from(j)))
//             .collect::<Result<Vec<Variable>, CircuitError>>()?;
//         circuit.enforce_equal(calc_hash, pi_hash)?;

//         Ok(circuit)
//     }

//     /// This function is used so that we can work with one historic root tree across the entire application.
//     fn get_key_store() -> &'static RwLock<HashMap<String, Key>> {
//         static KEY_STORE: OnceLock<RwLock<HashMap<String, Key>>> = OnceLock::new();
//         KEY_STORE.get_or_init(|| RwLock::new(HashMap::<String, Key>::new()))
//     }

//     /// This function is used so that we can work with one hash list.
//     fn get_hash_list() -> &'static RwLock<Vec<Fr254>> {
//         static HASH_LIST: OnceLock<RwLock<Vec<Fr254>>> = OnceLock::new();
//         HASH_LIST.get_or_init(|| RwLock::new(Vec::new()))
//     }
//     #[test]
//     #[ignore = "Only run this test on powerful machines"]
//     #[allow(clippy::type_complexity)]
//     fn test_preprocess_and_prove() -> Result<(), PlonkError> {
//         let now = ark_std::time::Instant::now();
//         struct TestProver;

//         let poseidon = Poseidon::<Fr254>::new();
//         let other_pi = (0u8..100).map(Fr254::from).collect::<Vec<Fr254>>();
//         let (circuits, hashes): (Vec<PlonkCircuit<Fr254>>, Vec<Vec<Fr254>>) =
//             cfg_into_iter!((0u64..64))
//                 .map(|i| {
//                     let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(i);
//                     let scalar = Fr254::rand(&mut rng);
//                     let hash = poseidon.hash(&[scalar]).unwrap();

//                     let mut circuit = if i < 32 {
//                         base_proof_circuit_generator(scalar, hash)?
//                     } else {
//                         base_proof_circuit_generator_two(scalar, hash)?
//                     };
//                     circuit.finalize_for_recursive_arithmetization::<RescueCRHF<Fq254>>()?;
//                     Ok((circuit, [&[hash], other_pi.as_slice()].concat()))
//                 })
//                 .collect::<Result<Vec<(PlonkCircuit<Fr254>, Vec<Fr254>)>, PlonkError>>()?
//                 .into_iter()
//                 .unzip();

//         ark_std::println!("made input circuits in: {:?}", now.elapsed());
//         let now = ark_std::time::Instant::now();

//         let rng = &mut ChaCha20Rng::seed_from_u64(0);
//         let kzg_srs: UnivariateUniversalParams<Bn254> =
//             FFTPlonk::<Kzg>::universal_setup_for_testing(1 << 26, rng).unwrap();
//         let ipa_srs: UnivariateUniversalIpaParams<Grumpkin> =
//             Zmorph::gen_srs_for_testing(rng, 18).unwrap();

//         let (pk_one, input_vk_one) = FFTPlonk::<Kzg>::preprocess(&kzg_srs, &circuits[0])?;
//         let (pk_two, input_vk_two) = FFTPlonk::<Kzg>::preprocess(&kzg_srs, &circuits[43])?;
//         ark_std::println!("Made proving key in: {:?}", now.elapsed());
//         // Scope the lock
//         {
//             let mut hash_list = get_hash_list().write().unwrap();
//             hash_list.push(input_vk_one.hash());
//             hash_list.push(input_vk_two.hash());
//             ark_std::println!("hash list: {:?}", hash_list);
//         }
//         let now = ark_std::time::Instant::now();
//         let input_outputs = cfg_iter!(circuits)
//             .enumerate()
//             .map(|(i, circuit)| {
//                 let rng = &mut ChaCha20Rng::seed_from_u64(0);
//                 let pk = if i < 32 { &pk_one } else { &pk_two };
//                 (
//                     FFTPlonk::<Kzg>::recursive_prove::<_, _, RescueTranscript<Fr254>>(
//                         rng, circuit, pk, None,
//                     )
//                     .unwrap(),
//                     pk.vk.clone(),
//                 )
//             })
//             .collect::<Vec<(Bn254Output, VerifyingKey<Kzg>)>>();
//         ark_std::println!("made input proofs in: {:?}", now.elapsed());
//         impl RecursiveProver for TestProver {
//             fn base_bn254_checks(
//                 specific_pis: &[Vec<Variable>],
//                 circuit: &mut PlonkCircuit<Fr254>,
//             ) -> Result<Vec<Variable>, CircuitError> {
//                 let in_one = RescueNativeGadget::<Fr254>::rescue_sponge_with_padding(
//                     circuit,
//                     &specific_pis[0],
//                     1,
//                 )?;
//                 let in_two = RescueNativeGadget::<Fr254>::rescue_sponge_with_padding(
//                     circuit,
//                     &specific_pis[1],
//                     1,
//                 )?;
//                 let calc_hash = circuit.poseidon_hash(&[in_one[0], in_two[0]])?;
//                 Ok(vec![calc_hash])
//             }

//             fn base_bn254_extra_checks(
//                 _specific_pis: &[Variable],
//                 _circuit: &mut PlonkCircuit<Fr254>,
//             ) -> Result<Vec<Variable>, CircuitError> {
//                 Ok(vec![])
//             }

//             fn base_grumpkin_checks(
//                 specific_pis: &[Vec<Variable>],
//                 _circuit: &mut PlonkCircuit<Fq254>,
//             ) -> Result<Vec<Variable>, CircuitError> {
//                 Ok(specific_pis.concat())
//             }

//             fn bn254_merge_circuit_checks(
//                 specific_pis: &[Vec<Variable>],
//                 circuit: &mut PlonkCircuit<Fr254>,
//             ) -> Result<Vec<Variable>, CircuitError> {
//                 let calc_hash = circuit.poseidon_hash(&[specific_pis[0][0], specific_pis[1][0]])?;
//                 Ok(vec![calc_hash])
//             }

//             fn grumpkin_merge_circuit_checks(
//                 specific_pis: &[Vec<Variable>],
//                 _circuit: &mut PlonkCircuit<Fq254>,
//             ) -> Result<Vec<Variable>, CircuitError> {
//                 Ok(specific_pis.concat())
//             }

//             fn store_base_bn254_pk(pk: ProvingKey<Kzg>) -> Option<()> {
//                 get_key_store()
//                     .write()
//                     .unwrap()
//                     .insert("bn254 base".to_string(), Key::FFT(pk));
//                 Some(())
//             }

//             fn decider_circuit_checks(
//                 _specific_pis: &[Vec<Variable>],
//                 _circuit: &mut PlonkCircuit<Fr254>,
//             ) -> Result<Vec<Variable>, CircuitError> {
//                 Ok(vec![])
//             }

//             fn get_vk_hash_list() -> Vec<Fr254> {
//                 get_hash_list().read().unwrap().clone()
//             }

//             fn get_base_grumpkin_pk() -> MLEProvingKey<Zmorph> {
//                 get_key_store()
//                     .read()
//                     .unwrap()
//                     .get("grumpkin base")
//                     .unwrap()
//                     .get_mle_pk()
//             }

//             fn get_base_bn254_pk() -> ProvingKey<Kzg> {
//                 get_key_store()
//                     .read()
//                     .unwrap()
//                     .get("bn254 base")
//                     .unwrap()
//                     .get_fft_pk()
//                     .clone()
//             }

//             fn get_merge_grumpkin_pk() -> MLEProvingKey<Zmorph> {
//                 get_key_store()
//                     .read()
//                     .unwrap()
//                     .get("grumpkin merge")
//                     .unwrap()
//                     .get_mle_pk()
//                     .clone()
//             }

//             fn get_merge_bn254_pk() -> ProvingKey<Kzg> {
//                 get_key_store()
//                     .read()
//                     .unwrap()
//                     .get("bn254 merge")
//                     .unwrap()
//                     .get_fft_pk()
//                     .clone()
//             }

//             fn get_decider_pk() -> JFProvingKey<Bn254> {
//                 get_key_store()
//                     .read()
//                     .unwrap()
//                     .get("decider")
//                     .unwrap()
//                     .get_decider_pk()
//                     .clone()
//             }

//             fn store_base_grumpkin_pk(pk: MLEProvingKey<Zmorph>) -> Option<()> {
//                 get_key_store()
//                     .write()
//                     .unwrap()
//                     .insert("grumpkin base".to_string(), Key::MLE(pk));
//                 Some(())
//             }

//             fn store_merge_grumpkin_pk(pk: MLEProvingKey<Zmorph>) -> Option<()> {
//                 get_key_store()
//                     .write()
//                     .unwrap()
//                     .insert("grumpkin merge".to_string(), Key::MLE(pk));
//                 Some(())
//             }

//             fn store_merge_bn254_pk(pk: ProvingKey<Kzg>) -> Option<()> {
//                 get_key_store()
//                     .write()
//                     .unwrap()
//                     .insert("bn254 merge".to_string(), Key::FFT(pk));
//                 Some(())
//             }

//             fn store_decider_pk(pk: JFProvingKey<Bn254>) -> Option<()> {
//                 get_key_store()
//                     .write()
//                     .unwrap()
//                     .insert("decider".to_string(), Key::Decider(pk));
//                 Some(())
//             }
//         }

//         TestProver::preprocess(
//             &input_outputs,
//             &hashes,
//             vec![vec![]; input_outputs.len() / 4].as_slice(),
//             &[],
//             &ipa_srs,
//             &kzg_srs,
//         )?;

//         // Now we test proof generation using the keys
//         ark_std::println!("begun prove test");
//         let (prove_inputs, hashes): (Vec<(Bn254Output, VerifyingKey<Kzg>)>, Vec<Vec<Fr254>>) =
//             cfg_into_iter!((0u64..256))
//                 .map(|i| {
//                     let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(i);
//                     let scalar = Fr254::rand(&mut rng);
//                     let hash = poseidon.hash(&[scalar]).unwrap();

//                     let mut circuit = if i < 74 {
//                         base_proof_circuit_generator(scalar, hash)?
//                     } else {
//                         base_proof_circuit_generator_two(scalar, hash)?
//                     };
//                     let pk = if i < 74 { &pk_one } else { &pk_two };
//                     circuit.finalize_for_recursive_arithmetization::<RescueCRHF<Fq254>>()?;
//                     let input_output =
//                         FFTPlonk::<Kzg>::recursive_prove::<_, _, RescueTranscript<Fr254>>(
//                             &mut rng, &circuit, pk, None,
//                         )?;
//                     Ok((
//                         (input_output, pk.vk.clone()),
//                         [&[hash], other_pi.as_slice()].concat(),
//                     ))
//                 })
//                 .collect::<Result<Vec<((Bn254Output, VerifyingKey<Kzg>), Vec<Fr254>)>, PlonkError>>(
//                 )?
//                 .into_iter()
//                 .unzip();

//         let now = ark_std::time::Instant::now();
//         let proof = TestProver::prove(
//             &prove_inputs,
//             &hashes,
//             &[],
//             vec![vec![]; prove_inputs.len() / 4].as_slice(),
//         )?;
//         ark_std::println!(
//             "Time taken to generate 256 recursive proofs: {:?}",
//             now.elapsed()
//         );

//         let field_pi = proof
//             .pi
//             .iter()
//             .flat_map(|f| f.into_bigint().to_bytes_be())
//             .collect::<Vec<u8>>();

//         let acc_elems = proof
//             .accumulators
//             .iter()
//             .flat_map(|acc| {
//                 let point = Point::<Fq254>::from(acc.comm);
//                 let opening_proof = Point::<Fq254>::from(acc.opening_proof.proof);
//                 point
//                     .coords()
//                     .iter()
//                     .chain(opening_proof.coords().iter())
//                     .flat_map(|coord| coord.into_bigint().to_bytes_be())
//                     .collect::<Vec<u8>>()
//             })
//             .collect::<Vec<u8>>();

//         let mut hasher = Keccak256::new();
//         hasher.update([field_pi, acc_elems].concat());
//         let buf = hasher.finalize();

//         // Generate challenge from state bytes using little-endian order
//         let pi_hash = Fr254::from_be_bytes_mod_order(&buf);

//         assert!(PlonkKzgSnark::<Bn254>::verify::<SolidityTranscript>(
//             &TestProver::get_decider_pk().vk,
//             &[pi_hash],
//             &proof.proof,
//             None
//         )
//         .is_ok());
//         Ok(())
//     }
// }
