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
    fn store_merge_bn254_pk_4(pk: ProvingKey<Kzg>) -> Option<()>;
    /// Stores the merge Bn254 proving key.
    fn store_merge_bn254_pk_16(pk: ProvingKey<Kzg>) -> Option<()>;
    /// Stores the decider proving key.
    fn store_decider_pk(pk: JFProvingKey<Bn254>) -> Option<()>;
    /// This function takes in the input proofs and outputs [`GrumpkinCircuitOutput`] to be taken in by the next function.
    fn base_grumpkin_circuit(
        outputs: &[Bn254Output; 2],
        specific_pi: &[Vec<Fr254>; 2],
        input_vks: &[VerifyingKey<Kzg>; 2],
        kzg_srs: &UnivariateUniversalParams<Bn254>,
    ) -> Result<GrumpkinOut, PlonkError> {
        // ark_std::println!("intermediate structure: base_grumpkin_circuit");
        // record how long it takes to run this function
        // let start  = Instant::now();
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

        // #[cfg(test)]
        // {
        //     ark_std::println!(
        //         "base grumpkin circuit size pre-finalize: {}",
        //         circuit.num_gates()
        //     );
        // }

        circuit.finalize_for_recursive_mle_arithmetization::<RescueCRHF<Fr254>>()?;

        // #[cfg(test)]
        // {
        //     let pi = circuit.public_input()?;
        //     circuit.check_circuit_satisfiability(&pi)?;
        //     ark_std::println!("base grumpkin circuit size: {}", circuit.num_gates());
        // }
        // let elapsed = start.elapsed();
        // ark_std::println!(
        //     "base_grumpkin_circuit took: {} seconds",
        //     elapsed.as_secs_f64()
        // );
        Ok((circuit, circuit_output))
    }
    /// This function takes in [`GrumpkinOut`] types, proves the circuits and then produces another circuit proving their correct accumulation.
    fn base_bn254_circuit(
        outputs: [GrumpkinOut; 2],
        base_grumpkin_pk: &MLEProvingKey<Zmorph>,
        input_vks: &[VerifyingKey<Kzg>; 4],
        extra_base_info: &[Fr254],
    ) -> Result<Bn254Out, PlonkError> {
        // ark_std::println!("intermediate structure: base_bn254_circuit");
        // let start  = Instant::now();
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

        // #[cfg(test)]
        // {
        //     ark_std::println!(
        //         "base bn254 circuit size pre-finalize: {}",
        //         circuit.num_gates()
        //     );
        // }

        circuit.finalize_for_recursive_arithmetization::<RescueCRHF<Fq254>>()?;

        // Run the following code only when testing
        // #[cfg(test)]
        // {
        //     let pi = circuit.public_input()?;
        //     circuit.check_circuit_satisfiability(&pi)?;
        //     ark_std::println!("base bn254 circuit size: {}", circuit.num_gates());
        // }
        // let elapsed = start.elapsed();
        // ark_std::println!(
        //     "base_bn254_circuit took: {} seconds",
        //     elapsed.as_secs_f64()
        // );
        Ok((circuit, bn254_circuit_out))
    }
    /// This function takes in [`Bn254Out`] types, proves the circuits and then produces another circuit proving their correct accumulation.
    fn merge_grumpkin_circuit(
        outputs: [Bn254Out; 2],
        base_bn254_pk: &ProvingKey<Kzg>,
        base_grumpkin_pk: &MLEProvingKey<Zmorph>,
    ) -> Result<GrumpkinOut, PlonkError> {
        // ark_std::println!("intermediate structure: merge_grumpkin_circuit");
        // let start  = Instant::now();
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

        // ark_std::println!(
        //     "JJ: am about to call prove_bn254_accumulation"
        // );

        let grumpkin_circuit_out = prove_bn254_accumulation::<false>(
            &bn254info,
            &[base_bn254_pk.vk.clone(), base_bn254_pk.vk.clone()],
            &base_grumpkin_pk.verifying_key,
            Self::grumpkin_merge_circuit_checks,
            &mut circuit,
        )?;

        // #[cfg(test)]
        // {
        //     ark_std::println!(
        //         "merge grumpkin circuit size pre-finalize: {}",
        //         circuit.num_gates()
        //     );
        // }

        circuit.finalize_for_recursive_mle_arithmetization::<RescueCRHF<Fr254>>()?;

        // Run the following code only when testing
        // #[cfg(test)]
        // {
        //     let pi = circuit.public_input()?;
        //     circuit.check_circuit_satisfiability(&pi)?;
        //     ark_std::println!("merge grumpkin circuit size: {}", circuit.num_gates());
        // }
        // let elapsed = start.elapsed();
        // ark_std::println!(
        //     "merge grumpkin circuit took: {} seconds",
        //     elapsed.as_secs_f64()
        // );

        Ok((circuit, grumpkin_circuit_out))
    }
    /// This function takes in [`GrumpkinOut`] types, proves the circuits and then produces another circuit proving their correct accumulation.
    fn merge_bn254_circuit(
        outputs: [GrumpkinOut; 2],
        merge_grumpkin_pk: &MLEProvingKey<Zmorph>,
        bn254_pk: &ProvingKey<Kzg>,
    ) -> Result<Bn254Out, PlonkError> {
        // ark_std::println!("intermediate structure: merge_bn254_circuit");
        // let start  = Instant::now();
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
        // let elapsed = start.elapsed();
        // ark_std::println!(
        //     "merge_bn254_circuit took: {} seconds",
        //     elapsed.as_secs_f64()
        // );
        Ok((circuit, bn254_circuit_out))
    }
    /// The decider circuit that fully verifies the Grumpkin accumulator along with the final grumpkin proof.
    fn decider_circuit(
        grumpkin_outputs: [GrumpkinOut; 2],
        extra_decider_info: &[Fr254],
        merge_grumpkin_pk: &MLEProvingKey<Zmorph>,
        merge_bn254_pk: &ProvingKey<Kzg>,
    ) -> Result<DeciderOut, PlonkError> {
        // ark_std::println!("intermediate structure: decider_circuit");
        // let start  = Instant::now();
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
        // ark_std::println!(
        //     "decider circuit circuit size pre-finalize: {}",
        //     circuit.num_gates()
        // );
        // }

        circuit.finalize_for_arithmetization()?;
        // ark_std::println!(
        //     "decider circuit circuit size after finalize: {}",
        //     circuit.num_gates()
        // );

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

        // let elapsed = start.elapsed();
        // ark_std::println!(
        //     "decider_circuit took: {} seconds",
        //     elapsed.as_secs_f64()
        // );
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
        ark_std::println!("JJ: am calling preprocess in Nightfish, plonk, src, recursion, RecursiveProver, but why?");
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

        // base_grumpkin_circuit(512) -> base_grumpkin_out 512, base_grumpkin_chunks 256
        let base_grumpkin_out_512 = cfg_chunks!(outputs, 2)
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
       let base_grumpkin_circuit_512 = &base_grumpkin_out_512[0].0;
        let base_pi = base_grumpkin_circuit_512.public_input().unwrap();
        base_grumpkin_circuit_512.check_circuit_satisfiability(&base_pi)?;
        let (base_grumpkin_pk_512, _) = MLEPlonk::<Zmorph>::preprocess(ipa_srs, base_grumpkin_circuit_512)?;
        
        // Produce and store the base Bn254 proving key
         let base_grumpkin_chunks_256: Vec<[GrumpkinOut; 2]> = base_grumpkin_out_512
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
        let base_bn254_out_256 = cfg_into_iter!(base_grumpkin_chunks_256)
            .zip(cfg_into_iter!(vk_chunks))
            .zip(cfg_into_iter!(extra_base_info))
            .map(|((chunk, vk_chunk), extra_info)| {
                Self::base_bn254_circuit(chunk, &base_grumpkin_pk_512, &vk_chunk, extra_info)
            })
            .collect::<Result<Vec<Bn254Out>, PlonkError>>()?;
        let base_bn254_circuit_256 = &base_bn254_out_256[0].0;
        let (base_bn254_pk_256, _) = FFTPlonk::<Kzg>::preprocess(kzg_srs, base_bn254_circuit_256)?;

        // Produce the Grumpkin merge proving key
        let base_bn254_chunks_128: Vec<[Bn254Out; 2]> = base_bn254_out_256
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
        let merge_grumpkin_out_128 = base_bn254_chunks_128
            .into_iter()
            .map(|chunk| Self::merge_grumpkin_circuit(chunk, &base_bn254_pk_256, &base_grumpkin_pk_512))
            .collect::<Result<Vec<GrumpkinOut>, PlonkError>>()?;

        let merge_grumpkin_circuit_128 = &merge_grumpkin_out_128[0].0;
 let (merge_grumpkin_pk_128, _) =
            MLEPlonk::<Zmorph>::preprocess(ipa_srs, merge_grumpkin_circuit_128)?;

        // Produce the Bn254 merge proving key
       let merge_grumpkin_chunks_64: Vec<[GrumpkinOut; 2]> = merge_grumpkin_out_128
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


      let merge_bn254_out_64 = cfg_into_iter!(merge_grumpkin_chunks_64)
            .map(|chunk| Self::merge_bn254_circuit(chunk, &merge_grumpkin_pk_128, &base_bn254_pk_256))
            .collect::<Result<Vec<Bn254Out>, PlonkError>>()?;

    let merge_bn254_circuit_64 = &merge_bn254_out_64[0].0;

        let (merge_bn254_pk_64, _) = FFTPlonk::<Kzg>::preprocess(kzg_srs, merge_bn254_circuit_64)?;

         let merge_bn254_chunks_32: Vec<[Bn254Out; 2]> = merge_bn254_out_64
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
let merge_grumpkin_out_32 = merge_bn254_chunks_32
            .into_iter()
            .map(|chunk| Self::merge_grumpkin_circuit(chunk, &merge_bn254_pk_64, &merge_grumpkin_pk_128))
            .collect::<Result<Vec<GrumpkinOut>, PlonkError>>()?;


        let merge_grumpkin_circuit_32 = &merge_grumpkin_out_32[0].0;

        let (merge_grumpkin_pk_32, _) =
            MLEPlonk::<Zmorph>::preprocess(ipa_srs, merge_grumpkin_circuit_32)?;


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
            .map(|chunk| Self::merge_bn254_circuit(chunk, &merge_grumpkin_pk_32, &merge_bn254_pk_64))
            .collect::<Result<Vec<Bn254Out>, PlonkError>>()?;

          let merge_bn254_circuit_16 = &merge_bn254_out_16[0].0;
        let (merge_bn254_pk_16, _) = FFTPlonk::<Kzg>::preprocess(kzg_srs, merge_bn254_circuit_16)?;
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

         let merge_grumpkin_circuit_8 = &merge_grumpkin_out_8[0].0;
        let (merge_grumpkin_pk_8, _) =
            MLEPlonk::<Zmorph>::preprocess(ipa_srs, merge_grumpkin_circuit_8)?;

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

        let merge_bn254_circuit_4 = &merge_bn254_out_4[0].0;

        let (merge_bn254_pk_4, _) = FFTPlonk::<Kzg>::preprocess(kzg_srs, merge_bn254_circuit_4)?;


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

        let merge_grumpkin_circuit_2 = &decider_input[0].0;

        let (merge_grumpkin_pk_2, _) =
            MLEPlonk::<Zmorph>::preprocess(ipa_srs, merge_grumpkin_circuit_2)?;

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
            &merge_grumpkin_pk_2,
            &merge_bn254_pk_4,
        )?;

        let (decider_pk, _) = PlonkKzgSnark::<Bn254>::preprocess(kzg_srs, &decider_out.circuit)?;

        Self::store_base_grumpkin_pk(base_grumpkin_pk_512).ok_or(PlonkError::InvalidParameters(
            "Could not store base Grumpkin proving key".to_string(),
        ))?;

        Self::store_base_bn254_pk(base_bn254_pk_256).ok_or(PlonkError::InvalidParameters(
            "Could not store base Bn254 proving key".to_string(),
        ))?;

        Self::store_merge_grumpkin_pk(merge_grumpkin_pk_2).ok_or(PlonkError::InvalidParameters(
            "Could not store merge Grumpkin proving key".to_string(),
        ))?;

        Self::store_merge_bn254_pk_4(merge_bn254_pk_16).ok_or(PlonkError::InvalidParameters(
            "Could not store merge Bn254 proving key".to_string(),
        ))?;

        Self::store_merge_bn254_pk_16(merge_bn254_pk_64).ok_or(PlonkError::InvalidParameters(
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
        // ark_std::println!("JJ: am provingg recursive proof");
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

        let base_grumpkin_pk_512 = Self::get_base_grumpkin_pk();
        let base_bn254_pk_256 = Self::get_base_bn254_pk();

        let merge_grumpkin_pk_128 = Self::get_merge_grumpkin_pk();
        let merge_grumpkin_pk_32 = Self::get_merge_grumpkin_pk();
        let merge_grumpkin_pk_8 = Self::get_merge_grumpkin_pk();
        let merge_grumpkin_pk_2 = Self::get_merge_grumpkin_pk();

        let merge_bn254_pk_64 = Self::get_merge_bn254_pk_16();
        let merge_bn254_pk_16 = Self::get_merge_bn254_pk_4();
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
        let base_grumpkin_out_512 = cfg_chunks!(outputs, 2)
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
        let base_grumpkin_chunks_256: Vec<[GrumpkinOut; 2]> = base_grumpkin_out_512
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

       let base_bn254_out_256 = cfg_into_iter!(base_grumpkin_chunks_256)
            .zip(cfg_into_iter!(vk_chunks))
            .zip(cfg_into_iter!(extra_base_info))
            .map(|((chunk, vk_chunk), extra_info)| {
                Self::base_bn254_circuit(chunk, &base_grumpkin_pk_512, &vk_chunk, extra_info)
            })
            .collect::<Result<Vec<Bn254Out>, PlonkError>>()?;

        let base_bn254_chunks_128: Vec<[Bn254Out; 2]> = base_bn254_out_256
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
        let merge_grumpkin_out_128 = base_bn254_chunks_128
            .into_iter()
            .map(|chunk| Self::merge_grumpkin_circuit(chunk, &base_bn254_pk_256, &base_grumpkin_pk_512))
            .collect::<Result<Vec<GrumpkinOut>, PlonkError>>()?;

let merge_grumpkin_chunks_64: Vec<[GrumpkinOut; 2]> = merge_grumpkin_out_128
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

        let merge_bn254_out_64 = cfg_into_iter!(merge_grumpkin_chunks_64)
            .map(|chunk| Self::merge_bn254_circuit(chunk, &merge_grumpkin_pk_128, &base_bn254_pk_256))
            .collect::<Result<Vec<Bn254Out>, PlonkError>>()?;

        let merge_bn254_chunks_32: Vec<[Bn254Out; 2]> = merge_bn254_out_64
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


        let merge_grumpkin_out_32 = merge_bn254_chunks_32
            .into_iter()
            .map(|chunk| Self::merge_grumpkin_circuit(chunk, &merge_bn254_pk_64, &merge_grumpkin_pk_128))
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
            .map(|chunk| Self::merge_bn254_circuit(chunk, &merge_grumpkin_pk_32, &merge_bn254_pk_64))
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
        // ark_std::println!("Recursive proof: {:?}", proof);
        // get the public inputs
        // let public_inputs = circuit.public_input().unwrap();
        // ark_std::println!("Public Inputs of this Recursive proof: {:?}", public_inputs);
        // ark_std::println!("specific_pi as return: {:?}", specific_pi);
        // ark_std::println!("accumulators as return: {:?}", accumulators);
        //verify this proof
        // ark_std::println!("JJ: Verifying recursive proof with decider vk");
        // PlonkKzgSnark::<Bn254>::verify::<SolidityTranscript>(
        //     &decider_pk.vk,
        //     &public_inputs,
        //     &proof,
        //     None,
        // )
        // .unwrap();
        // ark_std::println!("JJ: Verifying recursive proof with decider vk is done");

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
