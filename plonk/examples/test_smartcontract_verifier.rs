use ark_bn254::Bn254;
use ark_bn254::Fr as Fr254;

use ark_ed_on_bn254::Fq as FqEd254;
use ark_ff::PrimeField;
use hex::encode;
use jf_plonk::proof_system::structs::Proof;
use jf_plonk::proof_system::structs::VerifyingKey;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

use jf_plonk::proof_system::PlonkKzgSnark;
use jf_plonk::proof_system::UniversalSNARK;
use jf_plonk::transcript::SolidityTranscript;
use jf_relation::Arithmetization;
use jf_relation::{Circuit, PlonkCircuit};
use num_bigint::BigUint;

fn main() {
    let x = Fr254::from(3u64);
    let y = Fr254::from(4u64);

    let mut circuit: PlonkCircuit<FqEd254> = PlonkCircuit::new_ultra_plonk(10);

    let x_var = circuit.create_variable(x).unwrap();
    let y_var = circuit.create_public_variable(y).unwrap();
    let x_equal_y = circuit.is_equal(x_var, y_var).unwrap();
    circuit.enforce_false(x_equal_y.into()).unwrap();
    circuit.finalize_for_arithmetization().unwrap();
    let mut rng = jf_utils::test_rng();
    let srs_size = circuit.srs_size(true).unwrap();

    let srs = PlonkKzgSnark::<Bn254>::universal_setup_for_testing(srs_size, &mut rng).unwrap();
    let (pk, vk) = PlonkKzgSnark::<Bn254>::preprocess(&srs, None, &circuit).unwrap();
    let proof =
        PlonkKzgSnark::<Bn254>::prove::<_, _, SolidityTranscript>(&mut rng, &circuit, &pk, None, true)
            .unwrap();
    PlonkKzgSnark::<Bn254>::verify::<SolidityTranscript>(&vk, &[y], &proof, None).unwrap();

    let public_inputs = circuit.public_input().unwrap();
    let extra_transcript_init_msg = None;

    assert!(PlonkKzgSnark::<Bn254>::verify::<SolidityTranscript>(
        &vk,
        &public_inputs,
        &proof,
        extra_transcript_init_msg,
    )
    .is_ok());

    let proof_vec: Vec<ark_bn254::Fq> = proof.into();

    let mut hex_strings = Vec::new();
    for element in &public_inputs {
        let element_bigint: BigUint = element.into_bigint().into();
        let mut element_hex_string = element_bigint.to_str_radix(16);
        while element_hex_string.len() < 64 {
            element_hex_string.insert(0, '0');
        }
        hex_strings.push(element_hex_string);
    }
    for element in &proof_vec {
        let element_bigint: BigUint = element.into_bigint().into();
        let mut element_hex_string = element_bigint.to_str_radix(16);
        while element_hex_string.len() < 64 {
            element_hex_string.insert(0, '0');
        }
        hex_strings.push(element_hex_string);
    }
    let json_data = hex_strings.join("");
    let mut file = File::create("proof_ultra_nightfish.json").unwrap();
    file.write_all(json_data.as_bytes()).unwrap();
}
