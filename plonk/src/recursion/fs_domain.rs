//! Helpers for building Fiat-Shamir domain separation messages for recursive proving.

use ark_std::vec;

use ark_bn254::{Bn254, Fq, Fr, G1Affine};
use ark_ff::{BigInteger, PrimeField};
use ark_serialize::CanonicalSerialize;
use ark_std::vec::Vec;
use sha2::{Digest, Sha256};
use sha3::Keccak256;

use crate::proof_system::structs::VerifyingKey;

/// Stable, deterministic 32-byte hash helper
fn h256(bytes: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(&Sha256::digest(bytes));
    out
}

/// Canonical-serialize and hash a value (VK, SRS bits, etc.)
pub fn hash_canonical<T: CanonicalSerialize>(x: &T) -> [u8; 32] {
    let mut buf = Vec::new();
    x.serialize_compressed(&mut buf).unwrap();
    h256(&buf)
}

/// Computes the vk hash as in the Solidity verifier
pub fn compute_vk_hash(vk: &VerifyingKey<Bn254>) -> [u8; 32] {
    // Layout (matches the Solidity assembly):
    //  domain_size (last 8 bytes of a uint256) ||
    //  sigma_comms_1..6 (x||y each 32B) ||
    //  selector_comms_1..18 (x||y) ||
    //  k1..k6 (each 32B) ||
    //  range_table_comm, key_table_comm, table_dom_sep_comm, q_dom_sep_comm (x||y)
    //
    // Total = 8 + 6*64 + 18*64 + 6*32 + 4*64 = 1992 bytes
    let mut buf = Vec::with_capacity(1992);

    // 1) domain_size: Solidity writes only the last 8 bytes (big-endian)
    buf.extend_from_slice(&(vk.domain_size as u64).to_be_bytes());

    // Helpers: push canonical big-endian 32-byte encodings
    let push_fq = |buf: &mut Vec<u8>, x: &Fq| {
        let mut b = x.into_bigint().to_bytes_be(); // canonical integer, big-endian
        if b.len() < 32 {
            let mut z = vec![0u8; 32 - b.len()];
            z.extend_from_slice(&b);
            b = z;
        }
        buf.extend_from_slice(&b);
    };
    let push_fr = |buf: &mut Vec<u8>, x: &Fr| {
        let mut b = x.into_bigint().to_bytes_be(); // canonical integer, big-endian
        if b.len() < 32 {
            let mut z = vec![0u8; 32 - b.len()];
            z.extend_from_slice(&b);
            b = z;
        }
        buf.extend_from_slice(&b);
    };
    let push_g1 = |buf: &mut Vec<u8>, p: &G1Affine| {
        push_fq(buf, &p.x);
        push_fq(buf, &p.y);
    };

    // 2) sigma_comms_1..6
    vk.sigma_comms
        .iter()
        .take(6)
        .for_each(|c| push_g1(&mut buf, c));

    // 3) selector_comms_1..18
    vk.selector_comms
        .iter()
        .take(18)
        .for_each(|c| push_g1(&mut buf, c));

    // 4) k1..k6
    vk.k.iter().take(6).for_each(|x| push_fr(&mut buf, x));

    // 5) plookup commitments
    let pl = vk.plookup_vk.as_ref().expect("plookup_vk required");
    [
        &pl.range_table_comm,
        &pl.key_table_comm,
        &pl.table_dom_sep_comm,
        &pl.q_dom_sep_comm,
    ]
    .into_iter()
    .for_each(|c| push_g1(&mut buf, c));

    // keccak256 over the 1992-byte payload
    let h = Keccak256::digest(&buf);

    // Solidity returns keccak(buf) mod r, where r is BN254 scalar field modulus.
    // Use arkworks to reduce and then output the canonical 32-byte big-endian integer.
    let reduced = Fr::from_be_bytes_mod_order(&h);
    let mut out = reduced.into_bigint().to_bytes_be();
    if out.len() < 32 {
        let mut z = vec![0u8; 32 - out.len()];
        z.extend_from_slice(&out);
        out = z;
    }

    let mut arr = [0u8; 32];
    arr.copy_from_slice(&out);
    arr
}

/// Build a fixed-order, length-prefixed FS init message.
/// Use a simple TLV-ish format to avoid JSON/non-determinism.
#[allow(clippy::too_many_arguments)]
pub fn fs_domain_bytes(
    app_id: &'static str,
    proto: &'static str,
    role: &'static str,   // e.g. "rollup_prover"
    layer: &'static str,  // e.g. "base_bn254" | "merge_grumpkin" | "decider"
    vk_digest: [u8; 32],  // hash_canonical(&vk)
    srs_digest: [u8; 32], // KZG/IPA SRS digest
    recursion_depth: u32, // 0 at base, then increment upwards
    rollup_size: u32,     // number of leaf proofs in this batch
) -> Vec<u8> {
    let mut msg = Vec::new();
    let push = |m: &mut Vec<u8>, label: &str, v: &[u8]| {
        m.extend_from_slice(&(label.len() as u32).to_be_bytes());
        m.extend_from_slice(label.as_bytes());
        m.extend_from_slice(&(v.len() as u32).to_be_bytes());
        m.extend_from_slice(v);
    };
    push(&mut msg, "app_id", app_id.as_bytes());
    push(&mut msg, "proto", proto.as_bytes());
    push(&mut msg, "role", role.as_bytes());
    push(&mut msg, "layer", layer.as_bytes());
    push(&mut msg, "vk_digest", &vk_digest);
    push(&mut msg, "srs_digest", &srs_digest);
    push(&mut msg, "recursion_depth", &recursion_depth.to_be_bytes());
    push(&mut msg, "rollup_size", &rollup_size.to_be_bytes());
    msg
}

/// Deterministic ChaCha seed from FS domain msg
pub fn rng_seed_from_fs(fs_msg: &[u8]) -> [u8; 32] {
    h256(&[fs_msg].concat())
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
/// Struct for FS initialization message metadata.
pub struct FSInitMetadata {
    pub recursion_depth: usize,
    pub rollup_size: u32,
    pub srs_digest: [u8; 32],
}
