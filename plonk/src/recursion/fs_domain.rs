//! Helpers for building Fiat-Shamir domain separation messages for recursive proving.

use ark_serialize::CanonicalSerialize;
use ark_std::vec::Vec;
use sha2::{Digest, Sha256};

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

/// Build a fixed-order, length-prefixed FS init message.
/// Use a simple TLV-ish format to avoid JSON/non-determinism.
#[allow(clippy::too_many_arguments)]
pub fn fs_domain_bytes(
    app_id: &'static str,
    proto: &'static str,
    version: &'static str,
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
    push(&mut msg, "version", version.as_bytes());
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
