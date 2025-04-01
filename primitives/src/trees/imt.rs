//! This module contains all the code relating to indexed merkle trees.

use crate::poseidon::PoseidonError;

use super::{
    error::IndexedMerkleTreeError, index_to_directions, timber::Timber, CircuitInsertionInfo,
    Directions, MembershipProof, PathElement, Tree, TreeHasher,
};
use ark_ff::PrimeField;
use ark_std::{vec, vec::Vec, Zero};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
/// Struct used for an Indexed Merkle Tree.
#[derive(Clone, Debug)]
pub struct IndexedMerkleTree<N, H, DB> {
    /// The timber tree
    pub timber: Timber<N, H>,
    /// The database storing the leaf data that can be queried
    pub leaves_db: DB,
}

/// Struct storing relevant information about a nullifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LeafDBEntry<F> {
    /// The value of this nullifier
    pub value: F,
    /// The index of the leaf representing this nullifier in the Indexed Merkle tree
    pub index: u32,
    /// The index of the next highest value nullifier in the Indexed Merkle tree
    pub next_index: F,
    /// The value of the next highest value nullifier in the Indexed Merkle tree
    pub next_value: F,
}

impl<N: PrimeField> From<&LeafDBEntry<N>> for Vec<N> {
    fn from(entry: &LeafDBEntry<N>) -> Self {
        let index = N::from(entry.index);
        vec![entry.value, index, entry.next_index, entry.next_value]
    }
}

impl<N: PrimeField> TryFrom<&[N]> for LeafDBEntry<N> {
    type Error = IndexedMerkleTreeError;
    fn try_from(value: &[N]) -> Result<Self, Self::Error> {
        if value.len() != 4 {
            return Err(IndexedMerkleTreeError::DatabaseError);
        }
        let biguint: BigUint = value[1].into();
        let index = biguint
            .to_u32()
            .ok_or(IndexedMerkleTreeError::DatabaseError)?;
        Ok(LeafDBEntry {
            value: value[0],
            index,
            next_index: value[2],
            next_value: value[3],
        })
    }
}

impl<F: PrimeField> Default for LeafDBEntry<F> {
    fn default() -> Self {
        Self {
            value: F::zero(),
            index: 0,
            next_index: F::zero(),
            next_value: F::zero(),
        }
    }
}

impl<F: PrimeField> LeafDBEntry<F> {
    /// Creates a new instance of the struct.
    pub fn new(value: F, index: u32, next_index: F, next_value: F) -> Self {
        Self {
            value,
            index,
            next_index,
            next_value,
        }
    }
}

/// A databvase that stores nullifiers of spent commitments.
/// These are stored in the format they appear in the Indexed Merkle tree, that is
/// the "preimage" of the nullifier is stored in the form {nullifier_value, index, next_value} and
/// the key is a hash of this.
pub trait LeafDB<F> {
    /// Creates a new instance of the database. We have to do this because we need to insert the zero nullifier.
    fn new() -> Self;
    /// Stores a nullifier in the database. This functions works out the low nullifier and optionally takes the index in the tree of the new nullifier.
    fn store_nullifier(&mut self, nullifier: F, index: Option<u32>) -> Option<()>;
    /// Searches the database for a nullifier with the supplied fields. If it finds one, it returns it.
    fn get_nullifier(&self, value: Option<F>, next_value: Option<F>) -> Option<&LeafDBEntry<F>>;
    /// Searches the database for the nullifier that skips over the supplied value. That is finds the nullifier such that
    /// `low_nullifier.value` < `nullifier_value` < `low_nullifier.next_value`. If it finds one, it returns it.
    fn get_low_nullifier(&self, nullifier_value: &F) -> Option<&LeafDBEntry<F>>;
    /// Updates the nullifier entry stored with value `nullifier` with the new `next_value`.
    fn update_nullifier(
        &mut self,
        nullifier: F,
        new_next_index: F,
        new_next_value: F,
    ) -> Option<()>;
}

/// A struct that stores all the information needed to prove correct subtree insertion in a circuit context for an Indexed Merkle Tree.
#[derive(Clone, Debug)]
pub struct IMTCircuitInsertionInfo<N: PrimeField> {
    /// Root of the tree before any updates.
    pub old_root: N,
    /// The circuit info for the actual subtree insertion.
    pub circuit_info: CircuitInsertionInfo<N>,
    /// The initial index of the first leaf in the inserted subtree.
    pub first_index: u32,
    /// The low nullifiers for the leaves inserted with non_membership proofs.
    pub low_nullifiers: Vec<(LeafDBEntry<N>, MembershipProof<N>)>,
    /// The entries of the nullifiers inserted.
    pub pending_inserts: Vec<LeafDBEntry<N>>,
}

impl<N: PrimeField> From<IMTCircuitInsertionInfo<N>> for Vec<N> {
    fn from(info: IMTCircuitInsertionInfo<N>) -> Self {
        let mut vec = vec![];
        vec.push(info.old_root);
        let circuit_info_iter = Vec::<N>::from(info.circuit_info.clone()).into_iter();

        vec.extend(circuit_info_iter);
        vec.push(info.first_index.into());
        vec.extend(info.low_nullifiers.iter().flat_map(|(entry, proof)| {
            let mut vec: Vec<N> = vec![];
            let entry_iter = Vec::<N>::from(entry).into_iter();
            vec.extend(entry_iter);
            let proof_iter = Vec::<N>::from(proof.clone()).into_iter();
            vec.extend(proof_iter);
            vec
        }));
        vec.extend(info.pending_inserts.iter().flat_map(Vec::<N>::from));
        vec
    }
}

impl<N: PrimeField> From<&IMTCircuitInsertionInfo<N>> for Vec<N> {
    fn from(info: &IMTCircuitInsertionInfo<N>) -> Self {
        let mut vec = vec![];
        vec.push(info.old_root);
        let circuit_info_iter = Vec::<N>::from(info.circuit_info.clone()).into_iter();

        vec.extend(circuit_info_iter);
        vec.push(info.first_index.into());
        vec.extend(info.low_nullifiers.iter().flat_map(|(entry, proof)| {
            let mut vec: Vec<N> = vec![];
            let entry_iter = Vec::<N>::from(entry).into_iter();
            vec.extend(entry_iter);
            let proof_iter = Vec::<N>::from(proof.clone()).into_iter();
            vec.extend(proof_iter);
            vec
        }));
        vec.extend(info.pending_inserts.iter().flat_map(Vec::<N>::from));
        vec
    }
}

impl<N: PrimeField> IMTCircuitInsertionInfo<N> {
    /// Attempts to reconstruct a `IMTCircuitInsertionInfo` from a slice of `N`.
    pub fn from_slice(
        values: &[N],
        height: usize,
        no_leaves: usize,
    ) -> Result<Self, IndexedMerkleTreeError> {
        if values.len() < 2 {
            return Err(IndexedMerkleTreeError::InvalidBatchSize);
        }
        let old_root = values[0];

        let circuit_info_height = height - no_leaves.ilog2() as usize;
        let circuit_info_length = 3 + no_leaves + 2 + 2 * circuit_info_height;

        let circuit_info = CircuitInsertionInfo::from_slice(
            &values[1..1 + circuit_info_length],
            circuit_info_height,
        )?;
        let first_index: BigUint = values[1 + circuit_info_length].into();

        let low_nullifiers_length = (6 + 2 * height) * no_leaves;

        let mut low_nullifiers = vec![];
        values[2 + circuit_info_length..2 + circuit_info_length + low_nullifiers_length]
            .chunks(6 + 2 * height)
            .try_for_each(|chunk| {
                let entry = LeafDBEntry::try_from(&chunk[..4])?;
                let proof = MembershipProof::try_from(chunk[4..].to_vec())?;
                low_nullifiers.push((entry, proof));
                Result::<(), IndexedMerkleTreeError>::Ok(())
            })?;

        let mut pending_inserts = vec![];
        values[2 + circuit_info_length + low_nullifiers_length..]
            .chunks(4)
            .try_for_each(|chunk| {
                let entry = LeafDBEntry::try_from(chunk)?;
                pending_inserts.push(entry);
                Result::<(), IndexedMerkleTreeError>::Ok(())
            })?;

        Ok(IMTCircuitInsertionInfo {
            old_root,
            circuit_info,
            first_index: first_index
                .to_u32()
                .ok_or(IndexedMerkleTreeError::InvalidBatchSize)?,
            low_nullifiers,
            pending_inserts,
        })
    }
}

impl<N, H, DB> IndexedMerkleTree<N, H, DB>
where
    H: TreeHasher<N> + Clone + PartialEq + Send + Sync,
    N: Zero + PrimeField,
    DB: LeafDB<N>,
{
    /// Creates a new IndexedMerkleTree
    pub fn new(hasher: H, height: u32) -> Result<Self, IndexedMerkleTreeError> {
        let mut timber = Timber::<N, H>::new(hasher, height);
        let leaves_db: DB = LeafDB::<N>::new();
        let entry = leaves_db
            .get_nullifier(Some(N::zero()), None)
            .ok_or(IndexedMerkleTreeError::DatabaseError)?;
        let leaf_value = timber
            .hasher
            .hash(&[entry.value, entry.next_index, entry.next_value])?;
        timber.insert_leaf(leaf_value)?;
        Ok(IndexedMerkleTree { timber, leaves_db })
    }

    /// Inserts a leaf into the tree and stores the leaf in the leaves_db
    pub fn insert_leaf(&mut self, inner_leaf_value: N) -> Result<(), IndexedMerkleTreeError> {
        // First we check if the leaf is already in the tree or it has value zero
        if self
            .leaves_db
            .get_nullifier(Some(inner_leaf_value), None)
            .is_some()
            && (inner_leaf_value != N::zero())
        {
            return Err(IndexedMerkleTreeError::LeafAlreadyExists);
        }

        if inner_leaf_value != N::zero() {
            // Now we know the nullifier is not in the tree we can just insert the leaf.
            let low_nullifier = self
                .leaves_db
                .get_low_nullifier(&inner_leaf_value)
                .ok_or(IndexedMerkleTreeError::DatabaseError)?;
            let ln_value = low_nullifier.value;

            let ln_index = low_nullifier.index as usize;

            let leaf_value = self.timber.hasher.hash(&[
                inner_leaf_value,
                low_nullifier.next_index,
                low_nullifier.next_value,
            ])?;

            let directions = index_to_directions(ln_index, self.timber.height);

            self.leaves_db
                .store_nullifier(inner_leaf_value, Some(self.timber.leaf_count as u32))
                .ok_or(IndexedMerkleTreeError::DatabaseError)?;
            self.timber.insert_leaf(leaf_value)?;

            // Now we have to update the low_nullifier leaf in the tree so that it point to this new value.
            let new_low_nullifier = self
                .leaves_db
                .get_nullifier(Some(ln_value), None)
                .ok_or(IndexedMerkleTreeError::DatabaseError)?;

            let leaf_value = self.timber.hasher.hash(&[
                new_low_nullifier.value,
                new_low_nullifier.next_index,
                new_low_nullifier.next_value,
            ])?;

            Timber::<N, H>::update_leaf(&mut self.timber.tree, &directions, leaf_value)?;
            self.timber.root = self.timber.reduce_tree();
        } else {
            // If the nullifier has value zero it will be inserted into the tree but we won't put it in the DB so no other
            // leaf ever points to it. We insert it as a zero value.

            self.timber.insert_leaf(N::zero())?;
        }
        Ok(())
    }

    /// Inserts multiple leaves into the tree.
    pub fn insert_leaves(&mut self, inner_leaf_values: &[N]) -> Result<(), IndexedMerkleTreeError> {
        for leaf in inner_leaf_values {
            if self.leaves_db.get_nullifier(Some(*leaf), None).is_some() && !leaf.is_zero() {
                return Err(IndexedMerkleTreeError::LeafAlreadyExists);
            }
        }
        let mut current_index = self.timber.leaf_count;
        let mut low_nullifiers = inner_leaf_values
            .iter()
            .filter_map(|leaf| {
                if *leaf != N::zero() {
                    Some(
                        self.leaves_db
                            .get_low_nullifier(leaf)
                            .copied()
                            .ok_or(IndexedMerkleTreeError::DatabaseError),
                    )
                } else {
                    None
                }
            })
            .collect::<Result<Vec<LeafDBEntry<N>>, IndexedMerkleTreeError>>()?
            .iter()
            .map(|nullifier| nullifier.value)
            .collect::<Vec<N>>();
        low_nullifiers.sort();
        low_nullifiers.dedup();

        inner_leaf_values
            .iter()
            .try_for_each(|nullifier| {
                if *nullifier != N::zero() {
                    self.leaves_db
                        .store_nullifier(*nullifier, Some(current_index as u32))?;
                    current_index += 1;
                    Some(())
                } else {
                    Some(())
                }
            })
            .ok_or(IndexedMerkleTreeError::DatabaseError)?;
        let leaf_values = inner_leaf_values
            .iter()
            .map(|nullifier| {
                if *nullifier != N::zero() {
                    let db_entry = self
                        .leaves_db
                        .get_nullifier(Some(*nullifier), None)
                        .copied()
                        .ok_or(IndexedMerkleTreeError::DatabaseError)?;
                    self.timber
                        .hasher
                        .hash(&[db_entry.value, db_entry.next_index, db_entry.next_value])
                        .map_err(|e| e.into())
                } else {
                    Ok(*nullifier)
                }
            })
            .collect::<Result<Vec<N>, IndexedMerkleTreeError>>()?;

        self.timber.insert_leaves(&leaf_values)?;

        low_nullifiers.iter().try_for_each(|value| {
            let nullifier = self
                .leaves_db
                .get_nullifier(Some(*value), None)
                .copied()
                .ok_or(IndexedMerkleTreeError::DatabaseError)?;

            let leaf_value = self
                .timber
                .hasher
                .hash(&[nullifier.value, nullifier.next_index, nullifier.next_value])
                .map_err(|e| IndexedMerkleTreeError::TimberError(e.into()))?;
            let directions = index_to_directions(nullifier.index as usize, self.timber.height);
            Timber::<N, H>::update_leaf(&mut self.timber.tree, &directions, leaf_value)
                .map_err(IndexedMerkleTreeError::from)
        })?;

        self.timber.root = self.timber.reduce_tree();

        Ok(())
    }

    /// Returns a non-membership proof for a `inner_leaf_value`.
    pub fn non_membership_proof(
        &self,
        inner_leaf_value: N,
    ) -> Result<MembershipProof<N>, IndexedMerkleTreeError> {
        let low_nullifier = self
            .leaves_db
            .get_low_nullifier(&inner_leaf_value)
            .ok_or(IndexedMerkleTreeError::DatabaseError)?;

        let ln_index = low_nullifier.index as usize;
        let leaf_value = self.timber.hasher.hash(&[
            low_nullifier.value,
            low_nullifier.next_index,
            low_nullifier.next_value,
        ])?;

        let sibling_path = self.timber.get_sibling_path(leaf_value, ln_index).ok_or(
            IndexedMerkleTreeError::TimberError(super::error::TimberError::CannotGetPath),
        )?;

        Ok(MembershipProof {
            node_value: leaf_value,
            sibling_path,
            leaf_index: ln_index,
        })
    }

    /// Verifies a non-membership proof.
    pub fn verify_non_membership_proof(
        &self,
        proof: &MembershipProof<N>,
    ) -> Result<(), IndexedMerkleTreeError> {
        proof
            .verify(&self.timber.root, &self.timber.hasher)
            .map_err(|e| e.into())
    }

    /// Inserts a subtree into the Indexed Merkle Tree. First it checks that all leaves being inserted are not already in the tree.
    pub fn insert_subtree(
        &mut self,
        inner_leaf_values: &[N],
    ) -> Result<(), IndexedMerkleTreeError> {
        if inner_leaf_values.len().next_power_of_two() != inner_leaf_values.len() {
            return Err(IndexedMerkleTreeError::InvalidBatchSize);
        }

        // This is the index of the first leaf of the subtree in the main tree, counted from the left, starting at zero
        let mut first_index =
            ((self.timber.leaf_count as u32 - 1) / inner_leaf_values.len() as u32 + 1)
                * inner_leaf_values.len() as u32;
        let mut low_nullifiers = vec![];

        for leaf in inner_leaf_values {
            if self.leaves_db.get_nullifier(Some(*leaf), None).is_some() && (*leaf != N::zero()) {
                return Err(IndexedMerkleTreeError::LeafAlreadyExists);
            } else if !leaf.is_zero() {
                let low_nullifier = self
                    .leaves_db
                    .get_low_nullifier(leaf)
                    .ok_or(IndexedMerkleTreeError::DatabaseError)?;

                low_nullifiers.push(low_nullifier.value);
            } else {
                continue;
            }
        }

        low_nullifiers.sort();
        low_nullifiers.dedup();

        // First we loop through and store all the values if they aren't zero.
        for inner_value in inner_leaf_values {
            if !inner_value.is_zero() {
                self.leaves_db
                    .store_nullifier(*inner_value, Some(first_index))
                    .ok_or(IndexedMerkleTreeError::DatabaseError)?;
            }
            first_index += 1;
        }
        // Now we update all the low nullifiers in the tree
        low_nullifiers.iter().try_for_each(|value| {
            let nullifier = self
                .leaves_db
                .get_nullifier(Some(*value), None)
                .copied()
                .ok_or(IndexedMerkleTreeError::DatabaseError)?;

            let leaf_value = self
                .timber
                .hasher
                .hash(&[nullifier.value, nullifier.next_index, nullifier.next_value])
                .map_err(|e| IndexedMerkleTreeError::TimberError(e.into()))?;
            let directions = index_to_directions(nullifier.index as usize, self.timber.height);
            Timber::<N, H>::update_leaf(&mut self.timber.tree, &directions, leaf_value)
                .map_err(IndexedMerkleTreeError::from)
        })?;

        self.timber.root = self.timber.reduce_tree();

        let mut subtree_leaves = vec![];

        // Then after storing all of the values and updating low nullifiers (so all the entries will be correct) we calculate their leaf hashes
        for inner_value in inner_leaf_values {
            if *inner_value != N::zero() {
                let entry = self
                    .leaves_db
                    .get_nullifier(Some(*inner_value), None)
                    .ok_or(IndexedMerkleTreeError::DatabaseError)?;
                let leaf_value =
                    self.timber
                        .hasher
                        .hash(&[entry.value, entry.next_index, entry.next_value])?;
                subtree_leaves.push(leaf_value);
            } else {
                subtree_leaves.push(N::zero());
            }
        }

        let subtree_height = inner_leaf_values.len().ilog2();
        let subtree = Tree::<N>::build_from_values(subtree_height, &subtree_leaves);

        self.timber.insert_subtree(subtree)?;

        Ok(())
    }

    /// This function is used to insert a subtree into the IMT and returns all the relevant information so that we can later prove this in a circuit context.
    pub fn insert_for_circuit(
        &mut self,
        inner_values: &[N],
    ) -> Result<IMTCircuitInsertionInfo<N>, IndexedMerkleTreeError> {
        if inner_values.len().next_power_of_two() != inner_values.len() {
            return Err(IndexedMerkleTreeError::InvalidBatchSize);
        }

        let old_root = self.timber.root;

        // This is the index of the first leaf of the subtree in the main tree, counted from the left, starting at zero
        let mut initial_index = ((self.timber.leaf_count as u32 - 1) / inner_values.len() as u32
            + 1)
            * inner_values.len() as u32;

        let first_index = initial_index;

        let mut pending_inserts = vec![];
        let mut low_nullifiers = vec![];

        for &inner_value in inner_values.iter() {
            // First we get the low nullifier for the leaf we are inserting if the value is non-zero and does not already exist
            if self
                .leaves_db
                .get_nullifier(Some(inner_value), None)
                .is_some()
                && (!inner_value.is_zero())
            {
                return Err(IndexedMerkleTreeError::LeafAlreadyExists);
            }

            if !inner_value.is_zero() {
                let low_nullifier = self
                    .leaves_db
                    .get_low_nullifier(&inner_value)
                    .copied()
                    .ok_or(IndexedMerkleTreeError::DatabaseError)?;

                // Now we check if the low nullifier is in the tree already, if it is not then it is one of the pending inserts
                let proof = if let Ok(proof) = self.non_membership_proof(inner_value) {
                    self.leaves_db
                        .store_nullifier(inner_value, Some(initial_index))
                        .ok_or(IndexedMerkleTreeError::DatabaseError)?;

                    initial_index += 1;

                    let updated_nullifier = self
                        .leaves_db
                        .get_nullifier(Some(low_nullifier.value), None)
                        .ok_or(IndexedMerkleTreeError::DatabaseError)?;

                    let updated_leaf = self.timber.hasher.hash(&[
                        updated_nullifier.value,
                        updated_nullifier.next_index,
                        updated_nullifier.next_value,
                    ])?;
                    let ln_index = updated_nullifier.index as usize;
                    let directions = index_to_directions(ln_index, self.timber.height);

                    Timber::<N, H>::update_leaf(&mut self.timber.tree, &directions, updated_leaf)?;
                    self.timber.root = self.timber.reduce_tree();

                    proof
                } else {
                    // If we couldn't get the non-membership proof its because its a pending insert so we return a proof where everything is zero.

                    self.leaves_db
                        .store_nullifier(inner_value, Some(initial_index))
                        .ok_or(IndexedMerkleTreeError::DatabaseError)?;
                    initial_index += 1;
                    let node_value = self.timber.hasher.hash(&[
                        low_nullifier.value,
                        low_nullifier.next_index,
                        low_nullifier.next_value,
                    ])?;
                    MembershipProof {
                        node_value,
                        sibling_path: vec![
                            PathElement {
                                value: N::zero(),
                                direction: Directions::HashWithThisNodeOnLeft,
                            };
                            self.timber.height as usize
                        ],
                        leaf_index: initial_index as usize - 1,
                    }
                };

                low_nullifiers.push((low_nullifier, proof));
            } else {
                low_nullifiers.push((
                    LeafDBEntry::<N>::default(),
                    MembershipProof {
                        node_value: N::zero(),
                        sibling_path: vec![
                            PathElement {
                                value: N::zero(),
                                direction: Directions::HashWithThisNodeOnLeft,
                            };
                            self.timber.height as usize
                        ],
                        leaf_index: 0,
                    },
                ));
                initial_index += 1;
            }
        }

        for inner_value in inner_values.iter() {
            if !inner_value.is_zero() {
                let pending_insert = self
                    .leaves_db
                    .get_nullifier(Some(*inner_value), None)
                    .copied()
                    .ok_or(IndexedMerkleTreeError::DatabaseError)?;

                pending_inserts.push(pending_insert);
            } else {
                pending_inserts.push(LeafDBEntry::<N>::default());
            }
        }
        // Build the subtree to insert.
        let new_leaf_values = pending_inserts
            .iter()
            .map(|entry| {
                if !entry.value.is_zero() {
                    self.timber
                        .hasher
                        .hash(&[entry.value, entry.next_index, entry.next_value])
                } else {
                    Ok(N::zero())
                }
            })
            .collect::<Result<Vec<N>, PoseidonError>>()?;

        let circuit_info = self.timber.insert_for_circuit(&new_leaf_values)?;

        Ok(IMTCircuitInsertionInfo {
            old_root,
            circuit_info,
            first_index,
            low_nullifiers,
            pending_inserts,
        })
    }

    /// This function batch inserts a number of leaves into the IMT and returns all the relevant information so that we can later prove this in a circuit context.
    pub fn batch_insert_for_circuit(
        &mut self,
        inner_values: &[N],
    ) -> Result<Vec<IMTCircuitInsertionInfo<N>>, IndexedMerkleTreeError> {
        if inner_values.is_empty() {
            return Ok(vec![]);
        }
        let mut circuit_infos = vec![];
        for leaf_chunk in inner_values.chunks(8) {
            let circuit_info = self.insert_for_circuit(leaf_chunk)?;
            circuit_infos.push(circuit_info);
        }
        Ok(circuit_infos)
    }
}
