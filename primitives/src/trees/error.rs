//! Timber and tree related errors

use crate::poseidon::PoseidonError;
use ark_std::{
    error::Error,
    fmt::{Display, Formatter, Result},
    string::String,
};

#[derive(Debug)]
/// Error enum for the Timber tree.
pub enum TimberError {
    /// The tree is full and cannot accept any more leaves
    TreeIsFull,
    /// Cannot roll back to this leaf count, it is higher than the current leaf count
    CannotRollBack,
    /// Cannot get the sibling path for this leaf/node
    CannotGetPath,
    /// Wrong node type
    WrongNodeType,
    /// Can only insert a power of 2 sized batch
    InvalidBatchSize,
    /// Invalid membership proof
    InvalidMembershipProof,
    /// Hashing error
    HashingError(PoseidonError),
    /// Parameter error
    ParameterError(String),
}

impl Display for TimberError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            TimberError::TreeIsFull => write!(f, "Tree is full"),
            TimberError::CannotRollBack => write!(
                f,
                "Cannot roll back to this leaf count, it is higher than the current leaf count"
            ),
            TimberError::CannotGetPath => write!(f, "Cannot get the sibling path for this leaf"),
            TimberError::WrongNodeType => write!(f, "Wrong node type"),
            TimberError::InvalidBatchSize => write!(f, "Can only insert a power of 2 sized batch"),
            TimberError::InvalidMembershipProof => write!(f, "Invalid membership proof"),
            TimberError::HashingError(e) => write!(f, "Hashing error: {}", e),
            TimberError::ParameterError(e) => write!(f, "Parameter error: {}", e),
        }
    }
}

impl Error for TimberError {}

impl From<PoseidonError> for TimberError {
    fn from(e: PoseidonError) -> Self {
        TimberError::HashingError(e)
    }
}

#[derive(Debug)]
/// Enum for Indexed Merkle Tree related errors.
pub enum IndexedMerkleTreeError {
    /// The tree is full and cannot accept any more leaves
    TreeIsFull,
    /// Cannot roll back to this leaf count, it is higher than the current leaf count
    CannotRollBack,
    /// Cannot get the sibling path for this leaf/node
    LeafAlreadyExists,
    /// Error querying the leaf database
    DatabaseError,
    /// Division related error
    DivisionError,
    /// Hashing with Poseidon error
    HashingError(PoseidonError),
    /// Timber error
    TimberError(TimberError),
    /// Batch insert error
    InvalidBatchSize,
}

impl Display for IndexedMerkleTreeError {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            IndexedMerkleTreeError::TreeIsFull => write!(f, "Tree is full"),
            IndexedMerkleTreeError::CannotRollBack => write!(
                f,
                "Cannot roll back to this leaf count, it is higher than the current leaf count"
            ),
            IndexedMerkleTreeError::LeafAlreadyExists => write!(f, "Leaf already exists"),
            IndexedMerkleTreeError::DatabaseError => write!(f, "Database error"),
            IndexedMerkleTreeError::HashingError(e) => write!(f, "Hashing error: {}", e),
            IndexedMerkleTreeError::TimberError(e) => write!(f, "Timber error: {}", e),
            IndexedMerkleTreeError::DivisionError => write!(
                f,
                "number of leaves to insert was not divisible by batch_size"
            ),
            IndexedMerkleTreeError::InvalidBatchSize => {
                write!(f, "Can only insert a power of 2 sized batch")
            },
        }
    }
}

impl Error for IndexedMerkleTreeError {}

impl From<PoseidonError> for IndexedMerkleTreeError {
    fn from(e: PoseidonError) -> Self {
        IndexedMerkleTreeError::HashingError(e)
    }
}

impl From<TimberError> for IndexedMerkleTreeError {
    fn from(e: TimberError) -> Self {
        IndexedMerkleTreeError::TimberError(e)
    }
}
