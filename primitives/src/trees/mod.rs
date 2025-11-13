//! Code relating to Merkle Trees.

use crate::poseidon::{constants::PoseidonParams, FieldHasher, Poseidon, PoseidonError};
use ark_ff::PrimeField;
use ark_std::{
    boxed::Box, cfg_chunks, cfg_iter, cmp::max, collections::HashMap, string::ToString, vec,
    vec::Vec, Zero,
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use self::{
    error::TimberError,
    imt::{LeafDB, LeafDBEntry},
};
pub mod error;
pub mod imt;
pub mod timber;

// Note - in this module we unwrap() the result of hashing and of U32 <-> usize conversion. We're ok to do
// This because neither represents a recoverable situation; they're not errors we can handle but bugs
// that we need to fix see https://blog.burntsushi.net/unwrap/ condition 1.

/// We use this trait to specialise to the case when we use zero as an empty value in a tree.
/// If hashing zero against zero is not zero then we need to implement this trait.
pub trait TreeHasher<N>: FieldHasher<N> + Clone + PartialEq
where
    N: PrimeField,
{
    /// Perform a hash of the values, but if both values are zero then return zero.
    fn tree_hash(&self, values: &[N; 2]) -> Result<N, PoseidonError> {
        if values[0] == N::zero() && values[1] == N::zero() {
            return Ok(N::zero());
        }
        <Self as FieldHasher<N>>::hash(self, values)
    }
}

impl<N> TreeHasher<N> for Poseidon<N> where N: PoseidonParams {}

/// A struct storing a membership proof for a node in a Merkle Tree
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MembershipProof<N> {
    /// The leaf value for which the proof is generated.
    pub node_value: N,
    /// The sibling path for the leaf value.
    pub sibling_path: Vec<PathElement<N>>,
    /// The index of the leaf in the tree.
    pub leaf_index: usize,
}

impl<N: Copy + Zero> Default for MembershipProof<N> {
    fn default() -> Self {
        Self {
            node_value: N::zero(),
            sibling_path: vec![PathElement::default(); 32],
            leaf_index: 0,
        }
    }
}

impl<N: PrimeField> From<MembershipProof<N>> for Vec<N> {
    fn from(value: MembershipProof<N>) -> Self {
        let mut out = Vec::new();
        out.push(value.node_value);
        for path_element in value.sibling_path.iter() {
            match path_element.direction {
                Directions::HashWithThisNodeOnLeft => {
                    out.push(N::zero());
                },
                Directions::HashWithThisNodeOnRight => {
                    out.push(N::one());
                },
            }
            out.push(path_element.value);
        }
        out.push(N::from(value.leaf_index as u64));
        out
    }
}

impl<N: PrimeField> From<&MembershipProof<N>> for Vec<N> {
    fn from(value: &MembershipProof<N>) -> Self {
        let mut out = Vec::new();
        out.push(value.node_value);
        for path_element in value.sibling_path.iter() {
            match path_element.direction {
                Directions::HashWithThisNodeOnLeft => {
                    out.push(N::zero());
                },
                Directions::HashWithThisNodeOnRight => {
                    out.push(N::one());
                },
            }
            out.push(path_element.value);
        }
        out.push(N::from(value.leaf_index as u64));
        out
    }
}

impl<N: PrimeField> TryFrom<Vec<N>> for MembershipProof<N> {
    type Error = TimberError;

    fn try_from(value: Vec<N>) -> Result<Self, Self::Error> {
        if value.is_empty() {
            return Err(TimberError::ParameterError(
                "Vector cannot be empty".to_string(),
            ));
        }
        if value.len() % 2 != 0 {
            return Err(TimberError::ParameterError(
                "Vector length should be even".to_string(),
            ));
        }

        let node_value = value[0];
        let mut sibling_path = Vec::new();
        for chunk in value[1..].chunks_exact(2) {
            let direction_val = chunk[0];

            let value = chunk[1];
            let direction = if direction_val == N::zero() {
                Directions::HashWithThisNodeOnLeft
            } else if direction_val == N::one() {
                Directions::HashWithThisNodeOnRight
            } else {
                return Err(TimberError::ParameterError(
                    "Invalid direction value".to_string(),
                ));
            };
            let path_element = PathElement { direction, value };
            sibling_path.push(path_element);
        }
        let biguint: BigUint = value[value.len() - 1].into();
        let leaf_index = biguint.to_usize().unwrap();
        Ok(MembershipProof {
            node_value,
            sibling_path,
            leaf_index,
        })
    }
}

impl<N> MembershipProof<N> {
    /// Tries to verify the proof for a given root.
    pub fn verify<H: TreeHasher<N>>(&self, root: &N, hasher: &H) -> Result<(), TimberError>
    where
        N: PrimeField,
    {
        let mut leaf_value = self.node_value;
        for path_element in self.sibling_path.iter() {
            match path_element.direction {
                Directions::HashWithThisNodeOnLeft => {
                    leaf_value = hasher.tree_hash(&[path_element.value, leaf_value])?;
                },
                Directions::HashWithThisNodeOnRight => {
                    leaf_value = hasher.tree_hash(&[leaf_value, path_element.value])?;
                },
            }
        }
        if leaf_value != *root {
            return Err(TimberError::InvalidMembershipProof);
        }
        Ok(())
    }
}

/// A struct that stores all the information needed to prove correct subtree insertion in a circuit context.
#[derive(Clone, Debug)]
pub struct CircuitInsertionInfo<N> {
    /// The root of the tree before the insertion.
    pub old_root: N,
    /// The root of the tree after the insertion.
    pub new_root: N,
    /// Leaf count pre-insertion.
    pub leaf_count: usize,
    /// The leaves inserted.
    pub leaves: Vec<N>,
    /// The Merkle proof that proves the subtree we are inserting into was empty.
    pub proof: MembershipProof<N>,
}

impl<N: PrimeField> From<CircuitInsertionInfo<N>> for Vec<N> {
    fn from(value: CircuitInsertionInfo<N>) -> Self {
        let mut out = Vec::new();
        out.push(value.old_root);
        out.push(value.new_root);
        out.push(N::from(value.leaf_count as u64));
        out.extend(value.leaves);
        out.extend(Vec::<N>::from(value.proof));
        out
    }
}

impl<N: PrimeField> From<&CircuitInsertionInfo<N>> for Vec<N> {
    fn from(value: &CircuitInsertionInfo<N>) -> Self {
        let mut out = Vec::new();
        out.push(value.old_root);
        out.push(value.new_root);
        out.push(N::from(value.leaf_count as u64));
        for leaf in value.leaves.iter() {
            out.push(*leaf);
        }
        out.extend(Vec::<N>::from(&value.proof));
        out
    }
}

impl<N: PrimeField> CircuitInsertionInfo<N> {
    /// Tries to create a CircuitInsertionInfo from a slice. Here `height` is the height of the tree.
    pub fn from_slice(value: &[N], height: usize) -> Result<Self, TimberError> {
        // The length of the slice should be at least 3
        if value.len() < 3 {
            return Err(TimberError::InvalidMembershipProof);
        }
        let old_root = value[0];
        let new_root = value[1];
        let leaf_count_biguint: BigUint = value[2].into();
        let leaf_count = leaf_count_biguint
            .to_usize()
            .ok_or(TimberError::InvalidMembershipProof)?;
        // We can calculate the number of leaves because the last 2 + height * 2 elements are the membership proof.
        let membership_proof_size = 2 + height * 2;
        let leaves = value[3..value.len() - membership_proof_size].to_vec();

        let proof =
            MembershipProof::try_from(value[(value.len() - membership_proof_size)..].to_vec())?;
        Ok(CircuitInsertionInfo {
            old_root,
            new_root,
            leaf_count,
            leaves,
            proof,
        })
    }
}

/// This function is called recursively to traverse down the tree using the known insertion path
/// leafVal - the commitment has to be inserted into the tree
/// tree - The tree where leafVal will be inserted
/// path - The path down tree that leafVal will be inserted into
fn _insert_leaf<N: Zero + Clone>(leaf_val: N, tree: &Tree<N>, path: &[Directions]) -> Tree<N> {
    // The base case is when we have reached the end of the path, we return the the leafVal as a Leaf Object
    if path.is_empty() {
        return Tree::<N>::Leaf(leaf_val);
    }
    match tree {
        Tree::Branch(sub_trees) => {
            // decide whether to go left or right depending on the path (sub_trees.0 is the left sub tree).
            match path[0] {
                Directions::HashWithThisNodeOnLeft => Tree::Branch(Box::new((
                    _insert_leaf(leaf_val, &sub_trees.0, &path[1..]),
                    sub_trees.1.clone(),
                ))),
                Directions::HashWithThisNodeOnRight => Tree::Branch(Box::new((
                    sub_trees.0.clone(),
                    _insert_leaf(leaf_val, &sub_trees.1, &path[1..]),
                ))),
            }
        },
        // If we are at a leaf AND path.length > 0, we need to expand the undeveloped subtree
        // We then use the next element in path to decided which subtree to traverse
        Tree::Leaf(_) => match path[0] {
            Directions::HashWithThisNodeOnLeft => Tree::Branch(Box::new((
                _insert_leaf(leaf_val, &Tree::Leaf(N::zero()), &path[1..]),
                Tree::Leaf(N::zero()),
            ))),
            Directions::HashWithThisNodeOnRight => Tree::Branch(Box::new((
                Tree::Leaf(N::zero()),
                _insert_leaf(leaf_val, &Tree::Leaf(N::zero()), &path[1..]),
            ))),
        },
    }
}

/// This function is like _insertLeaf but it doesnt prune children on insertion
/// leafVal - the commitment has to be inserted into the tree
/// tree - The tree where leafVal will be inserted
/// path - The path down tree that leafVal will be inserted into
fn _safe_insert_leaf<N: Zero + PartialEq + Clone>(
    leaf_val: N,
    tree: &Tree<N>,
    path: &[Directions],
) -> Tree<N> {
    // The base case is when we have reached the end of the path, we return the the leafVal as a Leaf Object
    if path.is_empty() {
        match &tree {
            Tree::Branch(_) => {
                return tree.clone();
            },
            Tree::Leaf(value) => {
                if *value == N::zero() {
                    return Tree::Leaf(leaf_val);
                }
                return tree.clone();
            },
        }
    }
    match tree {
        Tree::Branch(sub_trees) => {
            // decide whether to go left or right depending on the path (sub_trees.0 is the left sub tree).
            match path[0] {
                Directions::HashWithThisNodeOnLeft => Tree::Branch(Box::new((
                    _safe_insert_leaf(leaf_val, &sub_trees.0, &path[1..]),
                    sub_trees.1.clone(),
                ))),
                Directions::HashWithThisNodeOnRight => Tree::Branch(Box::new((
                    sub_trees.0.clone(),
                    _safe_insert_leaf(leaf_val, &sub_trees.1, &path[1..]),
                ))),
            }
        },
        // If we are at a leaf AND path.length > 0, we need to expand the undeveloped subtree
        // We then use the next element in path to decided which subtree to traverse
        Tree::Leaf(_) => match path[0] {
            Directions::HashWithThisNodeOnLeft => Tree::Branch(Box::new((
                _safe_insert_leaf(leaf_val, &Tree::Leaf(N::zero()), &path[1..]),
                Tree::Leaf(N::zero()),
            ))),
            Directions::HashWithThisNodeOnRight => Tree::Branch(Box::new((
                Tree::Leaf(N::zero()),
                _safe_insert_leaf(leaf_val, &Tree::Leaf(N::zero()), &path[1..]),
            ))),
        },
    }
}

/// This function is like _insertLeaf but it doesnt prune children on insertion
/// leafVal - the commitment has to be inserted into the tree
/// tree - The tree where leafVal will be inserted
/// path - The path down tree that leafVal will be inserted into
fn insert_node<N: Zero + PartialEq + Clone>(
    node: &Tree<N>,
    tree: &mut Tree<N>,
    path: &[Directions],
) -> Option<()> {
    // The base case is when we have reached the end of the path, we return the the leafVal as a Leaf Object
    if path.is_empty() {
        *tree = node.clone();
        return Some(());
    }
    match tree {
        Tree::Branch(sub_trees) => {
            // decide whether to go left or right depending on the path (sub_trees.0 is the left sub tree).
            match path[0] {
                Directions::HashWithThisNodeOnLeft => {
                    insert_node(node, &mut sub_trees.0, &path[1..])
                },
                Directions::HashWithThisNodeOnRight => {
                    insert_node(node, &mut sub_trees.1, &path[1..])
                },
            }
        },
        // If we are at a leaf AND path.length > 0, we need to expand the undeveloped subtree
        // We then use the next element in path to decided which subtree to traverse
        Tree::Leaf(_) => {
            let remaining =
                path.iter()
                    .rev()
                    .fold(node.clone(), |acc, direction| match direction {
                        Directions::HashWithThisNodeOnLeft => {
                            Tree::Branch(Box::new((acc, Tree::Leaf(N::zero()))))
                        },
                        Directions::HashWithThisNodeOnRight => {
                            Tree::Branch(Box::new((Tree::Leaf(N::zero()), acc)))
                        },
                    });
            *tree = remaining;
            Some(())
        },
    }
}

/// This function is called recursively to traverse down the tree to find check set membership
/// leafVal - The commitment hash that is being checked
/// tree - The tree that will be checked
/// path - The path down tree that leafVal is stored
/// tree_hasher - This is the function that reduces the unexplored subtree (e.g. hash) in the membership check
/// acc - This is the array that contains the membership proof
fn _check_membership<N>(
    leaf_val: N,
    tree: &Tree<N>,
    path: &[Directions],
    tree_hasher: impl Fn(&Tree<N>) -> N,
    acc: &mut Vec<PathElement<N>>,
) -> Option<Vec<PathElement<N>>>
where
    N: PartialEq + Clone + PrimeField,
{
    match tree {
        Tree::Branch(sub_trees) => match path[0] {
            Directions::HashWithThisNodeOnLeft => {
                acc.push(PathElement {
                    direction: Directions::HashWithThisNodeOnRight,
                    value: tree_hasher(&sub_trees.1),
                });
                _check_membership(leaf_val, &sub_trees.0, &path[1..], tree_hasher, acc)
            },
            Directions::HashWithThisNodeOnRight => {
                acc.push(PathElement {
                    direction: Directions::HashWithThisNodeOnLeft,
                    value: tree_hasher(&sub_trees.0),
                });
                _check_membership(leaf_val, &sub_trees.1, &path[1..], tree_hasher, acc)
            },
        },
        // If we arrive at a leaf, we check if the value at the leaf matches the element we are looking for.
        Tree::Leaf(value) => {
            if *value != leaf_val {
                None
            } else {
                acc.reverse();
                Some(acc.to_vec())
            }
        },
    }
}

/// This function can be used to retrieve the sibling path of a node in the tree that may be in the empty part that is yet to be inserted.
/// This is useful in a circuit context when we do leaf insertions because we need to prove that the tree was empty in the location we are inserting into.
///
/// Here `height` is the distance from the node to the root of the tree i.e. if we had a tree of height 32 and we wanted to get the sibling path of a node
/// that was the root of a subtree of height 3 then the input `height` would be 29. The value `index` is how far along the tree it would be if the tree was only
/// `height` deep. So if we had a tree of height 32 and we wanted to get the sibling path of a node that was the root of a subtree of height 3 and was the 5th
/// node along the tree then the input `index` would be 4 because we zero index.
pub fn get_node_sibling_path<N>(
    tree: &Tree<N>,
    height: u32,
    index: usize,
    tree_hasher: impl Fn(&Tree<N>) -> N,
) -> Vec<PathElement<N>>
where
    N: Clone + PartialEq + PrimeField,
{
    let mut path = Vec::<PathElement<N>>::new();
    let mut current_tree = tree.clone();
    let mut sibling_path = index_to_directions(index, height);
    let mut current_height = height;

    while current_height > 0 {
        match current_tree {
            Tree::Branch(sub_trees) => {
                if let Directions::HashWithThisNodeOnLeft = sibling_path[0] {
                    path.push(PathElement {
                        direction: Directions::HashWithThisNodeOnRight,
                        value: tree_hasher(&sub_trees.1),
                    });
                    current_tree = sub_trees.0;
                    sibling_path.remove(0);
                } else {
                    path.push(PathElement {
                        direction: Directions::HashWithThisNodeOnLeft,
                        value: tree_hasher(&sub_trees.0),
                    });
                    current_tree = sub_trees.1;
                    sibling_path.remove(0);
                }
            },
            Tree::Leaf(_) => {
                if let Directions::HashWithThisNodeOnLeft = sibling_path[0] {
                    path.push(PathElement {
                        direction: Directions::HashWithThisNodeOnRight,
                        value: N::zero(),
                    });
                    current_tree = Tree::Leaf(N::zero());
                    sibling_path.remove(0);
                } else {
                    path.push(PathElement {
                        direction: Directions::HashWithThisNodeOnLeft,
                        value: N::zero(),
                    });
                    current_tree = Tree::Leaf(N::zero());
                    sibling_path.remove(0);
                }
            },
        }

        current_height -= 1;
    }
    path.into_iter().rev().collect::<Vec<_>>()
}

/// This function combines two fontier vectors by replacing
/// the first elements in the longer array with those in the
/// shorter array
fn combine_frontiers<N: Clone>(arr1: &[N], arr2: &[N]) -> Vec<N> {
    if arr1.len() > arr2.len() {
        let from_arr1 = &arr1[arr2.len()..];
        return [arr2, from_arr1].concat();
    }
    let from_arr2 = &arr2[arr1.len()..];
    [arr1, from_arr2].concat()
}

/// converts a leaf index into a path up the Merkle tree from the leaf
fn index_to_directions(index: usize, height: u32) -> Vec<Directions> {
    let mut path = Vec::<Directions>::new();
    for i in 0..height {
        let dir = index >> i & 1;
        if dir == 0 {
            path.push(Directions::HashWithThisNodeOnLeft)
        } else {
            path.push(Directions::HashWithThisNodeOnRight)
        }
    }
    path.reverse(); // the act of shifting digits off the end of 'index' gives us the path in reverse order compared to a string rep
    path
}

/// This function inserts a sibling path into a tree.
/// tree - The tree this sibling path will be inserted into.
/// sibling_path - The sibling path to be inserted.
/// index - The leafIndex corresponding to this valid sibling path.
/// value - The leafValue corresponding to this valid sibling path.
/// returns a tree object updated with this sibling path.
fn insert_sibling_path<N: Clone + Zero + PartialEq + Copy>(
    tree: Tree<N>,
    sibling_path: &[PathElement<N>],
    index: usize,
    value: N,
    height: u32,
) -> Tree<N> {
    let path_to_index = index_to_directions(index, height);
    let usize_ht = height as usize;
    let sibling_index_path = sibling_path
        .iter()
        .enumerate()
        .map(|(i, s)| match s.direction {
            Directions::HashWithThisNodeOnLeft => (
                s.value,
                [
                    &path_to_index[0..usize_ht - i - 1],
                    &[Directions::HashWithThisNodeOnLeft],
                ]
                .concat(),
            ),
            Directions::HashWithThisNodeOnRight => (
                s.value,
                [
                    &path_to_index[0..usize_ht - i - 1],
                    &[Directions::HashWithThisNodeOnRight],
                ]
                .concat(),
            ),
        })
        .collect::<Vec<_>>();
    let all_paths = [&[(value, path_to_index)], sibling_index_path.as_slice()].concat();
    all_paths
        .iter()
        .fold(tree, |acc, el| _safe_insert_leaf(el.0, &acc, &el.1))
}

/// We do batch insertions when doing stateless operations, the size of each batch is dependent on the tree structure.
/// Each batch insert has to be less than or equal to the next closest power of two - otherwise we may unbalance the tree.
/// E.g. original_leaf_count = 2, leaves.length = 9 -> BatchInserts = [2,4,3]
/// original_leaf_count - The leaf count of the tree we want to insert into.
/// leaves - The elements to be inserted into the tree
/// acc - Used to eliminate tail calls and make recursion more efficient.
/// returns an array of arrays containing paritioned elements of leaves, in the order to be inserted.
fn batch_leaves<'a, N: Clone>(
    original_leaf_count: usize,
    leaves: &'a [N],
    mut acc: Vec<&'a [N]>,
) -> Vec<&'a [N]> {
    if leaves.is_empty() {
        return acc.to_vec();
    }
    let output_leaf_count = original_leaf_count + leaves.len();
    let output_frontier_length = usize::try_from(output_leaf_count.ilog2() + 1).unwrap(); // this can only fail for an impractically large tree of height that won't fit a u32
                                                                                          // This is an array that counts the number of perfect trees at each depth for the current frontier.
                                                                                          // This is padded to be as long as the resultingFrontierSlot.
    let current_frontier_slot_array = vec![original_leaf_count; output_frontier_length]
        .iter()
        .enumerate()
        .map(|(i, a)| a / 2usize.pow(u32::try_from(i).unwrap())) // this can only fail for an impractically large tree of height that won't fit a u32
        .collect::<Vec<_>>();
    // The height of the subtree that would be created by the new leaves
    // log2.ceil for an unsigned integer 'a' is u32::BITS - (a-1).leading_zeros(); or whatever u type is of interest
    let sub_tree_height = usize::BITS - (leaves.len() - 1).leading_zeros();
    // Since we are trying to add in batches, we have to be sure that the
    // new tree created from the incoming leaves are added correctly to the existing tree
    // We need to work out if adding the subtree directly would impact the balance of the tree.
    // We achieve this by identifying if the perfect tree count at the height of the incoming tree, contains any odd counts
    let odd_depths = current_frontier_slot_array[0..usize::try_from(sub_tree_height).unwrap()]
        .iter()
        .map(|a| a % 2 != 0)
        .collect::<Vec<_>>();
    // If there are odd counts, we fix the lowest one first
    let odd_index = odd_depths.iter().position(|a| *a);
    if let Some(oi) = odd_index {
        // We can "round a tree out" (i.e. make it perfect) by inserting 2^depth leaves from the incoming set first.
        let leaves_to_slice = 2_usize.pow(oi.try_into().unwrap());
        let new_leaves = &leaves[leaves_to_slice..];
        return batch_leaves(original_leaf_count + leaves_to_slice, new_leaves, {
            acc.push(&leaves[0..leaves_to_slice]);
            acc
        });
    }
    acc.push(leaves);
    acc
}

/// This associated function helpfully traverses the tree to apply the function to all leaves.
/// f - A function that operates on Leaf values
/// tree - The tree to be traversed
/// returns the tree with f applied to all leaves
pub fn map_tree<M, N>(f: &M, tree: Tree<N>) -> Tree<N>
where
    M: Fn(N) -> N,
{
    match tree {
        Tree::Branch(sub_trees) => Tree::Branch(Box::new((
            map_tree(&f, sub_trees.0),
            map_tree(&f, sub_trees.1),
        ))),
        Tree::Leaf(value) => Tree::Leaf(f(value)),
    }
}

/// This associated function is like mapTree except the values are accumulated to the root
/// f - A binary (as in two parameters) function that operates on Leaf values
/// tree - The tree to be traversed
/// returns the result of the accumulations as due to f
fn reduce_tree<R, N>(f: &R, tree: &Tree<N>) -> N
where
    R: TreeHasher<N>,
    N: PrimeField,
{
    match tree {
        Tree::Branch(sub_trees) => f
            .tree_hash(&[reduce_tree(f, &sub_trees.0), reduce_tree(f, &sub_trees.1)])
            .unwrap(), // if the hashing fails then our tree is borked. We can't recover from that.
        Tree::Leaf(value) => *value,
    }
}

/// This associate function returns a closure that will hash the tree
/// tree - The tree that will be hashed
/// returns the hash result;
fn get_tree_hasher<H, N>(hasher: &H) -> impl Fn(&Tree<N>) -> N + '_
where
    H: TreeHasher<N>,
    N: PrimeField,
{
    |tree| reduce_tree::<H, N>(hasher, tree)
}

/// This recursive enum represents a general Merkle tree node, with either its two subtrees or a
/// leaf value (if it's at the tip of a branch).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tree<N> {
    /// A node in the tree with children
    Branch(Box<(Tree<N>, Tree<N>)>),
    /// A leaf in the tree or a pruned subtree
    Leaf(N),
}

impl<N> Tree<N> {
    /// This functions gets the height of a tree (it returns the longest path to a leaf)
    fn height(&self) -> u32 {
        match self {
            Tree::Branch(sub_trees) => 1 + max(sub_trees.0.height(), sub_trees.1.height()),
            Tree::Leaf(_) => 0,
        }
    }

    /// Returns the number of leaves in a tree
    fn leaf_count(&self) -> usize {
        match self {
            Tree::Branch(sub_trees) => sub_trees.0.leaf_count() + sub_trees.1.leaf_count(),
            Tree::Leaf(_) => 1,
        }
    }

    /// Builds a tree from a list of leaf values (in order)
    pub fn build_from_values(height: u32, values: &[N]) -> Self
    where
        N: Clone + Zero + PrimeField,
    {
        let min_length = 1 << height;

        let mut nodes = cfg_iter!(values)
            .map(|v| Tree::Leaf(*v))
            .collect::<Vec<_>>();
        nodes.resize_with(min_length, || Tree::Leaf(N::zero()));

        for _ in 0..height {
            nodes = cfg_chunks!(nodes, 2)
                .map(|sub_trees| {
                    Tree::Branch(Box::new((sub_trees[0].clone(), sub_trees[1].clone())))
                })
                .collect::<Vec<_>>();
        }

        nodes.pop().unwrap()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
/// A simple Enum intented to tell one how to hash a node with its counterpart.
pub enum Directions {
    /// Hash with this node on the left
    HashWithThisNodeOnLeft,
    /// Hash with this node on the right
    HashWithThisNodeOnRight,
}

/// an element in a Merkle proof path
#[derive(Clone, PartialEq, Debug, Deserialize, Serialize)]
pub struct PathElement<N> {
    /// Tells you whether to place this node on the left or right
    pub direction: Directions,
    /// The value to place in the hash function
    pub value: N,
}

impl<N: Zero + Copy> Default for PathElement<N> {
    fn default() -> Self {
        Self {
            direction: Directions::HashWithThisNodeOnLeft,
            value: N::zero(),
        }
    }
}

impl<F: PoseidonParams> LeafDB<F> for HashMap<F, LeafDBEntry<F>> {
    fn new() -> Self {
        let mut out = HashMap::<F, LeafDBEntry<F>>::new();
        out.insert(F::zero(), LeafDBEntry::<F>::default());
        out
    }

    fn store_nullifier(&mut self, nullifier: F, index: Option<u64>) -> Option<()> {
        // If the new nullifier is already in the db then we shouldn't store it.
        if self.get(&nullifier).is_some() {
            return None;
        }

        // If the new nullifier is not in the db then we should update the next value of its low nullifier.
        let low_nullifier = *self.get_low_nullifier(&nullifier)?;
        let index = if let Some(index) = index {
            index
        } else {
            self.values().map(|v| v.index as u64).max().unwrap_or(1) + 1
        };
        let entry = LeafDBEntry::<F>::new(
            nullifier,
            index,
            low_nullifier.next_index,
            low_nullifier.next_value,
        );
        let val = self.insert(nullifier, entry);
        // If we somehow got to here and the value was already in the db then we should return `None`.
        if val.is_some() {
            return None;
        }
        self.update_nullifier(low_nullifier.value, F::from(index), nullifier)?;

        Some(())
    }

    fn get_nullifier(&self, value: Option<F>, next_value: Option<F>) -> Option<&LeafDBEntry<F>> {
        match (value, next_value) {
            (Some(value), None) => self.get(&value),
            (None, Some(next_value)) => self.values().find(|v| v.next_value == next_value),

            (Some(value), Some(next_value)) => self.get(&value).and_then(|v| {
                if v.next_value == next_value {
                    Some(v)
                } else {
                    None
                }
            }),
            (None, None) => None,
        }
    }

    fn get_low_nullifier(&self, nullifier_value: &F) -> Option<&LeafDBEntry<F>> {
        let low_nullifier = self
            .keys()
            .filter(|k| *k < nullifier_value)
            .max()
            .copied()?;
        let next_value = self.get(&low_nullifier)?;
        if *nullifier_value < next_value.next_value || next_value.next_value == F::zero() {
            Some(next_value)
        } else {
            None
        }
    }

    fn update_nullifier(
        &mut self,
        nullifier: F,
        new_next_index: F,
        new_next_value: F,
    ) -> Option<()> {
        self.get_mut(&nullifier)
            .map(|v| *v = LeafDBEntry::<F>::new(nullifier, v.index, new_next_index, new_next_value))
    }
}
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use self::{
        imt::{IMTCircuitInsertionInfo, IndexedMerkleTree, LeafDB, LeafDBEntry},
        timber::Timber,
    };

    use super::*;
    use ark_bn254::Fr as Fr254;
    use ark_std::{rand::Rng, UniformRand};

    // Before we get on to the tests, define a few helper functions that help set up tests.

    /// This function makes a complete Merkle tree with every node value in a Vec. This is a very
    /// naive way to handle a tree but good for testing the more sophisticated approaches above.
    /// Nodes are numbered thusly:
    ///                                0
    ///             /                                     \
    ///             1                                     2
    ///      /              \                  /                    \
    ///      3              4                  5                     6
    ///   /     \         /     \           /     \               /     \
    /// 7       8        9      10         11     12             13     14
    /// This tree has a height of 3 (4 rows).
    fn make_complete_tree<N, H>(height: u32, hasher: &H, leaves: &[N]) -> Vec<N>
    where
        H: TreeHasher<N>,
        N: PrimeField,
    {
        let n_nodes = 2_usize.pow(height + 1) - 1;
        let n_leaves = 2_usize.pow(height);
        let first_leaf_index = n_nodes - n_leaves;
        let mut nodes = vec![N::zero(); n_nodes];
        let last_leaf_index = first_leaf_index + leaves.len();
        // copy the leaves into the leaf nodes
        nodes[first_leaf_index..last_leaf_index]
            .copy_from_slice(&leaves[..(last_leaf_index - first_leaf_index)]);
        for i in (0..n_nodes - 1).step_by(2) {
            let index = n_nodes - i - 1;
            // we're hashing from the right hand side so the starting node is guaranteed to be even
            // untouched nodes are always zero.
            if nodes[index - 1] == N::zero() && nodes[index] == N::zero() {
                nodes[index / 2 - 1] = N::zero();
            } else {
                nodes[index / 2 - 1] = hasher.tree_hash(&[nodes[index - 1], nodes[index]]).unwrap();
            }
            // compute the node at the next row up
        }
        nodes
    }

    /// makes a vecotr of n leaves with random values.
    fn make_rnd_leaves<N: UniformRand>(n: usize, mut rng: impl Rng) -> Vec<N> {
        let mut leaves = vec![];
        for _i in 0..n {
            leaves.push(N::rand(&mut rng));
        }
        leaves
    }

    /// used by several tests to compute a sibling path from scratch
    fn extract_sibling_path<N: Copy>(
        nodes: &[N],
        height: u32,
        leaf_index: usize,
    ) -> Vec<PathElement<N>> {
        let mut node_index = leaf_index + 2_usize.pow(height) - 1;
        let mut test_sibling_path = vec![];
        // and directly extract the sibling path, storing it as PathElements rather than primitive values
        for i in 0..usize::try_from(height).unwrap() {
            let direction = leaf_index >> i & 1;
            if direction == 1 {
                // sibling is to our left
                let path_element = PathElement {
                    direction: Directions::HashWithThisNodeOnLeft,
                    value: nodes[node_index - 1],
                };
                test_sibling_path.push(path_element);
                node_index = node_index / 2 - 1;
            } else {
                // sibling is to our right
                let path_element = PathElement {
                    direction: Directions::HashWithThisNodeOnRight,
                    value: nodes[node_index + 1],
                };
                test_sibling_path.push(path_element);
                node_index /= 2
            }
        }
        test_sibling_path
    }

    /// computes the frontier for a tree, given all its nodes.
    /// a frontier is the vector containing the rightmost left-hand nodes with a full tree under them
    fn compute_frontier<N: Copy + Zero + PartialEq>(nodes: &[N]) -> Vec<N> {
        // work out tree height and width
        let height = ((nodes.len() + 1) / 2).ilog2();
        let width = (nodes.len() + 1) / 2;
        // now we use an algorithm to extract the rightmost left (odd) nodes
        // first, find the last full node on the bottom row
        let mut last_full: usize = 0;
        // search across the indices in the bottom row only. We panic on a completely empty tree
        for (i, node) in nodes
            .iter()
            .enumerate()
            .take(2_usize.pow(height + 1) - 1)
            .skip(2_usize.pow(height) - 1)
        {
            if *node == N::zero() {
                last_full = i + 1 - 2_usize.pow(height);
                break;
            } // last_full counts from 1 so this is correct
        }
        // by dividing by successive powers of 2 we can find the nodes we want
        let mut frontier = vec![];
        for i in 0..width {
            let pow = 2_usize.pow(i.try_into().unwrap());
            let mut node_position_in_row: usize = last_full / pow;
            if node_position_in_row == 0 {
                break;
            } // there are no taller trees to find
            if node_position_in_row % 2 == 0 {
                node_position_in_row -= 1
            } // we only want the nearest odd node
              // for the row we are on, work out the leftmost index
            let left_most_index = 2_usize.pow(height - u32::try_from(i).unwrap()) - 1;
            // then we can find the sub tree top
            let tree_top = left_most_index + node_position_in_row - 1;
            // save it
            frontier.push(nodes[tree_top]);
        }
        frontier
    }

    #[test]
    fn test_tree_from_values() {
        const HEIGHT: u32 = 12;
        const LEAVES: usize = 2132;
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        let values = &make_rnd_leaves::<Fr254>(LEAVES, &mut rng);

        let nodes = make_complete_tree(HEIGHT, &poseidon, values);

        let test_tree = Tree::<Fr254>::build_from_values(HEIGHT, values);

        let test_root = reduce_tree(&poseidon, &test_tree);
        assert_eq!(test_root, nodes[0]);
    }

    #[test]
    fn test_update_node() {
        const HEIGHT: u32 = 5;
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        for sub_tree_leaves in [2, 4, 8] {
            let leaves = usize::rand(&mut rng) % 16 + 1;

            let mut values = make_rnd_leaves::<Fr254>(leaves, &mut rng);
            let mut tree = Tree::<Fr254>::build_from_values(HEIGHT, &values);
            let new_values = &make_rnd_leaves::<Fr254>(sub_tree_leaves, &mut rng);
            let sub_tree = Tree::<Fr254>::build_from_values(sub_tree_leaves.ilog2(), new_values);
            let remaining_height = HEIGHT - sub_tree_leaves.ilog2();
            let index = (leaves - 1) / sub_tree_leaves + 1;

            let directions = index_to_directions(index, remaining_height);

            insert_node(&sub_tree, &mut tree, &directions);
            if leaves % sub_tree_leaves != 0 {
                let resize = (leaves / sub_tree_leaves + 1) * sub_tree_leaves;
                values.resize(resize, Fr254::zero());
            }
            let full_leaves = [values.to_vec(), new_values.to_vec()].concat();
            let full_tree = Tree::<Fr254>::build_from_values(HEIGHT, &full_leaves);
            let test_root = reduce_tree(&poseidon, &tree);
            let full_root = reduce_tree(&poseidon, &full_tree);
            assert_eq!(test_root, full_root);
        }
    }

    #[test]
    fn test_insert_subtree() {
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        for sub_tree_leaves in [2, 4, 8] {
            let height = u32::rand(&mut rng) % 4 + 5;

            let leaves = usize::rand(&mut rng) % 16 + 1;

            let mut timber = Timber::<Fr254, Poseidon<Fr254>>::new(poseidon, height);
            let mut values = make_rnd_leaves::<Fr254>(leaves, &mut rng);
            timber.insert_leaves(&values).unwrap();

            let new_values = &make_rnd_leaves::<Fr254>(sub_tree_leaves, &mut rng);
            let sub_tree = Tree::<Fr254>::build_from_values(sub_tree_leaves.ilog2(), new_values);
            timber.insert_subtree(sub_tree).unwrap();

            if leaves % sub_tree_leaves != 0 {
                let resize = (leaves / sub_tree_leaves + 1) * sub_tree_leaves;
                values.resize(resize, Fr254::zero());
            }
            let full_leaves = [values.to_vec(), new_values.to_vec()].concat();
            let full_tree = Tree::<Fr254>::build_from_values(height, &full_leaves);

            let full_root = reduce_tree(&poseidon, &full_tree);
            assert_eq!(timber.root, full_root);
        }
    }

    /// we're testing a test here. Don't panic, it's to check against the ultimate authority
    #[test]
    fn test_compute_frontier() {
        const HEIGHT: u32 = 4; // other tests shouldn't care if you change this but this test will
        const LEAVES: usize = 9; // other tests shouldn't care if you change this but this test will
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        let nodes = make_complete_tree(HEIGHT, &poseidon, &make_rnd_leaves(LEAVES, &mut rng));
        let frontier = compute_frontier(&nodes);
        // for this particular tree size, we know what the answer should be, from the original timber repo
        assert_eq!(frontier[0], nodes[23]);
        assert_eq!(frontier[1], nodes[9]);
        assert_eq!(frontier[2], nodes[3]);
        assert_eq!(frontier[3], nodes[1]);
        assert_eq!(frontier.len(), 4);
    }
    #[test]
    fn test_compute_root_hash_and_frontier() {
        const HEIGHT: u32 = 4;
        let mut rng = ark_std::test_rng();
        // conduct the test for all possible leaf numbers
        for leaves in 0..2_usize.pow(HEIGHT) {
            let poseidon = Poseidon::<Fr254>::new();
            let nodes = make_complete_tree(HEIGHT, &poseidon, &make_rnd_leaves(leaves, &mut rng));
            let leaf_start_index = 2_usize.pow(HEIGHT) - 1;
            let leaves = &nodes[leaf_start_index..leaf_start_index + leaves];
            let test_root = nodes[0];
            let mut timber: Timber<Fr254, Poseidon<Fr254>> = Timber::new(poseidon, HEIGHT);
            timber.insert_leaves(leaves).unwrap();
            let test_leaves = Timber::<Fr254, Poseidon<Fr254>>::get_leaves(&timber.tree);
            assert_eq!(leaves, test_leaves);
            let root = timber.root;
            assert_eq!(root, test_root);
            assert_eq!(timber.frontier, compute_frontier(&nodes));
        }
    }
    #[test]
    fn test_get_sibling_path() {
        const HEIGHT: u32 = 6;

        for leaves_no in [8usize, 16, 24, 32, 40, 48, 56].into_iter() {
            let mut rng = ark_std::test_rng();
            let poseidon = Poseidon::<Fr254>::new();
            let nodes =
                make_complete_tree(HEIGHT, &poseidon, &make_rnd_leaves(leaves_no, &mut rng));
            let leaf_start_index = 2_usize.pow(HEIGHT) - 1;
            let leaves = &nodes[leaf_start_index..leaf_start_index + leaves_no];
            // run the test with all leaves
            for leaf_index in 0..leaves_no {
                // let leaf_index = rand::thread_rng().gen_range(0..LEAVES);
                let test_sibling_path = extract_sibling_path(&nodes, HEIGHT, leaf_index);
                // first, check our test sibling path works.
                let mut hash = leaves[leaf_index];
                for path_element in test_sibling_path
                    .iter()
                    .take(usize::try_from(HEIGHT).unwrap())
                {
                    match path_element.direction {
                        Directions::HashWithThisNodeOnLeft => {
                            hash = poseidon.hash(&[path_element.value, hash]).unwrap();
                        },
                        Directions::HashWithThisNodeOnRight => {
                            hash = poseidon.hash(&[hash, path_element.value]).unwrap();
                        },
                    }
                }
                assert_eq!(hash, nodes[0]);
                let mut timber: Timber<Fr254, Poseidon<Fr254>> = Timber::new(poseidon, HEIGHT);
                timber.insert_leaves(leaves).unwrap();
                let test_leaves = Timber::<Fr254, Poseidon<Fr254>>::get_leaves(&timber.tree);
                assert_eq!(leaves, test_leaves);
                let sibling_path = timber
                    .get_sibling_path(leaves[leaf_index], leaf_index)
                    .unwrap();
                // check the sibling paths agree
                assert_eq!(sibling_path, test_sibling_path);

                // check the trees are the same
                assert_eq!(nodes[0], timber.root);
                if leaf_index == leaves_no - 1 {
                    let tree_hasher = timber.get_tree_hasher();
                    let trial_sib_path =
                        get_node_sibling_path(&timber.tree, 3, leaves_no / 8, tree_hasher);

                    let mut hash = Fr254::zero();
                    for path_element in trial_sib_path.iter() {
                        match path_element.direction {
                            Directions::HashWithThisNodeOnLeft => {
                                hash = poseidon.tree_hash(&[path_element.value, hash]).unwrap();
                            },
                            Directions::HashWithThisNodeOnRight => {
                                hash = poseidon.tree_hash(&[hash, path_element.value]).unwrap();
                            },
                        }
                    }
                    assert_eq!(hash, timber.root);
                }
            }
        }
    }

    #[test]
    fn test_insert_for_circuit_timber() {
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        const TWO: usize = 2usize;
        const FOUR: usize = 4usize;
        const EIGHT: usize = 8usize;
        for sub_tree_leaves in [TWO, FOUR, EIGHT] {
            let height = u32::rand(&mut rng) % 4 + 5;

            let leaves = usize::rand(&mut rng) % 16 + 1;

            let mut timber = Timber::<Fr254, Poseidon<Fr254>>::new(poseidon, height);
            let values = make_rnd_leaves::<Fr254>(leaves, &mut rng);
            timber.insert_leaves(&values).unwrap();

            let new_values = make_rnd_leaves::<Fr254>(sub_tree_leaves, &mut rng);
            let sub_tree = Tree::<Fr254>::build_from_values(sub_tree_leaves.ilog2(), &new_values);
            let root = reduce_tree(&poseidon, &sub_tree);
            let info = timber.insert_for_circuit(&new_values).unwrap();
            let mut proof = info.proof.clone();
            // Verifies that the subtree was empty pre-insertion
            assert!(proof.verify(&info.old_root, &poseidon).is_ok());
            proof.node_value = root;
            // Verifies that we inserted the subtree in the correct location
            assert!(proof.verify(&info.new_root, &poseidon).is_ok());
        }
    }

    #[test]
    fn test_stateless_sibling_path() {
        const HEIGHT: u32 = 4;
        const LEAVES: usize = 9;
        if LEAVES > 2_usize.pow(HEIGHT) {
            panic!("Too many leaves")
        }
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        let nodes = make_complete_tree(HEIGHT, &poseidon, &make_rnd_leaves(LEAVES, &mut rng));
        let leaf_start_index = 2_usize.pow(HEIGHT) - 1;
        let leaves = &nodes[leaf_start_index..leaf_start_index + LEAVES];
        for leaf_index in 0..LEAVES {
            let test_sibling_path = extract_sibling_path(&nodes, HEIGHT, leaf_index);
            // make an empty, immutable timber instance
            let timber: Timber<Fr254, Poseidon<Fr254>> = Timber::new(poseidon, HEIGHT);
            let sibling_path = timber.stateless_sibling_path(leaves, leaf_index).unwrap();
            assert_eq!(sibling_path, test_sibling_path);
            // no need to check that timber is unchanged because we didn't make it mutable.
        }
    }

    #[test]
    fn test_stateless_increment_sibling_path() {
        // set up a tree
        const HEIGHT: u32 = 32;
        const LEAVES: usize = 3000;
        if LEAVES > 2_usize.pow(HEIGHT - 1) {
            panic!("Too many leaves")
        } // -1 because we end up adding the leaves to the tree twice
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        let leaves = make_rnd_leaves(LEAVES, &mut rng);

        let leaves_2 = make_rnd_leaves(LEAVES, &mut rng);
        // get the sibling path for the last leaf (most sensitive to leaf addition)
        let mut timber: Timber<Fr254, Poseidon<Fr254>> = Timber::new(poseidon, HEIGHT);
        timber.insert_leaves(&leaves).unwrap();
        let sibling_path = timber.get_sibling_path(leaves[0], 0).unwrap();
        // we can use the same leaves again for the purpose of testing.
        let incremented_sibling_path = timber
            .stateless_increment_sibling_path(&leaves_2, 0, leaves[0], &sibling_path)
            .unwrap();
        // now update the tree and see if we get the same answer
        timber.insert_leaves(&leaves_2).unwrap();
        let test_incremented_sibling_path = timber.get_sibling_path(leaves[0], 0).unwrap();
        assert_eq!(incremented_sibling_path, test_incremented_sibling_path);
    }

    #[test]
    fn test_stateless_increment_sibling_paths() {
        // set up a tree
        const HEIGHT: u32 = 32;
        const LEAVES: usize = 3000;
        let mut rng = ark_std::test_rng();
        let insert_leaves = 8usize * (usize::rand(&mut rng) % 16 + 1);
        if LEAVES > 2_usize.pow(HEIGHT - 1) {
            panic!("Too many leaves")
        } // -1 because we end up adding the leaves to the tree twice

        let poseidon = Poseidon::<Fr254>::new();
        let leaves = make_rnd_leaves(LEAVES, &mut rng);

        let leaves_2 = make_rnd_leaves(insert_leaves, &mut rng);
        // get the sibling path for the last leaf (most sensitive to leaf addition)
        let mut timber: Timber<Fr254, Poseidon<Fr254>> = Timber::new(poseidon, HEIGHT);
        timber.insert_leaves(&leaves).unwrap();

        let sibling_paths = (0..10)
            .map(|_| {
                let index = usize::rand(&mut rng) % 3000usize;
                let sibling_path = timber.get_sibling_path(leaves[index], index).unwrap();
                MembershipProof {
                    node_value: leaves[index],
                    sibling_path,
                    leaf_index: index,
                }
            })
            .collect::<Vec<MembershipProof<Fr254>>>();
        ark_std::println!("got to here");
        // we can use the same leaves again for the purpose of testing.
        let incremented_sibling_paths = timber
            .stateless_increment_sibling_paths(&leaves_2, &sibling_paths, 8)
            .unwrap();
        ark_std::println!("got to here 2");
        // now update the tree and see if we get the same answer
        for chunk in leaves_2.chunks(8) {
            let subtree = Tree::<Fr254>::build_from_values(3, chunk);
            timber.insert_subtree(subtree).unwrap();
        }
        for proof in incremented_sibling_paths.iter() {
            assert!(proof.verify(&timber.root, &poseidon).is_ok());
        }
    }

    #[test]
    fn test_rollback() {
        // setup some tree data
        const HEIGHT: u32 = 4;
        const LEAVES: usize = 5;
        if LEAVES > 2_usize.pow(HEIGHT - 1) {
            panic!("Too many leaves")
        }
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        let nodes = make_complete_tree(HEIGHT, &poseidon, &make_rnd_leaves(LEAVES, &mut rng));
        let leaf_start_index = 2_usize.pow(HEIGHT) - 1;
        let leaves = &nodes[leaf_start_index..leaf_start_index + LEAVES];
        // make a tree
        let mut timber1: Timber<Fr254, Poseidon<Fr254>> = Timber::new(poseidon, HEIGHT);
        timber1.insert_leaves(leaves).unwrap();
        // make a clone
        let mut timber2: Timber<Fr254, Poseidon<Fr254>> = timber1.clone();
        // add more leaves to the clone
        timber2.insert_leaves(leaves).unwrap();
        // check it's worked
        let t1_leaves = Timber::<Fr254, Poseidon<Fr254>>::get_leaves(&timber1.tree);
        let t2_leaves = Timber::<Fr254, Poseidon<Fr254>>::get_leaves(&timber2.tree);
        assert_eq!(leaves, t1_leaves);
        assert_eq!([leaves, leaves].concat().to_vec(), t2_leaves);
        // now roll it back
        timber2.rollback(LEAVES).unwrap();
        assert_eq!(timber1, timber2);
    }

    #[test]
    fn test_stateless_update() {
        // setup some tree data
        const HEIGHT: u32 = 4;
        // do multiple tests
        for n_leaves in 1..2_usize.pow(HEIGHT - 1) {
            let mut rng = ark_std::test_rng();
            let poseidon = Poseidon::<Fr254>::new();
            let nodes = make_complete_tree(HEIGHT, &poseidon, &make_rnd_leaves(n_leaves, &mut rng));
            let leaf_start_index = 2_usize.pow(HEIGHT) - 1;
            let leaves = &nodes[leaf_start_index..leaf_start_index + n_leaves];
            let leaves2 = &make_rnd_leaves(n_leaves - 1, rng);
            let nodes2 = make_complete_tree(HEIGHT, &poseidon, &[leaves, leaves2].concat());
            // make a tree
            let mut timber1: Timber<Fr254, Poseidon<Fr254>> = Timber::new(poseidon, HEIGHT);
            timber1.insert_leaves(leaves).unwrap();
            assert_eq!(timber1.root.clone(), nodes[0]);
            assert_eq!(timber1.frontier.clone(), compute_frontier(&nodes));
            timber1.insert_leaves(leaves2).unwrap();
            assert_eq!(timber1.root.clone(), nodes2[0]);
            assert_eq!(timber1.frontier.clone(), compute_frontier(&nodes2));
            // statelessly do the same thing to a new timber and compare (we need to add some leaves first because
            // if the tree is empty stateless_update just adds the leaves normally with insert_leaves).
            let mut timber2: Timber<Fr254, Poseidon<Fr254>> = Timber::new(poseidon, HEIGHT);
            timber2.insert_leaves(leaves).unwrap();
            assert_eq!(timber2.root.clone(), nodes[0]);
            assert_eq!(timber2.frontier.clone(), compute_frontier(&nodes));
            let height = timber2.height;
            timber2.stateless_update(leaves2, height);
            assert_eq!(timber1.leaf_count, timber2.leaf_count);
            assert_eq!(timber1.frontier, timber2.frontier);
            assert_eq!(timber1.height, timber2.height);
            assert_eq!(timber1.root, timber2.root);
            assert_eq!(timber1.hasher, timber2.hasher);
            //assert_ne!(timber1.tree, timber2.tree); // the tree doesn't get updated in timber2.
        }
    }

    #[test]
    fn test_stateless_update_subtrees() {
        // setup some tree data
        const HEIGHT: u32 = 32;
        const LEAVES: usize = 3000;
        // do multiple tests
        for n_leaves in 1..10 {
            let insert_leaf_count = 8 * n_leaves;
            let mut rng = ark_std::test_rng();
            let poseidon = Poseidon::<Fr254>::new();
            let leaves = &make_rnd_leaves(LEAVES, &mut rng);

            let leaves2 = &make_rnd_leaves(insert_leaf_count, rng);
            // make a tree
            let mut timber1: Timber<Fr254, Poseidon<Fr254>> = Timber::new(poseidon, HEIGHT);
            timber1.insert_leaves(leaves).unwrap();

            for chunk in leaves2.chunks(8) {
                let subtree = Tree::<Fr254>::build_from_values(3, chunk);
                timber1.insert_subtree(subtree).unwrap();
            }

            // statelessly do the same thing to a new timber and compare (we need to add some leaves first because
            // if the tree is empty stateless_update just adds the leaves normally with insert_leaves).
            let mut timber2: Timber<Fr254, Poseidon<Fr254>> = Timber::new(poseidon, HEIGHT);
            timber2.stateless_update(leaves, 32);

            timber2.stateless_update_subtrees(leaves2, 8, 32).unwrap();
            assert_eq!(timber1.leaf_count, timber2.leaf_count);
            assert_eq!(timber1.frontier, timber2.frontier);
            assert_eq!(timber1.height, timber2.height);
            assert_eq!(timber1.root, timber2.root);
            assert_eq!(timber1.hasher, timber2.hasher);
            //assert_ne!(timber1.tree, timber2.tree); // the tree doesn't get updated in timber2.
        }
    }

    #[test]
    fn test_insert_indexed_merkle_tree() {
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        for height in 2..6 {
            let no_leaves = 2_usize.pow(height) - 1;
            let mut nullifiers = make_rnd_leaves(no_leaves, &mut rng);

            let mut final_leaves = Vec::<Fr254>::new();
            let mut nullifier_db = <HashMap<Fr254, LeafDBEntry<Fr254>> as LeafDB<Fr254>>::new();
            for nullifier in nullifiers.iter() {
                nullifier_db.store_nullifier(*nullifier, None).unwrap();
            }
            nullifiers.insert(0, Fr254::zero());
            for nullifier in nullifiers.iter() {
                let entry = nullifier_db.get_nullifier(Some(*nullifier), None).unwrap();
                let leaf_val = poseidon
                    .hash(&[entry.value, entry.next_index, entry.next_value])
                    .unwrap();
                final_leaves.push(leaf_val);
            }

            let mut imt: IndexedMerkleTree<
                Fr254,
                Poseidon<Fr254>,
                HashMap<Fr254, LeafDBEntry<Fr254>>,
            > = IndexedMerkleTree::new(poseidon, height).unwrap();

            let mut imt_2: IndexedMerkleTree<
                Fr254,
                Poseidon<Fr254>,
                HashMap<Fr254, LeafDBEntry<Fr254>>,
            > = IndexedMerkleTree::new(poseidon, height).unwrap();

            for nullifier in nullifiers.iter().skip(1) {
                imt.insert_leaf(*nullifier).unwrap();
            }

            let leaves = Timber::<Fr254, Poseidon<Fr254>>::get_leaves(&imt.timber.tree);

            for (leaf, calc_leaf) in leaves.iter().zip(final_leaves.iter()) {
                assert_eq!(leaf, calc_leaf);
            }

            imt_2.insert_leaves(&nullifiers[1..]).unwrap();

            let leaves_2 = Timber::<Fr254, Poseidon<Fr254>>::get_leaves(&imt_2.timber.tree);
            for (leaf, calc_leaf) in leaves_2.iter().zip(final_leaves.iter()) {
                assert_eq!(leaf, calc_leaf);
            }
        }
    }

    #[test]
    fn test_non_membership() {
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        for height in 2..9 {
            let no_leaves = 2_usize.pow(height) - 1;
            let mut nullifiers = make_rnd_leaves(no_leaves, &mut rng);

            let mut nullifier_db = <HashMap<Fr254, LeafDBEntry<Fr254>> as LeafDB<Fr254>>::new();
            for nullifier in nullifiers.iter() {
                nullifier_db.store_nullifier(*nullifier, None).unwrap();
            }
            nullifiers.insert(0, Fr254::zero());

            let mut imt: IndexedMerkleTree<
                Fr254,
                Poseidon<Fr254>,
                HashMap<Fr254, LeafDBEntry<Fr254>>,
            > = IndexedMerkleTree::new(poseidon, height).unwrap();

            imt.insert_leaves(&nullifiers[1..no_leaves - 1]).unwrap();
            let path = imt.non_membership_proof(nullifiers[no_leaves - 1]).unwrap();
            let low_nullifier = imt
                .leaves_db
                .get_low_nullifier(&nullifiers[no_leaves - 1])
                .unwrap();
            let mut hash = poseidon
                .hash(&[
                    low_nullifier.value,
                    low_nullifier.next_index,
                    low_nullifier.next_value,
                ])
                .unwrap();
            for path_element in path.sibling_path.iter() {
                match path_element.direction {
                    Directions::HashWithThisNodeOnLeft => {
                        hash = poseidon.hash(&[path_element.value, hash]).unwrap();
                    },
                    Directions::HashWithThisNodeOnRight => {
                        hash = poseidon.hash(&[hash, path_element.value]).unwrap();
                    },
                }
            }
            assert_eq!(hash, imt.timber.root);

            // May as well test the verify function here as well.
            assert!(imt.verify_non_membership_proof(&path).is_ok());
        }
    }

    fn create_db_entries<F: PrimeField>(
        inner_values: &[F],
        start_leaves: usize,
        subtree_leaves: usize,
    ) -> Vec<LeafDBEntry<F>> {
        // To begin we calculate the index of each leaf in the tree.
        let mut values_indices = vec![];
        for (i, inner_value) in inner_values.iter().enumerate() {
            let index = if i < start_leaves {
                i
            } else {
                (((start_leaves - 1) / subtree_leaves + 1) * subtree_leaves) + (i - start_leaves)
            };

            values_indices.push((*inner_value, index))
        }

        // Now for each element in the list we find the smallest value in the list greater than it.
        // If it is the largest value we link it to F::zero() instead.
        let mut next_values = vec![];
        for inner_value in inner_values.iter() {
            let next_value = values_indices
                .iter()
                .filter(|x| (*inner_value < x.0) && !x.0.is_zero())
                .min()
                .copied()
                .unwrap_or((F::zero(), 0));
            next_values.push(next_value);
        }

        // Now we can create the entries.
        values_indices
            .into_iter()
            .zip(next_values)
            .enumerate()
            .map(|(i, (x, y))| {
                if i == 0 || !x.0.is_zero() {
                    let next_index = F::from(y.1 as u8);
                    LeafDBEntry {
                        value: x.0,
                        index: x.1 as u64,
                        next_index,
                        next_value: y.0,
                    }
                } else {
                    LeafDBEntry::default()
                }
            })
            .collect()
    }

    #[test]
    fn test_imt_subtree_insertion() {
        // For this test we have to construct a tree with a known structure and then test it against a subtree insertion.
        // If we have a tree with height `height` we will start with a tree that is les than half full and then insert a subtree
        // of maximum size half the tree. We will then check that the tree is correct.

        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        for _ in 0..10 {
            // Pick random height in the range [5, 8].
            let height = u32::rand(&mut rng) % 4 + 5;
            // Pick the number of leaves to start with in the range [0, 2^(height - 1) - 1).
            // We will add one to this later because an IMT always starts with one leaf.
            let start_leaves = usize::rand(&mut rng) % (2usize.pow(height - 1) - 1);
            // Pick the height of the subtree in the range [1, height - 1].
            let subtree_leaves_log = u32::rand(&mut rng) % (height - 2) + 1;
            // Total number of leaves in the subtree.
            let subtree_leaves = 2usize.pow(subtree_leaves_log);

            // Make the leaves for the start of the tree and the subtree.
            let start_values = make_rnd_leaves::<Fr254>(start_leaves, &mut rng);
            // The timber instance does not start with a zero leaf so we add one here.
            let timber_start_values = [vec![Fr254::zero()], start_values.clone()].concat();
            let subtree_values = (0..subtree_leaves)
                .map(|j| {
                    if j % 3 != 0 {
                        Fr254::rand(&mut rng)
                    } else {
                        Fr254::zero()
                    }
                })
                .collect::<Vec<Fr254>>();

            // Create a list of all values in the tree.
            let all_values = [timber_start_values, subtree_values.clone()].concat();

            // Use our function to compute what the DB entries will look like after all insertions.
            let entries = create_db_entries::<Fr254>(&all_values, start_leaves + 1, subtree_leaves);

            // Calculate the hashes of the entries.
            let start_leaf_values = entries
                .iter()
                .take(start_leaves + 1)
                .enumerate()
                .map(|(i, x)| {
                    if !x.value.is_zero() || i == 0 {
                        poseidon
                            .hash(&[x.value, x.next_index, x.next_value])
                            .unwrap()
                    } else {
                        Fr254::from(0u8)
                    }
                })
                .collect::<Vec<Fr254>>();
            let subtree_leaf_values = entries
                .iter()
                .skip(start_leaves + 1)
                .map(|x| {
                    if !x.value.is_zero() {
                        poseidon
                            .hash(&[x.value, x.next_index, x.next_value])
                            .unwrap()
                    } else {
                        Fr254::from(0u8)
                    }
                })
                .collect::<Vec<Fr254>>();

            // Start a new timebr instance and insert the leaves and the subtree.
            let mut timber = Timber::<Fr254, Poseidon<Fr254>>::new(poseidon, height);
            timber.insert_leaves(&start_leaf_values).unwrap();

            let subtree =
                Tree::<Fr254>::build_from_values(subtree_leaves_log, &subtree_leaf_values);
            timber.insert_subtree(subtree.clone()).unwrap();

            let mut imt = IndexedMerkleTree::<
                Fr254,
                Poseidon<Fr254>,
                HashMap<Fr254, LeafDBEntry<Fr254>>,
            >::new(poseidon, height)
            .unwrap();
            imt.insert_leaves(&start_values).unwrap();

            imt.insert_subtree(&subtree_values).unwrap();
            ark_std::println!("start_lenght: {}", start_leaves);
            ark_std::println!("subtree leaves: {}", subtree_leaves);
            for (i, (value, calc_entry_hash)) in all_values
                .iter()
                .zip(start_leaf_values.iter().chain(subtree_leaf_values.iter()))
                .enumerate()
            {
                ark_std::println!("i: {}", i);
                if !value.is_zero() || i.is_zero() {
                    let entry = imt.leaves_db.get_nullifier(Some(*value), None).unwrap();
                    let hash = poseidon
                        .hash(&[entry.value, entry.next_index, entry.next_value])
                        .unwrap();
                    assert_eq!(hash, *calc_entry_hash);
                    if i < start_leaves + 1 {
                        assert_eq!(i, entry.index as usize);
                    } else {
                        let j = (((start_leaves) / subtree_leaves + 1) * subtree_leaves)
                            + (i - start_leaves - 1);
                        assert_eq!(j, entry.index as usize);
                    }
                } else {
                    let j = if i < start_leaves + 1 {
                        i
                    } else {
                        (((start_leaves) / subtree_leaves + 1) * subtree_leaves)
                            + (i - start_leaves - 1)
                    };
                    let imt_sibling_path =
                        imt.timber.get_sibling_path(Fr254::from(0u8), j).unwrap();
                    let imt_m_proof = MembershipProof {
                        node_value: Fr254::from(0u8),
                        sibling_path: imt_sibling_path,
                        leaf_index: j,
                    };
                    assert!(imt_m_proof.verify(&imt.timber.root, &poseidon).is_ok());

                    let timber_sibling_path = timber.get_sibling_path(Fr254::from(0u8), j).unwrap();
                    let timber_m_proof = MembershipProof {
                        node_value: Fr254::from(0u8),
                        sibling_path: timber_sibling_path,
                        leaf_index: j,
                    };
                    assert!(timber_m_proof.verify(&timber.root, &poseidon).is_ok());
                }
            }

            assert_eq!(timber.root, imt.timber.root);
        }
    }

    #[test]
    fn test_insert_for_circuit() {
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        for _ in 0..10 {
            // Pick random height in the range [5, 8].
            let height = u32::rand(&mut rng) % 4 + 5;
            // Pick the number of leaves to start with in the range [0, 2^(height - 1) - 1).
            // We will add one to this later because an IMT always starts with one leaf.
            let start_leaves = usize::rand(&mut rng) % (2usize.pow(height - 1) - 1);
            // Pick the height of the subtree in the range [1, height - 1].
            let subtree_leaves_log = u32::rand(&mut rng) % (height - 2) + 1;
            // Total number of leaves in the subtree.
            let subtree_leaves = 2usize.pow(subtree_leaves_log);

            // Make the leaves for the start of the tree and the subtree.
            let start_values = make_rnd_leaves::<Fr254>(start_leaves, &mut rng);

            let subtree_values = make_rnd_leaves::<Fr254>(subtree_leaves, &mut rng);
            // Do a regular insert
            let mut imt_1 = IndexedMerkleTree::<
                Fr254,
                Poseidon<Fr254>,
                HashMap<Fr254, LeafDBEntry<Fr254>>,
            >::new(poseidon, height)
            .unwrap();
            imt_1.insert_leaves(&start_values).unwrap();
            imt_1.insert_subtree(&subtree_values).unwrap();
            // insert with the info
            let mut imt_2 = IndexedMerkleTree::<
                Fr254,
                Poseidon<Fr254>,
                HashMap<Fr254, LeafDBEntry<Fr254>>,
            >::new(poseidon, height)
            .unwrap();
            imt_2.insert_leaves(&start_values).unwrap();
            let info = imt_2.insert_for_circuit(&subtree_values).unwrap();

            assert_eq!(imt_1.timber.root, imt_2.timber.root);

            for nullifier in info.pending_inserts.iter() {
                let optional = imt_1.leaves_db.get_nullifier(Some(nullifier.value), None);
                assert!(optional.is_some());
                let entry = optional.unwrap();
                assert_eq!(entry.value, nullifier.value);
                assert_eq!(entry.next_index, nullifier.next_index);
                assert_eq!(entry.next_value, nullifier.next_value);
                assert_eq!(entry.index, nullifier.index);
            }
        }
    }

    #[test]
    fn test_vec_conversion() {
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        for _ in 0..10 {
            // Pick random height in the range [5, 8].
            let height = 32;
            // Pick the number of leaves to start with in the range [0, 2^(height - 1) - 1).
            // We will add one to this later because an IMT always starts with one leaf.
            let start_leaves = usize::rand(&mut rng) % 24;
            // Pick the height of the subtree in the range [1, height - 1].
            let subtree_leaves_log = 3;
            // Total number of leaves in the subtree.
            let subtree_leaves = 2usize.pow(subtree_leaves_log);

            // Make the leaves for the start of the tree and the subtree.
            let start_values = make_rnd_leaves::<Fr254>(start_leaves, &mut rng);

            let subtree_values = make_rnd_leaves::<Fr254>(subtree_leaves, &mut rng);

            // insert with the info
            let mut imt_2 = IndexedMerkleTree::<
                Fr254,
                Poseidon<Fr254>,
                HashMap<Fr254, LeafDBEntry<Fr254>>,
            >::new(poseidon, height)
            .unwrap();
            imt_2.insert_leaves(&start_values).unwrap();
            let info = imt_2.insert_for_circuit(&subtree_values).unwrap();

            let vec_info = Vec::<Fr254>::from(info.clone());

            let reconstructed_info = IMTCircuitInsertionInfo::<Fr254>::from_slice(
                &vec_info,
                height as usize,
                subtree_values.len(),
            )
            .unwrap();

            assert_eq!(info.old_root, reconstructed_info.old_root);
        }
    }
}
