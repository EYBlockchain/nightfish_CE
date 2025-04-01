//! This module contains code for timber trees. Timber trees are a variant of Merkle trees
//! that aim to store the minimal amount of data required to prove the inclusion of a leaf in the tree.

use super::{
    _check_membership, _insert_leaf, _safe_insert_leaf, batch_leaves, combine_frontiers,
    error::TimberError, get_node_sibling_path, get_tree_hasher, index_to_directions, insert_node,
    insert_sibling_path, reduce_tree, CircuitInsertionInfo, Directions, MembershipProof,
    PathElement, Tree, TreeHasher,
};
use ark_ff::PrimeField;
use ark_std::{boxed::Box, cfg_iter, vec, vec::Vec, Zero};
use rayon::prelude::*;

/// struct representing the fields of a Timber class which supports
/// construction and updating of a specific Merkle tree
#[derive(Clone, Debug, PartialEq)]
pub struct Timber<N, H> {
    /// The root of the tree
    pub root: N,
    /// The tree itself
    pub tree: Tree<N>,
    /// The frontier of the tree
    pub frontier: Vec<N>,
    /// The current number of leaves in the tree
    pub leaf_count: usize,
    /// The height of the tree
    pub height: u32,
    /// The maximum number of leaves in the tree
    pub width: usize,
    /// The hasher used to hash the nodes
    pub hasher: H,
}

impl<N, H> Timber<N, H>
where
    H: TreeHasher<N> + Clone + PartialEq + Send + Sync,
    N: Zero + PrimeField,
{
    /// This is the constructor for the Timber class
    pub fn new(hasher: H, height: u32) -> Self {
        let width = 2_usize.pow(height);
        Self {
            root: N::zero(),
            tree: Tree::Leaf(N::zero()),
            frontier: vec![],
            leaf_count: 0,
            height,
            width,
            hasher,
        }
    }

    /// This associated function helpfully traverses the tree to apply the function to all leaves.
    /// f - A function that operates on Leaf values
    /// tree - The tree to be traversed
    /// returns the tree with f applied to all leaves
    pub fn map_tree<M>(f: &M, tree: Tree<N>) -> Tree<N>
    where
        M: Fn(N) -> N,
    {
        match tree {
            Tree::Branch(sub_trees) => Tree::Branch(Box::new((
                Self::map_tree(&f, sub_trees.0),
                Self::map_tree(&f, sub_trees.1),
            ))),
            Tree::Leaf(value) => Tree::Leaf(f(value)),
        }
    }

    /// This associated function is like mapTree except the values are accumulated to the root
    /// f - A binary (as in two parameters) function that operates on Leaf values
    /// tree - The tree to be traversed
    /// returns the result of the accumulations as due to f
    pub fn reduce_tree(&self) -> N {
        reduce_tree(&self.hasher, &self.tree)
    }

    /// This associate function returns a closure that will hash the tree
    /// tree - The tree that will be hashed
    /// returns the hash result;
    pub fn get_tree_hasher(&self) -> impl Fn(&Tree<N>) -> N + '_ {
        |tree| reduce_tree(&self.hasher.clone(), tree)
    }

    /// This  associated function moves our "focus" from the current node down to a subtree
    /// tree - The tree where our focus is currently at the root of
    /// directions - The Directions that decide if we go left or right.
    /// returns the subtree where our focus is currently at the root of.
    fn move_down(tree: &Tree<N>, directions: &[Directions]) -> Tree<N> {
        if directions.is_empty() {
            return tree.clone();
        }
        match tree {
            Tree::Branch(sub_trees) => match directions[0] {
                Directions::HashWithThisNodeOnLeft => {
                    Self::move_down(&sub_trees.0, &directions[1..])
                },
                Directions::HashWithThisNodeOnRight => {
                    Self::move_down(&sub_trees.1, &directions[1..])
                },
            },
            Tree::Leaf(_) => tree.clone(),
        }
    }

    /// Traverse the tree to find the leaf value at the given index and replace it with the new leaf value.
    pub fn update_leaf(
        tree: &mut Tree<N>,
        directions: &[Directions],
        new_leaf_value: N,
    ) -> Result<(), TimberError> {
        if directions.is_empty() {
            match tree {
                Tree::Branch(_) => {
                    return Err(TimberError::WrongNodeType);
                },
                Tree::Leaf(value) => {
                    *value = new_leaf_value;
                    return Ok(());
                },
            }
        }
        match tree {
            Tree::Branch(sub_trees) => {
                if let Directions::HashWithThisNodeOnLeft = directions[0] {
                    Self::update_leaf(&mut sub_trees.0, &directions[1..], new_leaf_value)?;
                } else {
                    Self::update_leaf(&mut sub_trees.1, &directions[1..], new_leaf_value)?;
                }
            },

            Tree::Leaf(_) => {
                return Err(TimberError::WrongNodeType);
            },
        }
        Ok(())
    }

    /// Associated function to extract the leaves from a tree. This isn't in NF_3's equivalent code but we
    /// can't easily subvert 'reduce_tree' to do this as per NF_3 because of the strong typing
    pub fn get_leaves(tree: &Tree<N>) -> Vec<N> {
        fn recurse<N: Copy + Zero + PartialEq>(tree: &Tree<N>, leaves: &mut Vec<N>) -> Tree<N> {
            match tree {
                Tree::Branch(sub_trees) => Tree::Branch(Box::new((
                    recurse(&sub_trees.0, leaves),
                    recurse(&sub_trees.1, leaves),
                ))),
                Tree::Leaf(value) => {
                    if *value != N::zero() {
                        leaves.push(*value)
                    };
                    Tree::Leaf(*value)
                },
            }
        }
        let mut leaves = vec![];
        recurse(tree, &mut leaves);
        leaves
    }

    /// This method gets the path from a leaf index to the root.
    /// leaf_value the value of the leaf of interest.
    /// index - The index that the leaf of interest is located at.
    /// returns the sibling path for leafValue.
    /// If a leaf_index and a value are sent, the leaf index is used.
    pub fn get_sibling_path(&self, leaf_value: N, index: usize) -> Option<Vec<PathElement<N>>> {
        if self.leaf_count == 0 {
            return None;
        }
        let tree_hasher = self.get_tree_hasher();
        let path = index_to_directions(index, self.height); // should only fail on a < 32 machine.
        _check_membership(leaf_value, &self.tree, &path, tree_hasher, &mut vec![])
    }

    /// This function finds the next available empty subtree of the given `size`, here `size` is the number of leaves the subtree has,
    /// in the tree and returns the leaf index
    /// of the first leaf in that subtree.
    pub fn find_subtree_insertion_path(&self, subtree: &Tree<N>) -> Option<Vec<PathElement<N>>> {
        let subtree_height = subtree.height();
        let subtree_leaf_count = subtree.leaf_count();
        // Check we can fit the subtree in the tree
        if subtree_height > self.height {
            return None;
        }

        if subtree_leaf_count + self.leaf_count > self.width {
            return None;
        }
        // This is calculated using the fact that (self.leaf_count / subtree_leaf_count + 1) * subtree_leaf_count would give the index
        // of the first leaf of the new subtree after insertion. Then we divide by 2^subtree_height = subtree_leaf_count to get the index of the root of the subtree.
        let node_index = self.leaf_count / subtree_leaf_count + 1;

        let tree_hasher = self.get_tree_hasher();

        Some(get_node_sibling_path(
            &self.tree,
            self.height - subtree_height,
            node_index,
            &tree_hasher,
        ))
    }

    /// associated function gets the leaves of a tree and then
    /// finds the index of the leaf with the given value.
    pub fn index_from_leaf_value(tree: &Tree<N>, leaf_value: N) -> Option<usize> {
        let leaves = Self::get_leaves(tree);
        leaves.iter().position(|v| *v == leaf_value)
    }

    /// associated function to extract the leaf value if we know the index of the leaf.
    /// The index is always numbered from zero being the leftmost leaf in the tree.
    pub fn leaf_value_from_index(tree: &Tree<N>, index: usize) -> Option<N> {
        let leaves = Self::get_leaves(tree);
        if leaves.len() < index + 1 {
            return None;
        }
        Some(leaves[index])
    }

    /// This associated function verifies a sibling path for a given leafValue
    /// leafValue - The leafValue to get the path of.
    /// root - The root that the merkle proof should verify to.
    /// proofPath - The output from getSiblingPath.
    /// returns boolean if a path is ok or not.
    pub fn verify_sibling_path(
        leaf_value: N,
        root: N,
        sibling_path: Vec<PathElement<N>>,
        hasher: H,
    ) -> bool {
        if sibling_path.is_empty() {
            return false;
        };
        // we're stuck if the hashing fails, so throw in that case
        let calc_root = sibling_path
            .iter()
            .fold(leaf_value, |acc, el| match el.direction {
                Directions::HashWithThisNodeOnLeft => hasher.tree_hash(&[el.value, acc]).unwrap(),
                Directions::HashWithThisNodeOnRight => hasher.tree_hash(&[acc, el.value]).unwrap(),
            });
        calc_root == root
    }

    /// pads a vector with in a manner similar to the node .padStart command
    /// although we're dealing with a vector, not a string
    fn pad_vector<T: Clone>(initial: &[T], padder: T, length: usize) -> Vec<T> {
        if length < initial.len() {
            panic!("Vector is too long to pad");
        }
        let pads = length - initial.len();
        if pads == 0 {
            return initial.to_vec();
        }
        let mut padding = vec![padder; pads];
        padding.append(&mut initial.to_vec());
        padding
    }

    // This calculates the frontier for a tree
    // tree - The tree where our focus is currently at the root of
    // leafCount - leafCount of the tree
    // height - The height of the current tree, defaults to the TIMBER_HEIGHT
    // returns the frontier for the given tree
    fn calc_frontier(tree: &Tree<N>, leaf_count: usize, height: u32, hasher: &H) -> Vec<N> {
        if leaf_count == 0 {
            return vec![];
        }
        let usize_ht = usize::try_from(height).unwrap();
        // If there are some leaves in this tree, we can make our life easier
        // by locating "full" subtrees. We do this by using the height and leaf count
        let width = if leaf_count > 1 {
            2_usize.pow(height)
        } else {
            2
        };
        let num_frontier_points = usize::try_from(leaf_count.ilog2() + 1).unwrap();
        let tree_hasher = get_tree_hasher(hasher);
        // If this tree is full, we have a deterministic way to discover the frontier values
        if leaf_count == width {
            let mut dirs: Vec<Vec<Directions>> = vec![];
            // Dirs is an array of directions: ['0','10','110'...] or [Left, [Left, Right], [Left, Right, Right]...] (direction is reversed)
            for i in 0..num_frontier_points - 1 {
                let dir = Self::pad_vector(
                    &[Directions::HashWithThisNodeOnLeft],
                    Directions::HashWithThisNodeOnRight,
                    i + 1,
                );
                dirs.push(dir);
            }
            // The frontier points are then the root of the tree and our deterministic paths.
            let mut frontier = vec![tree_hasher(tree)];
            frontier.append(
                &mut dirs
                    .iter()
                    .map(|fp| tree_hasher(&Self::move_down(tree, fp)))
                    .collect::<Vec<N>>(),
            );
            return frontier;
        }
        // Our tree is not full at this height, but there will be a level where it will be full
        // unless there is only 1 leaf in this tree (which we will handle separately)

        // Firstly, we need to descend to a point where we are sitting over the subtree that forms
        // the frontier points.
        let last_index = leaf_count - 1;
        let directions = index_to_directions(last_index, height);
        let frontier_tree_root = Self::move_down(
            tree,
            &directions[0..usize::try_from(height).unwrap() - num_frontier_points],
        );
        // If the leaf count is 1 then our only option is left of the current location.
        if leaf_count == 1 {
            match Self::move_down(
                &frontier_tree_root,
                vec![Directions::HashWithThisNodeOnLeft; usize_ht].as_slice(),
            ) {
                Tree::Branch(_) => panic!(
                    "There was a branch when only a leaf should exist. This should never happen."
                ),
                Tree::Leaf(value) => {
                    return vec![value];
                },
            }
        }
        let left_leaf_count = 2_usize.pow(leaf_count.ilog2());
        let right_leaf_count = leaf_count - left_leaf_count;
        let mut left_sub_tree_dirs: Vec<Vec<Directions>> = vec![];
        for i in 0..num_frontier_points - 1 {
            let dir = Self::pad_vector(
                &[Directions::HashWithThisNodeOnLeft],
                Directions::HashWithThisNodeOnRight,
                i + 1,
            );
            left_sub_tree_dirs.push(dir);
        }
        match frontier_tree_root {
            Tree::Branch(sub_trees) => {
                let mut left_tree_frontier_points = vec![tree_hasher(&sub_trees.0)];
                left_tree_frontier_points.append(
                    &mut left_sub_tree_dirs
                        .iter()
                        .map(|fp| tree_hasher(&Self::move_down(&sub_trees.0, fp)))
                        .collect::<Vec<N>>(),
                );
                left_tree_frontier_points.reverse();
                let new_height = u32::try_from(num_frontier_points - 1).unwrap();
                combine_frontiers(
                    &left_tree_frontier_points,
                    &Self::calc_frontier(&sub_trees.1, right_leaf_count, new_height, hasher),
                )
            },
            Tree::Leaf(_) => panic!("This tree node should not be a leaf"),
        }
    }

    /// Inserts a single leaf into the tree
    /// leafValue - The commitment that will be inserted.
    /// returns updated timber instance.
    pub fn insert_leaf(&mut self, leaf_value: N) -> Result<(), TimberError> {
        if self.leaf_count == self.width {
            return Err(TimberError::TreeIsFull);
        }
        // New Leaf will be added at index leafCount - the leafCount is always one more than the index.
        let path_to_next_index = index_to_directions(self.leaf_count, self.height);
        self.tree = _insert_leaf(leaf_value, &self.tree, &path_to_next_index);
        self.leaf_count += 1;

        self.root = self.reduce_tree();
        self.frontier = Self::calc_frontier(&self.tree, self.leaf_count, self.height, &self.hasher);
        Ok(())
    }

    /// Inserts multiple  leaves into the tree
    /// leafValues - The commitments that will be inserted.
    /// returns updated timber instance.
    pub fn insert_leaves(&mut self, leaf_values: &[N]) -> Result<(), TimberError> {
        if leaf_values.is_empty() {
            return Ok(());
        }
        if self.leaf_count + leaf_values.len() > self.width {
            return Err(TimberError::TreeIsFull);
        }
        for (i, leaf_value) in leaf_values.iter().enumerate() {
            let node_index = index_to_directions(self.leaf_count + i, self.height);
            self.tree = _insert_leaf(*leaf_value, &self.tree, &node_index);
        }
        self.leaf_count += leaf_values.len();

        self.root = self.reduce_tree();
        self.frontier = Self::calc_frontier(&self.tree, self.leaf_count, self.height, &self.hasher);
        Ok(())
    }

    /// Inserts a subtree into the tree.
    pub fn insert_subtree(&mut self, subtree: Tree<N>) -> Result<(), TimberError> {
        let subtree_leaf_count = subtree.leaf_count();
        if self.leaf_count + subtree_leaf_count > self.width {
            return Err(TimberError::TreeIsFull);
        }

        // Here we calculate the difference in height of our main tree and our subtree
        let remaining_height = self.height - subtree.height();
        // This is the index of the root of the subtree in the main tree, counted from the left, starting at zero
        let index = if self.leaf_count != 0 {
            (self.leaf_count - 1) / subtree_leaf_count + 1
        } else {
            0
        };

        let directions = index_to_directions(index, remaining_height);

        insert_node(&subtree, &mut self.tree, &directions).ok_or(TimberError::TreeIsFull)?;
        self.root = self.reduce_tree();
        self.leaf_count = (index + 1) * subtree_leaf_count;
        self.frontier = Self::calc_frontier(&self.tree, self.leaf_count, self.height, &self.hasher);
        Ok(())
    }

    // This helpfully deletes the right subtree along a given path.
    // tree - The tree that deletion will be performed over.
    // pathToLeaf - The path along which every right subtree will be deleted.
    // returns a tree after deletion
    fn prune_right_sub_tree(tree: &Tree<N>, path_to_leaf: &[Directions]) -> Tree<N> {
        match &tree {
            Tree::Branch(sub_trees) => {
                match path_to_leaf[0] {
                    Directions::HashWithThisNodeOnLeft => Tree::Branch(Box::new((
                        Self::prune_right_sub_tree(&sub_trees.0, &path_to_leaf[1..]),
                        Tree::Leaf(N::zero()),
                    ))), // Going left, delete the right tree
                    Directions::HashWithThisNodeOnRight => Tree::Branch(Box::new((
                        sub_trees.0.clone(),
                        Self::prune_right_sub_tree(&sub_trees.1, &path_to_leaf[1..]),
                    ))), // Going right, but leave the left tree intact
                }
            },
            Tree::Leaf(_) => tree.clone(),
        }
    }

    /// This function updates the root, frontier and leafCount of a given timber instance based on incoming new leaves
    /// It does not update the tree - hence 'stateless' (although obvs it's not completely stateless),
    /// this is useful if we don't store the tree.
    /// leaves - The incoming new leaves
    /// returns a timber instance where everything but the tree is updated.
    /// NB: stateless_update assumes no two leaves are the same. It's root calculation may fail if this is not the case
    pub fn stateless_update(&mut self, leaves: &[N], height: u32) {
        if leaves.is_empty() {
            return;
        }
        // If the timber tree is empty, it's much simpler insert the leaves anyways.
        if self.leaf_count == 0 {
            self.insert_leaves(leaves).unwrap();
            return;
        }
        // Since we cannot rely on timber.tree, we have to work out how "full" the trees are
        // at each level using only their respective leaf counts.
        let output_leaf_count = self.leaf_count + leaves.len();
        let output_frontier_length = output_leaf_count.ilog2() + 1;
        // This is an array that counts the number of perfect trees at each depth for the final frontier.
        // E.g timber.leafCount = 8 --> [8, 4 , 2, 1]
        let resulting_frontier_slot =
            vec![output_leaf_count; output_frontier_length.try_into().unwrap()]
                .iter()
                .enumerate()
                .map(|(i, a)| a / 2_usize.pow(i.try_into().unwrap()))
                .collect::<Vec<usize>>();
        // This is an array that counts the number of perfect trees at each depth for the current frontier.
        // This is padded to be as long as the resultingFrontierSlot.
        let current_frontier_slot_array =
            vec![self.leaf_count; output_frontier_length.try_into().unwrap()]
                .iter()
                .enumerate()
                .map(|(i, a)| a / 2_usize.pow(i.try_into().unwrap()))
                .collect::<Vec<usize>>();
        // This is the array for the subtree frontier positions should be
        // this is calculated from the final and current frontier.
        // We back-calculate this as it helps work out the intermediate frontier
        let sub_tree_frontier_slot_array = resulting_frontier_slot
            .iter()
            .enumerate()
            .map(|(i, a)| a - current_frontier_slot_array[i])
            .collect::<Vec<usize>>();
        // The height of the subtree that would be created by the new leaves
        // log2.ceil for an unsigned integer 'a' is u32::BITS - (a-1).leading_zeros(); or whatever u type is of interest
        let sub_tree_height = usize::BITS - (leaves.len() - 1).leading_zeros();
        // Since we are trying to add in batches, we have to be sure that the
        // new tree created from the incoming leaves are added correctly to the existing tree
        // We need to work out if adding the subtree directly would impact the balance of the tree.
        // We achieve this by identifying if the perfect tree count at the height of the incoming tree, contains any odd counts
        let odd_depths = current_frontier_slot_array[0..sub_tree_height.try_into().unwrap()]
            .iter()
            .map(|a| a % 2 != 0)
            .collect::<Vec<bool>>();
        // If there are odd counts, we fix the lowest one first
        let odd_index = odd_depths.iter().position(|a| *a);
        if let Some(o_i) = odd_index {
            // We can "round a tree out" (i.e. make it perfect) by inserting 2^depth leaves from the incoming set first.
            let leaves_to_slice = 2_usize.pow(o_i.try_into().unwrap());
            let new_leaves = &leaves[leaves_to_slice..];
            self.stateless_update(&leaves[0..leaves_to_slice], height); // Update our frontier
            self.stateless_update(new_leaves, height);
            return;
        }
        // If we get to this point, then we are inserting our leaves into an existing balanced tree
        // This is ideal as it means we can batch insert out incoming leaves by making it into a mini-tree

        // This is the subtree consisting of the new leaves.
        let mut new_sub_tree = Timber::new(self.hasher.clone(), height);
        new_sub_tree.insert_leaves(leaves).unwrap(); // we can safely unwrap here because we know the leaves will fit
        let mut padded_sub_tree_frontier = new_sub_tree.frontier.clone();
        // Now we check if the calculated slots for the subtree frontier match the frontier we have calculated
        // as we may have increased the height of our tree.
        if new_sub_tree.frontier.len()
            < sub_tree_frontier_slot_array
                .iter()
                .filter(|&f| *f != 0)
                .count()
        {
            for i in new_sub_tree.frontier.len()..sub_tree_frontier_slot_array.len() {
                padded_sub_tree_frontier.push(
                    self.hasher
                        .tree_hash(&[self.frontier[i - 1], padded_sub_tree_frontier[i - 1]])
                        .unwrap(),
                );
            }
        }
        // Now we can calculate the updated frontiers based on all our information.
        let mut final_frontier: Vec<N> = vec![];
        for i in 0..resulting_frontier_slot.len() {
            let current_frontier_slot = current_frontier_slot_array[i];
            let sub_tree_frontier_slot = sub_tree_frontier_slot_array[i];
            // The rules for deciding if we should pick the existing frontier or override it with
            // the frontier from the newly created subtree.
            // 1) If either perfect tree counts at a depth are zero, we select the non-zero (our previous padding guarantees at least one is non-zero)
            // 2) If the perfect tree count for the existing tree is odd at a given depth, we select the frontier from the existing tree.
            //    This is because we know the perfect tree count for the incoming tree will be 1 (if it was > 1 we would need to do a small batch insert)
            // 3) If the perfect tree count for the existing tree is even at a given depth, we select the frontier from the incoming tree
            //    This is because the incoming tree will add to the perfect tree count and move the frontier to the right.
            if current_frontier_slot == 0 {
                final_frontier.push(padded_sub_tree_frontier[i]);
            } else if sub_tree_frontier_slot == 0 || current_frontier_slot % 2 != 0 {
                final_frontier.push(self.frontier[i]);
            }
            // ^-- This is safe because we ensure that subTreeFrontierSlot at this point can only be 1
            else {
                final_frontier.push(padded_sub_tree_frontier[i]);
            }
        }
        // Let's calculate the updated root now.
        let rightmost_element_sub_tree = index_to_directions(new_sub_tree.leaf_count - 1, height);
        // This is the height of our sub tree.
        // log2.ceil for an unsigned integer 'a' is u32::BITS - (a-1).leading_zeros(); or whatever u type is of interest
        let tree_height = usize::BITS - (new_sub_tree.leaf_count - 1).leading_zeros();
        // We can shortcut the root hash process by hashing our subtree.
        let mut root = reduce_tree(
            &self.hasher,
            &Self::move_down(
                &new_sub_tree.tree,
                &rightmost_element_sub_tree[0..(height - tree_height).try_into().unwrap()],
            ),
        );
        // Now we update the root hash by moving up to the height of the existing timber tree
        // If the root matches the timber frontier at that height - we hash it with zero
        // Otherwise we hash the frontier with our current root.
        for (i, slot) in current_frontier_slot_array
            .iter()
            .enumerate()
            .take(self.frontier.len())
            .skip(usize::try_from(tree_height).unwrap())
        {
            // We do this zero check because of past padding
            if slot % 2 == 0 || root == self.frontier[i] {
                root = self.hasher.tree_hash(&[root, N::zero()]).unwrap();
            } else {
                root = self.hasher.tree_hash(&[self.frontier[i], root]).unwrap();
            }
        }
        // From the last frontier of the existing tree, we hash up to the full height with zeroes
        for _j in self.frontier.len()..height.try_into().unwrap() {
            root = self.hasher.tree_hash(&[root, N::zero()]).unwrap();
        }
        self.root = root;
        self.frontier = final_frontier;
        self.leaf_count += leaves.len();
    }

    /// This function updates the root, frontier and leafCount of a given timber instance based on incoming new leaves
    /// It does not update the tree - hence 'stateless' (although obvs it's not completely stateless),
    /// this is useful if we don't store the tree. In contrast to [`Timber::stateless_update()`], this function inserts the leaves
    /// in subtrees of fixed size.
    /// leaves - The incoming new leaves
    /// returns a timber instance where everything but the tree is updated.
    /// NB: stateless_update assumes no two leaves are the same. It's root calculation may fail if this is not the case
    pub fn stateless_update_subtrees(
        &mut self,
        leaves: &[N],
        insertion_size: usize,
        height: u32,
    ) -> Result<(), TimberError> {
        if leaves.is_empty() {
            return Ok(());
        }

        if leaves.len() % insertion_size != 0 {
            return Err(TimberError::InvalidBatchSize);
        }
        // Since we cannot rely on timber.tree, we have to work out how "full" the trees are
        // at each level using only their respective leaf counts.
        let blank_leaf_count = if self.leaf_count % insertion_size == 0 {
            0
        } else {
            insertion_size - self.leaf_count % insertion_size
        };
        let output_leaf_count = self.leaf_count + blank_leaf_count + leaves.len();
        let output_frontier_length = output_leaf_count.ilog2() + 1;
        // This is an array that counts the number of perfect trees at each depth for the final frontier.
        // E.g timber.leafCount = 8 --> [8, 4 , 2, 1]
        let resulting_frontier_slot =
            vec![output_leaf_count; output_frontier_length.try_into().unwrap()]
                .iter()
                .enumerate()
                .map(|(i, a)| a / 2_usize.pow(i.try_into().unwrap()))
                .collect::<Vec<usize>>();
        // This is an array that counts the number of perfect trees at each depth for the current frontier.
        // This is padded to be as long as the resultingFrontierSlot.
        let current_frontier_slot_array =
            vec![self.leaf_count; output_frontier_length.try_into().unwrap()]
                .iter()
                .enumerate()
                .map(|(i, a)| a / 2_usize.pow(i.try_into().unwrap()))
                .collect::<Vec<usize>>();
        // This is the array for the subtree frontier positions should be
        // this is calculated from the final and current frontier.
        // We back-calculate this as it helps work out the intermediate frontier
        let sub_tree_frontier_slot_array = resulting_frontier_slot
            .iter()
            .enumerate()
            .map(|(i, a)| a - current_frontier_slot_array[i])
            .collect::<Vec<usize>>();
        // The height of the subtree that would be created by the new leaves
        // log2.ceil for an unsigned integer 'a' is u32::BITS - (a-1).leading_zeros(); or whatever u type is of interest
        let sub_tree_height = usize::BITS - (leaves.len() - 1).leading_zeros();
        // Since we are trying to add in batches, we have to be sure that the
        // new tree created from the incoming leaves are added correctly to the existing tree
        // We need to work out if adding the subtree directly would impact the balance of the tree.
        // We achieve this by identifying if the perfect tree count at the height of the incoming tree, contains any odd counts
        let odd_depths = current_frontier_slot_array[0..sub_tree_height.try_into().unwrap()]
            .iter()
            .map(|a| a % 2 != 0)
            .collect::<Vec<bool>>();
        // If there are odd counts, we fix the lowest one first
        let odd_index = odd_depths.iter().position(|a| *a);
        if let Some(o_i) = odd_index {
            // We can "round a tree out" (i.e. make it perfect) by inserting 2^depth leaves from the incoming set first.
            let leaves_to_slice = 2_usize.pow(o_i.try_into().unwrap());
            let new_leaves = &leaves[leaves_to_slice..];
            self.stateless_update(&leaves[0..leaves_to_slice], height); // Update our frontier
            self.stateless_update(new_leaves, height);
            return Ok(());
        }
        // If we get to this point, then we are inserting our leaves into an existing balanced tree
        // This is ideal as it means we can batch insert out incoming leaves by making it into a mini-tree

        // This is the subtree consisting of the new leaves.
        let mut new_sub_tree = Timber::new(self.hasher.clone(), height);
        new_sub_tree.insert_leaves(leaves).unwrap(); // we can safely unwrap here because we know the leaves will fit
        let mut padded_sub_tree_frontier = new_sub_tree.frontier.clone();
        // Now we check if the calculated slots for the subtree frontier match the frontier we have calculated
        // as we may have increased the height of our tree.
        if new_sub_tree.frontier.len()
            < sub_tree_frontier_slot_array
                .iter()
                .filter(|&f| *f != 0)
                .count()
        {
            for i in new_sub_tree.frontier.len()..sub_tree_frontier_slot_array.len() {
                padded_sub_tree_frontier.push(
                    self.hasher
                        .tree_hash(&[self.frontier[i - 1], padded_sub_tree_frontier[i - 1]])
                        .unwrap(),
                );
            }
        }
        // Now we can calculate the updated frontiers based on all our information.
        let mut final_frontier: Vec<N> = vec![];
        for i in 0..resulting_frontier_slot.len() {
            let current_frontier_slot = current_frontier_slot_array[i];
            let sub_tree_frontier_slot = sub_tree_frontier_slot_array[i];
            // The rules for deciding if we should pick the existing frontier or override it with
            // the frontier from the newly created subtree.
            // 1) If either perfect tree counts at a depth are zero, we select the non-zero (our previous padding guarantees at least one is non-zero)
            // 2) If the perfect tree count for the existing tree is odd at a given depth, we select the frontier from the existing tree.
            //    This is because we know the perfect tree count for the incoming tree will be 1 (if it was > 1 we would need to do a small batch insert)
            // 3) If the perfect tree count for the existing tree is even at a given depth, we select the frontier from the incoming tree
            //    This is because the incoming tree will add to the perfect tree count and move the frontier to the right.
            if current_frontier_slot == 0 {
                final_frontier.push(padded_sub_tree_frontier[i]);
            } else if sub_tree_frontier_slot == 0 || current_frontier_slot % 2 != 0 {
                final_frontier.push(self.frontier[i]);
            }
            // ^-- This is safe because we ensure that subTreeFrontierSlot at this point can only be 1
            else {
                final_frontier.push(padded_sub_tree_frontier[i]);
            }
        }
        // Let's calculate the updated root now.
        let rightmost_element_sub_tree = index_to_directions(new_sub_tree.leaf_count - 1, height);
        // This is the height of our sub tree.
        // log2.ceil for an unsigned integer 'a' is u32::BITS - (a-1).leading_zeros(); or whatever u type is of interest
        let tree_height = usize::BITS - (new_sub_tree.leaf_count - 1).leading_zeros();
        // We can shortcut the root hash process by hashing our subtree.
        let mut root = reduce_tree(
            &self.hasher,
            &Self::move_down(
                &new_sub_tree.tree,
                &rightmost_element_sub_tree[0..(height - tree_height).try_into().unwrap()],
            ),
        );
        // Now we update the root hash by moving up to the height of the existing timber tree
        // If the root matches the timber frontier at that height - we hash it with zero
        // Otherwise we hash the frontier with our current root.
        for (i, slot) in current_frontier_slot_array
            .iter()
            .enumerate()
            .take(self.frontier.len())
            .skip(usize::try_from(tree_height).unwrap())
        {
            // We do this zero check because of past padding
            if slot % 2 == 0 || root == self.frontier[i] {
                root = self.hasher.tree_hash(&[root, N::zero()]).unwrap();
            } else {
                root = self.hasher.tree_hash(&[self.frontier[i], root]).unwrap();
            }
        }
        // From the last frontier of the existing tree, we hash up to the full height with zeroes
        for _j in self.frontier.len()..height.try_into().unwrap() {
            root = self.hasher.tree_hash(&[root, N::zero()]).unwrap();
        }
        self.root = root;
        self.frontier = final_frontier;
        self.leaf_count += blank_leaf_count + leaves.len();

        Ok(())
    }

    /// This function statelessly (i.e. does not modify timber.tree) calculates the sibling path for the element leaves[leafIndex].
    /// It only requires the frontier and leafCount to do so.
    /// The elements that will be inserted.
    /// The index in leaves that the sibling path will be calculated for.
    ///  returns the sibling path for that element.
    pub fn stateless_sibling_path(
        &self,
        leaves: &[N],
        leaf_index: usize,
    ) -> Option<Vec<PathElement<N>>> {
        if leaves.is_empty() || leaf_index >= leaves.len() {
            return None;
        }
        let leaves_insert_order = batch_leaves(self.leaf_count, leaves, vec![]);
        let leaf_value = leaves[leaf_index];
        let leaf_index_after_insertion = leaf_index + self.leaf_count;
        let frontier_tree = self.frontier_to_tree();
        let init_tree = Timber {
            root: self.root,
            tree: frontier_tree,
            frontier: self.frontier.clone(),
            leaf_count: self.leaf_count,
            height: self.height,
            width: self.width,
            hasher: self.hasher.clone(),
        };
        let final_tree = leaves_insert_order
            .iter()
            .try_fold(init_tree, |mut acc, curr| {
                acc.insert_leaves(curr).ok().map(|_| acc)
            });
        // if we get Some(final tree), then call get_sibling_path on it and_then to avoid Option<Option...
        final_tree.and_then(|ft| ft.get_sibling_path(leaf_value, leaf_index_after_insertion))
    }

    /// This function converts a frontier array into a tree (unbalanced), this is useful if we need a tree-like structure to add
    /// new leaves to.
    /// returns An object that represents a tree formed from the frontier.
    fn frontier_to_tree(&self) -> Tree<N> {
        if self.frontier.is_empty() {
            return Tree::Leaf(N::zero());
        }
        let current_frontier_slot_array = vec![self.leaf_count; self.frontier.len()]
            .iter()
            .enumerate()
            .map(|(i, a)| a / 2_usize.pow(i.try_into().unwrap()))
            .collect::<Vec<usize>>();
        let mut frontier_paths = vec![];
        for (i, slot) in current_frontier_slot_array.iter().enumerate() {
            if slot % 2 == 0 {
                frontier_paths.push(index_to_directions(
                    slot - 2,
                    self.height - u32::try_from(i).unwrap(),
                ));
            } else {
                frontier_paths.push(index_to_directions(
                    slot - 1,
                    self.height - u32::try_from(i).unwrap(),
                ));
            }
        }
        self.frontier
            .iter()
            .enumerate()
            .fold(self.tree.clone(), |acc, (index, curr)| {
                _safe_insert_leaf(*curr, &acc, &frontier_paths[index])
            })
    }

    /// Rolls a tree back to a given leafcount
    /// leafCount - The leafcount to which the tree should be rolled back to.
    /// returns updated timber instance.
    pub fn rollback(&mut self, leaf_count: usize) -> Result<(), TimberError> {
        if leaf_count > self.leaf_count {
            return Err(TimberError::CannotRollBack);
        }
        if leaf_count == self.leaf_count {
            return Ok(());
        }
        let path_to_new_last_element = index_to_directions(leaf_count - 1, self.height);
        self.tree = if leaf_count == 0 {
            Tree::Leaf(N::zero())
        } else {
            Self::prune_right_sub_tree(&self.tree, &path_to_new_last_element)
        };
        self.leaf_count = leaf_count;
        self.frontier = Self::calc_frontier(&self.tree, self.leaf_count, self.height, &self.hasher);
        self.root = self.reduce_tree();
        Ok(())
    }
    /// Updates a sibling path statelessly, given the previous path and a set of leaves to update the tree.
    /// `leaves` - The elements that will be inserted.
    /// `leaf_index` - The index in leaves that the sibling path will be calculated for.
    /// `leaf_value` - The value of the leaf sibling path will be calculated for.
    /// `sibling_path` - The latest sibling path that for leafIndex that will be updated.
    /// returns updated siblingPath for leafIndex with leafValue.
    pub fn stateless_increment_sibling_path(
        &self,
        leaves: &[N],
        leaf_index: usize,
        leaf_value: N,
        sibling_path: &[PathElement<N>],
    ) -> Option<Vec<PathElement<N>>> {
        if leaves.is_empty() || leaf_index >= self.leaf_count {
            return Some(sibling_path.to_vec());
        }
        let leaves_insert_order = batch_leaves(self.leaf_count, leaves, vec![]);
        // Turn the frontier into a tree-like structure
        let new_tree = self.frontier_to_tree();
        // Add the sibling path to the frontier tree-like structure
        let sibling_tree =
            insert_sibling_path(new_tree, sibling_path, leaf_index, leaf_value, self.height);
        let init_tree = Timber {
            root: self.root,
            tree: sibling_tree,
            frontier: self.frontier.clone(),
            leaf_count: self.leaf_count,
            height: self.height,
            width: self.width,
            hasher: self.hasher.clone(),
        };
        // this is a little more complex than it otherwise would be because we must handle the possibility of insert_leaves failing.
        let final_tree = leaves_insert_order
            .iter()
            .try_fold(init_tree, |mut acc, curr| {
                acc.insert_leaves(curr).ok().map(|_| acc)
            });
        // if we get Some(final tree), then call get_sibling_path on it and_then to avoid Option<Option...
        final_tree.and_then(|ft| ft.get_sibling_path(leaf_value, leaf_index))
    }
    /// Updates a list of sibling path statelessly, given the previous paths and a set of leaves to update the tree.
    /// This is used in a Nightfall 4 client to update a clients sibling paths, for that reason it makes sure the leaves are a multiple of a
    /// fixed size (in Nightfall 4's case this will be 8).
    /// `leaves` - The elements that will be inserted.
    /// `leaf_index` - The index in leaves that the sibling path will be calculated for.
    /// `leaf_value` - The value of the leaf sibling path will be calculated for.
    /// `sibling_paths` - The latest sibling path that for leafIndex that will be updated.
    /// returns updated siblingPath for leafIndex with leafValue.
    pub fn stateless_increment_sibling_paths(
        &self,
        leaves: &[N],
        sibling_paths: &[MembershipProof<N>],
        insertion_size: usize,
    ) -> Result<Vec<MembershipProof<N>>, TimberError> {
        if leaves.len() % insertion_size != 0 {
            return Err(TimberError::InvalidBatchSize);
        }

        if leaves.is_empty() {
            return Ok(sibling_paths.to_vec());
        }
        // Turn the frontier into a tree-like structure.
        let mut new_tree = self.frontier_to_tree();
        // If the leaf_index of the sibling path is greater than the current leaf count, we should error as this means it corresponds to a leaf not in the tree.
        for proof in sibling_paths.iter() {
            // If a proof does not hash to the current root then it is not a valid path and we can't update it from the information we have.
            if proof.verify(&self.root, &self.hasher).is_err() {
                return Err(TimberError::InvalidMembershipProof);
            }

            if proof.leaf_index >= self.leaf_count {
                return Err(TimberError::WrongNodeType);
            }
            new_tree = insert_sibling_path(
                new_tree,
                &proof.sibling_path,
                proof.leaf_index,
                proof.node_value,
                self.height,
            );
        }

        let mut init_tree = Timber {
            root: self.root,
            tree: new_tree,
            frontier: self.frontier.clone(),
            leaf_count: self.leaf_count,
            height: self.height,
            width: self.width,
            hasher: self.hasher.clone(),
        };
        // this is a little more complex than it otherwise would be because we must handle the possibility of insert_leaves failing.
        for leaf_chunk in leaves.chunks(insertion_size) {
            let subtree = Tree::<N>::build_from_values(insertion_size.ilog2(), leaf_chunk);
            init_tree.insert_subtree(subtree)?;
        }

        // if we get Some(final tree), then call get_sibling_path on it and_then to avoid Option<Option...
        cfg_iter!(sibling_paths)
            .map(|proof| {
                init_tree
                    .get_sibling_path(proof.node_value, proof.leaf_index)
                    .map(|sibling_path| MembershipProof {
                        node_value: proof.node_value,
                        sibling_path,
                        leaf_index: proof.leaf_index,
                    })
                    .ok_or(TimberError::CannotGetPath)
            })
            .collect::<Result<Vec<MembershipProof<N>>, TimberError>>()
    }

    /// This function inserts a subtree into a tree and returns all relevant information needed to prove correctness of the insertion
    /// in a circuit.
    pub fn insert_for_circuit(
        &mut self,
        leaf_values: &[N],
    ) -> Result<CircuitInsertionInfo<N>, TimberError> {
        if leaf_values.len().next_power_of_two() != leaf_values.len() {
            return Err(TimberError::InvalidBatchSize);
        }
        let subtree_height = leaf_values.len().ilog2();
        let old_root = self.root;
        let leaf_count = self.leaf_count;

        let subtree = Tree::<N>::build_from_values(subtree_height, leaf_values);

        // Here we calculate the difference in height of our main tree and our subtree
        let remaining_height = self.height - subtree_height;
        // This is the index of the root of the subtree in the main tree, counted from the left, starting at zero
        let index: usize = if self.leaf_count != 0 {
            (self.leaf_count - 1) / leaf_values.len() + 1
        } else {
            0
        };

        let sibling_path =
            get_node_sibling_path(&self.tree, remaining_height, index, self.get_tree_hasher());

        let proof = MembershipProof {
            node_value: N::zero(),
            sibling_path,
            leaf_index: index,
        };
        self.insert_subtree(subtree)?;

        let new_root = self.root;

        Ok(CircuitInsertionInfo {
            old_root,
            new_root,
            leaf_count,
            leaves: leaf_values.to_vec(),
            proof,
        })
    }

    /// This function batch inserts leaves into the tree returning the circuit info after each batch.
    pub fn batch_insert_for_circuit(
        &mut self,
        leaf_values: &[N],
    ) -> Result<Vec<CircuitInsertionInfo<N>>, TimberError> {
        if leaf_values.is_empty() {
            return Ok(vec![]);
        }
        let mut circuit_infos = vec![];
        for leaf_chunk in leaf_values.chunks(8) {
            let circuit_info = self.insert_for_circuit(leaf_chunk)?;
            circuit_infos.push(circuit_info);
        }
        Ok(circuit_infos)
    }
}
