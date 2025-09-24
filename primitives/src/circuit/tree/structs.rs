//! Circuit version of various merkle tree related structs that we require.

use ark_ff::PrimeField;
use ark_std::{format, string::ToString, vec, vec::Vec};
use jf_relation::{errors::CircuitError, BoolVar, Circuit, PlonkCircuit, Variable};

use crate::{
    circuit::poseidon::PoseidonHashGadget,
    poseidon::constants::PoseidonParams,
    trees::{
        imt::{IMTCircuitInsertionInfo, LeafDBEntry},
        CircuitInsertionInfo, Directions, MembershipProof, PathElement,
    },
};

#[derive(Debug, Clone, Copy)]
/// Circuit variable used to represent a [`PathElement`] in a circuit.
pub struct PathElementVar {
    /// False or Zero for left, True or One for right.
    direction: BoolVar,
    /// The value of the node.
    value: Variable,
}

impl PathElementVar {
    /// Create a new path element variable.
    pub fn new(direction: BoolVar, value: Variable) -> Self {
        Self { direction, value }
    }

    /// Get the direction of the path element.
    pub fn direction(&self) -> &BoolVar {
        &self.direction
    }

    /// Get the value of the path element.
    pub fn value(&self) -> &Variable {
        &self.value
    }

    /// Create a new path element variable from a path element.
    pub fn from_path_element<F: PrimeField>(
        circuit: &mut PlonkCircuit<F>,
        path_element: &PathElement<F>,
    ) -> Result<Self, CircuitError> {
        let direction = if path_element.direction == Directions::HashWithThisNodeOnLeft {
            circuit.create_boolean_variable(false)?
        } else {
            circuit.create_boolean_variable(true)?
        };
        let value = circuit.create_variable(path_element.value)?;
        Ok(Self::new(direction, value))
    }
}

#[derive(Debug, Clone)]
/// Circuit variable used to represent a  [`MembershipProof`] in a circuit.
/// As we we will want to reuse a single [`MembershipProofVar`]s multiple times,
/// the value of the associated node is omitted from the struct.
pub struct MembershipProofVar {
    /// The path elements of the membership proof.
    path_elements: Vec<PathElementVar>,
}

impl MembershipProofVar {
    /// Create a new membership proof variable.
    pub fn new(path_elements: Vec<PathElementVar>) -> Self {
        Self { path_elements }
    }

    /// Get the path elements of the membership proof.
    pub fn path_elements(&self) -> &Vec<PathElementVar> {
        &self.path_elements
    }

    /// Create a new membership proof variable from a membership proof.
    pub fn from_membership_proof<F: PrimeField>(
        circuit: &mut PlonkCircuit<F>,
        membership_proof: &MembershipProof<F>,
    ) -> Result<Self, CircuitError> {
        let path_elements = membership_proof
            .sibling_path
            .iter()
            .map(|path_element| PathElementVar::from_path_element(circuit, path_element))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(path_elements))
    }

    /// Create a new membership proof variable from a vector of variables.
    pub fn from_vars<F: PrimeField>(
        circuit: &mut PlonkCircuit<F>,
        vars: &[Variable],
    ) -> Result<Self, CircuitError> {
        if vars.len() < 4 {
            return Err(CircuitError::ParameterError(format!(
                "Vector of vars length: ({}) must be at least 4",
                vars.len(),
            )));
        }
        if vars.len() % 2 != 0 {
            return Err(CircuitError::ParameterError(format!(
                "Vector of vars length: ({}) should be even",
                vars.len(),
            )));
        }

        let mut path_elements = Vec::new();
        // We skip the first `var`, as it represents the value of the MembershipProof,
        // which is not a part of MembershipProofVar
        let path_vars = vars[1..vars.len() - 1].to_vec();
        for chunk in path_vars.chunks_exact(2) {
            let direction_var = chunk[0];
            circuit.enforce_bool(direction_var)?;

            let value_var = chunk[1];

            let path_element_var = PathElementVar {
                direction: BoolVar(direction_var),
                value: value_var,
            };
            path_elements.push(path_element_var);
        }

        Ok(MembershipProofVar { path_elements })
    }

    /// Verifies a membership proof against a supplied value and root.
    pub fn verify_membership_proof<F: PoseidonParams>(
        &self,
        circuit: &mut PlonkCircuit<F>,
        value: &Variable,
        root: &Variable,
    ) -> Result<BoolVar, CircuitError> {
        let hash = self.calculate_new_root(circuit, value)?;
        circuit.is_equal(hash, *root)
    }

    /// Calculates the new root of the tree if we updated the value of the node this proof is for.
    pub fn calculate_new_root<F: PoseidonParams>(
        &self,
        circuit: &mut PlonkCircuit<F>,
        new_value: &Variable,
    ) -> Result<Variable, CircuitError> {
        let mut hash = *new_value;
        for path_element in self.path_elements.iter() {
            let node_value = *path_element.value();
            let selector = *path_element.direction();

            // First we calculate what the left output should be.
            let wires_in = [hash, selector.into(), node_value, selector.into()];
            let lc = [F::zero(), F::zero(), F::one(), F::zero()];
            let mul_selectors = [F::one(), -F::one()];
            let left = circuit.gen_quad_poly(&wires_in, &lc, &mul_selectors, F::zero())?;

            //Then we calculate what the right output should be.
            let wires_in = [node_value, selector.into(), hash, selector.into()];
            let lc = [F::zero(), F::zero(), F::one(), F::zero()];
            let mul_selectors = [F::one(), -F::one()];
            let right = circuit.gen_quad_poly(&wires_in, &lc, &mul_selectors, F::zero())?;

            hash = circuit.tree_hash(&[left, right])?;
        }
        Ok(hash)
    }
}

/// Circuit variable used to represent [`CircuitInsertionInfo`] in a circuit.
pub struct CircuitInsertionInfoVar {
    /// The root before the insertion.
    pub old_root: Variable,
    /// The root after the insertion.
    pub new_root: Variable,
    /// The value of the leaves.
    pub leaves: Vec<Variable>,
    /// The path elements of the membership proof.
    pub proof: MembershipProofVar,
}

impl CircuitInsertionInfoVar {
    /// Create a new circuit insertion info variable.
    pub fn new(
        old_root: Variable,
        new_root: Variable,
        leaves: Vec<Variable>,
        proof: MembershipProofVar,
    ) -> Self {
        Self {
            old_root,
            new_root,
            leaves,
            proof,
        }
    }

    /// Create a new circuit insertion info variable from a [`CircuitInsertionInfo`].
    pub fn from_circuit_insertion_info<F: PrimeField>(
        circuit: &mut PlonkCircuit<F>,
        circuit_insertion_info: &CircuitInsertionInfo<F>,
    ) -> Result<Self, CircuitError> {
        // The old and new roots will always be public inputs.
        let old_root = circuit.create_variable(circuit_insertion_info.old_root)?;
        let new_root = circuit.create_variable(circuit_insertion_info.new_root)?;
        let leaves = circuit_insertion_info
            .leaves
            .iter()
            .map(|leaf| circuit.create_variable(*leaf))
            .collect::<Result<Vec<_>, _>>()?;
        let proof =
            MembershipProofVar::from_membership_proof(circuit, &circuit_insertion_info.proof)?;
        Ok(Self::new(old_root, new_root, leaves, proof))
    }

    /// Create a new circuit insertion info variable from a slice of variables. Here `height` is the height of the tree.
    pub fn from_vars<F: PrimeField>(
        circuit: &mut PlonkCircuit<F>,
        vars: &[Variable],
        height: usize,
    ) -> Result<Self, CircuitError> {
        // The length of the `vars` vector should be at least 5 + height * 2.
        if vars.len() < 5 + height * 2 {
            return Err(CircuitError::ParameterError(format!(
                "Length of vars: ({}) must be at least 5 + 2 * height ({})",
                vars.len(),
                height,
            )));
        }
        let old_root = vars[0];
        let new_root = vars[1];

        // We can calculate the number of leaves because the last 2 + height * 2 elements are the membership proof.
        let membership_proof_size = 2 + height * 2;
        let leaves = vars[3..vars.len() - membership_proof_size].to_vec();

        let proof = MembershipProofVar::from_vars(
            circuit,
            &vars[(vars.len() - membership_proof_size)..vars.len()],
        )?;
        Ok(CircuitInsertionInfoVar {
            old_root,
            new_root,
            leaves,
            proof,
        })
    }

    /// Verifies an insertion into a tree using supplied [`CircuitInsertionInfo`].
    pub fn verify_subtree_insertion_gadget<F: PoseidonParams>(
        &self,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError> {
        let cii_var = self;

        // We first verify a membership proof to check the `old_root` is correct.
        // Here the `value` is zero, as this is before any subtree has been inserted.
        let init_check = MembershipProofVar::verify_membership_proof(
            &cii_var.proof,
            circuit,
            &circuit.zero(),
            &cii_var.old_root,
        )?;

        if cii_var.leaves.len().next_power_of_two() != cii_var.leaves.len() {
            return Err(CircuitError::ParameterError(
                "Need a power of two number of leaves".to_string(),
            ));
        }

        let mut nodes = cii_var.leaves.clone();

        for _ in 0..cii_var.leaves.len().ilog2() {
            nodes = nodes
                .chunks(2)
                .map(|chunk| circuit.tree_hash(&[chunk[0], chunk[1]]))
                .collect::<Result<Vec<_>, _>>()?;
        }
        // We now verify a membership proof to check the `new_root` is correct.
        // Here we enforce that the `value` is the root of the subtree, as this is after the subtree has been inserted.
        let final_check = MembershipProofVar::verify_membership_proof(
            &cii_var.proof,
            circuit,
            &nodes[0],
            &cii_var.new_root,
        )?;
        // We enforce that both checks are true
        circuit.mul_gate(init_check.into(), final_check.into(), circuit.one())?;
        Ok(())
    }

    /// Verify the insertion of a subtree into a tree is correct.
    pub fn verify_subtree_insertion<F: PoseidonParams>(
        circuit: &mut PlonkCircuit<F>,
        circuit_insertion_info: &CircuitInsertionInfo<F>,
    ) -> Result<(), CircuitError> {
        let cii_var =
            CircuitInsertionInfoVar::from_circuit_insertion_info(circuit, circuit_insertion_info)?;
        cii_var.verify_subtree_insertion_gadget(circuit)
    }
}

/// Variable used to represent an entry in the leaf database in a circuit.
pub struct LeafDBEntryVar {
    /// The value of this nullifier
    pub value: Variable,
    /// The index of the leaf representing this nullifier in the Indexed Merkle tree
    pub index: Variable,
    /// The index of the next highest value nullifier in the Indexed Merkle tree
    pub next_index: Variable,
    /// The value of the next highest value nullifier in the Indexed Merkle tree
    pub next_value: Variable,
}

impl LeafDBEntryVar {
    /// Create a new leaf database entry variable.
    pub fn new(
        value: Variable,
        index: Variable,
        next_index: Variable,
        next_value: Variable,
    ) -> Self {
        Self {
            value,
            index,
            next_index,
            next_value,
        }
    }

    /// Create a new leaf database entry variable from a leaf database entry.
    pub fn from_leaf_db_entry<F: PrimeField>(
        circuit: &mut PlonkCircuit<F>,
        leaf_db_entry: &LeafDBEntry<F>,
    ) -> Result<Self, CircuitError> {
        let value = circuit.create_variable(leaf_db_entry.value)?;
        let index = circuit.create_variable(F::from(leaf_db_entry.index))?;
        let next_index = circuit.create_variable(leaf_db_entry.next_index)?;
        let next_value = circuit.create_variable(leaf_db_entry.next_value)?;
        Ok(Self::new(value, index, next_index, next_value))
    }

    /// Hashes the variable.
    pub fn hash<F: PoseidonParams>(
        &self,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<Variable, CircuitError> {
        let wires_in = [self.value, self.next_index, self.next_value];
        circuit.poseidon_hash(&wires_in)
    }
}

/// Circuit variable used to represent [`IMTCircuitInsertionInfo`] in a circuit.
pub struct IMTCircuitInsertionInfoVar {
    /// Root of the tree before any updates.
    pub old_root: Variable,
    /// The circuit info for the actual subtree insertion.
    pub circuit_info: CircuitInsertionInfoVar,
    /// The index of the first leaf inserted.
    pub first_index: Variable,
    /// The low nullifiers for the leaves inserted with non_membership proofs.
    pub low_nullifiers: Vec<(LeafDBEntryVar, MembershipProofVar)>,
    /// The entries of the nullifiers inserted.
    pub pending_inserts: Vec<LeafDBEntryVar>,
}

impl IMTCircuitInsertionInfoVar {
    /// Create a new IMT circuit insertion info variable.
    pub fn new(
        old_root: Variable,
        circuit_info: CircuitInsertionInfoVar,
        first_index: Variable,
        low_nullifiers: Vec<(LeafDBEntryVar, MembershipProofVar)>,
        pending_inserts: Vec<LeafDBEntryVar>,
    ) -> Self {
        Self {
            old_root,
            circuit_info,
            first_index,
            low_nullifiers,
            pending_inserts,
        }
    }

    /// Create a new IMT circuit insertion info variable from an [`IMTCircuitInsertionInfo`].
    pub fn from_imt_circuit_insertion_info<F: PrimeField>(
        circuit: &mut PlonkCircuit<F>,
        imt_circuit_insertion_info: &IMTCircuitInsertionInfo<F>,
    ) -> Result<Self, CircuitError> {
        let old_root = circuit.create_variable(imt_circuit_insertion_info.old_root)?;
        // The old and new roots will always be public inputs.
        let old_root_inner =
            circuit.create_variable(imt_circuit_insertion_info.circuit_info.old_root)?;
        let new_root = circuit.create_variable(imt_circuit_insertion_info.circuit_info.new_root)?;
        let leaves = imt_circuit_insertion_info
            .circuit_info
            .leaves
            .iter()
            .map(|leaf| circuit.create_variable(*leaf))
            .collect::<Result<Vec<_>, _>>()?;
        let proof = MembershipProofVar::from_membership_proof(
            circuit,
            &imt_circuit_insertion_info.circuit_info.proof,
        )?;
        let circuit_info = CircuitInsertionInfoVar::new(old_root_inner, new_root, leaves, proof);

        let first_index =
            circuit.create_variable(F::from(imt_circuit_insertion_info.first_index))?;
        let low_nullifiers = imt_circuit_insertion_info
            .low_nullifiers
            .iter()
            .map(|(leaf_db_entry, membership_proof)| {
                Ok((
                    LeafDBEntryVar::from_leaf_db_entry(circuit, leaf_db_entry)?,
                    MembershipProofVar::from_membership_proof(circuit, membership_proof)?,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pending_inserts = imt_circuit_insertion_info
            .pending_inserts
            .iter()
            .map(|leaf_db_entry| LeafDBEntryVar::from_leaf_db_entry(circuit, leaf_db_entry))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(
            old_root,
            circuit_info,
            first_index,
            low_nullifiers,
            pending_inserts,
        ))
    }

    /// Verify the insertion of a subtree into an Indexed Merkle Tree is correct.
    /// Here, we assume the subtree depth is 3.
    pub fn verify_subtree_insertion_gadget<F: PoseidonParams>(
        &self,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<(), CircuitError> {
        let imt_insert_var = self;
        let inner_value_var = imt_insert_var
            .pending_inserts
            .iter()
            .map(|x| x.value)
            .collect::<Vec<Variable>>();
        let mut leaf_index = imt_insert_var.first_index;
        // We constrain `leaf_index` to match up with the membership proof of the subtree root.
        // Since a `true` direction means the node is on the left and a `false` direction means the node is on the right,
        // we calculate `2^3 * sum_i (1 - direction[i]) * 2^i`. The `2^3` is due to the subtree depth being 3.
        // Therefore, the initial 3 directions are all left.
        let (coeffs, path_vars): (Vec<F>, Vec<Variable>) = imt_insert_var
            .circuit_info
            .proof
            .path_elements()
            .iter()
            .enumerate()
            .map(|(i, path)| (-F::from(2u8).pow([i as u64 + 3]), path.direction.0))
            .collect::<Vec<(F, Variable)>>()
            .into_iter()
            .unzip();
        let constant = F::from(8u8)
            * (F::from(2u8).pow([imt_insert_var.circuit_info.proof.path_elements().len() as u64])
                - F::one());
        circuit.lin_comb_gate(&coeffs, &constant, &path_vars, &leaf_index)?;

        let mut pending_insertions = Vec::<LeafDBEntryVar>::new();
        let mut root = imt_insert_var.old_root;
        let mut zero_flags = Vec::<BoolVar>::new();
        for ((leaf_db_entry, membership_proof), inner_value) in imt_insert_var
            .low_nullifiers
            .iter()
            .zip(inner_value_var.iter())
        {
            // We check if the inner_value of the pending insert is zero.
            // If it is, we want to skip some of the subsequent checks
            let zero_flag = circuit.is_zero(*inner_value)?;

            // Check if the low_nullifier value is less than the value we are inserting
            let gt_check = circuit.is_gt(*inner_value, leaf_db_entry.value)?;
            // Check that the value we are inserting is either less than leaf_db_entry.next_value or leaf_db_entry.next_index is zero.
            let less_check_one = circuit.is_lt(*inner_value, leaf_db_entry.next_value)?;
            let less_check_two = circuit.is_zero(leaf_db_entry.next_index)?;

            let lt_check = circuit.logic_or(less_check_one, less_check_two)?;
            let ln_check = circuit.logic_and(gt_check, lt_check)?;

            // We perform a XOR to enforce that either `zero_flag` is true or `ln_check` is true but not both.
            // Since both are `BoolVar`s, we do this by enforcing that `zero_flag` + `and_check` = 1
            circuit.add_gate(zero_flag.into(), ln_check.into(), circuit.one())?;

            // We keep a list of `BoolVar`s that stores whenever we find a valid low nullifier for our inserted leaf.
            // A valid low nullifier can be either `leaf_db_entry` or one of the `pending_insertions`.
            // Ultimately this list must have precisely one valid low nullifier.
            let mut valid_low_nullifiers = Vec::<Variable>::new();

            // Calculate the appropriate hash of the low nullifier.
            let original_leaf = leaf_db_entry.hash(circuit)?;
            // Check whether the proof is correct or not for the low nullifier. If it's not, the low nullifier should be a pending insert.
            let correct_proof =
                membership_proof.verify_membership_proof(circuit, &original_leaf, &root)?;

            valid_low_nullifiers.push(correct_proof.into());

            // Calculate the updated leaf and hash it.
            let updated_nullifier = LeafDBEntryVar {
                value: leaf_db_entry.value,
                index: leaf_db_entry.index,
                next_index: leaf_index,
                next_value: *inner_value,
            };
            let updated_leaf = updated_nullifier.hash(circuit)?;

            // Then we will set the new root to be the value obtained when you insert the updated leaf into the tree down the path of the proof.
            // However, if the membership proof fails or `inner_value` is zero, we leave the root unchanged,
            // as, in that case, the low_nullifier is part of the subtree we are inserting or we're inserting a null leaf.
            let new_root = membership_proof.calculate_new_root(circuit, &updated_leaf)?;
            let update_root = BoolVar(circuit.gen_quad_poly(
                &[
                    correct_proof.into(),
                    zero_flag.into(),
                    circuit.zero(),
                    circuit.zero(),
                ],
                &[F::one(), F::zero(), F::zero(), F::zero()],
                &[-F::one(), F::zero()],
                F::zero(),
            )?);
            root = circuit.conditional_select(update_root, root, new_root)?;

            // Update the pending insertions if they need updating.
            // If zero_flag is true they do not need to be updated.
            // We also need to record the `next_index` and `next_value` of the leaf we are inserting.
            // We cannot necessarily use `leaf_db_entry.next_index` and `leaf_db_entry.next_value` as,
            // if our low nullifier is a pending insert, we do not know these to be correct.
            let mut next_index = leaf_db_entry.next_index;
            let mut next_value = leaf_db_entry.next_value;
            for pending_insert in pending_insertions.iter_mut() {
                let gt = circuit.is_gt(*inner_value, pending_insert.value)?;
                let lt_1 = circuit.is_lt(*inner_value, pending_insert.next_value)?;
                let lt_2 = circuit.is_zero(pending_insert.next_index)?;
                let or_check = circuit.logic_or(lt_1, lt_2)?;
                let ln_check = circuit.logic_and(gt, or_check)?;
                let zero_low_null = circuit.is_zero(pending_insert.value)?;
                // Only if our inequality checks hold and `pending_insert.value` is non-zero do we update.
                // Note that if `inner_value` is zero our checks automatically fail.
                let selector = BoolVar(circuit.gen_quad_poly(
                    &[
                        ln_check.into(),
                        zero_low_null.into(),
                        circuit.zero(),
                        circuit.zero(),
                    ],
                    &[F::one(), F::zero(), F::zero(), F::zero()],
                    &[-F::one(), F::zero()],
                    F::zero(),
                )?);

                // We record if we found a valid low nullifier or not.
                valid_low_nullifiers.push(selector.into());

                next_value =
                    circuit.conditional_select(selector, next_value, pending_insert.next_value)?;

                next_index =
                    circuit.conditional_select(selector, next_index, pending_insert.next_index)?;

                let updated_next_value = circuit.conditional_select(
                    selector,
                    pending_insert.next_value,
                    *inner_value,
                )?;

                let updated_next_index =
                    circuit.conditional_select(selector, pending_insert.next_index, leaf_index)?;

                *pending_insert = LeafDBEntryVar {
                    value: pending_insert.value,
                    index: pending_insert.index,
                    next_index: updated_next_index,
                    next_value: updated_next_value,
                };
            }

            // We ensure we have exactly one low nullifier (or `inner_value` is zero).
            let valid_low_null_count = circuit.lin_comb(
                &vec![F::one(); valid_low_nullifiers.len()],
                &F::zero(),
                &valid_low_nullifiers,
            )?;
            circuit.quad_poly_gate(
                &[
                    valid_low_null_count,
                    zero_flag.into(),
                    circuit.zero(),
                    circuit.zero(),
                    circuit.zero(),
                ],
                &[-F::one(), -F::one(), F::zero(), F::zero()],
                &[F::one(), F::zero()],
                F::zero(),
                F::one(),
            )?;

            let pending_insert = LeafDBEntryVar {
                value: *inner_value,
                index: leaf_index,
                next_index,
                next_value,
            };

            leaf_index = circuit.add_constant(leaf_index, &F::one())?;
            pending_insertions.push(pending_insert);
            zero_flags.push(zero_flag);
        }

        let leaf_values = pending_insertions
            .iter()
            .zip(zero_flags.into_iter())
            .map(|(x, zero_flag)| {
                let hash = circuit.poseidon_hash(&[x.value, x.next_index, x.next_value])?;
                circuit.conditional_select(zero_flag, hash, circuit.zero())
            })
            .collect::<Result<Vec<Variable>, CircuitError>>()?;

        let circuit_info = CircuitInsertionInfoVar {
            old_root: imt_insert_var.circuit_info.old_root,
            new_root: imt_insert_var.circuit_info.new_root,
            leaves: leaf_values,
            proof: imt_insert_var.circuit_info.proof.clone(),
        };

        circuit.enforce_equal(root, imt_insert_var.circuit_info.old_root)?;

        circuit_info.verify_subtree_insertion_gadget(circuit)
    }
    /// Verify the insertion of a subtree into an Indexed Merkle Tree is correct.
    pub fn verify_subtree_insertion<F: PoseidonParams>(
        circuit: &mut PlonkCircuit<F>,
        imt_insertion_info: &IMTCircuitInsertionInfo<F>,
    ) -> Result<(), CircuitError> {
        let cii_var = IMTCircuitInsertionInfoVar::from_imt_circuit_insertion_info(
            circuit,
            imt_insertion_info,
        )?;
        cii_var.verify_subtree_insertion_gadget(circuit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poseidon::Poseidon,
        trees::{get_node_sibling_path, imt::IndexedMerkleTree, timber::Timber},
    };
    use ark_bn254::Fr as Fr254;
    use ark_std::{collections::HashMap, rand::seq::SliceRandom, UniformRand};

    #[test]
    fn test_membership_proof_from_vars() {
        let mut circuit = PlonkCircuit::<Fr254>::new_turbo_plonk();
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        let mut tree = Timber::new(poseidon, 32);
        let mut proofs = Vec::new();
        let leaves: Vec<Fr254> = (0..48).map(|_| Fr254::rand(&mut rng)).collect();
        tree.insert_leaves(&leaves).unwrap();
        for (i, leaf) in leaves.iter().enumerate() {
            let sibling_path = get_node_sibling_path(&tree.tree, 32, i, tree.get_tree_hasher());
            let proof = MembershipProof {
                sibling_path,
                node_value: *leaf,
                leaf_index: i,
            };
            proofs.push(proof);
        }

        for proof in proofs.iter() {
            let field_elems = Vec::from(proof);
            let vars = field_elems
                .iter()
                .map(|&elem| circuit.create_variable(elem))
                .collect::<Result<Vec<Variable>, CircuitError>>()
                .unwrap();

            let proof_var_from_proof =
                MembershipProofVar::from_membership_proof(&mut circuit, proof).unwrap();

            let proof_var = MembershipProofVar::from_vars(&mut circuit, &vars).unwrap();

            assert_eq!(
                proof_var.path_elements.len(),
                proof_var_from_proof.path_elements.len()
            );

            for (path_var, path_var_from_proof) in proof_var
                .path_elements
                .iter()
                .zip(proof_var_from_proof.path_elements.iter())
            {
                assert_eq!(
                    circuit.witness(path_var.direction.0).unwrap(),
                    circuit.witness(path_var_from_proof.direction.0).unwrap()
                );
                assert_eq!(
                    circuit.witness(path_var.value).unwrap(),
                    circuit.witness(path_var_from_proof.value).unwrap()
                );
            }
        }
        circuit.check_circuit_satisfiability(&[]).unwrap();
    }

    #[test]
    fn test_circuit_insertion_info_from_vars() {
        let mut circuit = PlonkCircuit::<Fr254>::new_turbo_plonk();
        let mut rng = ark_std::test_rng();
        for _ in 0..25 {
            let poseidon = Poseidon::<Fr254>::new();
            let mut tree = Timber::new(poseidon, 32);
            let start_amount = usize::rand(&mut rng) % 2usize.pow(10);
            let leaves: Vec<Fr254> = (0..start_amount).map(|_| Fr254::rand(&mut rng)).collect();

            let subtree_leaves: Vec<Fr254> = (0..8).map(|_| Fr254::rand(&mut rng)).collect();
            tree.insert_leaves(&leaves).unwrap();

            let circuit_insertion_info = tree.insert_for_circuit(&subtree_leaves).unwrap();
            let field_elems = Vec::from(&circuit_insertion_info);
            let vars = field_elems
                .iter()
                .map(|&elem| circuit.create_variable(elem))
                .collect::<Result<Vec<Variable>, CircuitError>>()
                .unwrap();
            let circuit_insertion_info_var =
                CircuitInsertionInfoVar::from_vars(&mut circuit, &vars, 32 - 3).unwrap();
            let circuit_insertion_info_var_from_info =
                CircuitInsertionInfoVar::from_circuit_insertion_info(
                    &mut circuit,
                    &circuit_insertion_info,
                )
                .unwrap();
            assert_eq!(
                circuit
                    .witness(circuit_insertion_info_var.old_root)
                    .unwrap(),
                circuit
                    .witness(circuit_insertion_info_var_from_info.old_root)
                    .unwrap()
            );
            assert_eq!(
                circuit
                    .witness(circuit_insertion_info_var.new_root)
                    .unwrap(),
                circuit
                    .witness(circuit_insertion_info_var_from_info.new_root)
                    .unwrap()
            );
            assert_eq!(
                circuit_insertion_info_var.leaves.len(),
                circuit_insertion_info_var_from_info.leaves.len()
            );
            for (leaf, leaf_from_info) in circuit_insertion_info_var
                .leaves
                .iter()
                .zip(circuit_insertion_info_var_from_info.leaves.iter())
            {
                assert_eq!(
                    circuit.witness(*leaf).unwrap(),
                    circuit.witness(*leaf_from_info).unwrap()
                );
            }
            assert_eq!(
                circuit_insertion_info_var.proof.path_elements.len(),
                circuit_insertion_info_var_from_info
                    .proof
                    .path_elements
                    .len()
            );
            for (path_var, path_var_from_info) in
                circuit_insertion_info_var.proof.path_elements.iter().zip(
                    circuit_insertion_info_var_from_info
                        .proof
                        .path_elements
                        .iter(),
                )
            {
                assert_eq!(
                    circuit.witness(path_var.direction.0).unwrap(),
                    circuit.witness(path_var_from_info.direction.0).unwrap()
                );
                assert_eq!(
                    circuit.witness(path_var.value).unwrap(),
                    circuit.witness(path_var_from_info.value).unwrap()
                );
            }
        }
        circuit.check_circuit_satisfiability(&[]).unwrap();
    }

    #[test]
    fn test_verify_membership_proof() {
        let mut circuit = PlonkCircuit::<Fr254>::new_turbo_plonk();
        let mut rng = ark_std::test_rng();
        let poseidon = Poseidon::<Fr254>::new();
        let mut tree = Timber::new(poseidon, 32);
        let mut proofs = Vec::new();
        let leaves = vec![Fr254::rand(&mut rng); 4];
        tree.insert_leaves(&leaves).unwrap();
        for (i, leaf) in leaves.iter().enumerate() {
            let sibling_path = get_node_sibling_path(&tree.tree, 32, i, tree.get_tree_hasher());
            let proof = MembershipProof {
                sibling_path,
                node_value: *leaf,
                leaf_index: i,
            };
            proofs.push(proof);
        }
        let root = tree.reduce_tree();
        let root_var = circuit.create_variable(root).unwrap();

        for proof in proofs.iter() {
            proof.verify(&root, &poseidon).unwrap();
            let proof_var = MembershipProofVar::from_membership_proof(&mut circuit, proof).unwrap();
            let val_var = circuit.create_variable(proof.node_value).unwrap();
            proof_var
                .verify_membership_proof(&mut circuit, &val_var, &root_var)
                .unwrap();
        }

        circuit.check_circuit_satisfiability(&[]).unwrap();
        ark_std::println!("circuit num constraints: {}", circuit.num_gates() / 4);
    }

    #[test]
    fn test_verify_subtree_insertion() {
        let mut rng = ark_std::test_rng();
        for _ in 0..25 {
            let mut circuit = PlonkCircuit::<Fr254>::new_turbo_plonk();

            let poseidon = Poseidon::<Fr254>::new();
            let mut tree = Timber::new(poseidon, 32);
            let start_amount = usize::rand(&mut rng) % 2usize.pow(10);
            let leaves = vec![Fr254::rand(&mut rng); start_amount];

            let subtree_leaves = vec![Fr254::rand(&mut rng); 8];
            tree.insert_leaves(&leaves).unwrap();

            let circuit_insertion_info = tree.insert_for_circuit(&subtree_leaves).unwrap();

            CircuitInsertionInfoVar::verify_subtree_insertion(
                &mut circuit,
                &circuit_insertion_info,
            )
            .unwrap();
            circuit.check_circuit_satisfiability(&[]).unwrap();
        }
    }

    #[test]
    fn test_verify_imt_subtree_insertion() {
        let mut rng = ark_std::test_rng();
        for _ in 0..25 {
            let mut circuit = PlonkCircuit::<Fr254>::new_ultra_plonk(12);

            let poseidon = Poseidon::<Fr254>::new();
            let mut tree: IndexedMerkleTree<Fr254, Poseidon<Fr254>, HashMap<Fr254, LeafDBEntry<Fr254>>> = IndexedMerkleTree::<
                Fr254,
                Poseidon<Fr254>,
                HashMap<Fr254, LeafDBEntry<Fr254>>,
            >::new(poseidon, 32)
            .unwrap();

            let start_amount = u32::rand(&mut rng) % 2u32.pow(10);

            let mut leaves = (1..start_amount).collect::<Vec<u32>>();
            leaves.shuffle(&mut rng);
            let leaves = leaves.iter().map(|i| Fr254::from(*i)).collect::<Vec<_>>();

            let subtree_leaves = (0..8)
                .map(|j| {
                    if j % 3 != 0 {
                        Fr254::rand(&mut rng) + Fr254::from(start_amount)
                    } else {
                        Fr254::from(0u8)
                    }
                })
                .collect::<Vec<Fr254>>();

            tree.insert_leaves(&leaves).unwrap();

            let circuit_insertion_info = tree.insert_for_circuit(&subtree_leaves).unwrap();

            IMTCircuitInsertionInfoVar::verify_subtree_insertion(
                &mut circuit,
                &circuit_insertion_info,
            )
            .unwrap();
            circuit.check_circuit_satisfiability(&[]).unwrap();
        }
    }
}
