//! Circuit implementation of poseidon hash function.

use ark_ff::{Field, PrimeField};
use ark_std::{boxed::Box, collections::BTreeMap, string::ToString, vec, vec::Vec, Zero};
use jf_relation::{
    constants::GATE_WIDTH, errors::CircuitError, gates::*, Circuit, PlonkCircuit, Variable,
};

use crate::{
    circuit::poseidon::sponge::{PoseidonStateVar, SpongePoseidonHashGadget},
    poseidon::{
        constants::PoseidonParams,
        sponge::{PoseidonSponge, CRHF_RATE},
        PoseidonPerm, STATE_SIZE,
    },
};

#[derive(Debug, Clone)]
/// Used in full Poseidon rounds
pub struct Power5NonLinearGate<F> {
    pub(crate) matrix_vector: Vec<F>,
    pub(crate) constant: F,
}

impl<F: PrimeField> Gate<F> for Power5NonLinearGate<F> {
    fn name(&self) -> &'static str {
        "Full round gate"
    }

    fn q_hash(&self) -> [F; GATE_WIDTH] {
        [
            self.matrix_vector[0],
            self.matrix_vector[1],
            self.matrix_vector[2],
            self.matrix_vector[3],
        ]
    }

    fn q_c(&self) -> F {
        self.constant
    }

    fn q_o(&self) -> F {
        F::one()
    }
}

#[derive(Debug, Clone)]
/// Used in the final round of Poseidon
pub struct Power5NoConstantGate<F> {
    pub(crate) matrix_vector: Vec<F>,
}

impl<F: PrimeField> Gate<F> for Power5NoConstantGate<F> {
    fn name(&self) -> &'static str {
        "Full round no constant gate"
    }

    fn q_hash(&self) -> [F; GATE_WIDTH] {
        [
            self.matrix_vector[0],
            self.matrix_vector[1],
            self.matrix_vector[2],
            self.matrix_vector[3],
        ]
    }

    fn q_o(&self) -> F {
        F::one()
    }
}

#[derive(Debug, Clone)]
/// Used in partial rounds of Poseidon
pub struct NonFullRoundGate<F> {
    pub(crate) matrix_vector: Vec<F>,
    pub(crate) constant: F,
}

impl<F: PrimeField> Gate<F> for NonFullRoundGate<F> {
    fn name(&self) -> &'static str {
        "Partial round gate"
    }
    fn q_lc(&self) -> [F; GATE_WIDTH] {
        [
            F::zero(),
            self.matrix_vector[1],
            self.matrix_vector[2],
            self.matrix_vector[3],
        ]
    }

    fn q_hash(&self) -> [F; GATE_WIDTH] {
        [self.matrix_vector[0], F::zero(), F::zero(), F::zero()]
    }

    fn q_c(&self) -> F {
        self.constant
    }

    fn q_o(&self) -> F {
        F::one()
    }
}

#[derive(Debug, Clone)]
/// Used in partial rounds of Poseidon
pub struct NonFullRoundNoConstantGate<F> {
    pub(crate) matrix_vector: Vec<F>,
}

impl<F: PrimeField> Gate<F> for NonFullRoundNoConstantGate<F> {
    fn name(&self) -> &'static str {
        "Partial round no constant gate"
    }
    fn q_lc(&self) -> [F; GATE_WIDTH] {
        [
            self.matrix_vector[0],
            self.matrix_vector[1],
            self.matrix_vector[2],
            self.matrix_vector[3],
        ]
    }

    fn q_o(&self) -> F {
        F::one()
    }
}

#[derive(Debug, Clone)]
/// Used to raise an element to the power of 5
pub struct PowerFiveGate;

impl<F: Field> Gate<F> for PowerFiveGate {
    fn name(&self) -> &'static str {
        "Power 5 Gate"
    }

    fn q_hash(&self) -> [F; GATE_WIDTH] {
        [F::one(), F::zero(), F::zero(), F::zero()]
    }

    fn q_o(&self) -> F {
        F::one()
    }
}

#[derive(Debug, Clone)]
/// Most general gate needed for poseidon
pub struct GeneralPoseidonGate<F> {
    pub(crate) linear_coeff_vector: Vec<F>,
    pub(crate) power_coeff_vector: Vec<F>,
    pub(crate) constant: F,
}

impl<F: PrimeField> Gate<F> for GeneralPoseidonGate<F> {
    fn name(&self) -> &'static str {
        "General Poseidon Gate"
    }

    fn q_lc(&self) -> [F; GATE_WIDTH] {
        [
            self.linear_coeff_vector[0],
            self.linear_coeff_vector[1],
            self.linear_coeff_vector[2],
            self.linear_coeff_vector[3],
        ]
    }

    fn q_hash(&self) -> [F; GATE_WIDTH] {
        [
            self.power_coeff_vector[0],
            self.power_coeff_vector[1],
            self.power_coeff_vector[2],
            self.power_coeff_vector[3],
        ]
    }

    fn q_c(&self) -> F {
        self.constant
    }

    fn q_o(&self) -> F {
        F::one()
    }
}

#[derive(Copy, Clone)]
/// Used to store a `Variable` and its corresponding linear and power coefficients in a potential linear combination.
pub struct VariableCoefficents<F> {
    pub(crate) variable: Variable,
    pub(crate) linear_coeff: F,
    pub(crate) power_coeff: F,
}

#[derive(Clone)]
/// Used to store a several `VariableCoefficients` and a constant in a linear combination.
pub struct LinearCombination<F> {
    pub(crate) var_coeffs_vec: Vec<VariableCoefficents<F>>,
    pub(crate) constant: F,
}

/// Used to store a single `Variable` and a set of `LinearCombination`s. This is used in the `partial_round_relations_transformation` method.
pub struct PartialRoundLinearCombinations<F> {
    pub(crate) variable: Variable,
    pub(crate) lin_combs: Vec<LinearCombination<F>>,
}

fn dot_product<F: PrimeField>(vec_one: &[F], vec_two: &[F]) -> F {
    vec_one
        .iter()
        .zip(vec_two.iter())
        .fold(F::zero(), |acc, (x, y)| acc + (*x * *y))
}

/*fn non_linear<F: PrimeField>(input: &[F], matrix: &[Vec<F>], constant: &[F]) -> Vec<F> {
    let input = input.iter().map(|x| x.pow([5u64])).collect::<Vec<F>>();
    let input_length = input.len();

    let mut output = vec![F::zero(); input_length];

    for (o, matrix_row, c) in izip!(output.iter_mut(), matrix.iter(), constant.iter()) {
        *o = dot_product(&input, matrix_row) + c;
    }
    output
}*/

fn combine_linear_combinations<F: PrimeField>(
    lin_combs: &[LinearCombination<F>],
    coeffs_vec: &[F],
    constant: &F,
) -> LinearCombination<F> {
    let mut var_coeffs_map = BTreeMap::<Variable, (F, F)>::new();
    let mut constant_result = *constant;
    for (lin_comb, coeff) in lin_combs.iter().zip(coeffs_vec.iter()) {
        for var_coeffs in lin_comb.var_coeffs_vec.iter() {
            let entry = var_coeffs_map.entry(var_coeffs.variable);
            entry
                .and_modify(|(x, y)| {
                    *x += *coeff * var_coeffs.linear_coeff;
                    *y += *coeff * var_coeffs.power_coeff;
                })
                .or_insert((
                    *coeff * var_coeffs.linear_coeff,
                    *coeff * var_coeffs.power_coeff,
                ));
        }
        constant_result += *coeff * lin_comb.constant;
    }

    let mut var_coeffs_result = Vec::<VariableCoefficents<F>>::new();
    for (variable, (linear_coeff, power_coeff)) in var_coeffs_map {
        var_coeffs_result.push(VariableCoefficents {
            variable,
            linear_coeff,
            power_coeff,
        });
    }
    LinearCombination {
        var_coeffs_vec: var_coeffs_result,
        constant: constant_result,
    }
}

/// Given the length of the input vector to the Poseidon hash and the number of partial rounds `n` we want to perform,
/// this function outputs the decomposition of `n` into the optimal batching of `n` partial rounds.
/// The output `(x, v)` means we batch into rounds of `x` and then add on the elements of v. In particular,
/// `n - sum_v` is divisible by `x`.
fn optimal_partial_rounds_batching(
    input_len: usize,
    num_rounds: usize,
) -> Result<(usize, Vec<usize>), CircuitError> {
    match input_len {
        2 => {
            if num_rounds % 3 == 0 {
                Ok((3, [].to_vec()))
            } else {
                Ok((3, [num_rounds % 3].to_vec()))
            }
        },
        3 => {
            if num_rounds % 2 == 0 {
                Ok((2, [].to_vec()))
            } else {
                Ok((2, [1].to_vec()))
            }
        },
        4 => match num_rounds % 4 {
            0 => Ok((4, [].to_vec())),
            2 => Ok((4, [1, 1].to_vec())),
            _ => Ok((4, [num_rounds % 3].to_vec())),
        },
        5 => {
            if num_rounds % 6 == 0 {
                Ok((6, [].to_vec()))
            } else if num_rounds < 6 {
                Ok((6, [num_rounds].to_vec()))
            } else if [1, 2].contains(&(num_rounds % 6)) {
                Ok((6, [3, (num_rounds % 6) + 3].to_vec()))
            } else {
                Ok((6, [num_rounds % 6].to_vec()))
            }
        },
        6 => {
            if num_rounds % 5 == 0 {
                Ok((5, [].to_vec()))
            } else if num_rounds % 5 == 4 {
                Ok((5, [4].to_vec()))
            } else if num_rounds < 5 {
                Ok((5, [num_rounds].to_vec()))
            } else {
                Ok((5, [(num_rounds % 5) + 5].to_vec()))
            }
        },
        7 => {
            if num_rounds % 7 == 0 {
                Ok((7, [].to_vec()))
            } else if num_rounds < 4 {
                Ok((7, [num_rounds].to_vec()))
            } else if num_rounds % 7 < 4 {
                Ok((7, [(num_rounds % 7) + 7].to_vec()))
            } else {
                Ok((7, [num_rounds % 7].to_vec()))
            }
        },
        _ => Err(CircuitError::InternalError(
            "Invalid input length".to_string(),
        )),
    }
}

/// Used to perform a Poseidon hash inside a circuit with optimised gates.
pub trait PoseidonHashGadget<F: PoseidonParams> {
    /// Decompose a (potentially large) linear relationship into a number of general Poseidon gates and insert into the circuit
    fn decompose_into_general_poseidon_gates(
        &mut self,
        lin_comb: &LinearCombination<F>,
    ) -> Result<Variable, CircuitError>;

    /// Perform a full round of Poseidon hash inside a circuit.
    fn full_round_var(
        &mut self,
        state_vec: &[Variable],
        matrix: &[Vec<F>],
        round_constants: &[F],
    ) -> Result<Vec<Variable>, CircuitError>;

    /// Transforms a set of linear combinations of `Variable`s into the appropriate linear combinations after one partial round.
    fn partial_round_relations_transformation(
        &mut self,
        partial_round_lin_comb: &PartialRoundLinearCombinations<F>,
        matrix: &[Vec<F>],
        round_constants: &[F],
    ) -> Result<PartialRoundLinearCombinations<F>, CircuitError>;

    /// Perform a composition of partial rounds of Poseidon hash inside a circuit.
    fn composed_partial_round_var(
        &mut self,
        state_vec: &[Variable],
        matrix: &[Vec<F>],
        constants: &[Vec<F>],
        num_rounds: usize,
    ) -> Result<Vec<Variable>, CircuitError>;

    /// Raise a variable to the power of 5 inside a circuit.
    fn power_of_five(&mut self, var: Variable) -> Result<Variable, CircuitError>;

    /// Perform a Poseidon permutation inside a circuit.
    fn poseidon_perm(&mut self, inputs: &[Variable]) -> Result<Vec<Variable>, CircuitError>;

    /// Perform a Poseidon hash inside a circuit.
    fn poseidon_hash(&mut self, inputs: &[Variable]) -> Result<usize, CircuitError>;

    /// Performs a tree hash, where if both inputs are zero it outputs zero.
    fn tree_hash(&mut self, inputs: &[Variable; 2]) -> Result<Variable, CircuitError>;
}

impl<F: PoseidonParams> PoseidonHashGadget<F> for PlonkCircuit<F> {
    fn decompose_into_general_poseidon_gates(
        &mut self,
        lin_comb: &LinearCombination<F>,
    ) -> Result<Variable, CircuitError> {
        // We first add a general Poseidon gate to deal with the first 4 variables
        let mut wire_vars = [Variable::default(); 5];
        let mut val_vec = Vec::<F>::new();
        let mut power_val_vec = Vec::<F>::new();
        let mut linear_coeff_vector = Vec::<F>::new();
        let mut power_coeff_vector = Vec::<F>::new();
        for (i, var_coeffs) in lin_comb.var_coeffs_vec.iter().take(4).enumerate() {
            wire_vars[i] = var_coeffs.variable;
            let value = self.witness(var_coeffs.variable)?;
            val_vec.push(value);
            power_val_vec.push(value.pow([5u64]));
            linear_coeff_vector.push(var_coeffs.linear_coeff);
            power_coeff_vector.push(var_coeffs.power_coeff);
        }

        let output_val = dot_product(&val_vec, &linear_coeff_vector)
            + dot_product(&power_val_vec, &power_coeff_vector)
            + lin_comb.constant;
        let output_var = self.create_variable(output_val)?;
        wire_vars[4] = output_var;

        linear_coeff_vector.resize(4, F::zero());
        power_coeff_vector.resize(4, F::zero());

        self.insert_gate(
            &wire_vars,
            Box::new(GeneralPoseidonGate::<F> {
                linear_coeff_vector,
                power_coeff_vector,
                constant: lin_comb.constant,
            }),
        )?;

        // If we have fewer than or equal to 4 input `Variable`s we return the output `Variable`,
        // otherwise we recurse.
        if lin_comb.var_coeffs_vec.len() <= 4 {
            Ok(output_var)
        } else {
            let mut new_var_coeffs_vec = lin_comb.var_coeffs_vec[4..].to_vec();
            new_var_coeffs_vec.push(VariableCoefficents {
                variable: output_var,
                linear_coeff: F::one(),
                power_coeff: F::zero(),
            });

            self.decompose_into_general_poseidon_gates(&LinearCombination {
                var_coeffs_vec: new_var_coeffs_vec,
                constant: F::zero(),
            })
        }
    }

    fn full_round_var(
        &mut self,
        state_vec: &[Variable],
        matrix: &[Vec<F>],
        round_constants: &[F],
    ) -> Result<Vec<Variable>, CircuitError> {
        self.check_vars_bound(state_vec)?;

        // We first create a vector of `LinearCombination`s out of the `Variable`s to the power 5.
        let var_coeffs_vec = state_vec
            .iter()
            .map(|var| VariableCoefficents {
                variable: *var,
                linear_coeff: F::zero(),
                power_coeff: F::one(),
            })
            .collect::<Vec<VariableCoefficents<F>>>();
        let lin_comb_vec = var_coeffs_vec
            .iter()
            .map(|var_coeffs| LinearCombination {
                var_coeffs_vec: [*var_coeffs].to_vec(),
                constant: F::zero(),
            })
            .collect::<Vec<LinearCombination<F>>>();

        // Next we construct all the ouput `LinearCombination`s.
        let lin_comb_output_vec = matrix
            .iter()
            .zip(round_constants.iter())
            .map(|(matrix_row, constant)| {
                combine_linear_combinations(&lin_comb_vec, matrix_row, constant)
            })
            .collect::<Vec<LinearCombination<F>>>();

        // We now convert our `LinearCombination`s back into `Variable`s.
        lin_comb_output_vec
            .iter()
            .map(|lin_comb| self.decompose_into_general_poseidon_gates(lin_comb))
            .collect::<Result<Vec<Variable>, CircuitError>>()
    }

    fn partial_round_relations_transformation(
        &mut self,
        partial_round_lin_comb: &PartialRoundLinearCombinations<F>,
        matrix: &[Vec<F>],
        round_constants: &[F],
    ) -> Result<PartialRoundLinearCombinations<F>, CircuitError> {
        // We first create a `LinearCombination` out of `partial_round_lin_comb.variable` to the power 5,
        // so we can then combine it with the other `LinearCombination`s.
        let mut lin_comb_vec = Vec::<LinearCombination<F>>::new();
        lin_comb_vec.push(LinearCombination {
            var_coeffs_vec: [VariableCoefficents {
                variable: partial_round_lin_comb.variable,
                linear_coeff: F::zero(),
                power_coeff: F::one(),
            }]
            .to_vec(),
            constant: F::zero(),
        });
        // We now append all the other `LinearCombination`s.
        lin_comb_vec.extend(partial_round_lin_comb.lin_combs.clone());

        // Next we construct all the ouput `LinearCombination`s.
        let lin_comb_output_vec = matrix
            .iter()
            .zip(round_constants.iter())
            .map(|(matrix_row, constant)| {
                combine_linear_combinations(&lin_comb_vec, matrix_row, constant)
            })
            .collect::<Vec<LinearCombination<F>>>();

        // We need to create a `Variable` for the first output `LinearCombination`.
        let first_var = self.decompose_into_general_poseidon_gates(&lin_comb_output_vec[0])?;
        Ok(PartialRoundLinearCombinations {
            variable: first_var,
            lin_combs: lin_comb_output_vec[1..].to_vec(),
        })
    }

    fn composed_partial_round_var(
        &mut self,
        state_vec: &[Variable],
        matrix: &[Vec<F>],
        constants: &[Vec<F>],
        num_rounds: usize,
    ) -> Result<Vec<Variable>, CircuitError> {
        self.check_vars_bound(state_vec)?;
        if constants.len() != num_rounds {
            return Err(CircuitError::ParameterError(
                "constants matrix, wrong size".to_string(),
            ));
        }
        // We first transform our vector of `Variable`s into a `PartialRoundLinearCombinations`.
        let first_var = state_vec[0];
        let lin_comb_vec = state_vec
            .iter()
            .skip(1)
            .map(|var| LinearCombination {
                var_coeffs_vec: [VariableCoefficents {
                    variable: *var,
                    linear_coeff: F::one(),
                    power_coeff: F::zero(),
                }]
                .to_vec(),
                constant: F::zero(),
            })
            .collect::<Vec<LinearCombination<F>>>();
        let mut part_round_lin_combs = PartialRoundLinearCombinations {
            variable: first_var,
            lin_combs: lin_comb_vec,
        };
        // Perform the appropriate number of partial rounds.
        for constants_vec in constants.iter() {
            part_round_lin_combs = self.partial_round_relations_transformation(
                &part_round_lin_combs,
                matrix,
                constants_vec,
            )?;
        }
        // Convert the `PartialRoundLinearCombinations`s back into `Variables`.
        let mut output_vars = Vec::<Variable>::new();
        output_vars.push(part_round_lin_combs.variable);
        for lin_comb in part_round_lin_combs.lin_combs.iter() {
            output_vars.push(self.decompose_into_general_poseidon_gates(lin_comb)?);
        }
        Ok(output_vars)
    }

    fn power_of_five(&mut self, var: Variable) -> Result<Variable, CircuitError> {
        self.check_var_bound(var)?;
        let value = self.witness(var)?;
        let output = value.pow([5u64]);
        let out_var = self.create_variable(output)?;
        let wire_vars = &[var, 0, 0, 0, out_var];
        self.insert_gate(wire_vars, Box::new(PowerFiveGate))?;
        Ok(out_var)
    }

    fn poseidon_perm(&mut self, inputs: &[Variable]) -> Result<Vec<Variable>, CircuitError> {
        let t = inputs.len();
        let (constants, matrix, n_rounds_p) = F::params(t)
            .map_err(|_| CircuitError::InternalError("Couldn't get Poseidon params".to_string()))?;
        let n_rounds_f = 8;

        let mut state_vec_var = inputs.to_vec();

        for loop_val in state_vec_var.iter_mut().enumerate() {
            let (j, element): (usize, &mut usize) = loop_val;
            *element = self.add_constant(*element, &constants[0][j])?;
        }

        // For n in the range 1..(n_rounds_f/2 + 1) we perform a full round using constants[n].
        for constant in constants.iter().skip(1).take(n_rounds_f / 2) {
            state_vec_var = self.full_round_var(&state_vec_var, &matrix, constant)?;
        }

        // For n in the range (n_rounds_f/2 + 1)..(n_rounds_f/2 + n_rounds_p + 1) we perform a partial round using constants[n].
        // We use the `optimal_partial_rounds_batching` to optimally batch together multiple partial rounds.
        let (x, rem_vec) = optimal_partial_rounds_batching(t, n_rounds_p)?;
        let mut n = (n_rounds_f / 2) + 1;
        for r in rem_vec {
            state_vec_var =
                self.composed_partial_round_var(&state_vec_var, &matrix, &constants[n..n + r], r)?;
            n += r;
        }
        // Now we set `num_x_rounds` to be n_rounds_p - sum_{rem_vec}.
        let num_x_rounds = (n_rounds_p - n + (n_rounds_f / 2) + 1) / x;
        for _ in 0..num_x_rounds {
            state_vec_var =
                self.composed_partial_round_var(&state_vec_var, &matrix, &constants[n..n + x], x)?;
            n += x;
        }

        // For n in the range (n_rounds_f/2 + n_rounds_p + 1).. we perform a full round using constants[n].
        for constant in constants.iter().skip(n_rounds_f / 2 + n_rounds_p + 1) {
            let tmp = self.full_round_var(&state_vec_var, &matrix, constant)?;
            state_vec_var = tmp;
        }

        let zero_vec = vec![F::zero(); t];
        state_vec_var = self.full_round_var(&state_vec_var, &matrix, &zero_vec)?;

        Ok(state_vec_var)
    }

    fn poseidon_hash(&mut self, inputs: &[Variable]) -> Result<Variable, CircuitError> {
        let t = inputs.len() + 1;
        let mut state_vec_var = vec![usize::zero(); t];
        state_vec_var[1..].clone_from_slice(inputs);

        let state_vec_var = self.poseidon_perm(&state_vec_var)?;
        Ok(state_vec_var[0])
    }

    fn tree_hash(&mut self, inputs: &[Variable; 2]) -> Result<Variable, CircuitError> {
        let check_one = self.is_zero(inputs[0])?;
        let check_two = self.is_zero(inputs[1])?;
        let and = self.logic_and(check_one, check_two)?;
        let hash = self.poseidon_hash(inputs)?;
        self.conditional_select(and, hash, self.zero())
    }
}

/// Hash an arbitrary‑length slice of field elements with Poseidon.
/// Returns `out_len` digest field elements.
///
/// * `input`     – message as field elements
/// * `out_len`   – how many elements to squeeze (usually 1)
///
/// Follows the sponge rules in the Poseidon paper (§2.1, §4.2).
pub fn poseidon_hash_varlen<F>(
    circuit: &mut PlonkCircuit<F>,
    input: &[Variable],
    out_len: usize,
) -> Result<Vec<Variable>, CircuitError>
where
    F: PoseidonParams,
{
    //const CRHF_RATE: usize = 4;

    // ---------- 1. fresh zero state ----------
    let mut state = PoseidonStateVar([circuit.zero(); STATE_SIZE]);

    // ---------- 2. inject domain‑separation tag ----------
    // Variable‑length hash, 1 output  ⇒  tag = 2^64
    //let tag = circuit.create_variable(F::from(1u128 << 64))?;
    //state.0[CRHF_RATE] = tag; // write into *first* capacity word

    // ---------- 3. build padded message ----------
    let mut buf: Vec<Variable> = Vec::with_capacity(input.len()); //+ CRHF_RATE);
    buf.extend_from_slice(input);

    // 10* padding: a single '1' then zeros so that |buf| ≡ 0 (mod RATE)
    /*buf.push(circuit.one());
    let rem = (CRHF_RATE - (buf.len() % CRHF_RATE)) % CRHF_RATE;
    for _ in 0..rem {
        buf.push(circuit.zero());
    }*/

    // ---------- 4. absorb full (message || padding) ----------
    state = circuit.absorb(&state, &buf)?;

    // ---------- 5. squeeze requested number of elements ----------
    circuit.squeeze(&state, out_len)
}

/// Hash an arbitrary-length slice of field elements with the Poseidon sponge (native implementation).
/// Returns `out_len` digest field elements.
pub fn poseidon_hash_varlen_native<
    F: PrimeField + PoseidonParams + ark_crypto_primitives::sponge::Absorb,
>(
    input: &Vec<F>,
    out_len: usize,
) -> Vec<F> {
    use ark_crypto_primitives::sponge::CryptographicSponge;
    use ark_crypto_primitives::sponge::FieldBasedCryptographicSponge;

    // Initialize the Poseidon permutation and sponge
    let sponge_perm = PoseidonPerm::<F>::perm().expect("Invalid parameters");
    let mut sponge = PoseidonSponge::<F, CRHF_RATE>::new(&sponge_perm);

    // Absorb all input elements
    sponge.absorb(input);

    // Squeeze out_len elements and return
    sponge.squeeze_native_field_elements(out_len)
}

#[cfg(test)]
mod tests {
    use std::{dbg, println};

    use crate::{
        poseidon::{FieldHasher, Poseidon},
        trees::TreeHasher,
    };

    use super::*;
    use ark_bn254::Fr as Fr254;
    use ark_std::{test_rng, UniformRand};
    use itertools::izip;

    use jf_relation::Circuit;

    fn non_linear_partial<F: PrimeField>(input: &[F], matrix: &[Vec<F>], constant: &[F]) -> Vec<F> {
        let index_one = input.first().unwrap().pow([5u64]);

        let mut state = Vec::<F>::new();
        state.push(index_one);

        input.iter().skip(1).for_each(|x| state.push(*x));

        let input_length = input.len();

        let mut output = vec![F::zero(); input_length];

        for (o, matrix_row, c) in izip!(output.iter_mut(), matrix.iter(), constant.iter()) {
            *o = dot_product(&state, matrix_row) + c;
        }
        output
    }

    #[test]
    //the purpose of this test is to compare the `partial_round_var` and `composed_partial_round_var` functions.
    fn test_partial_round_var() -> Result<(), CircuitError> {
        let mut rng = test_rng();
        for m in 2..8 {
            let big_arr: Vec<Fr254> = (0..m)
                .map(|_| Fr254::rand(&mut rng))
                .collect::<Vec<Fr254>>();
            for n in 1..20 {
                let (constants, matrix, _) = Fr254::params(m).map_err(|_| {
                    CircuitError::InternalError("Couldn't get Poseidon params".to_string())
                })?;
                let trunc_constants = constants[0..n].to_vec();

                let mut output_val_vec = big_arr.clone();
                for row_constant in &trunc_constants {
                    output_val_vec = non_linear_partial(&output_val_vec, &matrix, row_constant);
                }

                let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_turbo_plonk();
                let big_arr_var = big_arr
                    .iter()
                    .map(|&x| circuit.create_variable(x).unwrap())
                    .collect::<Vec<_>>();

                let composed_part_round_var = circuit.composed_partial_round_var(
                    &big_arr_var,
                    &matrix,
                    &trunc_constants,
                    n,
                )?;

                circuit.check_circuit_satisfiability(&[])?;
                for (var, val) in composed_part_round_var.iter().zip(output_val_vec.iter()) {
                    assert_eq!(circuit.witness(*var)?, *val);
                }
            }
        }
        Ok(())
    }

    #[test]
    //the purpose of this test is to compare the hash result from plonk circuit with the hash
    //result from the primitive poseidon hash
    fn test_poseidon_hash_plonk_gadget() -> Result<(), CircuitError> {
        let mut rng = test_rng();
        for m in 1..7 {
            let big_arr: Vec<Fr254> = (0..m)
                .map(|_| Fr254::rand(&mut rng))
                .collect::<Vec<Fr254>>();
            let poseidon = Poseidon::<Fr254>::new();
            let expected_hash = poseidon.hash(&big_arr).unwrap();

            let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_turbo_plonk();
            let big_arr_var = big_arr
                .iter()
                .map(|&x| circuit.create_variable(x).unwrap())
                .collect::<Vec<_>>();

            let hash_plonk_var = circuit.poseidon_hash(&big_arr_var).unwrap();

            circuit.check_circuit_satisfiability(&[])?;

            let hash_plonk = circuit.witness(hash_plonk_var).unwrap();
            assert_eq!(hash_plonk.to_string(), expected_hash.to_string());
        }
        Ok(())
    }

    #[test]
    //the purpose of this test is to compare the hash result from plonk circuit with the hash
    //result from the primitive poseidon hash
    fn test_large_poseidon_hash() -> Result<(), CircuitError> {
        let mut rng = test_rng();

        let big_arr: Vec<Fr254> = (0..140)
            .map(|_| Fr254::rand(&mut rng))
            .collect::<Vec<Fr254>>();
        let poseidon = Poseidon::<Fr254>::new();

        //let expected_hash = poseidon.hash(&big_arr).unwrap();
        let expected_hash = poseidon_hash_varlen_native(&big_arr, 1);

        let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(16);
        let big_arr_var = big_arr
            .iter()
            .map(|&x| circuit.create_variable(x).unwrap())
            .collect::<Vec<_>>();

        let hash_plonk_var = poseidon_hash_varlen(&mut circuit, &big_arr_var, 1).unwrap();

        println!("num gates: {}", circuit.num_gates());
        circuit.check_circuit_satisfiability(&[])?;

        let hash_plonk = circuit.witness(hash_plonk_var[0]).unwrap();

        let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(16);
        let big_arr_var = big_arr
            .iter()
            .map(|&x| circuit.create_variable(x).unwrap())
            .collect::<Vec<_>>();

        dbg!("hash_plonk: ", &hash_plonk);
        dbg!("expected_hash: ", &expected_hash);
        assert_eq!(hash_plonk.to_string(), expected_hash[0].to_string());

        Ok(())
    }

    #[test]
    // This test builds a Merkle tree over 140 Fr254 leaves using Poseidon as the compression function,
    // then compares the plaintext Merkle root with the one computed inside a Plonk circuit.
    fn test_merkle_tree_poseidon_hash() -> Result<(), CircuitError> {
        let mut rng = test_rng();
        let num_leaves = 140 as u32;
        // Generate random leaves
        let mut leaves: Vec<Fr254> = (0..num_leaves).map(|i| Fr254::from(i)).collect();

        // Compute the next power of two for padding
        let pow2 = num_leaves.next_power_of_two();
        // Pad with zeros if needed
        leaves.resize(pow2 as usize, Fr254::zero());

        // Plaintext Merkle tree hashing function
        fn plaintext_tree_hash(poseidon: &Poseidon<Fr254>, leaves: &[Fr254]) -> Fr254 {
            let mut layer = leaves.to_vec();
            while layer.len() > 1 {
                let mut next = Vec::with_capacity((layer.len() + 1) / 2);
                for pair in layer.chunks(2) {
                    let left = pair[0];
                    let right = if pair.len() == 2 {
                        pair[1]
                    } else {
                        Fr254::zero()
                    };
                    next.push(poseidon.tree_hash(&[left, right]).unwrap());
                }
                dbg!("next layer: {:?}", &next);
                layer = next;
            }
            layer[0]
        }

        let poseidon = Poseidon::<Fr254>::new();
        let expected_root = plaintext_tree_hash(&poseidon, &leaves);

        // Build the circuit
        let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(16);
        // Allocate leaf variables and pad with zero variables
        let mut leaf_vars: Vec<Variable> = leaves
            .iter()
            .map(|&x| circuit.create_variable(x).unwrap())
            .collect();

        // Recursively build Merkle tree inside the circuit
        while leaf_vars.len() > 1 {
            let mut next_vars = Vec::with_capacity((leaf_vars.len() + 1) / 2);
            for pair in leaf_vars.chunks(2) {
                let left_var = pair[0];
                let right_var = if pair.len() == 2 {
                    pair[1]
                } else {
                    // pad with zero
                    circuit.zero()
                };
                // Use the provided tree_hash method to compress
                let parent = circuit.tree_hash(&[left_var, right_var]).unwrap();
                next_vars.push(parent);
            }
            dbg!(
                "next layer 2: {:?}",
                &next_vars
                    .iter()
                    .map(|v| circuit.witness(*v).unwrap())
                    .collect::<Vec<_>>()
            );
            leaf_vars = next_vars;
        }

        let root_var = leaf_vars[0];
        println!("num gates: {}", circuit.num_gates());
        circuit.check_circuit_satisfiability(&[])?;

        // Extract the witness root
        let root_plonk = circuit.witness(root_var).unwrap();

        // Compare
        assert_eq!(root_plonk, expected_root, "Merkle root mismatch");
        Ok(())
    }
}
