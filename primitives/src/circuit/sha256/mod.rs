//! Circuit implementation of keccak hash function.

use crate::{alloc::string::ToString, sha256::Sha256Params};
use ark_ff::PrimeField;
use ark_std::{vec, vec::Vec, Zero};
use jf_relation::{errors::CircuitError, BoolVar, Circuit, PlonkCircuit, Variable};
use num_bigint::BigUint;
use num_traits::{Pow, ToPrimitive};

// Returns 2^n in the field.
pub(crate) fn power_two_prime_field<F: PrimeField>(n: usize) -> F {
    if n == 0 {
        F::one()
    } else if n % 2 == 0 {
        let x: F = power_two_prime_field(n / 2);
        return x * x;
    } else {
        let x: F = power_two_prime_field(n / 2);
        return F::from(2_u32) * x * x;
    }
}

// Outline of method for computing the sha256 circuit:
//
// Sha256 generally deals with states of vectors of 32 bit elements.
// We encode such an element [a_0, ..., a_{31}] as the field element
// sum_{i=0}^{31} 2^{31 - i} * a_i, i.e. in big-endian form.
// However, it will be most useful to pass around the corresponding
// element sum_{i=0}^{31} 4^{31 - i} * a_i. We call the function from
// the first of these field elements to the second the 'spread'
// function. The reason for this is best seen in the circuits for the
// 'choice' and 'majority' operations. See
// https://zcash.github.io/halo2/design/gadgets/sha256/table16.html
// for a more detailed explanation of this methodology.
//
// To enforce that field elements are indeed of one of the two above
// forms, we make use of a lookup table. This has three columns. The
// first lists the integers val = 0, 1, ..., 2^{11}-1, the second
// lists spread(val) and the third lists floor(log_2(val)) + 1, which
// we take as 0 when val=0. This final column is essentially the number
// of digits needed to write val in binary. Its purpose is to avoid
// doing large range checks. So a single 32-bit element stored as
// field element x will often be passed around as a triple of field
// elements (x_0, x_1, x_2), where
// x = x_0 + 2^{11} * x_1 + 2^{22} * x_2 and each x_i is checked to
// appear in the first column of the loookup table. Sometimes we will
// check that x_2 < 2^{10} using the third column in the lookup table.
// Sometimes, however, this will not be needed, as it causes an error
// that will be caught elsewhere in the circuit. Simialrly we can
// store spread(x) as (spread(x_0), spread(x_1), spread(x_2)) with
// spread(x) = spread(x_0) + 2^{22} * spread(x_1) + 2^{44} * spread(x_2).
//
// Given an input of n field elements x_0, ..., x_{n-1}, we encode each
// x_i in a 256-bit chunk in big-endian form. The input to the hash
// function will then just be the concatenation of these n 256-bit
// chunks. We can, therefore, put n field elements into floor((n+1)/2)
// 512-bit "message blocks". (Note, if n is even, we need an extra
// message block to accomodate the "padding" bits.)

// Used to store a single 32-bit element as 3 field elements either in
// non-spread or spread form as described above.
type FieldChunksVar = [Variable; 3];
// Used to store a single 512-bit message block as 16 32-bit elements
// each represented as a field element in spread form.
type InitMessBlocksVar = [Variable; 16];
// Used to store the 64 32-bit output of the message scheduler. Each
// 32-bit element is represented as a field element in spread form.
type MessageScheduleVar = [Variable; 64];
// Used to store the 8 32-bit output of the compression function. Each
// 32-bit element is represented as a field element in spread form.
type CompOutputVar = [Variable; 8];

// Takes a non-spread BigUint element < 2^{11} and returns the spread version.
// Also returns the floor(log_2(input)) + 1. This is '0' if 'input = 0'.
// Returns 'None' if input is outside desired range.
// Takes in a BigUint and outputs an F, as this will be most useful.
fn field_to_spread_field<F: PrimeField>(non_spread_big_uint: &BigUint) -> Option<(F, u32)> {
    // We first check if input 'BigUint' is less than 2^{11}.
    if *non_spread_big_uint >= BigUint::from(2048_u32) {
        return None;
    }
    let mut spread_val = F::zero();
    let mut log_val = 0_u32;
    for i in 0..11 {
        let bit = (non_spread_big_uint / BigUint::from(1_u32 << i)) % BigUint::from(2_u8);
        if bit > BigUint::zero() {
            spread_val += F::from(1_u32 << (2 * i));
            log_val = i + 1;
        }
    }
    Some((spread_val, log_val))
}

// Essentially the inverse of 'field_to_spread_field'. Also returns the floor(log_2(input)) + 1.
// Returns 'None' if input has no inverse.
// Takes in a BigUint and outputs an F, as this will be most useful.
fn spread_field_to_field<F: PrimeField>(spread_big_uint: &BigUint) -> Option<(F, u32)> {
    // We first check if input 'BigUint' is less than 4^{11}.
    if *spread_big_uint >= BigUint::from(4194304_u32) {
        return None;
    }
    let mut non_spread_val = F::zero();
    let mut log_val = 0_u32;
    for i in 0..22 {
        let bit = (spread_big_uint / BigUint::from(1_u32 << i)) % BigUint::from(2_u8);
        if bit > BigUint::zero() {
            if i % 2 == 0 {
                non_spread_val += F::from(1_u32 << (i / 2));
                log_val = (i / 2) + 1;
            } else {
                return None;
            }
        }
    }
    Some((non_spread_val, log_val))
}

// Takes a non-spread element < 2^{32} and returns the spread version.
// Returns 'None' if input is outside desired range.
fn big_field_to_spread_field<F: PrimeField>(input: u32) -> F {
    let mut spread_val = 0u64;
    for i in 0..32 {
        if (input >> i) & 1 == 1 {
            spread_val += 1u64 << (2 * i);
        }
    }

    F::from(spread_val)
}

/// Used to perform a Sha256 hash inside a circuit.
pub trait Sha256HashGadget<F: Sha256Params> {
    /// Converts a 'Variable' representing a 'val' < 2^{32} into a pair of 'FieldChunksVar'.
    /// One represents 'val', the other 'spread(val)'.
    /// If 'range_check' is true, we do the range check 'val' < 2^{32}.
    fn non_spread_to_field_chunks(
        &mut self,
        var: &Variable,
        range_check: bool,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(FieldChunksVar, FieldChunksVar), CircuitError>;

    /// Converts a 'Variable' representing a 'spread(val)', with 'val' < 2^{32} into a pair of 'FieldChunksVar'.
    /// One represents 'val', the other 'spread(val)'.
    /// If 'range_check' is true, we do the range check 'val' < 2^{32}.
    fn spread_to_field_chunks(
        &mut self,
        var: &Variable,
        range_check: bool,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(FieldChunksVar, FieldChunksVar), CircuitError>;

    /// Given an input 'Variable' representing value 'val' < 2^{64}, this will constrain
    /// val = spread(val_even) + 2 * spread(val_odd)
    /// where val_even, val_odd < 2^{32}.
    /// The lookup table will be used to check the appropriate spreads.
    /// The output will be 'FieldChunksVar's representing (spread(val_even), spread(val_odd))
    /// or (val_even, val_odd) depending on 'spread' being 'true' or 'false'.
    fn constrain_spread_decomp(
        &mut self,
        var: &Variable,
        spread: bool,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(FieldChunksVar, FieldChunksVar), CircuitError>;

    /// Takes in a vector of 'Variable's and a 'constant'. Returns a 'Variable' representing the sum modulo 2^{32}.
    /// Note the 'Variable's represent elements in spread form but 'constant' will be given in non-spread form.
    fn mod_2_32_add(
        &mut self,
        vars: &[Variable],
        constant: &F,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<Variable, CircuitError>;

    /// Performs the choice operation on 3 32-bit elements represented as spread field elements.
    fn choice(
        &mut self,
        vars: &[Variable; 3],
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<Variable, CircuitError>;

    /// Performs the majority operation on 3 32-bit elements represented as spread field elements.
    fn majority(
        &mut self,
        vars: &[Variable; 3],
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<Variable, CircuitError>;

    /// Given an input 'Variable' representing value 'spread(val)', where 'val < 2^{32}', this will constrain
    /// spread(val) = spread(val_0) + 2^{2*rots[0]} * spread(val_1) + 2^{2*rots[1]} * spread(val_2) + 2^{2*rots[2]} * spread(val_3).
    /// Must have n_i > 0, for i=0,1,2,3, where n_0 = rots[0], n_i = rots[i] - rots[i-1], for i=1,2 and n_3 = 32 - rots[2].
    /// Also each val_i < 2^{n_i}.
    /// Returns 'Variable's representing '[spread(val_0), spread(val_1), spread(val_2), spread(val_3)]'.
    /// If this is going to be used for a σ_i rather than a Σ_i we need to range check val_0. (We set 'first_shift = true'.)
    /// This is not necessary for Σ_i, as any val_i that's too large will be caught when peforming Σ_i itself.
    fn split_for_rot(
        &mut self,
        var: &Variable,
        rots: &[u32; 3],
        first_shift: bool,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<[Variable; 4], CircuitError>;

    /// Right rotates by rots[0], rots[1] and rot[2] and outputs a 'Variable' XORing all three together.
    /// If 'first_shift = true', we shift by rots[0] instead of rotating.
    fn rot_sum(
        &mut self,
        var: &Variable,
        rots: &[u32; 3],
        first_shift: bool,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<Variable, CircuitError>;

    /// Given a vector of 'Variable's representing field elements, we output a vector of the appropriate number
    /// of 'InitMessBlocksVar's that represent the padded input into the 'prepare_message_schedule' function.
    fn preprocess(
        &mut self,
        input_vars: &[Variable],
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<Vec<InitMessBlocksVar>, CircuitError>;

    /// Given a vector of 'Variable's representing field elements and an "extra" bit, we output a vector of the appropriate
    /// number of 'InitMessBlocksVar's that represent the padded input into the 'prepare_message_schedule' function.
    fn preprocess_with_bit(
        &mut self,
        input_vars: &[Variable],
        bit_var: &BoolVar,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<Vec<InitMessBlocksVar>, CircuitError>;

    /// Given 16 'Variable's representing 32-bit elements, we return a
    /// 'MessageScheduleVar' representing the 64 element message schedule.
    fn prepare_message_schedule(
        &mut self,
        init_mess_blocks_var: &InitMessBlocksVar,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<MessageScheduleVar, CircuitError>;

    /// Given 'Variable's representing the 64 element message schedule and the output from the previous hash
    /// (or the initial H constants), we return a 'CompOutputVar' representing the output of sha256 compression function.
    fn iter_comp_func(
        &mut self,
        mess_sched_var: &MessageScheduleVar,
        comp_output_var: &CompOutputVar,
        k_constants: &[F; 64],
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<CompOutputVar, CircuitError>;

    /// Converts a 'CompOutputVar' into a pair of 'Variable's, one representing
    /// the lower 4 bits and one representing the upper 252 bits.
    fn hash_to_shifted_outputs(
        &mut self,
        comp_output_var: &CompOutputVar,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(Variable, Variable), CircuitError>;

    /// Given a vector of 'Variable's representing field elements,
    /// we return a pair of 'Variable's, one representing
    /// the lower 4 bits and one representing the upper 252 bits of
    /// the output of the sha256 hash.
    fn full_shifted_sha256_hash(
        &mut self,
        vars: &[Variable],
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(Variable, Variable), CircuitError>;

    /// Given a vector of 'Variable's representing field elements and an
    ///  "extra" bit, we concatenate the "extra" bit to the end of the final
    /// `Variable` and return a pair of 'Variable's, one representing
    /// the lower 4 bits and one representing the upper 252 bits of
    /// the output of the sha256 hash.
    fn full_shifted_sha256_hash_with_bit(
        &mut self,
        vars: &[Variable],
        bit_var: &BoolVar,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(Variable, Variable), CircuitError>;

    /// Generates the lookup table described at the beginning of the file.
    /// Inserts the table, along with the associated lookup 'Variable's, into the cicuit.
    /// Only call this function once all sha256 hashing is completed.
    fn finalize_for_sha256_hash(
        &mut self,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(), CircuitError>;
}

impl<F: Sha256Params> Sha256HashGadget<F> for PlonkCircuit<F> {
    fn non_spread_to_field_chunks(
        &mut self,
        var: &Variable,
        range_check: bool,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(FieldChunksVar, FieldChunksVar), CircuitError> {
        let big_uint_val: BigUint = self.witness(*var)?.into_bigint().into();

        let mut non_spread_chunks_var = [Variable::default(); 3];
        let mut spread_chunks_var = [Variable::default(); 3];

        for i in 0..3 {
            // 'chunk_val' will represent the next 11 least significant bits of 'big_uint_val'
            let chunk_val =
                (&big_uint_val / BigUint::from(2048_u32).pow(i as u32)) % BigUint::from(2048_u32);
            let non_spread_val =
                F::from(chunk_val.to_u32().ok_or(CircuitError::ParameterError(
                    "cannot convert BigUint into u32 as too big".to_string(),
                ))?);
            let (spread_val, log_val) = field_to_spread_field::<F>(&chunk_val).ok_or(
                CircuitError::ParameterError("invalid input to spread_field_to_field".to_string()),
            )?;
            let non_spread_var = self.create_variable(non_spread_val)?;
            let spread_var = self.create_variable(spread_val)?;
            let log_var = self.create_variable(F::from(log_val))?;
            lookup_vars.push((non_spread_var, spread_var, log_var));

            if range_check && i == 2 {
                // We check that 'log_var' represents a value less than or equal to 10.
                let shifted_log_var =
                    self.add_constant(log_var, &F::from(self.range_size()? as u32 - 11))?;
                self.add_range_check_variable(shifted_log_var)?;
            }

            non_spread_chunks_var[i as usize] = non_spread_var;
            spread_chunks_var[i as usize] = spread_var;
        }

        self.lin_comb_gate(
            // coefficients are [1, 2^{11}, 2^{22}]
            &[F::one(), F::from(2048_u64), F::from(4194304_u32)],
            &F::zero(),
            &non_spread_chunks_var,
            var,
        )?;
        Ok((non_spread_chunks_var, spread_chunks_var))
    }

    fn spread_to_field_chunks(
        &mut self,
        var: &Variable,
        range_check: bool,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(FieldChunksVar, FieldChunksVar), CircuitError> {
        let big_uint_val: BigUint = self.witness(*var)?.into_bigint().into();

        let mut non_spread_chunks_var = [Variable::default(); 3];
        let mut spread_chunks_var = [Variable::default(); 3];

        for i in 0..3 {
            // 'chunk_val' will represent the next 22 least significant bits of 'big_uint_val'
            let chunk_val = (&big_uint_val / BigUint::from(4194304_u32).pow(i as u32))
                % BigUint::from(4194304_u32);
            let spread_val = F::from(chunk_val.to_u32().ok_or(CircuitError::ParameterError(
                "cannot convert BigUint into u32 as too big".to_string(),
            ))?);
            let (non_spread_val, log_val) = spread_field_to_field::<F>(&chunk_val).ok_or(
                CircuitError::ParameterError("invalid input to spread_field_to_field".to_string()),
            )?;
            let non_spread_var = self.create_variable(non_spread_val)?;
            let spread_var = self.create_variable(spread_val)?;
            let log_var = self.create_variable(F::from(log_val))?;
            lookup_vars.push((non_spread_var, spread_var, log_var));

            if range_check && i == 2 {
                // We check that 'log_var' represents a value less than or equal to 10.
                let shifted_log_var =
                    self.add_constant(log_var, &F::from(self.range_size()? as u32 - 11))?;
                self.add_range_check_variable(shifted_log_var)?;
            }

            non_spread_chunks_var[i as usize] = non_spread_var;
            spread_chunks_var[i as usize] = spread_var;
        }

        self.lin_comb_gate(
            // coefficients are [1, 2^{22}, 2^{44}]
            &[F::one(), F::from(4194304_u32), F::from(17592186044416_u64)],
            &F::zero(),
            &spread_chunks_var,
            var,
        )?;
        Ok((non_spread_chunks_var, spread_chunks_var))
    }

    fn constrain_spread_decomp(
        &mut self,
        var: &Variable,
        spread: bool,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(FieldChunksVar, FieldChunksVar), CircuitError> {
        let val_big_uint: BigUint = self.witness(*var)?.into_bigint().into();
        // The values of the chunks represented by non-spread output
        let mut val_even = [F::zero(); 3];
        let mut val_odd = [F::zero(); 3];
        // The values of the chunks represented by spread output
        let mut spread_val_even = [F::zero(); 3];
        let mut spread_val_odd = [F::zero(); 3];
        // The log's of the chunks represented by non-spread output
        let mut log_val_even = [0_u32; 3];
        let mut log_val_odd = [0_u32; 3];
        // We calculate the values of the above elements
        for i in 0..64 {
            let j = i as usize / 22;
            let power = (i % 22) / 2;
            let bit = (&val_big_uint / BigUint::from(1_u64 << i)) % BigUint::from(2_u8);
            if bit > BigUint::zero() {
                if i % 2 == 0 {
                    val_even[j] += F::from(1_u32 << power);
                    spread_val_even[j] += F::from(1_u32 << (2 * power));
                    log_val_even[j] = power + 1;
                } else {
                    val_odd[j] += F::from(1_u32 << power);
                    spread_val_odd[j] += F::from(1_u32 << (2 * power));
                    log_val_odd[j] = power + 1;
                }
            }
        }

        let mut non_spread_vars_even = [Variable::default(); 3];
        let mut non_spread_vars_odd = [Variable::default(); 3];

        let mut spread_vars_even = [Variable::default(); 3];
        let mut spread_vars_odd = [Variable::default(); 3];
        // We create the appropriate 'Variable's, check them in the lookup table
        // and enforce the desired relation involving the original input 'var'.
        let mut coeffs = Vec::<F>::new();
        let mut spread_vars = Vec::<Variable>::new();
        for i in 0..3 {
            let var_even = self.create_variable(val_even[i])?;
            let var_odd = self.create_variable(val_odd[i])?;
            non_spread_vars_even[i] = var_even;
            non_spread_vars_odd[i] = var_odd;

            let spread_var_even = self.create_variable(spread_val_even[i])?;
            let spread_var_odd = self.create_variable(spread_val_odd[i])?;
            spread_vars_even[i] = spread_var_even;
            spread_vars_odd[i] = spread_var_odd;

            let log_var_even = self.create_variable(F::from(log_val_even[i]))?;
            let log_var_odd = self.create_variable(F::from(log_val_odd[i]))?;

            lookup_vars.push((var_even, spread_var_even, log_var_even));
            lookup_vars.push((var_odd, spread_var_odd, log_var_odd));

            // We push 2^{22i} and 2^{22i+1} to the 'coeffs' vector.
            coeffs.push(F::from(1_u128 << (22 * i)));
            coeffs.push(F::from(1_u128 << (22 * i + 1)));

            spread_vars.push(spread_var_even);
            spread_vars.push(spread_var_odd);
        }

        self.lin_comb_gate(&coeffs, &F::zero(), &spread_vars, var)?;

        if spread {
            Ok((spread_vars_even, spread_vars_odd))
        } else {
            Ok((non_spread_vars_even, non_spread_vars_odd))
        }
    }

    fn mod_2_32_add(
        &mut self,
        vars: &[Variable],
        constant: &F,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<Variable, CircuitError> {
        if vars.len() > 6 {
            return Err(CircuitError::ParameterError(
                "Can only add together up to 6 elements mod 2^{32}".to_string(),
            ));
        }
        // These 'FieldChunksVar's represent the non-spread version of the values represented by 'vars'.
        let mut non_spread_chunks_vars = Vec::<FieldChunksVar>::new();
        for var in vars {
            let (non_spread_chunks_var, _) =
                self.spread_to_field_chunks(var, false, lookup_vars)?;
            non_spread_chunks_vars.push(non_spread_chunks_var);
        }
        // The sum of all the values represented by the 'Variable's plus the constant term.
        let sum_big_uint: BigUint = non_spread_chunks_vars.iter().fold(
            constant.into_bigint().into(),
            |acc, [var_0, var_1, var_2]| {
                acc + self.witness(*var_0).unwrap().into_bigint().into()
                    + BigUint::from(2048_u32) * self.witness(*var_1).unwrap().into_bigint().into()
                    + BigUint::from(4194304_u32)
                        * self.witness(*var_2).unwrap().into_bigint().into()
            },
        );
        // The multiple of 2^{32} in 'sum_big_uint'. This will
        // be needed to constrain our mod 2^{32} relation.
        let overflow_big_uint: BigUint = sum_big_uint / BigUint::from(4294967296_u64);
        let overflow_var = self.create_variable(F::from(overflow_big_uint.to_u32().ok_or(
            CircuitError::ParameterError("cannot convert BigUint into u32 as too big".to_string()),
        )?))?;

        // 'overflow_big_uint' must be less than 2^{11}. (In practice it will actually be much smaller!)
        let (spread_overflow, log_overflow) = field_to_spread_field::<F>(&overflow_big_uint)
            .ok_or(CircuitError::ParameterError(
                "invalid input to field_to_spread_field".to_string(),
            ))?;

        let spread_overflow_var = self.create_variable(spread_overflow)?;
        let log_overflow_var = self.create_variable(F::from(log_overflow))?;
        lookup_vars.push((overflow_var, spread_overflow_var, log_overflow_var));

        // The coefficients of our mod 2^{32} relation are [1, 2^{11}, 2^{22}, ..., 1, 2^{11}, 2^{22}, -2^{32}]
        let mut coeffs = [F::one(), F::from(2048_u32), F::from(4194304_u32)]
            .into_iter()
            .cycle()
            .take(3 * vars.len())
            .collect::<Vec<F>>();
        coeffs.push(-F::from(4294967296_u64));
        // The 'Variable's of our mod 2^{32} relation.
        let mut non_spread_vars = non_spread_chunks_vars
            .into_iter()
            .flat_map(|non_spread_chunks_var| non_spread_chunks_var.to_vec())
            .collect::<Vec<Variable>>();
        non_spread_vars.push(overflow_var);
        // The mod 2^{32} relation.
        let non_spread_result_var = self.lin_comb(&coeffs, constant, &non_spread_vars)?;
        // We output the spread form of the output of the mod 2^{32} calculation.
        let (_, spread_result_chunks_var) =
            self.non_spread_to_field_chunks(&non_spread_result_var, true, lookup_vars)?;
        self.lin_comb(
            // coefficients are [1, 2^{22}, 2^{44}]
            &[F::one(), F::from(4194304_u32), F::from(17592186044416_u64)],
            &F::zero(),
            &spread_result_chunks_var,
        )
    }

    fn choice(
        &mut self,
        vars: &[Variable; 3],
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<Variable, CircuitError> {
        // The details of the choice operation are given in
        // the halo2 reference at the beginning of the file.
        let p_var = self.add(vars[0], vars[1])?;
        // 'even_const' is equal to sum_{i=0}^{31} 4^i
        let even_const = (F::from(18446744073709551616_u128) - F::one()) / F::from(3_u8);
        let q_var = self.lin_comb(&[-F::one(), F::one()], &even_const, &[vars[0], vars[2]])?;

        let (_, spread_p_odd_vars) = self.constrain_spread_decomp(&p_var, true, lookup_vars)?;
        let (_, spread_q_odd_vars) = self.constrain_spread_decomp(&q_var, true, lookup_vars)?;
        // coefficients are [1, 2^{22}, 2^{44}, 1, 2^{22}, 2^{44}]
        let coeffs = [
            F::one(),
            F::from(4194304_u32),
            F::from(17592186044416_u64),
            F::one(),
            F::from(4194304_u32),
            F::from(17592186044416_u64),
        ];
        let output_vars = [
            spread_p_odd_vars[0],
            spread_p_odd_vars[1],
            spread_p_odd_vars[2],
            spread_q_odd_vars[0],
            spread_q_odd_vars[1],
            spread_q_odd_vars[2],
        ];
        self.lin_comb(&coeffs, &F::zero(), &output_vars)
    }

    fn majority(
        &mut self,
        vars: &[Variable; 3],
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<Variable, CircuitError> {
        // The details of the 'majority' operation are given
        // in the halo2 reference at the beginning of the file.
        let m_var = self.lin_comb(&[F::one(); 3], &F::zero(), vars)?;

        let (_, spread_odd_var) = self.constrain_spread_decomp(&m_var, true, lookup_vars)?;

        self.lin_comb(
            // coefficients are [1, 2^{22}, 2^{44}]
            &[F::one(), F::from(4194304_u32), F::from(17592186044416_u64)],
            &F::zero(),
            &[spread_odd_var[0], spread_odd_var[1], spread_odd_var[2]],
        )
    }

    fn split_for_rot(
        &mut self,
        var: &Variable,
        rots: &[u32; 3],
        first_shift: bool,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<[Variable; 4], CircuitError> {
        let val_big_uint: BigUint = self.witness(*var)?.into_bigint().into();
        // We include 32 in our rotations for convenience
        let mut ext_rots = rots.to_vec();
        ext_rots.extend_from_slice(&[32]);
        let mut last_rot = 0_u32;
        let mut output_vars = [Variable::default(); 4];
        for (i, rot) in ext_rots.iter().enumerate() {
            // rot_diff is the size of the chunk of the input we want represented by 'output_var'
            let rot_diff = *rot as i32 - last_rot as i32;
            if rot_diff <= 0 || rot_diff >= 32 {
                return Err(CircuitError::ParameterError(
                    "rotation differences must be in (0,32)".to_string(),
                ));
            }
            // The value of the chunk in 'BigUint' form
            let mut chunk_big_uint = (&val_big_uint / BigUint::from(1_u64 << (2 * last_rot)))
                % BigUint::from(1_u64 << (2 * rot_diff));
            // We must divide our chunk into smaller chunks
            // of 11 in order to utilise our lookup table
            let mut coeffs = Vec::<F>::new();
            let mut spread_vars = Vec::<Variable>::new();
            for j in 0..((rot_diff - 1) / 11) + 1 {
                let spread_val = &chunk_big_uint % BigUint::from(1_u32 << 22);
                let spread_var = self.create_variable(F::from(spread_val.to_u32().ok_or(
                    CircuitError::ParameterError(
                        "cannot convert BigUint into u32 as too big".to_string(),
                    ),
                )?))?;
                spread_vars.push(spread_var);

                let (non_spread_val, log_val) =
                    spread_field_to_field(&spread_val).ok_or(CircuitError::ParameterError(
                        "invalid input to spread_field_to_field".to_string(),
                    ))?;
                let non_spread_var = self.create_variable(non_spread_val)?;
                let log_var = self.create_variable(F::from(log_val))?;
                lookup_vars.push((non_spread_var, spread_var, log_var));

                coeffs.push(F::from(1_u64 << (22 * j)));

                chunk_big_uint /= BigUint::from(1_u32 << 22);

                // We only need to range check the 'log_val' if we are shifting by rots[0] instead of rotating.
                // In this case we only need check the final 'log_val' of the first chunk.
                // If any of the other 'log_val's are "too large", this error will be caught in 'rot_sum'.
                if first_shift && i == 0 && j == (rot_diff - 1) / 11 {
                    let shifted_log_var =
                        self.add_constant(log_var, &F::from(self.range_size()? as u32 - 1 - *rot))?;
                    self.add_range_check_variable(shifted_log_var)?;
                }
            }
            let output_var = self.lin_comb(&coeffs, &F::zero(), &spread_vars)?;
            output_vars[i] = output_var;

            last_rot = *rot;
        }
        let mut coeffs = vec![F::one()];
        for rot in rots {
            coeffs.push(F::from(2u8).pow([*rot as u64 * 2]));
        }
        self.lin_comb_gate(&coeffs, &F::zero(), &output_vars, var)?;

        Ok(output_vars)
    }

    fn rot_sum(
        &mut self,
        var: &Variable,
        rots: &[u32; 3],
        first_shift: bool,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<Variable, CircuitError> {
        let split_vars = self.split_for_rot(var, rots, first_shift, lookup_vars)?;
        let bool_coeff = if first_shift { F::zero() } else { F::one() };
        // Right rotation/shift by 'rots[0]'
        let coeffs_0 = vec![
            F::one(),
            F::from(1_u64 << (2 * (rots[1] - rots[0]))),
            F::from(1_u64 << (2 * (rots[2] - rots[0]))),
            F::from(1_u64 << (2 * (32 - rots[0]))) * bool_coeff,
        ];
        let rot_var_0 = self.lin_comb(
            &coeffs_0,
            &F::zero(),
            &[split_vars[1], split_vars[2], split_vars[3], split_vars[0]],
        )?;
        // Right rotation by 'rots[1]'
        let coeffs_1 = vec![
            F::one(),
            F::from(1_u64 << (2 * (rots[2] - rots[1]))),
            F::from(1_u64 << (2 * (32 - rots[1]))),
            F::from(1_u64 << (2 * (32 + rots[0] - rots[1]))),
        ];
        let rot_var_1 = self.lin_comb(
            &coeffs_1,
            &F::zero(),
            &[split_vars[2], split_vars[3], split_vars[0], split_vars[1]],
        )?;
        // Right rotation by 'rots[2]'
        let coeffs_2 = vec![
            F::one(),
            F::from(1_u64 << (2 * (32 - rots[2]))),
            F::from(1_u64 << (2 * (32 + rots[0] - rots[2]))),
            F::from(1_u64 << (2 * (32 + rots[1] - rots[2]))),
        ];
        let rot_var_2 = self.lin_comb(
            &coeffs_2,
            &F::zero(),
            &[split_vars[3], split_vars[0], split_vars[1], split_vars[2]],
        )?;

        let sum_field_var = self.lin_comb(
            &[F::one(); 3],
            &F::zero(),
            &[rot_var_0, rot_var_1, rot_var_2],
        )?;
        let (spread_even_var, _) =
            self.constrain_spread_decomp(&sum_field_var, true, lookup_vars)?;
        // 'spread_even_var' records the XOR of the 3 'Variables' 'rot_var_0', 'rot_var_1' and 'rot_var_2'
        self.lin_comb(
            &[F::one(), F::from(4194304_u32), F::from(17592186044416_u64)],
            &F::zero(),
            &spread_even_var,
        )
    }

    fn preprocess(
        &mut self,
        input_vars: &[Variable],
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<Vec<InitMessBlocksVar>, CircuitError> {
        let n = input_vars.len();
        // The choice of 12 is actually pretty arbitrary. We can always change this at a later date.
        if n > 12 {
            return Err(CircuitError::ParameterError(
                "Can only hash together up to 12 field elements".to_string(),
            ));
        }
        // We must first express the inputs in spread form
        // This will hold 'Variables' representing the spread
        // 32-bit chunks of the inputs concatenated together
        let mut spread_field_vars = Vec::<Variable>::new();
        for input_var in input_vars {
            let big_uint: BigUint = self.witness(*input_var)?.into();
            // This stores the 'Variable's representing the non-spread 32-bit chunks of the input
            let mut non_spread_field_vars = Vec::<Variable>::new();
            let mut u32digits = big_uint.to_u32_digits();
            u32digits.resize(8, 0u32);
            for j in u32digits.into_iter().rev() {
                let field_var = self.create_variable(F::from(j))?;
                let (_, spread_field_chunks_var) =
                    self.non_spread_to_field_chunks(&field_var, true, lookup_vars)?;
                non_spread_field_vars.push(field_var);
                spread_field_vars.push(self.lin_comb(
                    &[F::one(), F::from(4194304_u32), F::from(17592186044416_u64)],
                    &F::zero(),
                    &spread_field_chunks_var,
                )?);
            }

            let coeffs = (0..8)
                .rev()
                .map(|i| F::from(BigUint::from(1u32) << (32 * i)))
                .collect::<Vec<F>>();
            // We must verify that the input is the appropriate linear combination of chunks
            self.lin_comb_gate(&coeffs, &F::zero(), &non_spread_field_vars, input_var)?;
        }
        // We put the spread 32-bit chunks into the message blocks in chunks of 16.
        // So each message block represents 2 field inputs.
        let mut init_mess_blocks_vars = Vec::<InitMessBlocksVar>::new();
        for spread_field_vec_var in spread_field_vars.chunks(16) {
            let mut init_mess_blocks_var = InitMessBlocksVar::default();
            for (i, spread_field_var) in spread_field_vec_var.iter().enumerate() {
                init_mess_blocks_var[i] = *spread_field_var;
            }
            init_mess_blocks_vars.push(init_mess_blocks_var);
        }
        // This is the length of the input needed for the padding of the message
        let spread_length_val = big_field_to_spread_field::<F>(256u32 * n as u32);
        // We put a 1 after all the inputs have been inserted into the message blocks.
        // The last few bits are then used to reprsent the length of the imput message.
        if n % 2 == 0 {
            let mut init_mess_blocks_var = InitMessBlocksVar::default();
            init_mess_blocks_var[0] =
                self.create_constant_variable(F::from(4611686018427387904_u64))?; // 2^{62}
            init_mess_blocks_var[15] = self.create_constant_variable(spread_length_val)?;
            init_mess_blocks_vars.push(init_mess_blocks_var);
        } else {
            init_mess_blocks_vars[n / 2][8] =
                self.create_constant_variable(F::from(4611686018427387904_u64))?; // 2^{62}
            init_mess_blocks_vars[n / 2][15] = self.create_constant_variable(spread_length_val)?;
        }

        Ok(init_mess_blocks_vars)
    }

    fn preprocess_with_bit(
        &mut self,
        input_vars: &[Variable],
        bit_var: &BoolVar,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<Vec<InitMessBlocksVar>, CircuitError> {
        let n = input_vars.len();
        // The choice of 12 is actually pretty arbitrary. We can always change this at a later date.
        if n > 12 {
            return Err(CircuitError::ParameterError(
                "Can only hash together up to 12 field elements".to_string(),
            ));
        }
        // We must first express the inputs in spread form
        // This will hold 'Variables' representing the spread
        // 32-bit chunks of the inputs concatenated together
        let mut spread_field_vars = Vec::<Variable>::new();
        for (idx, input_var) in input_vars.iter().enumerate() {
            let big_uint: BigUint = self.witness(*input_var)?.into();
            // This stores the 'Variable's representing the non-spread 32-bit chunks of the input
            let mut non_spread_field_vars = Vec::<Variable>::new();
            let mut u32digits = big_uint.to_u32_digits();
            u32digits.resize(8, 0u32);
            for (i, j) in u32digits.into_iter().rev().enumerate() {
                let field_var = self.create_variable(F::from(j))?;
                let (_, spread_field_chunks_var) =
                    self.non_spread_to_field_chunks(&field_var, true, lookup_vars)?;
                non_spread_field_vars.push(field_var);
                if i == 0 && idx == input_vars.len() - 1 {
                    // When we are dealing with the most significant chunk,
                    // we need to add the bit_var to the linear combination.
                    spread_field_vars.push(self.lin_comb(
                        &[
                            F::one(),
                            F::from(4194304_u32),
                            F::from(17592186044416_u64),
                            F::from(4611686018427387904_u64),
                        ],
                        &F::zero(),
                        &[spread_field_chunks_var.to_vec(), [bit_var.0].to_vec()].concat(),
                    )?);
                } else {
                    spread_field_vars.push(self.lin_comb(
                        &[F::one(), F::from(4194304_u32), F::from(17592186044416_u64)],
                        &F::zero(),
                        &spread_field_chunks_var,
                    )?);
                }
            }
            let coeffs = (0..8)
                .rev()
                .map(|i| F::from(BigUint::from(1u32) << (32 * i)))
                .collect::<Vec<F>>();

            // We must verify that the input is the appropriate linear combination of chunks
            self.lin_comb_gate(&coeffs, &F::zero(), &non_spread_field_vars, input_var)?;
        }
        // We put the spread 32-bit chunks into the message blocks in chunks of 16.
        // So each message block represents 2 field inputs.
        let mut init_mess_blocks_vars = Vec::<InitMessBlocksVar>::new();
        for spread_field_vec_var in spread_field_vars.chunks(16) {
            let mut init_mess_blocks_var = InitMessBlocksVar::default();
            for (i, spread_field_var) in spread_field_vec_var.iter().enumerate() {
                init_mess_blocks_var[i] = *spread_field_var;
            }
            init_mess_blocks_vars.push(init_mess_blocks_var);
        }
        // This is the length of the input needed for the padding of the message
        let spread_length_val = big_field_to_spread_field::<F>(256u32 * n as u32);
        // We put a 1 after all the inputs have been inserted into the message blocks.
        // The last few bits are then used to reprsent the length of the imput message.
        if n % 2 == 0 {
            let mut init_mess_blocks_var = InitMessBlocksVar::default();
            init_mess_blocks_var[0] =
                self.create_constant_variable(F::from(4611686018427387904_u64))?; // 2^{62}
            init_mess_blocks_var[15] = self.create_constant_variable(spread_length_val)?;
            init_mess_blocks_vars.push(init_mess_blocks_var);
        } else {
            init_mess_blocks_vars[n / 2][8] =
                self.create_constant_variable(F::from(4611686018427387904_u64))?; // 2^{62}
            init_mess_blocks_vars[n / 2][15] = self.create_constant_variable(spread_length_val)?;
        }

        Ok(init_mess_blocks_vars)
    }

    fn prepare_message_schedule(
        &mut self,
        init_mess_blocks_var: &InitMessBlocksVar,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<MessageScheduleVar, CircuitError> {
        // The 'Variable's reprsenting the output of the message schedule.
        let mut mess_sched_var: MessageScheduleVar = [Variable::default(); 64];
        mess_sched_var[..16].copy_from_slice(&init_mess_blocks_var[..16]);
        for i in 16..64 {
            let sigma_0 = self.rot_sum(&mess_sched_var[i - 15], &[3, 7, 18], true, lookup_vars)?;
            let sigma_1 = self.rot_sum(&mess_sched_var[i - 2], &[10, 17, 19], true, lookup_vars)?;
            mess_sched_var[i] = self.mod_2_32_add(
                &[
                    mess_sched_var[i - 7],
                    mess_sched_var[i - 16],
                    sigma_0,
                    sigma_1,
                ],
                &F::zero(),
                lookup_vars,
            )?;
        }
        Ok(mess_sched_var)
    }

    fn iter_comp_func(
        &mut self,
        mess_sched_var: &MessageScheduleVar,
        comp_output_var: &CompOutputVar,
        k_constants: &[F; 64],
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<CompOutputVar, CircuitError> {
        // The mutable 'Variable's that represent the input and output to the compression function
        let mut a_var = comp_output_var[0];
        let mut b_var = comp_output_var[1];
        let mut c_var = comp_output_var[2];
        let mut d_var = comp_output_var[3];
        let mut e_var = comp_output_var[4];
        let mut f_var = comp_output_var[5];
        let mut g_var = comp_output_var[6];
        let mut h_var = comp_output_var[7];

        for i in 0..64 {
            let upper_sigma_0_a_var = self.rot_sum(&a_var, &[2, 13, 22], false, lookup_vars)?;
            let upper_sigma_1_e_var = self.rot_sum(&e_var, &[6, 11, 25], false, lookup_vars)?;
            let maj_abc_var = self.majority(&[a_var, b_var, c_var], lookup_vars)?;
            let choice_efg_var = self.choice(&[e_var, f_var, g_var], lookup_vars)?;
            let t_1_vars = vec![
                h_var,
                upper_sigma_1_e_var,
                choice_efg_var,
                mess_sched_var[i],
            ];
            let t_1_var = self.mod_2_32_add(&t_1_vars, &k_constants[i], lookup_vars)?;
            let t_2_vars = vec![upper_sigma_0_a_var, maj_abc_var];
            let t_2_var = self.mod_2_32_add(&t_2_vars, &F::zero(), lookup_vars)?;

            h_var = g_var;
            g_var = f_var;
            f_var = e_var;
            e_var = self.mod_2_32_add(&[d_var, t_1_var], &F::zero(), lookup_vars)?;
            d_var = c_var;
            c_var = b_var;
            b_var = a_var;
            a_var = self.mod_2_32_add(&[t_1_var, t_2_var], &F::zero(), lookup_vars)?;
        }
        let a_var = self.mod_2_32_add(&[a_var, comp_output_var[0]], &F::zero(), lookup_vars)?;
        let b_var = self.mod_2_32_add(&[b_var, comp_output_var[1]], &F::zero(), lookup_vars)?;
        let c_var = self.mod_2_32_add(&[c_var, comp_output_var[2]], &F::zero(), lookup_vars)?;
        let d_var = self.mod_2_32_add(&[d_var, comp_output_var[3]], &F::zero(), lookup_vars)?;
        let e_var = self.mod_2_32_add(&[e_var, comp_output_var[4]], &F::zero(), lookup_vars)?;
        let f_var = self.mod_2_32_add(&[f_var, comp_output_var[5]], &F::zero(), lookup_vars)?;
        let g_var = self.mod_2_32_add(&[g_var, comp_output_var[6]], &F::zero(), lookup_vars)?;
        let h_var = self.mod_2_32_add(&[h_var, comp_output_var[7]], &F::zero(), lookup_vars)?;

        Ok([a_var, b_var, c_var, d_var, e_var, f_var, g_var, h_var])
    }

    fn hash_to_shifted_outputs(
        &mut self,
        comp_output_var: &CompOutputVar,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(Variable, Variable), CircuitError> {
        let all_limbs = comp_output_var
            .iter()
            .rev()
            .map(|var| {
                let (non_spread_field_chunks_var, _) =
                    self.spread_to_field_chunks(var, false, lookup_vars)?;
                self.lin_comb(
                    // coefficients are [1, 2^{11}, 2^{22}]
                    &[F::one(), F::from(2048_u32), F::from(4194304_u32)],
                    &F::zero(),
                    &non_spread_field_chunks_var,
                )
            })
            .collect::<Result<Vec<Variable>, CircuitError>>()?;
        let tmp: BigUint = self.witness(all_limbs[0])?.into();
        let first_limb = tmp.to_u32().ok_or(CircuitError::ParameterError(
            "cannot convert BigUint into u32 as too big".to_string(),
        ))?;
        let first_four_bits = first_limb & 15;
        let other_bits = first_limb >> 4;
        let low_bits_var = self.create_variable(F::from(first_four_bits))?;
        let high_bits_var = self.create_variable(F::from(other_bits))?;

        self.enforce_in_range(low_bits_var, 4)?;
        self.enforce_in_range(high_bits_var, 28)?;

        self.lc_gate(
            &[
                low_bits_var,
                high_bits_var,
                self.zero(),
                self.zero(),
                all_limbs[0],
            ],
            &[F::one(), F::from(16u32), F::zero(), F::zero()],
        )?;
        // This 'Variable' represents the top 252 bits of the output
        let mut acc = self.lc(
            &[all_limbs[7], all_limbs[6], all_limbs[5], all_limbs[4]],
            &[
                F::from(1u128 << 96),
                F::from(1u128 << 64),
                F::from(1u64 << 32),
                F::one(),
            ],
        )?;
        acc = self.lc(
            &[acc, all_limbs[3], all_limbs[2], all_limbs[1]],
            &[
                F::from(1u128 << 96),
                F::from(1u128 << 64),
                F::from(1u64 << 32),
                F::one(),
            ],
        )?;

        acc = self.lc(
            &[acc, high_bits_var, self.zero(), self.zero()],
            &[F::from(1u32 << 28), F::one(), F::zero(), F::zero()],
        )?;

        Ok((low_bits_var, acc))
    }

    fn full_shifted_sha256_hash(
        &mut self,
        input_vars: &[Variable],
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(Variable, Variable), CircuitError> {
        let init_mess_blocks_var = self.preprocess(input_vars, lookup_vars)?;

        let (h_constants, k_constants) = F::params().map_err(|_| {
            CircuitError::ParameterError("Could not retrieve sha256 constants".to_string())
        })?;
        let mut comp_output_var = CompOutputVar::default();
        for i in 0..8 {
            comp_output_var[i] = self.create_constant_variable(h_constants[i])?;
        }

        for blocks_var in init_mess_blocks_var {
            let mess_sched_var = self.prepare_message_schedule(&blocks_var, lookup_vars)?;

            comp_output_var =
                self.iter_comp_func(&mess_sched_var, &comp_output_var, &k_constants, lookup_vars)?;
        }
        self.hash_to_shifted_outputs(&comp_output_var, lookup_vars)
    }

    fn full_shifted_sha256_hash_with_bit(
        &mut self,
        input_vars: &[Variable],
        bit_var: &BoolVar,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(Variable, Variable), CircuitError> {
        let init_mess_blocks_var = self.preprocess_with_bit(input_vars, bit_var, lookup_vars)?;

        let (h_constants, k_constants) = F::params().map_err(|_| {
            CircuitError::ParameterError("Could not retrieve sha256 constants".to_string())
        })?;
        let mut comp_output_var = CompOutputVar::default();
        for i in 0..8 {
            comp_output_var[i] = self.create_constant_variable(h_constants[i])?;
        }

        for blocks_var in init_mess_blocks_var {
            let mess_sched_var = self.prepare_message_schedule(&blocks_var, lookup_vars)?;

            comp_output_var =
                self.iter_comp_func(&mess_sched_var, &comp_output_var, &k_constants, lookup_vars)?;
        }
        self.hash_to_shifted_outputs(&comp_output_var, lookup_vars)
    }

    fn finalize_for_sha256_hash(
        &mut self,
        lookup_vars: &mut Vec<(Variable, Variable, Variable)>,
    ) -> Result<(), CircuitError> {
        // We need to be able to range check up the 10:
        if self.range_bit_len()? < 4 {
            return Err(CircuitError::ParameterError(
                "'range_bit_length' must be as least 4".to_string(),
            ));
        }
        // The 'lookup_table' consists of the input field element 'val' (this is just
        // the index), the 'Variable' representing 'spread(val)' and the 'Variable'
        // representing 'floor(log_2(val)) + 1' (we take this as 0 when input=0).
        let mut log_vars = Vec::<Variable>::new();
        log_vars.push(self.zero());
        log_vars.push(self.one());
        for i in 2..=11 {
            let var = self.create_constant_variable(F::from(i as u32))?;
            log_vars.push(var);
        }

        let mut lookup_table = Vec::<(Variable, Variable)>::new();
        for i in 0..(1 << 11) {
            let mut lookup_val = F::zero();
            let mut log_val = 0;
            for j in 0..11 {
                let bit = (i >> j) & 1;
                if bit == 1 {
                    lookup_val += F::from(1_u32 << (2 * j));
                    log_val = j + 1;
                }
            }
            let var = self.create_constant_variable(lookup_val)?;
            lookup_table.push((var, log_vars[log_val]));
        }

        self.create_table_and_lookup_variables(lookup_vars, &lookup_table)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr as Fr254;
    use ark_std::UniformRand;
    use ark_std::{One, Zero};
    use digest::Digest;
    use jf_utils::test_rng;
    use sha2::Sha256;

    #[test]
    //testing the choice operation.
    fn choice_test() -> Result<(), CircuitError> {
        let rng = &mut test_rng();
        for _ in 0..20 {
            let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(4);
            let mut lookup_vars = Vec::<(Variable, Variable, Variable)>::new();
            let mut field_vec_val = [Fr254::zero(); 3];
            let mut final_field_val = Fr254::zero();
            for i in 0..32 {
                let mod8 = usize::rand(rng) % 8;
                let val_0 = mod8 % 2;
                let val_1 = (mod8 / 2) % 2;
                let val_2 = mod8 / 4;
                if val_0 * val_1 + (1 - val_0) * val_2 == 1 {
                    final_field_val += Fr254::from(1_u64 << (2 * i));
                }
                if val_0 == 1 {
                    field_vec_val[0] += Fr254::from(1_u64 << (2 * i));
                }
                if val_1 == 1 {
                    field_vec_val[1] += Fr254::from(1_u64 << (2 * i));
                }
                if val_2 == 1 {
                    field_vec_val[2] += Fr254::from(1_u64 << (2 * i));
                }
            }
            let field_vec_var: [Variable; 3] = [
                circuit.create_variable(field_vec_val[0])?,
                circuit.create_variable(field_vec_val[1])?,
                circuit.create_variable(field_vec_val[2])?,
            ];
            let res_var = circuit.choice(&field_vec_var, &mut lookup_vars)?;

            circuit.finalize_for_sha256_hash(&mut lookup_vars)?;
            assert_eq!(circuit.witness(res_var)?, final_field_val);
            circuit.check_circuit_satisfiability(&[])?;
        }
        Ok(())
    }

    #[test]
    //testing the choice operation.
    fn majority_test() -> Result<(), CircuitError> {
        let rng = &mut test_rng();
        for _ in 0..20 {
            let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(4);
            let mut lookup_vars = Vec::<(Variable, Variable, Variable)>::new();
            let mut field_vec_val = [Fr254::zero(); 3];
            let mut final_field_val = Fr254::zero();
            for i in 0..32 {
                let mod8 = usize::rand(rng) % 8;
                let val_0 = mod8 % 2;
                let val_1 = (mod8 / 2) % 2;
                let val_2 = mod8 / 4;
                if val_0 + val_1 + val_2 > 1 {
                    final_field_val += Fr254::from(1_u64 << (2 * i));
                }
                if val_0 == 1 {
                    field_vec_val[0] += Fr254::from(1_u64 << (2 * i));
                }
                if val_1 == 1 {
                    field_vec_val[1] += Fr254::from(1_u64 << (2 * i));
                }
                if val_2 == 1 {
                    field_vec_val[2] += Fr254::from(1_u64 << (2 * i));
                }
            }
            let field_vec_var: [Variable; 3] = [
                circuit.create_variable(field_vec_val[0])?,
                circuit.create_variable(field_vec_val[1])?,
                circuit.create_variable(field_vec_val[2])?,
            ];

            let res_var = circuit.majority(&field_vec_var, &mut lookup_vars)?;

            circuit.finalize_for_sha256_hash(&mut lookup_vars)?;
            assert_eq!(circuit.witness(res_var)?, final_field_val);
            circuit.check_circuit_satisfiability(&[])?;
        }
        Ok(())
    }

    #[test]
    //testing the rotate/shift operation.
    fn rot_sum_test() -> Result<(), CircuitError> {
        let rng = &mut test_rng();
        for _ in 0..20 {
            let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(4);
            let mut lookup_vars = Vec::<(Variable, Variable, Variable)>::new();
            let mut input_bits = [0_u8; 32];
            let mut output_bits = [0_u8; 32];
            for i in 0..32 {
                if usize::rand(rng) % 2 == 0 {
                    input_bits[i] = 1;
                    output_bits[(i + 30) % 32] = 1 - output_bits[(i + 30) % 32];
                    output_bits[(i + 19) % 32] = 1 - output_bits[(i + 19) % 32];
                    output_bits[(i + 10) % 32] = 1 - output_bits[(i + 10) % 32];
                }
            }
            let input_val = (0..32).fold(Fr254::zero(), |acc, i| {
                acc + Fr254::from(input_bits[i]) * Fr254::from(1_u64 << (2 * i))
            });
            let input_var = circuit.create_variable(input_val)?;

            let output_val = (0..32).fold(Fr254::zero(), |acc, i| {
                acc + Fr254::from(output_bits[i]) * Fr254::from(1_u64 << (2 * i))
            });

            let res_var = circuit.rot_sum(&input_var, &[2, 13, 22], false, &mut lookup_vars)?;

            circuit.finalize_for_sha256_hash(&mut lookup_vars)?;
            assert_eq!(circuit.witness(res_var)?, output_val);
            circuit.check_circuit_satisfiability(&[])?;
        }

        for _ in 0..20 {
            let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(4);
            let mut lookup_vars = Vec::<(Variable, Variable, Variable)>::new();
            let mut input_bits = [0_u8; 32];
            let mut output_bits = [0_u8; 32];
            for i in 0..32 {
                if usize::rand(rng) % 2 == 0 {
                    input_bits[i] = 1;
                    if i >= 3 {
                        output_bits[i - 3] = 1 - output_bits[i - 3];
                    }
                    output_bits[(i + 25) % 32] = 1 - output_bits[(i + 25) % 32];
                    output_bits[(i + 14) % 32] = 1 - output_bits[(i + 14) % 32];
                }
            }
            let input_val = (0..32).fold(Fr254::zero(), |acc, i| {
                acc + Fr254::from(input_bits[i]) * Fr254::from(1_u64 << (2 * i))
            });
            let input_var = circuit.create_variable(input_val)?;

            let output_val = (0..32).fold(Fr254::zero(), |acc, i| {
                acc + Fr254::from(output_bits[i]) * Fr254::from(1_u64 << (2 * i))
            });

            let res_var = circuit.rot_sum(&input_var, &[3, 7, 18], true, &mut lookup_vars)?;

            circuit.finalize_for_sha256_hash(&mut lookup_vars)?;
            assert_eq!(circuit.witness(res_var)?, output_val);
            circuit.check_circuit_satisfiability(&[])?;
        }
        Ok(())
    }

    #[test]
    //testing the mod 2^{32} operation.
    fn mod_2_32_test() -> Result<(), CircuitError> {
        let rng = &mut test_rng();
        for _ in 0..20 {
            let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(4);
            let mut lookup_vars = Vec::<(Variable, Variable, Variable)>::new();
            let num_summands = (usize::rand(rng) % 5) + 2;
            let mut non_spread_vals = vec![Fr254::zero(); num_summands];
            let mut spread_vals = vec![Fr254::zero(); num_summands];
            let mut non_spread_constant = Fr254::zero();
            for j in 0..32 {
                for i in 0..num_summands {
                    if usize::rand(rng) % 2 == 0 {
                        non_spread_vals[i] += Fr254::from(1_u32 << j);
                        spread_vals[i] += Fr254::from(1_u64 << (2 * j));
                    }
                }
                if usize::rand(rng) % 2 == 0 {
                    non_spread_constant += Fr254::from(1_u32 << j);
                }
            }
            let exp_sum_big_uint: BigUint = (non_spread_vals.iter().sum::<Fr254>()
                + non_spread_constant)
                .into_bigint()
                .into();
            let exp_non_spread_result_big_uint: BigUint =
                exp_sum_big_uint % BigUint::from(18446744073709551616_u128); // mod 2^{64}
            let mut exp_spread_result_val = Fr254::zero();
            for i in 0..32 {
                let bit = (&exp_non_spread_result_big_uint / BigUint::from(1_u32 << i))
                    % BigUint::from(2_u8);
                if bit > BigUint::zero() {
                    exp_spread_result_val += Fr254::from(1_u64 << (2 * i));
                }
            }

            let spread_vars = spread_vals
                .iter()
                .map(|val| circuit.create_variable(*val))
                .collect::<Result<Vec<Variable>, CircuitError>>()?;

            let res_var =
                circuit.mod_2_32_add(&spread_vars, &non_spread_constant, &mut lookup_vars)?;

            circuit.finalize_for_sha256_hash(&mut lookup_vars)?;
            assert_eq!(circuit.witness(res_var)?, exp_spread_result_val);
            circuit.check_circuit_satisfiability(&[])?;
        }
        Ok(())
    }

    #[test]
    // testing the preprocess operation.
    fn preprocess_test() -> Result<(), CircuitError> {
        let rng = &mut test_rng();
        for _ in 0..20 {
            let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(4);
            let mut lookup_vars = Vec::<(Variable, Variable, Variable)>::new();
            let num_field_elems = (usize::rand(rng) % 12) + 1;

            let mut field_vals = Vec::<Fr254>::new();
            let mut input_big_uints = Vec::<BigUint>::new();
            let mut field_vars = Vec::<Variable>::new();
            for _ in 0..num_field_elems {
                let val = Fr254::rand(rng);
                field_vals.push(val);
                input_big_uints.push(val.into_bigint().into());
                field_vars.push(circuit.create_variable(val)?);
            }

            let init_mess_blocks_vars = circuit.preprocess(&field_vars, &mut lookup_vars)?;
            assert_eq!(init_mess_blocks_vars.len(), num_field_elems / 2 + 1);

            let mut output_big_uints = Vec::<BigUint>::new();
            for i in 0..(2 * init_mess_blocks_vars.len()) {
                let mut output_big_uint = BigUint::zero();
                for j in 0..8 {
                    let l = 8 * (i % 2) + j;
                    let big_uint: BigUint = circuit
                        .witness(init_mess_blocks_vars[i / 2][l])?
                        .into_bigint()
                        .into();
                    for k in 0..32 {
                        let bit =
                            (&big_uint / BigUint::from(1_u64 << (2 * k))) % BigUint::from(4_u8);
                        if bit > BigUint::from(1_u8) {
                            return Err(CircuitError::ParameterError("bit too big!".to_string()));
                        }
                        if bit == BigUint::from(1_u8) {
                            output_big_uint += BigUint::from(2_u8).pow(32 * (7 - j) + k);
                        }
                    }
                }
                output_big_uints.push(output_big_uint);
            }

            for i in 0..num_field_elems {
                assert_eq!(input_big_uints[i], output_big_uints[i]);
            }

            let length_big_uint = BigUint::from(num_field_elems as u32 * 256);
            if num_field_elems % 2 == 0 {
                assert_eq!(
                    output_big_uints[num_field_elems],
                    BigUint::from(2_u8).pow(7_u32 * 32 + 31)
                );
                assert_eq!(output_big_uints[num_field_elems + 1], length_big_uint);
            } else {
                assert_eq!(
                    output_big_uints[num_field_elems],
                    BigUint::from(2_u8).pow(7_u32 * 32 + 31) + length_big_uint
                );
            }
            circuit.finalize_for_sha256_hash(&mut lookup_vars)?;
            circuit.check_circuit_satisfiability(&[])?;
        }
        Ok(())
    }

    #[test]
    //testing the sha256 hash.
    fn full_hash_test() -> Result<(), CircuitError> {
        let rng = &mut test_rng();
        for _ in 0..30 {
            let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(4);
            let mut lookup_vars = Vec::<(Variable, Variable, Variable)>::new();
            let num_field_elems = (usize::rand(rng) % 12) + 1;

            let mut field_vals = Vec::<Fr254>::new();
            let mut field_vars = Vec::<Variable>::new();
            for _ in 0..num_field_elems {
                let val = Fr254::rand(rng);
                field_vals.push(val);
                field_vars.push(circuit.create_variable(val)?);
            }

            let mut field_bytes = Vec::<u8>::new();
            for val in field_vals {
                let big_uint: BigUint = val.into_bigint().into();
                let mut bytes_vec = big_uint.to_bytes_be();
                while bytes_vec.len() < 32 {
                    bytes_vec.insert(0, 0_u8);
                }
                field_bytes.extend_from_slice(&bytes_vec);
            }

            let mut hasher = Sha256::new();
            hasher.update(field_bytes);
            let exp_hash_val = hasher.finalize();

            let (non_spread_lower_var, non_spread_upper_var) =
                circuit.full_shifted_sha256_hash(&field_vars, &mut lookup_vars)?;

            let lower_big_uint: BigUint =
                circuit.witness(non_spread_lower_var)?.into_bigint().into();
            let upper_big_uint: BigUint =
                circuit.witness(non_spread_upper_var)?.into_bigint().into();
            let big_uint_output = lower_big_uint + BigUint::from(16_u8) * upper_big_uint;

            let mut bytes_output = big_uint_output.to_bytes_be();
            while bytes_output.len() < 32 {
                bytes_output.insert(0, 0_u8);
            }

            circuit.finalize_for_sha256_hash(&mut lookup_vars)?;
            circuit.check_circuit_satisfiability(&[])?;

            for i in 0..32 {
                assert_eq!(bytes_output[i], exp_hash_val[i]);
            }
        }

        let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(4);
        let mut lookup_vars = Vec::<(Variable, Variable, Variable)>::new();
        let num_field_elems = 12;

        let mut field_vals = Vec::<Fr254>::new();
        let mut field_vars = Vec::<Variable>::new();
        for _ in 0..num_field_elems {
            let val = Fr254::from(0u8);
            field_vals.push(val);
            field_vars.push(circuit.create_variable(val)?);
        }

        let mut field_bytes = Vec::<u8>::new();
        for val in field_vals {
            let big_uint: BigUint = val.into_bigint().into();
            let mut bytes_vec = big_uint.to_bytes_be();
            while bytes_vec.len() < 32 {
                bytes_vec.insert(0, 0_u8);
            }
            field_bytes.extend_from_slice(&bytes_vec);
        }

        let mut hasher = Sha256::new();
        hasher.update(field_bytes);
        let exp_hash_val = hasher.finalize();

        let (non_spread_lower_var, non_spread_upper_var) =
            circuit.full_shifted_sha256_hash(&field_vars, &mut lookup_vars)?;

        let lower_big_uint: BigUint = circuit.witness(non_spread_lower_var)?.into_bigint().into();
        let upper_big_uint: BigUint = circuit.witness(non_spread_upper_var)?.into_bigint().into();
        let big_uint_output = lower_big_uint + BigUint::from(16_u8) * upper_big_uint;

        let mut bytes_output = big_uint_output.to_bytes_be();
        while bytes_output.len() < 32 {
            bytes_output.insert(0, 0_u8);
        }

        circuit.finalize_for_sha256_hash(&mut lookup_vars)?;
        circuit.check_circuit_satisfiability(&[])?;

        for i in 0..32 {
            assert_eq!(bytes_output[i], exp_hash_val[i]);
        }
        Ok(())
    }

    #[test]
    //testing the sha256 hash.
    fn full_hash_with_bit_test() -> Result<(), CircuitError> {
        let rng = &mut test_rng();
        for _ in 0..30 {
            let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(4);
            let mut lookup_vars = Vec::<(Variable, Variable, Variable)>::new();
            let num_field_elems = (usize::rand(rng) % 12) + 1;

            let mut field_vals = Vec::<Fr254>::new();
            let mut field_vars = Vec::<Variable>::new();
            for _ in 0..num_field_elems {
                let val = Fr254::rand(rng);
                field_vals.push(val);
                field_vars.push(circuit.create_variable(val)?);
            }
            let bit_var = circuit.create_boolean_variable(usize::rand(rng) % 2 == 0)?;

            let mut field_bytes = Vec::<u8>::new();
            for (i, val) in field_vals.iter().enumerate() {
                let big_uint: BigUint = val.into_bigint().into();
                let mut bytes_vec = big_uint.to_bytes_be();
                while bytes_vec.len() < 32 {
                    bytes_vec.insert(0, 0_u8);
                }
                if i == num_field_elems - 1 && circuit.witness(bit_var.0)? == Fr254::one() {
                    // If the bit is set, we add a 1 at the end of the last field element
                    bytes_vec[0] += 1u8 << 7; // Set the highest bit
                }
                field_bytes.extend_from_slice(&bytes_vec);
            }

            let mut hasher = Sha256::new();
            hasher.update(field_bytes);
            let exp_hash_val = hasher.finalize();

            let (non_spread_lower_var, non_spread_upper_var) = circuit
                .full_shifted_sha256_hash_with_bit(&field_vars, &bit_var, &mut lookup_vars)?;

            let lower_big_uint: BigUint =
                circuit.witness(non_spread_lower_var)?.into_bigint().into();
            let upper_big_uint: BigUint =
                circuit.witness(non_spread_upper_var)?.into_bigint().into();
            let big_uint_output = lower_big_uint + BigUint::from(16_u8) * upper_big_uint;

            let mut bytes_output = big_uint_output.to_bytes_be();
            while bytes_output.len() < 32 {
                bytes_output.insert(0, 0_u8);
            }

            circuit.finalize_for_sha256_hash(&mut lookup_vars)?;
            circuit.check_circuit_satisfiability(&[])?;

            for i in 0..32 {
                assert_eq!(bytes_output[i], exp_hash_val[i]);
            }
        }

        let mut circuit: PlonkCircuit<Fr254> = PlonkCircuit::new_ultra_plonk(4);
        let mut lookup_vars = Vec::<(Variable, Variable, Variable)>::new();
        let num_field_elems = 12;

        let mut field_vals = Vec::<Fr254>::new();
        let mut field_vars = Vec::<Variable>::new();
        for _ in 0..num_field_elems {
            let val = Fr254::from(0u8);
            field_vals.push(val);
            field_vars.push(circuit.create_variable(val)?);
        }
        let bit_var = circuit.create_boolean_variable(usize::rand(rng) % 2 == 0)?;

        let mut field_bytes = Vec::<u8>::new();
        for (i, val) in field_vals.iter().enumerate() {
            let big_uint: BigUint = val.into_bigint().into();
            let mut bytes_vec = big_uint.to_bytes_be();
            while bytes_vec.len() < 32 {
                bytes_vec.insert(0, 0_u8);
            }
            if i == num_field_elems - 1 && circuit.witness(bit_var.0)? == Fr254::one() {
                // If the bit is set, we add a 1 at the end of the last field element
                bytes_vec[0] += 1u8 << 7; // Set the highest bit
            }
            field_bytes.extend_from_slice(&bytes_vec);
        }

        let mut hasher = Sha256::new();
        hasher.update(field_bytes);
        let exp_hash_val = hasher.finalize();

        let (non_spread_lower_var, non_spread_upper_var) =
            circuit.full_shifted_sha256_hash_with_bit(&field_vars, &bit_var, &mut lookup_vars)?;

        let lower_big_uint: BigUint = circuit.witness(non_spread_lower_var)?.into_bigint().into();
        let upper_big_uint: BigUint = circuit.witness(non_spread_upper_var)?.into_bigint().into();
        let big_uint_output = lower_big_uint + BigUint::from(16_u8) * upper_big_uint;

        let mut bytes_output = big_uint_output.to_bytes_be();
        while bytes_output.len() < 32 {
            bytes_output.insert(0, 0_u8);
        }

        circuit.finalize_for_sha256_hash(&mut lookup_vars)?;
        circuit.check_circuit_satisfiability(&[])?;

        for i in 0..32 {
            assert_eq!(bytes_output[i], exp_hash_val[i]);
        }
        Ok(())
    }
}
