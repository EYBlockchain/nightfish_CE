//! Code for performing a Poseidon hash.
pub(crate) mod constants;
pub(crate) mod sponge;
pub use self::constants::PoseidonParams;
use ark_ff::PrimeField;
use ark_std::{error::Error, fmt::Display, marker::PhantomData, string::ToString, vec, vec::Vec};

/// The state size of poseidon hash.
pub const STATE_SIZE: usize = 4;

/// Error enum for the Poseidon hash function.
///
/// See Variants for more information about when this error is thrown.
#[derive(Debug, Clone, PartialEq)]
pub enum PoseidonError {
    /// Thrown if the user attempts to input a vector whose length is
    /// greater than the predefined threshold, e.g, 6 in NF.
    InvalidInputs,
}

/// Error messages for PoseidonError.
impl Display for PoseidonError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use PoseidonError::*;
        let msg = match self {
            InvalidInputs => "invalid inputs length".to_string(),
        };
        write!(f, "{}", msg)
    }
}

impl Error for PoseidonError {}

#[derive(Debug, Clone, PartialEq, Copy)]
/**
 * In the poseidon_hash_native gadget, Poseidon is defined as a struct,
 * which only has one field name: params. params is an instance of struct
 * PoseidonHashParameters<F>, where we define all the constant parameter.
 *
 * The implementation of trait FieldHasher is where all the magic take place to
 * compute the poseidon hash result.
 */
pub struct Poseidon<F: PoseidonParams>(PhantomData<F>);

impl<F: PoseidonParams> Poseidon<F> {
    /// Cosntructs a new Poseidon instance.
    pub fn new() -> Self {
        Poseidon(PhantomData)
    }
}
impl<F: PoseidonParams> Default for Poseidon<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Copy, Default)]
/// Data type for rescue prp inputs, keys and internal data
pub struct PoseidonVector<F> {
    pub(crate) vec: [F; STATE_SIZE],
}

impl<F: PrimeField> From<&Vec<F>> for PoseidonVector<F> {
    fn from(vector: &Vec<F>) -> Self {
        let vec: [F; STATE_SIZE] = (*vector).clone().try_into().unwrap();
        Self { vec }
    }
}

// Private functions
impl<F: PrimeField> PoseidonVector<F> {
    // To be consistent with `Poseidon::hash()`, we need start adding elements in position `STATE_SIZE - rate`.
    fn add_assign_elems(&mut self, elems: &[F], rate: usize) {
        self.vec
            .iter_mut()
            .skip(STATE_SIZE - rate)
            .zip(elems.iter())
            .for_each(|(a, b)| a.add_assign(b));
    }

    fn add_assign(&mut self, pos_vec: &PoseidonVector<F>) {
        for (a, b) in self.vec.iter_mut().zip(pos_vec.vec.iter()) {
            a.add_assign(b);
        }
    }

    fn full_pow(&mut self, exp: &[u64]) {
        self.vec.iter_mut().for_each(|elem| {
            *elem = elem.pow(exp);
        });
    }

    fn part_pow(&mut self, exp: &[u64]) {
        self.vec[0] = self.vec[0].pow(exp);
    }

    fn dot_product(&self, pos_vec: &PoseidonVector<F>) -> F {
        let mut r = F::zero();
        for (a, b) in self.vec.iter().zip(pos_vec.vec.iter()) {
            r.add_assign(&a.mul(b));
        }
        r
    }
}

/// A matrix that consists of `STATE_SIZE` number of rescue vectors.
#[derive(Debug, Clone)]
pub struct PoseidonMatrix<F> {
    matrix: [PoseidonVector<F>; STATE_SIZE],
}

impl<F: PrimeField> From<&Vec<Vec<F>>> for PoseidonMatrix<F> {
    fn from(matrix: &Vec<Vec<F>>) -> Self {
        let mat: [PoseidonVector<F>; STATE_SIZE] = matrix
            .iter()
            .map(PoseidonVector::from)
            .collect::<Vec<PoseidonVector<F>>>()
            .try_into()
            .unwrap();
        Self { matrix: mat }
    }
}

impl<F: PrimeField> PoseidonMatrix<F> {
    fn mul_vec(&self, vector: &PoseidonVector<F>) -> PoseidonVector<F> {
        let mut result = [F::zero(); STATE_SIZE];
        self.matrix
            .iter()
            .enumerate()
            .for_each(|(i, row)| result[i] = row.dot_product(vector));
        PoseidonVector { vec: result }
    }
}

/// A `PoseidonPerm` consists of a matrix, round constants and the number of partial rounds
#[derive(Debug, Clone)]
pub struct PoseidonPerm<F> {
    mds: PoseidonMatrix<F>,                  // poseidon permutation MDS matrix
    round_constants: Vec<PoseidonVector<F>>, // poseidon round constants
    num_part_rounds: usize,                  // number of partial rounds
}

impl<F: PoseidonParams> PoseidonPerm<F> {
    //#[cfg(test)]
    /// Retrieves the appropriate Poseidon parameters for the given field `F`.
    pub fn perm() -> Result<Self, PoseidonError> {
        let (constants, matrix, num_part_rounds) = F::params(STATE_SIZE)?;
        let round_constants = constants
            .iter()
            .map(PoseidonVector::from)
            .collect::<Vec<PoseidonVector<F>>>();
        Ok(PoseidonPerm {
            mds: PoseidonMatrix::<F>::from(&matrix),
            round_constants,
            num_part_rounds,
        })
    }

    /// Compute the permutation on PoseidonVector `input`
    pub fn eval(&self, input: &mut PoseidonVector<F>) {
        for (i, constants) in self
            .round_constants
            .iter()
            .enumerate()
            .take(8 + self.num_part_rounds)
        {
            input.add_assign(constants);
            if i < 4 || i >= 4 + self.num_part_rounds {
                input.full_pow(&[5u64]);
            } else {
                input.part_pow(&[5u64]);
            }
            *input = self.mds.mul_vec(input);
        }
    }
}

/**
 * A field hasher over a prime field `F` is any cryptographic hash function
 * that takes in a vector of elements of `F` and outputs a single element
 * of `F`.
 * JJ: will move this part to ports in future
 */
pub trait FieldHasher<F: PrimeField> {
    /// Hashes between 1 and 6 elements of the field `F`.
    fn hash(&self, inputs: &[F]) -> Result<F, PoseidonError>;
}
impl<F: PoseidonParams> FieldHasher<F> for Poseidon<F> {
    fn hash(&self, inputs: &[F]) -> Result<F, PoseidonError> {
        let t = inputs.len() + 1;

        let (constants, matrix, n_rounds_p) = F::params(t)?;

        let mut state = vec![F::zero(); t];
        state[1..].clone_from_slice(inputs);

        for (i, constant) in constants.iter().enumerate().take(8 + n_rounds_p) {
            //ark
            for (j, state_j) in state.iter_mut().enumerate() {
                *state_j += &constant[j];
            }
            //sub words x^5
            if i < 4 || i >= 4 + n_rounds_p {
                for state_j in state.iter_mut() {
                    *state_j = state_j.pow([5u64]);
                }
            } else {
                state[0] = state[0].pow([5u64]);
            }

            let tmp_state = state.clone();
            for (index, row) in matrix.iter().enumerate() {
                state[index] = row.iter().zip(tmp_state.iter()).map(|(x, y)| *x * y).sum();
            }
        }

        Ok(state[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ark_bn254::Fr as FqEd254;

    use ark_std::{vec::Vec, UniformRand};
    use byteorder::{LittleEndian, WriteBytesExt};
    use ff_ce::PrimeField;
    use jf_utils::to_bytes;
    use num::traits::Num;
    use num_bigint::BigUint;
    use poseidon_rs::{Fr, FrRepr, Poseidon as Poseidon_RS};

    #[test]
    // /the purpose of this test is to compare the results of our poseidon hash is the
    // /same as poseidon_rs library
    fn unit_test_poseidon_hash_native() {
        let poseidon_native = Poseidon::<FqEd254>::new();
        let result_native = poseidon_native.hash(&[FqEd254::from(1)]).unwrap();

        let poseidon_rs = Poseidon_RS::new();
        let result_poseidon_rs = poseidon_rs.hash(vec![Fr::from_str("1").unwrap()]).unwrap();

        let big_uint_value = BigUint::parse_bytes(
            b"29176100eaa962bdc1fe6c654d6a3c130e96a4d1168b33848b897dc502820133",
            16,
        )
        .unwrap();

        assert_eq!(result_native.to_string(), big_uint_value.to_string());
        assert_eq!(
            result_poseidon_rs.to_string(),
            "Fr(0x29176100eaa962bdc1fe6c654d6a3c130e96a4d1168b33848b897dc502820133)"
        );
        let unit_test_input = [1, 2, 0, 0, 0].map(FqEd254::from);
        let result_native = poseidon_native.hash(&unit_test_input).unwrap();

        assert_eq!(
            BigUint::parse_bytes(
                b"1018317224307729531995786483840663576608797660851238720571059489595066344487",
                10,
            )
            .unwrap(),
            ark_ff::PrimeField::into_bigint(result_native).into()
        );

        let unit_test_input = [1, 2, 0].map(FqEd254::from);
        let result_native = poseidon_native.hash(&unit_test_input).unwrap();
        assert_eq!(
            BigUint::parse_bytes(
                b"13831821852403126897479426070347226427183075710625481252219866028995538813194",
                10,
            )
            .unwrap(),
            ark_ff::PrimeField::into_bigint(result_native).into()
        );

        let unit_test_input = [3, 4, 5, 10, 23].map(FqEd254::from);
        let result_native = poseidon_native.hash(&unit_test_input).unwrap();
        assert_eq!(
            BigUint::parse_bytes(
                b"13034429309846638789535561449942021891039729847501137143363028890275222221409",
                10,
            )
            .unwrap(),
            ark_ff::PrimeField::into_bigint(result_native).into()
        );

        let unit_test_input = [3, 4, 0].map(FqEd254::from);
        let result_native = poseidon_native.hash(&unit_test_input).unwrap();

        assert_eq!(
            BigUint::parse_bytes(
                b"20920110273428514568010870241452043716145962658983296477235744958655125001669",
                10,
            )
            .unwrap(),
            ark_ff::PrimeField::into_bigint(result_native).into()
        );
    }

    #[test]
    fn random_test_poseidon_hash_native() {
        // Test local hash against known external implementation:
        let fr_modulus = BigUint::from_str_radix(
            "21888242871839275222246405745257275088548364400416034343698204186575808495617",
            16,
        )
        .unwrap();

        for _ in 0..=10 {
            let mut rng = ark_std::test_rng();
            let rand_native = FqEd254::rand(&mut rng);
            let now = ark_std::time::Instant::now();
            let poseidon_native = Poseidon::<FqEd254>::new();
            let result_native = poseidon_native.hash(&[rand_native]).unwrap();
            ark_std::println!("Time taken for native poseidon: {:?}", now.elapsed());
            let big_int = ark_ff::PrimeField::into_bigint(rand_native);
            let bytes = to_bytes![&big_int].unwrap();

            let mut limbs: [u64; 4] = [0; 4];
            for i in 0..4 {
                limbs[i] = u64::from_be_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
            }

            let big_uint_value = BigUint::from_bytes_le(&bytes);
            let reduced_big_uint = big_uint_value % &fr_modulus;

            let mut reduced_bytes = reduced_big_uint.to_bytes_le();
            reduced_bytes.resize(32, 0); // occasionally we'll randomly get something less that 32 bytes, which causes the test to fail.
            let mut reduced_limbs: [u64; 4] = [0; 4];
            for i in 0..4 {
                reduced_limbs[i] =
                    u64::from_le_bytes(reduced_bytes[i * 8..(i + 1) * 8].try_into().unwrap());
            }
            let reduced_fr_repr = FrRepr(reduced_limbs);
            let rand_rs: Fr = ff_ce::PrimeField::from_repr(reduced_fr_repr).unwrap();

            let poseidon_poseidon_rs = Poseidon_RS::new();
            let result_poseidon_rs = poseidon_poseidon_rs.hash(vec![rand_rs]).unwrap();

            let result_native_biguint =
                BigUint::from_str_radix(&result_native.to_string(), 10).unwrap();

            let result_poseidon_rs_biguint = {
                let fr_repr = ff_ce::PrimeField::into_repr(&result_poseidon_rs);
                let mut bytes = Vec::new();
                for limb in &fr_repr.0 {
                    bytes.write_u64::<LittleEndian>(*limb).unwrap();
                }
                BigUint::from_bytes_le(&bytes)
            };

            assert_eq!(result_native_biguint, result_poseidon_rs_biguint);
        }
    }
}
