use ark_crypto_primitives::sponge::{
    Absorb, CryptographicSponge, FieldBasedCryptographicSponge, FieldElementSize,
};
use ark_ff::PrimeField;
use ark_std::{vec, vec::Vec};

use super::{PoseidonParams, PoseidonPerm, PoseidonVector};

/// The rate of the sponge used in Poseidon.
pub const CRHF_RATE: usize = 3;

#[derive(Clone, Debug)]
/// A poseidon hash function consists of a permutation function and
/// an internal state.
pub struct PoseidonSponge<F: PoseidonParams, const RATE: usize> {
    pub(crate) state: PoseidonVector<F>,
    pub(crate) permutation: PoseidonPerm<F>,
}

impl<T: PoseidonParams + PrimeField, const RATE: usize> CryptographicSponge
    for PoseidonSponge<T, RATE>
{
    /// Config used by the sponge.
    type Config = PoseidonPerm<T>;

    /// Initialize a new instance of the sponge.
    fn new(permutation: &Self::Config) -> Self {
        Self {
            state: PoseidonVector::default(),
            permutation: permutation.clone(),
        }
    }

    /// Absorb an input into the sponge.
    /// This function will absorb the entire input, in chunks of `RATE`,
    /// even if the input lenght is not a multiple of `RATE`.
    fn absorb(&mut self, input: &impl Absorb) {
        let input_field_elements = input.to_sponge_field_elements_as_vec();

        // Absorb input.
        input_field_elements.chunks(RATE).for_each(|chunk| {
            self.state.add_assign_elems(chunk, RATE);
            self.permutation.eval(&mut self.state)
        });
    }

    /// WARNING! This trait method is unimplemented and should not be used.
    /// Only use the `CryptographicSponge` for squeezing native field elements.
    fn squeeze_bytes(&mut self, _num_bytes: usize) -> Vec<u8> {
        unimplemented!("Currently we only support squeezing native field elements!")
    }

    /// WARNING! This trait method is unimplemented and should not be used.
    /// Only use the `CryptographicSponge` for squeezing native field elements.
    fn squeeze_bits(&mut self, _num_bits: usize) -> Vec<bool> {
        unimplemented!("Currently we only support squeezing native field elements!")
    }

    /// WARNING! This trait method is unimplemented and should not be used.
    /// Use `squeeze_native_field_elements` instead.
    fn squeeze_field_elements_with_sizes<F: PrimeField>(
        &mut self,
        _sizes: &[FieldElementSize],
    ) -> Vec<F> {
        unimplemented!("Currently we only support squeezing native field elements!")
    }

    /// WARNING! This trait method is unimplemented and should not be used.
    /// Use `squeeze_native_field_elements` instead.
    fn squeeze_field_elements<F: PrimeField>(&mut self, _num_elements: usize) -> Vec<F> {
        unimplemented!("Currently we only support squeezing native field elements!")
    }

    /// Creates a new sponge with applied domain separation.
    fn fork(&self, domain: &[u8]) -> Self {
        let mut new_sponge = self.clone();

        let mut input = Absorb::to_sponge_bytes_as_vec(&domain.len());
        input.extend_from_slice(domain);
        new_sponge.absorb(&input);

        new_sponge
    }
}

/// The interface for field-based cryptographic sponge.
/// `T` is the native field used by the cryptographic sponge implementation.
impl<T: PoseidonParams, const RATE: usize> FieldBasedCryptographicSponge<T>
    for PoseidonSponge<T, RATE>
{
    /// Squeeze `num_elements` field elements from the sponge.
    fn squeeze_native_field_elements(&mut self, num_elements: usize) -> Vec<T> {
        // SQUEEZE PHASE
        let mut result = vec![];
        let mut remaining = num_elements;
        // extract current rate before calling Poseidon permutation again
        loop {
            let extract = remaining.min(RATE);
            result.extend_from_slice(&self.state.vec[0..extract]);
            remaining -= extract;
            if remaining == 0 {
                break;
            }
            self.permutation.eval(&mut self.state)
        }
        result
    }

    /// WARNING! This trait method is unimplemented and should not be used.
    /// Use `squeeze_native_field_elements` instead.
    fn squeeze_native_field_elements_with_sizes(&mut self, _sizes: &[FieldElementSize]) -> Vec<T> {
        unimplemented!("Currently we only support squeezing native field elements!")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::poseidon::{FieldHasher, Poseidon, STATE_SIZE};
    use ark_bn254::Fr;
    use ark_crypto_primitives::{
        absorb, collect_sponge_bytes, collect_sponge_field_elements, sponge::AbsorbWithLength,
    };
    use ark_ff::{One, UniformRand};
    use ark_std::Zero;
    use itertools::izip;
    use jf_utils::test_rng;

    fn assert_different_encodings<F: PoseidonParams, A: Absorb>(a: &A, b: &A) {
        let bytes1 = a.to_sponge_bytes_as_vec();
        let bytes2 = b.to_sponge_bytes_as_vec();
        assert_ne!(bytes1, bytes2);

        let sponge_param = PoseidonPerm::perm().unwrap();
        let mut sponge1 = PoseidonSponge::<F, 3>::new(&sponge_param);
        let mut sponge2 = PoseidonSponge::<F, 3>::new(&sponge_param);

        sponge1.absorb(&a);
        sponge2.absorb(&b);

        assert_ne!(
            sponge1.squeeze_native_field_elements(3),
            sponge2.squeeze_native_field_elements(3)
        );
    }

    #[test]
    fn single_field_element() {
        let mut rng = test_rng();
        let elem1 = Fr::rand(&mut rng);
        let elem2 = elem1 + Fr::one();

        assert_different_encodings::<Fr, _>(&elem1, &elem2)
    }

    #[test]
    fn list_with_constant_size_element() {
        let mut rng = test_rng();
        let lst1: Vec<_> = (0..1024 * 8).map(|_| Fr::rand(&mut rng)).collect();
        let mut lst2 = lst1.to_vec();
        lst2[3] += Fr::one();

        assert_different_encodings::<Fr, _>(&lst1, &lst2)
    }

    struct VariableSizeList(Vec<u8>);

    impl Absorb for VariableSizeList {
        fn to_sponge_bytes(&self, dest: &mut Vec<u8>) {
            self.0.to_sponge_bytes_with_length(dest)
        }

        fn to_sponge_field_elements<F: PrimeField>(&self, dest: &mut Vec<F>) {
            self.0.to_sponge_field_elements_with_length(dest)
        }
    }

    #[test]
    fn list_with_nonconstant_size_element() {
        let lst1 = vec![
            VariableSizeList(vec![1u8, 2, 3, 4]),
            VariableSizeList(vec![5, 6]),
        ];
        let lst2 = vec![
            VariableSizeList(vec![1u8, 2]),
            VariableSizeList(vec![3, 4, 5, 6]),
        ];

        assert_different_encodings::<Fr, _>(&lst1, &lst2);
    }

    #[test]
    fn test_macros() {
        let sponge_param = PoseidonPerm::perm().unwrap();
        let mut sponge1 = PoseidonSponge::<Fr, 3>::new(&sponge_param);
        sponge1.absorb(&vec![1u8, 2, 3, 4, 5, 6]);
        sponge1.absorb(&Fr::from(114514u128));

        let mut sponge2 = PoseidonSponge::<Fr, 3>::new(&sponge_param);
        absorb!(&mut sponge2, vec![1u8, 2, 3, 4, 5, 6], Fr::from(114514u128));

        let expected = sponge1.squeeze_native_field_elements(3);
        let actual = sponge2.squeeze_native_field_elements(3);

        assert_eq!(actual, expected);

        let mut expected = Vec::new();
        vec![6u8, 5, 4, 3, 2, 1].to_sponge_bytes(&mut expected);
        Fr::from(42u8).to_sponge_bytes(&mut expected);

        let actual = collect_sponge_bytes!(vec![6u8, 5, 4, 3, 2, 1], Fr::from(42u8));

        assert_eq!(actual, expected);

        let mut expected: Vec<Fr> = Vec::new();
        vec![6u8, 5, 4, 3, 2, 1].to_sponge_field_elements(&mut expected);
        Fr::from(42u8).to_sponge_field_elements(&mut expected);

        let actual: Vec<Fr> =
            collect_sponge_field_elements!(vec![6u8, 5, 4, 3, 2, 1], Fr::from(42u8));

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_against_hash() {
        let sponge_perm = PoseidonPerm::perm().unwrap();
        // We check that doing one round absorb and squeeze is the same as hashing.
        // We do this for various different input and squeeze sizes.
        let mut rng = test_rng();
        for (_, i, j) in izip!(0..10, 1..STATE_SIZE, 1..STATE_SIZE) {
            let mut input = (0..i).map(|_| Fr::rand(&mut rng)).collect::<Vec<Fr>>();
            let mut sponge = PoseidonSponge::<Fr, CRHF_RATE>::new(&sponge_perm);
            sponge.absorb(&input);

            let actual = sponge.squeeze_native_field_elements(j)[0];

            let poseidon = Poseidon::<Fr>::new();
            input.resize(CRHF_RATE, Fr::zero());
            let expected = poseidon.hash(&input).unwrap();

            assert_eq!(actual, expected);
        }
    }
}
