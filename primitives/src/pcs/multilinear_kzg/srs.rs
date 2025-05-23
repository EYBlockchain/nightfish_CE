// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Implementing Structured Reference Strings for multilinear polynomial KZG
use crate::pcs::{
    prelude::PCSError,
    univariate_kzg::srs::{
        UnivariateProverParam, UnivariateUniversalParams, UnivariateVerifierParam,
    },
    StructuredReferenceString,
};
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{format, vec::Vec};

/// Evaluations over {0,1}^n for G1 or G2
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct Evaluations<C: AffineRepr> {
    /// The evaluations.
    pub evals: Vec<C>,
}

/// Universal Parameter
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MultilinearUniversalParams<E: Pairing> {
    /// prover parameters
    pub prover_param: MultilinearProverParam<E>,
    /// h^randomness: h^t1, h^t2, ..., **h^{t_nv}**
    pub h_mask: Vec<E::G2Affine>,
}

/// Prover Config
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct MultilinearProverParam<E: Pairing> {
    /// number of variables
    pub num_vars: usize,
    /// `pp_{0}`, `pp_{1}`, ...,pp_{nu_vars} defined
    /// by XZZPD19 where pp_{nv-0}=g and
    /// pp_{nv-i}=g^{eq((t_1,..t_i),(X_1,..X_i))}
    pub powers_of_g: Vec<Evaluations<E::G1Affine>>,
    /// generator for G1
    pub g: E::G1Affine,
    /// generator for G2
    pub h: E::G2Affine,
}

impl<E: Pairing> Default for MultilinearProverParam<E> {
    fn default() -> Self {
        Self {
            num_vars: 0,
            powers_of_g: Vec::<Evaluations<E::G1Affine>>::new(),
            g: E::G1Affine::default(),
            h: E::G2Affine::default(),
        }
    }
}

/// Verifier Config
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct MultilinearVerifierParam<E: Pairing> {
    /// number of variables
    pub num_vars: usize,
    /// generator of G1
    pub g: E::G1Affine,
    /// generator of G2
    pub h: E::G2Affine,
    /// h^randomness: h^t1, h^t2, ..., **h^{t_nv}**
    pub h_mask: Vec<E::G2Affine>,
}

impl<E: Pairing> Default for MultilinearVerifierParam<E> {
    fn default() -> Self {
        Self {
            num_vars: 0,
            g: E::G1Affine::default(),
            h: E::G2Affine::default(),
            h_mask: Vec::<E::G2Affine>::new(),
        }
    }
}

impl<E: Pairing> StructuredReferenceString for MultilinearUniversalParams<E> {
    type ProverParam = MultilinearProverParam<E>;
    type VerifierParam = MultilinearVerifierParam<E>;
    type Item = E::G1Affine;

    /// Extract the prover parameters from the public parameters.
    fn extract_prover_param(&self, supported_num_vars: usize) -> Self::ProverParam {
        let to_reduce = self.prover_param.num_vars - supported_num_vars;

        Self::ProverParam {
            powers_of_g: self.prover_param.powers_of_g[to_reduce..].to_vec(),
            g: self.prover_param.g,
            h: self.prover_param.h,
            num_vars: supported_num_vars,
        }
    }

    /// Extract the verifier parameters from the public parameters.
    fn extract_verifier_param(&self, supported_num_vars: usize) -> Self::VerifierParam {
        let to_reduce = self.prover_param.num_vars - supported_num_vars;
        Self::VerifierParam {
            num_vars: supported_num_vars,
            g: self.prover_param.g,
            h: self.prover_param.h,
            h_mask: self.h_mask[to_reduce..].to_vec(),
        }
    }

    fn g(vp: &Self::VerifierParam) -> Self::Item {
        vp.g
    }

    /// Trim the universal parameters to specialize the public parameters
    /// for multilinear polynomials to the given `supported_num_vars`, and
    /// returns committer key and verifier key. `supported_num_vars` should
    /// be in range `1..=params.num_vars`
    fn trim(
        &self,
        supported_num_vars: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        if supported_num_vars > self.prover_param.num_vars {
            return Err(PCSError::InvalidParameters(format!(
                "SRS does not support target number of vars {supported_num_vars}"
            )));
        }

        let to_reduce = self.prover_param.num_vars - supported_num_vars;
        let ck = Self::ProverParam {
            powers_of_g: self.prover_param.powers_of_g[to_reduce..].to_vec(),
            g: self.prover_param.g,
            h: self.prover_param.h,
            num_vars: supported_num_vars,
        };
        let vk = Self::VerifierParam {
            num_vars: supported_num_vars,
            g: self.prover_param.g,
            h: self.prover_param.h,
            h_mask: self.h_mask[to_reduce..].to_vec(),
        };
        Ok((ck, vk))
    }

    #[cfg(any(test, feature = "test-srs"))]
    fn gen_srs_for_testing<R>(rng: &mut R, num_vars: usize) -> Result<Self, PCSError>
    where
        R: ark_std::rand::RngCore + ark_std::rand::CryptoRng,
    {
        tests::gen_srs_for_testing(rng, num_vars)
    }
}

// Implement `trait StructuredReferenceString` for (ML_pp, Uni_pp) to be used in
// MLE PCS.
impl<E: Pairing> StructuredReferenceString
    for (MultilinearUniversalParams<E>, UnivariateUniversalParams<E>)
{
    type ProverParam = (MultilinearProverParam<E>, UnivariateProverParam<E>);
    type VerifierParam = (MultilinearVerifierParam<E>, UnivariateVerifierParam<E>);
    type Item = (E::G1Affine, E::G1Affine);

    fn trim(
        &self,
        supported_degree: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        let ml_pp = <MultilinearUniversalParams<E> as StructuredReferenceString>::trim(
            &self.0,
            supported_degree,
        )?;
        let uni_pp = <UnivariateUniversalParams<E> as StructuredReferenceString>::trim(
            &self.1,
            supported_degree,
        )?;

        Ok(((ml_pp.0, uni_pp.0), (ml_pp.1, uni_pp.1)))
    }

    fn extract_prover_param(&self, supported_degree: usize) -> Self::ProverParam {
        let ml_prover_param =
            <MultilinearUniversalParams<E> as StructuredReferenceString>::extract_prover_param(
                &self.0,
                supported_degree,
            );
        let uni_prover_param =
            <UnivariateUniversalParams<E> as StructuredReferenceString>::extract_prover_param(
                &self.1,
                1 << supported_degree,
            );

        (ml_prover_param, uni_prover_param)
    }

    fn extract_verifier_param(&self, supported_degree: usize) -> Self::VerifierParam {
        let ml_verifier_param =
            <MultilinearUniversalParams<E> as StructuredReferenceString>::extract_verifier_param(
                &self.0,
                supported_degree,
            );
        let uni_verifier_param =
            <UnivariateUniversalParams<E> as StructuredReferenceString>::extract_verifier_param(
                &self.1,
                1 << supported_degree,
            );

        (ml_verifier_param, uni_verifier_param)
    }

    fn g(vp: &Self::VerifierParam) -> Self::Item {
        (vp.0.g, vp.1.g)
    }

    #[cfg(any(test, feature = "test-srs"))]
    fn gen_srs_for_testing<R>(rng: &mut R, supported_degree: usize) -> Result<Self, PCSError>
    where
        R: ark_std::rand::RngCore + ark_std::rand::CryptoRng,
    {
        let ml_pp =
            <MultilinearUniversalParams<E> as StructuredReferenceString>::gen_srs_for_testing(
                rng,
                supported_degree,
            )?;
        let uni_pp =
            <UnivariateUniversalParams<E> as StructuredReferenceString>::gen_srs_for_testing(
                rng,
                supported_degree,
            )?;
        Ok((ml_pp, uni_pp))
    }
}

#[cfg(any(test, feature = "test-srs"))]
mod tests {
    use super::*;
    use crate::pcs::multilinear_kzg::util::eq_eval;
    use ark_ec::{scalar_mul::fixed_base::FixedBase, CurveGroup};
    use ark_ff::{Field, PrimeField, Zero};
    use ark_poly::DenseMultilinearExtension;
    use ark_std::{
        collections::LinkedList,
        end_timer,
        iter::FromIterator,
        rand::{CryptoRng, RngCore},
        start_timer,
        string::ToString,
        vec,
        vec::Vec,
        UniformRand,
    };

    // fix first `pad` variables of `poly` represented in evaluation form to zero
    fn remove_dummy_variable<F: Field>(poly: &[F], pad: usize) -> Result<Vec<F>, PCSError> {
        if pad == 0 {
            return Ok(poly.to_vec());
        }
        if !poly.len().is_power_of_two() {
            return Err(PCSError::InvalidParameters(
                "Size of polynomial should be power of two.".to_string(),
            ));
        }
        let nv = ark_std::log2(poly.len()) as usize - pad;

        Ok((0..(1 << nv)).map(|x| poly[x << pad]).collect())
    }

    // Generate eq(t,x), a product of multilinear polynomials with fixed t.
    // eq(a,b) is takes extensions of a,b in {0,1}^num_vars such that if a and
    // b in {0,1}^num_vars are equal then this polynomial evaluates to 1.
    fn eq_extension<F: PrimeField>(t: &[F]) -> Vec<DenseMultilinearExtension<F>> {
        let start = start_timer!(|| "eq extension");

        let dim = t.len();
        let mut result = Vec::new();
        for (i, &ti) in t.iter().enumerate().take(dim) {
            let mut poly = Vec::with_capacity(1 << dim);
            for x in 0u64..(1 << dim) {
                let xi = if x >> i & 1 == 1 { F::one() } else { F::zero() };
                let ti_xi = ti * xi;
                poly.push(ti_xi + ti_xi - xi - ti + F::one());
            }
            result.push(DenseMultilinearExtension::from_evaluations_vec(dim, poly));
        }

        end_timer!(start);
        result
    }

    pub(crate) fn gen_srs_for_testing<E: Pairing, R: RngCore + CryptoRng>(
        rng: &mut R,
        num_vars: usize,
    ) -> Result<MultilinearUniversalParams<E>, PCSError> {
        if num_vars == 0 {
            return Err(PCSError::InvalidParameters(
                "constant polynomial not supported".to_string(),
            ));
        }

        let total_timer = start_timer!(|| "SRS generation");

        let pp_generation_timer = start_timer!(|| "Prover Param generation");

        let g = E::G1::rand(rng);
        let h = E::G2::rand(rng);

        let mut powers_of_g = Vec::new();

        let t: Vec<_> = (0..num_vars).map(|_| E::ScalarField::rand(rng)).collect();
        let scalar_bits = E::ScalarField::MODULUS_BIT_SIZE as usize;

        let mut eq: LinkedList<DenseMultilinearExtension<E::ScalarField>> =
            LinkedList::from_iter(eq_extension(&t));
        let mut eq_arr = LinkedList::new();
        let mut base = eq.pop_back().unwrap().evaluations;

        for i in (0..num_vars).rev() {
            eq_arr.push_front(remove_dummy_variable(&base, i)?);
            if i != 0 {
                let mul = eq.pop_back().unwrap().evaluations;
                base = base
                    .into_iter()
                    .zip(mul.into_iter())
                    .map(|(a, b)| a * b)
                    .collect();
            }
        }

        let mut pp_powers = Vec::new();
        let mut total_scalars = 0;
        for i in 0..num_vars {
            let eq = eq_arr.pop_front().unwrap();
            let pp_k_powers = (0..(1 << (num_vars - i))).map(|x| eq[x]);
            pp_powers.extend(pp_k_powers);
            total_scalars += 1 << (num_vars - i);
        }
        let window_size = FixedBase::get_mul_window_size(total_scalars);
        let g_table = FixedBase::get_window_table(scalar_bits, window_size, g);

        let pp_g = E::G1::normalize_batch(&FixedBase::msm(
            scalar_bits,
            window_size,
            &g_table,
            &pp_powers,
        ));

        let mut start = 0;
        for i in 0..num_vars {
            let size = 1 << (num_vars - i);
            let pp_k_g = Evaluations {
                evals: pp_g[start..(start + size)].to_vec(),
            };
            // check correctness of pp_k_g
            let t_eval_0 = eq_eval(&vec![E::ScalarField::zero(); num_vars - i], &t[i..num_vars])?;
            assert_eq!((g * t_eval_0).into_affine(), pp_k_g.evals[0]);

            powers_of_g.push(pp_k_g);
            start += size;
        }
        let gg = Evaluations {
            evals: [g.into_affine()].to_vec(),
        };
        powers_of_g.push(gg);

        let pp = MultilinearProverParam {
            num_vars,
            g: g.into_affine(),
            h: h.into_affine(),
            powers_of_g,
        };

        end_timer!(pp_generation_timer);

        let vp_generation_timer = start_timer!(|| "VP generation");
        let h_mask = {
            let window_size = FixedBase::get_mul_window_size(num_vars);
            let h_table = FixedBase::get_window_table(scalar_bits, window_size, h);
            E::G2::normalize_batch(&FixedBase::msm(scalar_bits, window_size, &h_table, &t))
        };
        end_timer!(vp_generation_timer);
        end_timer!(total_timer);
        Ok(MultilinearUniversalParams {
            prover_param: pp,
            h_mask,
        })
    }

    #[test]
    fn test_srs_gen() -> Result<(), PCSError> {
        use ark_bls12_381::Bls12_381;
        use jf_utils::test_rng;
        type E = Bls12_381;

        let mut rng = test_rng();
        for nv in 4..10 {
            let _ = MultilinearUniversalParams::<E>::gen_srs_for_testing(&mut rng, nv)?;
        }

        Ok(())
    }
}
