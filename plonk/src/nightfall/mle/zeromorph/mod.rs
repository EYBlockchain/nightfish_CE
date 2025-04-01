//! This module containes the implementation of the ZeromorphHelper struct which
//! contains several methods used in the Zeromorph protocol.

#![allow(dead_code)]
use ark_ff::{PrimeField, Zero};
use ark_poly::{
    univariate::DensePolynomial, DenseMultilinearExtension, DenseUVPolynomial, MultilinearExtension,
};
use ark_std::{ops::AddAssign, ops::Mul, vec, vec::Vec};

use core::marker::PhantomData;

pub mod zeromorph_protocol;
pub use zeromorph_protocol::Zeromorph;

/// Struct used to help with the Zeromorph protocol.
pub struct ZeromorphHelper<F: PrimeField> {
    /// Phantom data used so we can have an associated field.
    _phantom: PhantomData<F>,
}

impl<F: PrimeField> ZeromorphHelper<F> {
    fn _new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Returns 1 + X + ... + X^{2^num_vars - 1}
    fn phi_poly(num_vars: usize) -> DensePolynomial<F> {
        DensePolynomial::<F> {
            coeffs: vec![F::one(); 1 << num_vars],
        }
    }

    /// The efficient evaluation of phi_poly
    fn eval_phi_poly(num_vars: usize, point: &F) -> Option<F> {
        let x_pow = point.pow([1 << num_vars]);
        Some((x_pow - F::one()) * (*point - F::one()).inverse()?)
    }

    /// Performs the zeromorph isomorphism on the given multilinear extension.
    pub fn isom(mle: &DenseMultilinearExtension<F>) -> DensePolynomial<F> {
        DensePolynomial::<F>::from_coefficients_slice(&mle.evaluations)
    }

    /// Outputs polynomial f(X_0, ..., X_{k-1}, u_0, ..., u_{n-k-1})
    /// compare with the DenseMultilinearExtension method fix_variables
    /// which outputs f(u_0, ..., u_{n-k-1}, X_0, ..., X_{k-1})
    fn fix_variables_reverse(
        mle: &DenseMultilinearExtension<F>,
        partial_point: &[F],
    ) -> Option<DenseMultilinearExtension<F>> {
        if partial_point.len() > mle.num_vars {
            return None;
        }
        let mut evals = mle.evaluations.to_vec();
        let nv = mle.num_vars;
        let dim = partial_point.len();
        for i in 1..dim + 1 {
            let r = partial_point[dim - i];
            for b in 0..(1 << (nv - i)) {
                let left = evals[b];
                let right = evals[b + (1 << (nv - i))];
                evals[b] = left + r * (right - left);
            }
        }
        Some(DenseMultilinearExtension::<F>::from_evaluations_slice(
            nv - dim,
            &evals[..(1 << (nv - dim))],
        ))
    }
    /// Function to compute the q_k's as used in the zeromorph protocol.
    fn compute_qks(
        mle: &DenseMultilinearExtension<F>,
        u: &[F],
    ) -> Option<Vec<DenseMultilinearExtension<F>>> {
        if u.len() != mle.num_vars {
            return None;
        }
        let mut vec_q: Vec<DenseMultilinearExtension<F>> = Vec::new();
        let nv = mle.num_vars;
        for k in 0..nv {
            let mut u_slice = u[k..nv].to_vec();
            u_slice[0] += F::one();
            let left = Self::fix_variables_reverse(mle, &u_slice)?;
            u_slice[0] -= F::one();
            let right = Self::fix_variables_reverse(mle, &u_slice)?;
            vec_q.push(left - right);
        }
        Some(vec_q)
    }

    /// Function to compute the Z_x function used in the zeromorph protocol.
    /// We input the qk's as computed by the compute_qks function to avoid
    /// multiple calls to compute_qks.
    fn z_poly(
        mle: &DenseMultilinearExtension<F>,
        u: &[F],
        x: &F,
        qks: &[DenseMultilinearExtension<F>],
    ) -> Option<DensePolynomial<F>> {
        if u.len() != mle.num_vars || qks.len() != mle.num_vars {
            return None;
        }
        for (k, qk) in qks.iter().enumerate().take(mle.num_vars) {
            if qk.num_vars != k {
                return None;
            }
        }
        let v = mle.evaluate(u)?;
        let nv = mle.num_vars;
        let mut rhs_sum = DensePolynomial::<F>::zero();
        let mut x_pow: F = *x;
        for k in 0..nv {
            let scalar = u[k] * Self::eval_phi_poly(nv - k, &x_pow)?
                - x_pow * Self::eval_phi_poly(nv - k - 1, &(x_pow * x_pow))?;
            rhs_sum.add_assign(&Self::isom(&qks[k]).mul(scalar));
            x_pow *= x_pow;
        }
        let const_poly =
            DensePolynomial::<F>::from_coefficients_slice(&[Self::eval_phi_poly(nv, x)?.mul(-v)]);
        Some(Self::isom(mle) + const_poly + rhs_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;
    use ark_poly::Polynomial;
    use ark_std::{test_rng, UniformRand};

    #[test]
    fn phi_poly_test() {
        let rng = &mut test_rng();
        let max_num_vars = 20;
        for _ in 0..10 {
            let num_vars = (usize::rand(rng) % (max_num_vars - 10)) + 10;
            let point = Fq::rand(rng);
            let phi_poly = ZeromorphHelper::<Fq>::phi_poly(num_vars);
            let eval_phi_poly = ZeromorphHelper::<Fq>::eval_phi_poly(num_vars, &point).unwrap();
            assert_eq!(phi_poly.evaluate(&point), eval_phi_poly);
        }
    }

    #[test]
    fn zx_test() {
        let mut rng = test_rng();
        let n = 20;
        for _ in 0..10 {
            let mle = DenseMultilinearExtension::<Fq>::from_evaluations_vec(
                n,
                (0..(1 << n)).map(|_| Fq::rand(&mut rng)).collect(),
            );
            let u: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();
            let x = Fq::rand(&mut rng);
            let qks_wrapper = ZeromorphHelper::<Fq>::compute_qks(&mle, &u);
            assert!(qks_wrapper.is_some(), "q_wrapper is None");
            let qks = qks_wrapper.unwrap();
            let zx_wrapper = ZeromorphHelper::<Fq>::z_poly(&mle, &u, &x, &qks);
            assert!(zx_wrapper.is_some(), "zx_wrapper is None");
            let zx = zx_wrapper.unwrap();
            let result = zx.evaluate(&x);
            assert_eq!(result, Fq::zero());
        }
    }

    #[test]
    fn qk_test() {
        let mut rng = test_rng();
        let n = 20;
        for _ in 0..10 {
            let mle = DenseMultilinearExtension::<Fq>::from_evaluations_vec(
                n,
                (0..(1 << n)).map(|_| Fq::rand(&mut rng)).collect(),
            );
            let u: Vec<Fq> = (0..n).map(|_| Fq::rand(&mut rng)).collect();
            let qks_wrapper = ZeromorphHelper::<Fq>::compute_qks(&mle, &u);
            assert!(qks_wrapper.is_some(), "q_wrapper is None");
            let qks = qks_wrapper.unwrap();
            let mut sum_vec: Vec<Fq> = vec![Fq::zero(); 1 << n];
            for k in 0..n {
                sum_vec = sum_vec
                    .iter()
                    .enumerate()
                    .map(|(b, &x)| {
                        x + qks[k][b % (1 << k)] * (Fq::from(((b >> k) & 1) as u32) - u[k])
                    })
                    .collect();
            }
            let v = mle.evaluate(&u).unwrap();
            sum_vec = sum_vec.into_iter().map(|x| x + v).collect();
            assert_eq!(mle.evaluations, sum_vec);
        }
    }
}
