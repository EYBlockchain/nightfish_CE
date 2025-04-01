#![allow(dead_code)]
// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Useful utilities for KZG PCS

use crate::pcs::prelude::PCSError;
use ark_ff::PrimeField;
use ark_poly::{
    univariate::DensePolynomial, DenseMultilinearExtension, EvaluationDomain, Evaluations,
    MultilinearExtension, Polynomial, Radix2EvaluationDomain,
};
use ark_std::{end_timer, format, log2, start_timer, string::ToString, vec, vec::Vec};

use super::MLE;

/// Evaluate eq polynomial. use the public one later
#[cfg(any(test, feature = "test-srs"))]
pub(crate) fn eq_eval<F: PrimeField>(x: &[F], y: &[F]) -> Result<F, PCSError> {
    if x.len() != y.len() {
        return Err(PCSError::InvalidParameters(
            "x and y have different length".to_string(),
        ));
    }
    let start = start_timer!(|| "eq_eval");
    let mut res = F::one();
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let xi_yi = xi * yi;
        res *= xi_yi + xi_yi - xi - yi + F::one();
    }
    end_timer!(start);
    Ok(res)
}

/// Decompose an integer into a binary vector in little endian.
pub(crate) fn bit_decompose(input: u64, num_var: usize) -> Vec<bool> {
    let mut res = Vec::with_capacity(num_var);
    let mut i = input;
    for _ in 0..num_var {
        res.push(i & 1 == 1);
        i >>= 1;
    }
    res
}

/// For an MLE w with `mle_num_vars` variables, and `point_len` number of
/// points, compute the degree of the univariate polynomial `q(x):= w(l(x))`
/// where l(x) is a list of polynomials that go through all points.
// uni_degree is computed as `mle_num_vars * point_len`:
// - each l(x) is of degree `point_len`
// - mle has degree one
// - worst case is `\prod_{i=0}^{mle_num_vars-1} l_i(x) < point_len * mle_num_vars`
#[inline]
#[cfg(test)]
pub fn compute_qx_degree(mle_num_vars: usize, point_len: usize) -> usize {
    mle_num_vars * point_len
}

/// get the domain for the univariate polynomial
#[inline]
pub(crate) fn get_uni_domain<F: PrimeField>(
    uni_poly_degree: usize,
) -> Result<Radix2EvaluationDomain<F>, PCSError> {
    let domain = match Radix2EvaluationDomain::<F>::new(uni_poly_degree) {
        Some(p) => p,
        None => {
            return Err(PCSError::InvalidParameters(
                "failed to build radix 2 domain".to_string(),
            ))
        },
    };
    Ok(domain)
}

/// Compute W \circ l.
///
/// Given an MLE W, and a list of univariate polynomials l, generate the
/// univariate polynomial that composes W with l.
///
/// Returns an error if l's length does not matches number of variables in W.
pub(crate) fn compute_w_circ_l<F: PrimeField>(
    w: &DenseMultilinearExtension<F>,
    l: &[DensePolynomial<F>],
) -> Result<DensePolynomial<F>, PCSError> {
    let timer = start_timer!(|| "compute W \\circ l");

    if w.num_vars != l.len() {
        return Err(PCSError::InvalidParameters(format!(
            "l's length ({}) does not match num_variables ({})",
            l.len(),
            w.num_vars(),
        )));
    }

    let mut res_eval: Vec<F> = vec![];

    // TODO: consider to pass this in from caller
    // uni_degree is (product of each prefix's) + (2 * MLEs)
    // = (l.len() - (num_vars - log(l.len())) + 2) * l[0].degree
    let uni_degree = (l.len() - w.num_vars + log2(l.len()) as usize + 2) * l[0].degree();

    let domain = match Radix2EvaluationDomain::<F>::new(uni_degree) {
        Some(p) => p,
        None => {
            return Err(PCSError::InvalidParameters(
                "failed to build radix 2 domain".to_string(),
            ))
        },
    };
    for point in domain.elements() {
        // we reverse the order here because the coefficient vec are stored in
        // bit-reversed order
        let l_eval: Vec<F> = l.iter().rev().map(|x| x.evaluate(&point)).collect();
        res_eval.push(w.evaluate(l_eval.as_ref()).unwrap())
    }
    let evaluation = Evaluations::from_vec_and_domain(res_eval, domain);
    let res = evaluation.interpolate();

    end_timer!(timer);
    Ok(res)
}

/// Return the number of variables that one need for an MLE to
/// batch the list of MLEs
#[inline]
pub fn get_batched_nv(num_var: usize, polynomials_len: usize) -> usize {
    num_var + log2(polynomials_len) as usize
}

/// merge a set of polynomials. Returns an error if the
/// polynomials do not share a same number of nvs.
pub fn merge_polynomials<F: PrimeField>(
    polynomials: &[MLE<F>],
) -> Result<DenseMultilinearExtension<F>, PCSError> {
    let nv = polynomials[0].num_vars();
    for poly in polynomials.iter() {
        if nv != poly.num_vars() {
            return Err(PCSError::InvalidParameters(
                "num_vars do not match for polynomials".to_string(),
            ));
        }
    }

    let merged_nv = get_batched_nv(nv, polynomials.len());
    let mut scalars = vec![];
    for poly in polynomials.iter() {
        scalars.extend_from_slice(poly.to_evaluations().as_slice());
    }
    scalars.extend_from_slice(vec![F::zero(); (1 << merged_nv) - scalars.len()].as_ref());
    Ok(DenseMultilinearExtension::from_evaluations_vec(
        merged_nv, scalars,
    ))
}

/// Given a list of points, build `l(points)` which is a list of univariate
/// polynomials that goes through the points
pub(crate) fn build_l<F: PrimeField>(
    num_var: usize,
    points: &[Vec<F>],
    domain: &Radix2EvaluationDomain<F>,
) -> Result<Vec<DensePolynomial<F>>, PCSError> {
    let prefix_len = log2(points.len()) as usize;
    let mut uni_polys = Vec::new();

    // 1.1 build the indexes and the univariate polys that go through the indexes
    let indexes: Vec<Vec<bool>> = (0..points.len())
        .map(|x| bit_decompose(x as u64, prefix_len))
        .collect();
    for i in 0..prefix_len {
        let eval: Vec<F> = indexes
            .iter()
            .map(|x| F::from(x[prefix_len - i - 1]))
            .collect();

        uni_polys.push(Evaluations::from_vec_and_domain(eval, *domain).interpolate());
    }

    // 1.2 build the actual univariate polys that go through the points
    for i in 0..num_var {
        let mut eval: Vec<F> = points.iter().map(|x| x[i]).collect();
        eval.extend_from_slice(vec![F::zero(); domain.size as usize - eval.len()].as_slice());
        uni_polys.push(Evaluations::from_vec_and_domain(eval, *domain).interpolate())
    }

    Ok(uni_polys)
}

/// Input a list of multilinear polynomials and a list of points,
/// generate a list of evaluations.
// Note that this function is only used for testing verifications.
// In practice verifier does not see polynomials, and the `mle_values`
// are included in the `batch_proof`.
#[cfg(test)]
pub(crate) fn generate_evaluations<F: PrimeField>(
    polynomials: &[MLE<F>],
    points: &[Vec<F>],
) -> Result<Vec<F>, PCSError> {
    if polynomials.len() != points.len() {
        return Err(PCSError::InvalidParameters(
            "polynomial length does not match point length".to_string(),
        ));
    }

    let num_var = polynomials[0].num_vars;
    let uni_poly_degree = points.len();
    let merge_poly = merge_polynomials(polynomials)?;

    let domain = get_uni_domain::<F>(uni_poly_degree)?;
    let uni_polys = build_l(num_var, points, &domain)?;
    let mut mle_values = vec![];

    for i in 0..uni_poly_degree {
        let point: Vec<F> = uni_polys
            .iter()
            .rev()
            .map(|poly| poly.evaluate(&domain.element(i)))
            .collect();

        let mle_value = merge_poly.evaluate(&point).unwrap();
        mle_values.push(mle_value)
    }
    Ok(mle_values)
}
