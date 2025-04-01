//! Utility functions used in MLE-Plonk.

use super::{
    mle_structs::{
        GateInfo, MLEChallenges, MLELookupEvals, MLELookupProvingKey, MLEProofEvals,
        PolynomialError,
    },
    subroutines::lookupcheck::LogUpTable,
    virtual_polynomial::VirtualPolynomial,
};

use ark_ff::{Field, PrimeField};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};

use ark_std::{cfg_into_iter, cfg_iter_mut};
use dashmap::DashMap;

use jf_primitives::{pcs::PolynomialCommitmentScheme, rescue::RescueParameter};
use jf_relation::gadgets::{ecc::HasTEForm, EmulationConfig};
use rayon::prelude::*;

use ark_std::{cfg_iter, format, string::ToString, sync::Arc, vec, vec::Vec, One, Zero};

use crate::nightfall::mle::subroutines::{
    sumcheck::SumCheck, PolyOracle, SumCheckProof, VPSumCheck,
};
use crate::transcript::{RescueTranscript, Transcript};
use crate::{errors::PlonkError, nightfall::accumulation::AccumulationChallengesAndScalars};

/// This function computes the barycentric weights for a set of points.
pub fn compute_barycentric_weights<F: Field>(points: &[F]) -> Result<Vec<F>, PolynomialError> {
    cfg_iter!(points)
        .map(|point_i| {
            let weight = points
                .iter()
                .filter(|point_j| (*point_i != **point_j))
                .map(|point_j| *point_i - *point_j)
                .product::<F>();
            weight.inverse().ok_or(PolynomialError::UnreachableError)
        })
        .collect::<Result<Vec<F>, PolynomialError>>()
}

/// \hat f(x) = \sum_{x_i \in eval_x} f(x_i) eq(x, r)
/// where eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
pub fn build_f_hat<F: PrimeField>(
    poly: &VirtualPolynomial<F>,
    r: &[F],
) -> Result<VirtualPolynomial<F>, PolynomialError> {
    if poly.num_vars != r.len() {
        return Err(PolynomialError::ParameterError(
            "the number of variables is different from number of challenge r".to_string(),
        ));
    }

    let eq_x_r = Arc::new(build_eq_x_r(r));

    let mut res = poly.clone();

    res.mul_by_mle(eq_x_r)?;

    Ok(res)
}
/// This function build the eq(x, r) polynomial for any given r.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r<F: PrimeField>(r: &[F]) -> DenseMultilinearExtension<F> {
    DenseMultilinearExtension::from_evaluations_vec(r.len(), build_eq_x_r_vec(r))
}
/// This function build the eq(x, r) polynomial for any given r, and output the
/// evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r_vec<F: PrimeField>(r: &[F]) -> Vec<F> {
    // we build eq(x,r) from its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations
    let final_size = 1usize << r.len();
    let mut evals: Vec<F> = unsafe_allocate_zero_vec(final_size);
    let mut size = 1;
    evals[0] = F::one();

    for r in r.iter() {
        let (evals_left, evals_right) = evals.split_at_mut(size);
        let (evals_right, _) = evals_right.split_at_mut(size);

        evals_left
            .par_iter_mut()
            .zip(evals_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * *r;
                *x -= *y;
            });

        size *= 2;
    }

    evals
}

fn unsafe_allocate_zero_vec<F: PrimeField>(size: usize) -> Vec<F> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    // Check for safety of 0 allocation
    unsafe {
        let value = &F::zero();
        let ptr = value as *const F as *const u8;
        let bytes = ark_std::slice::from_raw_parts(ptr, ark_std::mem::size_of::<F>());
        assert!(bytes.iter().all(|&byte| byte == 0));
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<F>;
    unsafe {
        let layout = ark_std::alloc::Layout::array::<F>(size).unwrap();
        let ptr = ark_std::alloc::alloc_zeroed(layout) as *mut F;

        if ptr.is_null() {
            panic!("Zero vec allocaiton failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}

/// Evaluate eq polynomial.
pub fn eq_eval<F: PrimeField>(x: &[F], y: &[F]) -> Result<F, PolynomialError> {
    if x.len() != y.len() {
        return Err(PolynomialError::ParameterError(
            "x and y have different length".to_string(),
        ));
    }

    let res = cfg_iter!(x)
        .zip(cfg_iter!(y))
        .map(|(xi, yi)| *xi * *yi + *xi * *yi - xi - yi + F::one())
        .product::<F>();

    Ok(res)
}

/// Compute normalized multiplicity polynomial:
///        m(x) = m_f(x) / m_t(x),
/// where m_f(x) = count of value t(x) in lookup f
/// and m_t(x) = count of value t(x) in table t.
pub(super) fn compute_multiplicity_poly<F: PrimeField>(
    lookup_wire: &Arc<DenseMultilinearExtension<F>>,
    table: &LogUpTable<F>,
) -> Result<Arc<DenseMultilinearExtension<F>>, PolynomialError> {
    if lookup_wire.num_vars != table.get_table().num_vars {
        return Err(PolynomialError::ParameterError(
            "lookup_wire and table have different number of variables".to_string(),
        ));
    }
    let num_vars = lookup_wire.num_vars;

    #[cfg(feature = "parallel")]
    return {
        let lookup_multiplicities = DashMap::<F, F>::new();

        // Count number of occurences of each elements
        lookup_wire
            .evaluations
            .par_iter()
            .map(|&eval| {
                if table.get_multiplicity(eval).is_none() {
                    Err(PolynomialError::ParameterError(format!(
                        "Lookup value {eval} is not in table"
                    )))
                } else {
                    *lookup_multiplicities.entry(eval).or_insert_with(F::zero) += F::one();
                    Ok(())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let m_evals = table
            .get_table()
            .evaluations
            .par_iter()
            .map(|&value| {
                if let Some(lkup_m) = lookup_multiplicities.get(&value) {
                    // unwrap is safe because we are iterating over values in the table.
                    *lkup_m / table.get_multiplicity(value).unwrap()
                } else {
                    F::zero()
                }
            })
            .collect::<Vec<F>>();
        Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, m_evals,
        )))
    };

    #[cfg(not(feature = "parallel"))]
    return {
        let mut h_f = HashMap::new();
        let mut h_t = HashMap::new();

        // Count number of occurences of each elements
        for num in table.get_table().to_evaluations().iter() {
            *h_t.entry(*num).or_insert_with(F::zero) += F::one();
        }
        for num in lookup_wire.evaluations.iter() {
            if h_t.get(num).is_none() {
                return Err(PolyIOPErrors::InvalidProof(format!(
                    "Lookup value {num} is not in table"
                )));
            }
            *h_f.entry(*num).or_insert_with(F::zero) += F::one();
        }

        let m_evals = t
            .iter()
            .map(|value| {
                if let Some(h_f_val) = h_f.get(value) {
                    *h_f_val / *h_t.get(value).unwrap()
                } else {
                    F::zero()
                }
            })
            .collect::<Vec<F>>();

        Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, m_evals,
        )))
    };
}
/// Batches multilinear polynomials together into a single Sumcheck proof
/// and a relevant univariate polynomial proof. Note it is assumed the
/// relevant commitments have already been pushed to the transcript.
///
/// We do not assume that all polys have the same number of variables but we
/// treat them as if they do in the obvious way.
///
/// Similarly, we do not assume that we have a power of 2 number of polys
/// but we always make up to the next power of 2 by adding all extra zero
/// polys with zero associated points.
type BatchOpening<F> = (
    SumCheckProof<F, PolyOracle<F>>,
    F,
    Arc<DenseMultilinearExtension<F>>,
    Vec<F>,
);

pub fn mv_batch_open<P>(
    polys: &[Arc<DenseMultilinearExtension<P::ScalarField>>],
    points: &[Vec<P::ScalarField>],
    transcript: &mut RescueTranscript<P::BaseField>,
) -> Result<BatchOpening<P::ScalarField>, PolynomialError>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    let (proof, eval, g_prime_poly, accumulation_challenges_and_scalars) =
        challenges_and_scalars_from_mv_batch_open::<P>(polys, points, transcript)?;
    Ok((
        proof,
        eval,
        g_prime_poly,
        accumulation_challenges_and_scalars.a_2,
    ))
}

/// Batches multilinear polynomials together in preparation for a
/// single Sumcheck verification and a relevant univariate polynomial
/// verification.
type BatchVerification<F> = (F, Vec<F>, Vec<F>);

pub fn mv_batch_verify<P>(
    points: &[Vec<P::ScalarField>],
    values: &[P::ScalarField],
    proof: &SumCheckProof<P::ScalarField, PolyOracle<P::ScalarField>>,
    transcript: &mut RescueTranscript<P::BaseField>,
) -> Result<BatchVerification<P::ScalarField>, PolynomialError>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    if points.len() != values.len() {
        return Err(PolynomialError::ParameterError(format!(
            "Length of points vector {} and values vector {} do not match",
            points.len(),
            values.len()
        )));
    }

    for point in points.iter() {
        if point.len() != points[0].len() {
            return Err(PolynomialError::ParameterError(format!(
                "Length of points vector {} does not match length of first point vector {}",
                point.len(),
                points[0].len()
            )));
        }
    }
    let max_num_vars = points[0].len();

    let num_polys = points.len();

    let l = num_polys.next_power_of_two().ilog2() as usize;

    let t = transcript
        .squeeze_scalar_challenges::<P>(b"t", l)
        .map_err(|_| {
            PolynomialError::ParameterError("Couldn't squeeze a challenge scalar".to_string())
        })?;

    let deferred_check = VPSumCheck::<P>::verify(proof, transcript)
        .map_err(|_| PolynomialError::ParameterError("Couldn't verify Sumcheck".to_string()))?;

    if deferred_check.point.len() != l + max_num_vars {
        return Err(PolynomialError::ParameterError(format!(
            "Length of deferred_check point vector {} does not match l + max_num_vars {}",
            deferred_check.point.len(),
            l + max_num_vars
        )));
    }
    // We will repeatedly need the zero and one of the scalar field.
    let zero = P::ScalarField::zero();
    let one = P::ScalarField::one();

    // We calculate the evaluations for the mle eq_tilde
    let mut eq_tilde_coeff: Vec<P::ScalarField> = vec![zero; 1 << (l + max_num_vars)];
    for i in 0..num_polys {
        for j in 0..(1 << points[i].len()) {
            // The product eq(b,z_i), where b is the binary expansion of j.
            let mut eq_prod = one;
            for k in 0..points[i].len() {
                let field_bit = P::ScalarField::from(((j >> k) & 1) as u8);
                eq_prod *= field_bit * points[i][k] + (one - field_bit) * (one - points[i][k]);
            }
            // When k >= points[i].len(), we are taking points[i][k] to be zero, so multiplier is always 1.
            eq_tilde_coeff[i + (j << l)] = eq_prod;
        }
        // If j >= (1 << points[i].len()), then eq(b,z_i) = 0.
    }
    // If i > num_polys, eq(b,z_i) is only non-zero for b = (0, ... ,0).
    for coeff in eq_tilde_coeff.iter_mut().take(1 << l).skip(num_polys) {
        *coeff = one;
    }
    let eq_tilde_poly = DenseMultilinearExtension::<P::ScalarField>::from_evaluations_vec(
        l + max_num_vars,
        eq_tilde_coeff,
    );
    let a_1 = deferred_check.point[0..l].to_vec();
    let a_2 = deferred_check.point[l..l + max_num_vars].to_vec();

    let (zero_bit, one_bit): (Vec<P::ScalarField>, Vec<P::ScalarField>) = cfg_iter!(a_1)
        .zip(cfg_iter!(t))
        .map(|(a, t)| ((P::ScalarField::one() - *a - *t + (*a * *t)), (*a * *t)))
        .unzip();

    // We calculate the coefficients of the batched polynomials used to construct g_prime.
    let coeffs = cfg_into_iter!((0..num_polys))
        .map(|i| {
            cfg_into_iter!((0..l))
                .rev()
                .map(|j| {
                    if (i >> j) & 1 == 1 {
                        one_bit[j]
                    } else {
                        zero_bit[j]
                    }
                })
                .product::<P::ScalarField>()
        })
        .collect::<Vec<P::ScalarField>>();

    // We calculate s, the claimed result of the Sumcheck.
    let mut s = zero;
    for (i, value) in values.iter().take(num_polys).enumerate() {
        let mut eq_t_i = one;
        for (k, t) in t.iter().enumerate() {
            let field_bit = P::ScalarField::from(((i >> k) & 1) as u8);

            eq_t_i *= field_bit * t + (one - field_bit) * (one - t);
        }

        s += eq_t_i * value;
    }

    // The claimed Sumcheck evaluation must match the actual evaluation.
    if proof.eval != s {
        return Err(PolynomialError::ParameterError(
            "Sumcheck failed".to_string(),
        ));
    }

    let eq_tilde_eval =
        eq_tilde_poly
            .evaluate(&deferred_check.point)
            .ok_or(PolynomialError::ParameterError(
                "Couldn't evaluate eq_tilde at (a_1, a_2)".to_string(),
            ))?;

    let g_tilde_eval = deferred_check.eval
        * eq_tilde_eval
            .inverse()
            .ok_or(PolynomialError::ParameterError(
                "Couldn't invert eq_tilde_eval".to_string(),
            ))?;

    Ok((g_tilde_eval, coeffs, a_2))
}

/// Used to generate the challenges and scalars used during the
/// MLEProof accumulation process.
type BatchChallengesOpening<F> = (
    SumCheckProof<F, PolyOracle<F>>,
    F,
    Arc<DenseMultilinearExtension<F>>,
    AccumulationChallengesAndScalars<F>,
);

pub fn challenges_and_scalars_from_mv_batch_open<P>(
    polys: &[Arc<DenseMultilinearExtension<P::ScalarField>>],
    points: &[Vec<P::ScalarField>],
    transcript: &mut RescueTranscript<P::BaseField>,
) -> Result<BatchChallengesOpening<P::ScalarField>, PolynomialError>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    if polys.len() != points.len() {
        return Err(PolynomialError::ParameterError(format!(
            "Length of polys vector {} and points vector {} do not match",
            polys.len(),
            points.len()
        )));
    }

    for (poly, point) in polys.iter().zip(points.iter()) {
        if poly.num_vars() != point.len() || poly.num_vars() != polys[0].num_vars() {
            return Err(PolynomialError::ParameterError(format!(
                "Number of variables {} of poly does not match length of point vector {}, or does not match the number of variables of the first poly: {}",
                poly.num_vars(),
                point.len(),
                polys[0].num_vars()
            )));
        }
    }

    let num_vars = polys[0].num_vars();
    let num_polys = polys.len();

    let l = num_polys.next_power_of_two().ilog2() as usize;

    let t = transcript
        .squeeze_scalar_challenges::<P>(b"t", l)
        .map_err(|_| {
            PolynomialError::ParameterError("Couldn't squeeze a challenge scalar".to_string())
        })?;

    // We will repeatedly need the zero and one of the scalar field.
    let zero = P::ScalarField::zero();
    let one = P::ScalarField::one();

    // The coefficients used to construct the mle g_tilde
    let mut g_tilde_coeff = vec![zero; 1 << (l + num_vars)];

    // Since all later polys are zero, we need only have our sum run up to num_poly.
    let eq_t_vec = build_eq_x_r_vec(&t);

    let difference = (1 << l) - num_polys;
    let evals_slice = vec![P::ScalarField::zero(); 1 << num_vars];
    let zeroes_poly = Arc::new(DenseMultilinearExtension::from_evaluations_slice(
        num_vars,
        &evals_slice,
    ));
    let polys_2 = [polys, vec![zeroes_poly; difference].as_slice()].concat();
    cfg_iter_mut!(g_tilde_coeff)
        .enumerate()
        .for_each(|(i, coeff)| *coeff = eq_t_vec[i % (1 << l)] * polys_2[i % (1 << l)][i >> l]);

    // We calculate the evaluations for the mle eq_tilde
    let mut eq_tilde_coeff: Vec<P::ScalarField> = vec![zero; 1 << (l + num_vars)];

    let points_2 = [points, vec![vec![zero; num_vars]; difference].as_slice()].concat();
    let eq_points_vec = cfg_iter!(points_2)
        .map(|point| build_eq_x_r_vec(point))
        .collect::<Vec<Vec<P::ScalarField>>>();
    cfg_iter_mut!(eq_tilde_coeff)
        .enumerate()
        .for_each(|(i, coeff)| {
            *coeff = eq_points_vec[i % (1 << l)][i >> l];
        });

    let mles = [
        Arc::new(
            DenseMultilinearExtension::<P::ScalarField>::from_evaluations_vec(
                l + num_vars,
                g_tilde_coeff,
            ),
        ),
        Arc::new(
            DenseMultilinearExtension::<P::ScalarField>::from_evaluations_vec(
                l + num_vars,
                eq_tilde_coeff,
            ),
        ),
    ]
    .to_vec();

    // Copy of g_tilde needed later to construct g_prime after ownership is lost.
    let g_tilde_poly = mles[0].clone();

    let virtual_polynomial =
        VirtualPolynomial::<P::ScalarField>::new(2, l + num_vars, mles, vec![(one, vec![0, 1])]);

    let proof = VPSumCheck::<P>::prove(&virtual_polynomial, transcript)
        .map_err(|_| PolynomialError::ParameterError("Couldn't prove Sumcheck".to_string()))?;

    // The final challenge vector from the Sumcheck.
    let a_1 = proof.point[0..l].to_vec();
    let a_2 = proof.point[l..l + num_vars].to_vec();

    let (zero_bit, one_bit): (Vec<P::ScalarField>, Vec<P::ScalarField>) = cfg_iter!(a_1)
        .zip(cfg_iter!(t))
        .map(|(a, t)| ((P::ScalarField::one() - *a - *t + (*a * *t)), (*a * *t)))
        .unzip();

    // We calculate the coefficients of the batched polynomials used to construct g_prime.
    let coeffs = cfg_into_iter!((0..num_polys))
        .map(|i| {
            cfg_into_iter!((0..l))
                .rev()
                .map(|j| {
                    if (i >> j) & 1 == 1 {
                        one_bit[j]
                    } else {
                        zero_bit[j]
                    }
                })
                .product::<P::ScalarField>()
        })
        .collect::<Vec<P::ScalarField>>();

    // The polynomial g_prime polynomial in the batching protocol
    let g_prime_poly = g_tilde_poly.fix_variables(&a_1);
    let eval = g_prime_poly
        .evaluate(&a_2)
        .ok_or(PolynomialError::ParameterError(
            "Couldn't evaluate g_prime at a_2".to_string(),
        ))?;

    Ok((
        proof,
        eval,
        Arc::new(g_prime_poly),
        AccumulationChallengesAndScalars {
            t,
            a_1,
            a_2,
            coeffs,
        },
    ))
}

pub fn add_extra_variables<F>(
    mle: &Arc<DenseMultilinearExtension<F>>,
    num_vars: &usize,
) -> Result<Arc<DenseMultilinearExtension<F>>, PolynomialError>
where
    F: Field,
{
    if mle.num_vars > *num_vars {
        Err(PolynomialError::ParameterError(
            "polynomial has too many variables".to_string(),
        ))
    } else {
        let evals = mle.evaluations.as_slice();
        let eval_vec = vec![evals; 1 << (*num_vars - mle.num_vars)].concat();
        // for _ in mle.num_vars..*num_vars {
        //     let mut temp_eval_vec = eval_vec.clone();
        //     eval_vec.append(&mut temp_eval_vec);
        // }
        Ok(Arc::new(
            DenseMultilinearExtension::<F>::from_evaluations_vec(*num_vars, eval_vec),
        ))
    }
}

/// Evaluates the gate equation at a given point. Used by the verifier.
pub(crate) fn eval_gate_equation<F: PrimeField>(
    gate_info: &GateInfo<F>,
    selector_evals: &[F],
    wire_evals: &[F],
    pub_input_poly_eval: F,
) -> F {
    let evals = [wire_evals, selector_evals, &[pub_input_poly_eval]].concat();
    gate_info
        .products
        .iter()
        .fold(F::zero(), |acc, (coeff, prod)| {
            prod.iter().map(|i| evals[*i]).product::<F>() * coeff + acc
        })
}

/// Evaluate the permutation equation at a given point. Used by the verifier.
pub(crate) fn eval_permutation_equation<F: PrimeField>(
    perm_evals: &[F],
    wire_evals: &[F],
    challenges: &MLEChallenges<F>,
) -> F {
    let MLEChallenges { gamma, epsilon, .. } = *challenges;

    let pairs = perm_evals
        .chunks(3)
        .zip(wire_evals.chunks(3))
        .map(|(perm_chunk, wire_chunk)| {
            let denominator = wire_chunk.iter().map(|&eval| gamma - eval).product::<F>();
            let numerator = perm_chunk
                .iter()
                .zip(wire_chunk.iter())
                .fold(F::zero(), |acc, (perm, wire)| {
                    acc + (*perm * denominator / (gamma - wire))
                });
            [numerator, denominator]
        })
        .collect::<Vec<[F; 2]>>();

    epsilon * pairs[0][0]
        + epsilon * epsilon * pairs[1][0]
        + epsilon * epsilon * epsilon * pairs[0][1]
        + epsilon * epsilon * epsilon * epsilon * pairs[1][1]
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_lookup_equation<F: PrimeField>(
    wire_evals: &[F],
    range_table_eval: F,
    key_table_eval: F,
    table_dom_sep_eval: F,
    q_dom_sep_eval: F,
    q_lookup_eval: F,
    challenges: &MLEChallenges<F>,
    m_poly_eval: F,
) -> F {
    let MLEChallenges { tau, epsilon, .. } = *challenges;

    let epsilon_start = epsilon.pow([5u64]);
    let lookup_wire_eval = wire_evals[5]
        + tau
            * q_lookup_eval
            * (q_dom_sep_eval
                + tau * (wire_evals[0] + tau * (wire_evals[1] + tau * wire_evals[2])));

    let table_eval = range_table_eval
        + tau
            * q_lookup_eval
            * (table_dom_sep_eval
                + tau * (key_table_eval + tau * (wire_evals[3] + tau * wire_evals[4])));

    epsilon_start * (lookup_wire_eval + epsilon * table_eval + epsilon * epsilon * m_poly_eval)
}

pub(crate) fn build_zerocheck_eval<PCS>(
    evals: &MLEProofEvals<PCS>,
    lookup_evals: Option<&MLELookupEvals<PCS>>,
    gate_info: &GateInfo<PCS::Evaluation>,
    challenges: &MLEChallenges<PCS::Evaluation>,
    pi_eval: PCS::Evaluation,
    eq_eval: PCS::Evaluation,
) -> PCS::Evaluation
where
    PCS: PolynomialCommitmentScheme,
    PCS::Evaluation: PrimeField,
{
    let mut sc_eval = eval_gate_equation(
        gate_info,
        &evals.selector_evals[..17],
        &evals.wire_evals[..5],
        pi_eval,
    );

    sc_eval += eval_permutation_equation(&evals.permutation_evals, &evals.wire_evals, challenges);

    if let Some(lookup_evals) = lookup_evals {
        sc_eval += eval_lookup_equation::<PCS::Evaluation>(
            &evals.wire_evals,
            lookup_evals.range_table_eval,
            lookup_evals.key_table_eval,
            lookup_evals.table_dom_sep_eval,
            lookup_evals.q_dom_sep_eval,
            lookup_evals.q_lookup_eval,
            challenges,
            lookup_evals.m_poly_eval,
        );
    }
    sc_eval * eq_eval
}

pub(crate) fn scale_mle<F: PrimeField>(
    mle: &Arc<DenseMultilinearExtension<F>>,
    scalar: F,
) -> Vec<F> {
    mle.evaluations.iter().map(|eval| *eval * scalar).collect()
}

pub(crate) fn add_vecs<F: PrimeField>(a: &[F], b: &[F]) -> Result<Vec<F>, PlonkError> {
    if a.len() != b.len() {
        return Err(PlonkError::InvalidParameters(
            "Vectors have different lengths".to_string(),
        ));
    }
    Ok(a.iter().zip(b.iter()).map(|(a, b)| *a + *b).collect())
}

/// This function is used to build the [`VirtualPolynomial`] that we run a SumCheck on in our SNARK proofs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_sumcheck_poly<F, PCS>(
    wire_polys: &[Arc<DenseMultilinearExtension<F>>],
    selector_polys: &[Arc<DenseMultilinearExtension<F>>],
    permutation_polys: &[Arc<DenseMultilinearExtension<F>>],
    pub_input_poly: &Arc<DenseMultilinearExtension<F>>,
    eq_poly: Arc<DenseMultilinearExtension<F>>,
    gate_info: &GateInfo<F>,
    challenges: &MLEChallenges<F>,
    lookup_pk: Option<&MLELookupProvingKey<PCS>>,
    m_poly: Option<Arc<DenseMultilinearExtension<F>>>,
) -> Result<VirtualPolynomial<F>, PolynomialError>
where
    F: PrimeField,
    PCS: PolynomialCommitmentScheme<Evaluation = F>,
{
    let MLEChallenges { gamma, epsilon, .. } = *challenges;

    // First build the gate equation part of the VirtualPolynomial
    let mut vp = build_gate_equation(wire_polys, selector_polys, pub_input_poly, gate_info)?;

    // Add on the permutation polynomial
    let perm_vp = build_permutation_equation(wire_polys, permutation_polys, challenges)?;
    vp = &vp + &perm_vp;

    // If we have some lookup proving key then we add on the lookup part as well.
    if let (Some(lookup_pk), Some(m_poly)) = (lookup_pk, m_poly) {
        // The unwrap is safe because we have checked that we are using a lookup argument
        let lookup_vp = build_lookup_equation(wire_polys, lookup_pk, m_poly, challenges)?;
        vp = &vp + &lookup_vp;

        let first_perm_coeff = epsilon * epsilon * epsilon * gamma * gamma * gamma;
        let second_perm_coeff = first_perm_coeff * epsilon;

        let polys = vec![eq_poly.clone()];
        let products = vec![(first_perm_coeff + second_perm_coeff, vec![0])];

        let extra_term = VirtualPolynomial::new(1, eq_poly.num_vars, polys, products);
        vp.mul_by_mle(eq_poly)?;
        vp = &vp + &extra_term;
    } else {
        let first_perm_coeff = epsilon * epsilon * epsilon * gamma * gamma * gamma;
        let second_perm_coeff = epsilon * epsilon * epsilon * epsilon * gamma * gamma;

        let polys = vec![eq_poly.clone()];
        let products = vec![(first_perm_coeff + second_perm_coeff, vec![0])];

        let extra_term = VirtualPolynomial::new(1, eq_poly.num_vars, polys, products);
        vp.mul_by_mle(eq_poly)?;
        vp = &vp + &extra_term;
    }

    Ok(vp)
}

pub(crate) fn build_gate_equation<F: PrimeField>(
    wire_polys: &[Arc<DenseMultilinearExtension<F>>],
    selector_polys: &[Arc<DenseMultilinearExtension<F>>],
    pub_input_poly: &Arc<DenseMultilinearExtension<F>>,
    gate_info: &GateInfo<F>,
) -> Result<VirtualPolynomial<F>, PolynomialError> {
    // We take the first 5 wire polys, the first 17 selector polys, and the public input poly currently
    // in the future we will make this more customisable.
    let vp_mles = wire_polys
        .iter()
        .take(5)
        .chain(selector_polys.iter().take(17))
        .chain([pub_input_poly])
        .cloned()
        .collect::<Vec<_>>();

    Ok(VirtualPolynomial::from_gate_info_and_mles(
        gate_info, &vp_mles,
    ))
}

pub(crate) fn build_permutation_equation<F: PrimeField>(
    wire_polys: &[Arc<DenseMultilinearExtension<F>>],
    permutation_polys: &[Arc<DenseMultilinearExtension<F>>],
    challenges: &MLEChallenges<F>,
) -> Result<VirtualPolynomial<F>, PolynomialError> {
    // First we find the number of variables for the VirtualPolynomial
    let num_vars = wire_polys[0].num_vars;

    let MLEChallenges { gamma, epsilon, .. } = *challenges;
    let epsilon_sq = epsilon * epsilon;
    let epsilon_cube = epsilon_sq * epsilon;
    let epsilon_sqsq = epsilon_sq * epsilon_sq;
    let gamma_sq = gamma * gamma;

    // We need to check the evaluations over the boolean hyper cube of the product of the numerator polys and the product of the denominator polys.
    let polys = [permutation_polys, wire_polys].concat();

    // For now we just check if we have five or six wires to tell if we are using UltraPlonk or not.
    let products = match wire_polys.len() {
        5 => {
            vec![
                (epsilon * gamma_sq, vec![0]),
                (-epsilon * gamma, vec![0, 6]),
                (-epsilon * gamma, vec![0, 7]),
                (epsilon, vec![0, 6, 7]),
                (epsilon * gamma_sq, vec![1]),
                (-epsilon * gamma, vec![1, 5]),
                (-epsilon * gamma, vec![1, 7]),
                (epsilon, vec![1, 5, 7]),
                (epsilon * gamma_sq, vec![2]),
                (-epsilon * gamma, vec![2, 5]),
                (-epsilon * gamma, vec![2, 6]),
                (epsilon, vec![2, 5, 6]),
                (epsilon_sq * gamma, vec![3]),
                (-epsilon_sq, vec![3, 9]),
                (epsilon_sq * gamma, vec![4]),
                (-epsilon_sq, vec![4, 8]),
                (-epsilon_cube, vec![5, 6, 7]),
                (epsilon_cube * gamma, vec![5, 6]),
                (epsilon_cube * gamma, vec![5, 7]),
                (epsilon_cube * gamma, vec![6, 7]),
                (-epsilon_cube * gamma_sq, vec![5]),
                (-epsilon_cube * gamma_sq, vec![6]),
                (-epsilon_cube * gamma_sq, vec![7]),
                (epsilon_sqsq, vec![8, 9]),
                (-epsilon_sqsq * gamma, vec![8]),
                (-epsilon_sqsq * gamma, vec![9]),
            ]
        },
        6 => {
            vec![
                (epsilon * gamma_sq, vec![0]),
                (-epsilon * gamma, vec![0, 7]),
                (-epsilon * gamma, vec![0, 8]),
                (epsilon, vec![0, 7, 8]),
                (epsilon * gamma_sq, vec![1]),
                (-epsilon * gamma, vec![1, 6]),
                (-epsilon * gamma, vec![1, 8]),
                (epsilon, vec![1, 6, 8]),
                (epsilon * gamma_sq, vec![2]),
                (-epsilon * gamma, vec![2, 6]),
                (-epsilon * gamma, vec![2, 7]),
                (epsilon, vec![2, 6, 7]),
                (epsilon_sq * gamma_sq, vec![3]),
                (-epsilon_sq * gamma, vec![3, 10]),
                (-epsilon_sq * gamma, vec![3, 11]),
                (epsilon_sq, vec![3, 10, 11]),
                (epsilon_sq * gamma_sq, vec![4]),
                (-epsilon_sq * gamma, vec![4, 9]),
                (-epsilon_sq * gamma, vec![4, 11]),
                (epsilon_sq, vec![4, 9, 11]),
                (epsilon_sq * gamma_sq, vec![5]),
                (-epsilon_sq * gamma, vec![5, 9]),
                (-epsilon_sq * gamma, vec![5, 10]),
                (epsilon_sq, vec![5, 9, 10]),
                (-epsilon_cube, vec![6, 7, 8]),
                (epsilon_cube * gamma, vec![6, 7]),
                (epsilon_cube * gamma, vec![6, 8]),
                (epsilon_cube * gamma, vec![7, 8]),
                (-epsilon_cube * gamma_sq, vec![6]),
                (-epsilon_cube * gamma_sq, vec![7]),
                (-epsilon_cube * gamma_sq, vec![8]),
                (-epsilon_sqsq, vec![9, 10, 11]),
                (epsilon_sqsq * gamma, vec![9, 10]),
                (epsilon_sqsq * gamma, vec![9, 11]),
                (epsilon_sqsq * gamma, vec![10, 11]),
                (-epsilon_sqsq * gamma_sq, vec![9]),
                (-epsilon_sqsq * gamma_sq, vec![10]),
                (-epsilon_sqsq * gamma_sq, vec![11]),
            ]
        },
        _ => {
            return Err(PolynomialError::ParameterError(
                "Invalid number of wires".to_string(),
            ))
        },
    };

    Ok(VirtualPolynomial::new(3, num_vars, polys, products))
}

pub(crate) fn build_lookup_equation<F, PCS>(
    wire_polys: &[Arc<DenseMultilinearExtension<F>>],
    lookup_pk: &MLELookupProvingKey<PCS>,
    m_poly: Arc<DenseMultilinearExtension<F>>,
    challenges: &MLEChallenges<F>,
) -> Result<VirtualPolynomial<F>, PolynomialError>
where
    F: PrimeField,
    PCS: PolynomialCommitmentScheme<Evaluation = F>,
{
    // First we find the number of variables for the VirtualPolynomial
    let num_vars = wire_polys[0].num_vars;

    let MLEChallenges { tau, epsilon, .. } = *challenges;
    let tau_sq = tau * tau;
    let tau_cube = tau_sq * tau;
    let tau_sqsq = tau_sq * tau_sq;
    // There are four terms in the permutation argument so we start with epsilon^5.
    let epsilon_start = epsilon.pow([5u64]);

    // We need only check the lookup wire and lookup table.

    let polys = vec![
        wire_polys[5].clone(),
        lookup_pk.q_lookup_mle.clone(),
        lookup_pk.q_dom_sep_mle.clone(),
        wire_polys[0].clone(),
        wire_polys[1].clone(),
        wire_polys[2].clone(),
        lookup_pk.range_table_mle.clone(),
        lookup_pk.table_dom_sep_mle.clone(),
        lookup_pk.key_table_mle.clone(),
        wire_polys[3].clone(),
        wire_polys[4].clone(),
        m_poly,
    ];

    let products = vec![
        (epsilon_start, vec![0]),
        (epsilon_start * tau, vec![1, 2]),
        (epsilon_start * tau_sq, vec![1, 3]),
        (epsilon_start * tau_cube, vec![1, 4]),
        (epsilon_start * tau_sqsq, vec![1, 5]),
        (epsilon_start * epsilon, vec![6]),
        (epsilon_start * epsilon * tau, vec![1, 7]),
        (epsilon_start * epsilon * tau_sq, vec![1, 8]),
        (epsilon_start * epsilon * tau_cube, vec![1, 9]),
        (epsilon_start * epsilon * tau_sqsq, vec![1, 10]),
        (epsilon_start * epsilon * epsilon, vec![11]),
    ];

    Ok(VirtualPolynomial::new(2, num_vars, polys, products))
}
