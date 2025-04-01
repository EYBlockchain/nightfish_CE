//! In this module we define a `VirtualPolynomial<F>` struct that can be used to represent a
//! multivariate polynomial of higher degree in the Sumcheck protocol.

use super::mle_structs::PolynomialError;

use ark_ff::{Field, PrimeField};
use ark_poly::{
    evaluations::multivariate::{DenseMultilinearExtension, MultilinearExtension},
    univariate::DensePolynomial,
    Polynomial,
};
use ark_std::{
    borrow::Borrow,
    cfg_iter,
    cmp::max,
    ops::{Add, Mul},
    string::ToString,
    sync::Arc,
    vec::Vec,
};

use hashbrown::HashMap;
use rayon::prelude::*;

/// Used to extract information about a polynomial necessary for Sumcheck.
pub trait PolynomialInfo<F: Field> {
    /// The total degree of the polynomial, for a univariate this is the degree,
    /// for multilinear this is the number of variables and
    /// for a virtual polynomial this is the maximum degree of any monomial.
    fn max_degree(&self) -> usize;
    /// The number of variables the polynomial uses.
    fn num_vars(&self) -> usize;
}

impl<F: Field> PolynomialInfo<F> for DenseMultilinearExtension<F> {
    fn max_degree(&self) -> usize {
        self.num_vars
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field> PolynomialInfo<F> for DensePolynomial<F> {
    fn max_degree(&self) -> usize {
        self.degree()
    }

    fn num_vars(&self) -> usize {
        1
    }
}

/// A virtual polynomial is a multivariate polynomial f(x)=h(g1(x),...,gn(x))
/// where g1(x),...,gn(x) are multilinear polynomials.
/// The virtual polynomial is represented by a vector of multilinear polynomials, each `Vec<usize>` in
/// `self.products` tells us which polynomials to multiply for that term of the virtual polynomial.
/// The entire polynomial is the sum of all the terms i.e.
/// `f(x) = sum_{i=1}^{self.products.len()}prod_{j=0}^{self.products[i].len()}self.polys[self.products[i][j]].evaluate(x)`.
#[derive(Clone, Debug)]
pub struct VirtualPolynomial<F: PrimeField> {
    /// the maximum degree of any monomial in the polynomial
    pub max_degree: usize,
    /// the number of variables in the polynomial
    pub num_vars: usize,
    /// this vec stores the multilinear polynomials g1(x),...,gn(x)
    pub polys: Vec<Arc<DenseMultilinearExtension<F>>>,
    /// this vec stores the indices of the multilinear polynomials found in `self.polys`
    pub products: Vec<(F, Vec<usize>)>,
    /// Hashmap to store the index of the polynomial in `self.polys` given a pointer to it
    pub poly_mapping: HashMap<*const DenseMultilinearExtension<F>, usize>,
}

impl<F: PrimeField> VirtualPolynomial<F> {
    /// Create a new virtual polynomial
    pub fn new(
        max_degree: usize,
        num_vars: usize,
        polys: Vec<Arc<DenseMultilinearExtension<F>>>,
        products: Vec<(F, Vec<usize>)>,
    ) -> Self {
        let mut poly_mapping = HashMap::new();
        for (i, poly) in polys.iter().enumerate() {
            poly_mapping.insert(Arc::as_ptr(poly), i);
        }
        Self {
            max_degree,
            num_vars,
            polys,
            products,
            poly_mapping,
        }
    }

    /// Evaluate the virtual polynomial at the point `point`
    pub fn evaluate(&self, point: &[F]) -> Result<F, PolynomialError> {
        if point.len() != self.num_vars {
            return Err(PolynomialError::ParameterError(
                "point.len() != self.num_vars".to_string(),
            ));
        }

        let poly_evals = cfg_iter!(&self.polys)
            .map(|poly| {
                poly.evaluate(point).ok_or(PolynomialError::ParameterError(
                    "Point had wrong length".to_string(),
                ))
            })
            .collect::<Result<Vec<F>, PolynomialError>>()?;

        let result = cfg_iter!(self.products)
            .map(|(constant, product)| {
                cfg_iter!(product)
                    .map(|poly_index| poly_evals[*poly_index])
                    .product::<F>()
                    * constant
            })
            .sum::<F>();

        Ok(result)
    }

    /// Multiple the current VirtualPolynomial by an MLE:
    pub fn mul_by_mle(
        &mut self,
        mle: Arc<DenseMultilinearExtension<F>>,
    ) -> Result<(), PolynomialError> {
        if mle.num_vars != self.num_vars {
            return Err(PolynomialError::ParameterError(
                "product has a multiplicand with wrong number of variables".to_string(),
            ));
        }

        // Add the new MLE to the list of polynomials and get its index
        let mle_ptr = Arc::as_ptr(&mle);
        if let Some(index) = self.poly_mapping.get(&mle_ptr) {
            // If the MLE is already in the list of polynomials, just add it to the products
            for (_, product) in self.products.iter_mut() {
                product.push(*index);
            }
            return Ok(());
        } else {
            // If the MLE is not in the list of polynomials, add it
            let new_index = self.polys.len();
            self.polys.push(mle);
            self.poly_mapping.insert(mle_ptr, new_index);
            // Iterate through each product and add the new index
            for (_, product) in self.products.iter_mut() {
                product.push(new_index);
            }
        }

        // Update the maximum degree of the polynomial
        // Assuming that each MLE has a degree of 1
        self.max_degree += 1;

        Ok(())
    }

    /// add_mle_list
    pub fn add_mle_list(
        &mut self,
        mle_list: impl IntoIterator<Item = Arc<DenseMultilinearExtension<F>>>,
        constant: F,
    ) -> Result<(), PolynomialError> {
        let mle_list: Vec<Arc<DenseMultilinearExtension<F>>> = mle_list.into_iter().collect();

        let mut indexed_product: Vec<usize> = Vec::with_capacity(mle_list.len());

        if mle_list.is_empty() {
            return Err(PolynomialError::ParameterError(
                "input mle_list is empty".to_string(),
            ));
        }

        for mle in mle_list.iter() {
            if mle.num_vars != self.num_vars {
                return Err(PolynomialError::ParameterError(
                    "MLE has a different number of variables".to_string(),
                ));
            }
            let mle_ptr = Arc::as_ptr(mle);
            if let Some(index) = self.poly_mapping.get(&mle_ptr) {
                indexed_product.push(*index)
            } else {
                let curr_index = self.polys.len();
                self.polys.push(mle.clone());
                self.poly_mapping.insert(mle_ptr, curr_index);
                indexed_product.push(curr_index);
            }
        }

        self.products.push((constant, indexed_product));

        self.max_degree = max(self.max_degree, mle_list.len());

        Ok(())
    }
    /// Function to return a virtual polynomial describing the gate equation of a circuit.
    pub fn from_gate_info_and_mles(
        gate_info: &super::mle_structs::GateInfo<F>,
        mles: &[Arc<DenseMultilinearExtension<F>>],
    ) -> Self {
        let max_degree = gate_info.max_degree;
        let num_vars = mles[0].num_vars;

        Self::new(
            max_degree,
            num_vars,
            mles.to_vec(),
            gate_info.products.clone(),
        )
    }
}

impl<F: PrimeField> PolynomialInfo<F> for VirtualPolynomial<F> {
    fn max_degree(&self) -> usize {
        self.max_degree
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: PrimeField, T: Borrow<F>> Mul<T> for VirtualPolynomial<F> {
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self::Output {
        let rhs = rhs.borrow();
        for (constant, _) in self.products.iter_mut() {
            *constant *= rhs;
        }
        self
    }
}

impl<F: PrimeField> Add for &VirtualPolynomial<F> {
    type Output = VirtualPolynomial<F>;
    fn add(self, other: &VirtualPolynomial<F>) -> Self::Output {
        let mut res = self.clone();
        for products in other.products.iter() {
            let cur: Vec<Arc<DenseMultilinearExtension<F>>> =
                products.1.iter().map(|&x| other.polys[x].clone()).collect();

            res.add_mle_list(cur, products.0)
                .expect("add product failed");
        }

        res
    }
}
#[cfg(test)]
mod test {
    use super::*;
    use ark_bn254::Fq;
    use ark_ff::One;
    use ark_std::vec;

    #[test]
    fn test_virtual_polynomial_mul_by_mle() -> Result<(), PolynomialError> {
        let mle_1 = Arc::new(DenseMultilinearExtension::<Fq>::from_evaluations_vec(
            2,
            vec![Fq::from(1), Fq::from(2), Fq::from(3), Fq::from(4)],
        ));

        let mle_2 = Arc::new(DenseMultilinearExtension::<Fq>::from_evaluations_vec(
            2,
            vec![Fq::from(3), Fq::from(2), Fq::from(4), Fq::from(3)],
        ));
        let poly =
            VirtualPolynomial::<Fq>::new(2, 2, vec![mle_1, mle_2], vec![(Fq::one(), vec![0, 1])]);

        let mle_3 = Arc::new(DenseMultilinearExtension::<Fq>::from_evaluations_vec(
            2,
            vec![Fq::from(4), Fq::from(4), Fq::from(4), Fq::from(4)],
        ));
        let mut poly2 = poly.clone();
        poly2.mul_by_mle(mle_3.clone())?;
        let poly4 = VirtualPolynomial::<Fq>::new(2, 2, vec![mle_3], vec![(Fq::one(), vec![0])]);

        let a = poly.evaluate(&[Fq::from(1), Fq::from(2)])?;
        let b = poly4.evaluate(&[Fq::from(1), Fq::from(2)])?;
        let c = poly2.evaluate(&[Fq::from(1), Fq::from(2)])?;
        assert_eq!(a * b, c);
        assert_eq!(poly.max_degree + 1, poly2.max_degree);
        Ok(())
    }

    #[test]
    fn test_mul_scalar() {
        let virtual_poly = VirtualPolynomial::<Fq>::new(
            2,
            2,
            vec![
                Arc::new(DenseMultilinearExtension::<Fq>::from_evaluations_vec(
                    2,
                    vec![Fq::from(1), Fq::from(2), Fq::from(3), Fq::from(4)],
                )),
                Arc::new(DenseMultilinearExtension::<Fq>::from_evaluations_vec(
                    2,
                    vec![Fq::from(1), Fq::from(2), Fq::from(3), Fq::from(4)],
                )),
            ],
            vec![(Fq::one(), vec![0, 1])],
        );

        let value = virtual_poly.evaluate(&[Fq::from(1), Fq::from(2)]).unwrap();
        let alpha = Fq::from(2);
        let virtual_poly_2 = virtual_poly * alpha;
        let value_2 = virtual_poly_2
            .evaluate(&[Fq::from(1), Fq::from(2)])
            .unwrap();
        assert_eq!(value * alpha, value_2);
    }
}
