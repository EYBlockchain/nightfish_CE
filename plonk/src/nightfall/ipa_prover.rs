// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

use crate::{
    constants::domain_size_ratio,
    errors::{PlonkError, SnarkError::*},
    nightfall::ipa_structs::{
        eval_merged_lookup_witness, eval_merged_table, Challenges, MapKey, Oracles, PlookupOracles,
        ProvingKey,
    },
    proof_system::structs::{PlookupEvaluations, ProofEvaluations},
};

use ark_ec::{short_weierstrass::Affine, AffineRepr, CurveConfig};
use ark_ff::{batch_inversion, FftField, Field, One, PrimeField, UniformRand, Zero};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain,
    Polynomial, Radix2EvaluationDomain,
};
use ark_std::{
    cfg_iter,
    collections::BTreeMap,
    ops::Neg,
    rand::{CryptoRng, RngCore},
    string::ToString,
    vec,
    vec::Vec,
};

use core::ops::Mul;

use jf_primitives::{
    pcs::{PolynomialCommitmentScheme, StructuredReferenceString},
    rescue::RescueParameter,
};
use jf_relation::{constants::GATE_WIDTH, gadgets::ecc::HasTEForm, Arithmetization};
use jf_utils::par_utils::{parallelizable_btree_map, parallelizable_slice_iter};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

type CommitmentsAndPolys<PCS> = (
    Vec<<PCS as PolynomialCommitmentScheme>::Commitment>,
    Vec<DensePolynomial<<PCS as PolynomialCommitmentScheme>::Evaluation>>,
);

type MapAndOptionalPoly<PCS> = (
    BTreeMap<MapKey<PCS>, Vec<<PCS as PolynomialCommitmentScheme>::Polynomial>>,
    Option<Vec<<PCS as PolynomialCommitmentScheme>::Polynomial>>,
);

/// A Plonk IOP prover.
pub(crate) struct FFTProver<PCS: PolynomialCommitmentScheme> {
    pub(crate) domain: Radix2EvaluationDomain<PCS::Evaluation>,
    quot_domain: GeneralEvaluationDomain<PCS::Evaluation>,
}

impl<PCS, F, P> FFTProver<PCS>
where
    PCS: PolynomialCommitmentScheme<
        Evaluation = P::ScalarField,
        Polynomial = DensePolynomial<P::ScalarField>,
        Commitment = Affine<P>,
    >,
    PCS::SRS: StructuredReferenceString<Item = PCS::Commitment>,
    F: PrimeField + RescueParameter,
    P: HasTEForm<BaseField = F>,
{
    /// Construct a Plonk prover that uses a domain with size `domain_size` and
    /// quotient polynomial domain with a size that is larger than the degree of
    /// the quotient polynomial.
    /// * `num_wire_types` - number of wire types in the corresponding
    ///   constraint system.
    pub(crate) fn new(domain_size: usize, num_wire_types: usize) -> Result<Self, PlonkError> {
        let domain = Radix2EvaluationDomain::<P::ScalarField>::new(domain_size)
            .ok_or(PlonkError::DomainCreationError)?;
        let quot_domain = GeneralEvaluationDomain::<P::ScalarField>::new(
            domain_size * domain_size_ratio(domain_size, num_wire_types),
        )
        .ok_or(PlonkError::DomainCreationError)?;
        Ok(Self {
            domain,
            quot_domain,
        })
    }

    /// Round 1:
    /// 1. Compute and commit wire witness polynomials.
    /// 2. Compute public input polynomial.
    ///
    /// Return the wire witness polynomials and their commitments, also return the public input polynomial.
    pub(crate) fn run_1st_round<C: Arithmetization<PCS::Evaluation>, R: CryptoRng + RngCore>(
        &self,
        prng: &mut R,
        ck: &<PCS::SRS as StructuredReferenceString>::ProverParam,
        cs: &C,
    ) -> Result<(CommitmentsAndPolys<PCS>, DensePolynomial<P::ScalarField>), PlonkError> {
        let wire_polys: Vec<DensePolynomial<P::ScalarField>> = cs
            .compute_wire_polynomials()?
            .into_iter()
            .map(|poly| self.mask_polynomial(prng, poly, 1))
            .collect();
        let wires_poly_comms = cfg_iter!(wire_polys)
            .map(|wire_poly| PCS::commit(ck, wire_poly))
            .collect::<Result<Vec<PCS::Commitment>, _>>()?;
        let pub_input_poly = cs.compute_pub_input_polynomial()?;
        Ok(((wires_poly_comms, wire_polys), pub_input_poly))
    }

    /// Round 1.5 (Plookup): Compute and commit the polynomials that interpolate
    /// the sorted concatenation of the (merged) lookup table and the
    /// (merged) witnesses in lookup gates. Return the sorted vector, the
    /// polynomials and their commitments, as well as the merged lookup table.
    /// `cs` is guaranteed to support lookup.
    #[allow(clippy::type_complexity)]
    pub(crate) fn run_plookup_1st_round<
        C: Arithmetization<P::ScalarField>,
        R: CryptoRng + RngCore,
    >(
        &self,
        prng: &mut R,
        ck: &<PCS::SRS as StructuredReferenceString>::ProverParam,
        cs: &C,
        tau: P::ScalarField,
    ) -> Result<
        (
            CommitmentsAndPolys<PCS>,
            Vec<P::ScalarField>,
            Vec<P::ScalarField>,
        ),
        PlonkError,
    > {
        let merged_lookup_table = cs.compute_merged_lookup_table(tau)?;
        let (sorted_vec, h_1_poly, h_2_poly) =
            cs.compute_lookup_sorted_vec_polynomials(tau, &merged_lookup_table)?;
        let h_1_poly = self.mask_polynomial(prng, h_1_poly, 2);
        let h_2_poly = self.mask_polynomial(prng, h_2_poly, 2);
        let h_polys = vec![h_1_poly, h_2_poly];
        let h_poly_comms = h_polys
            .iter()
            .map(|h_poly| PCS::commit(ck, h_poly))
            .collect::<Result<Vec<PCS::Commitment>, _>>()?;
        Ok(((h_poly_comms, h_polys), sorted_vec, merged_lookup_table))
    }

    /// Round 2: Compute and commit the permutation grand product polynomial.
    /// Return the grand product polynomial and its commitment.
    pub(crate) fn run_2nd_round<C: Arithmetization<P::ScalarField>, R: CryptoRng + RngCore>(
        &self,
        prng: &mut R,
        ck: &<PCS::SRS as StructuredReferenceString>::ProverParam,
        cs: &C,
        challenges: &Challenges<P::ScalarField>,
    ) -> Result<(PCS::Commitment, DensePolynomial<P::ScalarField>), PlonkError> {
        let prod_perm_poly = self.mask_polynomial(
            prng,
            cs.compute_prod_permutation_polynomial(&challenges.beta, &challenges.gamma)?,
            2,
        );
        let prod_perm_comm = PCS::commit(ck, &prod_perm_poly)?;
        Ok((prod_perm_comm, prod_perm_poly))
    }

    /// Round 2.5 (Plookup): Compute and commit the Plookup grand product
    /// polynomial. Return the grand product polynomial and its commitment.
    /// `cs` is guaranteed to support lookup
    pub(crate) fn run_plookup_2nd_round<
        C: Arithmetization<P::ScalarField>,
        R: CryptoRng + RngCore,
    >(
        &self,
        prng: &mut R,
        ck: &<PCS::SRS as StructuredReferenceString>::ProverParam,
        cs: &C,
        challenges: &Challenges<P::ScalarField>,
        merged_lookup_table: Option<&Vec<P::ScalarField>>,
        sorted_vec: Option<&Vec<P::ScalarField>>,
    ) -> Result<(PCS::Commitment, DensePolynomial<P::ScalarField>), PlonkError> {
        if sorted_vec.is_none() {
            return Err(
                ParameterError("Run Plookup with empty sorted lookup vectors".to_string()).into(),
            );
        }

        let prod_lookup_poly = self.mask_polynomial(
            prng,
            cs.compute_lookup_prod_polynomial(
                &challenges.tau,
                &challenges.beta,
                &challenges.gamma,
                merged_lookup_table.unwrap(),
                sorted_vec.unwrap(),
            )?,
            2,
        );
        let prod_lookup_comm = PCS::commit(ck, &prod_lookup_poly)?;
        Ok((prod_lookup_comm, prod_lookup_poly))
    }

    /// Round 3: Return the splitted quotient polynomials and their commitments.
    /// Note that the first `num_wire_types`-1 splitted quotient polynomials
    /// have degree `domain_size`+1.
    pub(crate) fn run_3rd_round<R: CryptoRng + RngCore>(
        &self,
        prng: &mut R,
        ck: &<PCS::SRS as StructuredReferenceString>::ProverParam,
        pks: &ProvingKey<PCS>,
        challenges: &Challenges<P::ScalarField>,
        online_oracles: &Oracles<P::ScalarField>,
        num_wire_types: usize,
    ) -> Result<CommitmentsAndPolys<PCS>, PlonkError> {
        let quot_poly =
            self.compute_quotient_polynomial(challenges, pks, online_oracles, num_wire_types)?;
        let split_quot_polys = self.split_quotient_polynomial(prng, &quot_poly, num_wire_types)?;
        let split_quot_poly_comms = cfg_iter!(split_quot_polys)
            .map(|split_quot_poly| PCS::commit(ck, split_quot_poly))
            .collect::<Result<Vec<PCS::Commitment>, _>>()?;
        Ok((split_quot_poly_comms, split_quot_polys))
    }

    /// Round 4: Compute linearization polynomial and evaluate polynomials to be
    /// opened.
    ///
    /// Compute the polynomial evaluations for TurboPlonk.
    /// Return evaluations of the Plonk proof.
    pub(crate) fn compute_evaluations(
        &self,
        pk: &ProvingKey<PCS>,
        challenges: &Challenges<P::ScalarField>,
        online_oracles: &Oracles<P::ScalarField>,
        num_wire_types: usize,
    ) -> ProofEvaluations<P::ScalarField> {
        let wires_evals: Vec<P::ScalarField> =
            parallelizable_slice_iter(&online_oracles.wire_polys)
                .map(|poly| poly.evaluate(&challenges.zeta))
                .collect();
        let wire_sigma_evals: Vec<P::ScalarField> = parallelizable_slice_iter(&pk.sigmas)
            .take(num_wire_types - 1)
            .map(|poly| poly.evaluate(&challenges.zeta))
            .collect();
        let perm_next_eval = online_oracles
            .prod_perm_poly
            .evaluate(&(challenges.zeta * self.domain.group_gen));
        ProofEvaluations {
            wires_evals,
            wire_sigma_evals,
            perm_next_eval,
        }
    }

    /// Round 4.5 (Plookup): Compute and return evaluations of Plookup-related
    /// polynomials
    pub(crate) fn compute_plookup_evaluations(
        &self,
        pk: &ProvingKey<PCS>,
        challenges: &Challenges<P::ScalarField>,
        online_oracles: &Oracles<P::ScalarField>,
    ) -> Result<PlookupEvaluations<P::ScalarField>, PlonkError> {
        if pk.plookup_pk.is_none() {
            return Err(ParameterError(
                "Evaluate Plookup polynomials without supporting lookup".to_string(),
            )
            .into());
        }
        if online_oracles.plookup_oracles.h_polys.len() != 2 {
            return Err(ParameterError(
                "Evaluate Plookup polynomials without updating sorted lookup vector polynomials"
                    .to_string(),
            )
            .into());
        }

        let range_table_poly_ref = &pk.plookup_pk.as_ref().unwrap().range_table_poly;
        let key_table_poly_ref = &pk.plookup_pk.as_ref().unwrap().key_table_poly;
        let table_dom_sep_poly_ref = &pk.plookup_pk.as_ref().unwrap().table_dom_sep_poly;
        let q_dom_sep_poly_ref = &pk.plookup_pk.as_ref().unwrap().q_dom_sep_poly;

        let range_table_eval = range_table_poly_ref.evaluate(&challenges.zeta);
        let key_table_eval = key_table_poly_ref.evaluate(&challenges.zeta);
        let h_1_eval = online_oracles.plookup_oracles.h_polys[0].evaluate(&challenges.zeta);
        let q_lookup_eval = pk.q_lookup_poly()?.evaluate(&challenges.zeta);
        let table_dom_sep_eval = table_dom_sep_poly_ref.evaluate(&challenges.zeta);
        let q_dom_sep_eval = q_dom_sep_poly_ref.evaluate(&challenges.zeta);

        let zeta_mul_g = challenges.zeta * self.domain.group_gen;
        let prod_next_eval = online_oracles
            .plookup_oracles
            .prod_lookup_poly
            .evaluate(&zeta_mul_g);
        let range_table_next_eval = range_table_poly_ref.evaluate(&zeta_mul_g);
        let key_table_next_eval = key_table_poly_ref.evaluate(&zeta_mul_g);
        let h_1_next_eval = online_oracles.plookup_oracles.h_polys[0].evaluate(&zeta_mul_g);
        let h_2_next_eval = online_oracles.plookup_oracles.h_polys[1].evaluate(&zeta_mul_g);
        let q_lookup_next_eval = pk.q_lookup_poly()?.evaluate(&zeta_mul_g);
        let w_3_next_eval = online_oracles.wire_polys[3].evaluate(&zeta_mul_g);
        let w_4_next_eval = online_oracles.wire_polys[4].evaluate(&zeta_mul_g);
        let table_dom_sep_next_eval = table_dom_sep_poly_ref.evaluate(&zeta_mul_g);
        Ok(PlookupEvaluations {
            range_table_eval,
            key_table_eval,
            h_1_eval,
            q_lookup_eval,
            prod_next_eval,
            table_dom_sep_eval,
            q_dom_sep_eval,
            range_table_next_eval,
            key_table_next_eval,
            h_1_next_eval,
            h_2_next_eval,
            q_lookup_next_eval,
            w_3_next_eval,
            w_4_next_eval,
            table_dom_sep_next_eval,
        })
    }

    /// Compute linearization polynomial (excluding the quotient part)
    pub(crate) fn compute_non_quotient_component_for_lin_poly(
        &self,
        alpha_base: P::ScalarField,
        pk: &ProvingKey<PCS>,
        challenges: &Challenges<P::ScalarField>,
        online_oracles: &Oracles<P::ScalarField>,
        poly_evals: &ProofEvaluations<P::ScalarField>,
        plookup_evals: Option<&PlookupEvaluations<P::ScalarField>>,
    ) -> Result<DensePolynomial<P::ScalarField>, PlonkError> {
        let r_circ = Self::compute_lin_poly_circuit_contribution(pk, &poly_evals.wires_evals);
        let r_perm = Self::compute_lin_poly_copy_constraint_contribution(
            pk,
            challenges,
            poly_evals,
            &online_oracles.prod_perm_poly,
        );
        let mut lin_poly = r_circ + r_perm;
        // compute Plookup contribution if support lookup
        let r_lookup = plookup_evals.as_ref().map(|plookup_evals| {
            self.compute_lin_poly_plookup_contribution(
                pk,
                challenges,
                &poly_evals.wires_evals,
                plookup_evals,
                &online_oracles.plookup_oracles,
            )
        });

        if let Some(lookup_poly) = r_lookup {
            lin_poly = lin_poly + lookup_poly;
        }

        lin_poly = Self::mul_poly(&lin_poly, &alpha_base);
        Ok(lin_poly)
    }

    // Compute the Quotient part of the linearization polynomial:
    //
    // -Z_H(x) * [t1(X) + x^{n+2} * t2(X) + ... + x^{(num_wire_types-1)*(n+2)} *
    // t_{num_wire_types}(X)]
    pub(crate) fn compute_quotient_component_for_lin_poly(
        domain_size: usize,
        zeta: P::ScalarField,
        quot_polys: &[DensePolynomial<P::ScalarField>],
    ) -> Result<DensePolynomial<P::ScalarField>, PlonkError> {
        let vanish_eval = zeta.pow([domain_size as u64]) - P::ScalarField::one();
        let zeta_to_n_plus_2 = (vanish_eval + P::ScalarField::one()) * zeta * zeta;
        let mut r_quot = quot_polys.first().ok_or(PlonkError::IndexError)?.clone();
        let mut coeff = P::ScalarField::one();
        for poly in quot_polys.iter().skip(1) {
            coeff *= zeta_to_n_plus_2;
            r_quot = r_quot + Self::mul_poly(poly, &coeff);
        }
        r_quot = Self::mul_poly(&r_quot, &vanish_eval.neg());
        Ok(r_quot)
    }

    /// Compute (aggregated) polynomial opening proofs at point `zeta` and
    /// `zeta * domain_generator`. TODO: Parallelize the computation.
    pub(crate) fn create_polys_and_eval_sets<'a, 'b>(
        &self,
        pk: &'b ProvingKey<PCS>,
        zeta: &P::ScalarField,
        poly_evals: &'b ProofEvaluations<P::ScalarField>,
        oracles: &'b Oracles<P::ScalarField>,
        lin_poly: &'b DensePolynomial<P::ScalarField>,
    ) -> Result<MapAndOptionalPoly<PCS>, PlonkError>
    where
        'b: 'a,
    {
        let mut eval_sets_and_poly_refs =
            BTreeMap::<MapKey<PCS>, Vec<DensePolynomial<P::ScalarField>>>::new();

        let zeta_omega = self.domain.group_gen * zeta;

        let z_zeta = DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
            -*zeta,
            P::ScalarField::ONE,
        ]);
        let z_zeta_omega = DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
            -zeta_omega,
            P::ScalarField::ONE,
        ]);
        let z_poly = &z_zeta * &z_zeta_omega;

        let mut optional_lagrange_polys: Option<Vec<DensePolynomial<P::ScalarField>>> = None;
        let lookup_flag = pk.plookup_pk.is_some() && (oracles.plookup_oracles.h_polys.len() == 2);
        if lookup_flag {
            // List the polynomials to be opened at point `zeta`.
            let polys_ref = vec![
                lin_poly.clone(),
                &oracles.wire_polys[0]
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                        poly_evals.wires_evals[0],
                    ]),
                &oracles.wire_polys[1]
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                        poly_evals.wires_evals[1],
                    ]),
                &oracles.wire_polys[2]
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                        poly_evals.wires_evals[2],
                    ]),
                &oracles.wire_polys[5]
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                        poly_evals.wires_evals[5],
                    ]),
                &pk.sigmas[0]
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                        poly_evals.wire_sigma_evals[0],
                    ]),
                &pk.sigmas[1]
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                        poly_evals.wire_sigma_evals[1],
                    ]),
                &pk.sigmas[2]
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                        poly_evals.wire_sigma_evals[2],
                    ]),
                &pk.sigmas[3]
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                        poly_evals.wire_sigma_evals[3],
                    ]),
                &pk.sigmas[4]
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                        poly_evals.wire_sigma_evals[4],
                    ]),
                &pk.plookup_pk.as_ref().unwrap().q_dom_sep_poly
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![pk
                        .plookup_pk
                        .as_ref()
                        .unwrap()
                        .q_dom_sep_poly
                        .evaluate(zeta)]),
            ];

            eval_sets_and_poly_refs
                .entry(MapKey(1, z_zeta))
                .or_insert(polys_ref);

            // Now the polynomials that are only opened at 'zeta * omega'.
            let polys_ref = vec![
                &oracles.prod_perm_poly
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![oracles
                        .prod_perm_poly
                        .evaluate(&zeta_omega)]),
                &oracles.plookup_oracles.h_polys[1]
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![oracles
                        .plookup_oracles
                        .h_polys[1]
                        .evaluate(&zeta_omega)]),
                &oracles.plookup_oracles.prod_lookup_poly
                    - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![oracles
                        .plookup_oracles
                        .prod_lookup_poly
                        .evaluate(&zeta_omega)]),
            ];

            eval_sets_and_poly_refs
                .entry(MapKey(2, z_zeta_omega))
                .or_insert(polys_ref);

            // Now for all the polynomials evaluated at both.

            let (polys_ref, lagrange_polys) =
                Self::plookup_both_evals_polys_ref(oracles, pk, &[*zeta, zeta_omega])?;

            eval_sets_and_poly_refs
                .entry(MapKey(3, z_poly))
                .or_insert(polys_ref);

            optional_lagrange_polys = Some(lagrange_polys);
        } else {
            // List the polynomials to be opened at point `zeta`.
            let lin_poly_vec = vec![lin_poly.clone()];

            let wire_polys_zero = parallelizable_slice_iter(&oracles.wire_polys)
                .zip(poly_evals.wires_evals.par_iter())
                .map(|(poly, eval)| {
                    poly - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![*eval])
                })
                .collect::<Vec<DensePolynomial<P::ScalarField>>>();

            // Note we do not add the last wire sigma polynomial.
            let sigma_polys_zero = parallelizable_slice_iter(&pk.sigmas)
                .zip(poly_evals.wire_sigma_evals.par_iter())
                .take(pk.sigmas.len() - 1)
                .map(|(poly, eval)| {
                    poly - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![*eval])
                })
                .collect::<Vec<DensePolynomial<P::ScalarField>>>();
            let polys_ref = [lin_poly_vec, wire_polys_zero, sigma_polys_zero].concat();
            eval_sets_and_poly_refs
                .entry(MapKey(1, z_zeta))
                .or_insert(polys_ref);
            // Now the only one to be evaluated at "zeta * omega".
            eval_sets_and_poly_refs.insert(
                MapKey(2, z_zeta_omega),
                vec![
                    &oracles.prod_perm_poly
                        - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                            poly_evals.perm_next_eval,
                        ]),
                ],
            );
        }

        Ok((eval_sets_and_poly_refs, optional_lagrange_polys))
    }

    pub(crate) fn compute_q_comm_and_eval_polys(
        &self,
        ck: &<PCS::SRS as StructuredReferenceString>::ProverParam,
        polys_and_eval_points: &BTreeMap<MapKey<PCS>, Vec<DensePolynomial<P::ScalarField>>>,
        v: &P::ScalarField,
    ) -> Result<(PCS::Commitment, DensePolynomial<P::ScalarField>), PlonkError> {
        // We now compute q(x) from https://eprint.iacr.org/2020/1536.pdf.
        // For each polynomial f(x) in polys_and_evals we compute f(x)/(x-eval_1)...(x-eval_n) where
        // the eval_i are the elements of the set Omega in https://eprint.iacr.org/2020/1536.pdf.

        let polys_vec = parallelizable_btree_map(polys_and_eval_points)
            .map(|(map_key, polys)| {
                parallelizable_slice_iter(polys)
                    .map(|poly| poly / &map_key.1)
                    .collect::<Vec<DensePolynomial<P::ScalarField>>>()
            })
            .flatten()
            .collect::<Vec<DensePolynomial<P::ScalarField>>>();

        let length = polys_vec.len();

        let v_powers = vec![v; length];
        let v_powers = v_powers
            .iter()
            .enumerate()
            .map(|(power, scalar)| scalar.pow([power as u64]))
            .collect::<Vec<P::ScalarField>>();

        let q_poly = parallelizable_slice_iter(&polys_vec)
            .zip(v_powers.par_iter())
            .fold_with(
                DensePolynomial::<P::ScalarField>::zero(),
                |acc, (poly, coeff)| (acc + Self::mul_poly(poly, coeff)),
            )
            .reduce_with(|a, b| a + b)
            .ok_or(PlonkError::InvalidParameters(
                "Failed to sum polynomials correctly".to_string(),
            ))?;

        // Commit to q(x).
        let q_comm = PCS::commit(ck, &q_poly)?;

        Ok((q_comm, q_poly))
    }

    pub(crate) fn compute_g_poly_and_comm(
        &self,
        polys_and_eval_points: &BTreeMap<MapKey<PCS>, Vec<DensePolynomial<P::ScalarField>>>,
        optional_lagrange_polys: Option<Vec<DensePolynomial<P::ScalarField>>>,
        v: &P::ScalarField,
        u: &P::ScalarField,
        zeta: &P::ScalarField,
        q_poly: &DensePolynomial<P::ScalarField>,
    ) -> Result<DensePolynomial<P::ScalarField>, PlonkError> {
        let mut zeta = *zeta;
        let mut zeta_omega = zeta * self.domain.group_gen;

        // First we compute the polynomial z(x)
        let z_1_poly =
            DensePolynomial::from_coefficients_slice(&[*zeta.neg_in_place(), P::ScalarField::ONE]);
        let z_2_poly = DensePolynomial::from_coefficients_slice(&[
            *zeta_omega.neg_in_place(),
            P::ScalarField::ONE,
        ]);
        let z_poly = &z_1_poly * &z_2_poly;
        let eval_1 = z_2_poly.evaluate(u);
        let eval_2 = z_1_poly.evaluate(u);
        let q_coeff = z_poly.evaluate(u);

        // Compute the g(x) polynomial

        let polys_vec = if let Some(lagrange_polys) = optional_lagrange_polys {
            let polys_one = polys_and_eval_points
                .get(&MapKey(1, z_1_poly))
                .unwrap()
                .iter()
                .map(|poly| poly * eval_1)
                .collect::<Vec<DensePolynomial<P::ScalarField>>>();

            let polys_two = polys_and_eval_points
                .get(&MapKey(2, z_2_poly))
                .unwrap()
                .iter()
                .map(|poly| poly * eval_2)
                .collect::<Vec<DensePolynomial<P::ScalarField>>>();

            let polys_three = polys_and_eval_points
                .get(&MapKey(3, z_poly))
                .unwrap()
                .par_iter()
                .zip(lagrange_polys.par_iter())
                .map(|(poly, l_poly)| {
                    &(poly + l_poly)
                        - &DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                            l_poly.evaluate(u)
                        ])
                })
                .collect::<Vec<DensePolynomial<P::ScalarField>>>();

            [polys_one, polys_two, polys_three].concat()
        } else {
            let evals = [eval_1, eval_2, P::ScalarField::one()];
            parallelizable_btree_map(polys_and_eval_points)
                .map(|(map_key, polys)| {
                    parallelizable_slice_iter(polys)
                        .map(|poly| poly * evals[map_key.0 - 1])
                        .collect::<Vec<DensePolynomial<P::ScalarField>>>()
                })
                .flatten()
                .collect::<Vec<DensePolynomial<P::ScalarField>>>()
        };

        let sum_part = parallelizable_slice_iter(&polys_vec)
            .enumerate()
            .fold_with(
                DensePolynomial::<P::ScalarField>::zero(),
                |acc, (index, poly)| (acc + Self::mul_poly(poly, &v.pow([index as u64]))),
            )
            .reduce_with(|a, b| a + b)
            .ok_or(PlonkError::InvalidParameters(
                "Failed to sum polynomials correctly".to_string(),
            ))?;

        let z_q_poly = Self::mul_poly(q_poly, &q_coeff);

        let g_poly = &sum_part - &z_q_poly;

        Ok(g_poly)
    }
}

type TuplePolyVecs<P> = (
    Vec<DensePolynomial<<P as CurveConfig>::ScalarField>>,
    Vec<DensePolynomial<<P as CurveConfig>::ScalarField>>,
);

/// Private helper methods
impl<PCS, P> FFTProver<PCS>
where
    PCS: PolynomialCommitmentScheme<
        Evaluation = P::ScalarField,
        Polynomial = DensePolynomial<P::ScalarField>,
    >,
    PCS::Commitment: AffineRepr<Config = P, BaseField = P::BaseField, ScalarField = P::ScalarField>,
    P: HasTEForm,
    P::BaseField: RescueParameter + PrimeField,
{
    /// Return the list of plookup polynomials to be opened at point `zeta * g` and 'zeta'
    /// The order should be consistent with the verifier side.
    #[inline]
    fn plookup_both_evals_polys_ref<'a>(
        oracles: &'a Oracles<P::ScalarField>,
        pk: &'a ProvingKey<PCS>,
        points: &[P::ScalarField],
    ) -> Result<TuplePolyVecs<P>, PlonkError> {
        let poly_refs = vec![
            &pk.plookup_pk.as_ref().unwrap().range_table_poly,
            &pk.plookup_pk.as_ref().unwrap().key_table_poly,
            &oracles.plookup_oracles.h_polys[0],
            pk.q_lookup_poly()?,
            &oracles.wire_polys[3],
            &oracles.wire_polys[4],
            &pk.plookup_pk.as_ref().unwrap().table_dom_sep_poly,
        ];

        let result = poly_refs
            .into_par_iter()
            .map(|poly| {
                let evals = points
                    .par_iter()
                    .map(|point| poly.evaluate(point))
                    .collect::<Vec<P::ScalarField>>();
                let x_coeff = (evals[1] - evals[0]) / (points[1] - points[0]);
                let lagrange_poly = DensePolynomial::<P::ScalarField>::from_coefficients_vec(vec![
                    evals[0] - x_coeff * points[0],
                    x_coeff,
                ]);
                // let lagrange_poly = Self::lagrange_interpolate(points, &evals).unwrap();
                (poly - &lagrange_poly, lagrange_poly)
            })
            .collect::<(
                Vec<DensePolynomial<P::ScalarField>>,
                Vec<DensePolynomial<P::ScalarField>>,
            )>();

        Ok(result)
    }

    /// Mask the polynomial so that it remains hidden after revealing
    /// `hiding_bound` evaluations.
    fn mask_polynomial<R: CryptoRng + RngCore>(
        &self,
        prng: &mut R,
        poly: DensePolynomial<P::ScalarField>,
        hiding_bound: usize,
    ) -> DensePolynomial<P::ScalarField> {
        let mask_poly =
            <DensePolynomial<P::ScalarField> as DenseUVPolynomial<P::ScalarField>>::rand(
                hiding_bound,
                prng,
            )
            .mul_by_vanishing_poly(self.domain);
        mask_poly + poly
    }

    /// Compute the quotient polynomial via (i)FFTs.
    fn compute_quotient_polynomial(
        &self,
        challenges: &Challenges<P::ScalarField>,
        pk: &ProvingKey<PCS>,
        oracles: &Oracles<P::ScalarField>,
        num_wire_types: usize,
    ) -> Result<DensePolynomial<P::ScalarField>, PlonkError> {
        let n = self.domain.size();
        let m = self.quot_domain.size();
        ark_std::println!("m: {}, n: {}", m, n);
        let domain_size_ratio = m / n;
        // Compute 1/Z_H(w^i).
        let mut z_h_inv: Vec<P::ScalarField> = (0..domain_size_ratio)
            .map(|i| {
                (P::ScalarField::GENERATOR * self.quot_domain.element(i)).pow([n as u64])
                    - P::ScalarField::one()
            })
            .collect();

        batch_inversion(&mut z_h_inv);
        // Compute coset evaluations of the quotient polynomial.

        // TODO: figure out if the unwrap is safe/map error?
        let coset = self
            .quot_domain
            .get_coset(P::ScalarField::GENERATOR)
            .unwrap();
        // enumerate proving instances

        // lookup_flag = 1 if support Plookup argument.
        let lookup_flag = pk.plookup_pk.is_some();

        // Compute coset evaluations.
        let n_selectors = pk.selectors.len();
        let n_sigmas = pk.sigmas.len();
        let n_wires = oracles.wire_polys.len();
        for poly in pk.selectors.iter() {
            ark_std::println!("selector poly degree: {:?}", poly.degree());
        }
        let flattened = [
            pk.selectors.as_slice(),
            pk.sigmas.as_slice(),
            oracles.wire_polys.as_slice(),
        ]
        .concat();
        let coset_ffts = cfg_iter!(flattened)
            .map(|poly| coset.fft(poly.coeffs()))
            .collect::<Vec<Vec<P::ScalarField>>>();
        let selectors_coset_fft = coset_ffts[..n_selectors].to_vec();
        let sigmas_coset_fft = coset_ffts[n_selectors..n_selectors + n_sigmas].to_vec();
        let wire_polys_coset_fft =
            coset_ffts[n_selectors + n_sigmas..n_selectors + n_sigmas + n_wires].to_vec();
        // let prod_perm_poly_coset_fft = coset_ffts[n_selectors + n_sigmas + n_wires].clone();
        // let pub_input_poly_coset_fft = coset_ffts[n_selectors + n_sigmas + n_wires + 1].clone();
        // TODO: (binyi) we can also compute below in parallel with
        // `wire_polys_coset_fft`.
        let prod_perm_poly_coset_fft = coset.fft(oracles.prod_perm_poly.coeffs());
        let pub_input_poly_coset_fft = coset.fft(oracles.pub_inp_poly.coeffs());
        ark_std::println!("pub_input_poly degree: {}", oracles.pub_inp_poly.degree());

        let circ_poly = Self::compute_circuit_poly(
            &oracles.pub_inp_poly,
            pk.selectors.as_slice(),
            oracles.wire_polys.as_slice(),
        );
        let rng = &mut jf_utils::test_rng();
        for _ in 0..6 {
            let eval_elem =
                P::ScalarField::GENERATOR * self.quot_domain.element(usize::rand(rng) % m);
            let eval = circ_poly.evaluate(&eval_elem);
            ark_std::println!("circ_poly eval: {}", eval);
        }

        // Compute coset evaluations of Plookup online oracles.
        let (
            table_dom_sep_coset_fft,
            q_dom_sep_coset_fft,
            range_table_coset_fft,
            key_table_coset_fft,
            h_coset_ffts,
            prod_lookup_poly_coset_fft,
        ) = if let Some(lookup_key) = pk.plookup_pk.as_ref() {
            let [table_dom_sep_coset_fft, q_dom_sep_coset_fft, range_table_coset_fft, key_table_coset_fft, h_1_coset_fft, h_2_coset_fft, prod_lookup_poly_coset_fft]: [Vec<P::ScalarField>; 7] = parallelizable_slice_iter(&[lookup_key.table_dom_sep_poly.coeffs(), lookup_key.q_dom_sep_poly.coeffs(), lookup_key.range_table_poly.coeffs(), lookup_key.key_table_poly.coeffs(), oracles.plookup_oracles.h_polys[0].coeffs(), oracles.plookup_oracles.h_polys[1].coeffs(), oracles.plookup_oracles.prod_lookup_poly.coeffs()]).map(|coeffs| coset.fft(coeffs)).collect::<Vec<Vec<P::ScalarField>>>().try_into().unwrap();

            (
                Some(table_dom_sep_coset_fft),
                Some(q_dom_sep_coset_fft),
                Some(range_table_coset_fft),
                Some(key_table_coset_fft),
                Some(vec![h_1_coset_fft, h_2_coset_fft]),
                Some(prod_lookup_poly_coset_fft),
            )
        } else {
            (None, None, None, None, None, None)
        };

        // Compute coset evaluations of the quotient polynomial.

        let (quot_poly_coset_evals, t_circ_evals): (Vec<P::ScalarField>, Vec<P::ScalarField>) =
            parallelizable_slice_iter(&(0..m).collect::<Vec<_>>())
                .map(|&i| {
                    let (w, w_next): (Vec<P::ScalarField>, Vec<P::ScalarField>) = (0
                        ..num_wire_types)
                        .map(|j| {
                            (
                                wire_polys_coset_fft[j][i],
                                wire_polys_coset_fft[j][(i + domain_size_ratio) % m],
                            )
                        })
                        .unzip();

                    let eval_point = coset.element(i);
                    let t_circ = Self::compute_quotient_circuit_contribution(
                        i,
                        &w,
                        &pub_input_poly_coset_fft[i],
                        &selectors_coset_fft,
                    );
                    let (t_perm_1, t_perm_2) = Self::compute_quotient_copy_constraint_contribution(
                        i,
                        eval_point,
                        pk,
                        &w,
                        &prod_perm_poly_coset_fft[i],
                        &prod_perm_poly_coset_fft[(i + domain_size_ratio) % m],
                        challenges,
                        &sigmas_coset_fft,
                    );
                    let mut t1 = t_circ + t_perm_1;
                    let mut t2 = t_perm_2;

                    // add Plookup-related terms
                    if lookup_flag {
                        let (t_lookup_1, t_lookup_2) = self.compute_quotient_plookup_contribution(
                            i,
                            eval_point,
                            pk,
                            &w,
                            &w_next,
                            h_coset_ffts.as_ref().unwrap(),
                            prod_lookup_poly_coset_fft.as_ref().unwrap(),
                            range_table_coset_fft.as_ref().unwrap(),
                            key_table_coset_fft.as_ref().unwrap(),
                            selectors_coset_fft.last().unwrap(), /* TODO: add a method
                                                                  * to extract
                                                                  * q_lookup_coset_fft */
                            table_dom_sep_coset_fft.as_ref().unwrap(),
                            q_dom_sep_coset_fft.as_ref().unwrap(),
                            challenges,
                        );
                        t1 += t_lookup_1;
                        t2 += t_lookup_2;
                    }
                    let z = z_h_inv[i % domain_size_ratio];
                    (t1 * z + t2, t_circ * z)
                })
                .collect::<Vec<(P::ScalarField, P::ScalarField)>>()
                .into_iter()
                .unzip();

        // Compute the coefficient form of the quotient polynomial
        let quot_poly = DensePolynomial::from_coefficients_vec(coset.ifft(&quot_poly_coset_evals));
        let expected_degree = quotient_polynomial_degree(self.domain.size(), num_wire_types);
        if quot_poly.degree() != expected_degree {
            let t_circ_poly = DensePolynomial::from_coefficients_vec(coset.ifft(&t_circ_evals));
            ark_std::println!(
                "BAD! m: {}, n: {}, degree of quotient polynomial: {}, degree of t_circ: {}",
                m,
                n,
                quot_poly.degree(),
                t_circ_poly.degree(),
            );
            for poly in pk.selectors.iter() {
                ark_std::println!("BAD! selector poly degree: {:?}", poly.degree());
            }
            ark_std::println!(
                "BAD! pub_input_poly degree: {}",
                oracles.pub_inp_poly.degree(),
            );
        }
        Ok(quot_poly)
    }

    // Compute the i-th coset evaluation of the circuit part of the quotient
    // polynomial.
    fn compute_quotient_circuit_contribution(
        i: usize,
        w: &[P::ScalarField],
        pi: &P::ScalarField,
        selectors_coset_fft: &[Vec<P::ScalarField>],
    ) -> P::ScalarField {
        // Selectors
        // The order: q_lc, q_mul, q_hash, q_o, q_c, q_ecc
        // TODO: (binyi) get the order from a function.

        let q_mul: Vec<P::ScalarField> = (GATE_WIDTH..GATE_WIDTH + 2)
            .map(|j| selectors_coset_fft[j][i])
            .collect();
        let q_lc_sum: P::ScalarField = (0..GATE_WIDTH).fold(P::ScalarField::zero(), |acc, j| {
            acc + selectors_coset_fft[j][i] * w[j]
        });
        let q_hash_sum: P::ScalarField = (GATE_WIDTH + 2..2 * GATE_WIDTH + 2)
            .fold(P::ScalarField::zero(), |acc, j| {
                acc + selectors_coset_fft[j][i] * w[j - GATE_WIDTH - 2].pow([5])
            });
        let q_o = selectors_coset_fft[2 * GATE_WIDTH + 2][i];
        let q_c = selectors_coset_fft[2 * GATE_WIDTH + 3][i];
        let q_ecc = selectors_coset_fft[2 * GATE_WIDTH + 4][i];
        let q_x = selectors_coset_fft[2 * GATE_WIDTH + 5][i];
        let q_x2 = selectors_coset_fft[2 * GATE_WIDTH + 6][i];
        let q_y = selectors_coset_fft[2 * GATE_WIDTH + 7][i];
        let q_y2 = selectors_coset_fft[2 * GATE_WIDTH + 8][i];

        let w0w1 = w[0] * w[1];
        let w2w3 = w[2] * w[3];

        q_c + pi
            + q_lc_sum
            + w0w1 * (q_mul[0] + q_y2 * (w[0] + w[1]) + q_ecc * w2w3 * w[4])
            + w2w3 * (q_mul[1] + q_y * w2w3 + q_x * (w[0] * w[3] + w[1] * w[2]))
            + q_hash_sum
            + q_x2
                * (w[0] * (w[2] + P::ScalarField::from(2u8) * w[3])
                    + w[1] * (w[3] + P::ScalarField::from(2u8) * w[2]))
            - q_o * w[4]
    }

    // Compute the gate equation polynomial.
    fn compute_circuit_poly(
        pi_poly: &PCS::Polynomial,
        selector_polys: &[PCS::Polynomial],
        wire_polys: &[PCS::Polynomial],
    ) -> PCS::Polynomial {
        // Selectors
        // The order: q_lc, q_mul, q_hash, q_o, q_c, q_ecc

        let w0w1 = wire_polys[0].mul(&wire_polys[1]);
        let w2w3 = wire_polys[2].mul(&wire_polys[3]);

        let q_lc = (0..GATE_WIDTH).fold(PCS::Polynomial::zero(), |acc, i| {
            acc + selector_polys[i].mul(&wire_polys[i])
        });

        let q_mul: Vec<PCS::Polynomial> = (GATE_WIDTH..GATE_WIDTH + 2)
            .map(|i| selector_polys[i].clone())
            .collect();

        let wire_polys_pow = wire_polys
            .iter()
            .map(|poly| {
                let poly2 = poly.mul(poly);
                let poly4 = poly2.mul(&poly2);
                poly4.mul(poly)
            })
            .collect::<Vec<_>>();

        let q_hash = (GATE_WIDTH + 2..2 * GATE_WIDTH + 2)
            .fold(PCS::Polynomial::zero(), |acc, i| {
                acc + selector_polys[i].mul(&wire_polys_pow[i - GATE_WIDTH - 2])
            });
        let q_o = selector_polys[2 * GATE_WIDTH + 2].clone();
        let q_c = selector_polys[2 * GATE_WIDTH + 3].clone();
        let q_ecc = selector_polys[2 * GATE_WIDTH + 4].clone();
        let q_x = selector_polys[2 * GATE_WIDTH + 5].clone();
        let q_x2 = selector_polys[2 * GATE_WIDTH + 6].clone();
        let q_y = selector_polys[2 * GATE_WIDTH + 7].clone();
        let q_y2 = selector_polys[2 * GATE_WIDTH + 8].clone();

        q_c + pi_poly.clone()
            + q_lc
            + w0w1.mul(
                &(q_mul[0].clone()
                    + q_y2.mul(&(wire_polys[0].clone() + wire_polys[1].clone()))
                    + q_ecc.mul(&w2w3.mul(&wire_polys[4]))),
            )
            + w2w3.mul(
                &(q_mul[1].clone()
                    + q_y.mul(&w2w3)
                    + q_x.mul(
                        &(wire_polys[0].mul(&wire_polys[3]) + wire_polys[1].mul(&wire_polys[2])),
                    )),
            )
            + q_hash
            + q_x2.mul(
                &(wire_polys[0]
                    .mul(&(wire_polys[2].clone() + wire_polys[3].mul(P::ScalarField::from(2u8))))
                    + wire_polys[1].mul(
                        &(wire_polys[3].clone() + wire_polys[2].mul(P::ScalarField::from(2u8))),
                    )),
            )
            + wire_polys[4].mul(&-q_o)
    }

    /// Compute the i-th coset evaluation of the copy constraint part of the
    /// quotient polynomial.
    /// `eval_point` - the evaluation point.
    /// `w` - the wire polynomial coset evaluations at `eval_point`.
    /// `z_x` - the permutation product polynomial evaluation at `eval_point`.
    /// `z_xw`-  the permutation product polynomial evaluation at `eval_point *
    /// g`, where `g` is the root of unity of the original domain.
    #[allow(clippy::too_many_arguments)]
    fn compute_quotient_copy_constraint_contribution(
        i: usize,
        eval_point: P::ScalarField,
        pk: &ProvingKey<PCS>,
        w: &[P::ScalarField],
        z_x: &P::ScalarField,
        z_xw: &P::ScalarField,
        challenges: &Challenges<P::ScalarField>,
        sigmas_coset_fft: &[Vec<P::ScalarField>],
    ) -> (P::ScalarField, P::ScalarField) {
        let num_wire_types = w.len();
        let n = pk.domain_size();

        // The check that:
        //   \prod_i [w_i(X) + beta * k_i * X + gamma] * z(X)
        // - \prod_i [w_i(X) + beta * sigma_i(X) + gamma] * z(wX) = 0
        // on the vanishing set.
        // Delay the division of Z_H(X).
        //
        // Extended permutation values

        // Compute the 1st term.
        let (one, two) = w
            .iter()
            .zip((0..num_wire_types).map(|j| sigmas_coset_fft[j][i]))
            .enumerate()
            .fold((*z_x, *z_xw), |acc, (j, (&w, sigma))| {
                (
                    acc.0 * (w + pk.k()[j] * eval_point * challenges.beta + challenges.gamma),
                    acc.1 * (w + sigma * challenges.beta + challenges.gamma),
                )
            });

        // The check that z(x) = 1 at point 1.
        // (z(x)-1) * L1(x) * alpha^2 / Z_H(x) = (z(x)-1) * alpha^2 / (n * (x - 1))
        let result_2 = challenges.alpha.square() * (*z_x - P::ScalarField::one())
            / (P::ScalarField::from(n as u64) * (eval_point - P::ScalarField::one()));

        (challenges.alpha * (one - two), result_2)
    }

    /// Compute the i-th coset evaluation of the lookup constraint part of the
    /// quotient polynomial.
    /// `eval_point`: the evaluation point.
    /// `pk`: proving key.
    /// `lookup_w`: (merged) lookup witness coset evaluations at `eval_point`.
    /// `h_coset_ffts`: coset evaluations for the sorted lookup vector
    /// polynomials. `prod_lookup_coset_fft`: coset evaluations for the
    /// Plookup product polynomial. `challenges`: Fiat-shamir challenges.
    ///
    /// The coset evaluations should be non-empty. The proving key should be
    /// guaranteed to support lookup.
    #[allow(clippy::too_many_arguments)]
    fn compute_quotient_plookup_contribution(
        &self,
        i: usize,
        eval_point: P::ScalarField,
        pk: &ProvingKey<PCS>,
        w: &[P::ScalarField],
        w_next: &[P::ScalarField],
        h_coset_ffts: &[Vec<P::ScalarField>],
        prod_lookup_coset_fft: &[P::ScalarField],
        range_table_coset_fft: &[P::ScalarField],
        key_table_coset_fft: &[P::ScalarField],
        q_lookup_coset_fft: &[P::ScalarField],
        table_dom_sep_coset_fft: &[P::ScalarField],
        q_dom_sep_coset_fft: &[P::ScalarField],
        challenges: &Challenges<P::ScalarField>,
    ) -> (P::ScalarField, P::ScalarField) {
        assert!(pk.plookup_pk.is_some());
        assert_eq!(h_coset_ffts.len(), 2);

        let n = pk.domain_size();
        let m = self.quot_domain.size();
        let domain_size_ratio = m / n;
        let n_field = P::ScalarField::from(n as u64);
        let lagrange_n_coeff =
            self.domain.group_gen_inv / (n_field * (eval_point - self.domain.group_gen_inv));
        let lagrange_1_coeff =
            P::ScalarField::one() / (n_field * (eval_point - P::ScalarField::one()));
        let mut alpha_power = challenges.alpha * challenges.alpha * challenges.alpha;

        // extract polynomial evaluations
        let h_1_x = h_coset_ffts[0][i];
        let h_1_xw = h_coset_ffts[0][(i + domain_size_ratio) % m];
        let h_2_x = h_coset_ffts[1][i];
        let h_2_xw = h_coset_ffts[1][(i + domain_size_ratio) % m];
        let p_x = prod_lookup_coset_fft[i];
        let p_xw = prod_lookup_coset_fft[(i + domain_size_ratio) % m];
        let range_table_x = range_table_coset_fft[i];
        let key_table_x = key_table_coset_fft[i];
        let table_dom_sep_x = table_dom_sep_coset_fft[i];
        let q_dom_sep_x = q_dom_sep_coset_fft[i];

        let range_table_xw = range_table_coset_fft[(i + domain_size_ratio) % m];
        let key_table_xw = key_table_coset_fft[(i + domain_size_ratio) % m];
        let table_dom_sep_xw = table_dom_sep_coset_fft[(i + domain_size_ratio) % m];
        let merged_table_x = eval_merged_table::<P>(
            challenges.tau,
            range_table_x,
            key_table_x,
            q_lookup_coset_fft[i],
            w[3],
            w[4],
            table_dom_sep_x,
        );
        let merged_table_xw = eval_merged_table::<P>(
            challenges.tau,
            range_table_xw,
            key_table_xw,
            q_lookup_coset_fft[(i + domain_size_ratio) % m],
            w_next[3],
            w_next[4],
            table_dom_sep_xw,
        );
        let merged_lookup_x = eval_merged_lookup_witness::<P>(
            challenges.tau,
            w[5],
            w[0],
            w[1],
            w[2],
            q_lookup_coset_fft[i],
            q_dom_sep_x,
        );

        // The check that h1(X) - h2(wX) = 0 at point w^{n-1}
        //
        // Fh(X)/Z_H(X) = (Ln(X) * (h1(X) - h2(wX))) / Z_H(X) = (h1(X) - h2(wX)) *
        // w^{n-1} / (n * (X - w^{n-1}))
        let term_h = (h_1_x - h_2_xw) * lagrange_n_coeff;
        let mut result_2 = alpha_power * term_h;
        alpha_power *= challenges.alpha;

        // The check that p(X) = 1 at point 1.
        //
        // Fp1(X)/Z_H(X) = (L1(X) * (p(X) - 1)) / Z_H(X) = (p(X) - 1) / (n * (X - 1))
        let term_p_1 = (p_x - P::ScalarField::one()) * lagrange_1_coeff;
        result_2 += alpha_power * term_p_1;
        alpha_power *= challenges.alpha;

        // The check that p(X) = 1 at point w^{n-1}.
        //
        // Fp2(X)/Z_H(X) = (Ln(X) * (p(X) - 1)) / Z_H(X) = (p(X) - 1) * w^{n-1} / (n *
        // (X - w^{n-1}))
        let term_p_2 = (p_x - P::ScalarField::one()) * lagrange_n_coeff;
        result_2 += alpha_power * term_p_2;
        alpha_power *= challenges.alpha;

        // The relation check between adjacent points on the vanishing set.
        // Delay the division of Z_H(X).
        //
        // Fp3(X) = (X - w^{n-1}) * p(X) * (1+beta) * (gamma + merged_lookup(X)) *
        // [gamma*(1+beta) + merged_table(X) + beta * merged_table(Xw)]
        //        - (X - w^{n-1}) * p(Xw) * [gamma(1+beta) + h_1(X) + beta * h_1(Xw)] *
        //          [gamma(1+beta) + h_2(X) + beta * h_2(Xw)]
        let beta_plus_one = P::ScalarField::one() + challenges.beta;
        let gamma_mul_beta_plus_one = beta_plus_one * challenges.gamma;
        let term_p_3 = (eval_point - self.domain.group_gen_inv)
            * (p_x
                * beta_plus_one
                * (challenges.gamma + merged_lookup_x)
                * (gamma_mul_beta_plus_one + merged_table_x + challenges.beta * merged_table_xw)
                - p_xw
                    * (gamma_mul_beta_plus_one + h_1_x + challenges.beta * h_1_xw)
                    * (gamma_mul_beta_plus_one + h_2_x + challenges.beta * h_2_xw));
        let result_1 = alpha_power * term_p_3;

        (result_1, result_2)
    }

    /// Split the quotient polynomial into `num_wire_types` polynomials.
    /// The first `num_wire_types`-1 polynomials have degree `domain_size`+1.
    ///
    /// Let t(X) be the input quotient polynomial, t_i(X) be the output
    /// splitting polynomials. t(X) = \sum_{i=0}^{num_wire_types}
    /// X^{i*(n+2)} * t_i(X)
    ///
    /// NOTE: we have a step polynomial of X^(n+2) instead of X^n as in the
    /// GWC19 paper to achieve better balance among degrees of all splitting
    /// polynomials (especially the highest-degree/last one).
    fn split_quotient_polynomial<R: CryptoRng + RngCore>(
        &self,
        prng: &mut R,
        quot_poly: &DensePolynomial<P::ScalarField>,
        num_wire_types: usize,
    ) -> Result<Vec<DensePolynomial<P::ScalarField>>, PlonkError> {
        let expected_degree = quotient_polynomial_degree(self.domain.size(), num_wire_types);
        if quot_poly.degree() != expected_degree {
            return Err(WrongQuotientPolyDegree(quot_poly.degree(), expected_degree).into());
        }
        let n = self.domain.size();
        // compute the splitting polynomials t'_i(X) s.t. t(X) =
        // \sum_{i=0}^{num_wire_types} X^{i*(n+2)} * t'_i(X)
        let mut split_quot_polys: Vec<DensePolynomial<P::ScalarField>> =
            parallelizable_slice_iter(&(0..num_wire_types).collect::<Vec<_>>())
                .map(|&i| {
                    let end = if i < num_wire_types - 1 {
                        (i + 1) * (n + 2)
                    } else {
                        quot_poly.degree() + 1
                    };
                    // Degree-(n+1) polynomial has n + 2 coefficients.
                    DensePolynomial::<P::ScalarField>::from_coefficients_slice(
                        &quot_poly.coeffs[i * (n + 2)..end],
                    )
                })
                .collect();

        // mask splitting polynomials t_i(X), for i in {0..num_wire_types}.
        // t_i(X) = t'_i(X) - b_last_i + b_now_i * X^(n+2)
        // with t_lowest_i(X) = t_lowest_i(X) - 0 + b_now_i * X^(n+2)
        // and t_highest_i(X) = t_highest_i(X) - b_last_i
        let mut last_randomizer = P::ScalarField::zero();
        split_quot_polys
            .iter_mut()
            .take(num_wire_types - 1)
            .for_each(|poly| {
                let now_randomizer = P::ScalarField::rand(prng);

                poly.coeffs[0] -= last_randomizer;
                assert_eq!(poly.degree(), n + 1);
                poly.coeffs.push(now_randomizer);

                last_randomizer = now_randomizer;
            });
        // mask the highest splitting poly
        split_quot_polys[num_wire_types - 1].coeffs[0] -= last_randomizer;

        Ok(split_quot_polys)
    }

    // Compute the circuit part of the linearization polynomial
    fn compute_lin_poly_circuit_contribution(
        pk: &ProvingKey<PCS>,
        w_evals: &[P::ScalarField],
    ) -> DensePolynomial<P::ScalarField> {
        // The selectors order: q_lc, q_mul, q_hash, q_o, q_c, q_ecc
        // TODO: (binyi) get the order from a function.
        let q_lc = &pk.selectors[..GATE_WIDTH];
        let q_mul = &pk.selectors[GATE_WIDTH..GATE_WIDTH + 2];
        let q_hash = &pk.selectors[GATE_WIDTH + 2..2 * GATE_WIDTH + 2];
        let q_o = &pk.selectors[2 * GATE_WIDTH + 2];
        let q_c = &pk.selectors[2 * GATE_WIDTH + 3];
        let q_ecc = &pk.selectors[2 * GATE_WIDTH + 4];
        let q_x = &pk.selectors[2 * GATE_WIDTH + 5];
        let q_x2 = &pk.selectors[2 * GATE_WIDTH + 6];
        let q_y = &pk.selectors[2 * GATE_WIDTH + 7];
        let q_y2 = &pk.selectors[2 * GATE_WIDTH + 8];

        // TODO(binyi): add polynomials in parallel.
        // Note we don't need to compute the constant term of the polynomial.
        Self::mul_poly(&q_lc[0], &w_evals[0])
            + Self::mul_poly(&q_lc[1], &w_evals[1])
            + Self::mul_poly(&q_lc[2], &w_evals[2])
            + Self::mul_poly(&q_lc[3], &w_evals[3])
            + Self::mul_poly(&q_mul[0], &(w_evals[0] * w_evals[1]))
            + Self::mul_poly(&q_mul[1], &(w_evals[2] * w_evals[3]))
            + Self::mul_poly(&q_hash[0], &w_evals[0].pow([5]))
            + Self::mul_poly(&q_hash[1], &w_evals[1].pow([5]))
            + Self::mul_poly(&q_hash[2], &w_evals[2].pow([5]))
            + Self::mul_poly(&q_hash[3], &w_evals[3].pow([5]))
            + Self::mul_poly(
                q_ecc,
                &(w_evals[0] * w_evals[1] * w_evals[2] * w_evals[3] * w_evals[4]),
            )
            + Self::mul_poly(
                q_x,
                &(w_evals[0] * w_evals[3] * w_evals[2] * w_evals[3]
                    + w_evals[1] * w_evals[2] * w_evals[2] * w_evals[3]),
            )
            + Self::mul_poly(
                q_x2,
                &(w_evals[0] * w_evals[2]
                    + w_evals[1] * w_evals[3]
                    + P::ScalarField::from(2u8) * w_evals[0] * w_evals[3]
                    + P::ScalarField::from(2u8) * w_evals[1] * w_evals[2]),
            )
            + Self::mul_poly(q_y, &(w_evals[2] * w_evals[2] * w_evals[3] * w_evals[3]))
            + Self::mul_poly(
                q_y2,
                &(w_evals[0] * w_evals[0] * w_evals[1] + w_evals[0] * w_evals[1] * w_evals[1]),
            )
            + Self::mul_poly(q_o, &(-w_evals[4]))
            + q_c.clone()
    }

    // Compute the wire permutation part of the linearization polynomial
    fn compute_lin_poly_copy_constraint_contribution(
        pk: &ProvingKey<PCS>,
        challenges: &Challenges<P::ScalarField>,
        poly_evals: &ProofEvaluations<P::ScalarField>,
        prod_perm_poly: &DensePolynomial<P::ScalarField>,
    ) -> DensePolynomial<P::ScalarField> {
        let dividend = challenges.zeta.pow([pk.domain_size() as u64]) - P::ScalarField::one();
        let divisor = P::ScalarField::from(pk.domain_size() as u32)
            * (challenges.zeta - P::ScalarField::one());
        let lagrange_1_eval = dividend / divisor;

        // Compute the coefficient of z(X)
        let coeff = poly_evals.wires_evals.iter().enumerate().fold(
            challenges.alpha,
            |acc, (j, &wire_eval)| {
                acc * (wire_eval
                    + challenges.beta * pk.vk.k[j] * challenges.zeta
                    + challenges.gamma)
            },
        ) + challenges.alpha.square() * lagrange_1_eval;
        let mut r_perm = Self::mul_poly(prod_perm_poly, &coeff);

        // Compute the coefficient of the last sigma wire permutation polynomial
        let num_wire_types = poly_evals.wires_evals.len();
        let coeff = -poly_evals
            .wires_evals
            .iter()
            .take(num_wire_types - 1)
            .zip(poly_evals.wire_sigma_evals.iter())
            .fold(
                challenges.alpha * challenges.beta * poly_evals.perm_next_eval,
                |acc, (&wire_eval, &sigma_eval)| {
                    acc * (wire_eval + challenges.beta * sigma_eval + challenges.gamma)
                },
            );
        r_perm = r_perm + Self::mul_poly(&pk.sigmas[num_wire_types - 1], &coeff);
        r_perm
    }

    // Compute the Plookup part of the linearization polynomial
    fn compute_lin_poly_plookup_contribution(
        &self,
        pk: &ProvingKey<PCS>,
        challenges: &Challenges<P::ScalarField>,
        w_evals: &[P::ScalarField],
        plookup_evals: &PlookupEvaluations<P::ScalarField>,
        oracles: &PlookupOracles<P::ScalarField>,
    ) -> DensePolynomial<P::ScalarField> {
        let alpha_2 = challenges.alpha.square();
        let alpha_4 = alpha_2.square();
        let alpha_5 = alpha_4 * challenges.alpha;
        let alpha_6 = alpha_4 * alpha_2;
        let n = pk.domain_size();
        let one = P::ScalarField::one();
        let vanish_eval = challenges.zeta.pow([n as u64]) - one;

        // compute lagrange_1 and lagrange_n
        let divisor = P::ScalarField::from(n as u32) * (challenges.zeta - one);
        let lagrange_1_eval = vanish_eval / divisor;
        let divisor =
            P::ScalarField::from(n as u32) * (challenges.zeta - self.domain.group_gen_inv);
        let lagrange_n_eval = vanish_eval * self.domain.group_gen_inv / divisor;

        // compute the coefficient for polynomial `prod_lookup_poly`
        let merged_table_eval = eval_merged_table::<P>(
            challenges.tau,
            plookup_evals.range_table_eval,
            plookup_evals.key_table_eval,
            plookup_evals.q_lookup_eval,
            w_evals[3],
            w_evals[4],
            plookup_evals.table_dom_sep_eval,
        );
        let merged_table_next_eval = eval_merged_table::<P>(
            challenges.tau,
            plookup_evals.range_table_next_eval,
            plookup_evals.key_table_next_eval,
            plookup_evals.q_lookup_next_eval,
            plookup_evals.w_3_next_eval,
            plookup_evals.w_4_next_eval,
            plookup_evals.table_dom_sep_next_eval,
        );
        let merged_lookup_eval = eval_merged_lookup_witness::<P>(
            challenges.tau,
            w_evals[5],
            w_evals[0],
            w_evals[1],
            w_evals[2],
            plookup_evals.q_lookup_eval,
            plookup_evals.q_dom_sep_eval,
        );

        let beta_plus_one = one + challenges.beta;
        let zeta_minus_g_inv = challenges.zeta - self.domain.group_gen_inv;
        let coeff = alpha_4 * lagrange_1_eval
            + alpha_5 * lagrange_n_eval
            + alpha_6
                * zeta_minus_g_inv
                * beta_plus_one
                * (challenges.gamma + merged_lookup_eval)
                * (challenges.gamma * beta_plus_one
                    + merged_table_eval
                    + challenges.beta * merged_table_next_eval);
        let mut r_lookup = Self::mul_poly(&oracles.prod_lookup_poly, &coeff);

        // compute the coefficient for polynomial `h_2_poly`
        let coeff = -alpha_6
            * zeta_minus_g_inv
            * plookup_evals.prod_next_eval
            * (challenges.gamma * beta_plus_one
                + plookup_evals.h_1_eval
                + challenges.beta * plookup_evals.h_1_next_eval);
        r_lookup = r_lookup + Self::mul_poly(&oracles.h_polys[1], &coeff);

        r_lookup
    }

    #[inline]
    fn mul_poly(
        poly: &DensePolynomial<P::ScalarField>,
        coeff: &P::ScalarField,
    ) -> DensePolynomial<P::ScalarField> {
        DensePolynomial::<P::ScalarField>::from_coefficients_vec(
            parallelizable_slice_iter(&poly.coeffs)
                .map(|c| *coeff * c)
                .collect(),
        )
    }
}

#[inline]
fn quotient_polynomial_degree(domain_size: usize, num_wire_types: usize) -> usize {
    num_wire_types * (domain_size + 1) + 2
}

#[cfg(test)]
mod test {
    use crate::nightfall::UnivariateIpaPCS;

    use super::*;
    use ark_bls12_377::{g1::Config as Config377, Bls12_377};
    use ark_bls12_381::{g1::Config as Config381, Bls12_381};
    use ark_ec::pairing::Pairing;
    use jf_relation::gadgets::EmulationConfig;
    use jf_utils::test_rng;

    #[test]
    fn test_split_quotient_polynomial_wrong_degree() -> Result<(), PlonkError> {
        // test_split_quotient_polynomial_wrong_degree_helper::<Bn254>()?;
        test_split_quotient_polynomial_wrong_degree_helper::<Bls12_377, Config377>()?;
        test_split_quotient_polynomial_wrong_degree_helper::<Bls12_381, Config381>()
        // test_split_quotient_polynomial_wrong_degree_helper::<BW6_761>()
    }

    fn test_split_quotient_polynomial_wrong_degree_helper<E, P>() -> Result<(), PlonkError>
    where
        E: Pairing<G1Affine = Affine<P>>,
        <E as Pairing>::BaseField: RescueParameter,
        P: HasTEForm<BaseField = E::BaseField, ScalarField = E::ScalarField>,
        E::ScalarField: EmulationConfig<E::BaseField>,
    {
        let prover = FFTProver::<UnivariateIpaPCS<E>>::new(4, GATE_WIDTH + 1)?;
        let rng = &mut test_rng();
        let bad_quot_poly =
            <DensePolynomial<E::ScalarField> as DenseUVPolynomial<E::ScalarField>>::rand(25, rng);
        assert!(prover
            .split_quotient_polynomial(rng, &bad_quot_poly, GATE_WIDTH + 1)
            .is_err());
        Ok(())
    }
}
