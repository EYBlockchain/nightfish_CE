//! Module containing code to perform a lookup check using the protocol described in `https://eprint.iacr.org/2023/1284.pdf`.
//! The idea is that you have already constructed your merged lookup table and you wish to prove that a merged lookup wire
//! is contained within said table.
//!
//! The proving works using the GKR protocol to reduce the number of commitments sent to the SNARK verifier.
//! Verification is just a deferred check here and else where it must be confirmed that
//!               p(x,y) + lambda * q(x,y) = deferred_check_eval.
//! Since we have only one table and one lookup wire this check is that same as
//!              m(r)(alpha - w(r)) - (alpha - t(r)) + lambda * (alpha - t(r)) * (alpha - w(r))
//!                                                                           = deferred_check_eval.

use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{cfg_into_iter, cfg_iter, sync::Arc, vec, vec::Vec};
use dashmap::DashMap;
use jf_primitives::rescue::RescueParameter;
use jf_relation::gadgets::{ecc::HasTEForm, EmulationConfig};
use rayon::prelude::*;

use crate::{
    errors::PlonkError,
    nightfall::mle::{
        mle_structs::PolynomialError,
        subroutines::gkr::StructuredCircuit,
        utils::{compute_multiplicity_poly, scale_mle},
    },
    transcript::Transcript,
};

use super::gkr::{GKRDeferredCheck, GKRProof};

/// A struct used for proving and verifying lookup check claims.
pub struct LookupCheck<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
{
    _marker: ark_std::marker::PhantomData<P>,
}

/// The result of reducing a lookup check proof to a GKR proof.
pub struct LookupCheckReductionResult<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
{
    /// The p poly.
    pub p_poly: Arc<DenseMultilinearExtension<P::ScalarField>>,
    /// The q poly.
    pub q_poly: Arc<DenseMultilinearExtension<P::ScalarField>>,
}

impl<P> LookupCheckReductionResult<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
{
    /// A new LookupCheckReductionResult.
    pub fn new(
        p_poly: Arc<DenseMultilinearExtension<P::ScalarField>>,
        q_poly: Arc<DenseMultilinearExtension<P::ScalarField>>,
    ) -> Self {
        Self { p_poly, q_poly }
    }
}

/// A struct storing all the relevant information about a lookup table.
pub struct LogUpTable<F: PrimeField> {
    table: Arc<DenseMultilinearExtension<F>>,
    multiplicities: DashMap<F, F>,
}

impl<F: PrimeField> LogUpTable<F> {
    /// Create a new lookup table.
    pub fn new(table: Arc<DenseMultilinearExtension<F>>) -> Self {
        let multiplicities = DashMap::new();
        let table_evals = table.to_evaluations();
        cfg_iter!(table_evals).for_each(|&eval| {
            multiplicities
                .entry(eval)
                .and_modify(|v| *v += F::one())
                .or_insert(F::one());
        });
        Self {
            table,
            multiplicities,
        }
    }

    /// Get the multiplicity of a value in the lookup table.
    pub fn get_multiplicity(&self, value: F) -> Option<F> {
        self.multiplicities.get(&value).map(|v| *v)
    }

    /// Get the lookup table.
    pub fn get_table(&self) -> Arc<DenseMultilinearExtension<F>> {
        self.table.clone()
    }
}

impl<P> LookupCheck<P>
where
    P: HasTEForm,
    P::BaseField: PrimeField + RescueParameter,
    P::ScalarField: EmulationConfig<P::BaseField>,
{
    /// Create a new lookup check proof. We passs in the value alpha from the transcript because we wish to be able to compute all the challenges prior to running any SumChecks.
    pub fn prove<T: Transcript>(
        prepped_items: &LookupCheckReductionResult<P>,
        transcript: &mut T,
    ) -> Result<(GKRProof<P::ScalarField>, P::ScalarField), PlonkError> {
        let structured_circuit =
            StructuredCircuit::new(prepped_items.p_poly.clone(), prepped_items.q_poly.clone())?;
        let (_, q_out) = structured_circuit.output()?;
        let gkr_proof = structured_circuit.prove::<P, T>(transcript)?;

        Ok((gkr_proof, q_out))
    }

    /// Verify a lookup check proof. To perform the final deferred verification we recalculate the claimed evaluation from the final SumCheck proof
    /// using the claimed evaluations of the individual mles we are provided. These claimed mle evaluations are then checked via PCS opening proofs.
    pub fn verify<T: Transcript>(
        proof: &GKRProof<P::ScalarField>,
        transcript: &mut T,
    ) -> Result<GKRDeferredCheck<P::ScalarField>, PlonkError> {
        StructuredCircuit::verify::<P, T>(proof, transcript)
    }

    /// Reduces the lookup check proof to a GKR proof.
    pub fn reduce_to_gkr(
        table: &LogUpTable<P::ScalarField>,
        lookup_wire: Arc<DenseMultilinearExtension<P::ScalarField>>,
        m_poly: &Arc<DenseMultilinearExtension<P::ScalarField>>,
        alpha: P::ScalarField,
        zeta: P::ScalarField,
    ) -> Result<LookupCheckReductionResult<P>, PlonkError> {
        let num_vars = lookup_wire.num_vars();
        let alpha_zeta = alpha * zeta;
        let shifted_table = cfg_into_iter!(table.get_table().to_evaluations())
            .map(|eval| alpha_zeta - zeta * eval)
            .collect::<Vec<P::ScalarField>>();

        let shifted_lookup_wire = cfg_into_iter!(&lookup_wire.evaluations)
            .map(|eval| alpha_zeta - zeta * eval)
            .collect::<Vec<P::ScalarField>>();

        let scaled_m_poly = scale_mle(m_poly, zeta);

        let p_poly = Arc::new(
            DenseMultilinearExtension::<P::ScalarField>::from_evaluations_vec(
                num_vars + 1,
                [vec![-zeta; 1 << num_vars], scaled_m_poly].concat(),
            ),
        );
        let q_poly = Arc::new(
            DenseMultilinearExtension::<P::ScalarField>::from_evaluations_vec(
                num_vars + 1,
                [shifted_lookup_wire, shifted_table].concat(),
            ),
        );

        Ok(LookupCheckReductionResult::new(p_poly, q_poly))
    }

    /// Calculates the multiplicity polynomial for a lookup wire.
    pub fn calculate_m_poly(
        lookup_wire: &Arc<DenseMultilinearExtension<P::ScalarField>>,
        table: &LogUpTable<P::ScalarField>,
    ) -> Result<Arc<DenseMultilinearExtension<P::ScalarField>>, PolynomialError> {
        compute_multiplicity_poly(lookup_wire, table)
    }
}

#[cfg(test)]
mod tests {
    use crate::transcript::{RescueTranscript, Transcript};

    use super::*;
    use ark_bn254::{g1::Config as BnConfig, Fq, Fr};
    use ark_ff::{One, UniformRand, Zero};
    use ark_std::rand::{distributions::Uniform, prelude::*};

    fn build_table_and_lookup_wire<F: PrimeField>(
        num_vars: usize,
    ) -> (LogUpTable<F>, Arc<DenseMultilinearExtension<F>>) {
        let mut rng = ark_std::test_rng();
        let table = Arc::new(DenseMultilinearExtension::<F>::rand(num_vars, &mut rng));
        let n = 1 << num_vars;
        let lookup_evals = ark_std::test_rng()
            .sample_iter(&Uniform::new(0, n))
            .take(n)
            .map(|i| table.clone()[i])
            .collect::<Vec<F>>();
        let lookup_wire = Arc::new(DenseMultilinearExtension::<F>::from_evaluations_vec(
            num_vars,
            lookup_evals,
        ));
        let logup_table = LogUpTable::new(table);
        (logup_table, lookup_wire)
    }

    #[test]
    fn test_full_logup() {
        let mut rng = jf_utils::test_rng();

        for num_vars in 1usize..20 {
            let (logup_table, lookup_wire) = build_table_and_lookup_wire::<Fr>(num_vars);
            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let alpha = Fr::rand(&mut rng) * Fr::from(345u32);
            let zeta = Fr::rand(&mut rng) * Fr::from(345u32);
            let m_poly =
                LookupCheck::<BnConfig>::calculate_m_poly(&lookup_wire, &logup_table).unwrap();
            if num_vars == 3 {
                let mut num_sum = Fr::zero();
                let mut other_sum = Fr::zero();
                for i in 0..(1 << num_vars) {
                    ark_std::println!(
                        "i: {}, table: {}, lookup: {}, m_poly: {}",
                        i,
                        logup_table.table[i],
                        lookup_wire[i],
                        m_poly[i]
                    );
                    num_sum += (m_poly[i] / (alpha - logup_table.table[i]))
                        - (Fr::one() / (alpha - lookup_wire[i]));

                    other_sum += zeta
                        * (m_poly[i] * (alpha - lookup_wire[i]) - (alpha - logup_table.table[i]));
                    ark_std::println!("num_sum: {}, other sum : {}", num_sum, other_sum);
                }
            }
            let prepped_items = LookupCheck::<BnConfig>::reduce_to_gkr(
                &logup_table,
                lookup_wire.clone(),
                &m_poly,
                alpha,
                zeta,
            )
            .unwrap();
            let (proof, _) =
                LookupCheck::<BnConfig>::prove(&prepped_items, &mut transcript).unwrap();

            let mut transcript = <RescueTranscript<Fq> as Transcript>::new_transcript(b"test");
            let gkr_deferred_check =
                LookupCheck::<BnConfig>::verify(&proof, &mut transcript).unwrap();
            let evals = gkr_deferred_check.evals();
            assert_eq!(
                zeta * alpha - evals[2],
                zeta * lookup_wire.evaluate(gkr_deferred_check.point()).unwrap()
            );
            assert_eq!(
                zeta * alpha - evals[3],
                zeta * logup_table
                    .table
                    .evaluate(gkr_deferred_check.point())
                    .unwrap()
            );
        }
    }
}
