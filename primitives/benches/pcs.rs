#[cfg(not(feature = "test-srs"))]
fn main() {
    panic!("We need features=['bench'] to run this benchmark.");
}

#[cfg(feature = "test-srs")]
fn main() -> Result<(), jf_primitives::pcs::prelude::PCSError> {
    bench::bench_pcs()
}

#[cfg(feature = "test-srs")]
mod bench {
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ff::UniformRand;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{sync::Arc, time::Instant};
    use jf_primitives::pcs::{
        prelude::{MultilinearKzgPCS, PCSError, PolynomialCommitmentScheme},
        StructuredReferenceString,
    };
    use jf_utils::test_rng;

    pub(crate) fn bench_pcs() -> Result<(), PCSError> {
        let mut rng = test_rng();

        // normal polynomials
        let uni_params = MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, 18)?;

        for nv in 4..19 {
            let repetition = if nv < 10 {
                100
            } else if nv < 20 {
                50
            } else {
                10
            };

            let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
            let (ck, vk) = uni_params.trim(nv)?;

            let point: Vec<_> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

            // commit
            let com = {
                let start = Instant::now();
                for _ in 0..repetition {
                    let _commit = MultilinearKzgPCS::commit(&ck, &poly)?;
                }

                println!(
                    "KZG commit for {} variables: {} ns",
                    nv,
                    start.elapsed().as_nanos() / repetition as u128
                );

                MultilinearKzgPCS::commit(&ck, &poly)?
            };

            // open
            let (proof, value) = {
                let start = Instant::now();
                for _ in 0..repetition {
                    let _open = MultilinearKzgPCS::open(&ck, &poly, &point)?;
                }

                println!(
                    "KZG open for {} variables: {} ns",
                    nv,
                    start.elapsed().as_nanos() / repetition as u128
                );
                MultilinearKzgPCS::open(&ck, &poly, &point)?
            };

            // verify
            {
                let start = Instant::now();
                for _ in 0..repetition {
                    assert!(MultilinearKzgPCS::verify(
                        &vk, &com, &point, &value, &proof
                    )?);
                }
                println!(
                    "KZG verify for {} variables: {} ns",
                    nv,
                    start.elapsed().as_nanos() / repetition as u128
                );
            }

            println!("====================================");
        }

        Ok(())
    }
}
