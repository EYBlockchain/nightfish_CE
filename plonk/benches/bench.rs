// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

// For benchmark, run:
//     RAYON_NUM_THREADS=N cargo bench
// where N is the number of threads you want to use (N = 1 for single-thread).

use ark_bls12_377::{g1::Config as Config377, Bls12_377, Fq as Fq377, Fr as Fr377};
use ark_bls12_381::{Bls12_381, Fr as Fr381};
use ark_bn254::{g1::Config as Config254, Bn254, Fq as Fq254, Fr as Fr254};
use ark_bw6_761::{Fr as Fr761, BW6_761};
use nf_curves::grumpkin::{fields::Fr as GrumpkinFr, Fq as GrumpkinFq, Grumpkin};

use ark_ff::{One, PrimeField};
use ark_poly::evaluations::multivariate::{DenseMultilinearExtension, MultilinearExtension};

use ark_std::{
    rand::{distributions::Uniform, prelude::Distribution},
    sync::Arc,
    test_rng, UniformRand,
};

use jf_plonk::{
    errors::PlonkError,
    nightfall::{
        mle::{
            subroutines::{sumcheck::SumCheck, VPSumCheck},
            virtual_polynomial::VirtualPolynomial,
            zeromorph::Zeromorph,
            MLEPlonk,
        },
        FFTPlonk, PlonkIpaSnark, UnivariateIpaPCS,
    },
    proof_system::{PlonkKzgSnark, UniversalSNARK},
    transcript::{RescueTranscript, SolidityTranscript, StandardTranscript, Transcript},
    PlonkType,
};
use jf_primitives::{pcs::prelude::*, rescue::sponge::RescueCRHF};
use jf_relation::{Circuit, PlonkCircuit};
use std::time::Instant;

use criterion::{criterion_group, criterion_main, Criterion};

const NUM_REPETITIONS: usize = 10;
const NUM_GATES_LARGE: usize = 131072;
const NUM_GATES_SMALL: usize = 8192;

fn gen_circuit_for_bench<F: PrimeField>(
    num_gates: usize,
    plonk_type: PlonkType,
) -> Result<PlonkCircuit<F>, PlonkError> {
    let range_bit_len = 8;
    let mut cs: PlonkCircuit<F> = match plonk_type {
        PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
        PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(range_bit_len),
    };
    let mut a = cs.zero();
    for _ in 0..num_gates - 10 {
        a = cs.add(a, cs.one())?;
    }
    // Finalize the circuit.
    cs.finalize_for_arithmetization()?;

    Ok(cs)
}

fn gen_circuit_for_recursion_bench<F: PrimeField>(
    num_gates: usize,
    plonk_type: PlonkType,
) -> Result<PlonkCircuit<F>, PlonkError> {
    let range_bit_len = 8;
    let mut cs: PlonkCircuit<F> = match plonk_type {
        PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
        PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(range_bit_len),
    };
    let mut a = cs.zero();
    for _ in 0..num_gates - 10 {
        a = cs.add(a, cs.one())?;
    }
    // Finalize the circuit.
    cs.finalize_for_recursive_arithmetization::<RescueCRHF<Fq254>>()?;

    Ok(cs)
}

fn gen_circuit_for_mle_bench<F: PrimeField>(
    num_gates: usize,
    plonk_type: PlonkType,
) -> Result<PlonkCircuit<F>, PlonkError> {
    let range_bit_len = 8;
    let mut cs: PlonkCircuit<F> = match plonk_type {
        PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
        PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(range_bit_len),
    };
    let mut a = cs.zero();
    for _ in 0..num_gates - 10 {
        a = cs.add(a, cs.one())?;
    }
    // Finalize the circuit.
    cs.finalize_for_mle_arithmetization()?;

    Ok(cs)
}

fn gen_circuit_for_mle_recursion_bench<F: PrimeField>(
    num_gates: usize,
    plonk_type: PlonkType,
) -> Result<PlonkCircuit<F>, PlonkError> {
    let range_bit_len = 8;
    let mut cs: PlonkCircuit<F> = match plonk_type {
        PlonkType::TurboPlonk => PlonkCircuit::new_turbo_plonk(),
        PlonkType::UltraPlonk => PlonkCircuit::new_ultra_plonk(range_bit_len),
    };
    let mut a = cs.zero();
    for _ in 0..num_gates - 10 {
        a = cs.add(a, cs.one())?;
    }
    // Finalize the circuit.
    cs.finalize_for_recursive_mle_arithmetization::<RescueCRHF<Fr254>>()?;

    Ok(cs)
}

macro_rules! plonk_prove_bench {
    ($bench_curve:ty, $bench_field:ty, $bench_plonk_type:expr, $num_gates:expr) => {
        let rng = &mut jf_utils::test_rng();
        let cs = gen_circuit_for_bench::<$bench_field>($num_gates, $bench_plonk_type).unwrap();

        let max_degree = $num_gates + 2;
        let srs =
            PlonkKzgSnark::<$bench_curve>::universal_setup_for_testing(max_degree, rng).unwrap();

        let (pk, _) = PlonkKzgSnark::<$bench_curve>::preprocess(&srs, &cs).unwrap();

        let start = Instant::now();

        for _ in 0..NUM_REPETITIONS {
            let _ = PlonkKzgSnark::<$bench_curve>::prove::<_, _, StandardTranscript>(
                rng, &cs, &pk, None,
            )
            .unwrap();
        }

        println!(
            "proving time for {}, {}: {} ns/gate",
            stringify!($bench_curve),
            stringify!($bench_plonk_type),
            start.elapsed().as_nanos() / NUM_REPETITIONS as u128 / $num_gates as u128
        );
    };
}

fn ipa_prove_turbo_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit = gen_circuit_for_bench::<Fr377>(NUM_GATES_LARGE, PlonkType::TurboPlonk).unwrap();

    let max_degree = NUM_GATES_LARGE + 2;
    let srs = PlonkIpaSnark::<Bls12_377>::universal_setup_for_testing(max_degree, rng).unwrap();
    let (pk, _) = PlonkIpaSnark::<Bls12_377>::preprocess(&srs, &circuit).unwrap();

    c.bench_function("Ipa plonk prove", |b| {
        b.iter(|| {
            PlonkIpaSnark::<Bls12_377>::prove::<_, _, StandardTranscript>(rng, &circuit, &pk, None);
        })
    });
}

fn ipa_prove_ultra_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit = gen_circuit_for_bench::<Fr377>(NUM_GATES_LARGE, PlonkType::UltraPlonk).unwrap();

    let max_degree = NUM_GATES_LARGE + 2;
    let srs = PlonkIpaSnark::<Bls12_377>::universal_setup_for_testing(max_degree, rng).unwrap();
    let (pk, _) = PlonkIpaSnark::<Bls12_377>::preprocess(&srs, &circuit).unwrap();

    c.bench_function("Ipa ultraplonk prove", |b| {
        b.iter(|| {
            PlonkIpaSnark::<Bls12_377>::prove::<_, _, StandardTranscript>(rng, &circuit, &pk, None);
        })
    });
}

fn kzg_prove_turbo_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit = gen_circuit_for_bench::<Fr377>(NUM_GATES_LARGE, PlonkType::TurboPlonk).unwrap();

    let max_degree = NUM_GATES_LARGE + 2;
    let srs = PlonkKzgSnark::<Bls12_377>::universal_setup_for_testing(max_degree, rng).unwrap();
    let (pk, _) = PlonkKzgSnark::<Bls12_377>::preprocess(&srs, &circuit).unwrap();

    c.bench_function("Kzg plonk prove", |b| {
        b.iter(|| {
            PlonkKzgSnark::<Bls12_377>::prove::<_, _, RescueTranscript<Fq377>>(
                rng, &circuit, &pk, None,
            );
        })
    });
}

fn kzg_prove_ultra_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit = gen_circuit_for_bench::<Fr377>(NUM_GATES_LARGE, PlonkType::UltraPlonk).unwrap();

    let max_degree = NUM_GATES_LARGE + 2;
    let srs = PlonkKzgSnark::<Bls12_377>::universal_setup_for_testing(max_degree, rng).unwrap();
    let (pk, _) = PlonkKzgSnark::<Bls12_377>::preprocess(&srs, &circuit).unwrap();

    c.bench_function("Kzg ultraplonk prove", |b| {
        b.iter(|| {
            PlonkKzgSnark::<Bls12_377>::prove::<_, _, RescueTranscript<Fq377>>(
                rng, &circuit, &pk, None,
            );
        })
    });
}

fn fftplonk_prove_turbo_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit = gen_circuit_for_bench::<Fr377>(NUM_GATES_LARGE, PlonkType::TurboPlonk).unwrap();

    let max_degree = NUM_GATES_LARGE + 2;
    let srs = FFTPlonk::<UnivariateKzgPCS<Bls12_377>>::universal_setup_for_testing(max_degree, rng)
        .unwrap();
    let (pk, _) = FFTPlonk::<UnivariateKzgPCS<Bls12_377>>::preprocess(&srs, &circuit).unwrap();

    c.bench_function("Kzg FFT plonk prove", |b| {
        b.iter(|| {
            FFTPlonk::<UnivariateKzgPCS<Bls12_377>>::prove::<_, _, RescueTranscript<Fq377>>(
                rng, &circuit, &pk, None,
            );
        })
    });
}

fn fftplonk_prove_ultra_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit = gen_circuit_for_bench::<Fr377>(NUM_GATES_LARGE, PlonkType::UltraPlonk).unwrap();

    let max_degree = NUM_GATES_LARGE + 2;
    let srs = FFTPlonk::<UnivariateKzgPCS<Bls12_377>>::universal_setup_for_testing(max_degree, rng)
        .unwrap();
    let (pk, _) = FFTPlonk::<UnivariateKzgPCS<Bls12_377>>::preprocess(&srs, &circuit).unwrap();

    c.bench_function("Kzg FFT ultraplonk prove", |b| {
        b.iter(|| {
            FFTPlonk::<UnivariateKzgPCS<Bls12_377>>::prove::<_, _, RescueTranscript<Fq377>>(
                rng, &circuit, &pk, None,
            );
        })
    });
}

fn fftplonk_recursive_prove_ultra_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit =
        gen_circuit_for_recursion_bench::<Fr254>(NUM_GATES_LARGE, PlonkType::UltraPlonk).unwrap();

    let max_degree = NUM_GATES_LARGE + 2;
    let srs =
        FFTPlonk::<UnivariateKzgPCS<Bn254>>::universal_setup_for_testing(max_degree, rng).unwrap();
    let (pk, _) = FFTPlonk::<UnivariateKzgPCS<Bn254>>::preprocess(&srs, &circuit).unwrap();

    c.bench_function("Kzg FFT recursive ultraplonk prove", |b| {
        b.iter(|| {
            FFTPlonk::<UnivariateKzgPCS<Bn254>>::recursive_prove::<_, _, RescueTranscript<Fr254>>(
                rng, &circuit, &pk, None,
            );
        })
    });
}

fn mleplonk_prove_turbo_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit =
        gen_circuit_for_mle_bench::<Fr254>(NUM_GATES_LARGE, PlonkType::TurboPlonk).unwrap();

    let num_vars = NUM_GATES_LARGE.ilog2() as usize;
    let srs = MultilinearKzgPCS::<Bn254>::gen_srs_for_testing(rng, num_vars).unwrap();
    let (pk, _) = MLEPlonk::<MultilinearKzgPCS<Bn254>>::preprocess_helper(&circuit, &srs).unwrap();

    c.bench_function("Kzg MLE plonk prove", |b| {
        b.iter(|| {
            MLEPlonk::<MultilinearKzgPCS<Bn254>>::prove::<_, _, _, RescueTranscript<Fr254>>(
                &circuit, &pk,
            );
        })
    });
}

fn mleplonk_prove_ultra_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit =
        gen_circuit_for_mle_bench::<Fr254>(NUM_GATES_LARGE, PlonkType::UltraPlonk).unwrap();

    let num_vars = NUM_GATES_LARGE.ilog2() as usize;
    let srs = MultilinearKzgPCS::<Bn254>::gen_srs_for_testing(rng, num_vars).unwrap();
    let (pk, _) = MLEPlonk::<MultilinearKzgPCS<Bn254>>::preprocess_helper(&circuit, &srs).unwrap();

    c.bench_function("Kzg MLE ultraplonk prove", |b| {
        b.iter(|| {
            MLEPlonk::<MultilinearKzgPCS<Bn254>>::prove::<_, _, _, RescueTranscript<Fr254>>(
                &circuit, &pk,
            );
        })
    });
}

fn mleplonk_kzg_solidity_prove_ultra_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit =
        gen_circuit_for_mle_bench::<Fr254>(NUM_GATES_LARGE, PlonkType::UltraPlonk).unwrap();

    let num_vars = NUM_GATES_LARGE.ilog2() as usize;
    let srs = MultilinearKzgPCS::<Bn254>::gen_srs_for_testing(rng, num_vars).unwrap();
    let (pk, _) = MLEPlonk::<MultilinearKzgPCS<Bn254>>::preprocess_helper(&circuit, &srs).unwrap();

    c.bench_function("Kzg MLE Solidity ultraplonk prove", |b| {
        b.iter(|| {
            MLEPlonk::<MultilinearKzgPCS<Bn254>>::prove::<_, _, _, SolidityTranscript>(
                &circuit, &pk,
            );
        })
    });
}

fn mleplonk_prove_ultra_zeromorph_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit =
        gen_circuit_for_mle_recursion_bench::<GrumpkinFr>(NUM_GATES_LARGE, PlonkType::UltraPlonk)
            .unwrap();

    let num_vars = NUM_GATES_LARGE.ilog2() as usize;
    let srs = Zeromorph::<UnivariateIpaPCS<Grumpkin>>::gen_srs_for_testing(rng, num_vars).unwrap();
    let (pk, _) =
        MLEPlonk::<Zeromorph<UnivariateIpaPCS<Grumpkin>>>::preprocess_helper(&circuit, &srs)
            .unwrap();

    c.bench_function("Zeromorph MLE ultraplonk prove", |b| {
        b.iter(|| {
            let _ = MLEPlonk::<Zeromorph<UnivariateIpaPCS<Grumpkin>>>::recursive_prove::<
                _,
                _,
                RescueTranscript<Fr254>,
            >(rng, &circuit, &pk, None)
            .unwrap();
        })
    });
}

fn mleplonk_prove_ultra_zeromorph_solidity_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit =
        gen_circuit_for_mle_recursion_bench::<GrumpkinFr>(NUM_GATES_LARGE, PlonkType::UltraPlonk)
            .unwrap();

    let num_vars = NUM_GATES_LARGE.ilog2() as usize;
    let srs = Zeromorph::<UnivariateIpaPCS<Grumpkin>>::gen_srs_for_testing(rng, num_vars).unwrap();
    let (pk, _) =
        MLEPlonk::<Zeromorph<UnivariateIpaPCS<Grumpkin>>>::preprocess_helper(&circuit, &srs)
            .unwrap();

    c.bench_function("Zeromorph Solidity MLE ultraplonk prove", |b| {
        b.iter(|| {
            MLEPlonk::<Zeromorph<UnivariateIpaPCS<Grumpkin>>>::recursive_prove::<
                _,
                _,
                SolidityTranscript,
            >(rng, &circuit, &pk, None);
        })
    });
}

fn mleplonk_prove_turbo_zeromorph_benchmark(c: &mut Criterion) {
    let rng = &mut jf_utils::test_rng();
    let circuit =
        gen_circuit_for_mle_recursion_bench::<GrumpkinFr>(NUM_GATES_LARGE, PlonkType::TurboPlonk)
            .unwrap();

    let num_vars = NUM_GATES_LARGE.ilog2() as usize;
    let srs = Zeromorph::<UnivariateIpaPCS<Grumpkin>>::gen_srs_for_testing(rng, num_vars).unwrap();
    let (pk, _) =
        MLEPlonk::<Zeromorph<UnivariateIpaPCS<Grumpkin>>>::preprocess_helper(&circuit, &srs)
            .unwrap();

    c.bench_function("Zeromorph MLE turboplonk prove", |b| {
        b.iter(|| {
            MLEPlonk::<Zeromorph<UnivariateIpaPCS<Grumpkin>>>::recursive_prove::<
                _,
                _,
                RescueTranscript<Fr254>,
            >(rng, &circuit, &pk, None);
        })
    });
}

fn sumcheck_prove(c: &mut Criterion) {
    let mut rng = test_rng();

    let max_degree = usize::rand(&mut rng) % 10;
    let num_vars = 20;
    let mles = (0..20)
        .map(|_| Arc::new(DenseMultilinearExtension::<Fr254>::rand(num_vars, &mut rng)))
        .collect::<Vec<_>>();

    let range = Uniform::new(0usize, 20);
    let products = (0..3)
        .map(|_| {
            let mut product = Vec::new();
            for _ in 0..2 {
                product.push(range.sample(&mut rng));
            }
            (Fr254::one(), product)
        })
        .collect::<Vec<_>>();
    let virtual_polynomial =
        VirtualPolynomial::<Fr254>::new(max_degree, num_vars, mles.clone(), products.clone());

    let mut transcript = RescueTranscript::<Fr254>::new_transcript(b"test");

    c.bench_function("SumCheck Prover", |b| {
        b.iter(|| VPSumCheck::<Config254>::prove(&virtual_polynomial, &mut transcript))
    });
}

criterion_group! {
    name = plonk_ipa_benches;
    config = Criterion::default().sample_size(10).measurement_time(ark_std::time::Duration::from_secs(30)).warm_up_time(ark_std::time::Duration::from_secs(1));
    targets = ipa_prove_turbo_benchmark, kzg_prove_turbo_benchmark, fftplonk_prove_turbo_benchmark, mleplonk_prove_turbo_benchmark, sumcheck_prove, ipa_prove_ultra_benchmark, kzg_prove_ultra_benchmark, fftplonk_prove_ultra_benchmark, mleplonk_prove_ultra_benchmark, mleplonk_prove_ultra_zeromorph_benchmark, mleplonk_prove_turbo_zeromorph_benchmark, mleplonk_prove_ultra_zeromorph_solidity_benchmark, mleplonk_kzg_solidity_prove_ultra_benchmark, fftplonk_recursive_prove_ultra_benchmark
}

criterion_main!(plonk_ipa_benches);

macro_rules! plonk_ipa_prove_bench {
    ($bench_curve:ty, $bench_field:ty, $bench_plonk_type:expr, $num_gates:expr) => {
        let rng = &mut jf_utils::test_rng();
        let cs = gen_circuit_for_bench::<$bench_field>($num_gates, $bench_plonk_type).unwrap();

        let max_degree = $num_gates + 2;
        let srs =
            PlonkIpaSnark::<$bench_curve>::universal_setup_for_testing(max_degree, rng).unwrap();

        let (pk, _) = PlonkIpaSnark::<$bench_curve>::preprocess(&srs, &cs).unwrap();
        let start = Instant::now();

        for _ in 0..NUM_REPETITIONS {
            let _ = PlonkIpaSnark::<$bench_curve>::prove::<_, _, StandardTranscript>(
                rng, &cs, &pk, None,
            )
            .unwrap();
        }

        println!(
            "proving time for {}, {}: {} ns/gate",
            stringify!($bench_curve),
            stringify!($bench_plonk_type),
            start.elapsed().as_nanos() / NUM_REPETITIONS as u128 / $num_gates as u128
        );
    };
}

fn bench_prove() {
    plonk_prove_bench!(Bls12_381, Fr381, PlonkType::TurboPlonk, NUM_GATES_LARGE);
    plonk_prove_bench!(Bls12_377, Fr377, PlonkType::TurboPlonk, NUM_GATES_LARGE);
    plonk_prove_bench!(Bn254, Fr254, PlonkType::TurboPlonk, NUM_GATES_LARGE);
    plonk_prove_bench!(BW6_761, Fr761, PlonkType::TurboPlonk, NUM_GATES_SMALL);
    plonk_prove_bench!(Bls12_381, Fr381, PlonkType::UltraPlonk, NUM_GATES_LARGE);
    plonk_prove_bench!(Bls12_377, Fr377, PlonkType::UltraPlonk, NUM_GATES_LARGE);
    plonk_prove_bench!(Bn254, Fr254, PlonkType::UltraPlonk, NUM_GATES_LARGE);
    plonk_prove_bench!(BW6_761, Fr761, PlonkType::UltraPlonk, NUM_GATES_SMALL);
}

fn bench_ipa_prove() {
    plonk_ipa_prove_bench!(Bls12_377, Fr377, PlonkType::TurboPlonk, NUM_GATES_LARGE);
    plonk_ipa_prove_bench!(Bls12_377, Fr377, PlonkType::UltraPlonk, NUM_GATES_LARGE);
}

macro_rules! plonk_verify_bench {
    ($bench_curve:ty, $bench_field:ty, $bench_plonk_type:expr, $num_gates:expr) => {
        let rng = &mut jf_utils::test_rng();
        let cs = gen_circuit_for_bench::<$bench_field>($num_gates, $bench_plonk_type).unwrap();

        let max_degree = $num_gates + 2;
        let srs =
            PlonkKzgSnark::<$bench_curve>::universal_setup_for_testing(max_degree, rng).unwrap();

        let (pk, vk) = PlonkKzgSnark::<$bench_curve>::preprocess(&srs, &cs).unwrap();

        let proof =
            PlonkKzgSnark::<$bench_curve>::prove::<_, _, StandardTranscript>(rng, &cs, &pk, None)
                .unwrap();

        let start = Instant::now();

        for _ in 0..NUM_REPETITIONS {
            let _ =
                PlonkKzgSnark::<$bench_curve>::verify::<StandardTranscript>(&vk, &[], &proof, None)
                    .unwrap();
        }

        println!(
            "verifying time for {}, {}: {} ns",
            stringify!($bench_curve),
            stringify!($bench_plonk_type),
            start.elapsed().as_nanos() / NUM_REPETITIONS as u128
        );
    };
}

macro_rules! plonk_ipa_verify_bench {
    ($bench_curve:ty, $bench_field:ty, $bench_plonk_type:expr, $num_gates:expr) => {
        let rng = &mut jf_utils::test_rng();
        let cs = gen_circuit_for_bench::<$bench_field>($num_gates, $bench_plonk_type).unwrap();

        let max_degree = $num_gates + 2;
        let srs =
            PlonkIpaSnark::<$bench_curve>::universal_setup_for_testing(max_degree, rng).unwrap();

        let (pk, vk) = PlonkIpaSnark::<$bench_curve>::preprocess(&srs, &cs).unwrap();

        let proof =
            PlonkIpaSnark::<$bench_curve>::prove::<_, _, StandardTranscript>(rng, &cs, &pk, None)
                .unwrap();

        let start = Instant::now();

        for _ in 0..NUM_REPETITIONS {
            let _ =
                PlonkIpaSnark::<$bench_curve>::verify::<StandardTranscript>(&vk, &[], &proof, None)
                    .unwrap();
        }

        println!(
            "verifying time for {}, {}: {} ns",
            stringify!($bench_curve),
            stringify!($bench_plonk_type),
            start.elapsed().as_nanos() / NUM_REPETITIONS as u128
        );
    };
}

fn bench_verify() {
    plonk_verify_bench!(Bls12_381, Fr381, PlonkType::TurboPlonk, NUM_GATES_LARGE);
    plonk_verify_bench!(Bls12_377, Fr377, PlonkType::TurboPlonk, NUM_GATES_LARGE);
    plonk_verify_bench!(Bn254, Fr254, PlonkType::TurboPlonk, NUM_GATES_LARGE);
    plonk_verify_bench!(BW6_761, Fr761, PlonkType::TurboPlonk, NUM_GATES_SMALL);
    plonk_verify_bench!(Bls12_381, Fr381, PlonkType::UltraPlonk, NUM_GATES_LARGE);
    plonk_verify_bench!(Bls12_377, Fr377, PlonkType::UltraPlonk, NUM_GATES_LARGE);
    plonk_verify_bench!(Bn254, Fr254, PlonkType::UltraPlonk, NUM_GATES_LARGE);
    plonk_verify_bench!(BW6_761, Fr761, PlonkType::UltraPlonk, NUM_GATES_SMALL);
}

fn bench_ipa_verify() {
    plonk_ipa_verify_bench!(Bls12_377, Fr377, PlonkType::TurboPlonk, NUM_GATES_LARGE);
    plonk_ipa_verify_bench!(Bls12_377, Fr377, PlonkType::UltraPlonk, NUM_GATES_LARGE);
}

macro_rules! plonk_batch_verify_bench {
    ($bench_curve:ty, $bench_field:ty, $bench_plonk_type:expr, $num_proofs:expr) => {
        let rng = &mut jf_utils::test_rng();
        let cs = gen_circuit_for_bench::<$bench_field>(1024, $bench_plonk_type).unwrap();

        let max_degree = 1026;
        let srs =
            PlonkKzgSnark::<$bench_curve>::universal_setup_for_testing(max_degree, rng).unwrap();

        let (pk, vk) = PlonkKzgSnark::<$bench_curve>::preprocess(&srs, &cs).unwrap();

        let proof =
            PlonkKzgSnark::<$bench_curve>::prove::<_, _, StandardTranscript>(rng, &cs, &pk, None)
                .unwrap();

        let vks = vec![&vk; $num_proofs];
        let pub_input = vec![];
        let public_inputs_ref = vec![&pub_input[..]; $num_proofs];
        let proofs_ref = vec![&proof; $num_proofs];

        let start = Instant::now();

        for _ in 0..NUM_REPETITIONS {
            let _ = PlonkKzgSnark::<$bench_curve>::batch_verify::<StandardTranscript>(
                &vks,
                &public_inputs_ref[..],
                &proofs_ref,
                &vec![None; vks.len()],
            )
            .unwrap();
        }

        println!(
            "batch verifying time for {}, {}, {} proofs: {} ns/proof",
            stringify!($bench_curve),
            stringify!($bench_plonk_type),
            stringify!($num_proofs),
            start.elapsed().as_nanos() / NUM_REPETITIONS as u128 / $num_proofs as u128
        );
    };
}

fn bench_batch_verify() {
    plonk_batch_verify_bench!(Bls12_381, Fr381, PlonkType::TurboPlonk, 1000);
    plonk_batch_verify_bench!(Bls12_377, Fr377, PlonkType::TurboPlonk, 1000);
    plonk_batch_verify_bench!(Bn254, Fr254, PlonkType::TurboPlonk, 1000);
    plonk_batch_verify_bench!(BW6_761, Fr761, PlonkType::TurboPlonk, 1000);
    plonk_batch_verify_bench!(Bls12_381, Fr381, PlonkType::UltraPlonk, 1000);
    plonk_batch_verify_bench!(Bls12_377, Fr377, PlonkType::UltraPlonk, 1000);
    plonk_batch_verify_bench!(Bn254, Fr254, PlonkType::UltraPlonk, 1000);
    plonk_batch_verify_bench!(BW6_761, Fr761, PlonkType::UltraPlonk, 1000);
}

// fn main() {
//     bench_ipa_prove();
//     bench_ipa_verify();
//     bench_prove();
//     bench_verify();
//     bench_batch_verify();
// }
