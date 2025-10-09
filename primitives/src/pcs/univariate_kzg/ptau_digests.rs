//! Hardcoded SHA-256 digests for PPoT 28/0080 artifacts from the official PSE bucket.

/// Canonical SHA-256 digests for PPoT 28/0080 stream artifacts.
/// Keys are the server labels: `"01"`..`"27"` and `"final"`.
/// We temporarily only deal with degree from 1 to 26 as the max degree used in nightfall_4 is 2^26.
///
/// Hashing labels: 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27  final
///
/// ## Provenance / How we obtain these values
///
/// We compute the digests in CI by **streaming** each artifact from the official
/// PSE bucket directly into `sha256sum` (no files are persisted on disk), and
/// we publish the results in the job logs and the Step Summary. The job also
/// uploads a `hashes.txt` artifact for convenience.
///
/// The workflow below can be run manually (Actions -> “Run workflow”) or on PRs.
/// Its output is a ready-to-paste Rust array that we embed in source control
/// for deterministic verification.
///
/// ### GitHub Actions workflow (YAML)
///
/// ```yaml
/// name: Compute PTAU SHA-256 (pot28_0080)
///
/// on:
///   # Run manually from the Actions tab
///   workflow_dispatch: {}
///   # Optional: also run on pull requests
///   # pull_request:
///   #   types: [opened, synchronize, reopened]
///
/// jobs:
///   hash-ppot:
///     runs-on: ubuntu-latest
///     timeout-minutes: 120
///     steps:
///       - name: Compute SHA-256 digests by streaming (no files saved)
///         shell: bash
///         run: |
///           set -euo pipefail
///           BASE="https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080"
///           OUT="hashes.txt"
///           : > "$OUT"
///
///           labels="$(printf '%02d ' $(seq 1 27)) final"
///
///           echo "Hashing labels: $labels"
///           for L in $labels; do
///             URL="${BASE}/ppot_0080_${L}.ptau"
///             echo "-> $URL"
///             # Stream bytes directly; only the hex digest is kept
///             H="$(curl -fsSL "$URL" | sha256sum | awk '{print $1}')"
///             if [[ -z "$H" ]]; then
///               echo "ERROR: empty hash for label $L" >&2
///               exit 1
///             fi
///             echo "$L $H" | tee -a "$OUT"
///           done
///
///           {
///             echo "### PPoT 28/0080 SHA-256 digests"
///             echo
///             echo '```text'
///             cat "$OUT"
///             echo '```'
///             echo
///             echo "#### Rust snippet"
///             echo
///             echo '```rust'
///             echo 'pub const PSE_PPOT_SHA256: &[(&str, &str)] = &['
///             while read -r L H; do
///               printf '    (\"%s\", \"%s\"),\n' "$L" "$H"
///             done < "$OUT"
///             echo '];'
///             echo '```'
///           } >> "$GITHUB_STEP_SUMMARY"
///
///       - name: Upload hashes as artifact
///         uses: actions/upload-artifact@v4
///         with:
///           name: ppot28_0080-sha256
///           path: hashes.txt
///           if-no-files-found: error
/// ```
///
/// ### Operational notes
///
/// * We **do not** trust disk sidecars. The embedded table is the root of
///   trust; the binary verifies downloads against these digests on every run.
/// * For ad-hoc testing or alternative ceremonies, an environment override
///   is supported (e.g., `NIGHTFALL_PTAU_SHA256_07` or `NIGHTFALL_PTAU_SHA256_FINAL`).
/// * When upstream rotates artifacts, re-run the workflow, update this table,
///   and commit the change. Consider signing the release/tag that updates it.
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_01.ptau
// 01 c6874ee66aed417f1c6e8472f0851ae3125756c5855b7254f58e6c08b3fa6056
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_02.ptau
// 02 d1df5e998972f10d4d5295893ad8ffc6cfb2ba02134071705630f5392499173a
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_03.ptau
// 03 3b009905582ca9f1d11ca84d29f14a1d7a15aec10d6bd780fd7917dbdf8d86a1
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_04.ptau
// 04 e1fc46e17e9dcf344c0af3f4863874e9814a4d69e9e30ab47fa1c2673c3e86e6
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_05.ptau
// 05 9c1bfc4a2895b9beb1df6e95c96785dc9382bee84d84a0d8c56b1061e09d89d1
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_06.ptau
// 06 dc32ccae808b5ada09fdba465c939897b741cb321b48819f06b33def7bbbb208
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_07.ptau
// 07 cbc0287ae471c59e89f46ba0b61f19613364fb182ca544db8a5abc37765d80aa
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_08.ptau
// 08 5c411c13838e8e3ff80b6f87b81a0a92a66d2e64e82dd6af88ffc8ea89a548f6
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_09.ptau
// 09 67e59e346767e179a5bd88b4789a56ce376708244c01a32b520ee208dbfb2595
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_10.ptau
// 10 75b53cf75b5a1068d3a19575bf8164231b8a4babbf15383490de9c0144a15bf8
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_11.ptau
// 11 e6569fe93a6a3ba25b1d0aee9e314fee91820e4989b6b0eb34518b0149862284
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_12.ptau
// 12 35e163120e724a60853d0dd76ec54037f7c7b00584392255f71a4341d5a05c50
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_13.ptau
// 13 ccee28086e4b81d81a6e16fdee054d1dbd5276362e2662d4205d31de45cb930f
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_14.ptau
// 14 3ca1149e9349b22b0ee0649399cfb787677129b7b1189d1899fc0d615d9583db
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_15.ptau
// 15 4a8eb5a754ed4710d6ad603c2e1a89c72325a448e4474f8694d2c71c7c272f3d
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_16.ptau
// 16 ed3622a7c79b0b49aadd134ebbc5b77df8c8c59bccebdfd0d9bf2c1a51561cf9
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_17.ptau
// 17 f807e065fde53f72f4bf4d57140fab85b26daa6cc95bdfec7cce93622b3a367c
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_18.ptau
// 18 9693220206afab749e3d88d4ab5fdf5d36120ea102e7e587ccea0e7a5208e711
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_19.ptau
// 19 d0cadfb7d5bad6c5e0b012fb1c4bbaa3cc98fe249988f9f99d1818175e4c2015
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_20.ptau
// 20 560412532a1205145d5f21585274fb2cef61273496ddc7186aec855aab01a8cd
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_21.ptau
// 21 b4131202088141088d83fc05d654a142b51117df0960568613069e0719e7645a
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_22.ptau
// 22 9f50df02e370796098cecbb025d46a44c7165cac8bc5ebedd76d97f1129d9653
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_23.ptau
// 23 05085994c8aa34e4925585202e178e09fb8acc04e9565311751e8a04dec81cb1
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_24.ptau
// 24 d21a509863a643b8fd15af9b2f6f8af9b5928b3138af591edc2b7a731b8c2938
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_25.ptau
// 25 a91fd8e8ed3332b5ebafc90bc05329c874133f7d47c6bfe33e26dca0c2d11ee1
// -> https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_26.ptau
// 26 b354d098efff1c5ded84124fa9020eb2620b0faa62c2c7989217e062bf387651
pub const PSE_PPOT_SHA256: &[(&str, &str)] = &[
    (
        "01",
        "c6874ee66aed417f1c6e8472f0851ae3125756c5855b7254f58e6c08b3fa6056",
    ),
    (
        "02",
        "d1df5e998972f10d4d5295893ad8ffc6cfb2ba02134071705630f5392499173a",
    ),
    (
        "03",
        "3b009905582ca9f1d11ca84d29f14a1d7a15aec10d6bd780fd7917dbdf8d86a1",
    ),
    (
        "04",
        "e1fc46e17e9dcf344c0af3f4863874e9814a4d69e9e30ab47fa1c2673c3e86e6",
    ),
    (
        "05",
        "9c1bfc4a2895b9beb1df6e95c96785dc9382bee84d84a0d8c56b1061e09d89d1",
    ),
    (
        "06",
        "dc32ccae808b5ada09fdba465c939897b741cb321b48819f06b33def7bbbb208",
    ),
    (
        "07",
        "cbc0287ae471c59e89f46ba0b61f19613364fb182ca544db8a5abc37765d80aa",
    ),
    (
        "08",
        "5c411c13838e8e3ff80b6f87b81a0a92a66d2e64e82dd6af88ffc8ea89a548f6",
    ),
    (
        "09",
        "67e59e346767e179a5bd88b4789a56ce376708244c01a32b520ee208dbfb2595",
    ),
    (
        "10",
        "75b53cf75b5a1068d3a19575bf8164231b8a4babbf15383490de9c0144a15bf8",
    ),
    (
        "11",
        "e6569fe93a6a3ba25b1d0aee9e314fee91820e4989b6b0eb34518b0149862284",
    ),
    (
        "12",
        "35e163120e724a60853d0dd76ec54037f7c7b00584392255f71a4341d5a05c50",
    ),
    (
        "13",
        "ccee28086e4b81d81a6e16fdee054d1dbd5276362e2662d4205d31de45cb930f",
    ),
    (
        "14",
        "3ca1149e9349b22b0ee0649399cfb787677129b7b1189d1899fc0d615d9583db",
    ),
    (
        "15",
        "4a8eb5a754ed4710d6ad603c2e1a89c72325a448e4474f8694d2c71c7c272f3d",
    ),
    (
        "16",
        "ed3622a7c79b0b49aadd134ebbc5b77df8c8c59bccebdfd0d9bf2c1a51561cf9",
    ),
    (
        "17",
        "f807e065fde53f72f4bf4d57140fab85b26daa6cc95bdfec7cce93622b3a367c",
    ),
    (
        "18",
        "9693220206afab749e3d88d4ab5fdf5d36120ea102e7e587ccea0e7a5208e711",
    ),
    (
        "19",
        "d0cadfb7d5bad6c5e0b012fb1c4bbaa3cc98fe249988f9f99d1818175e4c2015",
    ),
    (
        "20",
        "560412532a1205145d5f21585274fb2cef61273496ddc7186aec855aab01a8cd",
    ),
    (
        "21",
        "b4131202088141088d83fc05d654a142b51117df0960568613069e0719e7645a",
    ),
    (
        "22",
        "9f50df02e370796098cecbb025d46a44c7165cac8bc5ebedd76d97f1129d9653",
    ),
    (
        "23",
        "05085994c8aa34e4925585202e178e09fb8acc04e9565311751e8a04dec81cb1",
    ),
    (
        "24",
        "d21a509863a643b8fd15af9b2f6f8af9b5928b3138af591edc2b7a731b8c2938",
    ),
    (
        "25",
        "a91fd8e8ed3332b5ebafc90bc05329c874133f7d47c6bfe33e26dca0c2d11ee1",
    ),
    (
        "26",
        "b354d098efff1c5ded84124fa9020eb2620b0faa62c2c7989217e062bf387651",
    ),
    // ("27", "PLACEHOLDER_FOR_27"), // Placeholder, replace with actual hash if needed
    // ("final", "PLACEHOLDER_FOR_FINAL"), // Placeholder, replace with actual hash if needed
];

pub fn expected_sha256_for_label(label: &str) -> Option<&'static str> {
    PSE_PPOT_SHA256
        .iter()
        .find(|(k, _)| *k == label)
        .map(|(_, v)| *v)
}
