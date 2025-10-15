extern crate std;
use ark_bn254::Fq2Config;
use ark_ec::short_weierstrass::{Affine, SWCurveConfig};
use ark_ff::{fields::PrimeField, BigInt, Fp2};
use ark_serialize::{CanonicalDeserialize, Read, SerializationError};
use ark_std::{boxed::Box, fmt::Display, io, string::String, vec, vec::Vec};
use byteorder::{LittleEndian, ReadBytesExt};
use std::{
    collections::BTreeMap,
    error::Error,
    fmt::Formatter,
    fs::File,
    io::{Seek, SeekFrom},
    path::PathBuf,
};
type IoResult<T> = Result<T, SerializationError>;
type G1Points<POne> = Vec<Affine<POne>>;
type G2Points<PTwo> = Vec<Affine<PTwo>>;
type ParseResult<POne, PTwo> = Result<(G1Points<POne>, G2Points<PTwo>), PtauError>;

/// Custom trait that includes `new_unchecked` for `PrimeField`.
pub trait PrimeFieldEx: PrimeField {
    /// Create a new field element without checking if it is in the field.
    fn new_unchecked(bigint: BigInt<4>) -> Self;
}

/// Implement the trait for `ark_bn254::Fq`.
impl PrimeFieldEx for ark_bn254::Fq {
    fn new_unchecked(bigint: BigInt<4>) -> Self {
        Self::new_unchecked(bigint)
    }
}
/// Helper function to deserialize a field element.
fn deserialize_field<R: Read, F>(reader: &mut R) -> IoResult<F>
where
    F: PrimeFieldEx + From<BigInt<4>>,
{
    let bigint = BigInt::<4>::deserialize_uncompressed(reader)?;
    Ok(F::new_unchecked(bigint))
}

/// The number of bytes in the magic string
const MAGIC_STRING_LEN: usize = 4;
/// The expected magic string
const MAGIC_STRING: &[u8; MAGIC_STRING_LEN] = b"ptau";
/// The expected version of the ptau file
const EXPECTED_VERSION: u32 = 1;
/// The expected number of sections in a ptau file
const EXPECTED_NUM_SECTIONS: u32 = 11;

// // -------------------------
// // | Parsing the ptau file |
// // -------------------------
#[derive(Debug)]
/// The error type for parsing a ptau file.
pub enum PtauError {
    /// The checksum of the downloaded file does not match the expected checksum.
    DownloadedChecksumMismatch {
        label: String,
        source: Box<dyn Error>,
    },
    /// IO error
    IoError(io::Error),
    /// Network error during download
    NetworkError(String),
    /// The max degree is invalid.
    InvalidMaxDegree,
    /// The magic string in the ptau file is invalid.
    InvalidMagicString,
    /// The version of the ptau file is invalid.
    InvalidVersion,
    /// The prime order in the ptau file is invalid.
    InvalidPrimeOrder,
    /// The number of sections in the ptau file is invalid.
    InvalidNumSections,
    /// The number of G1 points in the ptau file is invalid.
    InvalidNumG1Points,
    /// The number of G2 points in the ptau file is invalid.
    InvalidNumG2Points,
    /// An error occurred while reading a G1 point.
    InvalidG1Point,
    /// An error occurred while reading a G2 point.
    InvalidG2Point,
    /// An error occurred while reading the ptau file.
    InvalidPtauFile,
    /// An error occurred during serialization or deserialization.
    SerializationError(SerializationError),
}
impl Error for PtauError {}
impl Display for PtauError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DownloadedChecksumMismatch { label, source } => {
                write!(f, "Downloaded checksum mismatch for {}: {}", label, source)
            },
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::NetworkError(e) => write!(f, "Network error: {e}"),
            Self::InvalidMaxDegree => write!(f, "The max degree is invalid"),
            Self::InvalidMagicString => write!(f, "Failed to read magic string from ptau file"),
            Self::InvalidVersion => write!(f, "Failed to read version from ptau file"),
            Self::InvalidPrimeOrder => write!(f, "Failed to read prime order from ptau file"),
            Self::InvalidNumSections => {
                write!(f, "Failed to read number of sections from ptau file")
            },
            Self::InvalidNumG1Points => {
                write!(f, "Failed to read number of G1 points from ptau file")
            },
            Self::InvalidNumG2Points => {
                write!(f, "Failed to read number of G2 points from ptau file")
            },
            Self::InvalidG1Point => write!(f, "Failed to read G1 point from ptau file"),
            Self::InvalidG2Point => write!(f, "Failed to read G2 point from ptau file"),
            Self::InvalidPtauFile => write!(f, "Failed to read ptau file"),
            Self::SerializationError(e) => {
                write!(f, "{e}")
            },
        }
    }
}
impl From<SerializationError> for PtauError {
    fn from(err: SerializationError) -> PtauError {
        PtauError::SerializationError(err)
    }
}
impl From<io::Error> for PtauError {
    fn from(err: io::Error) -> PtauError {
        PtauError::IoError(err)
    }
}

/// Read the ptau file
pub fn parse_ptau_file<F, POne, PTwo>(
    ptau_file: &PathBuf,
    num_g1_points: usize,
    num_g2_points: usize,
) -> ParseResult<POne, PTwo>
where
    F: PrimeField + core::convert::From<ark_ff::BigInt<4>> + PrimeFieldEx,
    POne: SWCurveConfig<BaseField = F>, // G1 curve config
    PTwo: SWCurveConfig<BaseField = Fp2<Fq2Config>>, // G2 curve config
{
    let mut f = File::open(ptau_file).map_err(|_| PtauError::InvalidPtauFile)?;
    // Validate the magic string ("ptau").
    let mut magic_string_buf = [0u8; MAGIC_STRING_LEN];
    f.read_exact(&mut magic_string_buf)
        .map_err(|_| PtauError::InvalidMagicString)?;
    assert_eq!(&magic_string_buf, MAGIC_STRING);

    // Read and validate the version.
    let version = f
        .read_u32::<LittleEndian>()
        .map_err(|_| PtauError::InvalidVersion)?;
    assert_eq!(version, EXPECTED_VERSION);

    // Read the number of sections (a 32-bit little-endian uint)
    let num_sections = f
        .read_u32::<LittleEndian>()
        .map_err(|_| PtauError::InvalidNumSections)?;
    assert_eq!(num_sections, EXPECTED_NUM_SECTIONS);

    // Read the section offsets and sizes.
    let mut sections = BTreeMap::<usize, u64>::new();
    for _ in 0..num_sections {
        let section_num = f
            .read_u32::<LittleEndian>()
            .map_err(|_| PtauError::InvalidNumSections)?;
        let section_size = f
            .read_i64::<LittleEndian>()
            .map_err(|_| PtauError::InvalidNumSections)?;
        let pos = f
            .stream_position()
            .map_err(|_| PtauError::InvalidNumSections)?;
        f.seek(SeekFrom::Current(section_size))
            .map_err(|_| PtauError::InvalidNumSections)?;
        sections.insert(section_num as usize, pos);
    }

    // Read the header (section 1)
    f.seek(SeekFrom::Start(sections[&1]))
        .map_err(|_| PtauError::InvalidPrimeOrder)?;
    let n8 = f
        .read_u32::<LittleEndian>()
        .map_err(|_| PtauError::InvalidPrimeOrder)?;
    let mut q_buf = vec![0u8; n8 as usize];
    f.read_exact(&mut q_buf)
        .map_err(|_| PtauError::InvalidPrimeOrder)?;

    // Ensure q_buf is not all 0s.
    if q_buf.iter().all(|&b| b == 0u8) {
        return Err(PtauError::InvalidPrimeOrder);
    }

    // Read q_buf as an Fq element.
    let q = F::from_le_bytes_mod_order(&q_buf);
    if q != F::zero() {
        return Err(PtauError::InvalidPrimeOrder);
    }
    // Read the power and ceremony power.
    let power = f
        .read_u32::<LittleEndian>()
        .map_err(|_| PtauError::InvalidPrimeOrder)?;
    let _ceremony_power = f
        .read_u32::<LittleEndian>()
        .map_err(|_| PtauError::InvalidPrimeOrder)?;

    // Validate the number of points.

    let max_g2_points = 1 << power;
    let max_g1_points = max_g2_points * 2 - 1;
    if num_g1_points > max_g1_points {
        return Err(PtauError::InvalidNumG1Points);
    }
    if num_g2_points > max_g2_points {
        return Err(PtauError::InvalidNumG2Points);
    }

    // Read the G1 points
    let mut g1_points = Vec::<Affine<POne>>::with_capacity(num_g1_points);
    f.seek(SeekFrom::Start(sections[&2]))
        .map_err(|_| PtauError::InvalidG1Point)?;
    for _ in 0..num_g1_points {
        let mut x_buf = [0u8; 32];
        let mut y_buf = [0u8; 32];
        f.read_exact(&mut x_buf)
            .map_err(|_| PtauError::InvalidG1Point)?;
        f.read_exact(&mut y_buf)
            .map_err(|_| PtauError::InvalidG1Point)?;

        let x = deserialize_field(&mut &x_buf[..]).map_err(|_| PtauError::InvalidG1Point)?;
        let y = deserialize_field(&mut &y_buf[..]).map_err(|_| PtauError::InvalidG1Point)?;

        // Construct G1 point, ensuring it is on the curve.
        let g1 = Affine::<POne>::new(x, y);
        g1_points.push(g1);
    }

    // Read the G2 points
    let mut g2_points = Vec::<Affine<PTwo>>::with_capacity(num_g2_points);
    f.seek(SeekFrom::Start(sections[&3]))
        .map_err(|_| PtauError::InvalidG2Point)?;
    for _ in 0..num_g2_points {
        let mut x0_buf = [0u8; 32];
        let mut x1_buf = [0u8; 32];
        let mut y0_buf = [0u8; 32];
        let mut y1_buf = [0u8; 32];
        f.read_exact(&mut x0_buf)
            .map_err(|_| PtauError::InvalidG2Point)?;
        f.read_exact(&mut x1_buf)
            .map_err(|_| PtauError::InvalidG2Point)?;
        f.read_exact(&mut y0_buf)
            .map_err(|_| PtauError::InvalidG2Point)?;
        f.read_exact(&mut y1_buf)
            .map_err(|_| PtauError::InvalidG2Point)?;

        let x0 = deserialize_field(&mut &x0_buf[..]).map_err(|_| PtauError::InvalidG2Point)?;
        let x1 = deserialize_field(&mut &x1_buf[..]).map_err(|_| PtauError::InvalidG2Point)?;
        let y0 = deserialize_field(&mut &y0_buf[..]).map_err(|_| PtauError::InvalidG2Point)?;
        let y1 = deserialize_field(&mut &y1_buf[..]).map_err(|_| PtauError::InvalidG2Point)?;

        let x = Fp2::<ark_bn254::Fq2Config>::new(x0, x1);
        let y = Fp2::<ark_bn254::Fq2Config>::new(y0, y1);
        // Construct G2 point, ensuring it is on the curve.
        let g2 = Affine::<PTwo>::new(x, y);
        g2_points.push(g2);
    }

    Ok((g1_points, g2_points))
}
