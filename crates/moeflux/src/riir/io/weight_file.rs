//! Weight-file loader — Rust port of `infer.m`'s `WeightFile` +
//! `TensorManifest`.
//!
//! `model_weights.bin` is mmap'd; `model_weights.json` is parsed
//! into a per-tensor offset/shape/dtype index. Tensors are looked up
//! by name, returning a borrowed slice into the mmap. The slice's
//! lifetime is tied to [`WeightFile`]'s — drop the file and your
//! tensor views go away.
//!
//! ## Schema
//!
//! ```json
//! {
//!   "tensors": {
//!     "model.layers.0.self_attn.q_proj.weight": {
//!       "offset": 0,
//!       "size": 8388608,
//!       "shape": [4096, 4096],
//!       "dtype": "U32",
//!       "bits": 4
//!     },
//!     ...
//!   }
//! }
//! ```
//!
//! `bits` is only emitted for `U32`-packed quantized tensors by
//! `extract_weights.py >= 2026-04-26`. We default it to 4 for older
//! manifests so A17B / pre-bits extractions keep working — matches
//! the C path's behavior.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::{Mmap, MmapMut};
use serde::Deserialize;

/// Errors from the weight-file loader.
#[derive(Debug, thiserror::Error)]
pub enum WeightFileError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("manifest missing 'tensors' key")]
    MissingTensors,
    #[error("tensor '{name}' span [{offset}, {end}) extends past mmap'd file ({file_size} bytes)")]
    OutOfBounds {
        name: String,
        offset: u64,
        end: u64,
        file_size: u64,
    },
}

/// One tensor's metadata. Name lives in the index's [`HashMap`] key;
/// the index returns `(&str, &TensorInfo)` pairs.
///
/// `dtype` mirrors the C `dtype[8]` field — typical values are
/// `"U32"` (4-bit or 8-bit packed weights), `"BF16"` (scales /
/// biases), and `"F32"`. Use [`TensorInfo::dtype`] as a string-match
/// site rather than parsing into an enum: the C side does the same
/// and new dtypes can show up via the conversion script without a
/// crate-side change.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub offset: u64,
    pub size: u64,
    pub shape: Vec<usize>,
    pub dtype: String,
    /// Quantization bits for `U32`-packed tensors (typically 4 or
    /// 8). 0 means "not quantized" / non-`U32` dtype.
    pub bits: i32,
}

/// mmap'd weight blob plus tensor index. Construct via
/// [`WeightFile::open`]; tensor data via [`WeightFile::tensor_bytes`]
/// (slice view into the mmap, lives as long as `&self`).
pub struct WeightFile {
    backing: Backing,
    /// Bytes of weight data (the file length). The backing may be
    /// longer (owned copies are rounded up to a 16 KiB page).
    len: usize,
    tensors: HashMap<String, TensorInfo>,
}

/// How [`WeightFile`] holds the blob.
enum Backing {
    /// File-backed mmap: pages live in the OS page cache.
    Mmap(Mmap),
    /// Anonymous, page-aligned copy read with `F_NOCACHE`: counted in
    /// the process footprint, never in the page cache. Used under a
    /// hard memory cap.
    Owned(MmapMut),
}

impl Backing {
    fn bytes(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => m,
            Backing::Owned(m) => m,
        }
    }
}

/// How [`WeightFile::open_with`] loads the blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLoad {
    /// mmap the file (historical default; page-cache backed).
    Mmap,
    /// Read the whole file with `F_NOCACHE` into an anonymous,
    /// 16 KiB-page-aligned allocation.
    ReadNoCache,
}

/// Apple Silicon VM page size; also the alignment
/// `newBufferWithBytesNoCopy` requires.
const PAGE: usize = 16384;

/// Read all of `file` into a fresh anonymous mapping with the page
/// cache disabled for this fd.
fn read_nocache(file: &File, len: usize) -> Result<MmapMut, std::io::Error> {
    use std::os::unix::fs::FileExt;
    set_nocache(file)?;
    let mut buf = MmapMut::map_anon(len.div_ceil(PAGE).max(1) * PAGE)?;
    const CHUNK: usize = 64 << 20;
    let mut off = 0;
    while off < len {
        let end = (off + CHUNK).min(len);
        let n = file.read_at(&mut buf[off..end], off as u64)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("weights file ended at {off} of {len} bytes"),
            ));
        }
        off += n;
    }
    Ok(buf)
}

/// `fcntl(fd, F_NOCACHE, 1)`: reads on this fd bypass the unified
/// buffer cache, so loading model files doesn't grow the page cache.
pub fn set_nocache(file: &File) -> Result<(), std::io::Error> {
    use std::os::unix::io::AsRawFd;
    // SAFETY: valid owned fd for the duration of the call.
    let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1i32) };
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

impl WeightFile {
    /// Open a weight blob and its sibling manifest. The two paths
    /// are typically `<artifacts>/model_weights.bin` and
    /// `<artifacts>/model_weights.json` (matching the C `Ctx::open`
    /// argument order).
    pub fn open(bin_path: &Path, manifest_path: &Path) -> Result<Self, WeightFileError> {
        Self::open_with(bin_path, manifest_path, WeightLoad::Mmap)
    }

    /// [`Self::open`] with an explicit load strategy.
    pub fn open_with(
        bin_path: &Path,
        manifest_path: &Path,
        load: WeightLoad,
    ) -> Result<Self, WeightFileError> {
        let file = File::open(bin_path)?;
        let len = file.metadata()?.len() as usize;
        let backing = match load {
            // SAFETY: mmap of a regular file we just opened. The file
            // descriptor is closed when `file` drops (after Mmap takes
            // ownership of the mapping). On Linux/macOS a `MAP_PRIVATE`
            // mmap stays valid until munmap regardless of fd state.
            WeightLoad::Mmap => Backing::Mmap(unsafe { Mmap::map(&file)? }),
            WeightLoad::ReadNoCache => Backing::Owned(read_nocache(&file, len)?),
        };
        let tensors = parse_manifest(manifest_path)?;
        let wf = WeightFile { backing, len, tensors };
        wf.bounds_check_all()?;
        eprintln!(
            "[weights] {} {:.2} GB from {}",
            match load {
                WeightLoad::Mmap => "mmap'd",
                WeightLoad::ReadNoCache => "read (F_NOCACHE)",
            },
            wf.len as f64 / 1e9,
            bin_path.display()
        );
        eprintln!(
            "[manifest] Loaded {} tensors from {}",
            wf.tensors.len(),
            manifest_path.display()
        );
        Ok(wf)
    }

    /// Total bytes in the mmap'd file.
    pub fn file_size(&self) -> usize {
        self.len
    }

    /// Number of tensors in the manifest.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Tensor info by name, or `None` if missing. Cheap O(1) hash
    /// lookup.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Borrowed byte slice of tensor `name`'s data, or `None` if the
    /// tensor isn't in the manifest. The slice is exactly
    /// `info.size` bytes long, starting at `info.offset` in the
    /// mmap.
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensor_info(name)?;
        let start = info.offset as usize;
        let end = start + info.size as usize;
        Some(&self.backing.bytes()[start..end])
    }

    /// Borrowed byte slice `[offset, offset + len)` of the mmap'd
    /// weight file. Used by the graph compiler's CPU backend to
    /// resolve [`crate::riir::graph::WeightRef`] offsets — the
    /// producer code reads tensor offsets from
    /// [`crate::riir::layer_weight_cache::LayerWeightCache`] and the
    /// CPU encoder slices the mmap directly without going through
    /// the tensor-name map. Returns `None` if the range falls
    /// outside the mmap.
    pub fn bytes_at(&self, offset: u64, len: usize) -> Option<&[u8]> {
        let start = offset as usize;
        let end = start.checked_add(len)?;
        if end > self.len {
            return None;
        }
        Some(&self.backing.bytes()[start..end])
    }

    /// Iterator over `(name, info)` pairs. Order is HashMap-defined
    /// (i.e., unspecified). Useful for debugging / dumping.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.tensors.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Verify every tensor's `[offset, offset + size)` span fits
    /// inside the mmap. Run once at load time so later
    /// [`Self::tensor_bytes`] calls can skip per-call bounds checks
    /// — a typical forward pass walks ~1400 tensors per token, the
    /// per-call branch matters in the hot loop.
    fn bounds_check_all(&self) -> Result<(), WeightFileError> {
        let file_size = self.len as u64;
        for (name, info) in &self.tensors {
            let end = info.offset.saturating_add(info.size);
            if end > file_size {
                return Err(WeightFileError::OutOfBounds {
                    name: name.clone(),
                    offset: info.offset,
                    end,
                    file_size,
                });
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for WeightFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightFile")
            .field("file_size", &self.len)
            .field("num_tensors", &self.tensors.len())
            .finish()
    }
}

// --- Manifest parser ---------------------------------------------

#[derive(Debug, Deserialize)]
struct RawManifest {
    tensors: HashMap<String, RawTensor>,
}

/// Mirrors the per-tensor shape `extract_weights.py` emits. `bits`
/// is optional (older manifests don't carry it) — see module docs.
#[derive(Debug, Deserialize)]
struct RawTensor {
    offset: u64,
    size: u64,
    shape: Vec<usize>,
    dtype: String,
    #[serde(default)]
    bits: Option<i32>,
}

fn parse_manifest(path: &Path) -> Result<HashMap<String, TensorInfo>, WeightFileError> {
    let bytes = std::fs::read(path)?;
    let raw: RawManifest = serde_json::from_slice(&bytes)?;
    let mut out = HashMap::with_capacity(raw.tensors.len());
    for (name, t) in raw.tensors {
        // Default bits=4 for U32-packed tensors that lack the
        // explicit field (older manifests). Non-U32 tensors are
        // unquantized — bits=0.
        let bits = t
            .bits
            .unwrap_or_else(|| if t.dtype == "U32" { 4 } else { 0 });
        out.insert(
            name,
            TensorInfo {
                offset: t.offset,
                size: t.size,
                shape: t.shape,
                dtype: t.dtype,
                bits,
            },
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ReadNoCache` returns byte-identical tensors to `Mmap`, with a
    /// file length that isn't a page multiple (exercises the padded
    /// tail of the owned copy).
    #[test]
    fn read_nocache_matches_mmap() {
        let dir = std::env::temp_dir().join(format!("moeflux-wf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("w.bin");
        let manifest = dir.join("w.json");
        let len = PAGE * 3 + 123;
        let data: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::write(&bin, &data).unwrap();
        std::fs::write(
            &manifest,
            format!(
                r#"{{"tensors": {{
                    "a": {{"offset": 0, "size": 4096, "shape": [4096], "dtype": "U8"}},
                    "tail": {{"offset": {}, "size": 123, "shape": [123], "dtype": "U8"}}
                }}}}"#,
                PAGE * 3
            ),
        )
        .unwrap();
        let m = WeightFile::open_with(&bin, &manifest, WeightLoad::Mmap).unwrap();
        let o = WeightFile::open_with(&bin, &manifest, WeightLoad::ReadNoCache).unwrap();
        assert_eq!(m.file_size(), len);
        assert_eq!(o.file_size(), len);
        for name in ["a", "tail"] {
            assert_eq!(m.tensor_bytes(name), o.tensor_bytes(name), "{name}");
        }
        assert_eq!(o.tensor_bytes("tail").unwrap(), &data[PAGE * 3..]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Manifest parser handles the documented schema, including
    /// older manifests without `bits`. Doesn't need a real model.
    #[test]
    fn parse_manifest_with_and_without_bits() {
        let json = br#"{
            "tensors": {
                "weights_a": {
                    "offset": 0,
                    "size": 1024,
                    "shape": [16, 16],
                    "dtype": "U32",
                    "bits": 4
                },
                "weights_b_old": {
                    "offset": 1024,
                    "size": 2048,
                    "shape": [32, 16],
                    "dtype": "U32"
                },
                "scales": {
                    "offset": 3072,
                    "size": 64,
                    "shape": [32],
                    "dtype": "BF16"
                }
            }
        }"#;
        let raw: RawManifest = serde_json::from_slice(json).unwrap();
        assert_eq!(raw.tensors.len(), 3);
        let a = &raw.tensors["weights_a"];
        assert_eq!(a.bits, Some(4));
        assert_eq!(a.dtype, "U32");
        let b = &raw.tensors["weights_b_old"];
        assert_eq!(b.bits, None);
        assert_eq!(b.dtype, "U32");
    }
}
