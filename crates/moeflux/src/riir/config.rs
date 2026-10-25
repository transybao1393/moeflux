//! Runtime open-time configuration — [`OpenConfig`].
//!
//! Model *shape* stays compile-time ([`crate::riir::variants`]);
//! everything a deployment needs to tune per machine lives here:
//! context length (which sizes every KV cache), the expert-IO path,
//! and the IO thread count. [`crate::riir::RsCtx::open`] is
//! [`crate::riir::RsCtx::open_with`] with [`OpenConfig::default`],
//! so existing callers see no behavior change.

use crate::riir::budget::DEFAULT_RUNTIME_OVERHEAD;
use crate::riir::io::expert_io_mode::ExpertIoMode;
use crate::riir::variants::MAX_SEQ_LEN;

/// Default IO worker-thread count — the historical hard-coded value.
pub const DEFAULT_IO_THREADS: usize = 8;

/// Open-time knobs for [`crate::riir::RsCtx::open_with`].
///
/// Construct with [`OpenConfig::default`] and override fields:
///
/// ```ignore
/// let cfg = OpenConfig { n_ctx: 32_768, ..OpenConfig::default() };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenConfig {
    /// Maximum positions per sequence. Sizes every full-attention KV
    /// cache (`n_ctx × num_kv_heads × head_dim × 4 B` per side per
    /// layer). Must be in `1..=MAX_SEQ_LEN`. The default,
    /// `MAX_SEQ_LEN`, keeps the historical lazily-committed 1M-row
    /// reservation; set it to what you actually need.
    pub n_ctx: usize,
    /// Expert-IO path. `None` defers to `MOEFLUX_EXPERT_IO`, then to
    /// the RAM gate in [`crate::riir::io::expert_io_mode::select`].
    /// `Some(_)` wins over the env var.
    pub expert_io: Option<ExpertIoMode>,
    /// Worker threads for parallel expert `pread` and prefetch.
    /// Must be ≥ 1.
    pub io_threads: usize,
    /// Hard memory cap in bytes. `None` (default) = no cap, historical
    /// behavior. `Some(cap)`:
    /// - expert IO is forced to [`ExpertIoMode::Pread`]; asking for
    ///   `Mmap` is a config error (mmap'd pages escape accounting);
    /// - `open_with` fails if the [`crate::riir::budget::BudgetPlan`]
    ///   exceeds the cap;
    /// - every resource is built and the KV caches touched at open, then
    ///   the measured footprint is checked against the cap;
    /// - each `eval_*` call re-checks the footprint.
    pub mem_cap: Option<u64>,
    /// Bytes reserved inside `mem_cap` for Metal/driver/runtime
    /// overhead and scratch. Only used when `mem_cap` is set.
    pub runtime_overhead: u64,
    /// Size of the RAM hot tier for experts
    /// ([`crate::riir::io::hot_tier`]). `None`: under a cap, everything
    /// left after the budget plan; without a cap, no hot tier.
    /// `Some(0)` disables it. Requires `Pread` expert IO.
    pub hot_tier_bytes: Option<u64>,
    /// Routing profile to seed the hot tier from at open (written by
    /// [`crate::riir::RsCtx::save_routing_profile`]). Most-picked
    /// experts are loaded first.
    pub hot_tier_profile: Option<std::path::PathBuf>,
    /// Contiguous layer range this context runs (plan phase 3, mode B).
    /// `None` (default) = the whole model. `Some(a..b)` loads, plans and
    /// runs only layers `a..b`; the embedding exists only when `a == 0`
    /// and the final norm + LM head only when `b == num_layers`. Drive a
    /// partial range with [`crate::riir::RsCtx::eval_stage`]; the
    /// whole-model `eval_prompt` / `eval_token` need the full range.
    pub layer_range: Option<std::ops::Range<usize>>,
}

impl Default for OpenConfig {
    fn default() -> Self {
        Self {
            n_ctx: MAX_SEQ_LEN,
            expert_io: None,
            io_threads: DEFAULT_IO_THREADS,
            mem_cap: None,
            runtime_overhead: DEFAULT_RUNTIME_OVERHEAD,
            hot_tier_bytes: None,
            hot_tier_profile: None,
            layer_range: None,
        }
    }
}

impl OpenConfig {
    /// Check field ranges. Returns a human-readable reason on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.n_ctx == 0 || self.n_ctx > MAX_SEQ_LEN {
            return Err(format!(
                "n_ctx {} out of range 1..={MAX_SEQ_LEN}",
                self.n_ctx
            ));
        }
        if self.io_threads == 0 {
            return Err("io_threads must be ≥ 1".into());
        }
        if self.hot_tier_bytes.is_some_and(|b| b > 0) && self.expert_io == Some(ExpertIoMode::Mmap)
        {
            return Err("hot_tier_bytes needs expert_io = Pread (mmap mode binds the files)".into());
        }
        if self.mem_cap.is_some() && self.expert_io == Some(ExpertIoMode::Mmap) {
            return Err(
                "expert_io = Mmap is not allowed with mem_cap: mmap'd expert \
                 pages are file-backed and escape the cap"
                    .into(),
            );
        }
        if let Some(r) = &self.layer_range {
            crate::riir::range::validate(r)?;
        }
        Ok(())
    }

    /// The layer range to run: [`Self::layer_range`] or the whole model.
    pub fn resolved_layer_range(&self) -> std::ops::Range<usize> {
        self.layer_range
            .clone()
            .unwrap_or_else(crate::riir::range::full_range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_historical_behavior() {
        let c = OpenConfig::default();
        assert_eq!(c.n_ctx, MAX_SEQ_LEN);
        assert_eq!(c.expert_io, None);
        assert_eq!(c.io_threads, DEFAULT_IO_THREADS);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_rejects_out_of_range() {
        let bad_ctx = OpenConfig { n_ctx: 0, ..OpenConfig::default() };
        assert!(bad_ctx.validate().is_err());
        let too_big = OpenConfig { n_ctx: MAX_SEQ_LEN + 1, ..OpenConfig::default() };
        assert!(too_big.validate().is_err());
        let no_threads = OpenConfig { io_threads: 0, ..OpenConfig::default() };
        assert!(no_threads.validate().is_err());
    }

    #[test]
    fn mmap_rejected_under_cap() {
        let c = OpenConfig {
            mem_cap: Some(10 << 30),
            expert_io: Some(ExpertIoMode::Mmap),
            ..OpenConfig::default()
        };
        assert!(c.validate().is_err());
        let ok = OpenConfig { expert_io: Some(ExpertIoMode::Pread), ..c };
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn layer_range_validated() {
        let n = crate::riir::variants::VARIANT.num_layers;
        let c = OpenConfig { layer_range: Some(0..n), ..OpenConfig::default() };
        assert!(c.validate().is_ok());
        assert_eq!(OpenConfig::default().resolved_layer_range(), 0..n);
        let empty = OpenConfig { layer_range: Some(3..3), ..OpenConfig::default() };
        assert!(empty.validate().is_err());
        let past = OpenConfig { layer_range: Some(1..n + 1), ..OpenConfig::default() };
        assert!(past.validate().is_err());
    }
}
