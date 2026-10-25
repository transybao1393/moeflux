//! Hard memory cap — open-time plan, footprint measurement, checks.
//!
//! A context opened with [`crate::riir::OpenConfig::mem_cap`] set must
//! never push the process past the cap. Enforcement has three layers:
//!
//! 1. **Plan** ([`BudgetPlan::for_config`]) — before anything is
//!    allocated, add up every fixed account from the variant shape and
//!    the config. A plan over the cap fails `open`.
//! 2. **Measure at open** — under a cap, `open_with` builds every
//!    resource eagerly and touches the KV caches, then samples
//!    [`footprint`]. Over the cap fails `open`.
//! 3. **Tripwire** — each `eval_*` call re-samples the footprint and
//!    returns [`crate::riir::RsError::OverBudget`] if it crossed the
//!    cap (an accounting bug, by construction).
//!
//! `phys_footprint` counts anonymous and GPU (IOAccelerator) memory
//! but not clean file-backed pages, so mmap'd model files sit outside
//! it. Under a cap the engine reads experts with `pread` (never
//! mmap-direct); `F_NOCACHE` reads that keep them out of the page
//! cache are part of the same plan (`docs/fork-upgrade-plan.md`).

use crate::riir::io::expert_io_mode::ExpertIoMode;
use crate::riir::moe::expert_forward::MAX_K;
use crate::riir::variants::{AttnKind, LayerKind, VARIANT, Variant};

const GIB: u64 = 1 << 30;
const TWO_MIB: u64 = 2 << 20;

/// Default reservation for Metal/driver/shader overhead, thread
/// stacks, the heap and per-op scratch buffers. Replace with a
/// measured value per machine (Phase 0).
pub const DEFAULT_RUNTIME_OVERHEAD: u64 = GIB + GIB / 2;

/// One named slice of the cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Account {
    /// Non-expert weights (`model_weights.bin`).
    Weights,
    /// Full-attention KV caches at `n_ctx`.
    Kv,
    /// Linear-attention conv + recurrence state (host + GPU copies).
    LinearState,
    /// Expert slot ring: synced + two prefetch sets, 2 MiB-aligned.
    ExpertSlots,
    /// Prefill expert staging (pread mode only).
    PrefillStaging,
    /// Metal/driver/runtime overhead and scratch.
    RuntimeOverhead,
}

impl Account {
    /// Stable lowercase name for logs and errors.
    pub fn name(self) -> &'static str {
        match self {
            Account::Weights => "weights",
            Account::Kv => "kv",
            Account::LinearState => "linear_state",
            Account::ExpertSlots => "expert_slots",
            Account::PrefillStaging => "prefill_staging",
            Account::RuntimeOverhead => "runtime_overhead",
        }
    }
}

/// Byte totals per [`Account`] for one context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetPlan {
    /// `(account, bytes)` in a fixed order.
    pub entries: Vec<(Account, u64)>,
}

fn round_up(x: u64, to: u64) -> u64 {
    x.div_ceil(to) * to
}

/// KV bytes per position summed over every full-attn layer.
pub fn kv_bytes_per_token(v: Variant) -> u64 {
    let f32b = std::mem::size_of::<f32>() as u64;
    let full_layers = (0..v.num_layers)
        .filter(|&i| v.layer_kind(i) == LayerKind::FullAttn)
        .count() as u64;
    let per_layer = match v.attn_kind {
        AttnKind::Gqa => 2 * (v.num_kv_heads * v.head_dim) as u64 * f32b,
        AttnKind::Mla => (v.kv_lora_rank + v.qk_rope_head_dim) as u64 * f32b,
    };
    full_layers * per_layer
}

/// Linear-attn state bytes (host `LinearAttnState` + GPU mirror).
pub fn linear_state_bytes(v: Variant) -> u64 {
    let f32b = std::mem::size_of::<f32>() as u64;
    let linear_layers = (0..v.num_layers)
        .filter(|&i| v.layer_kind(i) == LayerKind::LinearAttn)
        .count() as u64;
    let conv = ((Variant::CONV_KERNEL_SIZE - 1) * v.linear_conv_dim()) as u64;
    let ssm = (v.linear_num_v_heads * Variant::LINEAR_VALUE_DIM * Variant::LINEAR_KEY_DIM) as u64;
    2 * linear_layers * (conv + ssm) * f32b
}

impl BudgetPlan {
    /// Plan for the active variant.
    ///
    /// `weights_bytes` is the size of `model_weights.bin`;
    /// `runtime_overhead` is [`crate::riir::OpenConfig::runtime_overhead`].
    pub fn for_config(
        n_ctx: usize,
        mode: ExpertIoMode,
        weights_bytes: u64,
        runtime_overhead: u64,
    ) -> Self {
        let v = VARIANT;
        let expert = v.expert_size_4bit() as u64;
        // `MoeBuffers::new`: MAX_K synced + 2 × MAX_K prefetch slots.
        let slots = 3 * MAX_K as u64 * round_up(expert, TWO_MIB);
        let staging = if mode.is_pread() {
            v.num_experts as u64 * expert
        } else {
            0
        };
        Self {
            entries: vec![
                (Account::Weights, weights_bytes),
                (Account::Kv, kv_bytes_per_token(v) * n_ctx as u64),
                (Account::LinearState, linear_state_bytes(v)),
                (Account::ExpertSlots, slots),
                (Account::PrefillStaging, staging),
                (Account::RuntimeOverhead, runtime_overhead),
            ],
        }
    }

    /// Sum of every account.
    pub fn total(&self) -> u64 {
        self.entries.iter().map(|&(_, b)| b).sum()
    }

    /// Bytes for one account (0 if absent).
    pub fn get(&self, account: Account) -> u64 {
        self.entries
            .iter()
            .find(|&&(a, _)| a == account)
            .map_or(0, |&(_, b)| b)
    }

    /// Fail if the plan exceeds `cap`, naming every account.
    pub fn check(&self, cap: u64) -> Result<(), String> {
        let total = self.total();
        if total <= cap {
            return Ok(());
        }
        Err(format!(
            "planned {} exceeds cap {} ({})",
            fmt_gib(total),
            fmt_gib(cap),
            self.breakdown()
        ))
    }

    /// `name=1.23 GiB, …` for logs and errors.
    pub fn breakdown(&self) -> String {
        self.entries
            .iter()
            .map(|&(a, b)| format!("{}={}", a.name(), fmt_gib(b)))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// `1.23 GiB`.
pub fn fmt_gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / GIB as f64)
}

/// Process memory as the OS accounts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footprint {
    /// Current `phys_footprint` in bytes.
    pub current: u64,
    /// Lifetime peak `phys_footprint` in bytes.
    pub peak: u64,
}

/// Sample this process's `phys_footprint` (what Activity Monitor's
/// "Memory" column and jetsam use). `None` if the syscall fails.
pub fn footprint() -> Option<Footprint> {
    let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a valid, writable `rusage_info_v4`, matching
    // the `RUSAGE_INFO_V4` flavor; the kernel writes at most its size.
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            (&mut info as *mut libc::rusage_info_v4).cast(),
        )
    };
    if rc != 0 {
        return None;
    }
    Some(Footprint {
        current: info.ri_phys_footprint,
        peak: info.ri_lifetime_max_phys_footprint,
    })
}

/// Fail if the current footprint is over `cap`. A failed sample is
/// treated as a failure — under a hard cap, unknown is not safe.
pub fn check_footprint(cap: u64, when: &str) -> Result<Footprint, String> {
    let fp = footprint().ok_or_else(|| format!("{when}: could not read phys_footprint"))?;
    if fp.current > cap {
        return Err(format!(
            "{when}: footprint {} exceeds cap {} (peak {})",
            fmt_gib(fp.current),
            fmt_gib(cap),
            fmt_gib(fp.peak)
        ));
    }
    Ok(fp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_scales_with_n_ctx() {
        let a = BudgetPlan::for_config(1024, ExpertIoMode::Pread, 0, 0);
        let b = BudgetPlan::for_config(2048, ExpertIoMode::Pread, 0, 0);
        assert_eq!(b.get(Account::Kv), 2 * a.get(Account::Kv));
        assert_eq!(a.get(Account::Kv), 1024 * kv_bytes_per_token(VARIANT));
    }

    #[test]
    fn staging_only_in_pread_mode() {
        let p = BudgetPlan::for_config(1, ExpertIoMode::Pread, 0, 0);
        let m = BudgetPlan::for_config(1, ExpertIoMode::Mmap, 0, 0);
        assert!(p.get(Account::PrefillStaging) > 0);
        assert_eq!(m.get(Account::PrefillStaging), 0);
    }

    #[test]
    fn total_and_check() {
        let p = BudgetPlan::for_config(8, ExpertIoMode::Pread, 1000, 2000);
        assert_eq!(p.total(), p.entries.iter().map(|e| e.1).sum::<u64>());
        assert!(p.check(p.total()).is_ok());
        let err = p.check(p.total() - 1).unwrap_err();
        assert!(err.contains("weights=") && err.contains("kv="), "{err}");
    }

    /// a3b numbers from the plan doc: 40 KiB of KV per token.
    #[cfg(feature = "model-qwen3-6-35b-a3b")]
    #[test]
    fn a3b_kv_per_token_is_40_kib() {
        assert_eq!(kv_bytes_per_token(VARIANT), 40 * 1024);
    }

    #[test]
    fn footprint_is_readable_and_nonzero() {
        let fp = footprint().expect("proc_pid_rusage");
        assert!(fp.current > 0);
        assert!(fp.peak >= fp.current);
        assert!(check_footprint(u64::MAX, "test").is_ok());
        assert!(check_footprint(1, "test").is_err());
    }
}
