# Changelog

## Unreleased

### Added

- `OpenConfig::max_checkpoints` (default 4): checkpoint buffers planned
  under a cap (`Account::Checkpoints`).

### Fixed

- Checkpoints allocated after open under a cap; now preallocated,
  touched at open and reused, with `restore_to` loading in place.
- Whole-model weights counted at raw length instead of whole pages.
- Parked request slots' host linear-attn state not touched at open.
- Intermittent `footprint_is_readable_and_nonzero` failure.

### Changed

- Under a cap, checkpoint buffers take cap memory at open (set
  `max_checkpoints: 0` if unused); `set_max_checkpoints` cannot raise
  the count past the opened value; `checkpoint_pos` can return
  `StateSnapshotError::CheckpointsFull`. `moeflux-node` plans none.

## 0.2.0-alpha.1 — 2026-10-26

First alpha of the fork revamp (`docs/fork-upgrade-plan.md`), on top
of moeflux `0.1.0-pre.4`. The version is `0.2.0` because, in semver,
`0.1.0-alpha.*` would sort before `0.1.0-pre.4`. APIs may still
change. Known gaps and precautions: [`docs/known-issues.md`](docs/known-issues.md).

### Added

- **Hard memory cap** (phase 1): `RsCtx::open_with(OpenConfig { mem_cap,
  n_ctx, … })`. A budget plan per account, reserve-and-touch at open,
  and footprint checks (current and lifetime peak); `open` fails with
  `Error::OverBudget` before anything else is affected. `open()` is
  unchanged for drama_llama.
- **`F_NOCACHE` streaming and a fixed hot tier** (phase 2):
  page-cache-free reads into aligned buffers, a frequency-based expert
  hot tier sized from what the cap leaves, routing-profile seeding,
  expert-I/O counters.
- **Layer-range execution** (phase 3): `layer_range`, `eval_stage`,
  f32 hidden in/out, per-range snapshots (v3).
- **Per-request state** (phase 4): `max_requests` pool,
  `admit_request` / `drop_request` / `eval_request`.
- **`moeflux-net`** (phase 5): frames, `moeflux-node`,
  `moeflux-coord` and a layer-range planner (TCP).
- **Streaming improvements** (phase 6): `ExpertIoStats` telemetry and
  CSV, bounded I/O (`io_queue_depth`, `io_bytes_per_sec`), coalesced
  prefill reads, cross-layer and lookahead prefetch,
  `fast_experts_dir` disk tier, `prefetch_experts` (default 4 under a
  cap, +4–15% tok/s on OLMoE).
- **OLMoE-1B-7B variant** (`model-olmoe-1b-7b`): converter, NumPy
  reference forward, and model tests under a cap — the test bed on
  hosts without a3b weights.
- **KV cache formats** (`kv_format`): F32 (default), F16, TurboQuant
  4/3-bit; snapshot v4. OLMoE, 3 GiB cap, n_ctx 4096: KV 1.00 → 0.56 /
  0.19 GiB (F16 / Turbo4), decode +29% / +44%.
- **Decode attention threads** (`attn_threads`, default 4):
  bit-identical; +55–58% tok/s at 2.7k context.
- **Tooling**: `scripts/check_test_host.sh`, `scripts/watch_run.sh`,
  `MOEFLUX_ROUTING_TRACE`, `tools/sim_expert_cache.py`.

### Fixed

- Prefill scratch sized for 8192 tokens outside the budget
  (`prefill_chunk`, `Account::PrefillScratch`).
- Allocations after open under a cap: pool re-coloring, first-submit
  driver state, untouched slot ring.
- `moeflux-net` session reset, rollback and input checks.
- `extract_weights.py` on Qwen3-MoE configs.

### Known gaps

- Checkpoints, whole-model weight rounding and parked request slots
  are not fully inside the cap plan (see `docs/known-issues.md`).
- a3b / a17b model-level tests have not run yet; verified on
  OLMoE-1B-7B plus unit / kernel tests on all four variants.
