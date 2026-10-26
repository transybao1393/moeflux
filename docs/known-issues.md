# Known issues

What has been fixed, what is still open, and what to watch out for in
the fork revamp (`docs/fork-upgrade-plan.md`). Add an entry when a bug
is fixed or a trap is found; move open items to *Fixed* when they land.
Older engine history (graph mode, prefill, kernels) is in
`.claude/memory/`.

## Open

| Issue | Impact | Plan |
| --- | --- | --- |
| **Checkpoints are outside the memory budget.** `checkpoint_pos` allocates a `state_size()` buffer per checkpoint (up to `max_checkpoints`) at runtime, and `restore_to` clones one more. | Under `mem_cap` this grows after open; only the after-eval footprint check catches it, once the memory is already used. A full F32 state is ~1 GiB on OLMoE at n_ctx 4096. | `Account::Checkpoints` = `max_checkpoints × state_size(n_ctx)` + 1, allocated and touched at open, reused as a ring; drama_llama's API unchanged. A quantized `kv_format` shrinks each checkpoint too. |
| **Whole-model weights counted at raw length**, while the owned copy is rounded up to a 16 KiB page. | Under-counts by < 16 KiB. | Round like `filtered_bytes` (the range path) already does. |
| **Parked request slots' host `LinearAttnState`** is planned but not touched at open. | The open-time footprint check doesn't see it (the cap still holds; it is reserved). | Touch it at open with the rest. |
| `resuming_prefill_after_seq_rm_matches_full_prefill` fails. | Known: linear-attn recurrence state is not position-truncatable (a partial truncate resets the layer). | `--skip` it; see the fn-level comment and `snapshot/state.rs`. |
| a3b / a17b model-level tests have not run on this host (no a3b weights). | Phases 1–6 and `kv_format` are verified on OLMoE-1B-7B only, plus unit / kernel tests on all four variants. | Run the `--include-ignored` gate on a host with the a3b model. |
| GPU time is not measured directly. | metal-rs exposes no GPU timestamps (`CmdbufStat::gpu_ns` stays 0); only wall time and I/O wait. | — |

## Fixed

### Memory cap

- **Prefill scratch outside the plan** (`bfd4070`). Every batched-prefill
  buffer was sized for 8192 tokens whatever `n_ctx` or the cap
  (OLMoE 2.6 GiB, a3b 3.6 GiB), with no account: a 3 GiB cap failed at
  open. Now `OpenConfig::prefill_chunk` (512 under a cap) sizes it and
  `Account::PrefillScratch` plans it exactly.
- **Expert slot ring never touched at open** (`bfd4070`) —
  heap-backed, so the open-time check missed it.
  `MetalBufferPool::touch_aligned`.
- **`commit_plan` re-colored the pool on the first graph** (+48 MB after
  open; `bfd4070`, `6f8cc46`). Under a cap the pool layout is frozen at
  open.
- **First GPU submit allocated ~80 MB of driver state after open**
  (`6f8cc46`). A warm submit now runs at open.
- **Tripwire only compared the current footprint** (`6f8cc46`), so a
  transient peak over the cap went unnoticed. `check_cap` also checks
  the lifetime-max footprint.
- **`DEFAULT_RUNTIME_OVERHEAD` 1.5 GiB → 512 MiB** (`bfd4070`): ~90 MiB
  is actually unitemized at peak on OLMoE.
- **`kv_format` scratch** (`ff6619d`): a quantized KV cache needs an f32
  prefill scratch; it has its own account (`kv_scratch`) and is touched
  at open.

### Correctness

- **`extract_weights.py` broke on Qwen3-MoE configs** (`7466141`): no
  linear-attn / shared-expert keys and no `rope_parameters`; now
  defaulted.
- **Metal kernel signature** (`ff6619d`, caught by the CPU-vs-Metal
  test): mixing `uint2` and `uint` thread-position inputs fails to
  compile at runtime. Use the same vector type for both.

### Speed

- **Prefetch read all K guesses** (`b9587b8`): 42% of decode reads were
  wasted on OLMoE at a 6 GiB cap. `prefetch_experts` = 4 under a cap.
- **F16 KV decode was slower than F32** (`ff6619d`): the f16 → f32
  conversion did not vectorize (19 ms vs 4.5 ms per layer). It now uses
  the hardware `fcvtl`; bit-exact over all 65,536 halves.
- **Decode attention ran on one core** (uncommitted): at 2.7k context it
  was ~half of decode time. `OpenConfig::attn_threads` (default 4)
  splits heads across a pool, bit-identical: needle decode
  303.6 → 192.6 ms/tok (F32), 205.8 → 132.8 (Turbo4).

## Precautions

### The shared server

- Other services run on this machine and must not be affected: run
  `scripts/check_test_host.sh` before every test phase, run model tests
  one process at a time (`--test-threads=1`), and keep caps within the
  reclaimable memory it reports. `scripts/watch_run.sh` samples memory,
  swap and pressure during a run.
- Host load changes timings a lot (load 12 vs 2 changed decode by
  > 30%). Compare speeds only within one batch of runs; rerun the
  baseline with every comparison.
- `attn_threads` and `io_threads` use CPU cores other services share;
  lower them if that matters more than speed.

### Configuration rules

- `expert_io = Mmap` is rejected with `mem_cap` (file-backed pages
  escape the cap); a hot tier needs `Pread`.
- `prefetch_lookahead` needs `prefetch_guess = PrevToken`.
- `kv_format`: MLA (Cogito) accepts only F32; Turbo needs a
  power-of-two `head_dim`. Non-F32 formats always use the CPU decode
  attention (the GPU SDPA fast path is F32-only).
- Snapshots are not converted between KV formats: loading a snapshot
  into a context with another `kv_format` fails with
  `ShapeMismatch { field: "kv_format" }`. F32 snapshots stay v2 / v3.

### Expected behavior, not bugs

- **Turbo KV changes greedy output wording early** (3–19 tokens on
  OLMoE) while answers stay correct and coherent; F16 matches F32 for
  hundreds of tokens. Logits cosine vs F32: F16 0.999999, Turbo4
  0.9928, Turbo3 0.9780.
- **Turbo CPU and Metal quantizers are not bit-identical**: their norm
  reductions sum in a different order, so a code at a level boundary
  can flip (reconstructions agree to ~1e-13 relative). Batched prefill
  vs per-token decode agree at cosine 0.9975 (Turbo4), not 1.0.
- **Turbo3 is slower than Turbo4**: its 3-bit unpack costs more than
  the 0.04 GiB of extra hot tier saves. Prefer Turbo4.

### Measuring

- **macOS `F_NOCACHE` only bypasses the page cache for page-aligned
  buffers with `F_RDAHEAD` off.** Python reads into unaligned buffers
  still fill the cache. Benchmark SSD reads into an `mmap` buffer with
  readahead off.
- **Files read recently are still in the page cache**: a "disk"
  benchmark on them measures RAM. Use a fresh cold copy, check
  residency, delete it afterwards.
- The SSD tops out at ~6.7 GB/s once ≥ 4 reads are in flight; decode is
  bandwidth-bound, so fewer bytes per token is what speeds it up.

### Tests and tooling

- `graph_diff_oracle.rs` and `checkpoint_restore.rs` compile only with
  `model-qwen3-6-35b-a3b`; the graph-diff kernel tests need no model
  files, so run them under a3b even on a host without a3b weights.
- `checkpoint_restore.rs` fails under a17b — expected.
- `cargo fmt` is not a gate (the crate is ~600 hunks off rustfmt); match
  the surrounding style, don't bulk-reformat.
- zsh does not word-split variables (`$G commit` fails): export
  `GIT_AUTHOR_*` / `GIT_COMMITTER_*` instead. `grep --include=*.rs`
  errors as an unmatched glob; quote it or drop it.

### Measured dead ends — don't retry without new evidence

- Predicting the next experts from a moving router score or recency:
  ties the previous-token guess (±1 pt) or loses.
- LRU or short-retention hot tiers: LFU wins.
- A 64-slot recent-read window: 0 hits in the engine (~145 slot fills
  per token).
- Re-laying-out experts on disk for coalesced reads: no gain (the SSD
  is flat once ≥ 4 reads are in flight).
- More from §8 of the plan: soft page-cache allowance, growing the hot
  tier when idle, lazily committed KV, LZ4 / LZFSE compression,
  per-expert files, all-2-bit experts.
