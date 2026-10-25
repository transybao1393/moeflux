# moeflux fork: upgrade plan

Oct 25, 2026 · companion to `moeflux fork proposed changes.md`

This plan turns the proposal into an ordered set of changes, corrected
against the moeflux 0.1.0-pre.4 source. Two principles drive it:

1. **The memory cap is a hard ceiling.** The server runs other things that
   must not be touched. Nothing the engine does may push total memory use
   past the configured cap (e.g. 10 GB) — including the OS page cache it
   causes, not just the memory it allocates.
2. **Streaming experts stays the core of the engine, and gets stronger.**
   Mode A (single node) streams experts inside the cap; mode B (pipeline)
   runs the same streaming code with a hot tier big enough to hold its whole
   layer range.

## 1. What the code says (corrections to the proposal)

| Proposal assumption | Actual (pre.4) | Source |
| --- | --- | --- |
| Model is ~17.5 GB at 4-bit | Experts alone are 16.9 GiB (18.1 GB): 1,769,472 B × 256 × 40, plus non-expert weights | `variants.rs` `expert_size_4bit` |
| Unknown full-attention interleave | Every 4th layer: 3, 7, …, 39. A 20+10+10 split holds 5 / 2 / 3 full-attn layers | `variants.rs:251`, `full_attn_interval: 4` |
| Unknown expert I/O path | Both: mmap (GPU reads files directly) or pread + prefetch, chosen at open | `io/expert_io_mode.rs` |
| Router is buried in the forward pass | Decode reads gate logits back to the CPU and runs `moe_router_cpu` — a clean hook point | `attn/linear_attn_forward.rs:1683`, `:1877`, prefill `~:2737` |
| 2-bit is a reference to borrow | `dequant_matvec_2bit` kernel exists; `open(_use_2bit)` is ignored; no 2-bit files or batched 2-bit kernel | `shaders.metal:762`, `mod.rs:441` |
| No telemetry | `cmdbuf_stats`, `prefetch_stats`, `gpu_capture`, `scripts/profile_smoke.sh` exist | `mod.rs`, `riir/gpu_capture.rs` |
| Hidden state as BF16 on the wire | Engine works in f32; BF16 would break exact network equivalence. Use f32 (8 KB/token) | — |
| Multi-request needs batching | Pipeline throughput needs per-request *state switching*, not batched decode | see Phase 4 |

## 2. What breaks a hard cap today

| Hidden growth | Where | Why it escapes the cap |
| --- | --- | --- |
| Expert reads fill the page cache | `io/expert_io.rs:432` (`file.read_at` on a normal fd) | Page cache is not in `phys_footprint` but takes RAM from other services |
| Every expert file is mmapped; mmap mode lets the GPU read it directly | `io/expert_io.rs:221` | File-backed pages grow as touched; up to 16.9 GiB |
| Non-expert weights are mmapped | `io/weight_file.rs:96` | Same: page cache, invisible to footprint |
| KV lazily committed for 1M tokens | `snapshot/state.rs:91`, `variants.rs:372` | 4 GiB virtual per full-attn layer; grows 40 KiB/token without limit |
| Automatic I/O mode check | `io/expert_io_mode.rs:110` | Compares experts to physical RAM, not the cap; picks mmap on a 24 GB M4 (16.9 < 18 GiB) and pread on a 16 GB M1 in mode B even when the node's range fits |
| Prefill staging | pread prefill path | Up to 453 MB per layer, not accounted |
| Metal/driver/shader overhead, thread stacks, heap | everywhere | Real but unaccounted |

## 3. Strict-cap rules

1. **No page cache.** Every model-file read (experts, non-expert weights,
   vocab) goes through an fd opened with `F_NOCACHE`. No mmap of model files
   when a cap is set, so mmap mode is off under a cap.
2. **Peak at open.** Every account is reserved *and touched* at open:
   weights, hot tier, slot ring, prefill staging, KV for the configured
   `n_ctx`, the request-state pool. If it does not fit, `open` fails before
   anything else is affected. After open, generation cannot grow memory.
3. **Admission, not growth.** A request reserves KV + DeltaNet state from the
   pre-reserved pool; when the pool is full, the request is rejected.
4. **Fixed-size hot tier.** Sized at open from what is left; eviction swaps
   contents, never changes total size.
5. **Runtime overhead is an account.** Measured in Phase 0 (Metal, driver,
   shaders, stacks), plus a safety margin, reserved inside the cap.
6. **I/O and CPU are bounded too.** `io_threads`, queue depth and an optional
   bytes/s limit come from config; speculative prefetch counts against the
   I/O limit, so the shared SSD and cores are not flooded.

**Enforcement layers**

- *Prevent:* a ledger refuses any allocation that would cross the cap, with
  an error naming the account.
- *Detect:* a watchdog thread samples `phys_footprint` (`task_info`,
  `TASK_VM_INFO`) every few hundred ms. Footprint above the ledger total, or
  within a margin of the cap, stops new work and reports — it means an
  accounting bug.
- *Verify:* tests check `mincore()` over model files shows ~zero resident
  pages after a long generation.

macOS has no supported per-process hard memory limit (no cgroups; jetsam
limits are private API; a VM gives a hard cap but loses Metal GPU
performance). The guarantee therefore comes from the engine itself: all
reservations at open, all model I/O through `F_NOCACHE`, the watchdog as a
tripwire.

## 4. Streaming design under the cap

### Budget (a3b, mode A, 10 GB)

| Account | Size | Note |
| --- | --- | --- |
| Non-expert weights | ~1.3 GB | Estimate — measure from `model_weights.bin` |
| KV, `n_ctx` = 32k | 1.25 GiB | 40 KiB/token (8k = 320 MB) |
| DeltaNet state per request | ~60 MB | 30 linear-attn layers |
| Slot ring | ~96 MiB | Exists: `MAX_K` × (1 synced + 2 prefetch sets), 2 MiB-aligned (`moe/expert_forward.rs:407`) |
| Prefill staging | 453 MB | One layer of experts |
| Runtime overhead + scratch | ~1 GB | Measured in Phase 0, with margin |
| **Hot tier (remainder)** | **~5.8 GB ≈ 3,300 experts** | ~32% of 10,240 |

### Per-token data path (decode, per layer)

```
GPU  CMD1: attention + router logits ─────────┐
I/O  prefetch guessed experts NOT in hot tier  │  (SSD → prefetch slots, F_NOCACHE pread)
CPU  read logits → top-8 (moe_router_cpu) ◄────┘
     for each of the 8 picks:
       hot tier hit   → bind hot buffer        (zero copy, zero I/O)
       prefetch hit   → bind prefetch slot
       miss           → read SSD → synced slot (parallel F_NOCACHE pread)
GPU  batched expert matvec (existing chunked dispatch)
CPU  off critical path: update frequency counts; maybe promote
     slot → hot tier, evicting the coldest (fixed size)
```

Core code change: add `SlotSource::Hot(id)` next to `Synced` /
`Prefetched(idx)` (`io/prefetch.rs:68`), a fixed hot-tier map, and skip
prefetch for experts already hot.

### Expected speed (external SSD, 2.8 GB/s assumed)

Each token needs 320 expert reads × 1.77 MB = 566 MB if nothing is cached.

| Picks served from RAM | Read per token | I/O ceiling |
| --- | --- | --- |
| 0% | 566 MB | ~5 tok/s |
| 50% | 283 MB | ~10 tok/s |
| 70% | 170 MB | ~16 tok/s |
| 80% | 113 MB | ~25 tok/s (likely compute-bound first) |

Expert use is skewed, so 32% of experts should serve more than 32% of
picks. Phase 0's routing profile measures how much more.

### Strengthening streaming (no extra memory)

1. **Profile-seeded hot tier:** count expert use per layer over
   representative prompts; load the most-used at open (no slow warm-up).
2. **Frequency-based promotion/eviction** at fixed size (recency-only LRU is
   what lost in flash-moe, `docs/io-and-gpu-exploration.md:92-110`).
3. **Better guesses:** run layer N+1's router on layer N's hidden state
   instead of reusing the last token's picks (25–30% hit today); only fetch
   confident, non-hot guesses.
4. **Overlap:** fetch the next layer's guessed cold experts while this layer
   computes (the two prefetch sets already exist for this).
5. **Prefill:** read only non-hot experts into staging; larger chunks spread
   reads over more tokens. A second staging buffer for read/compute overlap
   costs +453 MB — only if the budget allows.
6. **Disk tier (Option C):** keep the most important experts on the internal
   SSD (costs disk, not RAM). First check whether the whole 18 GB of experts
   simply fits on the internal SSD.
7. **Bounded I/O:** threads, queue depth, optional bytes/s from config.

### Mode B

Same engine, hot tier sized to the node's whole layer range, loaded at open
with `F_NOCACHE`, never misses. One code path for both modes keeps streaming
exercised everywhere.

## 5. Phases

Each phase ends with a measurement; the next phase is decided on numbers.

### Progress (Oct 25, 2026)

| Phase | State | Notes |
| --- | --- | --- |
| 0 | Partly done | `scripts/check_test_host.sh`; `RsCtx::expert_io_stats()` (disk reads/bytes, hot-tier hits/misses/admissions/evictions); `routing_counts()` / `save_routing_profile()`. Baselines need the M4 / M1 and the model. |
| 1 | Implemented | `OpenConfig` + `RsCtx::open_with` (`n_ctx`, `expert_io`, `io_threads`, `mem_cap`, `runtime_overhead`); KV sized from `n_ctx`; `budget.rs` plan, reserve-and-touch at open, `phys_footprint` check at open and after every `eval_*`; mmap refused under a cap. `layer_range` moves to Phase 3. |
| 2 | Implemented | Under a cap: weights read with `F_NOCACHE` into owned page-aligned memory, expert fds `F_NOCACHE` and not mmapped. `io/hot_tier.rs`: fixed-size, frequency-admitted, epoch-protected; decode binds hits zero-copy, prefetch skips hot experts, prefill copies hits. Seeding from a routing profile. |
| 3 | Implemented | `OpenConfig::layer_range`: only the range's tensors are loaded (compacted under a cap), planned (`BudgetPlan::for_range`) and run; `RsCtx::eval_stage(StageInput::{Tokens, Hidden}, pos, StageOutput::{Logits, Hidden, Discard})` with f32 hidden in/out; `chain_next` norm fusion broken at the range end; snapshot v3 header records the range. A mode B range seeds every expert of the range into the hot tier. |
| 4 | Implemented | `OpenConfig::max_requests`: a fixed pool of `SeqState`s (KV, linear-attn host + GPU recurrence, prefetch guesses), counted per slot in the plan and allocated at open under a cap; `admit_request` (`PoolFull` when full) / `drop_request` / `eval_request` / `state_{size,save,load}_for`. Switching requests swaps the live state with the parked one (no copies); the single-sequence API runs on `ReqId::DEFAULT`. |
| 5 | Implemented | `crates/moeflux-net`: versioned length-prefixed frames into fixed buffers; `moeflux-node` (one range per process, frame buffers counted in the cap via `OpenConfig::host_buffers`); `moeflux-coord run` (drives the chain, per-node decode latency) and `plan` (ranges from caps + measured speed). TCP loopback. |
| 6 | Implemented | Telemetry: `ExpertIoStats` from fixed per-layer atomics — reads/bytes by kind (demand / prefetch / prefill / seed, fast dir), per-layer hot / prefetch / demand counts, decode and prefill wall time with I/O wait, guess accuracy for both guessers; `csv_row` / `layer_csv`, `RsCtx::reset_expert_io_stats`. `OpenConfig::{io_queue_depth, io_bytes_per_sec}` (lock-free gate + GCRA limiter on every read). Prefill stages experts in place: deduped, sorted, coalesced runs of ≤ 8 on the I/O pool. `prefetch_guess = CrossLayer` (layer L's residual through L+1's router on the CPU). `prefetch_lookahead` (L+1's reads issued during L's attention). `fast_experts_dir` (per-layer disk tier). Every knob defaults off = Phase 2 behavior until measured. |
| 7–8 | Not started | Waiting on model-level verification of 1–6. |

Not yet verified on a model (this dev box has no weights and 18 GB):
the model groups of `tests/budget_cap.rs`, `tests/expert_source_equivalence.rs`,
`tests/layer_range.rs`, `tests/requests.rs`, `tests/streaming.rs` and
`moeflux-net/tests/loopback.rs` are `#[ignore]` until run on a capable
host. The cogito (MLA) decode path does not use the hot tier or
prefetch yet, and layer ranges / request switching are unit-tested but
not model-tested on cogito or a17b. Phase 6's exit numbers (read
bytes/token and tok/s vs Phase 2) come from
`tests/streaming.rs::streaming_stats_report` and the per-knob tests,
which print the stats CSV; flip a default only after it wins there.

### Deviations from the plan (phases 3–5)

- **Split equivalence is within tolerance, not bit-exact.** Breaking the
  `chain_next` norm fusion at a range boundary changes the order of one
  RMSNorm, so 40 vs 20+10+10 is compared with the oracle tolerance
  (`assert_logits_close`). A whole-range `eval_stage` matches
  `eval_prompt` / `eval_token` bit for bit, and network vs in-process
  split is bit for bit.
- **`eval_stage` with one row takes the decode path**; `eval_prompt` with
  one token takes the batched path. Compare like with like.
- **The deferred K-expert ring is not per request.** Every eval drains it
  before returning, so it is empty at every request switch; only the
  prefetch guesses move with the request.
- **Checkpoints (`checkpoint_pos` / `restore_to`) stay on the default
  request.** Per-request snapshots go through `state_save_for` /
  `state_load_for`.
- **Without a cap, the pool's GPU buffers are allocated lazily** at the
  first eval, like every other GPU resource on the uncapped path. Under
  a cap everything is allocated and touched at open.
- **Star topology.** Hidden states return to the coordinator between
  stages (one extra hop per stage). It keeps nodes independent and lets
  the coordinator time each node; a node-to-node ring is future work.
- **Network buffers are planned through `OpenConfig::host_buffers`**
  (new `host_buffers` budget account) rather than inside moeflux: the
  node allocates and touches its two frame buffers before opening.
- **The planner works from per-layer estimates** (manifest tensor sizes +
  the KV / linear-state formulas, fixed costs from `BudgetPlan`), with
  uniform per-layer work. Each node's own capped open stays the
  authoritative check.
- **The weight filter keeps unknown top-level tensors** on every range
  (only `model.layers.N.*`, the embedding and the head are split).

### Deviations from the plan (phase 6)

- **Nothing is on by default.** No speed could be measured here (no
  weights), so every Phase 6 knob ships off and the default path is
  the Phase 2 one; the stats CSV decides each default on the M4.
- **Cross-layer input is layer L's post-attention residual** (`h_mid`),
  not layer L's output: the output only exists once L's experts have
  run on the GPU. It goes through L+1's `post_attention_layernorm` and
  router gate (4- or 8-bit, serial CPU matvec into fixed buffers) at
  the top of L+1, after L's expert dispatch is committed. So the guess
  is better but issued no earlier than today's; the overlap item is
  separate. No confidence threshold: the top-k guess is always
  fetched, minus hot experts.
- **Today's prefetch already overlaps more than the plan assumed:**
  layer L+1's prefetch is issued while L's expert dispatch is still on
  the GPU, and runs through L+1's attention. Lookahead moves it one
  step earlier, into L's attention, using the two existing prefetch
  sets; the cost is that L's demand misses can queue behind L+1's
  speculative reads on the shared I/O pool (watch `io_wait_ns`). It
  needs the previous-token guess: a cross-layer guess isn't known that
  early, so `CrossLayer` + lookahead is a config error.
- **Prefill:** hot-tier experts are still copied into staging (the
  gather kernel needs one base buffer); only non-hot experts are read,
  each once per layer per chunk. Across chunks an expert is re-read,
  since staging holds one layer at a time; chunk size is unchanged and
  the second (+453 MB) staging buffer for read/compute overlap was not
  added. The old path's 1.77 MB host blob per layer is gone.
- **Queue depth** is a lock-free gate on reads in flight; `io_threads`
  already bounds pool reads, so it bites only below `io_threads` (or
  when the main thread reads). A coalesced prefill run is one permit
  and is charged its full bytes by the rate limiter (20 ms burst).
- **Disk tier is per layer, not per expert:** a layer file in
  `fast_experts_dir/packed_experts/` overrides the primary. Per-expert
  placement would need a new file format (per-expert files are
  rejected in §8). Copy the most-used layers (from the routing profile)
  or, if it fits, all 18 GB.
- **GPU time is not measured directly:** metal-rs exposes no GPU
  timestamps (`CmdbufStat::gpu_ns` stays 0). The stats give decode
  wall time and its I/O-wait share; GPU waits per command buffer label
  are in `RsCtx::cmdbuf_stats`.
- **No new budget accounts.** The guesser's buffers (≈ 17 KB on a3b)
  and the per-layer counters (≈ 2 KB) are fixed at open inside
  `runtime_overhead`; the decode additions are atomics and fixed
  arrays (hot-tier lookups still take the tier's existing mutex).

| # | Phase | Work | Exit criteria |
| --- | --- | --- | --- |
| 0 | Baseline + telemetry | Run a3b on M4 (mmap and pread, with/without `purge`) and one M1. Add streaming counters: bytes read per token/tier, hits per layer, I/O wait vs GPU time, guess accuracy; CSV output from `profile_smoke.sh`. Routing profile pass. Measure runtime overhead, SSD bandwidth (internal and external), 4 KB / 8 KB frame round-trip M4↔M1 | Baseline CSV; overhead number; expert popularity curve |
| 1 | Runtime config + ledger | `RsCtx::open_with(OpenConfig { mem_cap, n_ctx, expert_io, io_threads, layer_range, … })`; `open()` stays as a wrapper (drama_llama unchanged). KV sized from `n_ctx`. Ledger around `MetalBufferPool` and direct `new_buffer` sites. Reserve-and-touch at open. Watchdog | Oracles green; open fails cleanly over cap; footprint ≤ cap |
| 2 | `F_NOCACHE` + hot tier | `F_NOCACHE` fds for all model reads; weights read into owned buffers under a cap; fixed hot tier with `SlotSource::Hot`; profile seeding; frequency eviction | Identical logits at 0 / 32 / 100% hot tier; `mincore` ≈ 0; tok/s vs Phase 0 |
| 3 | Layer range | Load only the range's layers; `embed` / `forward_range` / `head`; break `chain_next` norm fusion at range end; per-range `state_save` / `state_load` | 40 vs 20+10+10 in one process match within oracle tolerance |
| 4 | Per-request state | Move `layer_states`, `LinearAttnBuffers.{conv_state, delta_state}`, deferred ring and prefetch guesses into `SeqState` keyed by `ReqId`; pre-reserved pool; admission; `drop_request` | Interleaved requests match sequential runs |
| 5 | Network + coordinator | `moeflux-net` frames (req id, position, f32 hidden / token id); loopback first; coordinator on M4; planner assigns ranges from budgets and measured speed | 3 processes over loopback match in-process split; latency measured |
| 6 | Streaming improvements | Cross-layer guesses, overlap, prefill non-hot reads, bounded I/O, optional internal-SSD tier | Read bytes/token and tok/s vs Phase 2 |
| 7 | Router policy | `RouterPolicy` hook after `moe_router_cpu`; Skip renormalises remaining weights (variable k already works through chunked matvec) | Task accuracy incl. JSON and tool calls |
| 8 | 2-bit (optional) | 2-bit repack, batched 2-bit experts kernel, mixed precision per layer | Oracle cosine + task accuracy |

## 6. Tests

- **Cap:** at caps of 6 / 8 / 10 GB, long generation at full `n_ctx` with the
  maximum request count: peak `phys_footprint` ≤ cap, `mincore` ≈ 0 on model
  files, identical logits across caps.
- **Over-cap open:** a configuration that cannot fit fails at `open` with an
  error naming the account.
- **Expert source equivalence:** mmap, pread, hot tier at 0 / 32 / 100% give
  identical logits on the same prompt (needs the per-context I/O mode from
  Phase 1; today only `tests/engine_op_diff.rs` exercises pread).
- **Split equivalence:** 40 layers vs 20+10+10, in-process and over loopback.
- **Request interleaving:** interleaved vs sequential requests match.
- **Streaming knobs** (`tests/streaming.rs`): cross-layer guesses,
  lookahead, bounded I/O and the fast experts dir each give
  bit-identical logits to the default path; prefill reads each routed
  expert once per layer; the stats test prints bytes/token, guess
  accuracy and I/O wait as CSV.
- **Policy quality:** skip / 2-bit judged on task accuracy, not perplexity
  alone (2-bit broke JSON and tool calling in flash-moe).
- **Speed regression gate:** per-machine tok/s baselines; a phase that drops
  streaming decode by more than ~5% is fixed or reverted.

The full oracle gate loads a3b and needs a high-RAM Mac; run it on the M4 or
a larger machine, not the 18 GB dev box.

## 7. Guardrails and upstream

- Keep mmap and pread paths; `ExpertSource` wraps them rather than replacing
  them. mmap mode remains available only when no cap is set.
- Budget, source and policy start as feature-gated modules inside `moeflux`
  (they need `MetalBufferPool`, `MoeBuffers`, `PrefetchState`); only
  `moeflux-net` and the coordinator binary are separate crates at first.
- Keep core diffs small so upstream (mdegans/moeflux) streaming work keeps
  merging. Phases 1, 2 and the telemetry are candidates for upstream PRs;
  coordinate before changing the public API.
- Keep `open()` working for drama_llama throughout.

## 8. Rejected or deferred

| Idea | Decision | Why |
| --- | --- | --- |
| Soft page-cache allowance (two fds + `mincore` governor) | Rejected | macOS cannot reliably evict a file range; overshoots the cap |
| Grow hot tier when the machine is idle | Rejected | Violates the fixed cap |
| mmap mode under a cap | Rejected | File-backed pages escape accounting |
| Lazily committed KV | Rejected | Grows after open |
| LZ4 / LZFSE / APFS compression | Rejected | Measured slower (`docs/optimization-experiments-q4.md`) |
| Per-expert files | Rejected | 15% slower (VFS overhead) |
| Recency-only LRU cache | Replaced by frequency-based | Lost to the OS page cache in flash-moe |
| All-2-bit experts | Deferred, opt-in only | Broke JSON and tool calling |

## 9. Measurements still needed

- [ ] Non-expert weight size (`model_weights.bin`) and Metal/runtime overhead
- [ ] Expert popularity per layer (share of picks covered by the top 32%)
- [ ] Internal vs external SSD random-read bandwidth with `F_NOCACHE`
- [ ] Frame round-trip M4↔M1 over Gigabit Ethernet and Thunderbolt bridge
- [ ] moeflux decode speed on an M1 GPU
- [ ] Confirm architecture figures against the model's `config.json`
