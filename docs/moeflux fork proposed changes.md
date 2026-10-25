# moeflux fork: proposed changes

Oct 25, 2026 · @Bao

## Context and goals

The fork must run Qwen3.6-35B-A3B in two modes from one codebase: a single Mac mini with a hard memory cap, and a three-node pipeline where the whole model sits in RAM. Everything else in this doc follows from those two modes.

**Model.** Qwen3.6-35B-A3B: about 35B total and 3B active parameters, 40 layers (30 Gated DeltaNet, 10 full attention), 256 experts per layer of about 3.15M parameters each, 8 routed plus 1 shared expert per token, hidden size 2048, 2 KV heads with head dim 256 ([architecture overview](https://exdai-qwen3-architecture.static.hf.space/index.html), [ApX](https://apxml.com/models/qwen36-35b-a3b)). At 4-bit the whole model is roughly 17.5 GB (calculated).

| Mode | Hardware | Where the weights live | Per-node memory budget |
| --- | --- | --- | --- |
| A. Single node, streaming | Mac mini M4, 24 GB, external SSD on Thunderbolt 4 | External SSD, hot experts cached in RAM | 10 GB, enforced in software |
| B. Three-node pipeline | Mac mini M4 + 2 × Mac mini M1 16 GB | Each node holds its own layer range in RAM | 10 GB each |

**Goals for the fork**

1. Run a contiguous range of layers in one process, so a model can be split across nodes (mode B).
2. Put every allocation under an explicit byte budget, with no help from the OS page cache.
3. Read experts from a pluggable source: fully resident, or streamed from SSD tiers (mode A).
4. Expose router decisions so a policy can skip an expert or use a 2-bit copy.
5. Serve several requests at once through the pipeline, for throughput.
6. Measure everything per token, per layer and per tier.

## What moeflux is today

moeflux 0.1.0-pre.4 is a pre-alpha, pure-Rust streaming-experts MoE runtime on Metal; only the Metal kernels come from Flash-MoE. It already supports our model, which is the main reason to fork it rather than start over.

| Area | Current state | Source |
| --- | --- | --- |
| Language and kernels | Rust host code; Metal shaders kept from Flash-MoE, embedded with `include_str!` and compiled at runtime | [README](https://docs.rs/crate/moeflux/latest/source/README.md) |
| Models | One compile-time feature per build: `model-qwen3-5-a17b`, `model-qwen3-6-35b-a3b`, `model-cogito-v2-671b` | [Cargo.toml](https://docs.rs/crate/moeflux/latest/source/Cargo.toml.orig) |
| Public API | `RsCtx::open`, `eval_prompt`, `eval_token`, `state_save`, `state_load`; `Ctx` is `Send` but not `Sync` and owns its Metal command queue | [lib.rs](https://docs.rs/crate/moeflux/latest/source/src/lib.rs), [README](https://docs.rs/crate/moeflux/latest/source/README.md) |
| Dependencies | `memmap2`, `metal`, `rayon`, `libc`, `bytemuck`, `objc`, `sysinfo`, `serde`, plus the `moeflux-metal` crate | [Cargo.toml](https://docs.rs/crate/moeflux/latest/source/Cargo.toml.orig) |
| Tests | Differential oracles: CPU vs Metal per op, batched vs per-token paths, and an MLX reference harness | [lib.rs](https://docs.rs/crate/moeflux/latest/source/src/lib.rs), [README](https://docs.rs/crate/moeflux/latest/source/README.md) |
| Weight prep | Python scripts `extract_weights.py`, `export_vocab.py`, `export_tokenizer.py` | [README](https://docs.rs/crate/moeflux/latest/source/README.md) |
| License, maturity | MIT; API not stable; no published throughput numbers | [README](https://docs.rs/crate/moeflux/latest/source/README.md) |

The upstream it came from, [Flash-MoE](https://github.com/danveloper/flash-moe), reads experts with parallel `pread` calls, relies on the OS page cache, and memory-maps the non-expert weights. Whether moeflux kept exactly that I/O path needs checking in its source; its `memmap2` dependency suggests memory-mapped reads somewhere.

## Gaps against our requirements

Six gaps stand between moeflux and the two modes; layer-range execution and memory accounting are the two that block everything else.

| Requirement | moeflux today | Gap | Mode |
| --- | --- | --- | --- |
| Run a layer range and hand over a hidden state | One context runs the whole model from tokens to logits | No entry point that takes or returns a hidden state | B |
| Hard memory budget | Allocations and mapped files are not tracked against a budget | Page cache and engine allocations can exceed the cap unseen | A, B |
| Pluggable expert source | Experts come from one built-in path | No way to swap resident, tiered or remote sources | A |
| Router hook | Routing happens inside the forward pass | No callback to skip an expert or pick a precision | A |
| Several requests in flight | One sequence per context; a batched path exists for evaluation | No per-request KV and no scheduler | B |
| Per-token telemetry | Correctness tests, no timing breakdown | No per-layer, per-tier timing or hit rates | A, B |

## Proposed changes by phase

Build the three-node pipeline first: it removes SSD reads from the hot path entirely and needs fewer changes than streaming. Each phase ends with a measurement, so the next phase is decided on numbers.

**Phase 0 · Baseline, no code changes**

- Run unmodified moeflux on the M4 with `model-qwen3-6-35b-a3b`; record decode tok/s, prefill tok/s and peak footprint.
- Repeat after `sudo purge` to see how much the page cache contributes.
- Run the same build on one M1 to measure its kernel speed.

**Phase 1 · Layer-range execution (mode B)**

- Add a `LayerRange { start, end }` to the context config; load only those layers’ weights, KV and DeltaNet state.
- Split the forward pass into three entry points: `embed(tokens) -> Hidden`, `forward_range(&Hidden) -> Hidden`, `head(&Hidden) -> Logits`.
- Make `state_save` and `state_load` work per range, so each node saves only its own state.
- Extend `extract_weights.py` to write one weight file per layer range.

**Phase 2 · Transport and coordinator (mode B)**

- New `moeflux-net` crate: length-prefixed frames carrying a request id, a position and a BF16 hidden state (2048 values, 4 KB per token), or a token id on the way back.
- A coordinator on the M4 drives the loop: embed, send, receive the next token, repeat.
- A planner reads each node’s budget and measured speed at start-up and assigns layer ranges.

**Phase 3 · Memory budget**

- New `moeflux-budget` crate: one ledger with named accounts (resident weights, KV, expert slots, I/O buffers, scratch).
- Route every Metal buffer allocation through the ledger; refuse an allocation that would cross the cap.
- Cross-check against the process footprint the OS reports, and fail loudly on a mismatch.

**Phase 4 · Expert sources (mode A)**

- Define an `ExpertSource` trait with a resident implementation (mode B) and a tiered one (mode A).
- The tiered source reads with `pread` and `F_NOCACHE` straight into a fixed pool of Metal slot buffers, external SSD first, internal SSD as an optional warm tier.
- Keep the expert file format byte-compatible with the kernels, so a slot can be bound without a copy.

**Phase 5 · Router hook and policy (mode A)**

- After routing each layer, call a `RouterPolicy` with the selected experts and gate weights; it returns, per expert, run at 4-bit, run at 2-bit, or skip.
- Renormalise the remaining gate weights when an expert is skipped.
- A 2-bit expert path needs its own kernel and file format; Flash-MoE ships a 2-bit expert mode that can serve as the reference ([Flash-MoE](https://github.com/danveloper/flash-moe)).

**Phase 6 · Several requests in flight (mode B)**

- Give each request its own KV and DeltaNet state on every node.
- A scheduler interleaves requests so all three nodes stay busy, extending the existing batched path.

**Phase 7 · Telemetry (all modes)**

- Per token: time per layer range, network wait, I/O wait per tier, compute, and policy decisions.
- Per tier: hit rates. Output as CSV for the Part 2 tables of the engineering post.

## Crate layout and API sketch

Keep the forked core small and put each new concern in its own crate, so upstream changes merge cleanly and each phase can be switched off.

&#91;embedded content: fork crate layout · 2 existing, 6 new\]

The coordinator talks to the core on every node through the network crate; the core alone drives the Metal kernels and calls the four support crates.

```rust
// moeflux (forked core)
pub struct LayerRange { pub start: usize, pub end: usize } // end exclusive

pub struct StageConfig {
    pub range: LayerRange,
    pub has_embedding: bool,          // first node only
    pub has_head: bool,               // last node only
    pub budget: moeflux_budget::Budget,
    pub experts: Box<dyn moeflux_source::ExpertSource>,
    pub policy: Option<Box<dyn moeflux_policy::RouterPolicy>>,
}

impl Stage {
    pub fn open(weights: &Path, cfg: StageConfig) -> Result<Self, Error>;
    pub fn embed(&mut self, req: ReqId, tokens: &[u32]) -> Result<Hidden, Error>;
    pub fn forward_range(&mut self, req: ReqId, h: &Hidden) -> Result<Hidden, Error>;
    pub fn head(&mut self, req: ReqId, h: &Hidden) -> Result<Logits, Error>;
    pub fn drop_request(&mut self, req: ReqId);   // frees its KV and DeltaNet state
}

// moeflux-source
pub trait ExpertSource {
    /// Make expert `e` of layer `l` available in a slot and return its buffer.
    fn acquire(&mut self, l: usize, e: usize, p: Precision) -> Result<SlotRef, Error>;
}

// moeflux-policy
pub enum Decision { Full, LowBit, Skip }
pub trait RouterPolicy {
    fn decide(&mut self, layer: usize, picks: &[(usize, f32)]) -> Vec<Decision>;
}
```

Names and signatures are a starting point; the real ones should follow moeflux’s existing `Ctx` style once the source is read.

## Testing and upstream plan

Every change must keep the model’s output identical to unmodified moeflux on the same prompt, and the existing differential oracles are the place to enforce that.

**Tests to add**

- **Split equivalence:** run the 40 layers as one range and as 20 + 10 + 10 in one process; the logits must match within the tolerance the existing oracles use.
- **Network equivalence:** the same split across three processes over loopback must match the in-process split exactly.
- **Budget:** a run that would cross the cap must fail with a clear error, never silently grow; the OS-reported footprint must stay under the cap through a long generation.
- **Expert source equivalence:** resident and tiered sources must give identical logits when the policy runs every expert at 4-bit.
- **Policy quality:** skip and 2-bit decisions are judged on task accuracy, not perplexity alone, since Flash-MoE found that all-2-bit experts broke JSON and tool calling.

**Upstream**

- Keep each phase behind its own Cargo feature, so the default build stays upstream-compatible.
- Phases 1, 3 and 7 are general enough to offer upstream as separate pull requests; moeflux is MIT-licensed and the README lists runtime model-variant dispatch as planned work, so coordinate before changing the public API.
- Keep the 2-bit kernels and the cluster coordinator in the fork until they are measured.

## Open questions and first measurements

Four unknowns decide whether the pipeline reaches its calculated ceiling of about 58 tok/s; measure them before writing Phase 2.

- [ ] Round-trip latency of a 4 KB frame between the M4 and an M1, over Gigabit Ethernet and over a Thunderbolt bridge. Each token crosses the network three times.
- [ ] moeflux decode speed on an M1 GPU, against the 68.25 GB/s memory bandwidth the ceiling assumes ([Low End Mac](https://lowendmac.com/2025/apple-silicon-m1-chip-specs/)).
- [ ] How the 10 full-attention layers are interleaved among the 40, from the model’s `config.json`; it decides which node holds which KV cache.
- [ ] Where moeflux reads expert weights today (memory map or `pread`), which decides how much of Phase 4 is new code.

The architecture figures in this doc come from a community architecture page, not Qwen’s official model card; confirm them against `config.json` before Phase 1.
