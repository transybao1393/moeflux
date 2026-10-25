//! Streaming improvements (plan phase 6): every knob changes where
//! expert bytes come from or when they are read, never the math, so
//! each must give **bit-identical** logits to the default path. The
//! stats test prints the telemetry the phase is judged on (read
//! bytes per token, guess accuracy, I/O wait vs. decode time) as CSV.
//!
//! Needs the a3b model; check the host first:
//! ```text
//! scripts/check_test_host.sh
//! cargo test -p moeflux --features model-qwen3-6-35b-a3b \
//!     --test streaming -- --include-ignored --test-threads=1 --nocapture
//! ```

#![cfg(all(target_os = "macos", feature = "model-qwen3-6-35b-a3b"))]

#[allow(dead_code)]
mod common;

use common::diff_helpers::default_a3b_paths;
use moeflux::{Ctx, ExpertIoMode, ExpertIoStats, OpenConfig};

const GIB: u64 = 1 << 30;
const N_CTX: usize = 2048;
const DECODE: usize = 24;

fn open(cfg: &OpenConfig) -> Ctx {
    let p = default_a3b_paths();
    Ctx::open_with(&p.weights, &p.manifest, &p.vocab, &p.root, false, cfg).expect("open")
}

/// Prefill a fixed prompt, then greedy-decode. Returns every step's
/// logits and the chosen tokens.
fn run(ctx: &mut Ctx) -> (Vec<Vec<f32>>, Vec<i32>) {
    let prompt: Vec<i32> = (0..64).map(|i| 1000 + 7 * i).collect();
    let mut logits = vec![0f32; ctx.n_vocab()];
    ctx.eval_prompt(&prompt, 0, 0, &mut logits).expect("prefill");
    let mut all = vec![logits.clone()];
    let mut toks = Vec::new();
    for pos in prompt.len()..prompt.len() + DECODE {
        let tok = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as i32)
            .unwrap();
        toks.push(tok);
        ctx.eval_token(tok, pos, 0, &mut logits).expect("decode");
        all.push(logits.clone());
    }
    (all, toks)
}

fn assert_same(label: &str, a: &(Vec<Vec<f32>>, Vec<i32>), b: &(Vec<Vec<f32>>, Vec<i32>)) {
    assert_eq!(a.1, b.1, "[{label}] greedy tokens differ");
    for (step, (x, y)) in a.0.iter().zip(&b.0).enumerate() {
        let bits_equal = x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
        assert!(bits_equal, "[{label}] logits differ at step {step}");
    }
}

/// Pread with a small hot tier, so decode both hits and misses.
fn base_cfg() -> OpenConfig {
    OpenConfig {
        n_ctx: N_CTX,
        expert_io: Some(ExpertIoMode::Pread),
        hot_tier_bytes: Some(GIB),
        ..OpenConfig::default()
    }
}

fn print_stats(label: &str, st: &ExpertIoStats) {
    eprintln!("[streaming] {label}");
    eprintln!("label,{}", ExpertIoStats::CSV_HEADER);
    eprintln!("{label},{}", st.csv_row());
    eprintln!("{}", ExpertIoStats::LAYER_CSV_HEADER);
    eprint!("{}", st.layer_csv());
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn streaming_stats_report() {
    let mut ctx = open(&base_cfg());
    ctx.reset_expert_io_stats();
    let _ = run(&mut ctx);
    let st = ctx.expert_io_stats();
    print_stats("default", &st);
    assert_eq!(st.decode_tokens, DECODE as u64);
    assert!(st.prefill_tokens > 0);
    assert!(st.decode_bytes_per_token().is_some());
    assert!(st.io_wait_ns <= st.decode_ns, "{st:?}");
    // Every pread decode layer after the first token is scored.
    assert!(st.guess_prev.layers > 0, "{st:?}");
}

#[test]
#[ignore = "needs a3b weights; run scripts/check_test_host.sh first"]
fn bounded_io_matches_and_respects_rate() {
    let base = run(&mut open(&base_cfg()));
    let rate = 400u64 << 20; // 400 MiB/s
    let mut ctx = open(&OpenConfig {
        io_queue_depth: Some(2),
        io_bytes_per_sec: Some(rate),
        ..base_cfg()
    });
    ctx.reset_expert_io_stats();
    let t0 = std::time::Instant::now();
    let got = run(&mut ctx);
    let secs = t0.elapsed().as_secs_f64();
    let st = ctx.expert_io_stats();
    print_stats("bounded", &st);
    assert_same("bounded io", &base, &got);
    // Bytes over wall time can exceed the rate by at most the burst.
    let burst = rate as f64 * moeflux::riir::io::io_limit::RATE_BURST.as_secs_f64();
    assert!(
        st.disk_bytes as f64 <= rate as f64 * secs + burst,
        "{} bytes in {secs:.2}s over {rate} B/s",
        st.disk_bytes
    );
}
