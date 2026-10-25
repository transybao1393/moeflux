//! End-to-end smoke for the OLMoE-1B-7B variant: open under a hard
//! memory cap, batched prefill of a chat prompt, greedy per-token
//! decode, compared against an MLX reference greedy run of the same
//! 4-bit weights.
//!
//! `#[ignore]` because it needs the converted model on disk. Paths
//! default to `$HOME/models/olmoe-1b-7b/{artifacts,root}`; override
//! the parent with `MOEFLUX_OLMOE_DIR`. Run with:
//!
//! ```text
//! cargo test -p moeflux --features model-olmoe-1b-7b --release \
//!     --test olmoe_smoke -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(all(target_os = "macos", feature = "model-olmoe-1b-7b"))]

use std::path::PathBuf;

use moeflux::{Ctx, Error, OpenConfig};

const VOCAB: usize = 50_304;
const EOS: i32 = 50_279;

/// "What is the capital of France? Answer in one sentence." in the
/// OLMoE chat template (`<|endoftext|><|user|>\n…\n<|assistant|>\n`).
const PROMPT: &[i32] = &[
    50279, 29, 93, 4537, 49651, 187, 1276, 310, 253, 5347, 273, 6181, 32, 37741, 275, 581, 6197,
    15, 187, 29, 93, 515, 5567, 49651, 187,
];
/// MLX greedy continuation: "The capital of France is Paris." + EOS.
const REF_TOKENS: &[i32] = &[510, 5347, 273, 6181, 310, 7785, 15, 50279];

fn model_dir() -> PathBuf {
    std::env::var_os("MOEFLUX_OLMOE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME")).join("models/olmoe-1b-7b")
        })
}

fn open() -> Result<Ctx, Error> {
    let dir = model_dir();
    let art = dir.join("artifacts");
    // Sized for an 18 GiB host with little headroom: the chunk-8192
    // prefill scratch alone is ~2.4 GiB, so a 3 GiB cap only fits
    // with no hot tier and a short KV cache (the prompt + decode use
    // < 40 positions).
    let cfg = OpenConfig {
        n_ctx: 512,
        mem_cap: Some(3482 << 20),
        hot_tier_bytes: Some(0),
        ..OpenConfig::default()
    };
    Ctx::open_with(
        &art.join("model_weights.bin"),
        &art.join("model_weights.json"),
        &art.join("vocab.bin"),
        &dir.join("root"),
        /* use_2bit = */ false,
        &cfg,
    )
}

fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    d / (na.sqrt() * nb.sqrt())
}

#[test]
#[ignore = "needs the OLMoE-1B-7B model on disk"]
fn olmoe_greedy_decode_matches_mlx_reference() -> Result<(), Error> {
    let mut ctx = open()?;
    let mut logits = vec![0.0f32; VOCAB];

    let t0 = std::time::Instant::now();
    ctx.eval_prompt(PROMPT, 0, 0, &mut logits)?;
    eprintln!(
        "[olmoe-smoke] prefill {} tokens in {:.2}s",
        PROMPT.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut generated = Vec::new();
    let mut pos = PROMPT.len();
    let t1 = std::time::Instant::now();
    for _ in 0..REF_TOKENS.len() {
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        let tok = argmax(&logits);
        eprintln!("[olmoe-smoke] step {}: token {tok} logit {:.3}", generated.len(), logits[tok as usize]);
        generated.push(tok);
        if tok == EOS {
            break;
        }
        ctx.eval_token(tok, pos, 0, &mut logits)?;
        pos += 1;
    }
    eprintln!(
        "[olmoe-smoke] decoded {:?} in {:.2}s",
        generated,
        t1.elapsed().as_secs_f64()
    );
    assert_eq!(generated, REF_TOKENS, "greedy tokens diverge from the MLX reference");
    Ok(())
}

/// Batched prefill (graph path) and token-by-token decode (per-token
/// path) of the same prompt must agree on the last position's logits.
#[test]
#[ignore = "needs the OLMoE-1B-7B model on disk"]
fn olmoe_batched_prefill_matches_per_token() -> Result<(), Error> {
    let mut batched = vec![0.0f32; VOCAB];
    {
        let mut ctx = open()?;
        ctx.eval_prompt(PROMPT, 0, 0, &mut batched)?;
    }
    let mut per_token = vec![0.0f32; VOCAB];
    {
        let mut ctx = open()?;
        for (pos, &tok) in PROMPT.iter().enumerate() {
            ctx.eval_token(tok, pos, 0, &mut per_token)?;
        }
    }
    let cos = cosine(&batched, &per_token);
    eprintln!(
        "[olmoe-smoke] batched vs per-token: cos={cos:.6} argmax {} / {}",
        argmax(&batched),
        argmax(&per_token)
    );
    assert_eq!(argmax(&batched), argmax(&per_token));
    assert!(cos > 0.999, "cosine {cos}");
    Ok(())
}
