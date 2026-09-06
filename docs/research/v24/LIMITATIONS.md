# Limitations — v2.4 RGB-T Fusion Campaign

## Data and evaluation scope

- Experiments used the **Fast Eval subset** (`max_train_images=24`, `max_val_images=12`) and **160×160** inputs.
- Metrics are informative for routing and smoke/formal-candidate gating on this subset, **not** a full-dataset or deployment-grade benchmark.
- Small-object AP (`APS`) is often unavailable or degenerate on this tiny subset (`-1` / unstable); do not over-interpret APS from these runs.

## Protocol and claims

- Lab `claim_level` enum forced `exploratory_comparison` even when `parameters.protocol` tagged `formal_candidate` / `mechanism_diagnosis`.
- Formal candidate forbids significance and SOTA claims by design (`allow_significance_claim=false`).
- Three matched seeds are **insufficient** for robust statistical significance or confidence intervals.

## Checkpoint and reuse

- Primary formal metric is **best-on-val mAP50_95** with earliest-epoch tie-break; last epoch is secondary.
- E10/E11 seed42 budget-scale runs were **not** admissible into the formal matrix (protocol / checkpoint audit mismatch).
- Intermediate DFINE checkpoints were pruned for disk; long-term retention emphasizes best/last weights and logs.

## Mechanism diagnosis bounds

- Diagnosis authority is **`diagnostic_only`**: explains failure modes; does not authorize performance superiority claims.
- Gradient / gate / feature probes sample fixed epochs (and best/last tags); some last-epoch probes can fail silently and reduce row counts.
- Detection error comparison uses a **score≥0.1 presence proxy**, not full COCO per-image matching.
- Diagnostic budget used 40 epochs for epoch-40 sampling while exploratory multi-seed arms used 20 epochs—compare within the stated protocol, not across blindly.

## P01 bounds

- Single seed (42), exploratory 20ep budget only.
- Positive Δ vs A3 does **not** imply multi-seed stability, realistic-resolution transfer, or formal promotion.
- Reopening P01 formal validation requires a **new frozen protocol**; exploratory numbers must not be silently upgraded.

## Inference latency bounds

- Existing A2/A4 latency numbers are forward-only, batch=1, synthetic-style benches.
- Mean/p50 and p95 can disagree; FLOPs, preprocess, postprocess, and multi-session variance were not fully controlled.
- Do not claim “A4 is faster” from the current bench alone.

## Hardware / software

- Results are tied to the local CUDA worker image `scientist-rgbt-detection:v2-cuda` and vendored D-FINE commit recorded in run metadata.
- Disk-full events previously corrupted or blocked checkpoint writes; formal runs added free-space gates (≥20GB start).

## What this stage does *not* claim

- SOTA on RGB-T detection.
- Production deployment readiness.
- That gated fusion, FDPN, or residual scaling is a generally better detector family.
- That the AI Scientist loop is fully automated end-to-end without human freeze/archive decisions.
