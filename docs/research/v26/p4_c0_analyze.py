"""C0 analysis of P4 RT-DETR F1 vs F3. Does not train. Does not enter ClaimGate."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "v26_p4_analysis"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_stats(sample_path: Path) -> dict:
    blob = _load(sample_path)
    preds = blob.get("predictions") or []
    max_scores: list[float] = []
    n_ge_01 = 0
    n_scores = 0
    for row in preds:
        scores = [float(x) for x in (row.get("scores") or [])]
        n_scores += len(scores)
        n_ge_01 += sum(1 for s in scores if s >= 0.1)
        if scores:
            max_scores.append(max(scores))
    return {
        "n_sample_images": len(preds),
        "n_sample_scores": n_scores,
        "n_sample_scores_ge_0_1": n_ge_01,
        "mean_max_score": mean(max_scores) if max_scores else None,
    }


def _pack(pack_id: str) -> dict:
    base = ROOT / "outputs" / pack_id
    aps = _load(base / "run" / "aps_lowlight.json")
    metrics = _load(base / "run" / "metrics.json")
    m = metrics["metrics"]
    training = metrics["training"]
    return {
        "pack_id": pack_id,
        "how": training.get("fusion_method"),
        "fusion_applied": training.get("fusion_applied"),
        "seed": training.get("seed"),
        "epochs_completed": training.get("epochs_completed"),
        "APS_lowlight": aps["APS_lowlight"],
        "mAP50_95_lowlight": aps["mAP50_95_lowlight"],
        "AP50_lowlight": aps["AP50_lowlight"],
        "Recall_small_lowlight": aps["Recall_small_lowlight"],
        "n_images_lowlight": aps["n_images_lowlight"],
        "n_annotations_lowlight": aps["n_annotations_lowlight"],
        "mAP50_full": m["mAP50"],
        "mAP50_95_full": m["mAP50_95"],
        "AP_small_full": m["AP_small"],
        "prediction_count": m["prediction_count"],
        "prediction_count_score_ge_0_1": m["prediction_count_score_ge_0_1"],
        "parameter_count": m["parameter_count"],
        "peak_gpu_memory_mb": m["peak_gpu_memory_mb"],
        "duration_seconds": m["duration_seconds"],
        "sample": _score_stats(base / "run" / "sample_predictions.json"),
    }


def main() -> None:
    f1 = _pack("v26_p4_r0")
    f3 = _pack("v26_p4_r1")
    dfine_f1 = 0.0045926865160844455
    dfine_f3_s42 = 0.02135704762627717
    rtdetr_delta = f3["APS_lowlight"] - f1["APS_lowlight"]
    dfine_delta = dfine_f3_s42 - dfine_f1
    mixed = {
        "APS_lowlight": rtdetr_delta,
        "mAP50_95_lowlight": f3["mAP50_95_lowlight"] - f1["mAP50_95_lowlight"],
        "AP50_lowlight": f3["AP50_lowlight"] - f1["AP50_lowlight"],
        "Recall_small_lowlight": f3["Recall_small_lowlight"] - f1["Recall_small_lowlight"],
        "mAP50_full": f3["mAP50_full"] - f1["mAP50_full"],
        "mAP50_95_full": f3["mAP50_95_full"] - f1["mAP50_95_full"],
    }
    signs = {k: ("up" if v > 0 else "down" if v < 0 else "flat") for k, v in mixed.items()}
    summary = {
        "schema_version": "1.0",
        "campaign": "v26_p4_c0_analysis",
        "gpu_trained": False,
        "claim_gate": "C0",
        "not_a_claim": True,
        "not_g2": True,
        "not_g3": True,
        "not_transfer_success": True,
        "parent_packs": ["outputs/v26_p4_r0", "outputs/v26_p4_r1"],
        "question": "Does F3 gated_multiscale transfer from D-FINE to RT-DETR as a detector-general strategy on low_light_subset_v1?",
        "f1": f1,
        "f3": f3,
        "deltas_rtdetr_f3_minus_f1": mixed,
        "delta_signs": signs,
        "transfer_delta_aps_lowlight": {
            "dfine_r0_f1_to_r1_f3_seed42": dfine_delta,
            "rtdetr_p4_f1_to_f3_seed42": rtdetr_delta,
            "rtdetr_over_dfine_ratio": (rtdetr_delta / dfine_delta) if dfine_delta else None,
        },
        "cost": {
            "param_ratio": f3["parameter_count"] / f1["parameter_count"],
            "peak_gpu_mem_ratio": f3["peak_gpu_memory_mb"] / f1["peak_gpu_memory_mb"],
            "duration_ratio": f3["duration_seconds"] / f1["duration_seconds"],
        },
        "interpretation": {
            "scientific_outcome": "INCONCLUSIVE",
            "reason": (
                "Primary APS_lowlight rose, but AP50_lowlight and full-set mAP50 fell. "
                "RT-DETR F3−F1 delta is an order of magnitude smaller than D-FINE F3−F1 at the same seed. "
                "Both P4 guns used only 2 epochs. Mixed signs block a transfer-success reading at C0."
            ),
        },
        "next_gpu_requires": "protocol amendment: P4 max_rounds is frozen at 2; extra seeds are out of protocol",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
