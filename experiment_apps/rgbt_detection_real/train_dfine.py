"""Run vendored D-FINE-S training under Scientist Lab budgets."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from artifact_writer import write_json
from claim_gate import claim_gate_metadata, resolve_protocol
from checkpoint_policy import (
    parse_checkpoint_policy,
    resolve_dfine_checkpoint_files,
    select_best_and_last_from_dfine_log,
)
from dfine_config_builder import dfine_root_from_app, write_dfine_fast_config
from dfine_dataset_stage import count_categories, stage_coco_for_dfine
from disk_gate import assert_disk_for_formal_train
from fusion_names import normalize_fusion_method
from models.fusion_factory import parse_fusion_config
from rgbt_pair_audit import write_rgbt_pair_audit


IMPLEMENTATION = "dfine_s_vendored_v0_8_9"
PROBE_EXECUTION_MODES = frozenset({"fast_eval", "smoke", "smoke_train", "debug"})


def resolve_stage_image_caps(
    execution_mode: str, params: Mapping[str, Any]
) -> tuple[int | None, int | None]:
    """Subset only probe/smoke HOW. Formal / full_train keep the full split."""
    if str(execution_mode) not in PROBE_EXECUTION_MODES:
        return None, None
    max_train = params.get("max_train_images")
    max_val = params.get("max_val_images")
    return (
        int(max_train or 16),
        int(max_val or 8),
    )


def run_dfine_train(
    *,
    data_root: Path,
    output_dir: Path,
    config: dict[str, Any],
    contract: dict[str, Any],
    seed: int,
    input_mode: str,
    fusion_method: str,
    app_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    app_dir = Path(app_dir or Path(__file__).resolve().parent)
    dfine_root = dfine_root_from_app(app_dir)
    if not (dfine_root / "train.py").is_file():
        raise FileNotFoundError(
            f"DFINE vendor missing at {dfine_root}; see third_party/VENDOR.md"
        )

    params = dict(config.get("parameters") or contract.get("parameters") or {})
    detector = str(
        params.get("detector") or params.get("dfine_backend") or "dfine"
    ).strip().lower().replace("-", "_")
    if detector in {"rt_detr"}:
        detector = "rtdetr"
    if detector not in {"dfine", "rtdetr", "dfine_s", "vendor"}:
        if detector not in {"auto", "standin", "torch_mini", "mini"}:
            raise RuntimeError(f"unsupported detector={detector!r}")
        detector = "dfine"
    if detector == "dfine_s" or detector == "vendor":
        detector = "dfine"
    baseline_key = "rtdetr_s" if detector == "rtdetr" else "dfine_s"
    implementation = (
        "rtdetr_s_lyuwenyu_zoo" if detector == "rtdetr" else IMPLEMENTATION
    )
    fusion_method = normalize_fusion_method(fusion_method)
    epochs = int(params.get("epochs") or 2)
    batch_size = int(params.get("batch_size") or 2)
    num_workers = int(params.get("num_workers") or 0)
    _proto_params = str(params.get("protocol") or "").strip().lower()
    _proto_task = str((contract.get("task_config") or {}).get("protocol") or "").strip().lower()
    _det_protocols = {
        "gate_j_deterministic_diagnosis",
        "gate_j_rng_hunt",
        "gate_j3b_rng_hunt",
    }
    deterministic = bool(
        params.get("deterministic_algorithms")
        or params.get("cudnn_deterministic")
        or _proto_params in _det_protocols
        or _proto_task in _det_protocols
        or "rng_hunt" in _proto_params
        or "rng_hunt" in _proto_task
    )
    if deterministic:
        # Diagnosis default: eliminate worker RNG drift.
        num_workers = 0
        params = dict(params)
        params["num_workers"] = 0
    image_height = params.get("image_height")
    image_width = params.get("image_width")
    image_size = params.get("image_size")
    # Legacy contracts used image_width as square side when height is omitted.
    if image_height is None and image_width is not None and image_size is None:
        image_size = image_width
        image_width = None
    if image_height is None and image_width is None and image_size is None:
        image_size = 160
    learning_rate = float(params.get("learning_rate") or 2e-4)
    warmup_raw = params.get("warmup_duration", params.get("warmup_steps"))
    warmup_duration = int(warmup_raw) if warmup_raw is not None else None
    ms_raw = params.get("lr_scheduler_milestones", params.get("multistep_milestones"))
    lr_scheduler_milestones = (
        [int(x) for x in ms_raw] if ms_raw is not None else None
    )
    gamma_raw = params.get("lr_scheduler_gamma", params.get("multistep_gamma"))
    lr_scheduler_gamma = float(gamma_raw) if gamma_raw is not None else None
    execution_mode = str(contract.get("execution_mode") or "fast_eval")
    max_train_images, max_val_images = resolve_stage_image_caps(execution_mode, params)
    protocol = resolve_protocol(
        contract=contract,
        parameters=params,
        execution_mode=execution_mode,
    )
    pretrained = bool(params.get("pretrained", False))
    local_model_dir = params.get("local_model_dir")
    checkpoint_policy = parse_checkpoint_policy(params)
    protocol_name = str(params.get("protocol") or "").strip().lower()
    task_protocol = str((contract.get("task_config") or {}).get("protocol") or "").strip().lower()
    enforce_disk = bool(params.get("enforce_disk_gate")) or epochs >= 40 or any(
        name.startswith("formal_candidate") for name in (protocol_name, task_protocol, protocol)
    )
    disk_status: dict[str, Any] | None = None
    if enforce_disk:
        disk_status = assert_disk_for_formal_train(output_dir)
        write_json(output_dir / "disk_gate.json", disk_status)
    # Only Fast Eval may shrink num_queries to fit tiny smoke resolutions.
    # Formal / full-size runs keep the model query structure unchanged.
    scale_queries = execution_mode in {"fast_eval", "smoke", "debug"}
    if "scale_queries_to_tokens" in params:
        scale_queries = bool(params.get("scale_queries_to_tokens"))

    if str(input_mode).lower() in {"rgbt", "rgb_thermal"}:
        write_rgbt_pair_audit(Path(data_root), output_dir / "rgbt_pair_audit.json")

    stage_root = output_dir / "_dfine_stage"
    stage_paths = stage_coco_for_dfine(
        Path(data_root),
        stage_root,
        input_mode=input_mode,
        fusion_method=fusion_method,
        label_map_path=output_dir / "category_label_map.json",
        max_train_images=max_train_images,
        max_val_images=max_val_images,
    )
    train_ann = json.loads(Path(stage_paths["train_ann"]).read_text(encoding="utf-8"))
    val_ann = json.loads(Path(stage_paths["val_ann"]).read_text(encoding="utf-8"))
    write_json(
        output_dir / "dfine_subset.json",
        {
            "execution_mode": execution_mode,
            "input_mode": input_mode,
            "fusion_method": fusion_method,
            "staging_mode": stage_paths.get("staging_mode"),
            "max_train_images": max_train_images,
            "max_val_images": max_val_images,
            "staged_train_images": len(train_ann.get("images") or []),
            "staged_val_images": len(val_ann.get("images") or []),
            "staged_train_annotations": len(train_ann.get("annotations") or []),
            "staged_val_annotations": len(val_ann.get("annotations") or []),
        },
    )
    num_classes = count_categories(stage_paths["train_ann"])
    dfine_out = output_dir / "_dfine_run"
    if dfine_out.exists():
        shutil.rmtree(dfine_out)
    dfine_out.mkdir(parents=True)

    disable_multiscale = deterministic or bool(params.get("disable_multiscale_collate"))
    cfg_path = write_dfine_fast_config(
        dfine_root=dfine_root,
        config_path=output_dir / "dfine_fast_config.yml",
        stage_paths=stage_paths,
        output_dir=dfine_out,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=int(image_size) if image_size is not None else None,
        image_height=int(image_height) if image_height is not None else None,
        image_width=int(image_width) if image_width is not None else None,
        learning_rate=learning_rate,
        num_classes=num_classes,
        seed=seed,
        scale_queries_to_tokens=scale_queries,
        budget_record_path=output_dir / "dfine_spatial_query_budget.json",
        pretrained=pretrained,
        local_model_dir=str(local_model_dir) if local_model_dir else None,
        disable_multiscale_collate=disable_multiscale,
        warmup_duration=warmup_duration,
        lr_scheduler_milestones=lr_scheduler_milestones,
        lr_scheduler_gamma=lr_scheduler_gamma,
        detector=detector,
    )
    # Keep a copy of the label map next to metrics for eval export / audit.
    label_map = stage_paths.get("category_label_map")
    if isinstance(label_map, dict):
        write_json(output_dir / "category_label_map.json", label_map)

    # Import vendored D-FINE.
    if str(dfine_root) not in sys.path:
        sys.path.insert(0, str(dfine_root))
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    from src.core import YAMLConfig  # type: ignore
    from src.solver import TASKS  # type: ignore

    if detector == "rtdetr":
        from vendor_rtdetr import register_rtdetr_modules

        register_rtdetr_modules()

    import torch

    determinism_report: dict[str, Any] | None = None
    order_audit: dict[str, Any] | None = None
    reseed_each_epoch = False
    disable_ema = False
    if deterministic:
        from determinism import (
            build_epoch0_order_audit,
            configure_determinism,
            rebuild_train_loader_seeded,
            write_determinism_artifacts,
        )

        determinism_report = configure_determinism(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yaml_cfg = YAMLConfig(str(cfg_path))
    # Disable distributed assumptions for single-process Lab runs.
    if hasattr(yaml_cfg, "yaml_cfg"):
        yaml_cfg.yaml_cfg["use_amp"] = bool(params.get("mixed_precision")) and device == "cuda"
        yaml_cfg.yaml_cfg["sync_bn"] = False
        if detector == "rtdetr":
            yaml_cfg.yaml_cfg["use_amp"] = False

    if deterministic:
        # Force AMP off under strict determinism diagnosis unless explicitly requested.
        if params.get("force_amp_under_determinism") is not True and hasattr(yaml_cfg, "yaml_cfg"):
            yaml_cfg.yaml_cfg["use_amp"] = False
        # Gate J3b: optimizer.yml defaults use_ema=True — disable for RNG hunt.
        rng_hunt = "rng_hunt" in str(protocol) or "rng_hunt" in _proto_params
        if "disable_ema" in params:
            disable_ema = bool(params.get("disable_ema"))
        else:
            disable_ema = rng_hunt
        if disable_ema and hasattr(yaml_cfg, "yaml_cfg"):
            yaml_cfg.yaml_cfg["use_ema"] = False
        if "reseed_each_epoch" in params:
            reseed_each_epoch = bool(params.get("reseed_each_epoch"))
        else:
            reseed_each_epoch = rng_hunt
        _ = yaml_cfg.train_dataloader  # materialize
        rebuild_train_loader_seeded(yaml_cfg=yaml_cfg, seed=seed, num_workers=num_workers)
        order_audit = build_epoch0_order_audit(
            dataset=yaml_cfg.train_dataloader.dataset,
            seed=seed,
            batch_size=batch_size,
        )
        write_determinism_artifacts(
            output_dir,
            configure_report=determinism_report or {},
            order_audit=order_audit,
            extra={
                "protocol": str(params.get("protocol") or protocol or "gate_j_deterministic_diagnosis"),
                "num_workers": num_workers,
                "disable_multiscale_collate": disable_multiscale,
                "disable_ema": disable_ema,
                "reseed_each_epoch": reseed_each_epoch,
                "mixed_precision": bool((yaml_cfg.yaml_cfg or {}).get("use_amp")),
                "use_ema": bool((yaml_cfg.yaml_cfg or {}).get("use_ema")),
            },
        )

    fusion_summary: dict[str, Any] | None = None
    wrap_summary: dict[str, Any] | None = None
    from models.how_plugin_loader import (
        HowPluginError,
        is_plugin_backbone_wrap,
        is_plugin_fusion_method,
        load_how_plugin,
        load_how_plugin_backbone_wrap,
        plugin_input_mode,
        plugin_path_for,
    )

    wrap_raw = ""
    wrap_blob = params.get("backbone_wrap")
    if isinstance(wrap_blob, dict):
        wrap_raw = str(wrap_blob.get("type") or "").strip()
    if is_plugin_backbone_wrap(wrap_raw):
        from apply_backbone_wrap import apply_plugin_backbone_wrap

        try:
            plugin_wrap = load_how_plugin_backbone_wrap(
                wrap_raw, yaml_cfg.model.backbone
            )
        except HowPluginError as exc:
            raise RuntimeError(f"HOW backbone_wrap plugin failed to load: {exc}") from exc
        wrap_mode = plugin_input_mode(plugin_path_for(wrap_raw))
        thermal_train = stage_paths.get("thermal_train_img")
        thermal_val = stage_paths.get("thermal_val_img")
        wrap_summary = apply_plugin_backbone_wrap(
            yaml_cfg,
            plugin_wrap,
            how_id=str(wrap_raw),
            input_mode=wrap_mode,
            thermal_train_img=Path(thermal_train) if thermal_train else None,
            thermal_val_img=Path(thermal_val) if thermal_val else None,
        )
        write_json(output_dir / "backbone_wrap_summary.json", wrap_summary)

    if is_plugin_fusion_method(fusion_method):
        if wrap_summary is not None:
            raise RuntimeError(
                "one slot per plugin: backbone_wrap and fusion plugin cannot both apply"
            )
        from apply_gated_fusion import apply_plugin_feature_fusion

        thermal_train = stage_paths.get("thermal_train_img")
        thermal_val = stage_paths.get("thermal_val_img")
        if thermal_train is None or thermal_val is None:
            raise RuntimeError(
                "plugin fusion staging missing thermal_* folders; "
                f"staging_mode={stage_paths.get('staging_mode')}"
            )
        try:
            plugin_fusion = load_how_plugin(str(fusion_method))
        except HowPluginError as exc:
            raise RuntimeError(f"HOW plugin failed to load: {exc}") from exc
        fusion_summary = apply_plugin_feature_fusion(
            yaml_cfg,
            plugin_fusion,
            thermal_train_img=Path(thermal_train),
            thermal_val_img=Path(thermal_val),
            how_id=str(fusion_method),
        )
        write_json(output_dir / "fusion_module_summary.json", fusion_summary)

    fusion_cfg = parse_fusion_config(params, fusion_method=fusion_method)
    if fusion_cfg is not None and fusion_summary is None:
        if wrap_summary is not None:
            raise RuntimeError(
                "one slot per plugin: backbone_wrap and gated fusion cannot both apply"
            )
        from apply_gated_fusion import apply_gated_multiscale_fusion

        thermal_train = stage_paths.get("thermal_train_img")
        thermal_val = stage_paths.get("thermal_val_img")
        if thermal_train is None or thermal_val is None:
            raise RuntimeError(
                "gated_multiscale staging missing thermal_* folders; "
                f"staging_mode={stage_paths.get('staging_mode')}"
            )
        fusion_summary = apply_gated_multiscale_fusion(
            yaml_cfg,
            fusion_cfg,
            thermal_train_img=Path(thermal_train),
            thermal_val_img=Path(thermal_val),
        )
        write_json(output_dir / "fusion_module_summary.json", fusion_summary)

    neck_summary: dict[str, Any] | None = None
    from models.neck_factory import parse_neck_config
    from apply_fdpn_neck import apply_neck

    neck_cfg = parse_neck_config(params)
    if detector == "rtdetr" and neck_cfg.type != "standard":
        raise RuntimeError(
            "RT-DETR Adapter does not implement N1/FDPN; "
            "do not stuff the D-FINE neck into the second detector"
        )
    if neck_cfg.type != "standard":
        neck_summary = apply_neck(yaml_cfg, neck_cfg)
        write_json(output_dir / "neck_module_summary.json", neck_summary)

    solver = TASKS[yaml_cfg.yaml_cfg["task"]](yaml_cfg)

    # FLOPs profiler (calflops) may probe with a square default that mismatches
    # non-square eval_spatial_size; never fail the run for profiling alone.
    try:
        from src.misc import profiler_utils as _profiler_utils  # type: ignore

        _orig_stats = _profiler_utils.stats

        def _safe_stats(cfg):  # noqa: ANN001
            try:
                return _orig_stats(cfg)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] model FLOPs/stats skipped: {type(exc).__name__}: {exc}")
                return 0, f"flops_unavailable: {type(exc).__name__}"

        _profiler_utils.stats = _safe_stats
    except Exception:  # noqa: BLE001
        pass

    if deterministic and reseed_each_epoch:
        from determinism import install_epoch_reseed

        install_epoch_reseed(solver, seed)

    mech_raw = params.get("mechanism_diagnosis")
    if mech_raw is None:
        mech_enabled = False
        mech: dict[str, Any] = {}
    elif isinstance(mech_raw, dict):
        mech = dict(mech_raw)
        mech_enabled = bool(mech.get("enabled", True))
    else:
        mech = {}
        mech_enabled = bool(mech_raw)
    if mech_enabled or protocol in {"mechanism_diagnosis", "diagnostic_only"}:
        from instrumented_fit import install_instrumented_fit

        arm_name = str(
            mech.get("arm")
            or params.get("experiment_arm")
            or (contract.get("task_config") or {}).get("experiment_id")
            or "ARM"
        )
        sample_epochs = list(mech.get("sample_epochs_1based") or [1, 20, 40])
        diag_dir = output_dir / "mechanism_diagnostics"
        install_instrumented_fit(
            solver,
            arm=arm_name,
            diag_dir=diag_dir,
            sample_epochs_1based=[int(x) for x in sample_epochs],
        )
        write_json(
            diag_dir / "probe_plan.json",
            {
                "arm": arm_name,
                "sample_epochs_1based": sample_epochs,
                "protocol": protocol,
                "formal_performance_claims_allowed": False,
            },
        )

    solver.fit()

    # Re-check disk before copying large checkpoints.
    if enforce_disk:
        disk_status = assert_disk_for_formal_train(output_dir)
        write_json(output_dir / "disk_gate.json", disk_status)

    selection = select_best_and_last_from_dfine_log(
        dfine_out, policy=checkpoint_policy
    )
    native_ckpts = resolve_dfine_checkpoint_files(dfine_out)
    ckpt_dir = output_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Formal names: best_weights (eval) + last_resume (recover) + legacy last.pt
    best_dst = ckpt_dir / "best_weights.pth"
    last_resume_dst = ckpt_dir / "last_resume.pth"
    last_dst = ckpt_dir / "last.pt"
    if native_ckpts["best_native"] is not None:
        shutil.copy2(native_ckpts["best_native"], best_dst)
    if native_ckpts["last_native"] is not None:
        shutil.copy2(native_ckpts["last_native"], last_resume_dst)
        shutil.copy2(native_ckpts["last_native"], last_dst)
    elif native_ckpts["best_native"] is not None:
        shutil.copy2(native_ckpts["best_native"], last_dst)
    else:
        torch.save({"vendor": IMPLEMENTATION, "note": "no_native_ckpt"}, last_dst)

    # Drop bulky intermediate DFINE artifacts to protect disk.
    for pattern in ("checkpoint*.pth",):
        for path in dfine_out.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass
    eval_dir = dfine_out / "eval"
    if eval_dir.is_dir():
        for path in eval_dir.glob("*.pth"):
            try:
                path.unlink()
            except OSError:
                pass

    write_json(
        output_dir / "checkpoint_selection.json",
        {
            **{k: v for k, v in selection.items() if k != "rows"},
            "native_best_path": (
                str(native_ckpts["best_native"]) if native_ckpts["best_native"] else None
            ),
            "native_last_path": (
                str(native_ckpts["last_native"]) if native_ckpts["last_native"] else None
            ),
            "lab_best_weights": str(best_dst) if best_dst.exists() else None,
            "lab_last_resume": str(last_resume_dst) if last_resume_dst.exists() else None,
        },
    )
    if selection.get("rows"):
        write_json(
            output_dir / "metrics_per_epoch.json",
            {
                "source": "dfine_log_txt",
                "selection_metric": "mAP50_95",
                "epochs": selection["rows"],
            },
        )

    # Prefer native eval metrics if solver exposed them; else zeros + resource fields.
    metrics_values = {
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "AP_small": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    try:
        if hasattr(solver, "val"):
            solver.val()
    except Exception:  # noqa: BLE001
        pass
    metrics_values.update(_read_dfine_eval_metrics(dfine_out))
    # Primary formal metrics come from best-on-val when available.
    best_row = selection.get("best") if selection.get("ok") else None
    last_row = selection.get("last") if selection.get("ok") else None
    if isinstance(best_row, dict):
        metrics_values["mAP50_95"] = float(best_row["mAP50_95"])
        metrics_values["mAP50"] = float(best_row["mAP50"])
        if best_row.get("APS") is not None:
            metrics_values["AP_small"] = float(best_row["APS"])
        metrics_values["AP75"] = (
            float(best_row["AP75"]) if best_row.get("AP75") is not None else 0.0
        )
    prediction_audit = _audit_val_predictions(
        solver,
        device=device,
        dump_path=output_dir / "sample_predictions.json",
        coco_dump_path=output_dir / "val_detections_coco.json",
        label_map=label_map if isinstance(label_map, dict) else None,
        score_threshold=0.05,
        max_dump_images=8,
    )
    write_json(output_dir / "prediction_audit.json", prediction_audit)
    metrics_values["prediction_count"] = float(prediction_audit.get("prediction_count") or 0)
    metrics_values["prediction_count_score_ge_0_1"] = float(
        prediction_audit.get("prediction_count_score_ge_0_1") or 0
    )
    metrics_values["nonzero_box_count"] = float(prediction_audit.get("nonzero_box_count") or 0)

    duration = time.time() - started
    peak_gpu = 0.0
    if device == "cuda" and torch.cuda.is_available():
        peak_gpu = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    param_count = 0.0
    try:
        model = getattr(solver, "model", None) or getattr(yaml_cfg, "model", None)
        if model is not None:
            param_count = float(sum(p.numel() for p in model.parameters()))
    except Exception:  # noqa: BLE001
        param_count = 0.0

    task_config = dict(contract.get("task_config") or {})
    claim_meta = claim_gate_metadata(protocol)
    checkpoint_metrics = {
        "best_on_validation": best_row,
        "last_epoch": last_row,
        "best_minus_last_mAP50_95": selection.get("best_minus_last_mAP50_95"),
        "best_epoch": selection.get("best_epoch"),
        "last_epoch_number": selection.get("last_epoch"),
    }
    metrics = {
        "schema_version": "1.0",
        "project_id": contract.get("project_id", "project_rgbt_002"),
        "node_id": contract.get("node_id", "rgbt_node"),
        "task_type": "rgbt_detection",
        "primary_metric": task_config.get("primary_metric", "mAP50_95"),
        "metrics": {
            **metrics_values,
            "duration_seconds": duration,
            "peak_gpu_memory_mb": peak_gpu,
            "parameter_count": param_count,
        },
        "checkpoint_policy": checkpoint_policy,
        "checkpoint_metrics": checkpoint_metrics,
        "training": {
            "seed": seed,
            "epochs_requested": epochs,
            "epochs_completed": epochs,
            "duration_seconds": duration,
            "backend": detector,
            "device": device,
            "trained": True,
            "baseline_key": baseline_key,
            "baseline_implementation": implementation,
            "vendor_commit": "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6",
            "config_path": str(cfg_path),
            "fusion_method": fusion_method,
            "staging_mode": stage_paths.get("staging_mode"),
            "fusion_applied": fusion_summary is not None
            or str(stage_paths.get("staging_mode") or "") == "early_concat_blend",
            "pretrained": pretrained,
            "neck_type": (neck_summary or {}).get("neck_type", "standard"),
            "neck_applied": neck_summary is not None and neck_cfg.type != "standard",
            "checkpoint_primary": checkpoint_policy.get("primary"),
            "best_epoch": selection.get("best_epoch"),
        },
        "evaluation_scope": task_config.get("evaluation_scope", "fast_eval_subset"),
        "claim_level": task_config.get("claim_level", "exploratory_comparison"),
        "execution_mode": contract.get("execution_mode", "fast_eval"),
        **claim_meta,
        "status": "completed",
    }
    summary = {
        "baseline_key": baseline_key,
        "baseline_implementation": implementation,
        "vendor_status": "vendored",
        "vendor_commit": "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6",
        "device": device,
        "parameter_count": param_count,
        "num_classes": num_classes,
        "input_mode": input_mode,
        "fusion_method": fusion_method,
        "fusion_module": fusion_summary,
        "neck_type": (neck_summary or {}).get("neck_type", "standard"),
        "neck_module": neck_summary,
        **claim_meta,
    }
    resources = {
        "cpu_count_requested": (contract.get("resources") or {}).get("cpu_count"),
        "memory_gb_requested": (contract.get("resources") or {}).get("memory_gb"),
        "gpu_count_requested": (contract.get("resources") or {}).get("gpu_count"),
        "peak_gpu_memory_mb": peak_gpu,
        "duration_seconds": duration,
        "device": device,
        "cuda_available": device == "cuda",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "model_summary.json", summary)
    write_json(output_dir / "resource_usage.json", resources)
    write_json(
        output_dir / "execution.json",
        {
            "status": "completed",
            "baseline_key": baseline_key,
            "baseline_implementation": implementation,
            "device": device,
            "protocol": protocol,
            "allow_scientific_claims": claim_meta["allow_scientific_claims"],
        },
    )
    # Prefer DFINE per-epoch log.txt; fall back to placeholder schema.
    _write_training_history_from_dfine_log(
        dfine_out=dfine_out,
        output_path=output_dir / "training_history.csv",
        epochs=epochs,
    )
    return metrics


def _write_training_history_from_dfine_log(
    *,
    dfine_out: Path,
    output_path: Path,
    epochs: int,
) -> None:
    """Export per-epoch learning curve from DFINE log.txt (JSON lines)."""
    import csv
    import json

    header = [
        "epoch",
        "train_loss",
        "loss_vfl",
        "loss_bbox",
        "loss_giou",
        "mAP50_95",
        "mAP50",
        "AP75",
        "APS",
        "lr",
        "backbone_actual_lr",
        "non_backbone_actual_lr",
        "scheduler_last_epoch",
        "train_time_seconds",
        "test_time_seconds",
    ]
    rows: list[dict[str, Any]] = []
    log_txt = Path(dfine_out) / "log.txt"
    if log_txt.is_file():
        for line in log_txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "epoch" not in payload:
                continue
            bbox = payload.get("test_coco_eval_bbox") or []
            backbone_lr = payload.get("backbone_actual_lr")
            if backbone_lr is None:
                backbone_lr = payload.get("train_lr")
            non_bb_lr = payload.get("non_backbone_actual_lr")
            rows.append(
                {
                    "epoch": int(payload.get("epoch", -1)) + 1,
                    "train_loss": payload.get("train_loss"),
                    "loss_vfl": payload.get("train_loss_vfl"),
                    "loss_bbox": payload.get("train_loss_bbox"),
                    "loss_giou": payload.get("train_loss_giou"),
                    "mAP50_95": float(bbox[0]) if len(bbox) > 0 else None,
                    "mAP50": float(bbox[1]) if len(bbox) > 1 else None,
                    "AP75": float(bbox[2]) if len(bbox) > 2 else None,
                    "APS": float(bbox[3]) if len(bbox) > 3 else None,
                    "lr": backbone_lr if backbone_lr is not None else payload.get("train_lr"),
                    "backbone_actual_lr": backbone_lr,
                    "non_backbone_actual_lr": non_bb_lr,
                    "scheduler_last_epoch": payload.get("scheduler_last_epoch"),
                    "train_time_seconds": payload.get("train_time"),
                    "test_time_seconds": payload.get("test_time") or payload.get("test_eval_time"),
                }
            )
    if not rows:
        rows = [
            {
                "epoch": i + 1,
                "train_loss": None,
                "loss_vfl": None,
                "loss_bbox": None,
                "loss_giou": None,
                "mAP50_95": None,
                "mAP50": None,
                "AP75": None,
                "APS": None,
                "lr": None,
                "backbone_actual_lr": None,
                "non_backbone_actual_lr": None,
                "scheduler_last_epoch": None,
                "train_time_seconds": None,
                "test_time_seconds": None,
            }
            for i in range(epochs)
        ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    # Also keep machine-readable curve for budget-scale analysis.
    curve_path = output_path.with_name("learning_curve.json")
    write_json(
        curve_path,
        {
            "source": "dfine_log_txt" if log_txt.is_file() else "placeholder",
            "n_epochs_logged": len(rows),
            "epochs_requested": epochs,
            "rows": rows,
        },
    )


def _audit_val_predictions(
    solver: Any,
    *,
    device: str,
    dump_path: Path | None = None,
    coco_dump_path: Path | None = None,
    label_map: Mapping[str, Any] | None = None,
    score_threshold: float = 0.1,
    max_dump_images: int = 8,
) -> dict[str, Any]:
    """Count non-empty detections on one val pass (Gate D metric-validity signal).

    Also dumps full-val COCO detections so host-side APS_lowlight can be computed
    on the frozen slice. Does not copy AP_small / mAP into APS_lowlight.
    """
    import torch

    model_to_dataset: dict[int, int] = {}
    if isinstance(label_map, Mapping):
        raw_map = label_map.get("model_label_to_dataset_category") or {}
        if isinstance(raw_map, Mapping):
            model_to_dataset = {int(k): int(v) for k, v in raw_map.items()}

    out: dict[str, Any] = {
        "ok": False,
        "prediction_count": 0,
        "prediction_count_score_ge_0_1": 0,
        "nonzero_box_count": 0,
        "batches": 0,
        "error": None,
        "score_threshold_for_dump": score_threshold,
        "coco_detections": 0,
    }
    per_image: list[dict[str, Any]] = []
    coco_dets: list[dict[str, Any]] = []
    try:
        model = getattr(solver, "ema", None)
        module = model.module if model is not None else getattr(solver, "model", None)
        post = getattr(solver, "postprocessor", None)
        loader = getattr(solver, "val_dataloader", None)
        if module is None or post is None or loader is None:
            out["error"] = "missing_model_or_loader"
            return out
        module.eval()
        pred_n = 0
        pred_ge = 0
        box_n = 0
        batches = 0
        with torch.no_grad():
            for samples, targets in loader:
                samples = samples.to(device)
                outputs = module(samples)
                orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
                results = post(outputs, orig_sizes)
                for img_i, (result, target) in enumerate(zip(results, targets)):
                    scores = result.get("scores")
                    boxes = result.get("boxes")
                    labels = result.get("labels")
                    if scores is None:
                        continue
                    pred_n += int(scores.numel())
                    pred_ge += int((scores >= 0.1).sum().item())
                    if boxes is not None and boxes.numel():
                        wh = boxes[:, 2:] - boxes[:, :2]
                        box_n += int(((wh[:, 0] > 1e-3) & (wh[:, 1] > 1e-3)).sum().item())
                    image_id = target.get("image_id")
                    if torch.is_tensor(image_id):
                        image_id = (
                            int(image_id.item())
                            if image_id.numel() == 1
                            else int(image_id.reshape(-1)[0].item())
                        )
                    score_list = [float(x) for x in scores.detach().cpu().tolist()]
                    label_list = (
                        [int(x) for x in labels.detach().cpu().tolist()] if labels is not None else []
                    )
                    box_list = boxes.detach().cpu().tolist() if boxes is not None else []
                    for score, label, box in zip(score_list, label_list, box_list):
                        x1, y1, x2, y2 = [float(v) for v in box[:4]]
                        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                        coco_dets.append(
                            {
                                "image_id": int(image_id),
                                "category_id": int(model_to_dataset.get(int(label), int(label))),
                                "bbox": [x1, y1, w, h],
                                "score": float(score),
                            }
                        )
                    if dump_path is not None and len(per_image) < int(max_dump_images):
                        keep = scores >= float(score_threshold)
                        gt_boxes = target.get("boxes")
                        gt_n = int(gt_boxes.shape[0]) if gt_boxes is not None and hasattr(gt_boxes, "shape") else 0
                        per_image.append(
                            {
                                "image_id": image_id,
                                "batch_index": batches,
                                "image_index": img_i,
                                "gt_count": gt_n,
                                "scores": [float(x) for x in scores[keep].detach().cpu().tolist()],
                                "labels": (
                                    [int(x) for x in labels[keep].detach().cpu().tolist()]
                                    if labels is not None
                                    else []
                                ),
                                "boxes_xyxy": (
                                    boxes[keep].detach().cpu().tolist()
                                    if boxes is not None
                                    else []
                                ),
                            }
                        )
                batches += 1
        out.update(
            {
                "ok": True,
                "prediction_count": pred_n,
                "prediction_count_score_ge_0_1": pred_ge,
                "nonzero_box_count": box_n,
                "batches": batches,
                "coco_detections": len(coco_dets),
            }
        )
        if dump_path is not None:
            write_json(
                Path(dump_path),
                {
                    "predictions": per_image,
                    "n_images": len(per_image),
                    "score_threshold": score_threshold,
                    "images": per_image,
                    "summary": {
                        "prediction_count": pred_n,
                        "prediction_count_score_ge_0_1": pred_ge,
                        "nonzero_box_count": box_n,
                    },
                },
            )
            out["dump_path"] = str(dump_path)
        if coco_dump_path is not None:
            write_json(
                Path(coco_dump_path),
                {
                    "schema_version": "1.0",
                    "format": "coco_bbox_detections",
                    "n": len(coco_dets),
                    "detections": coco_dets,
                },
            )
            out["coco_dump_path"] = str(coco_dump_path)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        if dump_path is not None and not Path(dump_path).exists():
            write_json(
                Path(dump_path),
                {"predictions": [], "error": out["error"]},
            )
    return out


def _find_checkpoint(dfine_out: Path) -> Path | None:
    candidates = sorted(dfine_out.rglob("*.pth")) + sorted(dfine_out.rglob("*.pt"))
    if not candidates:
        return None
    # Prefer last/best naming when present.
    for path in candidates:
        name = path.name.lower()
        if "best" in name or "last" in name:
            return path
    return candidates[-1]


def _read_dfine_eval_metrics(dfine_out: Path) -> dict[str, float]:
    import json

    # Prefer last line of DFINE log.txt: test_coco_eval_bbox = [AP, AP50, ...]
    log_txt = Path(dfine_out) / "log.txt"
    if log_txt.is_file():
        try:
            lines = [
                ln.strip()
                for ln in log_txt.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            for line in reversed(lines):
                payload = json.loads(line)
                bbox = payload.get("test_coco_eval_bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
                    out = {
                        "mAP50_95": float(bbox[0]),
                        "mAP50": float(bbox[1]),
                    }
                    if len(bbox) >= 4:
                        out["AP_small"] = float(bbox[3])
                    return out
        except Exception:  # noqa: BLE001
            pass

    for path in sorted(dfine_out.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        # Common coco-eval style keys.
        out: dict[str, float] = {}
        for key, aliases in (
            ("mAP50_95", ("mAP", "map", "AP", "AP50_95", "mAP50_95")),
            ("mAP50", ("AP50", "mAP50", "map50")),
            ("AP_small", ("APs", "AP_small")),
        ):
            for alias in aliases:
                if alias in payload:
                    try:
                        out[key] = float(payload[alias])
                        break
                    except (TypeError, ValueError):
                        continue
        if out:
            return out
    return {}
