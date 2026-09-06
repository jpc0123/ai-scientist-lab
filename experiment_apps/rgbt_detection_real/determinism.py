"""Deterministic training helpers for Gate J3 diagnosis."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


def reseed_all(seed: int) -> None:
    """Reseed Python / NumPy / Torch RNGs (used at init and each epoch)."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_determinism(seed: int) -> dict[str, Any]:
    """Lock major RNG sources before model/dataloader construction."""
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    # Must be set before first CUDA context when using deterministic algorithms.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    reseed_all(seed)

    torch.backends.cudnn.benchmark = False
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True

    # Reduce TF32 nondeterminism on Ampere+ (common residual drift source).
    tf32_disabled = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        tf32_disabled = True
    except Exception:  # noqa: BLE001
        tf32_disabled = False

    # DFINE uses ops without deterministic CUDA impls (e.g. grid_sampler_2d_backward).
    # Strict mode crashes mid-backward; warn_only keeps seeded RNG locks usable.
    det_algo_error: str | None = None
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        det_algo_ok = False  # warn_only by design for DFINE diagnosis
        det_algo_error = (
            "warn_only=True required: DFINE/grid_sampler lacks deterministic CUDA backward"
        )
    except Exception as exc:  # noqa: BLE001
        det_algo_ok = False
        det_algo_error = f"{type(exc).__name__}: {exc}"

    return {
        "seed": int(seed),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "tf32_disabled": tf32_disabled,
        "deterministic_algorithms_strict": det_algo_ok,
        "deterministic_algorithms_warn_only": True,
        "deterministic_algorithms_error": det_algo_error,
    }


def install_epoch_reseed(solver: Any, seed: int) -> None:
    """Reseed RNGs at the start of every training epoch (Gate J3b)."""
    original_fit = solver.fit

    def fit_with_reseed(*args: Any, **kwargs: Any) -> Any:
        train_loader = getattr(solver, "train_dataloader", None)
        set_epoch = getattr(train_loader, "set_epoch", None) if train_loader else None
        if callable(set_epoch):
            def set_epoch_reseed(epoch: int, *a: Any, **kw: Any) -> Any:
                # Distinct but deterministic stream per epoch.
                reseed_all(int(seed) + int(epoch) * 1009)
                return set_epoch(epoch, *a, **kw)

            train_loader.set_epoch = set_epoch_reseed  # type: ignore[method-assign]
        return original_fit(*args, **kwargs)

    solver.fit = fit_with_reseed  # type: ignore[method-assign]


def seed_worker(worker_id: int) -> None:
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _dataset_image_id(dataset: Any, index: int) -> str:
    try:
        ids = getattr(dataset, "ids", None)
        if ids is not None:
            image_id = ids[index]
            coco = getattr(dataset, "coco", None)
            if coco is not None:
                meta = coco.loadImgs(image_id)[0]
                return str(meta.get("file_name") or image_id)
            return str(image_id)
    except Exception:  # noqa: BLE001
        pass
    return str(index)


def build_epoch0_order_audit(*, dataset: Any, seed: int, batch_size: int) -> dict[str, Any]:
    """Compute epoch-0 sample order from a freshly seeded RandomSampler (no consume)."""
    import torch
    from torch.utils.data import RandomSampler

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = RandomSampler(dataset, generator=generator)
    order_idx = list(sampler)
    sample_ids = [_dataset_image_id(dataset, i) for i in order_idx]
    order_hash = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()

    batch_ids: list[list[str]] = []
    bs = max(1, int(batch_size))
    for start in range(0, min(len(sample_ids), bs * 20), bs):
        batch_ids.append(sample_ids[start : start + bs])

    return {
        "epoch": 0,
        "seed": int(seed),
        "num_samples": len(sample_ids),
        "first_100_sample_ids": sample_ids[:100],
        "first_20_batch_ids": batch_ids[:20],
        "order_hash": f"sha256:{order_hash}",
        "augmentation_hash": "sha256:noop_resize_only_or_multiscale_disabled",
        "sampler": "RandomSampler+Generator",
        "note": "Order derived from seeded RandomSampler before training; training loader uses a fresh Generator with the same seed.",
    }


def rebuild_train_loader_seeded(
    *,
    yaml_cfg: Any,
    seed: int,
    num_workers: int = 0,
) -> Any:
    """Replace train dataloader with an explicitly seeded Generator/worker_init_fn."""
    import torch
    from src.data import DataLoader  # type: ignore

    old = yaml_cfg.train_dataloader
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        old.dataset,
        batch_size=old.batch_size,
        shuffle=True,
        num_workers=int(num_workers),
        drop_last=getattr(old, "drop_last", False),
        collate_fn=old.collate_fn,
        pin_memory=getattr(old, "pin_memory", False),
        worker_init_fn=seed_worker if int(num_workers) > 0 else None,
        generator=generator,
    )
    loader.shuffle = True
    # YAMLConfig.train_dataloader is a read-only @property (no setter); BaseConfig
    # stores the instance on _train_dataloader. Assign the private field directly.
    yaml_cfg._train_dataloader = loader
    return loader


def write_determinism_artifacts(
    output_dir: Path,
    *,
    configure_report: dict[str, Any],
    order_audit: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "configure": configure_report,
        "epoch0_order": order_audit,
        **(extra or {}),
    }
    (output_dir / "determinism_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "epoch_sample_order_epoch0.json").write_text(
        json.dumps(order_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
