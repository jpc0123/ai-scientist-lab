"""Paired RGB+Thermal COCO dataset yielding 6-channel tensors for dual-stream fusion."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image


class PairedRGBTCocoDataset(torch.utils.data.Dataset):
    """Wrap a DFINE CocoDetection so each sample is ``[6, H, W]`` (RGB||Thermal).

    Spatial / photometric transforms are applied with a restored RNG state so both
    modalities share the same geometric draw (crop/flip). Photometric params also
    match because the RNG stream is replayed from the same seed state.
    """

    def __init__(
        self,
        base_dataset: Any,
        thermal_img_folder: str | Path,
    ) -> None:
        self.base = base_dataset
        self.thermal_img_folder = Path(thermal_img_folder)
        if not self.thermal_img_folder.is_dir():
            raise FileNotFoundError(
                f"thermal image folder missing: {self.thermal_img_folder}"
            )

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str) -> Any:
        # Proxy COCO / evaluator helpers (coco, ids, categories, …).
        if name in {"base", "thermal_img_folder"}:
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getitem__(self, idx: int):
        rgb_pil, target = self.base.load_item(idx)
        file_name = Path(str(target.get("image_path") or "")).name
        if not file_name:
            # Fallback via COCO ids when image_path is absent.
            image_id = int(self.base.ids[idx])
            file_name = self.base.coco.loadImgs(image_id)[0]["file_name"]
        thermal_path = self.thermal_img_folder / file_name
        if not thermal_path.is_file():
            raise FileNotFoundError(f"missing thermal pair: {thermal_path}")
        thermal_pil = Image.open(thermal_path).convert("RGB")
        if thermal_pil.size != rgb_pil.size:
            thermal_pil = thermal_pil.resize(rgb_pil.size, Image.BILINEAR)

        transforms = getattr(self.base, "_transforms", None)
        if transforms is None:
            rgb_t = _pil_to_tensor(rgb_pil)
            thr_t = _pil_to_tensor(thermal_pil)
            return torch.cat([rgb_t, thr_t], dim=0), target

        py_state = random.getstate()
        torch_state = torch.get_rng_state()
        numpy_state = None
        try:
            import numpy as np

            numpy_state = np.random.get_state()
        except Exception:  # noqa: BLE001
            numpy_state = None

        rgb_out, target_out, _ = transforms(rgb_pil, target, self.base)

        random.setstate(py_state)
        torch.set_rng_state(torch_state)
        if numpy_state is not None:
            import numpy as np

            np.random.set_state(numpy_state)

        thr_out, _, _ = transforms(thermal_pil, copy.deepcopy(target), self.base)
        if not torch.is_tensor(rgb_out) or not torch.is_tensor(thr_out):
            raise TypeError(
                "paired dataset expects tensor outputs from transforms; "
                f"got {type(rgb_out)} / {type(thr_out)}"
            )
        if rgb_out.shape != thr_out.shape:
            raise ValueError(
                "RGB/Thermal tensor shapes diverge after transforms: "
                f"{tuple(rgb_out.shape)} vs {tuple(thr_out.shape)}"
            )
        return torch.cat([rgb_out, thr_out], dim=0), target_out


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    import torchvision.transforms.functional as F

    return F.to_tensor(image)


def wrap_loader_dataset_with_thermal(
    loader: Any,
    thermal_img_folder: str | Path,
) -> Any:
    """Replace ``loader.dataset`` with a paired wrapper (in-place).

    PyTorch ≥2.6 forbids ``loader.dataset = ...`` after init; use
    ``object.__setattr__`` so DFINE's existing DataLoader (collate / set_epoch)
    stays intact.
    """
    dataset = loader.dataset
    # Some wrappers nest .dataset (Subset / Distributed).
    if hasattr(dataset, "dataset") and not hasattr(dataset, "load_item"):
        inner = dataset.dataset
        if hasattr(inner, "load_item"):
            object.__setattr__(
                dataset,
                "dataset",
                PairedRGBTCocoDataset(inner, thermal_img_folder),
            )
            return loader
    if not hasattr(dataset, "load_item"):
        raise TypeError(
            f"cannot wrap dataset type {type(dataset)!r}; expected CocoDetection"
        )
    object.__setattr__(loader, "dataset", PairedRGBTCocoDataset(dataset, thermal_img_folder))
    return loader
