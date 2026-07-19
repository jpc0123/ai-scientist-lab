"""Tiny CPU detector: small linear head over pooled features (NumPy)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TinyDetector:
    """Single-object smoke detector with explicit forward/backward."""

    in_channels: int
    num_classes: int
    feature_dim: int
    image_width: int
    image_height: int
    W1: np.ndarray
    b1: np.ndarray
    W_cls: np.ndarray
    b_cls: np.ndarray
    W_box: np.ndarray
    b_box: np.ndarray

    @classmethod
    def create(
        cls,
        *,
        in_channels: int,
        num_classes: int = 2,
        feature_dim: int = 64,
        image_width: int = 160,
        image_height: int = 128,
        seed: int = 42,
    ) -> "TinyDetector":
        rng = np.random.default_rng(seed)
        # 4x4 spatial pool => 16 cells
        in_dim = in_channels * 16
        scale = 1.0 / np.sqrt(in_dim)
        return cls(
            in_channels=in_channels,
            num_classes=num_classes,
            feature_dim=feature_dim,
            image_width=image_width,
            image_height=image_height,
            W1=rng.normal(0, scale, size=(in_dim, feature_dim)).astype(np.float32),
            b1=np.zeros(feature_dim, dtype=np.float32),
            W_cls=rng.normal(0, 0.05, size=(feature_dim, num_classes)).astype(np.float32),
            b_cls=np.zeros(num_classes, dtype=np.float32),
            W_box=rng.normal(0, 0.05, size=(feature_dim, 4)).astype(np.float32),
            b_box=np.zeros(4, dtype=np.float32),
        )

    def _pool(self, image: np.ndarray) -> np.ndarray:
        # image: H,W,C -> 4x4 mean pool -> flatten
        h, w, c = image.shape
        gh, gw = h // 4, w // 4
        feats = []
        for iy in range(4):
            for ix in range(4):
                patch = image[iy * gh : (iy + 1) * gh, ix * gw : (ix + 1) * gw, :]
                feats.append(patch.mean(axis=(0, 1)))
        return np.concatenate(feats, axis=0).astype(np.float32)

    def forward(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        x = self._pool(image)
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W_cls + self.b_cls
        box = 1.0 / (1.0 + np.exp(-(h @ self.W_box + self.b_box)))  # normalized xywh
        cache = {"x": x, "h": h, "logits": logits, "box": box}
        return logits, box, cache

    def loss_and_backward(
        self,
        image: np.ndarray,
        target_box: np.ndarray,
        target_label: int,
        *,
        lr: float,
    ) -> tuple[float, dict[str, float]]:
        logits, box_pred, cache = self.forward(image)
        # class CE
        logits = logits - logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        label = int(np.clip(target_label - 1, 0, self.num_classes - 1))  # coco id 1.. -> 0..
        cls_loss = float(-np.log(probs[label] + 1e-8))

        # target box normalized
        tw = max(float(self.image_width), 1.0)
        th = max(float(self.image_height), 1.0)
        tgt = np.asarray(
            [
                target_box[0] / tw,
                target_box[1] / th,
                target_box[2] / tw,
                target_box[3] / th,
            ],
            dtype=np.float32,
        )
        tgt = np.clip(tgt, 1e-3, 1.0 - 1e-3)
        box_loss = float(np.mean((box_pred - tgt) ** 2))
        loss = cls_loss + box_loss
        if not np.isfinite(loss):
            raise RuntimeError("nan_loss")

        # gradients
        dlogits = probs.copy()
        dlogits[label] -= 1.0
        h = cache["h"]
        x = cache["x"]
        dW_cls = np.outer(h, dlogits)
        db_cls = dlogits
        dh = self.W_cls @ dlogits

        # sigmoid box grad
        s = box_pred
        dbox = (2.0 / 4.0) * (box_pred - tgt)
        dpre = dbox * s * (1.0 - s)
        dW_box = np.outer(h, dpre)
        db_box = dpre
        dh = dh + self.W_box @ dpre

        # tanh hidden
        dpre_h = dh * (1.0 - h**2)
        dW1 = np.outer(x, dpre_h)
        db1 = dpre_h

        self.W_cls -= lr * dW_cls.astype(np.float32)
        self.b_cls -= lr * db_cls.astype(np.float32)
        self.W_box -= lr * dW_box.astype(np.float32)
        self.b_box -= lr * db_box.astype(np.float32)
        self.W1 -= lr * dW1.astype(np.float32)
        self.b1 -= lr * db1.astype(np.float32)

        return loss, {"cls_loss": cls_loss, "box_loss": box_loss}

    def predict(self, image: np.ndarray) -> tuple[int, np.ndarray, float]:
        logits, box, _ = self.forward(image)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        cls = int(np.argmax(probs)) + 1
        score = float(probs.max())
        tw = float(self.image_width)
        th = float(self.image_height)
        abs_box = np.asarray(
            [box[0] * tw, box[1] * th, box[2] * tw, box[3] * th], dtype=np.float32
        )
        return cls, abs_box, score

    def parameter_count(self) -> int:
        return int(
            self.W1.size
            + self.b1.size
            + self.W_cls.size
            + self.b_cls.size
            + self.W_box.size
            + self.b_box.size
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            image_width=self.image_width,
            image_height=self.image_height,
            W1=self.W1,
            b1=self.b1,
            W_cls=self.W_cls,
            b_cls=self.b_cls,
            W_box=self.W_box,
            b_box=self.b_box,
        )

    @classmethod
    def load(cls, path: Path) -> "TinyDetector":
        data = np.load(path, allow_pickle=False)
        return cls(
            in_channels=int(data["in_channels"]),
            num_classes=int(data["num_classes"]),
            feature_dim=int(data["feature_dim"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            W1=data["W1"].astype(np.float32),
            b1=data["b1"].astype(np.float32),
            W_cls=data["W_cls"].astype(np.float32),
            b_cls=data["b_cls"].astype(np.float32),
            W_box=data["W_box"].astype(np.float32),
            b_box=data["b_box"].astype(np.float32),
        )

    def summary(self, *, input_mode: str, fusion_method: str) -> dict[str, Any]:
        return {
            "model": "tiny_detector",
            "backend": "numpy",
            "input_mode": input_mode,
            "fusion_method": fusion_method,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "parameter_count": self.parameter_count(),
            "trainable_parameter_count": self.parameter_count(),
            "estimated_flops": None,
        }


def channels_for_mode(input_mode: str, fusion_method: str) -> int:
    if input_mode == "rgbt" and fusion_method in {"early_concat", "concat", "early"}:
        return 4
    return 3
