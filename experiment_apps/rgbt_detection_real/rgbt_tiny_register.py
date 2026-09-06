"""Gate F0 registrar for RGBT-Tiny: path-reference registration without image copies.

Discovers common raw layouts, pairs RGB/Thermal by sequence+frame, converts
annotations to unified COCO, performs sequence-level splits, audits quality,
and freezes manifests under datasets/registered/rgbt_tiny_v1/.

Images stay in the raw tree. Registered image dirs use hardlinks (preferred)
or symlinks — never shutil.copy of pixel data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Paper: ship, car, cyclist, pedestrian, bus, drone, plane
CANONICAL_CATEGORIES = [
    "ship",
    "car",
    "cyclist",
    "pedestrian",
    "bus",
    "drone",
    "plane",
]

RGB_DIR_NAMES = ("visible", "rgb", "vis", "Visible", "RGB", "VIS")
THR_DIR_NAMES = ("infrared", "thermal", "ir", "Infrared", "Thermal", "IR", "lwir")
ANN_DIR_NAMES = ("annotations", "labels", "gt", "Annotations", "Labels", "GT")


@dataclass
class FramePair:
    sequence_id: str
    frame_id: str
    rgb_path: Path
    thermal_path: Path
    sample_id: str  # globally unique basename stem


@dataclass
class BoxAnn:
    bbox: list[float]  # xywh
    category_name: str
    track_id: int | None = None
    area: float | None = None
    iscrowd: int = 0


@dataclass
class DiscoverResult:
    raw_root: Path
    layout: str
    sequences: dict[str, list[FramePair]] = field(default_factory=dict)
    # sequence -> frame_id -> list[BoxAnn]  (primary modality annotations)
    annotations: dict[str, dict[str, list[BoxAnn]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        remaining = max_bytes
        while True:
            chunk_size = 1024 * 1024
            if remaining is not None:
                if remaining <= 0:
                    break
                chunk_size = min(chunk_size, remaining)
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def resolve_raw_data_root(configured: Path) -> Path | None:
    """Pick the real data root under configured raw path (may be nested)."""
    configured = Path(configured)
    if not configured.exists():
        return None
    candidates: list[Path] = []
    if _looks_like_rgbt_root(configured):
        candidates.append(configured)
    for child in sorted(configured.iterdir()) if configured.is_dir() else []:
        if child.is_dir() and _looks_like_rgbt_root(child):
            candidates.append(child)
        nested = child / "RGBT-Tiny"
        if nested.is_dir() and _looks_like_rgbt_root(nested):
            candidates.append(nested)
    if not candidates:
        return configured if configured.is_dir() else None

    def _score(path: Path) -> tuple[int, int, str]:
        # Prefer official package root (annotations + images), not the images/ subfolder.
        has_ann = int(
            (path / "annotations_coco").is_dir()
            or (path / "annotations").is_dir()
            or (path / "data_split").is_dir()
        )
        has_images = int((path / "images").is_dir())
        # Higher is better; break ties by shorter path (package root over nested)
        return (has_ann, has_images, -len(path.parts), str(path))

    return max(candidates, key=_score)


def _looks_like_rgbt_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {p.name.lower() for p in path.iterdir()}
    # Official release: images/<seq>/{00,01} + annotations_coco|data_split
    images = path / "images"
    has_meta = (
        (path / "annotations_coco").is_dir()
        or (path / "annotations").is_dir()
        or (path / "data_split").is_dir()
    )
    if images.is_dir() and has_meta:
        for child in list(images.iterdir())[:20]:
            if not child.is_dir():
                continue
            sub = {p.name for p in child.iterdir() if p.is_dir()}
            if "00" in sub and "01" in sub:
                return True
        return True  # still treat as package root even before sampling seqs
    has_rgb = bool(names & {n.lower() for n in RGB_DIR_NAMES})
    has_thr = bool(names & {n.lower() for n in THR_DIR_NAMES})
    if has_rgb and has_thr:
        return True
    # sequence folders each containing named rgb+thermal (not bare 00/01-only trees)
    seq_hits = 0
    for child in list(path.iterdir())[:30]:
        if not child.is_dir():
            continue
        sub = {p.name.lower() for p in child.iterdir()} if any(child.iterdir()) else set()
        if (sub & {n.lower() for n in RGB_DIR_NAMES}) and (
            sub & {n.lower() for n in THR_DIR_NAMES}
        ):
            seq_hits += 1
    return seq_hits >= 1


def _pick_subdir(parent: Path, names: Iterable[str]) -> Path | None:
    lower_map = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for name in names:
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None


def _list_images(directory: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            out[path.stem] = path
    return out


def _normalize_category(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())
    aliases = {
        "ship": "ship",
        "boat": "ship",
        "vessel": "ship",
        "car": "car",
        "vehicle": "car",
        "cyclist": "cyclist",
        "bike": "cyclist",
        "bicycle": "cyclist",
        "pedestrian": "pedestrian",
        "person": "pedestrian",
        "people": "pedestrian",
        "bus": "bus",
        "drone": "drone",
        "uav": "drone",
        "plane": "plane",
        "airplane": "plane",
        "aircraft": "plane",
    }
    return aliases.get(key, str(name).strip().lower() or "unknown")


def discover_raw(raw_root: Path) -> DiscoverResult:
    """Discover pairs + annotations under an RGBT-Tiny-like tree."""
    raw_root = Path(raw_root)
    notes: list[str] = []

    # Layout Official: root/images/<seq>/{00=visible,01=thermal}/<frames>
    images_root = raw_root / "images"
    if images_root.is_dir():
        sequences = _pair_official_00_01(images_root)
        if sequences:
            notes.append(
                f"official 00/01 layout: {len(sequences)} sequences "
                "(COCO load deferred until subset selection)"
            )
            return DiscoverResult(
                raw_root=raw_root,
                layout="official_images_seq_00_01",
                sequences=sequences,
                annotations={},
                notes=notes,
            )

    # Layout A: root/{visible,infrared}/<seq>/<frames>
    rgb_top = _pick_subdir(raw_root, RGB_DIR_NAMES)
    thr_top = _pick_subdir(raw_root, THR_DIR_NAMES)
    if rgb_top and thr_top:
        sequences = _pair_modality_trees(rgb_top, thr_top)
        anns = _load_annotations_near(raw_root, sequences, notes)
        return DiscoverResult(
            raw_root=raw_root,
            layout="modality_then_sequence",
            sequences=sequences,
            annotations=anns,
            notes=notes,
        )

    # Layout B: root/<seq>/{visible,infrared}/
    sequences = {}
    for seq_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        rgb_d = _pick_subdir(seq_dir, RGB_DIR_NAMES)
        thr_d = _pick_subdir(seq_dir, THR_DIR_NAMES)
        if not rgb_d or not thr_d:
            continue
        pairs = _pair_two_dirs(seq_dir.name, rgb_d, thr_d)
        if pairs:
            sequences[seq_dir.name] = pairs
    if sequences:
        anns = _load_annotations_near(raw_root, sequences, notes)
        return DiscoverResult(
            raw_root=raw_root,
            layout="sequence_then_modality",
            sequences=sequences,
            annotations=anns,
            notes=notes,
        )

    notes.append("no recognizable RGB/Thermal sequence layout")
    return DiscoverResult(raw_root=raw_root, layout="unknown", notes=notes)


def _pair_official_00_01(images_root: Path) -> dict[str, list[FramePair]]:
    """Pair frames under images/<seq>/00 (RGB) and images/<seq>/01 (thermal)."""
    out: dict[str, list[FramePair]] = {}
    for seq_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        rgb_d = seq_dir / "00"
        thr_d = seq_dir / "01"
        if not rgb_d.is_dir() or not thr_d.is_dir():
            # also accept named modality folders under sequence
            rgb_d = _pick_subdir(seq_dir, RGB_DIR_NAMES) or rgb_d
            thr_d = _pick_subdir(seq_dir, THR_DIR_NAMES) or thr_d
        if not rgb_d.is_dir() or not thr_d.is_dir():
            continue
        pairs = _pair_two_dirs(seq_dir.name, rgb_d, thr_d)
        if pairs:
            out[seq_dir.name] = pairs
    return out


def _load_official_rgbt_tiny_coco(
    raw_root: Path,
    sequences: dict[str, list[FramePair]],
    notes: list[str],
    *,
    prefer_modality: str = "01",
) -> dict[str, dict[str, list[BoxAnn]]]:
    """Load official annotations_coco instances_{00|01}_{train|test}2017.json."""
    out: dict[str, dict[str, list[BoxAnn]]] = defaultdict(lambda: defaultdict(list))
    ann_root = raw_root / "annotations_coco"
    if not ann_root.is_dir():
        ann_root = raw_root / "annotations"
    if not ann_root.is_dir():
        notes.append("official annotations_coco missing")
        return out

    # Prefer thermal (01), fall back to visible (00)
    modality_order = [prefer_modality, "00" if prefer_modality == "01" else "01"]
    chosen_files: list[Path] = []
    for mod in modality_order:
        files = sorted(ann_root.glob(f"instances_{mod}_*2017.json"))
        if files:
            chosen_files = files
            notes.append(f"using modality={mod} COCO files: {[p.name for p in files]}")
            break
    if not chosen_files:
        # generic fallback
        chosen_files = sorted(ann_root.glob("instances_*.json"))
        notes.append(f"fallback COCO files: {[p.name for p in chosen_files]}")

    for path in chosen_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"failed to read {path.name}: {exc}")
            continue
        _ingest_coco_official(payload, sequences, out, notes, source=path)
    return out


def _ingest_coco_official(
    payload: dict[str, Any],
    sequences: dict[str, list[FramePair]],
    out: dict[str, dict[str, list[BoxAnn]]],
    notes: list[str],
    *,
    source: Path,
) -> None:
    """Match file_name like 'DJI_0022_1/01/00000.jpg' to (seq, frame)."""
    cats = {
        int(c["id"]): _normalize_category(c.get("name", str(c["id"])))
        for c in payload.get("categories") or []
    }
    images = {int(img["id"]): img for img in payload.get("images") or []}
    matched = 0
    for ann in payload.get("annotations") or []:
        img = images.get(int(ann["image_id"]))
        if not img:
            continue
        file_name = str(img.get("file_name") or "").replace("\\", "/")
        parts = Path(file_name).parts
        # expected: seq / modality / frame.ext
        if len(parts) >= 3:
            seq, _mod, frame = parts[-3], parts[-2], Path(parts[-1]).stem
        elif len(parts) == 2:
            seq, frame = parts[0], Path(parts[1]).stem
        else:
            continue
        if seq not in sequences:
            continue
        bbox = [float(x) for x in ann.get("bbox") or []]
        if len(bbox) != 4:
            continue
        cat = cats.get(int(ann.get("category_id", -1)), "unknown")
        out[seq][frame].append(
            BoxAnn(
                bbox=bbox,
                category_name=cat,
                track_id=ann.get("track_id") or ann.get("instance_id"),
                area=float(ann["area"]) if ann.get("area") is not None else bbox[2] * bbox[3],
                iscrowd=int(ann.get("iscrowd") or 0),
            )
        )
        matched += 1
    notes.append(f"{source.name}: matched {matched} annotation rows")


def load_official_sequence_split(
    raw_root: Path,
) -> tuple[dict[str, list[str]], list[str]] | tuple[None, list[str]]:
    """Parse data_split/train.txt & test.txt into sequence-level membership."""
    split_dir = Path(raw_root) / "data_split"
    train_file = split_dir / "train.txt"
    test_file = split_dir / "test.txt"
    if not train_file.is_file() or not test_file.is_file():
        return None, []

    def _seqs(path: Path) -> set[str]:
        out: set[str] = set()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip().replace("\\", "/")
            if not line:
                continue
            out.add(line.split("/")[0])
        return out

    train_seqs = _seqs(train_file)
    test_seqs = _seqs(test_file)
    leak = sorted(train_seqs & test_seqs)
    return {
        "train": sorted(train_seqs - test_seqs),
        "test": sorted(test_seqs - train_seqs),
    }, leak


def carve_val_from_train(
    split_map: dict[str, list[str]],
    *,
    seed: int,
    val_ratio: float = 0.15,
) -> dict[str, list[str]]:
    """Keep official test; move a sequence subset of train → val (no frame leakage)."""
    rng = random.Random(seed)
    train = list(split_map.get("train") or [])
    test = list(split_map.get("test") or [])
    rng.shuffle(train)
    if len(train) <= 2:
        return {"train": train, "val": [], "test": test}
    n_val = max(1, int(round(len(train) * val_ratio)))
    n_val = min(n_val, len(train) - 1)
    val = sorted(train[:n_val])
    train = sorted(train[n_val:])
    return {"train": train, "val": val, "test": sorted(test)}


def subset_within_splits(
    split_map: dict[str, list[str]],
    available: dict[str, list[FramePair]],
    *,
    max_sequences: int | None,
    max_frames_per_seq: int | None,
    frame_stride: int,
    seed: int,
    train_sequences: int | None = None,
    val_sequences: int | None = None,
    test_sequences: int | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[FramePair]]]:
    """Keep sequence-level split integrity while capping total sequences/frames."""
    rng = random.Random(seed)
    if max_sequences is None and train_sequences is None:
        quotas = {k: len(v) for k, v in split_map.items()}
    else:
        if train_sequences is not None or val_sequences is not None or test_sequences is not None:
            quotas = {
                "train": int(train_sequences or 0),
                "val": int(val_sequences or 0),
                "test": int(test_sequences or 0),
            }
        else:
            # Prefer more train: ~70/15/15 of the budget
            budget = int(max_sequences or 0)
            quotas = {
                "train": max(1, int(round(budget * 0.7))),
                "val": max(1, int(round(budget * 0.15))),
                "test": max(
                    0,
                    budget
                    - max(1, int(round(budget * 0.7)))
                    - max(1, int(round(budget * 0.15))),
                ),
            }
        for k in list(quotas):
            avail = [s for s in split_map.get(k, []) if s in available]
            quotas[k] = min(quotas[k], len(avail))

    chosen_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    subset: dict[str, list[FramePair]] = {}
    stride = max(1, int(frame_stride))
    for split, quota in quotas.items():
        cand = [s for s in split_map.get(split, []) if s in available]
        rng.shuffle(cand)
        picked = cand[:quota]
        chosen_split[split] = sorted(picked)
        for seq in picked:
            frames = list(available[seq])[::stride]
            if max_frames_per_seq is not None:
                frames = frames[: max(0, int(max_frames_per_seq))]
            if frames:
                subset[seq] = frames
    return chosen_split, subset


def _pair_modality_trees(rgb_top: Path, thr_top: Path) -> dict[str, list[FramePair]]:
    rgb_seqs = {p.name: p for p in rgb_top.iterdir() if p.is_dir()}
    thr_seqs = {p.name: p for p in thr_top.iterdir() if p.is_dir()}
    common = sorted(set(rgb_seqs) & set(thr_seqs))
    out: dict[str, list[FramePair]] = {}
    for seq in common:
        pairs = _pair_two_dirs(seq, rgb_seqs[seq], thr_seqs[seq])
        if pairs:
            out[seq] = pairs
    return out


def _pair_two_dirs(seq: str, rgb_dir: Path, thr_dir: Path) -> list[FramePair]:
    rgb_map = _list_images(rgb_dir)
    thr_map = _list_images(thr_dir)
    frames = sorted(set(rgb_map) & set(thr_map))
    pairs: list[FramePair] = []
    for frame in frames:
        sample_id = f"{_safe_token(seq)}__{_safe_token(frame)}"
        pairs.append(
            FramePair(
                sequence_id=seq,
                frame_id=frame,
                rgb_path=rgb_map[frame],
                thermal_path=thr_map[frame],
                sample_id=sample_id,
            )
        )
    return pairs


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _load_annotations_near(
    raw_root: Path,
    sequences: dict[str, list[FramePair]],
    notes: list[str],
) -> dict[str, dict[str, list[BoxAnn]]]:
    """Load COCO / DarkLabel-like annotations if present; prefer thermal then visible."""
    ann_root = None
    for name in ANN_DIR_NAMES:
        cand = raw_root / name
        if cand.is_dir():
            ann_root = cand
            break
    out: dict[str, dict[str, list[BoxAnn]]] = defaultdict(lambda: defaultdict(list))
    if ann_root is None:
        notes.append("no annotations directory found; boxes will be empty until labels exist")
        return out

    # Prefer thermal COCO, then visible, then any
    coco_candidates = sorted(ann_root.rglob("*.json"))
    preferred: list[Path] = []
    rest: list[Path] = []
    for path in coco_candidates:
        low = str(path).lower()
        if any(k in low for k in ("thermal", "infrared", "ir", "lwir")):
            preferred.append(path)
        elif any(k in low for k in ("visible", "rgb", "vis")):
            rest.insert(0, path)
        else:
            rest.append(path)
    loaded_any = False
    for path in preferred + rest:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict) or "images" not in payload:
            continue
        _ingest_coco(payload, sequences, out, notes, source=path)
        loaded_any = True
        notes.append(f"loaded COCO annotations from {path}")
        break  # primary modality only for fusion GT

    if not loaded_any:
        # DarkLabel / MOT txt: annotations/<seq>/*.txt or <seq>.txt
        txt_hits = 0
        for seq, pairs in sequences.items():
            frame_index = {p.frame_id: p for p in pairs}
            for txt in list(ann_root.rglob(f"*{seq}*"))[:200]:
                if txt.suffix.lower() not in {".txt", ".csv"}:
                    continue
                txt_hits += _ingest_txt_file(txt, seq, frame_index, out)
        if txt_hits:
            notes.append(f"loaded {txt_hits} txt annotation rows")
        else:
            notes.append("annotation files present but unrecognized; boxes empty")
    return out


def _ingest_coco(
    payload: dict[str, Any],
    sequences: dict[str, list[FramePair]],
    out: dict[str, dict[str, list[BoxAnn]]],
    notes: list[str],
    *,
    source: Path,
) -> None:
    cats = {
        int(c["id"]): _normalize_category(c.get("name", str(c["id"])))
        for c in payload.get("categories") or []
    }
    # Map file_name / id -> (seq, frame)
    stem_to_seq_frame: dict[str, tuple[str, str]] = {}
    for seq, pairs in sequences.items():
        for pair in pairs:
            stem_to_seq_frame[pair.frame_id] = (seq, pair.frame_id)
            stem_to_seq_frame[pair.sample_id] = (seq, pair.frame_id)
            stem_to_seq_frame[pair.rgb_path.stem] = (seq, pair.frame_id)
            # common: seq/frame or seq_frame in file_name
            stem_to_seq_frame[f"{seq}_{pair.frame_id}"] = (seq, pair.frame_id)

    images = {int(img["id"]): img for img in payload.get("images") or []}
    matched = 0
    for ann in payload.get("annotations") or []:
        img = images.get(int(ann["image_id"]))
        if not img:
            continue
        file_name = str(img.get("file_name") or "").replace("\\", "/")
        key = None
        # Prefer path that embeds sequence (avoids colliding frame stems across seqs)
        parts = Path(file_name).parts
        if len(parts) >= 2:
            seq_guess, frame_guess = parts[-2], Path(parts[-1]).stem
            if seq_guess in sequences:
                key = (seq_guess, frame_guess)
        if key is None:
            stem = Path(file_name).stem
            key = stem_to_seq_frame.get(stem)
            # disambiguate only if unique across sequences
            if key is not None:
                hits = [
                    (seq, pair.frame_id)
                    for seq, pairs in sequences.items()
                    for pair in pairs
                    if pair.frame_id == stem
                ]
                if len(hits) > 1:
                    key = None
        if key is None:
            continue
        seq, frame = key
        bbox = [float(x) for x in ann.get("bbox") or []]
        if len(bbox) != 4:
            continue
        cat = cats.get(int(ann.get("category_id", -1)), "unknown")
        out[seq][frame].append(
            BoxAnn(
                bbox=bbox,
                category_name=cat,
                track_id=ann.get("track_id") or ann.get("instance_id"),
                area=float(ann["area"]) if ann.get("area") is not None else bbox[2] * bbox[3],
                iscrowd=int(ann.get("iscrowd") or 0),
            )
        )
        matched += 1
    notes.append(f"{source.name}: matched {matched} annotation rows to frames")


def _ingest_txt_file(
    path: Path,
    seq: str,
    frame_index: dict[str, FramePair],
    out: dict[str, dict[str, list[BoxAnn]]],
) -> int:
    """Best-effort DarkLabel/MOT-ish parser: frame,id,x,y,w,h,class,..."""
    rows = 0
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\s,]+", line)
        if len(parts) < 6:
            continue
        try:
            # MOT: frame, id, x, y, w, h, ...
            frame_token = parts[0]
            x, y, w, h = map(float, parts[2:6])
            cls = parts[6] if len(parts) > 6 else "unknown"
            # if class is numeric keep as string id
        except ValueError:
            continue
        # resolve frame by numeric index or stem match
        frame_id = None
        if frame_token in frame_index:
            frame_id = frame_token
        else:
            # zero-padded stems
            for cand in (
                frame_token,
                f"{int(float(frame_token)):06d}" if frame_token.replace(".", "", 1).isdigit() else None,
                f"{int(float(frame_token)):05d}" if frame_token.replace(".", "", 1).isdigit() else None,
            ):
                if cand and cand in frame_index:
                    frame_id = cand
                    break
        if frame_id is None:
            continue
        out[seq][frame_id].append(
            BoxAnn(
                bbox=[x, y, w, h],
                category_name=_normalize_category(cls),
                track_id=int(float(parts[1])) if parts[1].replace(".", "", 1).isdigit() else None,
                area=w * h,
            )
        )
        rows += 1
    return rows


def select_subset(
    sequences: dict[str, list[FramePair]],
    *,
    max_sequences: int | None,
    max_frames_per_seq: int | None,
    frame_stride: int,
    seed: int,
) -> dict[str, list[FramePair]]:
    rng = random.Random(seed)
    seq_ids = sorted(sequences)
    rng.shuffle(seq_ids)
    if max_sequences is not None:
        seq_ids = seq_ids[: max(0, int(max_sequences))]
    stride = max(1, int(frame_stride))
    out: dict[str, list[FramePair]] = {}
    for seq in seq_ids:
        frames = list(sequences[seq])[::stride]
        if max_frames_per_seq is not None:
            frames = frames[: max(0, int(max_frames_per_seq))]
        if frames:
            out[seq] = frames
    return out


def split_sequences(
    seq_ids: list[str],
    *,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[str]]:
    """Sequence-level split — never split frames of one sequence across sets."""
    rng = random.Random(seed)
    ids = list(seq_ids)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": ids, "val": [], "test": []}
    if n == 2:
        return {"train": [ids[0]], "val": [ids[1]], "test": []}
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1) if n > 2 else 1
        n_train = max(1, n - n_val - (1 if n > n_train + n_val else 0))
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    if not test and len(val) > 1:
        test = [val.pop()]
    return {"train": train, "val": val, "test": test}


def link_file(src: Path, dst: Path) -> str:
    """Create hardlink (same volume) or symlink; never copy pixels."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        try:
            os.symlink(src, dst)
            return "symlink"
        except OSError as exc:
            raise RuntimeError(
                f"cannot hardlink/symlink {src} -> {dst}; refusing to copy images ({exc})"
            ) from exc


def build_category_mapping(
    used_names: Iterable[str],
) -> dict[str, Any]:
    names = []
    for n in CANONICAL_CATEGORIES:
        names.append(n)
    for n in sorted(set(used_names)):
        nn = _normalize_category(n)
        if nn not in names and nn != "unknown":
            names.append(nn)
    if "unknown" in {_normalize_category(x) for x in used_names}:
        names.append("unknown")
    # COCO 1-based ids for registered JSON; model map is 0-based
    coco_categories = [{"id": i + 1, "name": name} for i, name in enumerate(names)]
    name_to_coco = {c["name"]: c["id"] for c in coco_categories}
    dataset_to_model = {str(c["id"]): i for i, c in enumerate(coco_categories)}
    model_to_dataset = {str(i): c["id"] for i, c in enumerate(coco_categories)}
    return {
        "categories": coco_categories,
        "name_to_coco_id": name_to_coco,
        "dataset_category_to_model_label": dataset_to_model,
        "model_label_to_dataset_category": model_to_dataset,
        "num_classes": len(coco_categories),
    }


def audit_and_build(
    discovered: DiscoverResult,
    subset: dict[str, list[FramePair]],
    split_map: dict[str, list[str]],
    *,
    out_root: Path,
    annotation_modality: str,
    link_images: bool,
) -> dict[str, Any]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "annotations").mkdir(exist_ok=True)
    (out_root / "manifests").mkdir(exist_ok=True)

    used_cats: list[str] = []
    for seq, frames in subset.items():
        for pair in frames:
            for box in discovered.annotations.get(seq, {}).get(pair.frame_id, []):
                used_cats.append(box.category_name)
    cat_map = build_category_mapping(used_cats)
    name_to_id = cat_map["name_to_coco_id"]

    stats = {
        "pairing_missing": 0,
        "size_mismatch": 0,
        "duplicate_sample_ids": 0,
        "illegal_bbox": 0,
        "oob_bbox": 0,
        "unknown_category": 0,
        "empty_annotation_frames": 0,
        "per_class_instances": Counter(),
        "size_hist": Counter(),
        "split_sequence_counts": {k: len(v) for k, v in split_map.items()},
        "split_frame_counts": {k: 0 for k in split_map},
        "leakage_sequences": [],
        "link_mode_counts": Counter(),
    }

    # leakage check
    membership: dict[str, list[str]] = defaultdict(list)
    for split, seqs in split_map.items():
        for s in seqs:
            membership[s].append(split)
    stats["leakage_sequences"] = [s for s, sp in membership.items() if len(sp) > 1]

    sample_ids: set[str] = set()
    coco_by_split: dict[str, dict[str, Any]] = {}
    image_id = 1
    ann_id = 1

    for split, seqs in split_map.items():
        images = []
        annotations = []
        manifest_rows = []
        rgb_dir = out_root / "images" / split / "rgb"
        thr_dir = out_root / "images" / split / "thermal"
        if link_images:
            rgb_dir.mkdir(parents=True, exist_ok=True)
            thr_dir.mkdir(parents=True, exist_ok=True)

        for seq in seqs:
            for pair in subset.get(seq, []):
                if pair.sample_id in sample_ids:
                    stats["duplicate_sample_ids"] += 1
                    continue
                sample_ids.add(pair.sample_id)
                if not pair.rgb_path.is_file() or not pair.thermal_path.is_file():
                    stats["pairing_missing"] += 1
                    continue

                width = height = 0
                try:
                    from PIL import Image

                    with Image.open(pair.rgb_path) as im_r, Image.open(pair.thermal_path) as im_t:
                        rw, rh = im_r.size
                        tw, th = im_t.size
                        if (rw, rh) != (tw, th):
                            stats["size_mismatch"] += 1
                        width, height = tw, th  # thermal-aligned space preferred
                except Exception:  # noqa: BLE001
                    stats["pairing_missing"] += 1
                    continue

                file_name = f"{pair.sample_id}.jpg"
                if link_images:
                    # preserve original suffix in link target name when needed
                    ext = pair.rgb_path.suffix.lower() or ".jpg"
                    file_name = f"{pair.sample_id}{ext}"
                    mode_r = link_file(pair.rgb_path, rgb_dir / file_name)
                    mode_t = link_file(pair.thermal_path, thr_dir / file_name)
                    stats["link_mode_counts"][mode_r] += 1
                    stats["link_mode_counts"][mode_t] += 1

                images.append(
                    {
                        "id": image_id,
                        "file_name": file_name,
                        "width": width,
                        "height": height,
                        "sequence_id": seq,
                        "frame_id": pair.frame_id,
                    }
                )
                boxes = discovered.annotations.get(seq, {}).get(pair.frame_id, [])
                if not boxes:
                    stats["empty_annotation_frames"] += 1
                for box in boxes:
                    x, y, w, h = box.bbox
                    # Match rgbt_pair_audit: negative origin or non-positive size is illegal.
                    illegal = w <= 0 or h <= 0 or x < 0 or y < 0
                    oob = (not illegal) and (
                        x + w > width + 1e-3 or y + h > height + 1e-3
                    )
                    if illegal:
                        stats["illegal_bbox"] += 1
                        continue
                    if oob:
                        stats["oob_bbox"] += 1
                    cat_name = _normalize_category(box.category_name)
                    if cat_name not in name_to_id:
                        stats["unknown_category"] += 1
                        cat_name = "unknown" if "unknown" in name_to_id else list(name_to_id)[0]
                    coco_cid = name_to_id[cat_name]
                    area = float(box.area if box.area is not None else w * h)
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": coco_cid,
                            "bbox": [float(x), float(y), float(w), float(h)],
                            "area": area,
                            "iscrowd": int(box.iscrowd),
                            **({"track_id": box.track_id} if box.track_id is not None else {}),
                        }
                    )
                    stats["per_class_instances"][cat_name] += 1
                    if area < 16 * 16:
                        stats["size_hist"]["lt_16"] += 1
                    elif area < 32 * 32:
                        stats["size_hist"]["16_to_32"] += 1
                    else:
                        stats["size_hist"]["ge_32"] += 1
                    ann_id += 1

                manifest_rows.append(
                    {
                        "sample_id": pair.sample_id,
                        "sequence_id": seq,
                        "frame_id": pair.frame_id,
                        "split": split,
                        "rgb_path": str(pair.rgb_path.resolve()),
                        "thermal_path": str(pair.thermal_path.resolve()),
                        "file_name": file_name,
                        "width": width,
                        "height": height,
                    }
                )
                image_id += 1
                stats["split_frame_counts"][split] += 1

        coco_by_split[split] = {
            "images": images,
            "annotations": annotations,
            "categories": cat_map["categories"],
        }
        ann_path = out_root / "annotations" / f"instances_{split}.json"
        ann_path.write_text(json.dumps(coco_by_split[split], indent=2), encoding="utf-8")
        man_path = out_root / "manifests" / f"{split}_pairs.csv"
        with man_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "sample_id",
                    "sequence_id",
                    "frame_id",
                    "split",
                    "rgb_path",
                    "thermal_path",
                    "file_name",
                    "width",
                    "height",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    # Serialize counters
    quality = {
        "annotation_modality_priority": annotation_modality,
        "pairing_missing": stats["pairing_missing"],
        "size_mismatch": stats["size_mismatch"],
        "duplicate_sample_ids": stats["duplicate_sample_ids"],
        "illegal_bbox": stats["illegal_bbox"],
        "oob_bbox": stats["oob_bbox"],
        "unknown_category": stats["unknown_category"],
        "empty_annotation_frames": stats["empty_annotation_frames"],
        "per_class_instances": dict(stats["per_class_instances"]),
        "target_size_distribution": dict(stats["size_hist"]),
        "split_sequence_counts": stats["split_sequence_counts"],
        "split_frame_counts": stats["split_frame_counts"],
        "cross_split_sequence_leakage": stats["leakage_sequences"],
        "leakage_check": "passed" if not stats["leakage_sequences"] else "failed",
        "link_mode_counts": dict(stats["link_mode_counts"]),
        "discovery_layout": discovered.layout,
        "discovery_notes": discovered.notes,
    }
    (out_root / "quality_audit.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_root / "category_mapping.json").write_text(
        json.dumps(cat_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    yaml_text = f"""dataset_key: rgbt_tiny_v1
task_type: rgbt_detection
annotation_format: coco
source_dataset: RGBT-Tiny
claim_level: pipeline_validation_only

modalities:
  - rgb
  - thermal

splits:
  train:
    rgb_dir: images/train/rgb
    thermal_dir: images/train/thermal
    annotation: annotations/instances_train.json
  val:
    rgb_dir: images/val/rgb
    thermal_dir: images/val/thermal
    annotation: annotations/instances_val.json
  test:
    rgb_dir: images/test/rgb
    thermal_dir: images/test/thermal
    annotation: annotations/instances_test.json

path_policy: hardlink_or_symlink_no_copy
raw_root: {discovered.raw_root.as_posix()}
"""
    (out_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    return {
        "quality": quality,
        "category_mapping": cat_map,
        "coco_by_split": {k: {"n_images": len(v["images"]), "n_anns": len(v["annotations"])} for k, v in coco_by_split.items()},
    }


def write_freeze(
    out_root: Path,
    *,
    raw_root: Path,
    quality: dict[str, Any],
    category_mapping: dict[str, Any],
    subset_policy: dict[str, Any],
    split_map: dict[str, list[str]],
) -> dict[str, Any]:
    def _hash(path: Path) -> str | None:
        return sha256_file(path) if path.is_file() else None

    freeze = {
        "dataset_name": "RGBT-Tiny",
        "registered_key": "rgbt_tiny_v1",
        "version": "v1",
        "split_method": "sequence_level",
        "raw_root": str(raw_root.resolve()),
        "registered_root": str(out_root.resolve()),
        "subset_policy": subset_policy,
        "rgb_count": quality.get("split_frame_counts"),
        "thermal_count": quality.get("split_frame_counts"),
        "annotation_count": {
            split: None  # filled below
            for split in ("train", "val", "test")
        },
        "categories": category_mapping.get("categories"),
        "category_mapping": {
            "dataset_category_to_model_label": category_mapping.get(
                "dataset_category_to_model_label"
            ),
            "num_classes": category_mapping.get("num_classes"),
        },
        "manifest_hashes": {
            f"{s}_pairs.csv": _hash(out_root / "manifests" / f"{s}_pairs.csv")
            for s in ("train", "val", "test")
        },
        "annotation_hashes": {
            f"instances_{s}.json": _hash(out_root / "annotations" / f"instances_{s}.json")
            for s in ("train", "val", "test")
        },
        "sequence_split": split_map,
        "leakage_check": quality.get("leakage_check"),
        "path_policy": "hardlink_or_symlink_no_copy",
        "gate": "F0",
        "claim_authority": "registration_only",
    }
    for split in ("train", "val", "test"):
        ann = out_root / "annotations" / f"instances_{split}.json"
        if ann.is_file():
            payload = json.loads(ann.read_text(encoding="utf-8"))
            freeze["annotation_count"][split] = len(payload.get("annotations") or [])
    (out_root / "DATASET_FREEZE.json").write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = [
        "# RGBT-Tiny Gate F0 Registration Report",
        "",
        f"- raw_root: `{raw_root}`",
        f"- registered: `{out_root}`",
        f"- layout: `{quality.get('discovery_layout')}`",
        f"- leakage_check: **{quality.get('leakage_check')}**",
        f"- sequences: `{quality.get('split_sequence_counts')}`",
        f"- frames: `{quality.get('split_frame_counts')}`",
        f"- pairing_missing: {quality.get('pairing_missing')}",
        f"- size_mismatch: {quality.get('size_mismatch')}",
        f"- illegal_bbox: {quality.get('illegal_bbox')}",
        f"- oob_bbox: {quality.get('oob_bbox')}",
        f"- unknown_category: {quality.get('unknown_category')}",
        f"- empty_annotation_frames: {quality.get('empty_annotation_frames')}",
        f"- per_class: `{dict(quality.get('per_class_instances') or {})}`",
        f"- size_hist: `{dict(quality.get('target_size_distribution') or {})}`",
        f"- link_modes: `{dict(quality.get('link_mode_counts') or {})}`",
        "",
        "## Notes",
        *[f"- {n}" for n in (quality.get("discovery_notes") or [])],
        "",
        "## Policy",
        "- No raw/converted image copies; hardlink/symlink only.",
        "- Sequence-level split only (no frame-level random split).",
        "- F0 does not train; F1 is a small read-only/CUDA probe.",
    ]
    (out_root / "DATASET_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return freeze


def run_f0(
    *,
    configured_raw: Path,
    out_root: Path,
    max_sequences: int | None = 12,
    max_frames_per_seq: int | None = 40,
    frame_stride: int = 5,
    seed: int = 42,
    link_images: bool = True,
    annotation_modality: str = "thermal",
    train_sequences: int | None = None,
    val_sequences: int | None = None,
    test_sequences: int | None = None,
) -> dict[str, Any]:
    configured_raw = Path(configured_raw)
    out_root = Path(out_root)

    resolved = resolve_raw_data_root(configured_raw)
    empty_raw = resolved is None or (
        resolved.is_dir() and not any(resolved.iterdir())
    )
    if empty_raw:
        empty = {
            "gate": "F0",
            "status": "waiting_for_raw",
            "dataset_name": "RGBT-Tiny",
            "configured_raw": str(configured_raw),
            "resolved_raw": str(resolved) if resolved else None,
            "message": "D:\\datasets\\RGBT-Tiny\\raw is empty. Apply for download, extract, then re-run F0.",
            "download_forms": {
                "google": "https://forms.gle/EeRooNEYzXXporQt9",
                "microsoft": "https://forms.cloud.microsoft/r/nN7JmKn4eJ",
                "project": "https://github.com/XinyiYing/RGBT-Tiny",
            },
            "next": [
                "Fill official form and download RGBT-Tiny",
                "Extract under D:\\datasets\\RGBT-Tiny\\raw\\",
                "Confirm real root with Get-ChildItem -Depth 2",
                "Re-run: python scripts/register_rgbt_tiny_gate_f.py f0 --subset",
            ],
        }
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "GATE_F_STATUS.json").write_text(
            json.dumps(empty, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return empty

    discovered = discover_raw(resolved)
    if not discovered.sequences:
        blocked = {
            "gate": "F0",
            "status": "layout_unrecognized",
            "resolved_raw": str(resolved),
            "notes": discovered.notes,
            "hint": "Inspect tree and adjust discover_raw heuristics, or place visible/infrared sequence folders at raw root.",
        }
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "GATE_F_STATUS.json").write_text(
            json.dumps(blocked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return blocked

    official_split, leak = load_official_sequence_split(resolved)
    if official_split is not None:
        if leak:
            discovered.notes.append(f"official split leakage sequences: {leak}")
        full_split = carve_val_from_train(official_split, seed=seed, val_ratio=0.15)
        # Restrict to sequences present on disk
        full_split = {
            k: [s for s in v if s in discovered.sequences] for k, v in full_split.items()
        }
        split_map, subset = subset_within_splits(
            full_split,
            discovered.sequences,
            max_sequences=max_sequences,
            max_frames_per_seq=max_frames_per_seq,
            frame_stride=frame_stride,
            seed=seed,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            test_sequences=test_sequences,
        )
        discovered.notes.append(
            f"official data_split used; carved val; subset seqs="
            f"{ {k: len(v) for k, v in split_map.items()} }"
        )
    else:
        subset = select_subset(
            discovered.sequences,
            max_sequences=max_sequences,
            max_frames_per_seq=max_frames_per_seq,
            frame_stride=frame_stride,
            seed=seed,
        )
        split_map = split_sequences(sorted(subset), seed=seed)
        discovered.notes.append("no official data_split; random sequence split")

    # Load annotations only for selected sequences (still parses COCO JSON once)
    prefer = "01" if annotation_modality.lower() in {"thermal", "01", "infrared", "ir"} else "00"
    if discovered.layout == "official_images_seq_00_01":
        discovered.annotations = _load_official_rgbt_tiny_coco(
            resolved, subset, discovered.notes, prefer_modality=prefer
        )
        if not any(discovered.annotations.values()):
            discovered.annotations = _load_annotations_near(
                resolved, subset, discovered.notes
            )

    built = audit_and_build(
        discovered,
        subset,
        split_map,
        out_root=out_root,
        annotation_modality=annotation_modality,
        link_images=link_images,
    )
    subset_policy = {
        "max_sequences": max_sequences,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "test_sequences": test_sequences,
        "max_frames_per_seq": max_frames_per_seq,
        "frame_stride": frame_stride,
        "seed": seed,
        "full_corpus": max_sequences is None
        and train_sequences is None
        and max_frames_per_seq is None
        and frame_stride == 1,
        "official_split": official_split is not None,
        "source_raw": str(resolved),
        "gate_h_expand": True,
    }
    freeze = write_freeze(
        out_root,
        raw_root=resolved,
        quality=built["quality"],
        category_mapping=built["category_mapping"],
        subset_policy=subset_policy,
        split_map=split_map,
    )
    result = {
        "gate": "F0",
        "status": "registered_subset" if not subset_policy["full_corpus"] else "registered_full",
        "freeze": freeze,
        "quality": built["quality"],
        "coco_summary": built["coco_by_split"],
    }
    (out_root / "GATE_F_STATUS.json").write_text(
        json.dumps(
            {
                "gate": "F0",
                "status": result["status"],
                "allow_f1": built["quality"].get("leakage_check") == "passed"
                and built["quality"].get("pairing_missing", 1) == 0,
                "registered_root": str(out_root),
                "source_raw": str(resolved),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
