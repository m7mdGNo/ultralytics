from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def build_hrnet_pose_data_dict(data_root: str | Path) -> dict[str, Any]:
    """Build the Ultralytics ``data`` dict for HRNet CSV folder datasets (used by train and standalone val)."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HRNet dataset root is not a directory: {root}")

    train_samples, train_names = parse_hrnet_pose_split(root / "train")
    val_samples, val_names = parse_hrnet_pose_split(resolve_hrnet_validation_split(root))
    test_samples, test_names = parse_hrnet_pose_split(root / "test")
    names = sorted(set(train_names) | set(val_names) | set(test_names))
    if not names:
        raise RuntimeError(f"No classes found under {root}")

    name_to_idx = {n: i for i, n in enumerate(names)}
    for split in (train_samples, val_samples, test_samples):
        for s in split:
            s["cls"] = [name_to_idx[n] for n in s["cls_name"]]

    return {
        "path": root,
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "names": {i: n for i, n in enumerate(names)},
        "nc": len(names),
        "channels": 3,
        "kpt_shape": [1, 3],
    }


def resolve_hrnet_validation_split(data_root: str | Path) -> Path:
    """Return path to validation split: prefers ``valid/``, falls back to ``val/``.

    Looks for ``_annotations.csv`` in each candidate folder.
    """
    root = Path(data_root)
    for name in ("valid", "val"):
        split = root / name
        if (split / "_annotations.csv").exists():
            return split
    v1, v2 = root / "valid" / "_annotations.csv", root / "val" / "_annotations.csv"
    raise FileNotFoundError(f"Expected {v1} or {v2}.")


def parse_hrnet_pose_split(split_dir: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse a split folder containing images and `_annotations.csv`."""
    split_path = Path(split_dir)
    csv_path = split_path / "_annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected annotation file at {csv_path}")

    rows_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    classes: set[str] = set()

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["filename"]
            cls_name = row["class"]
            classes.add(cls_name)
            rows_by_file[name].append(
                {
                    "class": cls_name,
                    "xmin": float(row["xmin"]),
                    "ymin": float(row["ymin"]),
                    "xmax": float(row["xmax"]),
                    "ymax": float(row["ymax"]),
                }
            )

    class_names = sorted(classes)
    samples: list[dict[str, Any]] = []

    for fname, ann in rows_by_file.items():
        im_file = split_path / fname
        if not im_file.exists():
            continue
        boxes = []
        labels = []
        for a in ann:
            x1, y1, x2, y2 = a["xmin"], a["ymin"], a["xmax"], a["ymax"]
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            cx = x1 + w * 0.5
            cy = y1 + h * 0.5
            boxes.append([cx, cy, w, h])
            labels.append(a["class"])
        samples.append({"im_file": str(im_file), "boxes_xywh": np.array(boxes, dtype=np.float32), "cls_name": labels})

    return samples, class_names


class HRNetPoseDataset(Dataset):
    """Dataset for bbox-annotated keypoint-center training."""

    def __init__(self, samples: list[dict[str, Any]], imgsz: int = 640, augment: bool = False):
        self.samples = samples
        self.imgsz = int(imgsz)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        im = cv2.imread(sample["im_file"])
        if im is None:
            raise FileNotFoundError(f"Could not read image: {sample['im_file']}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h0, w0 = im.shape[:2]

        im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(im).permute(2, 0, 1).contiguous().float() / 255.0

        boxes = sample["boxes_xywh"].copy()
        boxes[:, 0] /= max(w0, 1)
        boxes[:, 1] /= max(h0, 1)
        boxes[:, 2] /= max(w0, 1)
        boxes[:, 3] /= max(h0, 1)

        n = len(boxes)
        keypoints = np.zeros((n, 1, 3), dtype=np.float32)
        keypoints[:, 0, 0] = boxes[:, 0]
        keypoints[:, 0, 1] = boxes[:, 1]
        keypoints[:, 0, 2] = 1.0

        return {
            "img": img,
            "cls": torch.as_tensor(sample["cls"], dtype=torch.float32).view(-1, 1),
            "bboxes": torch.as_tensor(boxes, dtype=torch.float32),
            "keypoints": torch.as_tensor(keypoints, dtype=torch.float32),
            "im_file": sample["im_file"],
            "ori_shape": (h0, w0),
            "imgsz": (self.imgsz, self.imgsz),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        imgs = torch.stack([b["img"] for b in batch], 0)
        cls = []
        bboxes = []
        keypoints = []
        bidx = []
        im_file = []
        ori_shape = []

        for i, b in enumerate(batch):
            n = b["cls"].shape[0]
            if n:
                cls.append(b["cls"])
                bboxes.append(b["bboxes"])
                keypoints.append(b["keypoints"])
                bidx.append(torch.full((n,), i, dtype=torch.long))
            im_file.append(b["im_file"])
            ori_shape.append(b["ori_shape"])

        if cls:
            cls = torch.cat(cls, 0)
            bboxes = torch.cat(bboxes, 0)
            keypoints = torch.cat(keypoints, 0)
            bidx = torch.cat(bidx, 0)
        else:
            cls = torch.zeros((0, 1), dtype=torch.float32)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            keypoints = torch.zeros((0, 1, 3), dtype=torch.float32)
            bidx = torch.zeros((0,), dtype=torch.long)

        return {
            "img": imgs,
            "cls": cls,
            "bboxes": bboxes,
            "keypoints": keypoints,
            "batch_idx": bidx,
            "im_file": im_file,
            "ori_shape": ori_shape,
        }
