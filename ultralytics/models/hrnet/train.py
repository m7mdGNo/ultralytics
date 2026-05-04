from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from ultralytics.data.build import build_dataloader
from ultralytics.data.hrnet_pose import HRNetPoseDataset, parse_hrnet_pose_split
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.hrnet.nn import HRNetPoseModel
from ultralytics.models.hrnet.val import HRNetPoseValidator
from ultralytics.utils import DEFAULT_CFG, RANK


class HRNetPoseTrainer(BaseTrainer):
    """Trainer for HRNet keypoint-center model with class heatmaps and offsets."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = overrides or {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

    def get_dataset(self) -> dict[str, Any]:
        data_root = Path(self.args.data)
        if not data_root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {data_root}")

        train_samples, train_names = parse_hrnet_pose_split(data_root / "train")
        val_samples, val_names = parse_hrnet_pose_split(data_root / "val")
        test_samples, test_names = parse_hrnet_pose_split(data_root / "test")
        names = sorted(set(train_names) | set(val_names) | set(test_names))
        if not names:
            raise RuntimeError(f"No classes found in {data_root}")

        name_to_idx = {n: i for i, n in enumerate(names)}
        for split in (train_samples, val_samples, test_samples):
            for s in split:
                s["cls"] = [name_to_idx[n] for n in s["cls_name"]]

        return {
            "path": data_root,
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
            "names": {i: n for i, n in enumerate(names)},
            "nc": len(names),
            "channels": 3,
            "kpt_shape": [1, 3],
        }

    def build_dataset(self, img_path, mode: str = "train", batch: int | None = None):
        samples = self.data[mode]
        return HRNetPoseDataset(samples=samples, imgsz=self.args.imgsz, augment=mode == "train")

    def get_dataloader(self, dataset_path, batch_size: int = 16, rank: int = 0, mode: str = "train") -> DataLoader:
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=mode == "train",
            rank=rank,
            drop_last=False,
        )

    def get_model(self, cfg: str | Path | dict[str, Any] | None = None, weights: str | Path | None = None, verbose=True):
        model = HRNetPoseModel(cfg if isinstance(cfg, dict) else None, nc=self.data["nc"], ch=3, verbose=verbose and RANK == -1)
        if weights:
            model.load_state_dict(weights.state_dict() if hasattr(weights, "state_dict") else weights, strict=False)
        return model

    def get_validator(self):
        self.loss_names = "heatmap_loss", "offset_loss"
        return HRNetPoseValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        vals = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, vals))

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )
