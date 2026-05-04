from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from ultralytics.data.build import build_dataloader
from ultralytics.data.hrnet_pose import HRNetPoseDataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER


class HRNetPoseValidator(BaseValidator):
    """Validator for center-keypoint HRNet predictions."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "pose"
        # BaseTrainer._setup_train expects self.metrics.keys (like DetMetrics.keys).
        self.metrics = SimpleNamespace(keys=["metrics/l2", "metrics/acc8"])
        self.total_points = 0
        self.total_l2 = 0.0
        self.total_hits = 0

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def postprocess(self, preds):
        return preds

    def get_dataloader(self, dataset_path, batch_size: int):
        """Build val loader from ``data['val']`` sample list (HRNet CSV layout)."""
        if not isinstance(dataset_path, list):
            raise TypeError(
                f"HRNetPoseValidator expected a sample list for split={self.args.split!r}; got {type(dataset_path)}"
            )
        dataset = HRNetPoseDataset(samples=dataset_path, imgsz=self.args.imgsz, augment=False)
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=False,
            pin_memory=self.training,
        )

    def init_metrics(self, model):
        self.total_points = 0
        self.total_l2 = 0.0
        self.total_hits = 0
        self.names = model.names
        self.nc = len(self.names)

    def update_metrics(self, preds, batch):
        decoded = self._decode_preds(preds, conf=self.args.conf, max_det=self.args.max_det)
        bsz = len(batch["im_file"])
        h_img, w_img = batch["img"].shape[2:]
        stride = 4.0

        for i in range(bsz):
            gt = batch["bboxes"][batch["batch_idx"] == i]
            gt_cls = batch["cls"][batch["batch_idx"] == i].view(-1).long()
            if gt.numel() == 0:
                continue
            gt_xy = gt[:, :2] * gt.new_tensor([w_img, h_img])
            pred_xy = decoded[i]["xy"] * stride
            pred_cls = decoded[i]["cls"].long() if decoded[i]["cls"].numel() else decoded[i]["cls"]

            for j, p in enumerate(gt_xy):
                cls = gt_cls[j]
                cls_mask = pred_cls == cls
                if cls_mask.any():
                    d = torch.cdist(p.unsqueeze(0), pred_xy[cls_mask]).min().item()
                    self.total_l2 += d
                    self.total_hits += float(d <= 8.0)
                else:
                    self.total_l2 += float(max(h_img, w_img))
                self.total_points += 1

    def get_stats(self):
        if self.total_points == 0:
            return {"metrics/l2": float("inf"), "metrics/acc8": 0.0, "fitness": 0.0}
        mean_l2 = self.total_l2 / self.total_points
        acc8 = self.total_hits / self.total_points
        return {"metrics/l2": mean_l2, "metrics/acc8": acc8, "fitness": acc8}

    def print_results(self):
        stats = self.get_stats()
        LOGGER.info(f"HRNetPose - L2: {stats['metrics/l2']:.3f}, Acc@8px: {stats['metrics/acc8']:.4f}")

    def get_desc(self):
        return "%22s%11s%11s" % ("Class", "L2", "Acc@8px")

    @staticmethod
    def _decode_preds(preds: tuple[torch.Tensor, torch.Tensor], conf: float, max_det: int):
        hm, off = preds
        hm = hm.sigmoid()
        b, c, _, _ = hm.shape
        out = []
        for bi in range(b):
            cb, yb, xb = torch.where(hm[bi] > conf)
            if cb.numel() == 0:
                out.append(
                    {"xy": torch.zeros((0, 2), device=hm.device), "cls": torch.zeros((0,), device=hm.device), "conf": torch.zeros((0,), device=hm.device)}
                )
                continue
            scores = hm[bi, cb, yb, xb]
            if scores.numel() > max_det:
                keep = torch.topk(scores, max_det).indices
                cb, yb, xb, scores = cb[keep], yb[keep], xb[keep], scores[keep]
            xy = torch.stack((xb.float() + off[bi, 0, yb, xb], yb.float() + off[bi, 1, yb, xb]), 1)
            out.append({"xy": xy, "cls": cb.float(), "conf": scores})
        return out
