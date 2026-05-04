from __future__ import annotations

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class HRNetPosePredictor(BasePredictor):
    """Predictor that decodes class heatmap and offset maps into point keypoints."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        hm, off = preds
        hm = hm.sigmoid()
        results = []
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        for bi, orig_img in enumerate(orig_imgs):
            c, y, x = torch.where(hm[bi] > self.args.conf)
            if c.numel() == 0:
                results.append(Results(orig_img, path=self.batch[0][bi], names=self.model.names))
                continue

            scores = hm[bi, c, y, x]
            if scores.numel() > self.args.max_det:
                keep = torch.topk(scores, self.args.max_det).indices
                c, y, x, scores = c[keep], y[keep], x[keep], scores[keep]

            xy = torch.stack((x.float() + off[bi, 0, y, x], y.float() + off[bi, 1, y, x]), 1) * 4.0
            kpts = torch.zeros((xy.shape[0], 1, 3), device=xy.device)
            kpts[:, 0, :2] = xy
            kpts[:, 0, 2] = scores
            kpts = ops.scale_coords(img.shape[2:], kpts, orig_img.shape)

            # Build small visualization boxes around keypoints for compatibility with Results plotting.
            half = 3.0
            boxes = torch.zeros((xy.shape[0], 6), device=xy.device)
            boxes[:, 0] = kpts[:, 0, 0] - half
            boxes[:, 1] = kpts[:, 0, 1] - half
            boxes[:, 2] = kpts[:, 0, 0] + half
            boxes[:, 3] = kpts[:, 0, 1] + half
            boxes[:, 4] = scores
            boxes[:, 5] = c.float()

            results.append(Results(orig_img, path=self.batch[0][bi], names=self.model.names, boxes=boxes, keypoints=kpts))
        return results
