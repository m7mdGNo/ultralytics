from __future__ import annotations

"""HRNet pose head. ``HRNET_NN_REV`` bumps when training hotfixes land (verify after pip install)."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Bump when training hotfixes land (verify after pip install).
HRNET_NN_REV = 3  # 3: criterion placeholder for BaseTrainer epoch-end hook


class ConvBNAct(nn.Module):
    """Simple Conv-BN-ReLU block."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(nn.Conv2d(c1, c2, k, s, p, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HRNetBackbone(nn.Module):
    """Compact HRNet-like backbone that keeps a high-resolution stream."""

    def __init__(self, ch: int = 3, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(ch, width, 3, 2), ConvBNAct(width, width, 3, 2))
        self.high = nn.Sequential(
            ConvBNAct(width, width, 3, 1),
            ConvBNAct(width, width, 3, 1),
            ConvBNAct(width, width, 3, 1),
        )
        self.low = nn.Sequential(
            ConvBNAct(width, width * 2, 3, 2),
            ConvBNAct(width * 2, width * 2, 3, 1),
            ConvBNAct(width * 2, width * 2, 3, 1),
        )
        self.fuse = ConvBNAct(width + width * 2, width * 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)  # stride 4
        high = self.high(x)
        low = self.low(high)
        low_up = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((high, low_up), 1))


class HRNetPoseModel(nn.Module):
    """
    HRNet-style keypoint detector with class heatmap and offset heads.

    Training target is generated from bbox centers:
      - heatmap: one channel per class
      - offset: sub-pixel x/y offset at positive locations
    """

    def __init__(self, cfg: dict[str, Any] | None = None, nc: int = 1, ch: int = 3, verbose: bool = True):
        super().__init__()
        cfg = cfg or {}
        width = int(cfg.get("width", 32))
        self.stride = torch.tensor([4.0])
        self.nc = int(nc)
        self.channels = int(ch)
        self.names = {i: f"class_{i}" for i in range(self.nc)}

        self.backbone = HRNetBackbone(ch=self.channels, width=width)
        hidden = width * 2
        self.head = ConvBNAct(hidden, hidden, 3, 1)
        self.hm_head = nn.Conv2d(hidden, self.nc, 1)
        self.off_head = nn.Conv2d(hidden, 2, 1)
        nn.init.constant_(self.hm_head.bias, -2.19)  # low initial confidence
        # BaseTrainer._do_train calls hasattr(self.criterion, "update"); YOLO uses a loss module there.
        self.criterion = nn.Module()  # no-op; no .update — hook is skipped

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor], *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x)
        return self.predict(x)

    def predict(self, x: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # Match device/dtype to weights (AMP keeps params fp32; CPU batches must be moved in trainer).
        p = next(self.parameters())
        x = x.to(device=p.device, dtype=p.dtype)
        feat = self.backbone(x)
        feat = self.head(feat)
        return self.hm_head(feat), self.off_head(feat)

    def _build_targets(
        self,
        batch: dict[str, torch.Tensor],
        out_h: int,
        out_w: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = batch["img"].shape[0]
        dt = dtype if dtype is not None else batch["img"].dtype
        hm_t = torch.zeros((b, self.nc, out_h, out_w), device=device, dtype=dt)
        off_t = torch.zeros((b, 2, out_h, out_w), device=device, dtype=dt)
        mask = torch.zeros((b, 1, out_h, out_w), device=device, dtype=dt)

        if batch["bboxes"].numel() == 0:
            return hm_t, off_t, mask

        # bboxes are normalized xywh
        boxes = batch["bboxes"]
        cls = batch["cls"].view(-1).long().clamp(0, self.nc - 1)
        bidx = batch["batch_idx"].view(-1).long()

        cx = boxes[:, 0] * out_w
        cy = boxes[:, 1] * out_h
        gx = torch.clamp(cx.long(), 0, out_w - 1)
        gy = torch.clamp(cy.long(), 0, out_h - 1)
        ox = cx - gx.float()
        oy = cy - gy.float()

        for i in range(boxes.shape[0]):
            bi, ci, x, y = bidx[i], cls[i], gx[i], gy[i]
            hm_t[bi, ci, y, x] = 1.0
            off_t[bi, 0, y, x] = ox[i]
            off_t[bi, 1, y, x] = oy[i]
            mask[bi, 0, y, x] = 1.0
            for yy in range(max(0, int(y) - 1), min(out_h, int(y) + 2)):
                for xx in range(max(0, int(x) - 1), min(out_w, int(x) + 2)):
                    d2 = (xx - x.float()) ** 2 + (yy - y.float()) ** 2
                    hm_t[bi, ci, yy, xx] = torch.maximum(hm_t[bi, ci, yy, xx], torch.exp(-d2 / 2))

        return hm_t, off_t, mask

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        preds: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hm_p, off_p = preds if preds is not None else self.predict(batch["img"])
        hm_t, off_t, mask = self._build_targets(
            batch, hm_p.shape[-2], hm_p.shape[-1], hm_p.device, hm_p.dtype
        )

        hm_loss = F.binary_cross_entropy_with_logits(hm_p, hm_t)
        denom = mask.sum().clamp(min=1.0)
        off_loss = (F.l1_loss(off_p * mask, off_t * mask, reduction="sum")) / denom
        total = hm_loss + off_loss
        return total, torch.stack((hm_loss.detach(), off_loss.detach()))

    @torch.no_grad()
    def decode(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        conf: float = 0.25,
        max_det: int = 300,
    ) -> list[dict[str, torch.Tensor]]:
        hm, off = preds
        hm = hm.sigmoid()
        b, _, _, _ = hm.shape
        outputs: list[dict[str, torch.Tensor]] = []

        for bi in range(b):
            hmb = hm[bi]
            offb = off[bi]
            cb, yb, xb = torch.where(hmb > conf)
            if cb.numel() == 0:
                outputs.append(
                    {
                        "xy": torch.zeros((0, 2), device=hm.device),
                        "conf": torch.zeros((0,), device=hm.device),
                        "cls": torch.zeros((0,), device=hm.device),
                    }
                )
                continue

            scores = hmb[cb, yb, xb]
            if scores.numel() > max_det:
                keep = torch.topk(scores, max_det).indices
                cb, yb, xb, scores = cb[keep], yb[keep], xb[keep], scores[keep]
            ox = offb[0, yb, xb]
            oy = offb[1, yb, xb]
            xy = torch.stack((xb.float() + ox, yb.float() + oy), 1) * self.stride[0]
            outputs.append({"xy": xy, "conf": scores, "cls": cb.float()})

        return outputs
