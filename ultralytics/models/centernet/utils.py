# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision


def decode_centernet_outputs(
    hm: torch.Tensor,
    reg: torch.Tensor,
    wh: torch.Tensor,
    stride: float,
    conf_thres: float = 0.3,
    iou_thres: float = 0.45,
    max_det: int = 300,
    agnostic: bool = False,
    classes: list[int] | None = None,
) -> list[torch.Tensor]:
    """Decode CenterNet heatmap / offset / WH tensors to per-image ``[N, 6]`` xyxy + conf + cls (letterbox space).

    Args:
        hm (torch.Tensor): Heatmap logits ``(B, nc, H, W)``.
        reg (torch.Tensor): Sub-pixel offset ``(B, 2, H, W)`` added to integer cell indices.
        wh (torch.Tensor): Log width/height ``(B, 2, H, W)`` (exp to pixels).
        stride (float): Input stride (heatmap cell size in input pixels).
        conf_thres (float): Confidence threshold on heatmap after sigmoid.
        iou_thres (float): IoU threshold for NMS.
        max_det (int): Maximum boxes per image after NMS.
        agnostic (bool): Class-agnostic NMS.
        classes (list[int], optional): Filter to these class indices.

    Returns:
        (list[torch.Tensor]): One tensor per batch element, each ``(N, 6)`` as ``xyxy, conf, cls``.
    """
    hm_s = hm.sigmoid()
    pool = F.max_pool2d(hm_s, kernel_size=3, stride=1, padding=1)
    peak_mask = (hm_s == pool) & (hm_s >= conf_thres)
    B, nc, H, W = hm_s.shape
    device = hm.device
    dtype = hm.dtype
    out: list[torch.Tensor] = []

    for b in range(B):
        dets = []
        for c in range(nc):
            heat = hm_s[b, c] * peak_mask[b, c].float()
            flat = heat.flatten()
            k = min(max_det * 4, flat.numel())
            if k == 0:
                continue
            vals, inds = torch.topk(flat, k)
            for v, ix in zip(vals, inds):
                if float(v) < conf_thres:
                    break
                ix = int(ix)
                yi, xi = divmod(ix, W)
                ox = (float(xi) + float(reg[b, 0, yi, xi])) * stride
                oy = (float(yi) + float(reg[b, 1, yi, xi])) * stride
                ww = float(torch.exp(wh[b, 0, yi, xi]))
                hh = float(torch.exp(wh[b, 1, yi, xi]))
                row = torch.tensor(
                    [ox - ww / 2, oy - hh / 2, ox + ww / 2, oy + hh / 2, float(v), float(c)],
                    device=device,
                    dtype=dtype,
                )
                dets.append(row)

        if not dets:
            out.append(torch.zeros(0, 6, device=device, dtype=dtype))
            continue

        pred = torch.stack(dets, 0)
        boxes, scores, clses = pred[:, :4], pred[:, 4], pred[:, 5]
        if classes is not None:
            m = torch.zeros_like(scores, dtype=torch.bool)
            for ci in classes:
                m |= clses == ci
            pred, boxes, scores, clses = pred[m], boxes[m], scores[m], clses[m]
        if scores.shape[0] == 0:
            out.append(torch.zeros(0, 6, device=device, dtype=dtype))
            continue
        labels = torch.zeros_like(clses, dtype=torch.long) if agnostic else clses.long()
        nms_idx = torchvision.ops.batched_nms(boxes, scores, labels, iou_thres)
        nms_idx = nms_idx[scores[nms_idx].argsort(descending=True)[:max_det]]
        out.append(pred[nms_idx])
    return out
