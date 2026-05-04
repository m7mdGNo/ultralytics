# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision


def centernet_output_stride(stride: torch.Tensor | int | float | list | tuple) -> float:
    """Normalize ``model.stride`` to a float for heatmap decode (``Tensor`` from nn.Module; ``int`` from ``AutoBackend``)."""
    if isinstance(stride, torch.Tensor):
        return float(stride.flatten()[0].item())
    if isinstance(stride, (list, tuple)) and len(stride) > 0:
        x = stride[0]
        return float(x.flatten()[0].item()) if isinstance(x, torch.Tensor) else float(x)
    return float(stride)


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

    Vectorized over classes and candidates (no Python per-pixel loops) so validation stays responsive on large ``nc``.

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
    # Relaxed local-max: strict equality fails under fp16/AMP when neighbors tie or differ by tiny eps,
    # which yields empty peaks, no boxes, and mAP stuck at 0 while loss still decreases.
    eps = 1e-2 if hm_s.dtype in (torch.float16, torch.bfloat16) else 1e-4
    peak_mask = (hm_s >= pool - eps) & (hm_s >= conf_thres)
    masked = hm_s * peak_mask.to(dtype=hm_s.dtype)

    B, nc, H, W = masked.shape
    HW = H * W
    device = hm.device
    dtype = hm.dtype
    k_pool = min(max(max_det * 25, 256), nc * HW)
    out: list[torch.Tensor] = []

    for b in range(B):
        flat = masked[b].reshape(-1)
        k = min(k_pool, flat.numel())
        vals, idx = torch.topk(flat, k)
        sel = vals >= conf_thres
        vals, idx = vals[sel], idx[sel]
        # Weak heatmaps: no local-max survives conf — fall back to global top-k with a per-image floor so
        # val metrics are not stuck at P=R=mAP=0 while reg/wh losses still train.
        if idx.numel() == 0:
            flat_u = hm_s[b].reshape(-1)
            ku = min(k_pool, flat_u.numel())
            vals, idx = torch.topk(flat_u, ku)
            if vals.numel() == 0:
                out.append(torch.zeros(0, 6, device=device, dtype=dtype))
                continue
            vmax = float(vals[0].item())
            thr_u = min(float(conf_thres), max(vmax * 0.25, 1e-8))
            sel = vals >= thr_u
            vals, idx = vals[sel], idx[sel]
        if idx.numel() == 0:
            out.append(torch.zeros(0, 6, device=device, dtype=dtype))
            continue

        c = idx // HW
        rem = idx % HW
        yi = rem // W
        xi = rem % W

        reg_b = reg[b]
        wh_b = wh[b]
        dt = reg_b.dtype
        ox = (xi.to(dt) + reg_b[0, yi, xi]) * stride
        oy = (yi.to(dt) + reg_b[1, yi, xi]) * stride
        ww = torch.exp(wh_b[0, yi, xi])
        hh = torch.exp(wh_b[1, yi, xi])
        x1 = ox - ww * 0.5
        y1 = oy - hh * 0.5
        x2 = ox + ww * 0.5
        y2 = oy + hh * 0.5

        pred = torch.stack((x1, y1, x2, y2, vals.to(dt), c.to(dt)), dim=1)

        if classes is not None:
            cls_keep = torch.tensor(classes, device=device, dtype=torch.long)
            pred = pred[torch.isin(pred[:, 5].long(), cls_keep)]
        if pred.shape[0] == 0:
            out.append(torch.zeros(0, 6, device=device, dtype=dtype))
            continue

        boxes = pred[:, :4]
        scores = pred[:, 4]
        clses = pred[:, 5]
        labels = torch.zeros(pred.shape[0], dtype=torch.long, device=device) if agnostic else clses.long()
        # float32 boxes for NMS stability under AMP/half
        nms_idx = torchvision.ops.batched_nms(
            boxes.float(), scores.float(), labels, float(iou_thres)
        )
        if nms_idx.numel() == 0:
            out.append(torch.zeros(0, 6, device=device, dtype=dtype))
            continue
        nms_idx = nms_idx[scores[nms_idx].argsort(descending=True)[:max_det]]
        out.append(pred[nms_idx].to(dtype=dtype))
    return out
