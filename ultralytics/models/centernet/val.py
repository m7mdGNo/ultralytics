# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.torch_utils import unwrap_model

from .utils import centernet_output_stride, decode_centernet_outputs


class CenterNetValidator(DetectionValidator):
    """Validate CenterNet by decoding heatmaps + offset + WH, then running NMS and standard detection metrics."""

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Track output stride for heatmap decoding."""
        super().init_metrics(model)
        m = unwrap_model(model)
        self._cn_stride = centernet_output_stride(m.stride)

    def postprocess(self, preds: torch.Tensor | tuple[torch.Tensor, ...]) -> list[dict[str, torch.Tensor]]:
        """Decode CenterNet tuple outputs to the same dict structure as :class:`DetectionValidator`."""
        if isinstance(preds, (list, tuple)) and len(preds) == 3:
            hm, reg, wh = preds
            raw = decode_centernet_outputs(
                hm,
                reg,
                wh,
                self._cn_stride,
                conf_thres=self.args.conf,
                iou_thres=self.args.iou,
                max_det=self.args.max_det,
                agnostic=self.args.single_cls or self.args.agnostic_nms,
                classes=self.args.classes,
            )
            return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in raw]
        return super().postprocess(preds)
