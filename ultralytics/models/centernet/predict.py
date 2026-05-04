# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import unwrap_model

from .utils import decode_centernet_outputs


class CenterNetPredictor(DetectionPredictor):
    """Run inference for CenterNet models (heatmap + offset + WH decode + NMS)."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Decode CenterNet tuple outputs, then build ``Results`` like the detection predictor."""
        if isinstance(preds, (list, tuple)) and len(preds) == 3:
            hm, reg, wh = preds
            stride = float(unwrap_model(self.model).stride.view(-1)[0].item())
            preds_list = decode_centernet_outputs(
                hm,
                reg,
                wh,
                stride,
                conf_thres=self.args.conf,
                iou_thres=self.args.iou,
                max_det=self.args.max_det,
                agnostic=self.args.agnostic_nms,
                classes=self.args.classes,
            )
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
            return self.construct_results(preds_list, img, orig_imgs, **kwargs)
        return super().postprocess(preds, img, orig_imgs, **kwargs)
