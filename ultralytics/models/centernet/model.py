# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import CenterNetDetectionModel
from ultralytics.utils import ROOT

from .predict import CenterNetPredictor
from .train import CenterNetTrainer
from .val import CenterNetValidator


class CenterNet(Model):
    """CenterNet object detector: heatmap peaks, sub-pixel offset, and width/height regression.

    Examples:
        >>> from ultralytics import CenterNet
        >>> model = CenterNet("centernet.yaml")
        >>> model.train(data="coco8.yaml", epochs=1, imgsz=640)
    """

    def __init__(
        self, model: str | Path | None = None, task: str | None = None, verbose: bool = False
    ) -> None:
        """Initialize CenterNet; ``task`` is fixed to ``detect`` (same dataset format as YOLO detection)."""
        _ = task  # API compatibility with YOLO(..., task=...); CenterNet is always detection-format data.
        if model is None:
            model = ROOT / "cfg/models/centernet/centernet.yaml"
        super().__init__(model=model, task="detect", verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map detect task to CenterNet-specific trainer, validator, and predictor."""
        return {
            "detect": {
                "model": CenterNetDetectionModel,
                "trainer": CenterNetTrainer,
                "validator": CenterNetValidator,
                "predictor": CenterNetPredictor,
            },
        }
