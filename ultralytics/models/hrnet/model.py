from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model
from ultralytics.models.hrnet.nn import HRNetPoseModel
from ultralytics.models.hrnet.predict import HRNetPosePredictor
from ultralytics.models.hrnet.train import HRNetPoseTrainer
from ultralytics.models.hrnet.val import HRNetPoseValidator


class HRNetPose(Model):
    """HRNet-style keypoint-center model (heatmap + offset)."""

    def __init__(self, model: str | Path = "hrnet_pose.yaml", task: str | None = "pose", verbose: bool = False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "pose": {
                "model": HRNetPoseModel,
                "trainer": HRNetPoseTrainer,
                "validator": HRNetPoseValidator,
                "predictor": HRNetPosePredictor,
            }
        }
