# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import CenterNetDetectionModel
from ultralytics.utils import RANK

from .val import CenterNetValidator


class CenterNetTrainer(DetectionTrainer):
    """Trainer for CenterNet heatmap + offset + WH detection."""

    def get_model(self, cfg: dict | str | None = None, weights: str | None = None, verbose: bool = True):
        """Build and optionally load a :class:`CenterNetDetectionModel`."""
        model = CenterNetDetectionModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a CenterNet validator and three-component loss names."""
        self.loss_names = "hm_loss", "reg_loss", "wh_loss"
        return CenterNetValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
