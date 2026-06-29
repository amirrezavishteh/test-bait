"""Persistence for the CASA-Lite conformal-quantile threshold artifact."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from casa_lite.conformal_quantile import QuantileCalibration


@dataclass
class LiteCalibrationArtifact:
    """A saved CASA-Lite threshold calibration.

    Attributes:
        detection: The conformal-quantile calibration, or ``None``.
        version: Artifact schema version.
    """

    detection: Optional[QuantileCalibration] = None
    version: int = 1

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "detection": self.detection.to_dict() if self.detection else None,
        }

    def save(self, path: str) -> None:
        """Write the artifact to ``path`` (creating parent dirs)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "LiteCalibrationArtifact":
        """Load an artifact written by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        det = data.get("detection")
        detection = (
            QuantileCalibration(
                threshold=det["threshold"],
                alpha=det["alpha"],
                n_models=det["n_models"],
                rank=det["rank"],
                certified=det.get("certified", True),
                date=det.get("date", ""),
            )
            if det
            else None
        )
        return cls(detection=detection, version=int(data.get("version", 1)))


def load_artifact_if_exists(path: Optional[str]) -> Optional[LiteCalibrationArtifact]:
    """Load a CASA-Lite calibration artifact if present, else ``None``."""
    if path and os.path.exists(path):
        return LiteCalibrationArtifact.load(path)
    return None
