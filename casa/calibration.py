"""Persistence for conformal calibration artifacts.

A single JSON artifact stores the match-threshold (``beta``) and/or detection
(z-score) calibrations so they need not be recomputed for every scan.  The
format records, per the spec, the calibration date, sample counts, target rates,
achieved bounds and calibrated values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from casa.conformal import DetectionCalibration, MatchCalibration


@dataclass
class CalibrationArtifact:
    """Bundle of optional match and detection calibrations.

    Attributes:
        match: Match-threshold calibration, or ``None``.
        detection: Detection-threshold calibration, or ``None``.
        version: Artifact schema version.
    """

    match: Optional[MatchCalibration] = None
    detection: Optional[DetectionCalibration] = None
    version: int = 1

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return {
            "version": self.version,
            "match": self.match.to_dict() if self.match else None,
            "detection": self.detection.to_dict() if self.detection else None,
        }

    def save(self, path: str) -> None:
        """Write the artifact to ``path`` (creating parent dirs)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationArtifact":
        """Load an artifact written by :meth:`save`.

        Args:
            path: Artifact file path.

        Returns:
            The reconstructed :class:`CalibrationArtifact`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        match = _match_from_dict(data.get("match"))
        detection = _detection_from_dict(data.get("detection"))
        return cls(match=match, detection=detection, version=int(data.get("version", 1)))

    def merge(self, other: "CalibrationArtifact") -> "CalibrationArtifact":
        """Return a copy with ``other``'s non-null calibrations overlaid."""
        return CalibrationArtifact(
            match=other.match or self.match,
            detection=other.detection or self.detection,
            version=max(self.version, other.version),
        )


def _match_from_dict(d: Optional[dict]) -> Optional[MatchCalibration]:
    if not d:
        return None
    return MatchCalibration(
        beta=d["beta"],
        target_match_error=d["target_match_error"],
        achieved_loss=d["achieved_loss"],
        n_match_pairs=d["n_match_pairs"],
        n_total_pairs=d["n_total_pairs"],
        certified=d.get("certified", True),
        date=d.get("date", ""),
    )


def _detection_from_dict(d: Optional[dict]) -> Optional[DetectionCalibration]:
    if not d:
        return None
    return DetectionCalibration(
        threshold=d["threshold"],
        target_far=d["target_far"],
        failure_prob=d["failure_prob"],
        n_models=d["n_models"],
        achieved_ucb=d["achieved_ucb"],
        certified=d.get("certified", True),
        date=d.get("date", ""),
    )


def load_artifact_if_exists(path: Optional[str]) -> Optional[CalibrationArtifact]:
    """Load a calibration artifact if ``path`` is set and exists, else ``None``."""
    if path and os.path.exists(path):
        return CalibrationArtifact.load(path)
    return None
