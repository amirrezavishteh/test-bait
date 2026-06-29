"""Smoke test: the seminar demo runs end-to-end and shows the key contrast."""

from __future__ import annotations

import io
import os
import runpy
from contextlib import redirect_stdout

_DEMO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "examples", "casa_demo.py"
)


def test_demo_runs_and_reports() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        runpy.run_path(_DEMO, run_name="__main__")
    out = buf.getvalue()
    # Worked example, both detectors, and the decisive multi-target contrast.
    assert "0.442" in out                       # spec §7.3 backdoor score
    assert "BACKDOORED" in out                  # CASA-Lite flags
    assert "MULTI-TARGET" in out and "MISS" in out  # BAIT Q-Score misses it
