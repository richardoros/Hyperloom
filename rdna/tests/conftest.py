"""Conftest for rdna/tests/ — sys.path adjustments for the bench module.

The H0.0 bench script lives under rdna/bench/measure.py and is not a
proper installed package. The runtime tests under rdna/tests/ need to
import it via path manipulation; doing it here keeps the tests clean.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))