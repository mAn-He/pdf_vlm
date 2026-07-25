"""Reproducibility helpers for evaluation runs."""

from __future__ import annotations

import os
import random
from typing import Any


def set_global_seed(seed: int) -> dict[str, Any]:
    """Set Python / NumPy / env seeds. Returns a snapshot of what was applied."""
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    snapshot: dict[str, Any] = {"seed": seed, "PYTHONHASHSEED": str(seed), "numpy": False, "torch": False}
    try:
        import numpy as np

        np.random.seed(seed)
        snapshot["numpy"] = True
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        snapshot["torch"] = True
    except Exception:
        pass
    return snapshot
