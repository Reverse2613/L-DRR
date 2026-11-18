"""Pin the framework API and keep its settings/cache inside writable folders."""

from __future__ import annotations

import os
import tempfile
from importlib import metadata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_VERSION = "8.3.0"
_configured = False
# Set before any CUDA work (config imports runtime before trainer setup).
# This removes avoidable cuBLAS warnings; CUDA grid_sample backward still
# cannot be promised bit-for-bit deterministic.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def require_framework() -> None:
    """Configure local settings before importing the supported framework.

    Returns:
        None. No model weights or dataset files are downloaded.

    Raises:
        RuntimeError: Ultralytics is missing or has a different API version.
    """
    global _configured
    if _configured:
        return
    try:
        installed = metadata.version("ultralytics")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("Install requirements-yolo-rgbt.txt in a separate environment.") from error
    if installed != FRAMEWORK_VERSION:
        raise RuntimeError(f"This adapter requires ultralytics=={FRAMEWORK_VERSION}; found {installed}.")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / "outputs" / "_yolo_rgbt_settings"))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rio_yolo_rgbt_matplotlib"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics.utils import SETTINGS

    # Local metrics/checkpoints are sufficient; external experiment services
    # are not part of this experiment. Never modify the user's other settings.
    disabled = ("sync", "hub", "wandb", "clearml", "comet", "mlflow", "neptune", "dvc", "raytune")
    SETTINGS.update({key: False for key in disabled if key in SETTINGS})
    _configured = True
