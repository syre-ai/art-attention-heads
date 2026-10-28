"""YAML config loading with path resolution."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config file by name (without extension).

    Parameters
    ----------
    name : str
        Config file name, e.g. "model", "features", "clustering", "dataset".

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return the rd1_attention_taxonomy root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_results_dir(subdir: str = "") -> Path:
    """Return (and create) a results subdirectory."""
    d = get_project_root() / "results" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d
