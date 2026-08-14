"""Configuration loading for the extracted main method."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run = config["runs"][0]
    merged = dict(config)
    merged.update(run)
    return merged
