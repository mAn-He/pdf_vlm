"""IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    base = root or project_root()
    return (base / p).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML: {path}")
    return data


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_model(path: str | Path, model: type[T]) -> T:
    return model.model_validate(load_json(path))


def save_model(path: str | Path, obj: BaseModel) -> None:
    save_json(path, obj.model_dump(mode="json"))


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_name: str = "default.yaml", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load configs/<name> and optionally merge nested overrides."""
    cfg_path = resolve_path(f"configs/{config_name}")
    cfg = load_yaml(cfg_path)
    if overrides:
        cfg = merge_dicts(cfg, overrides)
    return cfg


def load_named_config(relative: str) -> dict[str, Any]:
    """Load a config path relative to configs/, e.g. models/gemma3_4b_qat.yaml."""
    return load_yaml(resolve_path(f"configs/{relative}"))
