"""
Configuration loader for BTC-Pulse.
Reads config.yaml and provides typed access to settings.
"""

import os
import yaml
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

_config = None


def load_config(path: str | None = None) -> dict:
    """Load and cache configuration from YAML file."""
    global _config
    if _config is not None and path is None:
        return _config

    config_path = Path(path) if path else _CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Copy config.yaml.example to config.yaml and fill in your keys."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)

    return _config


def get_project_root() -> Path:
    return _PROJECT_ROOT


def get_db_path() -> Path:
    cfg = load_config()
    db_rel = cfg.get("data", {}).get("db_path", "data/btc_pulse.db")
    return _PROJECT_ROOT / db_rel


def get_snapshot_dir() -> Path:
    cfg = load_config()
    snap_rel = cfg.get("data", {}).get("snapshot_dir", "data/snapshots")
    p = _PROJECT_ROOT / snap_rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_log_dir() -> Path:
    cfg = load_config()
    log_rel = cfg.get("data", {}).get("log_dir", "logs")
    p = _PROJECT_ROOT / log_rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_api_key(name: str) -> str:
    cfg = load_config()
    key = cfg.get("api_keys", {}).get(name, "")
    # Also check environment variable override
    env_key = os.environ.get(f"BTCPULSE_{name.upper()}_KEY", "")
    return env_key or key
