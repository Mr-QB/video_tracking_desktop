import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] Failed to load config.yaml: {e}")
    else:
        print(f"[WARN] config.yaml not found at {CONFIG_PATH}")
    return {}

APP_CONFIG = load_config()

def get_path(key: str, default_relative: str = "") -> Path:
    """
    Get a resolved absolute Path object for a given config key.
    If the path specified in config (or default) is relative, it will be resolved relative to PROJECT_ROOT.
    """
    raw_path = APP_CONFIG.get("paths", {}).get(key)
    if not raw_path:
        raw_path = default_relative
    p = Path(raw_path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p

