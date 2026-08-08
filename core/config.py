import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

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
