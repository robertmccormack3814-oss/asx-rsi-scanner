from pathlib import Path
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIG = json.loads((ROOT / "config.json").read_text())

def load_json(path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))

def now_iso():
    return datetime.now(timezone.utc).isoformat()
