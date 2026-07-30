import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
_MODELS_DIR = Path(__file__).parent.parent / "models"

def load_model_config() -> Dict[str, Any]:
    config_path = _MODELS_DIR / "model_config.json"
    return json.loads(config_path.read_text()) if config_path.exists() else {}

def validate_model_version(model_path: Path, expected_version: str) -> bool:
    try:
        model_info_path = _MODELS_DIR / "model_info.json"
        model_info = json.loads(model_info_path.read_text()) if model_info_path.exists() else {}
        return model_info.get("version") == expected_version
    except Exception:
        return False

def get_model_cache_key(config: Dict[str, Any]) -> str:
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def get_latest_model_version() -> Optional[str]:
    try:
        model_info_path = _MODELS_DIR / "model_info.json"
        model_info = json.loads(model_info_path.read_text()) if model_info_path.exists() else {}
        return model_info.get("version")
    except Exception:
        return None

def backup_current_model(model_path: Path):
    if not model_path.exists():
        return

    timestamp = Path(__file__).parent.parent / "backups"
    timestamp.mkdir(exist_ok=True)

    backup_path = timestamp / f"model_{model_path.name}_backup_{int(Path.mtime(model_path))}.pt"
    import shutil
    shutil.copy2(model_path, backup_path)
    logger.info(f"Model backed up to {backup_path}")

def check_disk_space(required_mb: int) -> bool:
    import shutil
    total, used, free = shutil.disk_usage(Path.cwd())
    free_mb = free // (1024 * 1024)
    return free_mb >= required_mb