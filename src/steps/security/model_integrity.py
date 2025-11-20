"""Model integrity helpers for MLSecOps requirements."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict


def _iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _calculate_hash(models_dir: str) -> str:
    root = Path(models_dir)
    digest = hashlib.sha256()
    for file_path in _iter_files(root):
        digest.update(str(file_path.relative_to(root)).encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def generate_model_hash(models_dir: str, hash_file: Path) -> str:
    """Create SHA256 hash over every file inside models_dir and persist it."""
    hash_value = _calculate_hash(models_dir)
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(hash_value, encoding="utf-8")
    return hash_value


def verify_model_hash(models_dir: str, hash_file: Path) -> bool:
    """Verify model hash if file exists."""
    if not hash_file.exists():
        return False
    expected = hash_file.read_text(encoding="utf-8").strip()
    current = _calculate_hash(models_dir)
    return expected == current


def record_model_integrity(models_dir: str, output_dir: Path) -> Dict[str, str | bool]:
    """Helper that wraps hash generation + verification."""
    hash_path = output_dir / "model.sha256"
    if hash_path.exists():
        verified = verify_model_hash(models_dir, hash_path)
        hash_value = hash_path.read_text(encoding="utf-8").strip()
    else:
        hash_value = generate_model_hash(models_dir, hash_path)
        verified = True
    return {"model_hash": hash_value, "verified": verified}
