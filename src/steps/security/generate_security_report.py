"""Generate final MLSecOps / MITRE ATLAS security report."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def generate_security_report(
    output_path: Path,
    data_checks: Dict[str, Any],
    adversarial_results: Dict[str, Any],
    model_integrity: Dict[str, Any],
    dependency_results: Dict[str, Any],
    atlas_summary: Any,
) -> Path:
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_security": data_checks,
        "adversarial_test": adversarial_results,
        "model_integrity": model_integrity,
        "dependency_scan": dependency_results,
        "mitre_atlas_summary": atlas_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path

