"""Simple dependency scanning + supply-chain safeguards."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

KNOWN_VULNS = {
    "requests": {"2.19.0": "CVE-2018-18074"},
    "pyyaml": {"5.1": "CVE-2017-18342"},
}


def parse_requirement(line: str) -> tuple[str, str | None]:
    line = line.strip()
    if not line or line.startswith("#"):
        return "", None
    if "==" in line:
        pkg, version = line.split("==", 1)
        return pkg.lower().strip(), version.strip()
    return line.lower().strip(), None


def scan_dependencies(requirements_path: str) -> Dict[str, List[str] | str | int]:
    path = Path(requirements_path)
    if not path.exists():
        return {"file_hash": "", "checked": 0, "vulnerabilities": []}

    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    vulns: List[str] = []
    checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        pkg, version = parse_requirement(line)
        if not pkg:
            continue
        checked += 1
        version_map = KNOWN_VULNS.get(pkg)
        if version_map and version in version_map:
            vulns.append(f"{pkg}=={version} -> {version_map[version]}")

    return {
        "file_hash": file_hash,
        "checked": checked,
        "vulnerabilities": vulns,
    }

