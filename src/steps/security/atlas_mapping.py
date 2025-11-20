"""Map security findings to MITRE ATLAS technique categories."""
from __future__ import annotations

from typing import Any, Dict, List


def map_to_atlas(
    data_checks: Dict[str, Any],
    adversarial_results: Dict[str, Any],
    dependency_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return a MITRE ATLAS aligned summary."""
    mappings: List[Dict[str, Any]] = []

    if sum(data_checks["pii_matches"].values()) > 0:
        mappings.append(
            {
                "technique": "Data Poisoning / PII Leakage",
                "atlas_id": "TA0001",
                "severity": "high",
                "details": "Potential PII detected in processed dataset.",
            }
        )
    else:
        mappings.append(
            {
                "technique": "Data Poisoning / PII Leakage",
                "atlas_id": "TA0001",
                "severity": "low",
                "details": "No PII signatures found.",
            }
        )

    if adversarial_results["change_ratio"] > 0.1:
        mappings.append(
            {
                "technique": "Model Evasion",
                "atlas_id": "TA0005",
                "severity": "medium",
                "details": f"{adversarial_results['change_ratio']:.2%} predictions flipped under noise.",
            }
        )
    else:
        mappings.append(
            {
                "technique": "Model Evasion",
                "atlas_id": "TA0005",
                "severity": "low",
                "details": "Model stable under basic adversarial noise.",
            }
        )

    if dependency_results["vulnerabilities"]:
        mappings.append(
            {
                "technique": "Supply Chain Compromise",
                "atlas_id": "TA0006",
                "severity": "medium",
                "details": f"Dependencies flagged: {dependency_results['vulnerabilities']}",
            }
        )
    else:
        mappings.append(
            {
                "technique": "Supply Chain Compromise",
                "atlas_id": "TA0006",
                "severity": "low",
                "details": "No known vulnerable dependencies found.",
            }
        )

    return mappings

