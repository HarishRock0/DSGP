from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class InsightQuerySignal:
    district: str

@dataclass
class InsightSignal:
    district: str
    insights: Dict[str, Any]
