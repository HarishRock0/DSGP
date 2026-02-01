from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ChildInsightQuerySignal:
    district: str

@dataclass
class ChildInsightSignal:
    district: str
    insights: Dict[str, Any]