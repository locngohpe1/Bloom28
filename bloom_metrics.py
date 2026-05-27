from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class BloomJoinMetrics:
    strategy: str
    execution_mode: str
    join_type: str
    small_rows: Optional[int] = None
    small_distinct_keys: Optional[int] = None
    large_rows_before_filter: Optional[int] = None
    large_rows_after_filter: Optional[int] = None
    joined_rows: Optional[int] = None
    partition_count: Optional[int] = None
    configured_false_positive_rate: Optional[float] = None
    candidate_ratio: Optional[float] = None
    filter_reduction_ratio: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def pretty(self) -> str:
        lines = [f"{k}: {v}" for k, v in self.to_dict().items()]
        return "\n".join(lines)