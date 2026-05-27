from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pyspark.sql import DataFrame

from utils import KeyCols, estimate_distinct_keys


@dataclass
class AdaptiveDecision:
    strategy: str
    notes: str
    estimated_small_rows: Optional[int] = None
    estimated_small_distinct_keys: Optional[int] = None


class AdaptiveBloomPlanner:
    """
    Chooses a strategy based on size and data shape.

    Strategies:
      - broadcast_hash
      - global_bloom
      - partitioned_bloom
      - semi_join
      - direct_join
    """

    def __init__(
        self,
        broadcast_row_threshold: int = 100_000,
        bloom_distinct_threshold: int = 5_000_000,
        prefer_partitioned_when_partition_col_present: bool = True,
        approx_rsd: float = 0.05,
        count_small_rows: bool = True,
        use_semi_join_when_distinct_small: bool = False,
        semi_join_distinct_threshold: int = 300_000,
    ) -> None:
        self.broadcast_row_threshold = broadcast_row_threshold
        self.bloom_distinct_threshold = bloom_distinct_threshold
        self.prefer_partitioned_when_partition_col_present = prefer_partitioned_when_partition_col_present
        self.approx_rsd = approx_rsd
        self.count_small_rows = count_small_rows
        self.use_semi_join_when_distinct_small = use_semi_join_when_distinct_small
        self.semi_join_distinct_threshold = semi_join_distinct_threshold

    def choose(
        self,
        small_df: DataFrame,
        key_cols: KeyCols,
        partition_col: Optional[str] = None,
    ) -> AdaptiveDecision:
        small_rows = int(small_df.count()) if self.count_small_rows else None
        distinct_keys = estimate_distinct_keys(small_df, key_cols, rsd=self.approx_rsd)

        if small_rows is not None and small_rows <= self.broadcast_row_threshold:
            return AdaptiveDecision(
                strategy="broadcast_hash",
                notes=f"small_df row count {small_rows} <= broadcast threshold {self.broadcast_row_threshold}",
                estimated_small_rows=small_rows,
                estimated_small_distinct_keys=distinct_keys,
            )

        if self.use_semi_join_when_distinct_small and distinct_keys <= self.semi_join_distinct_threshold:
            return AdaptiveDecision(
                strategy="semi_join",
                notes=f"distinct key count {distinct_keys} <= semi-join threshold {self.semi_join_distinct_threshold}",
                estimated_small_rows=small_rows,
                estimated_small_distinct_keys=distinct_keys,
            )

        if partition_col and self.prefer_partitioned_when_partition_col_present and distinct_keys <= self.bloom_distinct_threshold:
            return AdaptiveDecision(
                strategy="partitioned_bloom",
                notes="partition column provided and bloom distinct threshold acceptable",
                estimated_small_rows=small_rows,
                estimated_small_distinct_keys=distinct_keys,
            )

        if distinct_keys <= self.bloom_distinct_threshold:
            return AdaptiveDecision(
                strategy="global_bloom",
                notes=f"distinct keys {distinct_keys} within bloom threshold {self.bloom_distinct_threshold}",
                estimated_small_rows=small_rows,
                estimated_small_distinct_keys=distinct_keys,
            )

        return AdaptiveDecision(
            strategy="direct_join",
            notes=f"distinct keys {distinct_keys} exceed bloom threshold {self.bloom_distinct_threshold}",
            estimated_small_rows=small_rows,
            estimated_small_distinct_keys=distinct_keys,
        )