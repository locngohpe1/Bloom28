from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from pybloom_live import BloomFilter
from pyspark.sql import DataFrame

from utils import (
    KeyCols,
    distinct_key_df,
    distinct_partitioned_key_df,
    estimate_distinct_keys,
    normalize_key_cols,
    validate_columns,
)
@dataclass
class BloomBuildResult:
    bloom_filter: Optional[BloomFilter] = None
    partitioned_filters: Optional[Dict[str, BloomFilter]] = None
    key_set: Optional[set] = None
    partitioned_key_sets: Optional[Dict[str, set]] = None
    distinct_count: int = 0
    partition_count: int = 0
    collected_count: int = 0
    key_cols: Optional[List[str]] = None
    partition_col: Optional[str] = None
    mode: str = "global"


class BloomFilterBuilder:
    """
    Builds Python bloom filters from Spark distinct keys.

    Good for:
      - dimension / lookup tables
      - medium distinct-key volumes
      - broadcasting compact probabilistic structures

    Guardrails:
      - max_collect_distinct_keys prevents accidental driver overload
      - partition-aware bloom can reduce false positives significantly
    """

    def __init__(
        self,
        expected_items: int = 1_000_000,
        false_positive_rate: float = 0.01,
        max_collect_distinct_keys: int = 5_000_000,
        estimate_before_collect: bool = True,
        approx_rsd: float = 0.05,
    ) -> None:
        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        self.max_collect_distinct_keys = max_collect_distinct_keys
        self.estimate_before_collect = estimate_before_collect
        self.approx_rsd = approx_rsd

    def _new_bloom(self, capacity: Optional[int] = None) -> BloomFilter:
        return BloomFilter(
            capacity=capacity or self.expected_items,
            error_rate=self.false_positive_rate,
        )

    def _guard_collect_size(self, df: DataFrame, key_cols: KeyCols) -> int:
        if not self.estimate_before_collect:
            return -1

        approx_cnt = estimate_distinct_keys(df, key_cols, rsd=self.approx_rsd)
        if approx_cnt > self.max_collect_distinct_keys:
            raise ValueError(
                f"Estimated distinct keys ({approx_cnt}) exceed max_collect_distinct_keys "
                f"({self.max_collect_distinct_keys}). Use fallback strategy or increase threshold."
            )
        return approx_cnt

    def build_global(self, df: DataFrame, key_cols: KeyCols) -> BloomBuildResult:
        cols = normalize_key_cols(key_cols)
        validate_columns(df, cols, "small_df")

        approx_cnt = self._guard_collect_size(df, cols)

        key_df = distinct_key_df(df, cols)
        bloom = self._new_bloom(capacity=max(approx_cnt, self.expected_items) if approx_cnt > 0 else self.expected_items)

        collected = 0
        key_set = set()
        for row in key_df.toLocalIterator():
            key = row["__bf_key__"]
            bloom.add(key)
            key_set.add(key)
            collected += 1
            if collected > self.max_collect_distinct_keys:
                raise ValueError(
                    f"Collected distinct keys exceeded max_collect_distinct_keys={self.max_collect_distinct_keys}"
                )

        return BloomBuildResult(
            bloom_filter=bloom,
            key_set=key_set,
            distinct_count=collected,
            collected_count=collected,
            key_cols=cols,
            mode="global",
        )

    def build_partitioned(
        self,
        df: DataFrame,
        key_cols: KeyCols,
        partition_col: str,
        max_partitions: int = 50_000,
    ) -> BloomBuildResult:
        cols = normalize_key_cols(key_cols)
        validate_columns(df, [partition_col, *cols], "small_df")

        approx_cnt = self._guard_collect_size(df, cols)

        pair_df = distinct_partitioned_key_df(df, cols, partition_col=partition_col)
        partitioned: Dict[str, BloomFilter] = {}
        collected = 0

        partitioned_key_sets: Dict[str, set] = {}
        for row in pair_df.toLocalIterator():
            p = str(row[partition_col]) if row[partition_col] is not None else "__NULL_PARTITION__"
            if p not in partitioned:
                if len(partitioned) >= max_partitions:
                    raise ValueError(
                        f"Partition count exceeded max_partitions={max_partitions}. "
                        f"Partitioned bloom is not a good fit."
                    )
                partitioned[p] = self._new_bloom()
                partitioned_key_sets[p] = set()
            key = row["__bf_key__"]
            partitioned[p].add(key)
            partitioned_key_sets[p].add(key)
            collected += 1
            if collected > self.max_collect_distinct_keys:
                raise ValueError(
                    f"Collected partitioned keys exceeded max_collect_distinct_keys={self.max_collect_distinct_keys}"
                )

        return BloomBuildResult(
            partitioned_filters=partitioned,
            partitioned_key_sets=partitioned_key_sets,
            distinct_count=collected,
            partition_count=len(partitioned),
            collected_count=collected,
            key_cols=cols,
            partition_col=partition_col,
            mode="partitioned",
        )