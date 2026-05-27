from __future__ import annotations

from typing import Literal, Optional, Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

from adaptive_bloom_join import AdaptiveBloomPlanner
from bloom_broadcast import BloomBroadcast
from bloom_builder import BloomBuildResult, BloomFilterBuilder
from bloom_metrics import BloomJoinMetrics
from utils import (
    KeyCols,
    add_serialized_key_column,
    normalize_key_cols,
    safe_ratio,
    validate_columns,
)


ExecutionMode = Literal["udf", "rdd"]
Strategy = Literal["auto", "broadcast_hash", "global_bloom", "partitioned_bloom", "semi_join", "direct_join"]


class BloomJoin:
    """
    Advanced Bloom Join orchestration for PySpark.

    Features:
      - global bloom filter
      - partition-aware bloom filter
      - composite keys
      - adaptive strategy selection
      - udf or rdd pre-filter execution
      - metrics
      - fallback to broadcast hash / semi-join / direct join
    """

    def __init__(
        self,
        spark,
        expected_items: int = 1_000_000,
        false_positive_rate: float = 0.01,
        max_collect_distinct_keys: int = 5_000_000,
        planner: Optional[AdaptiveBloomPlanner] = None,
        default_execution_mode: ExecutionMode = "udf",
    ) -> None:
        self.spark = spark
        self.builder = BloomFilterBuilder(
            expected_items=expected_items,
            false_positive_rate=false_positive_rate,
            max_collect_distinct_keys=max_collect_distinct_keys,
        )
        self.broadcast_helper = BloomBroadcast(spark)
        self.planner = planner or AdaptiveBloomPlanner()
        self.default_execution_mode = default_execution_mode
        self.false_positive_rate = false_positive_rate

    # ---------- bloom membership helpers ----------

    def _filter_global_udf(
        self,
        large_df: DataFrame,
        key_cols: KeyCols,
        bloom_bc,
        delimiter: str = "||",
    ) -> DataFrame:
        # Thay UDF bằng isin() — collect key set về driver rồi dùng Spark native filter
        # bloom_bc.value là BloomFilter, ta cần tập keys thực — được truyền vào qua build_result
        # Nên method này nhận key_set trực tiếp thay vì bloom_bc
        key_set = bloom_bc.value
        keyed = add_serialized_key_column(large_df, key_cols, "__bf_key__", delimiter)
        return keyed.filter(F.col("__bf_key__").isin(list(key_set))).drop("__bf_key__")

    def _filter_partitioned_udf(
        self,
        large_df: DataFrame,
        key_cols: KeyCols,
        partition_col: str,
        partitioned_bc,
        delimiter: str = "||",
    ) -> DataFrame:
        # Tương tự: dùng isin() per partition thay vì UDF
        bf_map = partitioned_bc.value
        keyed = add_serialized_key_column(large_df, key_cols, "__bf_key__", delimiter)
        conditions = None
        for p, key_set in bf_map.items():
            keys = list(key_set)
            if not keys:
                continue
            if p == "__NULL_PARTITION__":
                part_cond = F.col(partition_col).isNull()
            else:
                part_cond = F.col(partition_col) == p
            cond = part_cond & F.col("__bf_key__").isin(keys)
            conditions = cond if conditions is None else (conditions | cond)
        if conditions is None:
            return keyed.filter(F.lit(False)).drop("__bf_key__")
        return keyed.filter(conditions).drop("__bf_key__")

    def _filter_global_rdd(
        self,
        large_df: DataFrame,
        key_cols: KeyCols,
        bloom_bc,
        delimiter: str = "||",
    ) -> DataFrame:
        cols = normalize_key_cols(key_cols)
        schema = large_df.schema

        def serialize_row(row) -> str:
            parts = []
            for c in cols:
                val = row[c]
                parts.append("__NULL__" if val is None else str(val))
            return delimiter.join(parts)

        filtered_rdd = large_df.rdd.filter(lambda row: serialize_row(row) in bloom_bc.value)
        return self.spark.createDataFrame(filtered_rdd, schema=schema)

    def _filter_partitioned_rdd(
        self,
        large_df: DataFrame,
        key_cols: KeyCols,
        partition_col: str,
        partitioned_bc,
        delimiter: str = "||",
    ) -> DataFrame:
        cols = normalize_key_cols(key_cols)
        schema = large_df.schema

        def serialize_row(row) -> str:
            parts = []
            for c in cols:
                val = row[c]
                parts.append("__NULL__" if val is None else str(val))
            return delimiter.join(parts)

        def keep(row) -> bool:
            p = str(row[partition_col]) if row[partition_col] is not None else "__NULL_PARTITION__"
            bf = partitioned_bc.value.get(p)
            if bf is None:
                return False
            return serialize_row(row) in bf

        filtered_rdd = large_df.rdd.filter(keep)
        return self.spark.createDataFrame(filtered_rdd, schema=schema)

    def _apply_bloom_filter(
        self,
        large_df: DataFrame,
        build_result: BloomBuildResult,
        execution_mode: ExecutionMode,
        delimiter: str = "||",
    ) -> tuple:
        """Returns (filtered_df, broadcast_variable) — caller must unpersist the broadcast variable."""
        if build_result.mode == "global":
            if execution_mode == "rdd":
                bloom_bc = self.broadcast_helper.broadcast(build_result.bloom_filter)
                return self._filter_global_rdd(large_df, build_result.key_cols, bloom_bc, delimiter), bloom_bc
            # udf mode: collect key set thay vì broadcast BloomFilter object
            key_set_bc = self.broadcast_helper.broadcast(build_result.key_set)
            return self._filter_global_udf(large_df, build_result.key_cols, key_set_bc, delimiter), key_set_bc

        if build_result.mode == "partitioned":
            if execution_mode == "rdd":
                partitioned_bc = self.broadcast_helper.broadcast(build_result.partitioned_filters)
                return self._filter_partitioned_rdd(
                    large_df,
                    build_result.key_cols,
                    build_result.partition_col,
                    partitioned_bc,
                    delimiter,
                ), partitioned_bc
            # udf mode: convert từng BloomFilter sang set of keys
            key_sets_bc = self.broadcast_helper.broadcast(build_result.partitioned_key_sets)
            return self._filter_partitioned_udf(
                large_df,
                build_result.key_cols,
                build_result.partition_col,
                key_sets_bc,
                delimiter,
            ), key_sets_bc
        raise ValueError(f"Unsupported build_result mode: {build_result.mode}")

    # ---------- exact strategies ----------

    def _broadcast_hash_join(
        self,
        large_df: DataFrame,
        small_df: DataFrame,
        key_cols: KeyCols,
        join_type: str,
    ) -> DataFrame:
        return large_df.join(F.broadcast(small_df), on=normalize_key_cols(key_cols), how=join_type)

    def _semi_prefilter_join(
        self,
        large_df: DataFrame,
        small_df: DataFrame,
        key_cols: KeyCols,
        join_type: str,
    ) -> DataFrame:
        cols = normalize_key_cols(key_cols)
        distinct_small = small_df.select(*cols).distinct()
        candidates = large_df.join(F.broadcast(distinct_small), on=cols, how="left_semi")
        return candidates.join(small_df, on=cols, how=join_type)

    def _direct_join(
        self,
        large_df: DataFrame,
        small_df: DataFrame,
        key_cols: KeyCols,
        join_type: str,
    ) -> DataFrame:
        return large_df.join(small_df, on=normalize_key_cols(key_cols), how=join_type)

    # ---------- public ----------

    def join(
        self,
        large_df: DataFrame,
        small_df: DataFrame,
        key_cols: KeyCols,
        join_type: str = "inner",
        strategy: Strategy = "auto",
        partition_col: Optional[str] = None,
        execution_mode: Optional[ExecutionMode] = None,
        collect_metrics: bool = False,
        delimiter: str = "||",
    ):
        """
        Returns:
          - DataFrame if collect_metrics=False
          - (DataFrame, BloomJoinMetrics) if collect_metrics=True
        """
        execution_mode = execution_mode or self.default_execution_mode
        key_cols_list = normalize_key_cols(key_cols)

        validate_columns(large_df, key_cols_list, "large_df")
        validate_columns(small_df, key_cols_list, "small_df")
        if partition_col:
            validate_columns(large_df, [partition_col], "large_df")
            validate_columns(small_df, [partition_col], "small_df")

        chosen_strategy = strategy
        planner_decision = None

        if strategy == "auto":
            planner_decision = self.planner.choose(small_df, key_cols_list, partition_col)
            chosen_strategy = planner_decision.strategy

        is_bloom_strategy = strategy == "global_bloom" or strategy == "partitioned_bloom" or (
                strategy == "auto" and planner_decision is not None and
                planner_decision.strategy in {"global_bloom", "partitioned_bloom"}
        )
        large_before = large_df.count() if (collect_metrics and is_bloom_strategy) else None
        small_rows = small_df.count() if collect_metrics else None
        small_distinct = None
        partition_count = None

        if chosen_strategy == "broadcast_hash":
            joined = self._broadcast_hash_join(large_df, small_df, key_cols_list, join_type)

        elif chosen_strategy == "semi_join":
            joined = self._semi_prefilter_join(large_df, small_df, key_cols_list, join_type)


        elif chosen_strategy == "global_bloom":

            build_result = self.builder.build_global(small_df, key_cols_list)

            small_distinct = build_result.distinct_count

            filtered_large, bloom_bc = self._apply_bloom_filter(

                large_df=large_df,

                build_result=build_result,

                execution_mode=execution_mode,

                delimiter=delimiter,

            )

            joined = filtered_large.join(small_df, on=key_cols_list, how=join_type)

            self.broadcast_helper.unpersist(bloom_bc)


        elif chosen_strategy == "partitioned_bloom":

            if not partition_col:
                raise ValueError("partition_col is required for partitioned_bloom strategy")

            build_result = self.builder.build_partitioned(small_df, key_cols_list, partition_col=partition_col)

            small_distinct = build_result.distinct_count

            partition_count = build_result.partition_count

            filtered_large, bloom_bc = self._apply_bloom_filter(

                large_df=large_df,

                build_result=build_result,

                execution_mode=execution_mode,

                delimiter=delimiter,

            )

            joined = filtered_large.join(small_df, on=[partition_col, *key_cols_list], how=join_type)

            self.broadcast_helper.unpersist(bloom_bc)

        elif chosen_strategy == "direct_join":
            joined = self._direct_join(large_df, small_df, key_cols_list, join_type)

        else:
            raise ValueError(f"Unsupported strategy: {chosen_strategy}")

        if not collect_metrics:
            if chosen_strategy in {"global_bloom", "partitioned_bloom"}:
                self.broadcast_helper.unpersist(bloom_bc)
            return joined

        large_after = None
        if chosen_strategy in {"global_bloom", "partitioned_bloom"}:
            large_after = filtered_large.count()
            self.broadcast_helper.unpersist(bloom_bc)

        joined_rows = joined.count()

        metrics = BloomJoinMetrics(
            strategy=chosen_strategy,
            execution_mode=execution_mode,
            join_type=join_type,
            small_rows=small_rows,
            small_distinct_keys=small_distinct if small_distinct is not None else (
                planner_decision.estimated_small_distinct_keys if planner_decision else None
            ),
            large_rows_before_filter=large_before,
            large_rows_after_filter=large_after,
            joined_rows=joined_rows,
            partition_count=partition_count,
            configured_false_positive_rate=self.false_positive_rate,
            candidate_ratio=safe_ratio(large_after or 0, large_before or 0) if large_after is not None else None,
            filter_reduction_ratio=(
                        1.0 - safe_ratio(large_after or 0, large_before or 0)) if large_after is not None else None,
            notes=planner_decision.notes if planner_decision else None,
        )
        return joined, metrics

    def recommend_strategies(self) -> list[str]:
        return [
            "1. broadcast_hash -> use when small_df is truly small",
            "2. global_bloom -> use when distinct keys are medium and large_df is huge",
            "3. partitioned_bloom -> use when both tables share a natural partition column such as tenant/date/country",
            "4. semi_join -> use when exact candidate pruning is preferred over probabilistic pruning",
            "5. direct_join -> use when bloom build cost is not justified",
        ]