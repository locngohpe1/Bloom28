from __future__ import annotations

from typing import List, Sequence, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column


KeyCols = Union[str, Sequence[str]]


def normalize_key_cols(key_cols: KeyCols) -> List[str]:
    if isinstance(key_cols, str):
        return [key_cols]
    return list(key_cols)


def validate_columns(df: DataFrame, required_cols: Sequence[str], df_name: str = "DataFrame") -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def build_serialized_key_expr(key_cols: KeyCols, delimiter: str = "||") -> Column:
    cols = normalize_key_cols(key_cols)
    return F.concat_ws(
        delimiter,
        *[F.coalesce(F.col(c).cast("string"), F.lit("__NULL__")) for c in cols]
    )


def add_serialized_key_column(
    df: DataFrame,
    key_cols: KeyCols,
    output_col: str = "__bf_key__",
    delimiter: str = "||",
) -> DataFrame:
    return df.withColumn(output_col, build_serialized_key_expr(key_cols, delimiter))

def estimate_distinct_keys(
    df: DataFrame,
    key_cols: KeyCols,
    rsd: float = 0.05,
    temp_key_col: str = "__bf_key__",
) -> int:
    tmp = add_serialized_key_column(df, key_cols, temp_key_col)
    return int(
        tmp.select(F.approx_count_distinct(F.col(temp_key_col), rsd).alias("cnt"))
        .first()["cnt"]
    )


def distinct_key_df(
    df: DataFrame,
    key_cols: KeyCols,
    temp_key_col: str = "__bf_key__",
) -> DataFrame:
    cols = normalize_key_cols(key_cols)
    return add_serialized_key_column(df.select(*cols), cols, temp_key_col).select(temp_key_col).distinct()


def distinct_partitioned_key_df(
    df: DataFrame,
    key_cols: KeyCols,
    partition_col: str,
    temp_key_col: str = "__bf_key__",
) -> DataFrame:
    cols = normalize_key_cols(key_cols)
    return (
        add_serialized_key_column(df.select(partition_col, *cols), cols, temp_key_col)
        .select(partition_col, temp_key_col)
        .distinct()
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
