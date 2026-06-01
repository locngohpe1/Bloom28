"""
BENCHMARK 1 — Synthetic Cold Chain Data
========================================
Mô phỏng bài toán: Food_Batches (nhỏ) join IoT_Temperature_Logs (lớn)
Mục tiêu: So sánh 3 strategies trên laptop (driver.memory=2g, local[*])

Scale laptop-friendly:
  - food_batches : 500 rows  (small dimension table)
  - iot_logs     : 50,000 rows (large fact table, ~10% match rate)

Strategies so sánh:
  1. direct_join       → Sort Merge Join (shuffle, no optimization)
  2. broadcast_hash    → BHJ (Spark native broadcast)
  3. global_bloom/udf  → Bloom Filter Join
  4. partitioned_bloom → Partition-aware Bloom Filter

Output: in kết quả từng strategy ra console + ghi JSON
"""

import sys, os, time, json, random
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Nếu chạy từ thư mục khác, chỉnh path này về folder chứa bloom_join.py
BLOOM_LIB_PATH = os.environ.get("BLOOM_LIB_PATH", ".")
sys.path.insert(0, BLOOM_LIB_PATH)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType

from bloom_join import BloomJoin


# ── Config ─────────────────────────────────────────────────────────────────

N_BATCHES   = 500       # số lô hàng (small table)
N_LOGS      = 50_000    # số sensor logs (large table)
MATCH_RATE  = 0.10      # 10% logs khớp với batch thật
BREACH_RATE = 0.20      # 20% logs có nhiệt độ > 8°C
SEED        = 42
TEMP_BREACH_THRESHOLD = 8.0

WAREHOUSES  = ["WH_HCM_01", "WH_HN_02", "WH_DN_03", "WH_HP_04", "WH_CT_05"]
CATEGORIES  = ["dairy", "meat", "seafood", "vegetables", "fruit"]

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ── Spark Session ───────────────────────────────────────────────────────────

def create_spark() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("Benchmark1_Synthetic")
        .config("spark.sql.shuffle.partitions", "4")   # laptop: ít partition cho nhanh
        .config("spark.driver.memory", "2g")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")  # tắt auto-broadcast để so sánh fair
        .getOrCreate()
    )


# ── Data Generators ─────────────────────────────────────────────────────────

def generate_food_batches(spark: SparkSession, n: int = N_BATCHES) -> DataFrame:
    """
    Bảng nhỏ: n lô hàng thực phẩm, mỗi lô có batch_id, warehouse, category, expiry.
    """
    random.seed(SEED)
    rows = []
    for i in range(n):
        rows.append((
            f"BATCH_{i:05d}",
            random.choice(WAREHOUSES),
            random.choice(CATEGORIES),
            (datetime(2024, 1, 1) + timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d"),
            round(random.uniform(1.0, 6.0), 1),   # safe_temp_max (°C)
        ))

    schema = StructType([
        StructField("batch_id",       StringType()),
        StructField("warehouse_id",   StringType()),
        StructField("category",       StringType()),
        StructField("expiry_date",    StringType()),
        StructField("safe_temp_max",  FloatType()),
    ])
    return spark.createDataFrame(rows, schema)


def generate_iot_logs(
    spark: SparkSession,
    n: int = N_LOGS,
    match_rate: float = MATCH_RATE,
    breach_rate: float = BREACH_RATE,
    skew: bool = False,
) -> DataFrame:
    """
    Bảng lớn: n sensor logs.
    - match_rate: tỷ lệ logs có batch_id khớp với Food_Batches
    - breach_rate: tỷ lệ logs có nhiệt độ vượt ngưỡng
    - skew=True: 80% logs từ 1 warehouse (WH_HCM_01) → test data skew
    """
    random.seed(SEED + 1)
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    rows = []

    for i in range(n):
        # batch_id: match_rate% khớp thật, còn lại là noise
        if random.random() < match_rate:
            batch_id = f"BATCH_{random.randint(0, N_BATCHES - 1):05d}"
        else:
            batch_id = f"BATCH_{random.randint(N_BATCHES, N_BATCHES * 20):05d}"

        # warehouse: uniform hoặc skewed
        if skew:
            warehouse = "WH_HCM_01" if random.random() < 0.80 else random.choice(WAREHOUSES[1:])
        else:
            warehouse = random.choice(WAREHOUSES)

        # temperature: breach_rate% vượt ngưỡng
        temp = (
            round(random.uniform(8.1, 25.0), 1)
            if random.random() < breach_rate
            else round(random.uniform(-2.0, 7.9), 1)
        )

        rows.append((
            f"LOG_{i:08d}",
            batch_id,
            warehouse,
            temp,
            (base_time + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S"),
        ))

    schema = StructType([
        StructField("log_id",        StringType()),
        StructField("batch_id",      StringType()),
        StructField("warehouse_id",  StringType()),
        StructField("temperature_c", FloatType()),
        StructField("recorded_at",   StringType()),
    ])
    return spark.createDataFrame(rows, schema)


# ── Benchmark Result ────────────────────────────────────────────────────────

@dataclass
class BenchmarkRow:
    benchmark:               str
    strategy:                str
    execution_mode:          str
    n_batches:               int
    n_logs:                  int
    exec_time_sec:           float
    joined_rows:             Optional[int]
    large_before_filter:     Optional[int]
    large_after_filter:      Optional[int]
    filter_reduction_pct:    Optional[float]   # % dòng bị loại bỏ
    candidate_ratio:         Optional[float]
    fpr_configured:          Optional[float]
    notes:                   str = ""
    error:                   str = ""


# ── Runner ──────────────────────────────────────────────────────────────────

def run_one(
    spark: SparkSession,
    food: DataFrame,
    logs: DataFrame,
    strategy: str,
    execution_mode: str = "udf",
    partition_col: Optional[str] = None,
    fpr: float = 0.01,
    benchmark_label: str = "",
) -> BenchmarkRow:
    """Chạy 1 strategy, đo thời gian và metrics."""
    bj = BloomJoin(
        spark=spark,
        expected_items=N_BATCHES * 2,
        false_positive_rate=fpr,
        max_collect_distinct_keys=N_BATCHES * 10,
    )

    # Cache cả 2 bảng trước khi đo → loại bỏ I/O khỏi timing
    food.cache(); food.count()
    logs.cache(); logs.count()

    start = time.perf_counter()
    try:
        result, metrics = bj.join(
            large_df=logs,
            small_df=food,
            key_cols="batch_id",
            join_type="inner",
            strategy=strategy,
            partition_col=partition_col,
            execution_mode=execution_mode,
            collect_metrics=True,
        )
        elapsed = round(time.perf_counter() - start, 3)

        frr = metrics.filter_reduction_ratio
        return BenchmarkRow(
            benchmark=benchmark_label,
            strategy=strategy,
            execution_mode=execution_mode,
            n_batches=N_BATCHES,
            n_logs=N_LOGS,
            exec_time_sec=elapsed,
            joined_rows=metrics.joined_rows,
            large_before_filter=metrics.large_rows_before_filter,
            large_after_filter=metrics.large_rows_after_filter,
            filter_reduction_pct=round(frr * 100, 1) if frr is not None else None,
            candidate_ratio=metrics.candidate_ratio,
            fpr_configured=fpr,
            notes=metrics.notes or "",
        )
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 3)
        return BenchmarkRow(
            benchmark=benchmark_label,
            strategy=strategy,
            execution_mode=execution_mode,
            n_batches=N_BATCHES,
            n_logs=N_LOGS,
            exec_time_sec=elapsed,
            joined_rows=None,
            large_before_filter=None,
            large_after_filter=None,
            filter_reduction_pct=None,
            candidate_ratio=None,
            fpr_configured=fpr,
            error=str(e),
        )
    finally:
        food.unpersist()
        logs.unpersist()


# ── Benchmark Suite ─────────────────────────────────────────────────────────

def bench_A_baseline_comparison(spark: SparkSession) -> list[BenchmarkRow]:
    """
    Bench A: So sánh 4 strategies trên uniform data.
    Câu hỏi: Strategy nào nhanh nhất? Filter bao nhiêu % dữ liệu?
    """
    print("\n" + "="*60)
    print("BENCH A — Baseline Comparison (uniform data)")
    print(f"  Food_Batches : {N_BATCHES} rows")
    print(f"  IoT_Logs     : {N_LOGS:,} rows | match_rate={MATCH_RATE}")
    print("="*60)

    food = generate_food_batches(spark)
    logs = generate_iot_logs(spark, skew=False)
    results = []

    strategies = [
        ("direct_join",       "udf",  None,            0.01),
        ("broadcast_hash",    "udf",  None,            0.01),
        ("global_bloom",      "udf",  None,            0.01),
        ("global_bloom",      "rdd",  None,            0.01),
        ("partitioned_bloom", "udf",  "warehouse_id",  0.01),
    ]

    for strat, mode, pcol, fpr in strategies:
        label = f"{strat}/{mode}"
        print(f"  Running: {label:<30}", end="", flush=True)
        r = run_one(spark, food, logs, strat, mode, pcol, fpr, benchmark_label="A_baseline")
        results.append(r)
        if r.error:
            print(f"ERROR: {r.error}")
        else:
            frr = f"{r.filter_reduction_pct}%" if r.filter_reduction_pct is not None else "N/A"
            print(f"→ {r.exec_time_sec}s | rows_after_filter={r.large_after_filter} | filter_reduction={frr}")

    return results


def bench_B_fpr_sensitivity(spark: SparkSession) -> list[BenchmarkRow]:
    """
    Bench B: Thay đổi FPR từ 0.001 → 0.20, đo filter_reduction.
    Câu hỏi: FPR ảnh hưởng thế nào đến hiệu quả lọc?
    """
    print("\n" + "="*60)
    print("BENCH B — FPR Sensitivity (global_bloom/udf)")
    print("="*60)

    food = generate_food_batches(spark)
    logs = generate_iot_logs(spark, skew=False)
    results = []

    for fpr in [0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.20]:
        print(f"  FPR={fpr:<6}", end="", flush=True)
        r = run_one(spark, food, logs, "global_bloom", "udf", None, fpr, benchmark_label="B_fpr")
        results.append(r)
        if r.error:
            print(f"  ERROR: {r.error}")
        else:
            frr = f"{r.filter_reduction_pct}%" if r.filter_reduction_pct is not None else "N/A"
            print(f"→ {r.exec_time_sec}s | filter_reduction={frr} | rows_after={r.large_after_filter}")

    return results


def bench_C_skew(spark: SparkSession) -> list[BenchmarkRow]:
    """
    Bench C: So sánh direct_join vs global_bloom vs partitioned_bloom trên dữ liệu skewed.
    80% logs từ 1 warehouse → test khả năng chịu skew.
    """
    print("\n" + "="*60)
    print("BENCH C — Data Skew (80% logs từ WH_HCM_01)")
    print("="*60)

    food = generate_food_batches(spark)
    logs_skewed = generate_iot_logs(spark, skew=True)
    results = []

    for strat, mode, pcol in [
        ("direct_join",       "udf",  None),
        ("global_bloom",      "udf",  None),
        ("partitioned_bloom", "udf",  "warehouse_id"),
    ]:
        print(f"  Running: {strat:<25}", end="", flush=True)
        r = run_one(spark, food, logs_skewed, strat, mode, pcol, 0.01, benchmark_label="C_skew")
        results.append(r)
        if r.error:
            print(f"ERROR: {r.error}")
        else:
            frr = f"{r.filter_reduction_pct}%" if r.filter_reduction_pct is not None else "N/A"
            print(f"→ {r.exec_time_sec}s | filter_reduction={frr}")

    return results


def bench_D_e2e_breach_detection(spark: SparkSession) -> None:
    """
    Bench D: End-to-end pipeline — phát hiện lô hàng bị vi phạm nhiệt độ.
    Đây là use case thực tế: filter hot_logs → bloom join → aggregate.
    """
    print("\n" + "="*60)
    print("BENCH D — End-to-End Breach Detection Pipeline")
    print("="*60)

    food = generate_food_batches(spark)
    logs = generate_iot_logs(spark, skew=False)

    # Pre-filter: chỉ giữ logs vượt ngưỡng
    hot_logs = logs.filter(F.col("temperature_c") > TEMP_BREACH_THRESHOLD)
    print(f"  Total logs     : {logs.count():,}")
    print(f"  Hot logs (>8°C): {hot_logs.count():,}")

    bj = BloomJoin(
        spark=spark,
        expected_items=N_BATCHES * 2,
        false_positive_rate=0.01,
        max_collect_distinct_keys=N_BATCHES * 10,
    )

    start = time.perf_counter()
    joined, metrics = bj.join(
        large_df=hot_logs,
        small_df=food,
        key_cols="batch_id",
        join_type="inner",
        strategy="global_bloom",
        collect_metrics=True,
    )

    # Xử lý duplicate warehouse_id column
    joined = joined.drop(food["warehouse_id"])

    # Aggregate: xếp hạng lô hàng nguy hiểm nhất
    at_risk = (
        joined
        .groupBy("batch_id", "warehouse_id", "category", "expiry_date")
        .agg(
            F.count("*").alias("breach_count"),
            F.max("temperature_c").alias("max_temp_c"),
            F.min("recorded_at").alias("first_breach_at"),
            F.max("recorded_at").alias("last_breach_at"),
        )
        .orderBy(F.desc("max_temp_c"))
    )
    elapsed = round(time.perf_counter() - start, 3)

    total_at_risk = at_risk.count()
    print(f"\n  → Execution time  : {elapsed}s")
    print(f"  → At-risk batches : {total_at_risk}")
    print(f"  → Filter reduction: {metrics.filter_reduction_ratio:.1%}" if metrics.filter_reduction_ratio else "")
    print("\n  Top 10 most dangerous batches:")
    at_risk.show(10, truncate=False)


# ── Print Summary ────────────────────────────────────────────────────────────

def print_summary(all_results: list[BenchmarkRow]) -> None:
    print("\n" + "="*70)
    print("SUMMARY — Benchmark 1 (Synthetic)")
    print("="*70)
    print(f"{'Bench':<6} {'Strategy':<22} {'Mode':<5} {'Time(s)':<9} {'Filter%':<10} {'Rows→After'}")
    print("-"*70)
    for r in all_results:
        frr  = f"{r.filter_reduction_pct}%" if r.filter_reduction_pct is not None else "N/A"
        after = str(r.large_after_filter) if r.large_after_filter is not None else "N/A"
        err  = f" ← ERROR: {r.error[:30]}" if r.error else ""
        print(f"{r.benchmark:<6} {r.strategy:<22} {r.execution_mode:<5} "
              f"{r.exec_time_sec:<9} {frr:<10} {after}{err}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")
    print(f"\nSpark version : {spark.version}")
    print(f"N_BATCHES     : {N_BATCHES}")
    print(f"N_LOGS        : {N_LOGS:,}")

    all_results: list[BenchmarkRow] = []

    try:
        all_results += bench_A_baseline_comparison(spark)
        all_results += bench_B_fpr_sensitivity(spark)
        all_results += bench_C_skew(spark)
        bench_D_e2e_breach_detection(spark)

        print_summary(all_results)

        # Ghi kết quả ra JSON
        out_path = "benchmark_1_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2, ensure_ascii=False)
        print(f"\nResults saved → {out_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
