"""
TEST 2 — Intel Berkeley Research Lab Sensor Dataset
======================================================================
Dataset : http://db.csail.mit.edu/labdata/labdata.html

"""

import sys
import os
import time
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

# ── Fix Python version mismatch trên Windows (phải đặt TRƯỚC SparkSession) ──
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ── Path tới BloomJoin library ───────────────────────────────────────────────
BLOOM_LIB_PATH = os.environ.get("BLOOM_LIB_PATH", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BLOOM_LIB_PATH)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, IntegerType, DoubleType
)
from bloom_join import BloomJoin


# ── Spark Session ────────────────────────────────────────────────────────────

def create_spark() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("Benchmark2_IntelLabData")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )


# ── Load Intel Lab Data ──────────────────────────────────────────────────────

def load_intel_lab_txt(spark: SparkSession, path: str) -> DataFrame:
    """
    Load file .txt space-separated từ MIT.
    Format: date time epoch moteid temperature humidity light voltage
    Ví dụ : 2004-03-31 03:38:15.757551 2 1 122.153 -3.91901 11.04 2.03397

    Lưu ý:
      - Không có header
      - Có dòng thiếu field (< 8 parts) → bỏ qua
      - Có dòng có thừa khoảng trắng → split() tự xử lý
      - temperature = 122°C là faulty reading → lọc ở bước sau
    """
    raw = spark.read.text(path)

    # Parse bằng Spark SQL expressions thay vì Python UDF
    # → tránh Python worker, tránh lỗi version mismatch
    split_col = F.split(F.trim(F.col("value")), r"\s+")

    parsed = (
        raw
        # Lọc dòng trống hoặc quá ngắn
        .filter(F.trim(F.col("value")) != "")
        .filter(F.size(F.split(F.trim(F.col("value")), r"\s+")) >= 8)
        .select(
            split_col.getItem(0).alias("date"),
            split_col.getItem(1).alias("time_str"),
            split_col.getItem(2).cast(IntegerType()).alias("epoch"),
            split_col.getItem(3).cast(IntegerType()).alias("moteid"),
            split_col.getItem(4).cast(DoubleType()).alias("temperature"),
            split_col.getItem(5).cast(DoubleType()).alias("humidity"),
            split_col.getItem(6).cast(DoubleType()).alias("light"),
            split_col.getItem(7).cast(DoubleType()).alias("voltage"),
        )
        # Bỏ rows mà cast thất bại (null sau cast)
        .filter(
            F.col("moteid").isNotNull() &
            F.col("temperature").isNotNull() &
            F.col("epoch").isNotNull()
        )
    )
    return parsed


def load_intel_lab_csv(spark: SparkSession, path: str) -> DataFrame:
    """
    Load file CSV từ Kaggle (có header).
    """
    schema = StructType([
        StructField("date",        StringType()),
        StructField("time_str",    StringType()),
        StructField("epoch",       IntegerType()),
        StructField("moteid",      IntegerType()),
        StructField("temperature", DoubleType()),
        StructField("humidity",    DoubleType()),
        StructField("light",       DoubleType()),
        StructField("voltage",     DoubleType()),
    ])
    return (
        spark.read
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .schema(schema)
        .csv(path)
        .filter(F.col("moteid").isNotNull() & F.col("temperature").isNotNull())
    )


def prepare_iot_logs(raw_df: DataFrame) -> DataFrame:
    """
    Biến đổi Intel Lab raw → schema chuẩn cho BloomJoin:
      batch_id      : "SENSOR_01" ... "SENSOR_54"  (key join)
      warehouse_id  : "ROOM_01"   ... "ROOM_54"    (partition col)
      temperature_c : float
      recorded_at   : "yyyy-MM-dd HH:mm:ss"
      humidity, light, voltage giữ nguyên để dùng trong analysis

    Lọc outliers:
      temperature > 100°C hoặc < -40°C
      → Intel Lab có known faulty readings khi voltage thấp (< 2.4V)
        nhiệt độ lên tới 122°C, rõ ràng là sensor lỗi
    """
    return (
        raw_df
        .filter(
            (F.col("temperature") > -40) &
            (F.col("temperature") < 100)
        )
        .select(
            # log_id: ghép epoch + moteid để unique
            F.concat(
                F.lit("LOG_"),
                F.lpad(F.col("moteid").cast("string"), 2, "0"),
                F.lit("_"),
                F.lpad(F.col("epoch").cast("string"), 7, "0"),
            ).alias("log_id"),

            # batch_id: key dùng để join với SensorMetadata
            F.concat(
                F.lit("SENSOR_"),
                F.lpad(F.col("moteid").cast("string"), 2, "0"),
            ).alias("batch_id"),

            # warehouse_id: partition col
            F.concat(
                F.lit("ROOM_"),
                F.lpad(F.col("moteid").cast("string"), 2, "0"),
            ).alias("warehouse_id"),

            # temperature
            F.col("temperature").cast(FloatType()).alias("temperature_c"),

            # timestamp
            F.concat_ws(" ", F.col("date"), F.col("time_str")).alias("recorded_at"),

            # giữ thêm để dùng trong analysis
            F.col("humidity").cast(FloatType()),
            F.col("light").cast(FloatType()),
            F.col("voltage").cast(FloatType()),
        )
    )


# ── Build SensorMetadata (bảng nhỏ, 54 rows) ────────────────────────────────

# Tọa độ thực tế 54 sensors, nguồn: http://db.csail.mit.edu/labdata/mote_locs.txt
MOTE_LOCATIONS = {
    1:(2.0,7.6),  2:(2.4,7.6),  3:(9.2,7.1),  4:(9.0,7.1),  5:(7.2,6.6),
    6:(9.2,5.1),  7:(7.5,5.0),  8:(7.5,4.0),  9:(7.5,3.0),  10:(6.5,3.0),
    11:(5.5,3.0), 12:(4.5,3.0), 13:(3.5,3.0), 14:(2.5,3.0), 15:(1.5,3.0),
    16:(0.5,3.0), 17:(2.0,2.0), 18:(4.0,2.0), 19:(6.0,2.0), 20:(8.0,2.0),
    21:(10.0,2.0),22:(10.0,4.0),23:(10.0,6.0),24:(10.0,8.0),25:(8.0,8.0),
    26:(6.0,8.0), 27:(4.0,8.0), 28:(2.0,8.0), 29:(0.0,8.0), 30:(0.0,6.0),
    31:(0.0,4.0), 32:(0.0,2.0), 33:(0.0,0.0), 34:(2.0,0.0), 35:(4.0,0.0),
    36:(6.0,0.0), 37:(8.0,0.0), 38:(10.0,0.0),39:(10.0,-2.0),40:(8.0,-2.0),
    41:(6.0,-2.0),42:(4.0,-2.0),43:(2.0,-2.0),44:(0.0,-2.0),45:(5.0,5.0),
    46:(5.0,7.0), 47:(3.0,7.0), 48:(3.0,5.0), 49:(1.0,7.0), 50:(1.0,5.0),
    51:(1.0,1.0), 52:(3.0,1.0), 53:(5.0,1.0), 54:(7.0,1.0),
}


def build_sensor_metadata(spark: SparkSession) -> DataFrame:
    """
    Bảng nhỏ 54 rows — metadata của từng sensor.
    Đóng vai 'Food_Batches' trong BloomJoin framework.
    """
    rows = []
    for moteid in range(1, 55):
        x, y = MOTE_LOCATIONS.get(moteid, (0.0, 0.0))

        # Phân loại room theo vị trí
        if moteid in range(1, 10):
            room = "lab_main"
            safe_min, safe_max = 18.0, 30.0
        elif moteid in range(10, 20):
            room = "corridor"
            safe_min, safe_max = 17.0, 32.0
        elif moteid in range(20, 35):
            room = "office"
            safe_min, safe_max = 18.0, 30.0
        elif moteid in range(35, 45):
            room = "storage"
            safe_min, safe_max = 10.0, 28.0
        else:
            room = "server_room"
            safe_min, safe_max = 15.0, 35.0

        rows.append((
            f"SENSOR_{moteid:02d}",   # batch_id
            f"ROOM_{moteid:02d}",     # warehouse_id
            room,
            float(x), float(y),
            safe_min, safe_max,
        ))

    schema = StructType([
        StructField("batch_id",      StringType()),
        StructField("warehouse_id",  StringType()),
        StructField("room_type",     StringType()),
        StructField("x_loc",         FloatType()),
        StructField("y_loc",         FloatType()),
        StructField("safe_temp_min", FloatType()),
        StructField("safe_temp_max", FloatType()),
    ])
    return spark.createDataFrame(rows, schema)


# ── Benchmark Result ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkRow:
    benchmark:            str
    strategy:             str
    execution_mode:       str
    n_sensors:            int
    n_logs:               int
    exec_time_sec:        float
    joined_rows:          Optional[int]
    large_before_filter:  Optional[int]
    large_after_filter:   Optional[int]
    filter_reduction_pct: Optional[float]
    fpr_configured:       Optional[float]
    notes:                str = ""
    error:                str = ""


def run_one(
    spark: SparkSession,
    metadata: DataFrame,
    logs: DataFrame,
    strategy: str,
    execution_mode: str = "udf",
    partition_col: Optional[str] = None,
    fpr: float = 0.01,
    benchmark_label: str = "",
    n_sensors: int = 54,
    n_logs: int = 0,
) -> BenchmarkRow:

    bj = BloomJoin(
        spark=spark,
        expected_items=100,
        false_positive_rate=fpr,
        max_collect_distinct_keys=500,
    )

    metadata.cache(); metadata.count()
    logs.cache(); logs.count()

    start = time.perf_counter()
    try:
        result, metrics = bj.join(
            large_df=logs,
            small_df=metadata,
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
            n_sensors=n_sensors,
            n_logs=n_logs,
            exec_time_sec=elapsed,
            joined_rows=metrics.joined_rows,
            large_before_filter=metrics.large_rows_before_filter,
            large_after_filter=metrics.large_rows_after_filter,
            filter_reduction_pct=round(frr * 100, 1) if frr is not None else None,
            fpr_configured=fpr,
            notes=metrics.notes or "",
        )
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 3)
        return BenchmarkRow(
            benchmark=benchmark_label,
            strategy=strategy,
            execution_mode=execution_mode,
            n_sensors=n_sensors,
            n_logs=n_logs,
            exec_time_sec=elapsed,
            joined_rows=None,
            large_before_filter=None,
            large_after_filter=None,
            filter_reduction_pct=None,
            fpr_configured=fpr,
            error=str(e),
        )
    finally:
        metadata.unpersist()
        logs.unpersist()


# ── Benchmark A — Strategy Comparison ────────────────────────────────────────

def bench_A_strategy_comparison(
    spark: SparkSession,
    iot_logs: DataFrame,
    sensor_meta: DataFrame,
    n_logs: int,
) -> list:

    print("\n" + "="*60)
    print("BENCH A — Strategy Comparison (Intel Lab Real Data)")
    print(f"  Sensors (small) : {sensor_meta.count()} rows")
    print(f"  IoT logs (large): {n_logs:,} rows")
    print("="*60)

    strategies = [
        ("direct_join",       "udf",  None,           0.01),
        ("broadcast_hash",    "udf",  None,           0.01),
        ("global_bloom",      "udf",  None,           0.01),
        ("global_bloom",      "rdd",  None,           0.01),
        ("partitioned_bloom", "udf",  "warehouse_id", 0.01),
    ]

    results = []
    for strat, mode, pcol, fpr in strategies:
        label = f"{strat}/{mode}"
        print(f"  Running: {label:<30}", end="", flush=True)
        r = run_one(
            spark, sensor_meta, iot_logs,
            strat, mode, pcol, fpr,
            benchmark_label="A_strategy",
            n_sensors=54, n_logs=n_logs,
        )
        results.append(r)
        if r.error:
            print(f"ERROR: {r.error[:80]}")
        else:
            frr = f"{r.filter_reduction_pct}%" if r.filter_reduction_pct is not None else "N/A"
            print(f"→ {r.exec_time_sec}s | filter_reduction={frr} | joined={r.joined_rows}")

    return results


# ── Benchmark B — Anomaly Detection E2E ──────────────────────────────────────

def bench_B_anomaly_detection(
    spark: SparkSession,
    iot_logs: DataFrame,
    sensor_meta: DataFrame,
) -> None:

    print("\n" + "="*60)
    print("BENCH B — Anomaly Detection E2E (Intel Lab Real Data)")
    print("  Tìm sensor có nhiệt độ bất thường (< 10°C hoặc > 32°C)")
    print("="*60)

    # Pre-filter chỉ giữ readings bất thường
    anomalous = iot_logs.filter(
        (F.col("temperature_c") < 10.0) | (F.col("temperature_c") > 32.0)
    )

    total    = iot_logs.count()
    n_anomal = anomalous.count()
    print(f"  Total readings     : {total:,}")
    print(f"  Anomalous readings : {n_anomal:,}  ({n_anomal / max(total,1):.1%})")

    bj = BloomJoin(
        spark=spark,
        expected_items=100,
        false_positive_rate=0.01,
        max_collect_distinct_keys=500,
    )

    start = time.perf_counter()
    joined, metrics = bj.join(
        large_df=anomalous,
        small_df=sensor_meta,
        key_cols="batch_id",
        join_type="inner",
        strategy="global_bloom",
        collect_metrics=True,
    )

    # Xử lý duplicate warehouse_id
    joined = joined.drop(sensor_meta["warehouse_id"])

    # Aggregate per sensor
    report = (
        joined
        .groupBy("batch_id", "warehouse_id", "room_type", "x_loc", "y_loc",
                 "safe_temp_min", "safe_temp_max")
        .agg(
            F.count("*").alias("anomaly_count"),
            F.min("temperature_c").alias("min_temp"),
            F.max("temperature_c").alias("max_temp"),
            F.round(F.avg("temperature_c"), 2).alias("avg_temp"),
            F.min("recorded_at").alias("first_anomaly"),
            F.max("recorded_at").alias("last_anomaly"),
        )
        .orderBy(F.desc("anomaly_count"))
    )
    elapsed = round(time.perf_counter() - start, 3)

    n_faulty = report.count()
    print(f"\n  → Execution time      : {elapsed}s")
    print(f"  → Sensors với anomaly : {n_faulty} / 54")
    frr = metrics.filter_reduction_ratio
    if frr is not None:
        print(f"  → Filter reduction    : {frr:.1%}")

    print("\n  Top 15 sensors có nhiều anomaly nhất:")
    report.show(15, truncate=False)


# ── Print Summary ─────────────────────────────────────────────────────────────

def print_summary(results: list) -> None:
    print("\n" + "="*72)
    print("SUMMARY — Benchmark 2 (Intel Lab Real Data)")
    print("="*72)
    print(f"{'Bench':<10} {'Strategy':<22} {'Mode':<5} {'Time(s)':<10} {'Filter%':<10} {'Joined'}")
    print("-"*72)
    for r in results:
        frr    = f"{r.filter_reduction_pct}%" if r.filter_reduction_pct is not None else "N/A"
        joined = str(r.joined_rows) if r.joined_rows is not None else "N/A"
        err    = f"  ← ERROR: {r.error[:40]}" if r.error else ""
        print(f"{r.benchmark:<10} {r.strategy:<22} {r.execution_mode:<5} "
              f"{r.exec_time_sec:<10} {frr:<10} {joined}{err}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark 2 — Intel Lab Sensor Data")
    parser.add_argument(
        "--data_path", type=str,
        default=r"C:\Users\ASUS\Desktop\Bloom28\data.txt",
        help='Path tới file data. Ví dụ: "C:/Users/ASUS/data/data.txt"'
    )
    parser.add_argument(
        "--format", type=str, default="txt", choices=["txt", "csv"],
        help="txt = MIT space-separated (default) | csv = Kaggle với header"
    )
    args = parser.parse_args()

    # Normalize path cho Windows (backslash → forward slash)
    data_path = args.data_path.replace("\\", "/")

    if not os.path.exists(data_path):
        print(f"\nERROR: Không tìm thấy file: {data_path}")
        print("Download tại: https://www.kaggle.com/datasets/divyansh22/intel-berkeley-research-lab-sensor-data")
        sys.exit(1)

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\nSpark version : {spark.version}")
    print(f"Data path     : {data_path}")
    print(f"Format        : {args.format}")

    # Load raw data
    print("\nLoading Intel Lab Data...")
    if args.format == "txt":
        raw = load_intel_lab_txt(spark, data_path)
    else:
        raw = load_intel_lab_csv(spark, data_path)

    # Transform sang schema BloomJoin
    iot_logs     = prepare_iot_logs(raw)
    sensor_meta  = build_sensor_metadata(spark)

    # Đếm sau khi filter outliers
    raw_count  = raw.count()
    clean_count = iot_logs.count()
    print(f"Raw rows      : {raw_count:,}")
    print(f"Clean rows    : {clean_count:,}  (sau khi lọc outliers temperature > 100°C hoặc < -40°C)")
    print(f"Dropped       : {raw_count - clean_count:,}  ({(raw_count - clean_count)/max(raw_count,1):.1%} faulty readings)")
    print(f"Sensor count  : {sensor_meta.count()} sensors")

    # Cache clean logs một lần, dùng cho cả 2 bench
    iot_logs.cache()
    iot_logs.count()  # trigger cache

    all_results = []
    try:
        all_results += bench_A_strategy_comparison(spark, iot_logs, sensor_meta, clean_count)
        bench_B_anomaly_detection(spark, iot_logs, sensor_meta)
        print_summary(all_results)

        out_path = "benchmark_2_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2, ensure_ascii=False)
        print(f"\nResults saved → {out_path}")

    finally:
        iot_logs.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()