"""Join Food_Batches (small) with Temperature_Logs (large) to find batches exposed to dangerous temperatures.
"""
import sys

sys.path.insert(0, "/mnt/project")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from datetime import datetime, timedelta
import random

from bloom_join import BloomJoin
from adaptive_bloom_join import AdaptiveBloomPlanner


def create_spark():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("ColdChainBloomTest")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )


def generate_food_batches(spark, n_batches=200):
    """Small table: food batches across warehouses."""
    warehouses = ["WH_HCM_01", "WH_HN_02", "WH_DN_03", "WH_HP_04"]
    categories = ["dairy", "meat", "seafood", "vegetables", "fruit"]

    rows = []
    for i in range(n_batches):
        rows.append((
            f"BATCH_{i:05d}",
            random.choice(warehouses),
            random.choice(categories),
            (datetime.now() + timedelta(days=random.randint(1, 30))).isoformat(),
        ))

    schema = StructType([
        StructField("batch_id", StringType()),
        StructField("warehouse_id", StringType()),
        StructField("category", StringType()),
        StructField("expiry_date", StringType()),
    ])
    return spark.createDataFrame(rows, schema)


def generate_temperature_logs(spark, n_logs=5000):
    """
    Large table: sensor telemetry.
    ~10% of logs will reference batch IDs that exist in food_batches,
    rest are noise from other batches / sensors.
    """
    warehouses = ["WH_HCM_01", "WH_HN_02", "WH_DN_03", "WH_HP_04", "WH_CT_05"]
    base_time = datetime.now() - timedelta(hours=48)

    rows = []
    for i in range(n_logs):
        # 10% chance to match a real batch
        if random.random() < 0.10:
            batch_id = f"BATCH_{random.randint(0, 199):05d}"
        else:
            batch_id = f"BATCH_{random.randint(500, 9999):05d}"

        # ~20% of logs simulate a temperature breach (> 8°C for cold chain)
        if random.random() < 0.20:
            temp = round(random.uniform(8.1, 25.0), 1)
        else:
            temp = round(random.uniform(-2.0, 7.9), 1)

        rows.append((
            f"LOG_{i:07d}",
            batch_id,
            random.choice(warehouses),
            temp,
            (base_time + timedelta(minutes=5 * i // 10)).isoformat(),
        ))

    schema = StructType([
        StructField("log_id", StringType()),
        StructField("batch_id", StringType()),
        StructField("warehouse_id", StringType()),
        StructField("temperature_c", FloatType()),
        StructField("recorded_at", StringType()),
    ])
    return spark.createDataFrame(rows, schema)


# ── Tests ────────────────────────────────────────────────────────────


def test_global_bloom_join(spark):
    """Global bloom: join on batch_id only."""
    food = generate_food_batches(spark)
    logs = generate_temperature_logs(spark)

    bj = BloomJoin(
        spark=spark,
        expected_items=500,
        false_positive_rate=0.01,
        max_collect_distinct_keys=10_000,
    )

    result, metrics = bj.join(
        large_df=logs,
        small_df=food,
        key_cols="batch_id",
        join_type="inner",
        strategy="global_bloom",
        execution_mode="udf",
        collect_metrics=True,
    )

    print("\n=== Global Bloom Join ===")
    print(metrics.pretty())
    print(f"Result columns: {result.columns}")

    assert result.count() > 0, "Join should produce results"
    assert "temperature_c" in result.columns
    assert "category" in result.columns
    print("PASSED\n")


def test_partitioned_bloom_join(spark):
    """Partitioned bloom: join on batch_id, partitioned by warehouse_id."""
    food = generate_food_batches(spark)
    logs = generate_temperature_logs(spark)

    bj = BloomJoin(
        spark=spark,
        expected_items=500,
        false_positive_rate=0.01,
        max_collect_distinct_keys=10_000,
    )

    result, metrics = bj.join(
        large_df=logs,
        small_df=food,
        key_cols="batch_id",
        join_type="inner",
        strategy="partitioned_bloom",
        partition_col="warehouse_id",
        execution_mode="udf",
        collect_metrics=True,
    )

    print("=== Partitioned Bloom Join ===")
    print(metrics.pretty())

    assert result.count() > 0, "Partitioned join should produce results"
    print("PASSED\n")


def test_auto_strategy(spark):
    """Let the planner choose the strategy."""
    food = generate_food_batches(spark)
    logs = generate_temperature_logs(spark)

    planner = AdaptiveBloomPlanner(
        broadcast_row_threshold=50,   # force bloom (200 batches > 50)
        bloom_distinct_threshold=10_000,
    )

    bj = BloomJoin(
        spark=spark,
        expected_items=500,
        false_positive_rate=0.01,
        planner=planner,
    )

    result, metrics = bj.join(
        large_df=logs,
        small_df=food,
        key_cols="batch_id",
        join_type="inner",
        strategy="auto",
        collect_metrics=True,
    )

    print("=== Auto Strategy ===")
    print(metrics.pretty())

    assert metrics.strategy in ("global_bloom", "partitioned_bloom", "broadcast_hash", "semi_join", "direct_join")
    assert result.count() > 0
    print("PASSED\n")


def test_breach_detection_e2e(spark):
    """
    End-to-end: join -> filter breaches -> identify at-risk batches
    for discount rerouting.
    """
    food = generate_food_batches(spark)
    logs = generate_temperature_logs(spark)

    TEMP_THRESHOLD = 8.0

    # Pre-filter: only logs with temperature breach
    hot_logs = logs.filter(F.col("temperature_c") > TEMP_THRESHOLD)

    bj = BloomJoin(
        spark=spark,
        expected_items=500,
        false_positive_rate=0.01,
        max_collect_distinct_keys=10_000,
    )

    joined = bj.join(
        large_df=hot_logs,
        small_df=food,
        key_cols="batch_id",
        join_type="inner",
        strategy="global_bloom",
    )

    # Drop cột warehouse_id trùng từ food trước khi groupBy
    joined = joined.drop(food["warehouse_id"])

    # Aggregate: which batches had the worst breaches?
    at_risk = (
        joined
        .groupBy("batch_id", "warehouse_id", "category", "expiry_date")
        .agg(
            F.count("*").alias("breach_count"),
            F.max("temperature_c").alias("max_temp"),
            F.min("recorded_at").alias("first_breach"),
            F.max("recorded_at").alias("last_breach"),
        )
        .orderBy(F.desc("max_temp"))
    )

    print("=== Breach Detection E2E ===")
    print(f"Total at-risk batches: {at_risk.count()}")
    at_risk.show(10, truncate=False)

    assert "breach_count" in at_risk.columns
    assert "max_temp" in at_risk.columns
    print("PASSED\n")


if __name__ == "__main__":
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    try:
        test_global_bloom_join(spark)
        test_partitioned_bloom_join(spark)
        test_auto_strategy(spark)
        test_breach_detection_e2e(spark)
        print("All tests passed.")
    finally:
        spark.stop()
