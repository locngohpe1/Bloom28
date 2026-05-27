from bloom_join import BloomJoin

bj = BloomJoin(
    spark=spark,
    expected_items=2_000_000,
    false_positive_rate=0.01,
    max_collect_distinct_keys=3_000_000,
)

result, metrics = bj.join(
    large_df=fact_df,
    small_df=dim_df,
    key_cols="security_id",
    join_type="inner",
    strategy="global_bloom",
    execution_mode="udf",
    collect_metrics=True,
)

print(metrics.pretty())