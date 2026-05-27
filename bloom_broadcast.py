from __future__ import annotations

from pyspark import Broadcast
from pyspark.sql import SparkSession


class BloomBroadcast:
    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark

    def broadcast(self, obj) -> Broadcast:
        return self.spark.sparkContext.broadcast(obj)

    def unpersist(self, bc: Broadcast, blocking: bool = False) -> None:
        if bc is not None:
            bc.unpersist(blocking=blocking)

    def destroy(self, bc: Broadcast, blocking: bool = False) -> None:
        if bc is not None:
            bc.destroy(blocking=blocking)