
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

private[crowd] case class RaykarBinaryStatistics(example: Long, a: Double, b: Double)

