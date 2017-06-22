
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

private[crowd] case class RaykarBinaryExample(example: Long, mu: Double, featureVector: Array[Double], a: Double, b: Double, p: Double)
