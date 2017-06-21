
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

case class RaykarBinaryParams(alpha: Array[Double], beta: Array[Double], w: Array[Double], 
                                    a: Array[Array[Double]], b: Array[Array[Double]], wp: Array[Array[Double]])

