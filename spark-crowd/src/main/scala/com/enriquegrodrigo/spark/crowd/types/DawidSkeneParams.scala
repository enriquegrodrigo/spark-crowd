
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

case class DawidSkeneParams(pi: Array[Array[Array[Double]]], w: Array[Double])

