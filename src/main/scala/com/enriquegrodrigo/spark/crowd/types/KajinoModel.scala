
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast


private[crowd] case class KajinoModel(estimation: Dataset[BinarySoftLabel], w0: Array[Double], w: Array[Array[Double]]) 

