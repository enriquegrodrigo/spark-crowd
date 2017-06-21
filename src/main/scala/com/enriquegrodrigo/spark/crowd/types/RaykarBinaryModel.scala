
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast


case class RaykarBinaryModel(estimation: Dataset[BinarySoftLabel], params: RaykarBinaryParams, logLikelihood: Double) 


