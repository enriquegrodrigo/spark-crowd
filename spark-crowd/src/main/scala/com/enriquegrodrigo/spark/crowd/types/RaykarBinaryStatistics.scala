
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
 * Class for storing the RaykarBinary annotation statistics needed for the learning process [[com.enriquegrodrigo.spark.crowd.methods.RaykarBinary]].
 *
 * @param example example to which the information refers 
 * @param a a statistic 
 * @param b b statistic 
 * @author enrique.grodrigo
 * @version 0.1
 */
private[crowd] case class RaykarBinaryStatistics(example: Long, a: Double, b: Double)

