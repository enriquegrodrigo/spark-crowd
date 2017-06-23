
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
 * Class for storing the Glad method params [[com.enriquegrodrigo.spark.crowd.methods.Glad]].
 *
 * @param alpha reliability factor for each annotator 
 * @param w weight for each class
 * @author enrique.grodrigo
 * @version 0.1
 */
case class GladParams(alpha: Array[Double], w: Array[Double])

