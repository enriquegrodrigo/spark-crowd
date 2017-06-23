
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
 * Class for storing the DawidSkene method params [[com.enriquegrodrigo.spark.crowd.methods.DawidSkene]].
 *
 * @param pi confusion matrix for each annotator 
 * @param w weight for each class
 * @author enrique.grodrigo
 * @version 0.1
 */
case class DawidSkeneParams(pi: Array[Array[Array[Double]]], w: Array[Double])

