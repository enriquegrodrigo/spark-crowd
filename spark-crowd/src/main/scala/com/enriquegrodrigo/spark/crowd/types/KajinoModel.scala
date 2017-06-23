
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
 * Class for storing the results from the Kajino method [[com.enriquegrodrigo.spark.crowd.methods.Kajino]].
 *
 * @param dataset Ground truth values for the input annotation dataset 
 * @param w0 weights for the general logistic regression model. 
 * @param w weights for the annotators logistic regression model. 
 * @author enrique.grodrigo
 * @version 0.1
 */
private[crowd] case class KajinoModel(estimation: Dataset[BinarySoftLabel], w0: Array[Double], w: Array[Array[Double]]) 

