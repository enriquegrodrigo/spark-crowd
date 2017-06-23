
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
 * Class for storing the results from the RaykarBinary method [[com.enriquegrodrigo.spark.crowd.methods.RaykarBinary]].
 *
 * @param estimation Ground truth values for the input annotation dataset 
 * @param params model parameters, which include the annotator reliability and the logistic regression coefficients [[com.enriquegrodrigo.spark.crowd.types.RaykarBinaryParams]]
 * @param logLikelihood likelihood of the resulting model 
 * @author enrique.grodrigo
 * @version 0.1
 */
case class RaykarBinaryModel(estimation: Dataset[BinarySoftLabel], params: RaykarBinaryParams, logLikelihood: Double) 


