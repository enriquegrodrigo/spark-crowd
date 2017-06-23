
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
 * Class for storing the results from the DawidSkene method [[com.enriquegrodrigo.spark.crowd.methods.Glad]].
 *
 * @param dataset Ground truth values probabilities for the input annotation dataset 
 * @param params model parameters, which include the annotator reliability and the class weights [[com.enriquegrodrigo.spark.crowd.types.GladParams]]
 * @param logLikelihood likelihood of the resulting model 
 * @author enrique.grodrigo
 * @version 0.1
 */
case class GladModel(dataset: Dataset[BinarySoftLabel], 
                      params: GladParams, logLikelihood: Double)

