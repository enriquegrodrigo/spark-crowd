
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset

/**
 * Class for storing the results from the DawidSkene method [[com.enriquegrodrigo.spark.crowd.methods.DawidSkene]].
 *
 * @param dataset Ground truth values for the input annotation dataset 
 * @param params model parameters, which include the annotator reliability and the class weights [[com.enriquegrodrigo.spark.crowd.types.DawidSkeneParams]]
 * @param logLikelihood likelihood of the resulting model 
 * @author enrique.grodrigo
 * @version 0.1
 */
case class DawidSkeneModel(dataset: Dataset[MulticlassLabel], 
                                    params: DawidSkeneParams, 
                                    logLikelihood: Double) 

