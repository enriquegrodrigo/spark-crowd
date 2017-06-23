
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
 * Class for storing the paratial results from the Glad method [[com.enriquegrodrigo.spark.crowd.methods.Glad]].
 *
 * @param dataset Annotations with the ground truth estimation [[com.enriquegrodrigo.spark.crowd.types.GladPartial]]
 * @param params model parameters, which include the annotator reliability and the class weights [[com.enriquegrodrigo.spark.crowd.types.GladParams]]
 * @param logLikelihood likelihood of the resulting model 
 * @param improvement improvement in the likelihood of the model 
 * @param nAnnotators number of annotators present in the dataset 
 * @author enrique.grodrigo
 * @version 0.1
 */
private[crowd] case class GladPartialModel(dataset: Dataset[GladPartial], params: Broadcast[GladParams], 
                                  logLikelihood: Double, improvement: Double, 
                                  nAnnotators: Int) {

  def modify(nDataset: Dataset[GladPartial] =dataset, 
      nParams: Broadcast[GladParams] =params, 
      nLogLikelihood: Double =logLikelihood, 
      nImprovement: Double =improvement, 
      nNAnnotators: Int =nAnnotators) = 
        new GladPartialModel(nDataset, nParams, 
                                                nLogLikelihood, 
                                                nImprovement, 
                                                nNAnnotators)
}

