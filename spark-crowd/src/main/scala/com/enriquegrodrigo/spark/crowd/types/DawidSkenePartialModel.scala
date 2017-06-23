
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
 * Class for storing the partials results from the DawidSkene method [[com.enriquegrodrigo.spark.crowd.methods.DawidSkene]].
 *
 * @param dataset Dataset with both annotations and ground truth estimation
 * @param params model parameters, which include the annotator reliability and the class weights [[com.enriquegrodrigo.spark.crowd.types.DawidSkeneParams]]
 * @param logLikelihood likelihood of the resulting model 
 * @param improvement improvement from the previous step 
 * @param nClasses number of classes in the annotation dataset
 * @param nAnnotators number of annotators in the annotation dataset 
 * @author enrique.grodrigo
 * @version 0.1
 */
private[crowd] case class DawidSkenePartialModel(dataset: Dataset[DawidSkenePartial], params: Broadcast[DawidSkeneParams], 
                                  logLikelihood: Double, improvement: Double, nClasses: Int, 
                                  nAnnotators: Long) {

  def modify(nDataset: Dataset[DawidSkenePartial] =dataset, 
      nParams: Broadcast[DawidSkeneParams] =params, 
      nLogLikelihood: Double =logLikelihood, 
      nImprovement: Double =improvement, 
      nNClasses: Int =nClasses, 
      nNAnnotators: Long =nAnnotators) = 
        new DawidSkenePartialModel(nDataset, nParams, 
                                                nLogLikelihood, 
                                                nImprovement, 
                                                nNClasses, 
                                                nNAnnotators)
}

