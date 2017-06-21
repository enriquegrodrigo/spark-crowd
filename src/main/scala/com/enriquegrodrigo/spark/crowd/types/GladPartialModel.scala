
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast


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

