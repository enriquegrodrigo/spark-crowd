
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast


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

