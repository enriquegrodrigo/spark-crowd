
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame
import org.apache.spark.broadcast.Broadcast



private[crowd] case class RaykarBinaryPartialModel(dataset: DataFrame, annotatorData: Dataset[BinaryAnnotation], 
                                    mu: Dataset[BinarySoftLabel], dataStatistics: Dataset[RaykarBinaryStatistics],
                                    params: Broadcast[RaykarBinaryParams], logLikelihood: Double, 
                                    improvement: Double, nAnnotators: Int, nFeatures: Int) {

  def modify(nDataset: DataFrame =dataset, 
      nAnnotatorData: Dataset[BinaryAnnotation] =annotatorData, 
      nMu: Dataset[BinarySoftLabel] =mu, 
      nDataStatistics: Dataset[RaykarBinaryStatistics] = dataStatistics, 
      nParams: Broadcast[RaykarBinaryParams] =params, 
      nLogLikelihood: Double =logLikelihood, 
      nImprovement: Double =improvement, 
      nNAnnotators: Int =nAnnotators, 
      nNFeatures: Int =nFeatures) = 
        new RaykarBinaryPartialModel(nDataset, nAnnotatorData, nMu, nDataStatistics, 
          nParams, nLogLikelihood, nImprovement, nNAnnotators, nNFeatures)
}

