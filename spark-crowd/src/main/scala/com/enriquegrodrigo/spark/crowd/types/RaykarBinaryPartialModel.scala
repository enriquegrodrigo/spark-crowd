
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame
import org.apache.spark.broadcast.Broadcast


/**
 * Class for storing the partial results from the RaykarBinary method [[com.enriquegrodrigo.spark.crowd.methods.RaykarBinary]].
 *
 * @param dataset Dataframe with the feature data 
 * @param annotatorData Dataset of annotations 
 * @param mu estimation of the ground truth label for each example 
 * @param dataStatistics statistics precomputed about the data [[com.enriquegrodrigo.spark.crowd.types.RaykarBinaryStatistics]]
 * @param params model params as the realiability of annotators [[com.enriquegrodrigo.spark.crowd.types.RaykarBinaryParams]]
 * @param logLikelihood likelihood of the resulting model 
 * @param improvement improvement from last step of EM algorithm 
 * @param nAnnotators number of annotators in the annotation data 
 * @param nFeatures number of features in the dataset 
 * @author enrique.grodrigo
 * @version 0.1
 */
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

