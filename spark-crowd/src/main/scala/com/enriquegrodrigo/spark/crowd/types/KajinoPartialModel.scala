
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame
import org.apache.spark.broadcast.Broadcast


private[crowd] case class KajinoPartialModel(dataset: DataFrame, annotatorData: Dataset[BinaryAnnotation], 
                                    w0: Broadcast[Array[Double]], w: Array[Array[Double]], priors: Broadcast[KajinoPriors], 
                                    variation: Double, nAnnotators: Int, nFeatures: Int) {

  def modify(nDataset: DataFrame =dataset, 
      nAnnotatorData: Dataset[BinaryAnnotation] =annotatorData, 
      nW0: Broadcast[Array[Double]] =w0, 
      nW: Array[Array[Double]] = w, 
      nPriors: Broadcast[KajinoPriors] =priors, 
      nVariation: Double =variation, 
      nNAnnotators: Int =nAnnotators, 
      nNFeatures: Int =nFeatures) = 
      new KajinoPartialModel(nDataset, nAnnotatorData, nW0, nW, nPriors,
        nVariation, nNAnnotators, nNFeatures)
}

