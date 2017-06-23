
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame
import org.apache.spark.broadcast.Broadcast

/**
 * Class for storing the results from the Kajino method [[com.enriquegrodrigo.spark.crowd.methods.Kajino]].
 *
 * @param dataset feature dataframe 
 * @param annotatorData annotation dataset 
 * @param w0 weights for the general logistic regression model
 * @param w weights for each annotator logistic regression model
 * @param priors values for the priors parameters [[com.enriquegrodrigo.spark.crowd.types.KajinoPriors]]
 * @param variation variation in the componentes for the logistic regression general model 
 * @param nAnnotators the number of annotators in the annotation dataset 
 * @param nFeatures the number of features in the feature dataset 
 * @author enrique.grodrigo
 * @version 0.1
 */
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

