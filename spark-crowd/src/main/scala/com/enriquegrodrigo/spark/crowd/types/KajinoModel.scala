
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
*  Kajino model returned by the Kajino method 
*
*  @param mu label estimation returned from the model.
*  @param w0 weights of the aggregated logistic regression model.
*  @param weights logistic regresion weights for each annotator.  
*  @param logLikelihood logLikelihood of the final estimation of the model.  
*  @author enrique.grodrigo
*  @version 0.1 
*/
class KajinoModel(mu: Dataset[BinarySoftLabel], 
                          w0: Array[Double],  
                          weights: Array[Array[Double]]) extends Model[BinarySoftLabel] {
                            
  /**
  *  Method that returns the probabilistic estimation of the true label 
  *
  *  @return [[org.apache.spark.sql.Dataset]]
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getMu(): Dataset[BinarySoftLabel] = mu 

  /**
  *  Method that returns the annotator precision information 
  *
  *  @return Double 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getAnnotatorWeights(annotator: Int): Array[Double] = weights(annotator) 

  /**
  *  Method that returns the weights of the logistic regresion model 
  *
  *  @return Double 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getModelWeights(): Array[Double] = w0 
}
