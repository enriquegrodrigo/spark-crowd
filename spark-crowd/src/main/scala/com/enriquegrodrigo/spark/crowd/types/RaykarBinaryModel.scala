
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
*  Raykar Binary model returned by the RaykarBinary method 
*
*  @param mu label estimation returned from the model.
*  @param alpha alpha estimation of the method.
*  @param beta beta estimation of the method.
*  @param weights logistic regresion weights.  
*  @param logLikelihood logLikelihood of the final estimation of the model.  
*  @author enrique.grodrigo
*  @version 0.1 
*/
class RaykarBinaryModel(mu: Dataset[BinarySoftLabel], 
                          alpha: Array[Double],  
                          beta: Array[Double],  
                          weights: Array[Double],
                          logLikelihood: Double) extends Model[BinarySoftLabel] {

  /**
  *  Method that returns the probabilistic estimation of the true label 
  *
  *  @return org.apache.spark.sql.Dataset
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getMu(): Dataset[BinarySoftLabel] = mu 

  /**
  *  Method that returns the likelihood of the model 
  *
  *  @return Double 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getLogLikelihood(): Double = logLikelihood 

  /**
  *  Method that returns the annotator precision information 
  *
  *  @return Double 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getAnnotatorPrecision(): Tuple2[Array[Double], Array[Double]] = (alpha,beta)

  /**
  *  Method that returns the weights of the logistic regresion model 
  *
  *  @return Double 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getModelWeights(): Array[Double] = weights 
}
