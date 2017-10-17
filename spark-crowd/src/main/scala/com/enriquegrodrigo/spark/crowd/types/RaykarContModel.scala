
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
*  Raykar Continuous model returned by the RaykarCont method 
*
*  @param mu label estimation returned from the model.
*  @param lambda annotator precision estimation.
*  @param weights logistic regresion weights.  
*  @param logLikelihood logLikelihood of the final estimation of the model.  
*  @author enrique.grodrigo
*  @version 0.1 
*/
class RaykarContModel(mu: Dataset[RealLabel], 
                          lambdas: Dataset[RealLabel],  
                          weights: Array[Double],
                          logLikelihood: Double) extends Model[RealLabel] {
                            
  /**
  *  Method that returns the estimation of the true label 
  *
  *  @return [[org.apache.spark.sql.Dataset]]
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getMu(): Dataset[RealLabel] = mu 

  /**
  *  Method that returns the mean square error of the model 
  *
  *  @return Double 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getLogLikelihood(): Double = logLikelihood 

  /**
  *  Method that returns the annotator precision as a real parameter 
  *
  *  @return Double 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getAnnotatorPrecision(): Dataset[RealLabel] = lambdas

  /**
  *  Method that returns the weights of the linear regresion model 
  *
  *  @return Double 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getModelWeights(): Array[Double] = weights 
}
