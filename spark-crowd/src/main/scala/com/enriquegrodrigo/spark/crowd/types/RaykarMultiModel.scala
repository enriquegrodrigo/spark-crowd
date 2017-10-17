
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
*  Raykar Multiclass model returned by the RaykarMulti method 
*
*  @param mu label estimation returned from the model.
*  @param prec Annotator precision object [[com.enriquegrodrigo.spark.crowd.types.DiscreteAnnotatorPrecision]]
*  @param weights logistic regresion weights.  
*  @param logLikelihood logLikelihood of the final estimation of the model.  
*  @author enrique.grodrigo
*  @version 0.1 
*/
class RaykarMultiModel(mu: Dataset[MulticlassSoftProb], 
                          prec: Dataset[DiscreteAnnotatorPrecision],  
                          weights: Array[Array[Double]],
                          logLikelihood: Double) extends Model[MulticlassSoftProb] {
                            
  /**
  *  Method that returns the probabilistic estimation of the true label 
  *
  *  @return [[org.apache.spark.sql.Dataset]]
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getMu(): Dataset[MulticlassSoftProb] = mu 

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
  def getAnnotatorPrecision(): Dataset[DiscreteAnnotatorPrecision] = prec 

  /**
  *  Method that returns the weights of the logistic regresion model for a specific class
  *
  *  @return Double 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getModelWeights(c: Int): Array[Double] =  weights(c) 
}
