
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
*  DawidSkene model returned by the DawidSkene method 
*
*  @param mu label estimation returned from the model.
*  @param prec dataset with annotator precision information 
*  @param logLikelihood logLikelihood of the final estimation of the model.  
*  @author enrique.grodrigo
*  @version 0.1 
*/
class DawidSkeneModel(mu: Dataset[MulticlassLabel], 
                          prec: Array[Array[Array[Double]]],
                          logLikelihood: Double) extends Model[MulticlassLabel] {
                            
  /**
  *  Method that returns the probabilistic estimation of the true label 
  *
  *  @return [[org.apache.spark.sql.Dataset]]
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getMu(): Dataset[MulticlassLabel] = mu 

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
  def getAnnotatorPrecision(): Array[Array[Array[Double]]] = prec 
}
