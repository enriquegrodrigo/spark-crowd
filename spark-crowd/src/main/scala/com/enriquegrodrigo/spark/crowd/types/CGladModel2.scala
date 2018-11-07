package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql._
import org.apache.spark.broadcast.Broadcast

/**
*  Glad model returned by the Glad method 
*
*  @param mu label estimation returned from the model.
*  @param alphas precision of annotator given by the Glad model.
*  @param betas instance difficulty given by Glad model. 
*  @param logLikelihood logLikelihood of the final estimation of the model.  
*  @author enrique.grodrigo
*  @version 0.2 
*/
class CGladModel2(mu: Dataset[BinarySoftLabel], 
                          prec: Array[Double], 
                          diffic: Array[Double], 
                          clusters: DataFrame
                          ) extends Model[BinarySoftLabel] {
                            
  /**
  *  Method that returns the probabilistic estimation of the true label 
  *
  *  @return org.apache.spark.sql.Dataset
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
  def getAnnotatorPrecision(): Array[Double] = prec 

  /**
  *  Method that returns information about instance difficulty
  *
  *  @return Double 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getInstanceDifficulty(): Array[Double] = diffic 
  def getInstanceCluster(): DataFrame = clusters 

}
