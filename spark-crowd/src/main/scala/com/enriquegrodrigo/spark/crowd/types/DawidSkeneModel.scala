/*
 * MIT License 
 *
 * Copyright (c) 2017 Enrique González Rodrigo 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this 
 * software and associated documentation files (the "Software"), to deal in the Software 
 * without restriction, including without limitation the rights to use, copy, modify, 
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
 * persons to whom the Software is furnished to do so, subject to the following conditions: 
 *
 * The above copyright notice and this permission notice shall be included in all copies or 
 * substantial portions of the Software.  
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

/**
*  DawidSkene model returned by the DawidSkene method 
*
*  @param mu label estimation returned from the model.
*  @param prec dataset with annotator precision information 
*  @param logLikelihood logLikelihood of the final estimation of the model.  
*  @author enrique.grodrigo
*  @version 0.2 
*/
class DawidSkeneModel(mu: Dataset[MulticlassLabel], 
                          prec: DataFrame) extends Model[MulticlassLabel] {
                            
  /**
  *  Method that returns the probabilistic estimation of the true label 
  *
  *  @return org.apache.spark.sql.Dataset
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getMu(): Dataset[MulticlassLabel] = mu 

  /**
  *  Method that returns the annotator precision information 
  *
  *  @return Double 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def getAnnotatorPrecision(): Dataset[DiscreteAnnotatorPrecision] = {
    import prec.sparkSession.implicits._
    prec.select(col("annotator"), col("j") as "c", col("l") as "k", col("pi") as "prob").as[DiscreteAnnotatorPrecision]
  }
}
