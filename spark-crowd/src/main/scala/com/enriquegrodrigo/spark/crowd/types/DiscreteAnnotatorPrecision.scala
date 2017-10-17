
package com.enriquegrodrigo.spark.crowd.types

/**
*  Probability that the annotator "annotator" labels an example 
*  with true class "c" as "k".
*
*  @param annotator annotator of the relation 
*  @param c true class 
*  @param k labeled class 
*  @author enrique.grodrigo
*  @version 0.1 
*/
case class DiscreteAnnotatorPrecision(annotator: Long, c: Int, k: Int, prob: Double)
