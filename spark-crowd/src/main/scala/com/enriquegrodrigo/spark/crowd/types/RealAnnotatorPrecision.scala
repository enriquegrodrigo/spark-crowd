
package com.enriquegrodrigo.spark.crowd.types

/**
*  Given a normal probability for each N("true label",1/lambda), lambda 
*  is the parameter that represents the precision of the annotator 
*  labelling.  
*
*  @param annotator annotator of the relation 
*  @param lambda the precision of the annotator
*  @author enrique.grodrigo
*  @version 0.1 
*/
case class RealAnnotatorPrecision(annotator: Long, lambda: Int)
