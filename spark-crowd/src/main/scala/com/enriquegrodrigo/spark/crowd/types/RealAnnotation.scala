package com.enriquegrodrigo.spark.crowd.types

/**
 * Class for storing real annotations.
 *
 * @param example example for which the annotation is made
 * @param annotator annotator that made the annotation
 * @param value value of the annotation 
 * @author enrique.grodrigo
 * @version 0.1
 */
case class RealAnnotation(example: Long, annotator: Long, value: Double) 

