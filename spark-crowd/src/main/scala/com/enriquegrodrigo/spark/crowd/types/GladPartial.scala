
package com.enriquegrodrigo.spark.crowd.types

/**
 * Class for storing the Glad method partial annotation information [[com.enriquegrodrigo.spark.crowd.methods.DawidSkene]].
 *
 * @param example example to which the information refers 
 * @param annotator annotator to which the record refers
 * @param value value of the annotation 
 * @param est estimation of the ground truth label 
 * @param beta difficulty of the example 
 * @author enrique.grodrigo
 * @version 0.1
 */
private[crowd] case class GladPartial(example: Long, annotator: Int, value: Int, est: Double, beta: Double)
