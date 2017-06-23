
package com.enriquegrodrigo.spark.crowd.types
/**
 * Class for storing the RaykarBinary method partial annotation information [[com.enriquegrodrigo.spark.crowd.methods.RaykarBinary]].
 *
 * @param example example to which the information refers 
 * @param annotator annotator to which the record refers
 * @param value value of the annotation 
 * @param mu estimation of the ground truth label 
 * @author enrique.grodrigo
 * @version 0.1
 */
private[crowd] case class RaykarBinaryPartial(example: Long, annotator: Int, value: Int, mu: Double)
