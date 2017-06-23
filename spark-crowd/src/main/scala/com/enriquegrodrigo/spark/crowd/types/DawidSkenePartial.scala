
package com.enriquegrodrigo.spark.crowd.types

/**
 * Class for storing the DawidSkene method partial annotation information [[com.enriquegrodrigo.spark.crowd.methods.DawidSkene]].
 *
 * @param example example to which the information refers 
 * @param annotator annotator to which the record refers
 * @param value value of the annotation 
 * @param est estimation of the ground truth label 
 * @author enrique.grodrigo
 * @version 0.1
 */
private[crowd] case class DawidSkenePartial(example: Long, annotator: Long, value: Int, est: Int)
