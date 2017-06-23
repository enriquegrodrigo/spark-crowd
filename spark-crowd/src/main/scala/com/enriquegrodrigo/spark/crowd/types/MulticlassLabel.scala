
package com.enriquegrodrigo.spark.crowd.types

/**
 * Class for storing a multiclass label.
 *
 * @param example example to which label refers to  
 * @param value value of the label
 * @author enrique.grodrigo
 * @version 0.1
 */
case class MulticlassLabel(example: Long, value: Int) 


