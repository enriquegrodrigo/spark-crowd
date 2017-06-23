package com.enriquegrodrigo.spark.crowd.types

/**
 * Class for storing the ground thruth estimation of the Kajino method [[com.enriquegrodrigo.spark.crowd.methods.Kajino]].
 *
 * @param example example to which the information refers 
 * @param est estimation of the ground truth label 
 * @author enrique.grodrigo
 * @version 0.1
 */
private[crowd] case class KajinoEstimation(example: Long, est: Double)


