
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset
import org.apache.spark.broadcast.Broadcast

/**
 * Class for storing the RaykarBinary method params [[com.enriquegrodrigo.spark.crowd.methods.RaykarBinary]].
 *
 * @param alpha Reliability in positive class 
 * @param beta Reliability in negative class 
 * @param weights of the logistic regression model
 * @param a prior for the reliability in positive class 
 * @param b prior for the reliability in negative class 
 * @param wp prior for the logistic regression model weights
 * @author enrique.grodrigo
 * @version 0.1
 */
case class RaykarBinaryParams(alpha: Array[Double], beta: Array[Double], w: Array[Double], 
                                    a: Array[Array[Double]], b: Array[Array[Double]], wp: Array[Array[Double]])

