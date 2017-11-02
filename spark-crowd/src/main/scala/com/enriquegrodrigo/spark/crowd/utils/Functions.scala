
package com.enriquegrodrigo.spark.crowd.utils

import scala.math.exp
import scala.math.log
import scala.math.pow

private[spark] object Functions {

  val COMPTHRES = pow(10, -5) 
  val BIGNUMBER = pow(10, 2) 

  def nearZero(x:Double): Boolean = (x < COMPTHRES) && (x > -COMPTHRES)

  def sigmoid(x: Double): Double = 1 / (1 + exp(-x))

  def prodlog(x:Double,l:Double) = if (nearZero(x)) 0 
                                   else x * logLim(l)
                                   
  def logLim(x:Double) = if ( nearZero(x) ) (-BIGNUMBER) else log(x)
}
