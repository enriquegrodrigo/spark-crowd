
package com.enriquegrodrigo.spark.crowd.aggregators 

import com.enriquegrodrigo.spark.crowd.types.GladPartial
import com.enriquegrodrigo.spark.crowd.types.GladParams
import com.enriquegrodrigo.spark.crowd.utils.Functions
import org.apache.spark.sql.{Encoder,Encoders}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.broadcast.Broadcast

import scala.math.log

private[crowd] class GladLogLikelihoodAggregator(params: Broadcast[GladParams]) 
  extends Aggregator[GladPartial, GladLogLikelihoodAggregatorBuffer, Double]{

  def zero: GladLogLikelihoodAggregatorBuffer = GladLogLikelihoodAggregatorBuffer(0,-1)

  def reduce(b: GladLogLikelihoodAggregatorBuffer, a: GladPartial) : GladLogLikelihoodAggregatorBuffer = {
    val alphaVal = params.value.alpha(a.annotator.toInt)
    val betaVal = a.beta
    val sig = Functions.sigmoid(alphaVal*betaVal) 
    val p0 = 1-a.est
    val p1 = a.est
    val k0 = if (a.value == 0) sig else 1-sig 
    val k1 = if (a.value == 1) sig else 1-sig 
    GladLogLikelihoodAggregatorBuffer(b.agg + Functions.prodlog(p0,k0) 
                                          + Functions.prodlog(p1,k1), p1) 
  }

  def merge(b1: GladLogLikelihoodAggregatorBuffer, b2: GladLogLikelihoodAggregatorBuffer) : GladLogLikelihoodAggregatorBuffer = { 
    GladLogLikelihoodAggregatorBuffer(b1.agg + b2.agg, if (b1.classProb == -1) b2.classProb else b1.classProb)
  }

  def finish(reduction: GladLogLikelihoodAggregatorBuffer) =  {
    val agg = reduction.agg
    val w0 = params.value.w(0)
    val w1 = params.value.w(1)
    val lastVal = reduction.agg + Functions.prodlog((1-reduction.classProb),params.value.w(0)) + 
                      Functions.prodlog(reduction.classProb,params.value.w(1))
    lastVal
  }


  def bufferEncoder: Encoder[GladLogLikelihoodAggregatorBuffer] = Encoders.product[GladLogLikelihoodAggregatorBuffer]

  def outputEncoder: Encoder[Double] = Encoders.scalaDouble
}

private[crowd] case class GladLogLikelihoodAggregatorBuffer(agg: Double, classProb: Double)
