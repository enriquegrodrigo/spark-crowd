
package com.enriquegrodrigo.spark.crowd.aggregators 

import com.enriquegrodrigo.spark.crowd.types.GladPartial
import com.enriquegrodrigo.spark.crowd.types.GladParams
import com.enriquegrodrigo.spark.crowd.utils.Functions
import org.apache.spark.sql.{Encoder,Encoders}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.broadcast.Broadcast

private[crowd] class GladBetaAggregator(params: Broadcast[GladParams], learningRate: Double) 
  extends Aggregator[GladPartial, GladBetaAggregatorBuffer, Double]{

  def zero: GladBetaAggregatorBuffer = GladBetaAggregatorBuffer(0,-1) 
  
  def reduce(b: GladBetaAggregatorBuffer, a: GladPartial) : GladBetaAggregatorBuffer = {
    val alpha = params.value.alpha
    val al = alpha(a.annotator)
    val bet = a.beta
    val aest = a.est
    val sigmoidValue = Functions.sigmoid(alpha(a.annotator)*a.beta)
    val p = if (a.value == 1) a.est else (1-a.est)
    val term = (p - sigmoidValue)*alpha(a.annotator)
    GladBetaAggregatorBuffer(b.agg + term, a.beta)
  }

  def merge(b1: GladBetaAggregatorBuffer, b2: GladBetaAggregatorBuffer) : GladBetaAggregatorBuffer = { 
    GladBetaAggregatorBuffer(b1.agg + b2.agg, if (b1.beta == -1) b2.beta else b1.beta)
  }

  def finish(reduction: GladBetaAggregatorBuffer) = {
    reduction.beta + learningRate * reduction.agg
  }

  def bufferEncoder: Encoder[GladBetaAggregatorBuffer] = Encoders.product[GladBetaAggregatorBuffer]

  def outputEncoder: Encoder[Double] = Encoders.scalaDouble
}

private[crowd] case class GladBetaAggregatorBuffer(agg: Double, beta: Double)
