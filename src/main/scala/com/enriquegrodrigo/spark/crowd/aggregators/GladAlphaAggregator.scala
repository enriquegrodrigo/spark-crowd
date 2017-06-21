
package com.enriquegrodrigo.spark.crowd.aggregators 

import com.enriquegrodrigo.spark.crowd.types.GladPartial
import com.enriquegrodrigo.spark.crowd.types.GladParams
import com.enriquegrodrigo.spark.crowd.utils.Functions
import org.apache.spark.sql.{Encoder,Encoders}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.broadcast.Broadcast

private[crowd] class GladAlphaAggregator(params: Broadcast[GladParams], learningRate: Double) 
  extends Aggregator[GladPartial, GladAlphaAggregatorBuffer, Double]{

  def zero: GladAlphaAggregatorBuffer = GladAlphaAggregatorBuffer(0,-1) 
  
  def reduce(b: GladAlphaAggregatorBuffer, a: GladPartial) : GladAlphaAggregatorBuffer = {
    val alpha = params.value.alpha
    val al = alpha(a.annotator)
    val bet = a.beta
    val aest = a.est
    val sigmoidValue = Functions.sigmoid(alpha(a.annotator)*a.beta)
    val p = if (a.value == 1) a.est else (1-a.est)
    val term = (p - sigmoidValue)*bet
    GladAlphaAggregatorBuffer(b.agg + term, al)
  }

  def merge(b1: GladAlphaAggregatorBuffer, b2: GladAlphaAggregatorBuffer) : GladAlphaAggregatorBuffer = { 
    GladAlphaAggregatorBuffer(b1.agg + b2.agg, if (b1.alpha == -1) b2.alpha else b1.alpha )
  }

  def finish(reduction: GladAlphaAggregatorBuffer) = {
    reduction.alpha + learningRate * reduction.agg
  }

  def bufferEncoder: Encoder[GladAlphaAggregatorBuffer] = Encoders.product[GladAlphaAggregatorBuffer]

  def outputEncoder: Encoder[Double] = Encoders.scalaDouble
}

private[crowd] case class GladAlphaAggregatorBuffer(agg: Double, alpha: Double)
