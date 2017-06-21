
package com.enriquegrodrigo.spark.crowd.aggregators 

import com.enriquegrodrigo.spark.crowd.types.GladPartial
import com.enriquegrodrigo.spark.crowd.types.GladParams
import com.enriquegrodrigo.spark.crowd.utils.Functions
import org.apache.spark.sql.{Encoder,Encoders}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.broadcast.Broadcast

import scala.math.{log,exp}

private[crowd] class GladEAggregator(params: Broadcast[GladParams]) 
  extends Aggregator[GladPartial, GladEAggregatorBuffer, Double]{

  def zero: GladEAggregatorBuffer = GladEAggregatorBuffer(Vector.fill(2)(1)) //Binary
  
  def reduce(b: GladEAggregatorBuffer, a: GladPartial) : GladEAggregatorBuffer = {
    val alpha = params.value.alpha
    val sigmoidValue = Functions.sigmoid(alpha(a.annotator)*a.beta)
    val p0 = if (a.value == 0) sigmoidValue else (1 - sigmoidValue)
    val p1 = if (a.value == 1) sigmoidValue else (1 - sigmoidValue) 
    GladEAggregatorBuffer(Vector(Functions.logLim(p0) + b.aggVect(0), Functions.logLim(p1) + b.aggVect(1)))
  }

  def merge(b1: GladEAggregatorBuffer, b2: GladEAggregatorBuffer) : GladEAggregatorBuffer = { 
    GladEAggregatorBuffer(b1.aggVect.zip(b2.aggVect).map(x => x._1 * x._2))
  }

  def finish(reduction: GladEAggregatorBuffer) = {
    val w = params.value.w
    val negative = exp(reduction.aggVect(0) + Functions.logLim(w(0)))
    val positive = exp(reduction.aggVect(1) + Functions.logLim(w(1)))
    val norm = negative + positive
    positive/norm
  }

  def bufferEncoder: Encoder[GladEAggregatorBuffer] = Encoders.product[GladEAggregatorBuffer]

  def outputEncoder: Encoder[Double] = Encoders.scalaDouble
}

private[crowd] case class GladEAggregatorBuffer(aggVect: scala.collection.Seq[Double])
