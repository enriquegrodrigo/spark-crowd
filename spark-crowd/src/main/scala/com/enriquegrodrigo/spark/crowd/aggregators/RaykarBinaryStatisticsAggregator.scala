
package com.enriquegrodrigo.spark.crowd.aggregators 

import com.enriquegrodrigo.spark.crowd.types.RaykarBinaryPartial
import com.enriquegrodrigo.spark.crowd.types.RaykarBinaryParams
import com.enriquegrodrigo.spark.crowd.utils.Functions
import org.apache.spark.sql.{Encoder,Encoders}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.broadcast.Broadcast

private[crowd] class RaykarBinaryStatisticsAggregator(params: Broadcast[RaykarBinaryParams]) 
  extends Aggregator[RaykarBinaryPartial, RaykarBinaryStatisticsAggregatorBuffer, (Double,Double)]{

  def zero: RaykarBinaryStatisticsAggregatorBuffer = RaykarBinaryStatisticsAggregatorBuffer(1,1) //Binary
  
  def reduce(b: RaykarBinaryStatisticsAggregatorBuffer, a: RaykarBinaryPartial) : RaykarBinaryStatisticsAggregatorBuffer = {
    val alphaValue = params.value.alpha(a.annotator)
    val alphaTerm = if (a.value == 1) alphaValue else 1-alphaValue
    val betaValue = params.value.beta(a.annotator)
    val betaTerm = if (a.value == 0) betaValue else 1-betaValue 
    RaykarBinaryStatisticsAggregatorBuffer(b.a * alphaTerm, b.b * betaTerm)
  }

  def merge(b1: RaykarBinaryStatisticsAggregatorBuffer, b2: RaykarBinaryStatisticsAggregatorBuffer) : RaykarBinaryStatisticsAggregatorBuffer = { 
    RaykarBinaryStatisticsAggregatorBuffer(b1.a * b2.a, b1.b*b2.b)
  }

  def finish(reduction: RaykarBinaryStatisticsAggregatorBuffer) = {
    (reduction.a,reduction.b)
  }

  def bufferEncoder: Encoder[RaykarBinaryStatisticsAggregatorBuffer] = Encoders.product[RaykarBinaryStatisticsAggregatorBuffer]

  def outputEncoder: Encoder[(Double,Double)] = Encoders.product[(Double,Double)]
}

private[crowd] case class RaykarBinaryStatisticsAggregatorBuffer(a: Double, b: Double)
