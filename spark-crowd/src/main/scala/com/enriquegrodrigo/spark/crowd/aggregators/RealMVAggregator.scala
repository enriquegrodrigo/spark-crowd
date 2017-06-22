
package com.enriquegrodrigo.spark.crowd.aggregators

import com.enriquegrodrigo.spark.crowd.types.RealAnnotation
import org.apache.spark.sql.{Encoder,Encoders}
import org.apache.spark.sql.expressions.Aggregator

private[crowd] class RealMVAggregator extends Aggregator[RealAnnotation, RealMVPartial, Double]{

  def zero: RealMVPartial = RealMVPartial(0,0)

  def reduce(b: RealMVPartial, a: RealAnnotation) : RealMVPartial = 
    RealMVPartial(b.aggValue + a.value, b.count + 1) 

  def merge(b1: RealMVPartial, b2: RealMVPartial) : RealMVPartial = 
    RealMVPartial(b1.aggValue + b2.aggValue, b1.count + b2.count) 

  def finish(reduction: RealMVPartial) = {
    if (reduction.aggValue == 0) 
      throw new IllegalArgumentException()
    else 
      reduction.aggValue / reduction.count 
  }

  def bufferEncoder: Encoder[RealMVPartial] = Encoders.product[RealMVPartial]

  def outputEncoder: Encoder[Double] = Encoders.scalaDouble
}


