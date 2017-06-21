
package com.enriquegrodrigo.spark.crowd.aggregators 

import org.apache.spark.sql.{Encoder, Encoders}
import com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation
import org.apache.spark.sql.expressions.Aggregator

  private[crowd] class BinarySoftMVAggregator extends Aggregator[BinaryAnnotation, BinaryMVPartial, Double]{
  
    def zero: BinaryMVPartial = BinaryMVPartial(0,0)
  
    def reduce(b: BinaryMVPartial, a: BinaryAnnotation) : BinaryMVPartial = 
      BinaryMVPartial(b.aggValue+a.value, b.count + 1) 
  
    def merge(b1: BinaryMVPartial, b2: BinaryMVPartial) : BinaryMVPartial = 
      BinaryMVPartial(b1.aggValue + b2.aggValue, b1.count + b2.count) 
  
    def finish(reduction: BinaryMVPartial) =  {
      val numerator: Double = reduction.aggValue
      val denominator: Double = reduction.count
      if (denominator == 0) 
        throw new IllegalArgumentException() 
      else {
        (numerator / denominator)
      }
    }
  
    def bufferEncoder: Encoder[BinaryMVPartial] = Encoders.product[BinaryMVPartial]
  
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }


