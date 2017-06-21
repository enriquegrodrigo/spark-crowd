
package com.enriquegrodrigo.spark.crowd.aggregators 

import org.apache.spark.sql.{Encoder, Encoders}
import com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation
import org.apache.spark.sql.expressions.Aggregator

private[crowd] class BinaryMVAggregator extends Aggregator[BinaryAnnotation, BinaryMVPartial, Int]{
  
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
      else if ( (numerator / denominator) >= 0.5 ) 
        1 
      else 
        0
    }
  
    def bufferEncoder: Encoder[BinaryMVPartial] = Encoders.product[BinaryMVPartial]
  
    def outputEncoder: Encoder[Int] = Encoders.scalaInt
  }


