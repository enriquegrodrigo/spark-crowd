/*
 * MIT License 
 *
 * Copyright (c) 2017 Enrique González Rodrigo 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this 
 * software and associated documentation files (the "Software"), to deal in the Software 
 * without restriction, including without limitation the rights to use, copy, modify, 
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
 * persons to whom the Software is furnished to do so, subject to the following conditions: 
 *
 * The above copyright notice and this permission notice shall be included in all copies or 
 * substantial portions of the Software.  
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

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


