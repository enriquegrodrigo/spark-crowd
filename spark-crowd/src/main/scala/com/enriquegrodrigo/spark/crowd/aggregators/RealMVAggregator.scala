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


