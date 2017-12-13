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

import com.enriquegrodrigo.spark.crowd.types.MulticlassAnnotation
import org.apache.spark.sql.{Encoder,Encoders}
import org.apache.spark.sql.expressions.Aggregator

private[crowd] class MulticlassMVAggregator(nClasses: Int) extends Aggregator[MulticlassAnnotation, MulticlassMVPartial, Int]{

  def zero: MulticlassMVPartial = MulticlassMVPartial(Vector.fill(nClasses)(0),0)

  def reduce(b: MulticlassMVPartial, a: MulticlassAnnotation) : MulticlassMVPartial = {
    MulticlassMVPartial(b.aggVect.updated(a.value, b.aggVect(a.value) + 1), b.count + 1) 
  }

  def merge(b1: MulticlassMVPartial, b2: MulticlassMVPartial) : MulticlassMVPartial = { 
    MulticlassMVPartial(b1.aggVect.zip(b2.aggVect).map(x => x._1 + x._2), b1.count + b2.count) 
  }

  def finish(reduction: MulticlassMVPartial) = {
      reduction.aggVect.indexOf(reduction.aggVect.max)
  }


  def bufferEncoder: Encoder[MulticlassMVPartial] = Encoders.product[MulticlassMVPartial]

  def outputEncoder: Encoder[Int] = Encoders.scalaInt
}


