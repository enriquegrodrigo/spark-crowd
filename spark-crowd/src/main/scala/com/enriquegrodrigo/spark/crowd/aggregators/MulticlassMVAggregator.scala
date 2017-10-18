
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


