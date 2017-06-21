
package com.enriquegrodrigo.spark.crowd.aggregators 

import com.enriquegrodrigo.spark.crowd.types.DawidSkenePartial
import com.enriquegrodrigo.spark.crowd.types.DawidSkeneParams
import org.apache.spark.sql.{Encoder,Encoders}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.broadcast.Broadcast

private[crowd] class DawidSkeneEAggregator(params: Broadcast[DawidSkeneParams], nClasses: Int) 
  extends Aggregator[DawidSkenePartial, DawidSkeneAggregatorBuffer, Int]{

  def zero: DawidSkeneAggregatorBuffer = DawidSkeneAggregatorBuffer(Vector.fill(nClasses)(1))
  
  def reduce(b: DawidSkeneAggregatorBuffer, a: DawidSkenePartial) : DawidSkeneAggregatorBuffer = {
    val pi = params.value.pi 
    val classCondi = Vector.range(0,nClasses).map( c => pi(a.annotator.toInt)(c)(a.value))
    val newVect = classCondi.zip(b.aggVect).map(x => x._1 * x._2)
    DawidSkeneAggregatorBuffer(newVect) 
  }

  def merge(b1: DawidSkeneAggregatorBuffer, b2: DawidSkeneAggregatorBuffer) : DawidSkeneAggregatorBuffer = { 
    val buf = DawidSkeneAggregatorBuffer(b1.aggVect.zip(b2.aggVect).map(x => x._1 * x._2))
    buf
  }

  def finish(reduction: DawidSkeneAggregatorBuffer) = {
    val result = reduction.aggVect.zipWithIndex.maxBy(x => x._1*params.value.w(x._2))._2
    result
  }

  def bufferEncoder: Encoder[DawidSkeneAggregatorBuffer] = Encoders.product[DawidSkeneAggregatorBuffer]

  def outputEncoder: Encoder[Int] = Encoders.scalaInt
}

private[crowd] case class DawidSkeneAggregatorBuffer(aggVect: scala.collection.Seq[Double])
