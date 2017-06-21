
package com.enriquegrodrigo.spark.crowd.aggregators 

import com.enriquegrodrigo.spark.crowd.types.DawidSkenePartial
import com.enriquegrodrigo.spark.crowd.types.DawidSkeneParams
import org.apache.spark.sql.{Encoder,Encoders}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.broadcast.Broadcast

import scala.math.log

private[crowd] class DawidSkeneLogLikelihoodAggregator(params: Broadcast[DawidSkeneParams]) 
  extends Aggregator[DawidSkenePartial, DawidSkeneLogLikelihoodAggregatorBuffer, Double]{

  private def sumKey(map: Map[Int,Double], pair: (Int,Double)) = {
      val key = pair._1
      val value = pair._2
      val new_value = map.get(key) match {
        case Some(x) => x + value
        case None => value 
      }
      map.updated(key, new_value)
  }

  def zero: DawidSkeneLogLikelihoodAggregatorBuffer = DawidSkeneLogLikelihoodAggregatorBuffer(0, -1)

  def reduce(b: DawidSkeneLogLikelihoodAggregatorBuffer, a: DawidSkenePartial) : DawidSkeneLogLikelihoodAggregatorBuffer = {
    val pival = params.value.pi(a.annotator.toInt)(a.est)(a.value)
    DawidSkeneLogLikelihoodAggregatorBuffer(b.agg + log(pival), a.est) 
  }

  def merge(b1: DawidSkeneLogLikelihoodAggregatorBuffer, b2: DawidSkeneLogLikelihoodAggregatorBuffer) : DawidSkeneLogLikelihoodAggregatorBuffer = { 
    DawidSkeneLogLikelihoodAggregatorBuffer(b1.agg + b2.agg, if (b1.predClass == -1) b2.predClass else b1.predClass) 
  }

  def finish(reduction: DawidSkeneLogLikelihoodAggregatorBuffer) =  {
    reduction.agg + log(params.value.w(reduction.predClass))
  }


  def bufferEncoder: Encoder[DawidSkeneLogLikelihoodAggregatorBuffer] = Encoders.product[DawidSkeneLogLikelihoodAggregatorBuffer]

  def outputEncoder: Encoder[Double] = Encoders.scalaDouble
}

private[crowd] case class DawidSkeneLogLikelihoodAggregatorBuffer(agg: Double, predClass: Int)
