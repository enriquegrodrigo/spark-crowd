
package com.enriquegrodrigo.spark.crowd.methods

import scala.collection.mutable.ArrayBuilder
import com.enriquegrodrigo.spark.crowd.aggregators._
import com.enriquegrodrigo.spark.crowd.types._
import org.apache.spark.sql.Dataset

/**
 *  Provides functions for transforming an annotation dataset into 
 *  a standard label dataset using the majority voting approach
 *  
 *  This object provides several functions for using majority voting style
 *  algorithms over annotations datasets (spark datasets with types [[com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation]], 
 *  [[com.enriquegrodrigo.spark.crowd.types.MulticlassAnnotation]], or [[com.enriquegrodrigo.spark.crowd.types.RealAnnotation]]). For discrete types 
 *  ([[com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation]], [[com.enriquegrodrigo.spark.crowd.types.MulticlassAnnotation]]) the method uses the '''most 
 *  frequent''' class. For continuous types, the '''mean''' is used. 
 *
 *  The object also provides methods for estimating the probability of a class
 *  for the discrete type, computing, for the binary case, the mean of the 
 *  positive class and, for the multiclass case, the one vs all mean of a
 *  class against the others. 
 *
 *  @example
 *  {{{
 *    result: Dataset[BinaryLabel] = MajorityVoting.transformBinary(dataset)
 *  }}}
 *
 *  @author enrique.grodrigo
 *  @version 0.1 
 */
object MajorityVoting {


  /**
   * Obtains the most frequent class for BinaryAnnotation datasets
   * @param dataset The annotations dataset to be aggregated
   */
  def transformBinary(dataset: Dataset[BinaryAnnotation]): Dataset[BinaryLabel] = {
    import dataset.sparkSession.implicits._
    val aggFunction = (new BinaryMVAggregator()).toColumn 
    dataset.groupByKey( (x: BinaryAnnotation) => x.example)
            .agg(aggFunction)
            .map((t: (Long, Int)) => BinaryLabel(t._1, t._2))
  }
  
  /**
   * Obtains probability of the class being positive for BinaryAnnotation datasets
   * @param dataset The annotations dataset to be aggregated
   */
  def transformSoftBinary(dataset: Dataset[BinaryAnnotation]): Dataset[BinarySoftLabel] = {
    import dataset.sparkSession.implicits._
    val aggFunction = (new BinarySoftMVAggregator()).toColumn
    dataset.groupByKey( (x: BinaryAnnotation) => x.example)
            .agg(aggFunction)
            .map((t: (Long, Double)) => BinarySoftLabel(t._1, t._2))
  }

  /**
   * Obtain the mean of the annotations for a given example.
   * @param dataset The annotations dataset to be aggregated
   */
  def transformReal(dataset: Dataset[RealAnnotation]): Dataset[RealLabel] = {
    import dataset.sparkSession.implicits._
    val aggFunction = (new RealMVAggregator()).toColumn
    dataset.groupByKey(_.example).agg(aggFunction).map((t: (Long, Double)) => RealLabel(t._1, t._2))
  }

  /**
   * Obtain the most frequent class for all examples in the annotation dataset. 
   * @param dataset The annotations dataset to be aggregated
   */
  def transformMulticlass(dataset: Dataset[MulticlassAnnotation]): Dataset[MulticlassLabel] = {
    import dataset.sparkSession.implicits._
    val nClasses = dataset.select($"value").distinct().count().toInt
    val aggFunction = (new MulticlassMVAggregator(nClasses)).toColumn
    dataset.groupByKey(_.example).agg(aggFunction).map((t: (Long, Int)) => MulticlassLabel(t._1, t._2))
  }
  
  /**
   * Obtain a list of datasets resulting of applying [[transformSoftBinary]] to
   * each class against the others
   * @param dataset The annotations dataset to be aggregated
   */
  def transformSoftMulti(dataset: Dataset[MulticlassAnnotation]): List[Dataset[BinarySoftLabel]] = {
    import dataset.sparkSession.implicits._
    val nClasses = dataset.select($"value").distinct().count().toInt
    List.range(0,nClasses).map(i => transformSoftBinary( //For each class
        dataset.map(d => BinaryAnnotation(d.example,d.annotator,if(d.value==i) 1 else 0))
      )
    )
  }

}
