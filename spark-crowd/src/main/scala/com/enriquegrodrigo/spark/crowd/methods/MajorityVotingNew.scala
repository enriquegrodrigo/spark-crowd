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
/*
package com.enriquegrodrigo.spark.crowd.methods

import scala.collection.mutable.ArrayBuilder
import com.enriquegrodrigo.spark.crowd.aggregators._
import com.enriquegrodrigo.spark.crowd.types._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.functions.{count,lit,sum,col, explode, array}


/**
 *  Provides functions for transforming an annotation dataset into 
 *  a standard label dataset using the majority voting approach
 *  
 *  This object provides several functions for using majority voting style
 *  algorithms over annotations datasets (spark datasets with types [[types.BinaryAnnotation]], 
 *  [[types.MulticlassAnnotation]], or [[types.RealAnnotation]]). For discrete types 
 *  ([[types.BinaryAnnotation]], [[types.MulticlassAnnotation]]) the method uses the '''most 
 *  frequent''' class. For continuous types, the '''mean''' is used. 
 *
 *  The object also provides methods for estimating the probability of a class
 *  for the discrete type, computing, for the binary case, the proportion of the 
 *  positive class and, for the multiclass case, the proportion of each of the classes.
 *
 *  The next example can be found in the examples folder of the project. 
 *
 *  @example
 *  {{{
 *  
 *  import com.enriquegrodrigo.spark.crowd.methods.MajorityVoting
 *  import com.enriquegrodrigo.spark.crowd.types._
 *  
 *  val exampleFile = "data/binary-ann.parquet"
 *  val exampleFileMulti = "data/multi-ann.parquet"
 *  val exampleFileCont = "data/cont-ann.parquet"
 *  
 *  val exampleDataBinary = spark.read.parquet(exampleFile).as[BinaryAnnotation] 
 *  val exampleDataMulti = spark.read.parquet(exampleFileMulti).as[MulticlassAnnotation] 
 *  val exampleDataCont = spark.read.parquet(exampleFileCont).as[RealAnnotation] 
 *  
 *  //Applying the learning algorithm
 *  //Binary class
 *  val muBinary = MajorityVoting.transformBinary(exampleDataBinary)
 *  val muBinaryProb = MajorityVoting.transformSoftBinary(exampleDataBinary)
 *  //Multiclass
 *  val muMulticlass = MajorityVoting.transformMulticlass(exampleDataMulti)
 *  val muMulticlassProb = MajorityVoting.transformSoftMulti(exampleDataMulti)
 *  //Continuous case
 *  val muCont = MajorityVoting.transformReal(exampleDataCont)
 *
 *  }}}
 *
 *  @author enrique.grodrigo
 *  @version 0.1.3 
 */
object MajorityVoting {
  
  /****************************************************/
  /****************** CASE CLASSES ********************/
  /****************************************************/

  /**
   *  Combination example class for complete multiclass probability estimation 
   *  @author enrique.grodrigo
   *  @version 0.1.3 
   */
  private[spark] case class ExampleClassCombination(example: Long, clas: Int)

  /**
   *  Frequency of a concrete class for example  
   *  @author enrique.grodrigo
   *  @version 0.1.3 
   */
  private[spark] case class ExampleClassFrequency(example: Long, clas: Int, freq: Double)

  /**
   *  Number of labels for an example 
   *  @author enrique.grodrigo
   *  @version 0.1.3
   */
  private[spark] case class ExampleFrequency(example: Long, freq: Double)



  /****************************************************/
  /****************** AGGREGATORS ********************/
  /****************************************************/

  /**
   *  Obtain multiclass soft probability estimation 
   *  @author enrique.grodrigo
   *  @version 0.1.3
   */
  private[spark] class MulticlassMajorityEstimation() extends Aggregator[(ExampleClassCombination, MulticlassAnnotation), Double, Double] {
    def zero: Double = 0.0 
    def reduce(b: Double, a: (ExampleClassCombination,MulticlassAnnotation)) : Double =  a match {
      case (_,null) => b //For dealing with incomplete cases
      case (_,ann) => b + 1 
    }
    def merge(b1: Double, b2: Double) : Double = b1 + b2 
    def finish(b: Double) = b 
    def bufferEncoder: Encoder[Double] = Encoders.scalaDouble
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }
 

  /****************************************************/
  /******************** METHODS **********************/
  /****************************************************/

  /******************** BINARY **********************/

  /**
   * Obtains the most frequent class (0 or 1) for binary annotations dataset.  
   * @param dataset The annotations dataset 
   */
  def binary(dataset: DataFrame): DataFrame = {
    return df.groupBy("example")
             .agg(when(avg(col("value")) > 0.5, 1).otherwise(0) as "mu")
  }

   /**
   * Obtains a probabilistic estimation of a binary class in (0,1).  
   * @param dataset The annotations dataset 
   */
  def softBinary(dataset: DataFrame): DataFrame = {
    return df.groupBy("example")
             .agg(avg(col("value")) as "mu")
  }

  /******************** MULTICLASS **********************/

  def multi(dataset: DataFrame): DataFrame = {
    return df.groupBy("example", "value")
             .count()
  }
 
  /**
   * Obtain the mean of the annotations for each example from a [[types.RealAnnotation]].
   * @param dataset The annotations dataset (spark Dataset of type [[types.RealAnnotation]]) 
   *  to be aggregated
   */
  def transformReal(dataset: Dataset[RealAnnotation]): Dataset[RealLabel] = {
    import dataset.sparkSession.implicits._
    val aggFunction = (new RealMVAggregator()).toColumn
    //Groups by example and obtains the mean of the annotations for each example
    dataset.groupByKey(_.example).agg(aggFunction).map((t: (Long, Double)) => RealLabel(t._1, t._2))
  }

  /**
   * Obtain the most frequent class for each example of the a [[types.MulticlassAnnotation]] dataset. 
   * @param dataset The annotations dataset (spark Dataset of type [[types.MulticlassAnnotation]]) 
   *  to be aggregated
   */
  def transformMulticlass(dataset: Dataset[MulticlassAnnotation]): Dataset[MulticlassLabel] = {
    import dataset.sparkSession.implicits._
    val nClasses = dataset.select($"value").distinct().count().toInt
    val aggFunction = (new MulticlassMVAggregator(nClasses)).toColumn
    //Groups by example and obtains the most frequent class for each example
    dataset.groupByKey(_.example).agg(aggFunction).map((t: (Long, Int)) => MulticlassLabel(t._1, t._2))
  }
  
  /**
   * Obtain a list of datasets resulting of applying [[transformSoftBinary]] to
   * each class against the others (One vs All) on a  [[types.MulticlassAnnotation]] dataset. 
   *
   * It supposes classes go from 0 to nClasses. For example, for a three class problem, there
   * should be classes {0,1,2}. 
   *
   * @param dataset The annotations dataset (spark Dataset of type [[types.MulticlassAnnotation]]) 
   *  to be aggregated
   */
  def transformSoftMulti(dataset: Dataset[MulticlassAnnotation]): Dataset[MulticlassSoftProb] = {
    import dataset.sparkSession.implicits._
    val nClasses = dataset.select($"value").distinct().count().toInt
    //Obtains all combinations example, clas (for imcomplete cases)
    val exampleClass = dataset.map(_.example)
                                    .distinct
                                    .withColumnRenamed("value", "example")
                                    .withColumn("clas", explode(array((0 until nClasses).map(lit): _*)))
                                    .as[ExampleClassCombination]
    //Obtains the frequencies for each class for each example. If incomplete case (the annotations do not cover all classes)
    // the method return 0 for that frequency.
    val classFrequencies = exampleClass.joinWith(dataset, 
                                            exampleClass.col("example") === dataset.col("example") &&  
                                              exampleClass.col("clas") === dataset.col("value"),
                                          "left_outer") 
                                       .as[(ExampleClassCombination, MulticlassAnnotation)]
                                       .groupByKey(_._1) 
                                       .agg((new MulticlassMajorityEstimation()).toColumn)
                                       .as[(ExampleClassCombination, Double)]
                                       .map(x => ExampleClassFrequency(x._1.example, x._1.clas, x._2))
                                       .as[ExampleClassFrequency]

    //Obtains the number of annotations for each example
    val exampleFrequencies =  dataset.groupBy(col("example"))
                                     .agg(count(col("annotator")) as "freq")
                                     .as[ExampleFrequency]

    //Estimates the probability of each class using standard division
    val estimation = exampleFrequencies.joinWith(classFrequencies, 
                                          exampleFrequencies.col("example") === classFrequencies.col("example"))
                                       .as[(ExampleFrequency, ExampleClassFrequency)]
                                       .map(x => MulticlassSoftProb(x._1.example, x._2.clas, x._2.freq/x._1.freq))
                                       .as[MulticlassSoftProb]
    return estimation
  }
}
*/
