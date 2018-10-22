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



package com.enriquegrodrigo.spark.crowd.methods

import com.enriquegrodrigo.spark.crowd.types._

import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Column
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
import org.apache.spark.broadcast.Broadcast

import org.apache.commons.math3.distribution.ChiSquaredDistribution

import scala.util.Random
import scala.math.{sqrt => scalaSqrt, exp => scalaExp, abs => scalaAbs}

/**
 *  Provides functions for transforming an annotation dataset into 
 *  a standard label dataset using the PM algorithm.
 *
 *  This algorithm only works with continuous target variables. Thus you need an 
 *  annotation dataset of [[types.RealAnnotation]]: 
 *
 *  The algorithm returns a [[PM.PMModel]], with information about 
 *  the class true label estimation and the annotators weight
 *
 *  @example
 *  {{{
 *   import com.enriquegrodrigo.spark.crowd.methods.PM
 *   import com.enriquegrodrigo.spark.crowd.types._
 *   
 *   sc.setCheckpointDir("checkpoint")
 *   
 *   val annFile = "data/real-ann.parquet"
 *   
 *   val annData = spark.read.parquet(annFile)
 *   
 *   //Applying the learning algorithm
 *   val mode = PM(annData)
 *   
 *   //Get MulticlassLabel with the class predictions
 *   val pred = mode.getMu()
 *   
 *   //Annotator weights
 *   val annweights = mode.getAnnotatorWeights()
 *   
 *  }}}
 *  @see Q. Li, Y. Li, J. Gao, B. Zhao, W. Fan, and J. Han. Resolving conflicts in heterogeneous data by truth discovery and source reliability estimation. In SIGMOD, pages 1187–1198, 2014.
 *  @version 0.2.0
 */
object PM {

  /****************************************************/
  /****************** CASE CLASSES ********************/
  /****************************************************/

  private[crowd] case class InternalModel(annotations: DataFrame, mu: DataFrame, weights: DataFrame, difference: Double)  

  /**
  *  Model returned by the learning algorithm.
  *
  *  @author enrique.grodrigo
  *  @version 0.2.0
  */
  class PMModel(mu: DataFrame, weights: DataFrame) {

    /**
    *  Estimated ground truth.
    *
    *  @author enrique.grodrigo
    *  @version 0.2.0
    */
    def getMu(): Dataset[RealLabel] = {
      import mu.sparkSession.implicits._
      mu.select(col("example"), col("mu") as "value").as[RealLabel]
    }

    /**
    *  Estimated annotator weights.
    *
    *  @author enrique.grodrigo
    *  @version 0.2.0
    */
    def getAnnotatorWeights(): Dataset[RealAnnotatorWeight] = {

      import weights.sparkSession.implicits._
      weights.select(col("annotator"), col("w") as "weight").as[RealAnnotatorWeight]
    }
  }
  /****************************************************/
  /********************   UDF   **********************/
  /****************************************************/
  
  private[crowd] def squaredDistance(gt: Double, y: Double): Double = {
    return math.pow((gt - y),2)
  }
 

  /****************************************************/
  /******************** METHODS **********************/
  /****************************************************/

  private[crowd] def initialization(df: DataFrame): InternalModel = {
    //Obtains the mean for each example
    val mu = df.groupBy("example").agg(avg("value") as "mu").cache()
    return InternalModel(df.cache(), mu, mu, -1) //Second mu is a placeholder
  }
  
  
  private[crowd] def seNormWeights(annotations: DataFrame, mu: DataFrame): DataFrame = {
    val squaredDistanceF = udf(squaredDistance(_: Double,_: Double))
    val joined = annotations.join(mu, "example").cache()
    val stdnorm = joined.groupBy("example")
                        .agg(stddev_pop("value") as "norm")
    val distances = joined.join(stdnorm, "example")
                          .select(col("annotator"), squaredDistanceF(col("mu"), col("value"))/col("norm") as "distance")
                          .rollup("annotator")
                          .agg(sum(col("distance")) as "distance")
                          .cache()
  
    val denom = distances.where(col("annotator").isNull).collect()(0).getAs[Double](1)
    val weights = distances.where(col("annotator").isNotNull).select(col("annotator"), -log(col("distance")/denom) as "w")
    return weights
  }
  
  private[crowd] def gtEstimation(df: DataFrame, weights: DataFrame): DataFrame = {
    return df.join(weights, "annotator")
             .groupBy("example")
             .agg(sum(col("value") * col("w")) as "num", sum(col("w")) as "denom")
             .select(col("example"), col("num")/col("denom") as "mu")
             .cache()
  }
  
  private[crowd] def mse(mu1: DataFrame, mu2: DataFrame): Double = {
    return mu1.join(mu2.toDF("example", "mu2"), "example")
              .select((sum(pow(col("mu")-col("mu2"), 2))/count("example")) as "mse")
              .collect()(0).getAs[Double](0)
  }
  
  private[crowd] def step(m: InternalModel, i: Integer): InternalModel = {
    val weights = seNormWeights(m.annotations, m.mu)
    val mu = gtEstimation(m.annotations,weights).checkpoint()
    val mseDifference = mse(m.mu,mu)
    return InternalModel(m.annotations, mu, weights, mseDifference)  
  }

 /**
  *  Apply the IBCC Algorithm.
  *
  *  @param dataset The dataset (spark dataset of [[types.RealAnnotation]]) 
  *  @param iterations Iterations of the learning algorithm 
  *  @param threshold Minimum MSE for the algorithm to continue iterating 
  *  @author enrique.grodrigo
  *  @version 0.2.0
  */ 
  def apply(dataset: Dataset[RealAnnotation], iterations: Int = 5, threshold: Double = 0.1): PMModel = {

    //Initialization
    val d = dataset.toDF()
    val initModel = initialization(d)
    //Prepare for steps
    val stepF = (model: InternalModel,i:Int) => step(model,i)
    val first = stepF(initModel, 0)
    val firstFixed = InternalModel(first.annotations, first.mu, first.weights, 1.0) 

    //Repeats until some condition is met
    val s = Stream.range(2,iterations).scanLeft(firstFixed)(stepF)
                                    .takeWhile( (model) => model.difference > threshold )
                  
    val l = s.last
    //Results: Ground Truth estimation, class prior estimation and annotator quality matrices
    (new PMModel(l.mu, l.weights))
  }

}
