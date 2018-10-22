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
 *  a standard label dataset using the CATD algorithm.
 *
 *  This algorithm only works with continuous label datasets of type [[types.RealAnnotation]]: 
 *
 *  The algorithm returns a [[CATD.CATDModel]], with information about 
 *  the class true label estimation and the annotators weight
 *
 *  @example
 *  {{{
 *   import com.enriquegrodrigo.spark.crowd.methods.CATD
 *   import com.enriquegrodrigo.spark.crowd.types._
 *   
 *   sc.setCheckpointDir("checkpoint")
 *   
 *   val annFile = "data/real-ann.parquet"
 *   
 *   val annData = spark.read.parquet(annFile)
 *   
 *   //Applying the learning algorithm
 *   val mode = CATD(annData.as[RealAnnotation])
 *   
 *   //Get MulticlassLabel with the class predictions
 *   val pred = mode.getMu()
 *   
 *   //Annotator weights
 *   val annweights = mode.getAnnotatorWeights()
 *   
 *  }}}
 *  @see Q. Li, Y. Li, J. Gao, L. Su, B. Zhao, M. Demirbas, W. Fan, and J. Han. A confidence-aware approach for truth discovery on long-tail data. PVLDB, 8(4):425–436, 2014.
 *  @version 0.2.0
 */
object CATD {

  /****************************************************/
  /****************** CASE CLASSES ********************/
  /****************************************************/

  /* Internal model for the iterations */
  private[crowd] case class InternalModel(annotations: DataFrame, mu: DataFrame, weights: DataFrame, difference: Double)  


  /**
   *  Model returned by the CATD algorithm  
   *
   *  @author enrique.grodrigo
   *  @version 0.2.0 
   */
  class CATDModel(mu: DataFrame, weights: DataFrame) {
    /**
     *  Dataset of [[types.RealLabel]] with the ground truth estimation
     *
     *  @author enrique.grodrigo
     *  @version 0.2.0 
     */
    def getMu(): Dataset[RealLabel] = {
      import mu.sparkSession.implicits._
      mu.select(col("example"), col("mu") as "value").as[RealLabel]
    }

    /**
     *  Dataset of [[types.RealAnnotatorWeight]] with the annotator weights used for the aggregation
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
  
  /*ChiSquared probability*/ 
  private[crowd] def chisq(ns: Long, alpha: Double): Double = {
    val md = new ChiSquaredDistribution(ns)
    return md.inverseCumulativeProbability(1 - alpha/2)
  }

  /*Weight calculation for the annotator*/ 
  private[crowd] def weightExpression(alpha: Double)(ns: Long, denom: Double): Double = {
    return chisq(ns,alpha)/denom
  }


  
  /****************************************************/
  /******************** METHODS **********************/
  /****************************************************/

  private[crowd] def initialization(df: DataFrame): InternalModel = {
    //Obtains the mean for each example
    val mu = df.groupBy("example").agg(avg("value") as "mu").cache()
    return InternalModel(df.cache(), mu, mu, -1) //Second mu is a placeholder
  }
  
  private[crowd] def weightEstimation(df: DataFrame, mu: DataFrame, alpha: Double): DataFrame = {
    val aggregation = df.join(mu, "example")
                        .groupBy("annotator")
                        .agg(sum(pow(col("value")-col("mu"), 2)) as "denom", count("example") as "ns")
    val weightf = udf((ns: Long, denom: Double) => weightExpression(alpha)(ns,denom))
    val annWeights = aggregation.select(col("annotator"), weightf(col("ns"), col("denom")) as "w")
    return annWeights.cache()
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
  
  private[crowd] def step(alpha: Double)(m: InternalModel, i: Integer): InternalModel = {
    val weights = weightEstimation(m.annotations, m.mu, alpha)
    val mu = gtEstimation(m.annotations,weights).checkpoint()
    val mseDifference = mse(m.mu,mu)
    return InternalModel(m.annotations, mu, weights, mseDifference)  
  }

 /**
   *  Applies the CATD learning algorithm.
   *
   *  @param dataset The dataset over which the algorithm will execute ([[types.RealAnnotation]]
   *  @param iterations Maximum number of iterations of the algorithm 
   *  @param threshold Minimum change in MSE needed for continuing with the execution 
   *  @param alpha Chi-square alpha value for the weight calculation
   *  @return [[CATD.CATDModel]]
   *
   *  @author enrique.grodrigo
   *  @version 0.2.0 
   */
  def apply(dataset: Dataset[RealAnnotation], iterations: Int = 5, threshold: Double = 0.1, alpha: Double = 0.05): CATDModel = {
    val d = dataset.toDF()

    //Initialization
    val initModel = initialization(d)
    //Prepare for steps
    val stepF = (model: InternalModel,i:Int) => step(alpha)(model,i)
    val first = stepF(initModel, 0)
    val firstFixed = InternalModel(first.annotations, first.mu, first.weights, 1.0) 

    //Repeats until some condition is met
    val s = Stream.range(2,iterations).scanLeft(firstFixed)(stepF)
                                    .takeWhile( (model) => model.difference > threshold )
                  
    val l = s.last
    //Results: Ground Truth estimation, class prior estimation and annotator quality matrices
    (new CATDModel(l.mu, l.weights))
  }

}
  
