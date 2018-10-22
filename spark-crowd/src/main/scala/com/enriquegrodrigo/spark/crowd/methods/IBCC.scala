
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


import scala.util.Random
import scala.math.{sqrt => scalaSqrt, exp => scalaExp, abs => scalaAbs}

/**
 *  Provides functions for transforming an annotation dataset into 
 *  a standard label dataset using the IBCC algorithm.
 *
 *  This algorithm only works with multiclass target variables (Datasets of 
 *  type [[types.MulticlassAnnotation]]
 *
 *  The algorithm returns a [[IBCC.IBCCModel]], with information about 
 *  the class true label estimation, the annotators precision, and the 
 *  class prior estimation 
 *
 *  @example
 *  {{{
 *   import com.enriquegrodrigo.spark.crowd.methods.IBCC
 *   import com.enriquegrodrigo.spark.crowd.types._
 *   
 *   sc.setCheckpointDir("checkpoint")
 *   
 *   val annFile = "data/binary-ann.parquet"
 *   
 *   val annData = spark.read.parquet(annFile)
 *   
 *   //Applying the learning algorithm
 *   val mode = IBCC(annData)
 *   
 *   //Get MulticlassLabel with the class predictions
 *   val pred = mode.getMu() 
 *   
 *   //Annotator precision matrices
 *   val annprec = mode.getAnnotatorPrecision()
 *   
 *   //Annotator precision matrices
 *   val classPrior = mode.getClassPrior()
 *  }}}
 *  @see H.-C. Kim and Z. Ghahramani. Bayesian classifier combination. In AISTATS, pages 619–627, 2012.
 *  @version 0.2.0
 */
object IBCC {

  /****************************************************/
  /****************** CASE CLASSES ********************/
  /****************************************************/

  private[crowd] case class InternalModel(mu: DataFrame, p: DataFrame, pi: DataFrame, likelihood: Double, improvement: Double)  
  
  /**
  *  Model returned by the learning algorithm.
  *
  *  @author enrique.grodrigo
  *  @version 0.2.0
  */
  class IBCCModel(mu: DataFrame, p: DataFrame, pi: DataFrame) {
    /**
    *  Estimated probabilities for each example 
    *
    *  @author enrique.grodrigo
    *  @version 0.2.0
    */
    def getMu(): Dataset[MulticlassSoftProb] = {
      import mu.sparkSession.implicits._
      mu.select(col("example"), col("class") as "clas", col("mu") as "prob").as[MulticlassSoftProb]
    }

    /**
    *  Estimated annotator precision 
    *
    *  @author enrique.grodrigo
    *  @version 0.2.0
    */
    def getAnnotatorPrecision(): Dataset[DiscreteAnnotatorPrecision] = {
      import pi.sparkSession.implicits._
      pi.select(col("annotator"), col("c"), col("k"), col("pi") as "prob").as[DiscreteAnnotatorPrecision] 
    }

    /**
    *  Estimated class prior.  
    *
    *  @author enrique.grodrigo
    *  @version 0.2.0
    */
    def getClassPrior(): DataFrame = p 
  }

  /****************************************************/
  /******************     UDAF    ********************/
  /****************************************************/

  private[crowd] class Prod extends UserDefinedAggregateFunction {
    def inputSchema: StructType = StructType(Array(StructField("pi", DoubleType)))
    def bufferSchema: StructType = StructType(Array(StructField("result", DoubleType)))
    def dataType: DataType = DoubleType
    def deterministic: Boolean = true
    def initialize(buffer: MutableAggregationBuffer): Unit = {
      buffer(0) = 1.0
    }
    def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
      buffer(0) = buffer.getAs[Double](0) * input.getAs[Double](0)
    }
    def merge(buffer1: MutableAggregationBuffer, buffer2: Row) = {
      buffer1(0) = buffer1.getAs[Double](0) * buffer2.getAs[Double](0)
    }
    def evaluate(buffer: Row): Any = {
      buffer(0)
    }
  }

  /****************************************************/
  /********************   UDF   **********************/
  /****************************************************/
  
  private[crowd] def addClassPriors(classPrior: Array[Double], nExamples: Long)(c: Int, num: Double) = {
    (num + classPrior(c))/(classPrior.sum + nExamples)
  }

  private[crowd] def addAnnotatorPrior(annPrior: Map[String, Array[Array[Double]]])(j: String, c: Int, k: Int, num: Double, denom: Double) = {
    (annPrior(j)(c)(k) + num)/(annPrior(j)(c).sum + denom)
  }


  /****************************************************/
  /******************** METHODS **********************/
  /****************************************************/

  private[crowd] def probmvoting(annotations: DataFrame, nClasses: Long, allClasses: Column): DataFrame = {
  //Generating example-class combination. Take care of examples without some classes combinations. 
  val exampleClass = annotations.select(col("example")).distinct().withColumn("value", explode(allClasses))
  val joined = exampleClass.join(annotations, Seq("example", "value"), "left_outer").withColumnRenamed("value", "class")
  
  //Obtaining probability of class
  val ceroNull = when(isnull(col("annotator")), lit(0)).otherwise(lit(1))
  val rolled = joined.rollup(col("example"), col("class")).agg(sum(ceroNull) as "count").cache()
  val denom = rolled.where(col("class").isNull).select(col("example"), col("count")).toDF("example_denom", "denom")
  val num = rolled.where(col("class").isNotNull).toDF("example_num", "class_num", "num")
  val classProbs = num.join(denom, col("example_num") === col("example_denom"))
                      .select(col("example_num") as "example", col("class_num") as "class", col("num") / col("denom") as "mu")
  return classProbs
}
  
  /**
  *  Annotator prior matrix default.
  *
  *  @author enrique.grodrigo
  *  @version 0.2.0
  */
  private[crowd] def priorMatrixGen(nClasses: Int): Array[Array[Double]] = {
    return Array.fill[Double](nClasses,nClasses)(1.0)
  }

  /**
  *  Class prior default.
  *
  *  @author enrique.grodrigo
  *  @version 0.2.0
  */
  private[crowd] def classPrior(nClasses: Int): Array[Double] = {
    return Array.fill[Double](nClasses)(1.0)
  }

  /**
  *  Initialization.
  *
  *  @author enrique.grodrigo
  *  @version 0.2.0
  */
  private[crowd] def initialization(annotations: DataFrame): (DataFrame, DataFrame, DataFrame, Long, Long, Int, Column) = {
    val anncached = annotations.cache()
    val nClasses = anncached.select(col("value")).distinct().count().toInt
    val nAnnotators = anncached.select(col("annotator")).distinct().count()
    
    
    val allClasses = array((0 until nClasses).map(lit):_*)
    val nExamples = anncached.select(col("example")).distinct().count()
    val jck = annotations.select(col("annotator") as "j")
                         .distinct().withColumn("c", explode(allClasses))
                         .withColumn("k", explode(allClasses)).cache()
    val mu = probmvoting(anncached, nClasses, allClasses).cache()
    return (anncached, mu, jck, nExamples, nAnnotators, nClasses, allClasses)
  }

  /**
  *  M Step of the EM algorithm.
  *
  *  @author enrique.grodrigo
  *  @version 0.2.0
  */
  private[crowd] def mStep(annotations: DataFrame, mu: DataFrame, jck: DataFrame, nExamples: Long, 
            classDirichlet: Array[Double], annPrior: Map[String, Array[Array[Double]]], 
            allClasses: Column): (DataFrame, DataFrame) = {

  //Obtains class priors
    val classPriorExpression = udf((c: Int,num: Double) => addClassPriors(classDirichlet, nExamples)(c,num))
    val prior = mu.groupBy("class")
                  .agg(sum(col("mu")) as "num")
                  .select(col("class"), classPriorExpression(col("class"), col("num")) as "p")
  
    
    //Add possible classes to data
    val dfC = annotations.withColumn("c", explode(allClasses))
    
    //Obtains full M step data taking into account missing combinations
    val jckr = jck.toDF("annotator", "c", "value") //jck combinations 
    val mData = jckr.join(dfC, Seq("annotator", "c", "value"), "left_outer") 
                    .join(mu.toDF("example", "c", "mu"), Seq("example","c"), "left_outer")
    
    //Obtains the groups we are interested in (j,c,k) for num and (j,c) for denom 
    val grouped = mData.rollup("annotator", "c", "value")
                       .agg(sum(when(isnull(col("mu")), 0).otherwise(col("mu"))) as "stat")
                       .where(col("annotator").isNotNull)
                       .where(col("c").isNotNull).cache()
    
    //Obtains pi by joining num and denom
    val num = grouped.where(col("value").isNotNull)
                     .select(col("annotator"), col("c"), col("value") as "k", col("stat") as "num")
                     .toDF("annotator", "c", "k", "num")

    val denom = grouped.where(col("value").isNull)
                       .select(col("annotator"), col("c"), col("stat") as "denom")
                       .toDF("annotator", "c", "denom")

    val annotatorQualityExpression = udf(
      (annotator: String, c: Int, k: Int, num: Double, denom: Double) => {
        addAnnotatorPrior(annPrior)(annotator, c, k, num, denom)
      }
    )

    val pi = num.join(denom, Seq("annotator", "c"))
                .select(col("annotator"), col("c"), col("k"), 
                          annotatorQualityExpression(col("annotator"), col("c"), 
                          col("k"), col("num"), col("denom")) as "pi")
  
    return (prior.cache(), pi.cache())
  }

  /**
  *  E Step of the EM algorithm.
  *
  *  @author enrique.grodrigo
  *  @version 0.2.0
  */
  private[crowd] def eStep(annotations: DataFrame, pi: DataFrame, classPrior:DataFrame): DataFrame = {
    //Prepares data for E step
    val eData = annotations.withColumnRenamed("value", "k").join(pi, Seq("annotator", "k"))
    //Prod for annotations taking pi into account
    val prod = new Prod
    val pi_grouped = eData.groupBy("example", "c").agg(prod(col("pi")) as "stat")
    val num = pi_grouped.join(classPrior.withColumnRenamed("class", "c"), "c")
                        .select(col("example"), col("c"), col("stat") * col("p") as "num").cache()
    val denom = num.groupBy("example").agg(sum(col("num")) as "denom").toDF("example", "denom")
    val mu = num.join(denom, "example").select(col("example"), col("c") as "class", col("num") / col("denom") as "mu")
    return mu.cache()
  }

  /**
  *  Step of the EM algorithm.
  *
  *  @author enrique.grodrigo
  *  @version 0.2.0
  */
  private[crowd] def step(annotations: DataFrame, jck: DataFrame, nExamples: Long, 
            allClasses: Column, annotatorPrior: Map[String, Array[Array[Double]]], 
            classPrior: Array[Double])(model: InternalModel, i: Int): InternalModel = {

    val (p, pi) = mStep(annotations, model.mu, jck, nExamples, classPrior, annotatorPrior, allClasses)
    val mu = eStep(annotations, pi, p).checkpoint()
    val like = likelihood(annotations, pi, p, allClasses)
    val improvement = scalaAbs(like - model.likelihood) 
    return InternalModel(mu, p, pi,like, improvement)
  }
  
  private[crowd] def likelihood(annotations: DataFrame, pi: DataFrame, classPrior: DataFrame, allClasses: Column): Double = {
    val prod = new Prod
    return annotations.withColumn("c", explode(allClasses))
                            .select(col("example"), col("annotator"), col("c"), col("value") as "k")
                            .join(pi, Seq("annotator", "c", "k"))
                            .groupBy("example", "c")
                            .agg(prod(col("pi")) as "annlike")
                            .join(classPrior.toDF("c", "p"), "c")
                            .select(log(sum(col("annlike") * col("p"))) as "loglike")
                            .collect()(0).getAs[Double](0)
  }

  /**
  *  Apply the IBCC Algorithm.
  *
  *  @param dataset The dataset (spark dataset of MulticlassAnnotation
  *  @param eMIters Number of iterations for the EM algorithm
  *  @param eMThreshold LogLikelihood variability threshold for the EM algorithm
  *  @param annDirich Dirichlech prior for annotators. By default, a uniform prior.  
  *  @param classDirich Dirichlech prior for classes. By default, a uniform prior.  
  *  @author enrique.grodrigo
  *  @version 0.2.0
  */
  def apply(dataset: Dataset[MulticlassAnnotation], eMIters: Int = 5, eMThreshold: Double = 0.1, 
            annDirich: Option[Map[String,Array[Array[Double]]]] = None, 
            classDirich: Option[Array[Double]] = None): IBCCModel = {

    //Initialization
    val d = dataset.toDF()
    val (annotations, mu, jck, nExamples, nAnnotators, nClasses, allClasses)  = initialization(d)
    val annPrior = annDirich.getOrElse(Map((0 until nAnnotators.toInt).map(x => (x.toString,priorMatrixGen(nClasses))):_*))
    val classP = classDirich.getOrElse(classPrior(nClasses))

    //The internal model asks for 3 dataframes and the improvement. We use mu as placeholder
    val initModel = InternalModel(mu, mu, mu, 0, 0) 

    //Prepare for steps
    val stepF = (model: InternalModel,i:Int) => step(annotations, jck, nExamples, allClasses, annPrior, classP)(model,i)
    val first = stepF(initModel, 0)
    val firstFixed = InternalModel(first.mu, first.p, first.pi, 0, eMThreshold+1 ) 

    //Repeats until some condition is met
    val l = Stream.range(2,eMIters).scanLeft(firstFixed)(stepF)
                                    .takeWhile( (model) => model.improvement > eMThreshold )
                                    .last

    //Results: Ground Truth estimation, class prior estimation and annotator quality matrices
    (new IBCCModel(l.mu, l.p, l.pi))
  }

}


