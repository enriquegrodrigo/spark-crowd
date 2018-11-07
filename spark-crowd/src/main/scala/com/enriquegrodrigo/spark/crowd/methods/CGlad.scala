
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


import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.{Vectors => VectorsMl}
import scala.collection.JavaConversions.asScalaBuffer
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.clustering.KMeans
import scala.util.Random
import scala.math.exp
import scala.math.log
import scala.math.pow
import scala.math.{sqrt => scalaSqrt}
import org.apache.spark.mllib.optimization._

object CGlad {

  /****************************************************/
  /****************** CASE CLASSES ********************/
  /****************************************************/

  case class CGladParams(alpha: Array[Double], w: Array[Double], beta: Array[Double])

  /****************************************************/
  /******************     UDAF    ********************/
  /****************************************************/

class MuEstimate(alpha: Array[Double], beta: Array[Double], weights: Array[Double]) extends UserDefinedAggregateFunction {
  def inputSchema: StructType = StructType(Array(StructField("annotator", IntegerType),StructField("cluster", IntegerType),StructField("value", DoubleType)))
  def bufferSchema: StructType = StructType(Array(StructField("result0", DoubleType), StructField("result1", DoubleType)))
  def dataType: DataType = DoubleType
  def deterministic: Boolean = true
  def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = 1.0
    buffer(1) = 1.0
  }
  def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val annotator = input.getAs[Integer](0)
    val cluster = input.getAs[Integer](1)
    val value = input.getAs[Double](2)
    val sigmoidValue = sigmoid(alpha(annotator)*beta(cluster))
    val p = if (value == 0) sigmoidValue else (1 - sigmoidValue)
    //buffer(0) = buffer.getAs[Double](0) + logLim(p)
    //buffer(1) = buffer.getAs[Double](1) + logLim(1-p)
    buffer(0) = buffer.getAs[Double](0) * p
    buffer(1) = buffer.getAs[Double](1) * (1-p)
  }
  def merge(buffer1: MutableAggregationBuffer, buffer2: Row) = {
    buffer1(0) = buffer1.getAs[Double](0) * buffer2.getAs[Double](0)
    buffer1(1) = buffer1.getAs[Double](1) * buffer2.getAs[Double](1)
    //buffer1(0) = buffer1.getAs[Double](0) + buffer2.getAs[Double](0)
    //buffer1(1) = buffer1.getAs[Double](1) + buffer2.getAs[Double](1)
  }
  def evaluate(buffer: Row): Any = {
      val b0 = buffer.getAs[Double](0)
      val b1 = buffer.getAs[Double](1)
      //val negative = exp(b0 + logLim(weights(0)))
      //val positive = exp(b1 + logLim(weights(1)))
      val negative = b0 * weights(0)
      val positive = b1 * weights(1)
      val norm = negative + positive
      positive/norm
  }
}

class LogLikelihoodEstimate(alpha: Array[Double], beta: Array[Double], weights: Array[Double]) extends UserDefinedAggregateFunction {
  def inputSchema: StructType = StructType(Array(StructField("annotator", IntegerType),StructField("cluster", IntegerType),StructField("value", DoubleType),StructField("mu", DoubleType)))
  def bufferSchema: StructType = StructType(Array(StructField("result0", DoubleType), StructField("result1", DoubleType)))
  def dataType: DataType = DoubleType
  def deterministic: Boolean = true
  def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = 0.0
    buffer(1) = -1.0
  }
  def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val annotator = input.getAs[Integer](0)
    val cluster = input.getAs[Integer](1)
    val value = input.getAs[Double](2)
    val mu = input.getAs[Double](3)
    val sigmoidValue = sigmoid(alpha(annotator)*beta(cluster))
    val k =  if (value == 0) (sigmoidValue) else (1-sigmoidValue)
    val p = 1-mu
    buffer(0) = buffer.getAs[Double](0) + prodlog(p,k) + prodlog(1-p,1-k)
    buffer(1) = p
  }
  def merge(buffer1: MutableAggregationBuffer, buffer2: Row) = {
    buffer1(0) = buffer1.getAs[Double](0) + buffer2.getAs[Double](0)
    buffer1(1) = if (buffer1.getAs[Double](1) < 0) buffer2.getAs[Double](1) else buffer1.getAs[Double](1)
  }
  def evaluate(buffer: Row): Any = {
      val agg = buffer.getAs[Double](0)
      val p0 = buffer.getAs[Double](1)
      val p1 = 1-p0
      val w0 = weights(0)
      val w1 = weights(1)
      val result = agg + prodlog(p1,w1) + prodlog(p0,w0) 
      result
  }
}


  /****************************************************/
  /******************** Gradient **********************/
  /****************************************************/

class CGladGradient(nExamples: Long, nAnnotators: Integer, nClusters: Integer) extends Gradient {

    override def compute(data: org.apache.spark.mllib.linalg.Vector,label: Double,weights: org.apache.spark.mllib.linalg.Vector,cumGradient: org.apache.spark.mllib.linalg.Vector): Double = {
      
      val cumG = cumGradient.toDense.values
      
      val w = weights.toArray 
      val alphas = w.slice(0, nAnnotators)
      val betas = w.slice(nAnnotators, nAnnotators+nClusters)

      val v: Array[Double] = data.toArray
      val example: Long = v(0).toLong
      val annotator: Int = v(1).toInt
      val value: Integer = v(2).toInt
      val mu: Double = v(3).toDouble
      val cluster: Integer = v(4).toInt
      //val beta: Double = v(5).toDouble
      
      //Alpha
      val sigmoidValueAlpha = sigmoid( alphas(annotator) * betas(cluster) )
      val pAlpha = if (value == 1) mu else (1-mu)
      val termAlpha = (pAlpha - sigmoidValueAlpha)* betas(cluster)
      
      val annIndex = annotator
      cumG(annIndex) += termAlpha
      
      //Beta
      
      val sigmoidValueBeta = sigmoid(alphas(annotator)*betas(cluster))
      val pBeta = if (value == 1) mu else (1-mu)
      val termBeta = (pBeta - sigmoidValueBeta)*alphas(annotator)
      val clusterIndex = nAnnotators+cluster
      cumG(clusterIndex)+=termBeta
      //println(cumG.mkString(","))
      
      //Loss
      
      val alphaVal = alphas(annotator)
      val betaVal = betas(cluster)
      val sig = sigmoid(alphaVal*betaVal) 
      val p = 1-mu
      val k = if (value == 0) sig else 1-sig 
      val lossTerm = prodlog(p,k) + prodlog(1-p,1-k)
    
      lossTerm
    }
}


class CGladUpdater() extends Updater {
    def compute(weightsOld: org.apache.spark.mllib.linalg.Vector, gradient: org.apache.spark.mllib.linalg.Vector, stepSize: Double, iter: Int, regParam: Double) = {

      val stepS = stepSize //Atenuates step size

      //Full update
      val fullGradient = gradient.toArray
      val newWeights = weightsOld.toArray.zip(fullGradient).map{ case (wold,fg) => wold + stepS*fg }
      val newVector = Vectors.dense(newWeights)
      (newVector, 0) //Second parameter is not used
    }
  }

  /****************************************************/
  /******************** METHODS **********************/
  /****************************************************/

def votingbin(dataset: DataFrame) = {
  //dataset.groupBy("example").agg(when(mean(col("value")) > 0.5, 1).otherwise(0) as "value")
  dataset.groupBy("example").agg(mean(col("value")) as "mu")
}

 val COMPTHRES = pow(10, -2) 
 val BIGNUMBER = pow(10, 5) 

 def nearZero(x:Double): Boolean = (x < COMPTHRES) && (x > -COMPTHRES)

 def sigmoid(x: Double): Double = 1 / (1 + exp(-x))

 def prodlog(x:Double,l:Double) = if (nearZero(x)) 0 
                                   else x * logLim(l)
                                   
 def logLim(x:Double) = if ( nearZero(x) ) (-BIGNUMBER) else log(x)

def rowToVectorMu(r: Row): (Double,Vector) = {
  val s: Seq[Double] = r.toSeq.map( x => 
          x match {
            case d: Double => d
            case d: Int => d.toDouble
            case d: Long => d.toDouble
      })
  (0.0,Vectors.dense(s.toArray))
}

def mStep(data: DataFrame, gradIters: Integer, gradThreshold: Double, gradLearningRate: Double, gradientDataFraction: Double, alpha: Array[Double], beta: Array[Double], 
            classWeights: Array[Double], nExamples: Long, nAnnotators: Int, nClusters: Int): (Array[Double], Array[Double], Array[Double]) = {
  val dataset = data.cache()
  val optiData = dataset.rdd.map(rowToVectorMu _)
  val grad = new CGladGradient(nExamples, nAnnotators, nClusters)
  val updater = new CGladUpdater()
  //val rand = new Random() //First weight estimation is random
  val initialWeights = Vectors.dense(alpha++beta++classWeights) //Vectors.dense(Array.tabulate(nAnnotators+nClusters)(x => rand.nextDouble())) 
  //val initialWeights = Vectors.dense(Array.tabulate(nAnnotators+nClusters)(x => rand.nextDouble())) 
  //val initialWeights = Vectors.dense(Array.tabulate(nAnnotators+nClusters)(x => 0.5)) 
  val opt = GradientDescent.runMiniBatchSGD(optiData,grad,updater,gradLearningRate,gradIters,0,gradientDataFraction,initialWeights,gradThreshold)._1
  val optWeights: Array[Double] = opt.toArray
  val nAlpha = optWeights.slice(0,nAnnotators)
  val nBeta = optWeights.slice(nAnnotators, nAnnotators+nClusters)
  val positiveWeight = dataset.select("example", "mu").distinct.agg(mean("mu")).collect()(0).getDouble(0)
  (nAlpha,nBeta,Array(1-positiveWeight, positiveWeight))
}

def eStep(dataset: DataFrame, alpha: Array[Double], beta: Array[Double], classWeights: Array[Double]): DataFrame = {
  val partialDataset = dataset.cache()
  val mu_af = new MuEstimate(alpha, beta, classWeights)
  val mu = partialDataset.groupBy("example").agg(mu_af(col("annotator"), col("cluster"), col("value")) as "mu")
  partialDataset.select("example", "annotator", "value", "cluster").join(mu, "example")
}

def nLogLikelihood(partialDataset: DataFrame, alpha: Array[Double], beta: Array[Double], classWeights: Array[Double]): Double = {
  val like_uf = new LogLikelihoodEstimate(alpha, beta, classWeights)
  -partialDataset.groupBy("example").agg(like_uf(col("annotator"), col("cluster"), col("value"), col("mu")) as "like").agg(sum(col("like"))).collect()(0).getDouble(0)
}

def initialization(dataset: DataFrame, rank: Integer, k: Integer, seed: Long): (DataFrame, Array[Double],Array[Double],Array[Double], Long, Int, Dataset[ExampleRanks]) = {
  import dataset.sparkSession.implicits._
  val datasetCached = dataset.cache() 
  val nAnnotators = datasetCached.select("annotator").distinct().count()
  val nExamples = datasetCached.select("example").distinct().count()
  val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("annotator").setItemCol("example").setRatingCol("value").setRank(rank)
  val alsmodel = als.fit(datasetCached)
  val itemfactors = alsmodel.itemFactors
  val featurevectors = itemfactors.map((r: Row) => ExampleRanks(r.getInt(0), 
                              VectorsMl.dense(asScalaBuffer(r.getList[Float](1)).toArray.map(x => x.toDouble))))
                                    .as[ExampleRanks]
  val kmeans = new KMeans().setK(k).setSeed(seed)
  val kmodel = kmeans.fit(featurevectors)
  val clusters = kmodel.transform(featurevectors)
  val mapping = clusters.select($"id" as "example", $"prediction" as "cluster")
  val pred = votingbin(datasetCached)
  val joined = datasetCached.join(pred, "example").join(mapping, "example").select(col("*"))
  val rand = new Random(seed) //First weight estimation is random
  //val betaInit = Array.tabulate(k)(x => rand.nextDouble())
  val betaInit = Array.tabulate(k)(x => 0.5)
  val alphaInit = Array.tabulate(nAnnotators.toInt)(x => 0.5)
  val classWeights = Array.fill(2)(0.5) //Placeholder
  (joined, alphaInit, betaInit, classWeights, nExamples, nAnnotators.toInt, featurevectors)
}

def step(partialDataset: DataFrame, alpha: Array[Double], beta: Array[Double], 
          classWeights: Array[Double], logLike: Double, gradIters: Int, gradThreshold: Double, 
          gradLearningRate: Double, gradDataFraction: Double, backtrackingLimit: Double, nExamples: Long, nAnnotators: Int, nClusters: Int, iter: Int, reductionFactor: Double = 1): (DataFrame, Array[Double], Array[Double], Array[Double], Double, Double, Double) = {

    val (nAlpha, nBeta, nWeights) = mStep(partialDataset, gradIters, gradThreshold, gradLearningRate, gradDataFraction, alpha, beta, classWeights, nExamples, nAnnotators, nClusters)
    //println("***Step***")
    //println("OldAlpha")
    //println(alpha.mkString(","))
    //println("NewAlpha")
    //println(nAlpha.mkString(","))
    //println("OldBeta")
    //println(beta.mkString(","))
    //println("NewBeta")
    //println(nBeta.mkString(","))
    //println("OldWeights")
    //println(classWeights.mkString(","))
    //println("NewWeights")
    //println(nWeights.mkString(","))
    val nPartialDataset = eStep(partialDataset, nAlpha, nBeta, nWeights).cache().localCheckpoint()
    //nPartialDataset.filter("example < 5").show(50, false)
    val nLogLike = nLogLikelihood(nPartialDataset, nAlpha, nBeta, nWeights)
    val improvement = if (nLogLike.isNaN) -1 else logLike -  nLogLike 
    //println("NegativeLogLikelihood")
    //println(nLogLike)
    val rateSize = gradLearningRate/(3 * reductionFactor)
    if (iter > 1 && improvement < 0 && rateSize > backtrackingLimit) {
      //println("Gradient Learning rate is too big")
      //println("New learning rate" + rateSize)
      return step(partialDataset, alpha, beta, 
          classWeights, logLike, gradIters, gradThreshold, rateSize, gradDataFraction, backtrackingLimit,
           nExamples, nAnnotators, nClusters, iter, reductionFactor+1) 
    } else {
      return (nPartialDataset, nAlpha, nBeta, nWeights, nLogLike, improvement, gradLearningRate)
    }
}
  

  def apply(dataset: Dataset[BinaryAnnotation], eMIters: Int = 10, eMThreshold: Double = 0,  
            gradIters: Int = 100, gradThreshold: Double = 0, gradLearningRate: Double=0.01, gradDataFraction: Double = 1.0,
            backtrackingLimit: Double = 0.1, rank: Integer = 8, k: Integer= 32, seed: Long = 1L) = {

    import dataset.sparkSession.implicits._
    //Initialization
    val d = dataset.toDF()
    //d.filter("example < 5").show(50, false)

    val (partialDataset, alpha, beta, classWeights, nExamples, nAnnotators, exampleRank) = initialization(d, rank, k, seed)
//partialDataset: DataFrame, alpha: Array[Double], 
//                  beta: Array[Double], 
//                  classWeights: Array[Double], logLike: Double, improvement: Double,
    //Prepare for steps
    //partialDataset.filter("example < 5").show(50, false)
    val stepF = (iterObject: (DataFrame, Array[Double], Array[Double], Array[Double], Double, Double, Double),
                  i:Int) => step(iterObject._1, iterObject._2, iterObject._3, iterObject._4, iterObject._5, gradIters, 
                                gradThreshold, iterObject._7, gradDataFraction, backtrackingLimit, nExamples, nAnnotators, k, i)
    val (nPartialDataset, nAlpha, nBeta, nClassWeights, nLogLike, nImprovement, nLearning) = stepF((partialDataset, 
      alpha, beta, classWeights, 0.0, 100000.0, gradLearningRate), 1)

    //Repeats until some condition is met
    val l = Stream.range(2,eMIters).scanLeft((nPartialDataset, nAlpha, nBeta, nClassWeights, nLogLike, 1.0, nLearning))(stepF)
      .takeWhile( { case (_,_,_,_,_,improvement,_) => improvement > eMThreshold })
                                    .last
    //l._1.show()


    //Results: Ground Truth estimation, class prior estimation and annotator quality matrices
    val preparedDataset = l._1.select(col("example"), col("mu") as "value")
    new CGladModel(preparedDataset.as[BinarySoftLabel], //Ground truth estimate
                        l._2, //Model parameters 
                        l._3, //Difficulty for each example
                        l._1.select("example", "cluster").as[ExampleCluster],
                        exampleRank)
  }

}


