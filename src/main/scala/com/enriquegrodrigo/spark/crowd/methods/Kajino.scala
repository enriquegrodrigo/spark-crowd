
package com.enriquegrodrigo.spark.crowd.methods

import com.enriquegrodrigo.spark.crowd.types.KajinoPartialModel
import com.enriquegrodrigo.spark.crowd.types.KajinoModel
import com.enriquegrodrigo.spark.crowd.types.KajinoPriors
import com.enriquegrodrigo.spark.crowd.types.KajinoEstimation
import com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation
import com.enriquegrodrigo.spark.crowd.types.BinarySoftLabel
import com.enriquegrodrigo.spark.crowd.aggregators.RaykarBinaryStatisticsAggregator 
import com.enriquegrodrigo.spark.crowd.utils.Functions

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions.lit
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.optimization.Gradient
import org.apache.spark.mllib.optimization.LogisticGradient
import org.apache.spark.mllib.optimization.SimpleUpdater
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.optimization.Updater
import org.apache.spark.mllib.optimization.LBFGS
import org.apache.spark.mllib.optimization.GradientDescent
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors

import scala.util.Random
import scala.math._

/**
 *  Provides functions for transforming an annotation dataset into 
 *  a standard label dataset using the Kajino algorithm 
 *
 *  This algorithm only works with [[com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation]] datasets
 *
 *  @example
 *  {{{
 *    result: KajinoModel  = Kajino(dataset, annotations)
 *  }}}
 *  @author enrique.grodrigo
 *  @version 0.1 
 *  @see Kajino, Hiroshi, Yuta Tsuboi, and Hisashi Kashima. "A Convex Formulation 
 *  for Learning from Crowds." Transactions of the Japanese Society for Artificial 
 *  Intelligence 27.3 (2012): 133-142. 
 *  
 */
object Kajino {

  /**
  * Computes logistic function values.
  */
  private def computeSigmoid(x: Array[Double], w: Array[Double]): Double = {
      val vectMult = x.zip(w).map{case (x,w) =>  x*w}
      Functions.sigmoid(vectMult.sum)
  }

  /**
  * Computes loss from the likelihood function
  */
  private def computePointLoss(est: Double, sig: Double): Double = {
    -(Functions.prodlog(est,sig) + Functions.prodlog(est,sig))
  }  

  /**
  * Updater for the SGD algorithm 
  */
  private class KajinoUpdater(w0: Broadcast[Array[Double]], priors: Broadcast[KajinoPriors]) extends Updater {
    def compute(weightsOld:Vector, gradient: Vector, stepSize: Double, iter: Int, regParam: Double) = {
      val lambda = priors.value.lambda
      val step = stepSize/sqrt(iter)
      val secTerm = weightsOld.toArray.zip(w0.value).map{ case (old,v0) => lambda * (old - v0) }
      val fullGradient = gradient.toArray.zip(secTerm).map{case (g,t) => g + t}
      val newWeights = weightsOld.toArray.zip(fullGradient).map{ case (wold,fg) => wold - step*fg }
      val newVector = Vectors.dense(newWeights)
      (newVector, lambda)
    }
  }

  /**
  * Gradient calculation for the SGD algorithm
  */
  private class KajinoGradient() extends Gradient {

    override def compute(data: Vector, label: Double, weights: Vector, cumGradient:Vector): Double = {
      val w = weights.toArray 
      val x: Array[Double] = data.toArray
      val sigm = computeSigmoid(x,w)
      val innerPart = label-sigm
      val sumTerm = x.map(_ * innerPart)
      val cumGradientArray = cumGradient.toDense.values
      cumGradient.foreachActive({ case (i,gi) => cumGradientArray(i) += -sumTerm(i) })
      val loss = computePointLoss(label,sigm)
      loss
    }
  }


  /**
   * Compute probability with a weight vector
   */
  private def computeProb(bweights: Broadcast[Array[Double]])(example: Row): BinarySoftLabel = {
       val weights = bweights.value
       val exampleID = example.getLong(0)
       val s: Array[Double] = example.toSeq.map( x => 
            x match {
              case d: Double => d
              case d: Int => d.toDouble 
              case d: Long => d.toDouble 
            }
      ).toArray.tail

      val toSigm = s.zip(weights).map{case (x,w) => x*w }.reduce(_+_)
      val sigm = Functions.sigmoid(toSigm) 
      BinarySoftLabel(exampleID,sigm)
  }

  /**
  *  Applies the learning algorithm
  *
  *  @param dataset the dataset with feature vectors.
  *  @param annDataset the dataset with the annotations.
  *  @param maxIters number of iterations of the Kajino algorithm 
  *  @param iterThreshold threshold for the improvement in the likelihood 
  *  @param lambda lambda parameter of the algorithm
  *  @param eta eta parameter of the algorithm
  *  @param gradIters maximum number of iterations for the gradient descent algorithm
  *  @param gradLearning learning rate for the gradient descent algorithm 
  *  @param gradThreshold threshold for the log likelihood variability for the gradient descent algorithm
  *  @return [[com.enriquegrodrigo.spark.crowd.types.KajinoModel]]
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def apply(dataset: DataFrame, annDataset: Dataset[BinaryAnnotation], maxIters: Int = 1, iterThreshold: Double = 0.001, lambda: Double=1, eta: Double=1, gradIters: Int = 100, gradLearning: Double = 1.0, gradThreshold: Double= 0.01): KajinoModel = {
    import dataset.sparkSession.implicits._
    val sc = dataset.sparkSession.sparkContext
    val priors = KajinoPriors(lambda,eta)
    val dataFixed = dataset.withColumn("com.enriquegrodrigo.temp.independent", lit(1))
    val initialModel = initialization(dataFixed, annDataset, priors, gradIters, gradLearning, gradThreshold)
    val secondModel = step(gradIters, gradLearning, gradThreshold)(initialModel,0)
    val fixed = secondModel.modify(nVariation=1)
    val l = Stream.range(1,maxIters).scanLeft(fixed)(step(gradIters, gradLearning, gradThreshold))
                                    .takeWhile( (model) => model.variation > iterThreshold )
                                    .last
    val sw0 = l.w0.value.mkString(" ")                                   
    val fet = l.nFeatures                                   
    val w = sc.broadcast(l.w(1))
    val estimation = l.dataset.map(computeProb(l.w0)).as[BinarySoftLabel]  
    new KajinoModel(estimation,l.w0.value, l.w)
  }

  /**
   * Converts row to RDD for SGD
   */
  private def rowToRDD(r: Row, colNames: Array[String]): (Double,Vector) = {

      val annotationIndex = colNames.indexOf("com_enriquegrodrigo_spark_crowd_temp_value")
      val annotation = r.getInt(annotationIndex).toDouble

      val s: Seq[Double] = r.toSeq.map( x => 
            x match {
              case d: Double => d
              case d: Int => d.toDouble 
            }
      ).toArray

      val noAnn  = s.zipWithIndex.filterNot{ case (v,i) => i == annotationIndex }.map(x => x._1).toArray
      val v = Vectors.dense(noAnn)
      (annotation, v)
  }
 
  /**
   * Initilalization of weights for each annotator
   */
  private def initializeWJ(j: Int, data: DataFrame, annotatorData: Dataset[BinaryAnnotation], nFeatures: Int, gradIters: Int , gradLearning: Double, gradThreshold: Double): Array[Double] = { 
    import data.sparkSession.implicits._

    val jAnnData = annotatorData.filter(_.annotator == j)
                                .toDF()
                                .drop("annotator")
                                .withColumnRenamed("example", "com_enriquegrodrigo_spark_crowd_temp_example")
                                .withColumnRenamed("value", "com_enriquegrodrigo_spark_crowd_temp_value")

    val joined = data.join(jAnnData, data.col("example") === jAnnData.col("com_enriquegrodrigo_spark_crowd_temp_example"))
                     .drop("example")
                     .drop("com_enriquegrodrigo_spark_crowd_temp_example")
    val colNames = joined.columns
    val optiData = joined.map(r => rowToRDD(r,colNames)).as[(Double,Vector)].rdd
    
    val grad = new LogisticGradient()
    val updaterL = new SimpleUpdater()
    val lbfgs = new LBFGS(grad, updaterL) 

    val randAr = Array.tabulate(nFeatures)(x => Random.nextDouble)
    val initialWeights = Vectors.dense(randAr)
    val opt = GradientDescent.runMiniBatchSGD(optiData,grad,updaterL,gradLearning,gradIters,0,1,initialWeights,gradThreshold)._1
    //val opt = lbfgs.optimize(optiData,initialWeights)
    val optWeights = opt.toArray
    optWeights
  }

  /**
   * Initialization step of the algorithm
   */
  private def initialization(dataset: DataFrame, annotatorData: Dataset[BinaryAnnotation], priors: KajinoPriors, gradIters: Int , gradLearning: Double, gradThreshold: Double):KajinoPartialModel = {
    val sc = dataset.sparkSession.sparkContext
    import dataset.sparkSession.implicits._
    val annCached = annotatorData.cache() 
    val datasetCached = dataset.cache() 
    val nFeatures = datasetCached.take(1)(0).length - 1 //example
    val nAnnotators = annCached.select($"annotator").distinct().count().toInt
    val kajprior = sc.broadcast(priors)
    val wJ: Array[Array[Double]] = Array.tabulate(nAnnotators)(j => initializeWJ(j, dataset, annotatorData,nFeatures, gradIters, gradLearning, gradThreshold)) 
    val w0 = Array.fill(nFeatures)(0.0)
    KajinoPartialModel(dataset, annotatorData, sc.broadcast(w0), wJ, kajprior, 0,
      nAnnotators.toInt, nFeatures.toInt) 
  }


  /**
   * Estimation of the weights for the general logistic regresion model 
   */
  private def optimizeW0(model: KajinoPartialModel): KajinoPartialModel = {

    def mergeWeights(x: Array[Double], y: Array[Double]) : Array[Double] = {
      x.zip(y).map{ case (x,y) => x + y } 
    }

    val sc = model.dataset.sparkSession.sparkContext
    val weights = model.w
    val eta = model.priors.value.eta
    val lambda = model.priors.value.lambda
    val nAnn = model.nAnnotators
    val wMat = weights.foldLeft("")((s,x) => s + "\n" + x.map(k => f"$k%2.4f").mkString(" "))
    val w0 = weights.reduce(mergeWeights(_,_))
                    .map((x:Double) => (x*lambda)/(eta + nAnn*lambda))
    val sw0 = w0.mkString(" ")                                   

    val variation = model.w0.value.zip(w0).map{ case (oldval,newval) => abs(oldval - newval) }.sum
    model.modify(nW0 = sc.broadcast(w0), nVariation = variation)
  }

  /**
   * Calculation of the weights for the personal logistic regresion model for each annotator
   */
  private def computeWJ(j: Int, model: KajinoPartialModel, gradIters: Int , gradLearning: Double, gradThreshold: Double): Array[Double] = { 
     import model.dataset.sparkSession.implicits._
   
    val jAnnData = model.annotatorData.filter(_.annotator == j)
                                .drop("annotator")
                                .withColumnRenamed("example", "com_enriquegrodrigo_spark_crowd_temp_example")
                                .withColumnRenamed("value", "com_enriquegrodrigo_spark_crowd_temp_value")
                                .toDF()

    val joined = model.dataset.join(jAnnData, model.dataset.col("example") === jAnnData.col("com_enriquegrodrigo_spark_crowd_temp_example"))
            .drop("example")
            .drop("com_enriquegrodrigo_spark_crowd_temp_example")
    val colNames = joined.columns
    val optiData = joined.map(r => rowToRDD(r,colNames)).as[(Double,Vector)].rdd
    
    val grad = new LogisticGradient()
    val updaterL = new SimpleUpdater()
    val updater = new KajinoUpdater(model.w0, model.priors)
    val lbfgs = new LBFGS(grad, updaterL) 

    val rand = new Random()
    val randAr = Array.tabulate(model.nFeatures)(x => Random.nextDouble)
    val initialWeights = Vectors.dense(randAr)
    val opt = GradientDescent.runMiniBatchSGD(optiData,grad,updater,gradLearning,gradIters,0,1,initialWeights,gradThreshold)._1
    //val opt = lbfgs.optimize(optiData,initialWeights)
    val optWeights = opt.toArray
    optWeights
  }

  /**
   * Obtains all annotator logistic regression models 
   */
  private def optimizeWJ(model: KajinoPartialModel, gradIters: Int , gradLearning: Double, gradThreshold: Double): KajinoPartialModel = {
    val sc = model.dataset.sparkSession.sparkContext
    import model.dataset.sparkSession.implicits._
    val wJ = Array.tabulate(model.nAnnotators)(j => computeWJ(j,model, gradIters, gradLearning, gradThreshold)) 
    model.modify(nW = wJ)
  }

  /**
   * Full iteration of the algorithm
   */
  private def step(gradIters: Int , gradLearning: Double, gradThreshold: Double)(model: KajinoPartialModel, i: Int): KajinoPartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val m = optimizeW0(model)
    val result = optimizeWJ(m, gradIters, gradLearning, gradThreshold) 
    result 
  }
}

