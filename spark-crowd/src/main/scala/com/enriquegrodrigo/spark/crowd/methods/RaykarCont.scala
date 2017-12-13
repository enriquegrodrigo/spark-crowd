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
import com.enriquegrodrigo.spark.crowd.utils.Functions

import org.apache.spark.sql._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.optimization._
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Vector,Vectors}

import scala.util.Random
import scala.math.{pow => powMath}

/**
 *  Provides functions for transforming an annotation dataset into 
 *  a standard label dataset using the Raykar algorithm for multiclass
 *
 *  This algorithm only works with [[types.RealAnnotation]] datasets. There are versions for the 
 *  [[types.BinaryAnnotation]] ([[RaykarBinary]]) and [[types.MulticlassAnnotation]] ([[RaykarMulti]]).
 *
 *  It will return a [[types.RaykarContModel]] with information about the estimation of the 
 *  ground truth for each example, the annotator precision estimation 
 *  of the model, the weights of the linear regression model learned and 
 *  the MAE of the model. 
 *
 *  The next example can be found in the examples folders. In it, the user may also find an example
 *  of how to add prior confidence on the annotators.
 *
 *  @example
 *  {{{
 *    import com.enriquegrodrigo.spark.crowd.methods.RaykarCont
 *    import com.enriquegrodrigo.spark.crowd.types._
 *    
 *    sc.setCheckpointDir("checkpoint")
 *    
 *    val exampleFile = "data/cont-data.parquet"
 *    val annFile = "data/cont-ann.parquet"
 *    
 *    val exampleData = spark.read.parquet(exampleFile)
 *    val annData = spark.read.parquet(annFile).as[RealAnnotation] 
 *    
 *    //Applying the learning algorithm
 *    val mode = RaykarCont(exampleData, annData)
 *    
 *    //Get MulticlassLabel with the class predictions
 *    val pred = mode.getMu().as[RealLabel] 
 *    
 *    //Annotator precision matrices
 *    val annprec = mode.getAnnotatorPrecision()
 *    
 *    //Annotator likelihood 
 *    val like = mode.getLogLikelihood()
 *  }}}
 *  @author enrique.grodrigo
 *  @version 0.1.5
 *  @see Raykar, Vikas C., et al. "Learning from crowds." Journal of Machine
 *  Learning Research 11.Apr (2010): 1297-1322.
 *  
 */

object RaykarCont {

  /****************************************************/
  /****************** CASE CLASSES ********************/
  /****************************************************/

 /**
   *  RaykarCon partial model shared through iterations 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class RaykarContPartialModel(dataset: DataFrame, annotatorData: Dataset[RealAnnotation], 
                                    mu: Dataset[RealLabel], lambda: Dataset[RealAnnotatorPrecision], 
                                    weights: Broadcast[Array[Double]], logLikelihood: Double, 
                                    improvement: Double, nAnnotators: Int, nFeatures: Int) {

    def modify(nDataset: DataFrame =dataset, 
        nAnnotatorData: Dataset[RealAnnotation] =annotatorData, 
        nMu: Dataset[RealLabel] =mu, 
        nLambdas: Dataset[RealAnnotatorPrecision] =lambda, 
        nWeights: Broadcast[Array[Double]] =weights, 
        nLogLikelihood: Double =logLikelihood, 
        nImprovement: Double =improvement, 
        nNAnnotators: Int =nAnnotators, 
        nNFeatures: Int =nFeatures) = 
          new RaykarContPartialModel(nDataset, nAnnotatorData, nMu, nLambdas, 
            nWeights, nLogLikelihood, nImprovement, nNAnnotators, nNFeatures)
  }

  /**
   *  Mu estimation for an example 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class MuEstimation(example: Long, mu:Double) 
  
  /**
  *  Linear regresion prediction for an example
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] case class LinearPrediction(example: Long, mu:Double) 

  /**
  *  Lambda estimation for each annotator
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] case class AnnotatorLambda(annotator: Long, lambda:Double) 

  /**
  *  Annotation with prediction information for the example 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] case class AnnotationsWithPredictions(example: Long, annotator: Long, value:Double, prediction:Double) 

  /**
  *  Annotation with lambda information of the annotator
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] case class AnnotationsWithLambda(example: Long, annotator: Long, value:Double, lambda:Double) 

  /****************************************************/
  /****************** AGGREGATORS ********************/
  /****************************************************/

  /**
   *  Obtains a estimation of lambda for an annotator 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] class LambdaAggregator() extends Aggregator[AnnotationsWithPredictions, (Double,Double), Double] {
    def zero: (Double,Double)= (0.0,0.0)
    def reduce(b: (Double, Double), a: AnnotationsWithPredictions) : (Double,Double) = (b._1 + powMath((a.value - a.prediction),2), b._2 + 1) 
    def merge(b1: (Double,Double), b2: (Double,Double)) : (Double,Double) = (b1._1 + b2._1, b1._2 + b2._2)
    def finish(b: (Double,Double)) = b._2/b._1 
    def bufferEncoder: Encoder[(Double,Double)] = Encoders.product[(Double,Double)]
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }

  /**
   *  Obtains an estimation of mu for each example 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] class MuAggregator() extends Aggregator[AnnotationsWithLambda, (Double,Double), Double] {
    def zero: (Double,Double)= (0.0,0.0)
    def reduce(b: (Double, Double), a: AnnotationsWithLambda) : (Double,Double) = (b._1 + (a.lambda * a.value) , b._2 + a.lambda) 
    def merge(b1: (Double,Double), b2: (Double,Double)) : (Double,Double) = (b1._1 + b2._1, b1._2 + b2._2)
    def finish(b: (Double,Double)) = b._1/b._2 
    def bufferEncoder: Encoder[(Double,Double)] = Encoders.product[(Double,Double)]
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }

  /****************************************************/
  /******************** GRADIENT **********************/
  /****************************************************/

  /**
   *  Computes square error of an instance 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] def computeLikeInstance(mui: Double, predi: Double) = powMath(mui-predi,2) 

  /**
   *  Gradient descent method for the linear regresion part of the model 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] class RaykarContGradient() extends Gradient {

    override def compute(data: Vector, label: Double, weights: Vector, cumGradient:Vector): Double = {
      val w = weights.toArray 
      val s: Array[Double] = data.toArray
      val x = s.drop(1)
      val xw = (for { (value,i) <- x.zipWithIndex } yield value*w(i+1)).sum + w(0)
      val innerPart = label-xw
      val sumTerm = (Array(1.0)++x).map(_ * innerPart)
      val cumGradientArray = cumGradient.toDense.values
      cumGradient.foreachActive({ case (i,gi) => cumGradientArray(i) -= sumTerm(i) })
      val loss = computeLikeInstance(label,xw)
      loss
    }

  }

  /****************************************************/
  /******************** METHODS **********************/
  /****************************************************/

  /**
  *  Applies the learning algorithm
  *
  *  @param dataset the dataset with feature vectors (spark Dataframe).
  *  @param annDataset the dataset with the annotations (spark Dataset of [[types.RealAnnotation]].
  *  @param iters number of iterations for the EM algorithm
  *  @param threshold logLikelihood variability threshold for the EM algorithm
  *  @param gradIters maximum number of iterations for the GradientDescent algorithm
  *  @param gradThreshold threshold for the log likelihood variability for the gradient descent algorithm
  *  @param gradLearning learning rate for the gradient descent algorithm 
  *  @return [[com.enriquegrodrigo.spark.crowd.types.RaykarContModel]]
  *  @author enrique.grodrigo
  *  @version 0.1.5
  */
  def apply(dataset: DataFrame, annDataset: Dataset[RealAnnotation], eMIters: Int = 5, 
            eMThreshold: Double = 0.001, gradIters: Int = 100, gradThreshold: Double = 0.1, 
            gradLearning: Double=0.1): RaykarContModel = {
    import dataset.sparkSession.implicits._
    val datasetFixed = dataset.withColumn("comenriquegrodrigotempindependent", lit(1))
    val initialModel = initialization(datasetFixed, annDataset)
    val secondModel = step(gradIters, gradThreshold, gradLearning)(initialModel,0)
    val fixed = secondModel.modify(nImprovement=1)
    //Loop until the conditions are met
    val l = Stream.range(1,eMIters).scanLeft(fixed)(step(gradIters, gradThreshold, gradLearning))
                                    .takeWhile( (model) => model.improvement > eMThreshold )
                                    .last
    new RaykarContModel(l.mu, l.lambda, l.weights.value)
  }

 /**
  *  Initialize the parameters.  
  *  First ground truth estimation is done using the majority voting algorithm
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def initialization(dataset: DataFrame, annotatorData: Dataset[RealAnnotation]): RaykarContPartialModel = {
    val sc = dataset.sparkSession.sparkContext
    import dataset.sparkSession.implicits._
    val annCached = annotatorData.cache() 
    val datasetCached = dataset.cache() 
    val nFeatures = datasetCached.take(1)(0).length - 1 //Class and example
    val nAnnotators = annCached.select($"annotator").distinct().count().toInt

    val mu = MajorityVoting.transformReal(annCached)
    val placeholderLambda = Seq(RealAnnotatorPrecision(0,0.0)).toDS()
    RaykarContPartialModel(dataset, annotatorData, mu, 
                                placeholderLambda, 
                                sc.broadcast(Array.fill(nFeatures)(0.0)), 
                            0,0, nAnnotators.toInt, nFeatures.toInt) 
  }

 /**
  *  Compute prediction for an example with the weights vector.  
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def computePred(weights: Broadcast[Array[Double]], featureVector: Row) = {
    val x: Seq[Double] = featureVector.toSeq.tail.tail.map(castRowMember)   //example and mu
    val w = weights.value
    val xw = (for { (value,i) <- x.zipWithIndex } yield value*w(i)).sum 
    xw
  }

 /**
  *  Compute lambda estimation 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def computeLambda(joined: Dataset[AnnotationsWithPredictions]): Dataset[RealAnnotatorPrecision] = {
    import joined.sparkSession.implicits._
    joined.groupByKey(_.annotator).agg((new LambdaAggregator()).toColumn)
                                  .as[(Long,Double)]
                                  .map{ case (annotator, lambda) => RealAnnotatorPrecision(annotator, lambda) }
                                  .as[RealAnnotatorPrecision]
  }

 /**
  *  Estimation of the ground truth 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def computeYPred(annotations: Dataset[RealAnnotation], lambdas: Dataset[RealAnnotatorPrecision]) = {
    import annotations.sparkSession.implicits._
    annotations.joinWith(lambdas, annotations.col("annotator") === lambdas.col("annotator"))
               .as[(RealAnnotation, RealAnnotatorPrecision)]
               .map{case (RealAnnotation(example, annotator, value), RealAnnotatorPrecision(_,lambda)) => 
                              AnnotationsWithLambda(example,annotator,value,lambda)}
               .as[AnnotationsWithLambda]
               .groupByKey(_.example)
               .agg((new MuAggregator).toColumn)
               .as[(Long, Double)]
               .map{ case (example, prediction) => RealLabel(example, prediction) }
               .as[RealLabel]
  }

  /**
  *  Cast row member 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def castRowMember(m: Any) : Double = m match {
            case m: Double => m 
            case m: Int => m.toDouble
  }
  
  /**
  *  Prepare data for MLlib gradient descent algorithm 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def prepareDataGradient(data: DataFrame, mu: Dataset[RealLabel]) : 
          DataFrame = {

      val muFixed = mu.withColumnRenamed("value", "comenriquegrodrigotempmu") 
      val fullData = data.join(muFixed, "example")
      val features = fullData.columns.filter(x => (!x.startsWith("comenriquegrodrigotemp") && (x != "example")))
                                     .map(col)
      fullData.select((Array(col("comenriquegrodrigotempmu"),
                              col("example"),
                              col("comenriquegrodrigotempindependent")) ++
                              features):_*)
    }

  /**
  *  Convert prepared data to RDD  
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def gradientDataToRDD(data: DataFrame) : RDD[(Double, Vector)] = {
   data.rdd.map((x: Row) => 
           (x.getDouble(0), 
             Vectors.dense(Array.range(2,x.size).map(i => castRowMember(x.get(i))))))
  }

  /**
  *  Step of the iterative algorithm
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def step(gradIters: Int, gradThreshold: Double, gradLearning: Double)(model: RaykarContPartialModel, i: Int): RaykarContPartialModel = {
    import model.dataset.sparkSession.implicits._ 
    
    //Linear regression optimization
    val preparedData: DataFrame = prepareDataGradient(model.dataset, model.mu)
    val preparedRDD: RDD[(Double, Vector)] =  gradientDataToRDD(preparedData)
    val grad = new RaykarContGradient()
    val updater = new SimpleUpdater()
    val rand = new Random(0) //First weight estimation is random
    val initialWeights = Vectors.dense(Array.tabulate(model.nFeatures)(x => rand.nextDouble())) 
    val opt = GradientDescent.runMiniBatchSGD(preparedRDD,grad,updater,gradLearning,gradIters,0,1,initialWeights,gradThreshold)._1
    val optWeights = opt.toArray
    val weights = model.mu.sparkSession.sparkContext.broadcast(optWeights)
    val pred = preparedData.map(r => RealLabel(r.getLong(1), computePred(weights, r))) 
    
    val joined: Dataset[AnnotationsWithPredictions] = 
      model.annotatorData.joinWith(pred, 
                                     model.annotatorData.col("example") === pred.col("example"))
                         .as[(RealAnnotation,RealLabel)]
                         .map(x => AnnotationsWithPredictions(x._1.example, 
                                                                x._1.annotator, 
                                                                x._1.value, 
                                                                x._2.value))
    //Estimation of annotator precision
    val lambdas: Dataset[RealAnnotatorPrecision] = computeLambda(joined)

   
    //Estimation of ground truth EStep
    val mu: Dataset[RealLabel] = computeYPred(model.annotatorData, lambdas)

    //Estimation of least squares error 
    val like = pred.joinWith(mu, pred.col("example") === mu.col("example")).map(x => computeLikeInstance(x._2.value, x._1.value)).reduce(_+_) / model.dataset.count()

    val muFix = mu.map{ case RealLabel(annotator, mu) => RealLabel(annotator, mu) }
    model.modify(nMu=muFix.cache(), nWeights=weights, nLambdas = lambdas, nLogLikelihood = like, nImprovement = model.logLikelihood-like)
  }

}

