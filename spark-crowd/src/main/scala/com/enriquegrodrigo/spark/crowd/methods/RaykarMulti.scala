
package com.enriquegrodrigo.spark.crowd.methods

import com.enriquegrodrigo.spark.crowd.types._
import com.enriquegrodrigo.spark.crowd.utils.Functions

import org.apache.spark.sql._
import org.apache.spark.sql.expressions._
import org.apache.spark.mllib.optimization._
import org.apache.spark.sql.functions._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.broadcast.Broadcast

import scala.util.Random
import scala.math._

/**
 *  Provides functions for transforming an annotation dataset into 
 *  a standard label dataset using the Raykar algorithm for multiclass
 *
 *  This algorithm only works with [[types.MulticlassAnnotation]] datasets. There are versions for the 
 *  [[types.BinaryAnnotation]] ([[RaykarBinary]]) and [[types.RealAnnotation]] ([[RaykarCont]]).
 *
 *  It will return a [[types.RaykarMultiModel]] with information about the estimation of the 
 *  ground truth for each example (probability for each class), the annotator precision estimation 
 *  of the model, the weights of the three (one vs all) logistic regression model learned and 
 *  the log-likelihood of the model. 
 *
 *  The next example can be found in the examples folders. In it, the user may also find an example
 *  of how to add prior confidence on the annotators.
 *
 *  @example
 *  {{{
 *    import com.enriquegrodrigo.spark.crowd.methods.RaykarMulti
 *    import com.enriquegrodrigo.spark.crowd.types._
 *    
 *    sc.setCheckpointDir("checkpoint")
 *    
 *    val exampleFile = "data/multi-data.parquet"
 *    val annFile = "data/multi-ann.parquet"
 *    
 *    val exampleData = spark.read.parquet(exampleFile)
 *    val annData = spark.read.parquet(annFile).as[MulticlassAnnotation] 
 *    
 *    //Applying the learning algorithm
 *    val mode = RaykarMulti(exampleData, annData)
 *    
 *    //Get MulticlassLabel with the class predictions
 *    val pred = mode.getMu().as[MulticlassSoftProb] 
 *    
 *    //Annotator precision matrices
 *    val annprec = mode.getAnnotatorPrecision()
 *    
 *    //Annotator likelihood 
 *    val like = mode.getLogLikelihood()
 *  }}}
 *  @author enrique.grodrigo
 *  @version 0.1 
 *  @see Raykar, Vikas C., et al. "Learning from crowds." Journal of Machine
 *  Learning Research 11.Apr (2010): 1297-1322.
 *  
 */
object RaykarMulti {

  /****************************************************/
  /****************** CASE CLASSES ********************/
  /****************************************************/
  
  /**
   * Partial object get data from one step to another.
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class RaykarMultiPartialModel(dataset: DataFrame, 
                                      annotations: Dataset[MulticlassAnnotation],
                                      mu: Dataset[MulticlassSoftProb],
                                      annotatorPrecision: Dataset[DiscreteAnnotatorPrecision],
                                      logisticWeights: Array[Array[Double]],
                                      logisticPrediction: Dataset[LogisticMultiPrediction],
                                      annotationsLikelihood: Dataset[MulticlassSoftProb],
                                      annotatorPriorMatrix: Broadcast[Array[Array[Array[Double]]]],
                                      weightsPriorMatrix: Array[Broadcast[Array[Array[Double]]]],
                                      likelihood: Double,
                                      improvement: Double,
                                      annotatorClassCombination: Dataset[AnnotatorClassCombination],
                                      nFeatures: Int,
                                      nClasses: Int,
                                      nAnnotators: Int)  {
    def apply(dataset: DataFrame= dataset, 
                annotations: Dataset[MulticlassAnnotation]= annotations, 
                mu: Dataset[MulticlassSoftProb]= mu, 
                annotatorPrecision: Dataset[DiscreteAnnotatorPrecision]= annotatorPrecision,
                logisticWeights: Array[Array[Double]]= logisticWeights, 
                logisticPrediction: Dataset[LogisticMultiPrediction]= logisticPrediction, 
                annotationsLikelihood: Dataset[MulticlassSoftProb]= annotationsLikelihood, 
                annotatorPriorMatrix: Broadcast[Array[Array[Array[Double]]]]= annotatorPriorMatrix, 
                weightsPriorMatrix: Array[Broadcast[Array[Array[Double]]]]= weightsPriorMatrix, 
                likelihood: Double= likelihood, 
                improvement: Double= improvement, 
                annotatorClassCombination: Dataset[AnnotatorClassCombination]=annotatorClassCombination, 
                nFeatures: Int= nFeatures, 
                nClasses: Int= nClasses, 
                nAnnotator: Int= nAnnotators) = {

                  RaykarMultiPartialModel(dataset, 
                                            annotations, 
                                            mu, 
                                            annotatorPrecision, 
                                            logisticWeights, 
                                            logisticPrediction, 
                                            annotationsLikelihood, 
                                            annotatorPriorMatrix, 
                                            weightsPriorMatrix, 
                                            likelihood, 
                                            improvement, 
                                            annotatorClassCombination, 
                                            nFeatures, 
                                            nClasses, 
                                            nAnnotators)

    }


  }

  /**
   *  Combinations of annotator and classes for frequency calculation taking into account all combinations 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class AnnotatorClassCombination(annotator: Long, clas: Int, k: Int)
  
  /**
   *  Dataset with both annotations and class probability estimated in the previous step, for obtaining the 
   *  soft frequency  matrices.
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class AnnotationWithClassProb(example: Long, clas: Int, prob: Double, annotator: Long, annotation: Int)

  /**
   *  Dataset with soft frequency of (annotator,c,k) in the annotations dataset. 
   *  Represents the numerator in the corresponding element of the 
   *  precisions matrices. 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class AnnotatorFrequency(annotator: Long, clas: Int, k:Int, frequency: Double)

  /**
   *  Dataset with soft frequency of (annotator,c) in the annotations dataset
   *  Represents the denominator in the corresponding element of the 
   *  precisions matrices. 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class AnnotatorClassFrequency(annotator: Long, clas: Int, frequency: Double)

  /**
   *  Aggregation of annotator parameters for each example in the one vs all approach for 
   *  logistic regression.
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class LogisticParams(example: Long, a: Double, b: Double)

  /**
   *  Mu estimate with logistic params for the example 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class MuWithLogisticParams(example: Long, mu: Double, a: Double, b:Double)
  
  /**
   *  Logistic Annotator params for the LogisticParams aggregator
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class LogisticAnnotatorParams(a: Double, b:Double)

  /**
   *  Logistic prediction for the one vs all approach  
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class LogisticPrediction(example: Long, prob: Double)

  /**
   *  Logistic prediction for the full multiclass problem 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class LogisticMultiPrediction(example: Long, clas: Int, prob: Double)

  /**
   *  Normalizer for logistic predictions 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class Normalizer(example: Long, norm: Double)


  /**
   *  Annotations with logistic prediction information for EStep 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class AnnotationsWithLogisticPrediction(example: Long, clas: Int, prediction: Double, annotator: Long, annotation: Int) 
  
  /**
   *  EStep estimation point with information about annotation probability and the logistic prediction. 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class EStepEstimationPoint(example: Long, clas: Int, prediction: Double, annotator: Long, annotation: Int, annotationProb: Double) 

  /**
   *  Likelihood estimation point with annotation likelihood as well as the true class estimation form E Step 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] case class LikelihoodPoint(example: Long, clas: Int, mu: Double, annotationsLikelihood: Double) 


  /****************************************************/
  /****************** AGGREGATORS ********************/
  /****************************************************/

  /**
   *  Obtains the likelihood for each example given a class (grouping keys) 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] class AnnotationsLikelihoodAggregator() extends Aggregator[EStepEstimationPoint, (Double,Double), Double] {
    def zero: (Double,Double)= (1.0,-1.0)
    def reduce(b: (Double, Double), a: EStepEstimationPoint) : (Double,Double) =  (a.annotationProb * b._1, a.prediction)
    def merge(b1: (Double,Double), b2: (Double,Double)) : (Double,Double) = (b1._1 * b2._1, if (b1._2 >= 0) b1._2 else b2._2)
    def finish(b: (Double,Double)) = b._1 * b._2 
    def bufferEncoder: Encoder[(Double,Double)] = Encoders.product[(Double,Double)]
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }
 
  /**
   *  Obtains the soft frequency of appearance of the key (j,c,k) 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] class FrequencyAggregator() extends Aggregator[Tuple2[AnnotatorClassCombination,AnnotationWithClassProb], Double, Double] {
    def zero: Double = 0
    def reduce(b: Double, a: (AnnotatorClassCombination, AnnotationWithClassProb)) = a match {
      case (_, null) => b
      case (_, a) => a.prob + b
    }
    def merge(b1: Double, b2: Double) = b1 + b2 
    def finish(b: Double) = b 
    def bufferEncoder: Encoder[Double] = Encoders.scalaDouble
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }
  
  /**
   *  Obtains the soft frequency of appearance of the key (j,c) 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] class ClassFrequencyAggregator() extends Aggregator[Tuple2[AnnotatorClassCombination, AnnotationWithClassProb], Double, Double] {
    def zero: Double = 0
    def reduce(b: Double, a: Tuple2[AnnotatorClassCombination, AnnotationWithClassProb]) = a match {
      case (_, null) => b
      case (_, a) => a.prob + b
    }
    def merge(b1: Double, b2: Double) = b1 + b2 
    def finish(b: Double) = b 
    def bufferEncoder: Encoder[Double] = Encoders.scalaDouble
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }

  /**
   *  Obtains the soft frequency of appearance of the key (j,c) 
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] class LogisticParamAggregator() extends Aggregator[Tuple2[MulticlassAnnotation, DiscreteAnnotatorPrecision], LogisticAnnotatorParams, LogisticAnnotatorParams] {
    def zero: LogisticAnnotatorParams = LogisticAnnotatorParams(1,1) 
    def reduce(b: LogisticAnnotatorParams, a: Tuple2[MulticlassAnnotation, DiscreteAnnotatorPrecision]) = a match {
      case (_, DiscreteAnnotatorPrecision(_,0,_,prob)) => LogisticAnnotatorParams(b.a, b.b * prob) 
      case (_, DiscreteAnnotatorPrecision(_,1,_,prob)) => LogisticAnnotatorParams(b.a * prob, b.b) 
    }
    def merge(b1: LogisticAnnotatorParams, b2: LogisticAnnotatorParams) = LogisticAnnotatorParams(b1.a * b2.a, b1.b*b2.b) 
    def finish(b: LogisticAnnotatorParams) = b 
    def bufferEncoder: Encoder[LogisticAnnotatorParams] = Encoders.product[LogisticAnnotatorParams]
    def outputEncoder: Encoder[LogisticAnnotatorParams] = Encoders.product[LogisticAnnotatorParams]
  }


  /****************************************************/
  /******************** GRADIENT **********************/
  /****************************************************/

  /**
  * Computes the logistic function for a data point 
  */
  private[crowd] def computeSigmoid(x: Array[Double], w: Array[Double]): Double = {
      val vectMult = x.zip(w).map{case (x,w) =>  x*w}
      Functions.sigmoid(vectMult.sum)
  }

  /**
  * Computes the negative likelihood of a point (loss)
  */
  private[crowd] def computePointLoss(mui: Double, pi: Double, ai: Double, bi: Double): Double = {
    val mulaipi = ai*pi
    val mulbipi = bi*(1-pi)
    -(Functions.prodlog(mui,mulaipi) + Functions.prodlog((1-mui),mulbipi))
  }  

  /**
  * Matrix multiplication (TODO: improving using libraries)
  */
  private[crowd] def matMult (mat: Array[Array[Double]], v: Array[Double]): Array[Double] = {
    mat.map(mv => mv.zip(v).map{ case (x,y) => x*y }.reduce(_ + _))
  }

  /**
  * Computes the gradient for the SGD algorithm 
  */
  private[crowd] class RaykarMultiGradient() extends Gradient {

    override def compute(data: Vector, label: Double, weights: Vector, cumGradient:Vector): Double = {
      val w = weights.toArray 
      val s: Array[Double] = data.toArray

      //First 2 columns are special parameters
      val a = s(0)
      val b = s(1)

      // The data point
      val x = s.drop(2)

      //Gradient calculation
      val sigm = computeSigmoid(x,w)
      val innerPart = label-sigm
      val sumTerm = x.map(_ * innerPart)
      val cumGradientArray = cumGradient.toDense.values
      cumGradient.foreachActive({ case (i,gi) => cumGradientArray(i) += sumTerm(i) })

      //Point loss 
      val loss = computePointLoss(label,sigm,a,b)
      loss
    }
  }


  /**
  * Computes updater for the SGD algorithm.
  * Adds the regularization priors.
  */
  private[crowd] class RaykarMultiUpdater(weightsPrior: Broadcast[Array[Array[Double]]]) extends Updater {
    def compute(weightsOld:Vector, gradient: Vector, stepSize: Double, iter: Int, regParam: Double) = {

      val regTerm = matMult(weightsPrior.value, weightsOld.toArray) //Regularization with prior weights

      val stepS = stepSize/scala.math.sqrt(iter) //Atenuates step size

      //Full update
      val fullGradient = gradient.toArray.zip(regTerm).map{case (g,t) => g - t}
      //val fullGradient = gradient.toArray
      val newWeights: Array[Double] = weightsOld.toArray.zip(fullGradient).map{ case (wold,fg) => wold + stepS*fg }
      val newVector = Vectors.dense(newWeights)
      (newVector, 0) //Second parameter is not used
    }
  }


  /****************************************************/
  /******************** METHODS **********************/
  /****************************************************/

  /**
  *  Applies the learning algorithm
  *
  *  @param dataset the dataset with feature vectors (spark Dataframe).
  *  @param annDataset the dataset with the annotations (spark Dataset of [[types.MulticlassAnnotation]]).
  *  @param emIters number of iterations for the EM algorithm
  *  @param emThreshold logLikelihood variability threshold for the EM algorithm
  *  @param gradIters maximum number of iterations for the GradientDescent algorithm
  *  @param gradThreshold threshold for the log likelihood variability for the gradient descent algorithm
  *  @param gradLearning learning rate for the gradient descent algorithm 
  *  @param k_prior prior (Dirichlet distribution hyperparameters) for the estimation
  *  of the probability that an annotator correctly a class given another 
  *  @param w_prior prior for the weights of the logistic regression model
  *  @return [[com.enriquegrodrigo.spark.crowd.types.RaykarBinaryModel]]
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def apply(dataset: DataFrame, annDataset: Dataset[MulticlassAnnotation], eMIters: Int = 5, 
            eMThreshold: Double = 0.001, gradIters: Int = 100, gradThreshold: Double = 0.1, 
            gradLearning: Double=0.1, 
            k_prior: Option[Array[Array[Array[Double]]]]= None, 
            w_prior: Option[Array[Array[Array[Double]]]]= None): RaykarMultiModel = {

    import dataset.sparkSession.implicits._

    //Adds ones column
    val datasetFixed = dataset.withColumn("comenriquegrodrigotempindependent", lit(1))

    //Model execution
    val initialModel = initialization(datasetFixed, annDataset, k_prior, w_prior)
    val secondModel = step(gradIters, gradThreshold, gradLearning)(initialModel,0)
    val fixed = secondModel(improvement=1)
    val l = Stream.range(1,eMIters).scanLeft(fixed)(step(gradIters, gradThreshold, gradLearning))
                                    .takeWhile((model) => model.improvement > eMThreshold)
                                    .last
    //Real model result
    return new RaykarMultiModel(l.mu, 
                                  l.annotatorPrecision,  
                                  l.logisticWeights, 
                                  l.likelihood)
  }
  
  /**
  *  Step of the iterative algorithm
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def step(gradIters: Int, gradThreshold: Double, gradLearning: Double)(model: RaykarMultiPartialModel, 
                            i: Int): RaykarMultiPartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val m = mStep(model, gradIters, gradThreshold, gradLearning)
    val e = eStep(m)
    val result = logLikelihood(e)
    result(mu=result.mu.checkpoint)
  }
  
 /**
  *  Initialize the parameters.  
  *  First ground truth estimation is done using the majority voting algorithm
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def initialization(dataset: DataFrame, annotatorData: Dataset[MulticlassAnnotation], 
                      k_prior: Option[Array[Array[Array[Double]]]], 
                      w_prior: Option[Array[Array[Array[Double]]]]): RaykarMultiPartialModel = {
    val sc = dataset.sparkSession.sparkContext
    val sparkSession = dataset.sparkSession
    import dataset.sparkSession.implicits._

    //Obtaining useful metadata
    val annCached = annotatorData.cache() 
    val datasetCached = dataset.cache() 
    val nFeatures = datasetCached.take(1)(0).length - 1 //Example
    val nAnnotators = annCached.select($"annotator").distinct().count().toInt
    val nClasses = annCached.select($"value").distinct().count().toInt

    //Processing priors (adds uniform prior if user does not provide any)
    val annotatorPrior = k_prior match {
      case Some(arr) => arr 
      case None => Array.fill(nAnnotators,nClasses,nClasses)(2.0) 
    }

    val weightsPrior: Array[Array[Array[Double]]] = w_prior match {
      case Some(arr) => arr 
      case None => Array.tabulate(nClasses,nFeatures,nFeatures){ case (c,x,y) => if (x == y) (if (x == 0) 0 else 1) else 0 } 
    }

    //MajorityVoting estimation
    val estimation = MajorityVoting.transformSoftMulti(annCached)

    //Annotator-Class-Class combinations 
    val combinations = annotatorData.map(_.annotator)
                                    .distinct
                                    .withColumnRenamed("value", "annotator")
                                    .withColumn("clas", explode(array((0 until nClasses).map(lit): _*)))
                                    .withColumn("k", explode(array((0 until nClasses).map(lit): _*)))
                                    .as[AnnotatorClassCombination]

    //Creating Partial model object
    RaykarMultiPartialModel(datasetCached, 
                            annCached, 
                            estimation.cache(), 
                            sparkSession.emptyDataset[DiscreteAnnotatorPrecision],
                            Array.ofDim[Double](nClasses, nFeatures),
                            sparkSession.emptyDataset[LogisticMultiPrediction],
                            sparkSession.emptyDataset[MulticlassSoftProb],
                            sc.broadcast(annotatorPrior),
                            weightsPrior.map(x => sc.broadcast(x)), 
                            -1,
                            -1,
                            combinations.cache(),
                            nFeatures, 
                            nClasses,
                            nAnnotators
                            ) 
  }


 /**
  *  M Step of the EM algorithm.  
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def mStep(model: RaykarMultiPartialModel, gradIters: Int, gradThreshold: Double, gradLearning: Double): RaykarMultiPartialModel = {

    import model.dataset.sparkSession.implicits._ 
    val sc = model.dataset.sparkSession.sparkContext


    /*
     * Obtains frequencies for the combination (annotator, c, k), being c the given class and k the class the annotator
     * annotates.
     */
    def annotatorFrequency( annWithClassProb: Dataset[(AnnotatorClassCombination, AnnotationWithClassProb)] ) : 
          Dataset[AnnotatorFrequency] = {

      annWithClassProb.groupByKey(x => x._1)
                      .agg((new FrequencyAggregator()).toColumn)
                      .map(x => AnnotatorFrequency(x._1.annotator, x._1.clas, x._1.k, x._2))
                      .as[AnnotatorFrequency]
    }

    /*
     *  Obtains frequencies of combinations of (annotator, c), being c the given class for an example
     */
    def annotatorClasFrequency( annWithClassProb: Dataset[(AnnotatorClassCombination, AnnotationWithClassProb)] ) : 
          Dataset[AnnotatorClassFrequency] = {

      annWithClassProb.groupByKey(x => (x._1.annotator, x._1.clas))
                              .agg((new ClassFrequencyAggregator).toColumn)
                              .map(x => AnnotatorClassFrequency(x._1._1, x._1._2, x._2))
                              .as[AnnotatorClassFrequency]

    }


    /**
     * Obtains prediction for the annotator precision (confusion matrix)
     */
    def annotatorPrecision( frequencies: Dataset[AnnotatorFrequency], 
        classFreq: Dataset[AnnotatorClassFrequency] ) : Dataset[DiscreteAnnotatorPrecision] = {
          def getPrecisionWithPrior(annotator: Int, clas: Int, k: Int, num: Double, denom: Double) = {
            val numPrior = model.annotatorPriorMatrix.value(annotator)(clas)(k)
            val denomPrior = model.annotatorPriorMatrix.value(annotator)(clas).sum
            val withoutPrior = num/denom 
            val precision = ((numPrior - 1) + num)/((denomPrior - model.nClasses) + denom)
            precision
          }

      frequencies.joinWith(classFreq, frequencies.col("annotator") === classFreq.col("annotator") &&
                                            frequencies.col("clas") === classFreq.col("clas"))
                 .as[(AnnotatorFrequency, AnnotatorClassFrequency)]
                 .map(x => DiscreteAnnotatorPrecision(x._1.annotator, 
                                x._1.clas, 
                                x._1.k, 
                                getPrecisionWithPrior(x._1.annotator.toInt, x._1.clas, x._1.k, x._1.frequency, x._2.frequency)
                  ))
                 .as[DiscreteAnnotatorPrecision]
    }
    

    /*
     * Obtains params a, b for the one vs all logistic regression
     */
    def logisticRegressionParams( c: Int, frequencies: Dataset[AnnotatorFrequency], classFrequencies: Dataset[AnnotatorClassFrequency])
      : Dataset[LogisticParams] = {

        def binarizeClas(k: Int) = if(k == c) 1 else 0

        val fixFrequencies = frequencies.map(x => AnnotatorFrequency(x.annotator, binarizeClas(x.clas), binarizeClas(x.k), x.frequency))
                                        .groupByKey(x => (x.annotator, x.clas, x.k))
                                        .reduceGroups((af1, af2) => AnnotatorFrequency(af1.annotator, af1.clas, af1.k, af1.frequency + af2.frequency)) 
                                        .as[((Long, Int, Int), AnnotatorFrequency)]
                                        .map(x => x._2)
                                        .as[AnnotatorFrequency]

        val fixClasFrequencies = frequencies.map(x => AnnotatorClassFrequency(x.annotator, binarizeClas(x.clas), x.frequency))
                                            .groupByKey(x => (x.annotator, x.clas))
                                            .reduceGroups((af1, af2) => AnnotatorClassFrequency(af1.annotator, af1.clas, af1.frequency + af2.frequency)) 
                                            .as[((Long, Int), AnnotatorClassFrequency)]
                                            .map(x => x._2)
                                            .as[AnnotatorClassFrequency]
        val annTable = annotatorPrecision(fixFrequencies, fixClasFrequencies)

        //Obtains RaykarBinary Logistic Regression annotator params 
        model.annotations.joinWith(annTable, model.annotations.col("annotator") === annTable.col("annotator") &&
                                              model.annotations.col("value") === annTable.col("k"))
                         .as[(MulticlassAnnotation, DiscreteAnnotatorPrecision)]
                         .groupByKey(x => x._1.example)
                         .agg((new LogisticParamAggregator()).toColumn)
                         .as[Tuple2[Long, LogisticAnnotatorParams]]
                         .map(x => LogisticParams(x._1, x._2.a, x._2.b))
    }

    /**
     * Casting for spark Row members
     */
    def castRowMember(m: Any) = m match {
            case m: Double => m 
            case m: Int => m.toDouble
    }

    /**
     * Prepares data for MLlib gradient descent
     */
    def prepareDataLogisticGradient(logParams: Dataset[LogisticParams], mu: Dataset[BinarySoftLabel]) : 
          RDD[(Double, Vector)] = {

      val muWithParams = mu.joinWith(logParams, mu.col("example") === logParams.col("example")) 
                           .as[(BinarySoftLabel, LogisticParams)]
                           .map(x => MuWithLogisticParams(x._1.example, x._1.value, x._2.a, x._2.b))
                           .as[MuWithLogisticParams]
                           .withColumnRenamed("mu", "comenriquegrodrigotempmu")
                           .withColumnRenamed("a", "comenriquegrodrigotempa")
                           .withColumnRenamed("b", "comenriquegrodrigotempb")


      val fullData = model.dataset.join(muWithParams, "example")
      val features = fullData.columns.filter(x => (!x.startsWith("comenriquegrodrigotemp") && (x != "example")))
                                     .map(col)
      fullData.select((Array(col("comenriquegrodrigotempmu"),
                      col("comenriquegrodrigotempa"),
                      col("comenriquegrodrigotempb"),
                      col("comenriquegrodrigotempindependent")) ++
                      features):_*)
              .rdd
              .map((x: Row) => (x.getDouble(0), Vectors.dense(Array.range(1,x.size).map(i => castRowMember(x.get(i))))))
 
    }

    /**
     * Apply weights of a model to obtain the class prediction
     */
    def applyModel(weights: Array[Double]) : Dataset[LogisticPrediction] = {
      model.dataset.select((Array(col("example"),col("comenriquegrodrigotempindependent")) ++ model.dataset.columns.tail.tail.map(col)):_*) 
                   .map((r : Row) => LogisticPrediction(r.getLong(0),Functions.sigmoid(Array.range(1,r.size).map(i => castRowMember(r.get(i))).zip(weights).foldLeft(0.0)((x,y) => x + y._1 * y._2 )))) 
    }

    /**
     * Obtains the logistic regression models
     */
    def logisticRegression( frequencies: Dataset[AnnotatorFrequency], 
      classFrequencies: Dataset[AnnotatorClassFrequency] ) : (Array[Array[Double]], Dataset[LogisticMultiPrediction]) = {
      
      val predictions = Array.ofDim[Dataset[LogisticPrediction]](model.nClasses)
      val logisticWeights =  Array.ofDim[Double](model.nClasses, model.nFeatures)


      for ( c <- (0 until model.nClasses) ) {
        //Obtains RaykarBinary Logistic Regression annotator Params
        val params = logisticRegressionParams(c, frequencies, classFrequencies)

        //Prepares multiclass mu for one vs all
        val muFixed = model.mu.filter(x => x.clas == c).map(x => BinarySoftLabel(x.example, x.prob)).as[BinarySoftLabel] 

        //Prepare data for logistic gradient api
        val preparedRDD = prepareDataLogisticGradient(params, muFixed) 
       
        //Obtains optmized logistic regression
        val grad = new RaykarMultiGradient()
        val updater = new RaykarMultiUpdater(model.weightsPriorMatrix(c))
        val rand = new Random(0) //First weight estimation is random
        val initialWeights = Vectors.dense(Array.tabulate(model.nFeatures)(x => rand.nextDouble())) 
        val opt = GradientDescent.runMiniBatchSGD(preparedRDD,grad,updater,gradLearning,gradIters,0,1,initialWeights,gradThreshold)._1
        val optWeights = opt.toArray
        logisticWeights(c) = optWeights

        //Obtains predictions for class c 
        predictions(c) = applyModel(optWeights).withColumn("clas", lit(c)).as[LogisticPrediction]
      }

      //Prepare logistic prediction results and logistic weights matrix.
      val fullUnnormalizedPredictions = predictions.reduce(_ union _).as[LogisticMultiPrediction].cache()
      val normalizer = fullUnnormalizedPredictions.groupByKey(_.example).agg(sum(col("prob")).as[Double]).as[(Long,Double)].map(x => Normalizer(x._1, x._2))
      val normalizedPredictions = fullUnnormalizedPredictions
                                    .joinWith(normalizer,  fullUnnormalizedPredictions.col("example") === normalizer.col("example"))
                                    .as[(LogisticMultiPrediction, (Long, Double))]
                                    .map(x => LogisticMultiPrediction(x._1.example, x._1.clas, x._1.prob/x._2._2))
                                    .as[LogisticMultiPrediction]
                                    .cache()
      (logisticWeights, normalizedPredictions)
    }



    //Obtains annotator frequency matrix. Combines with exploded (annotator,class,class) so that all combinations are computed. 
    val annotationsWithClassProbs = model.mu.alias("mu").joinWith(model.annotations.alias("ann"), 
                                                  $"mu.example" === $"ann.example")
                                            .as[(MulticlassSoftProb, MulticlassAnnotation)]
                                            .map(x => AnnotationWithClassProb(x._1.example, x._1.clas, x._1.prob, 
                                                                              x._2.annotator, x._2.value))
                                            .as[AnnotationWithClassProb]
    val fullCombination = model.annotatorClassCombination.alias("A").joinWith(annotationsWithClassProbs.alias("B"), 
                                                                    $"A.annotator" === $"B.annotator" && 
                                                                    $"A.clas" === $"B.clas" &&
                                                                    $"A.k" === $"B.annotation", 
                                                                    "left_outer")
                                                                .as[(AnnotatorClassCombination, AnnotationWithClassProb)]
                                                                .cache()

    val freqMatrix = annotatorFrequency(fullCombination).cache()
    val freqClasMatrix = annotatorClasFrequency(fullCombination).cache()

    //Obtains annotator precision matrix
    val annotatorPrec = annotatorPrecision(freqMatrix, freqClasMatrix)

    //Obtains logistic regression via the one vs all approach
    val (logWeights, prediction) = logisticRegression( freqMatrix, freqClasMatrix ) 

    //Saving results
    model(annotatorPrecision=annotatorPrec.cache(), logisticWeights = logWeights, logisticPrediction = prediction.cache())   
  }

 /**
  *  E Step of the EM algorithm.  
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def eStep(model: RaykarMultiPartialModel): RaykarMultiPartialModel = {

    import model.dataset.sparkSession.implicits._ 

    //Obtains EStepData with logistic and annotators probabilities
    val annWithPred = model.logisticPrediction
                           .joinWith(model.annotations, model.logisticPrediction.col("example") === model.annotations.col("example")) 
                           .as[(LogisticMultiPrediction, MulticlassAnnotation)]
                           .map(x => AnnotationsWithLogisticPrediction(x._1.example, x._1.clas, x._1.prob, x._2.annotator, x._2.value))
                           .as[AnnotationsWithLogisticPrediction]
    val eStepData = annWithPred.joinWith(model.annotatorPrecision, annWithPred.col("annotator") === model.annotatorPrecision.col("annotator") &&
                                                                    annWithPred.col("clas") === model.annotatorPrecision.col("c") &&
                                                                    annWithPred.col("annotation") === model.annotatorPrecision("k"))
                               .as[(AnnotationsWithLogisticPrediction, DiscreteAnnotatorPrecision)]
                               .map(x => EStepEstimationPoint(x._1.example, x._1.clas, x._1.prediction, x._1.annotator, x._1.annotation, x._2.prob))
                               .as[EStepEstimationPoint]

    
    //Computes ground truth estimation of each example, returning a probability for each class.
    val numerator = eStepData.groupByKey(x => (x.example, x.clas))
                             .agg((new AnnotationsLikelihoodAggregator()).toColumn)
                             .as[Tuple2[Tuple2[Long,Int],Double]]
                             .map(x => MulticlassSoftProb(x._1._1, x._1._2, x._2))
                             .as[MulticlassSoftProb]
                             .cache()

    val denominator = numerator.groupByKey(_.example) 
                               .agg(sum(col("prob")).as[Double])
                               .as[(Long,Double)]
                               .map(x => Normalizer(x._1, x._2))
                               .as[Normalizer]

    val estimation = numerator.joinWith(denominator, numerator.col("example") === denominator.col("example"))
                              .as[(MulticlassSoftProb,(Long,Double))]
                              .map(x => MulticlassSoftProb(x._1.example, x._1.clas, x._1.prob/x._2._2))
                              .as[MulticlassSoftProb]
    

    //Save results
    model(mu=estimation.cache(), annotationsLikelihood=numerator)
  }

 /**
  *  Obtains the likelihood of the partial model.  
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def logLikelihood(model: RaykarMultiPartialModel): RaykarMultiPartialModel = {
    import model.dataset.sparkSession.implicits._ 

    //Takes advantage of EStep results about likelihood of the annotators and computes the data likelihood
    val likelihoodData = model.annotationsLikelihood.joinWith(model.mu, model.annotationsLikelihood.col("example") === model.mu.col("example") && 
                                                                        model.annotationsLikelihood.col("clas") === model.mu.col("clas"))  
                                                    .as[(MulticlassSoftProb, MulticlassSoftProb)]
                                                    .map(x => LikelihoodPoint(x._1.example, x._1.clas, x._2.prob, x._1.prob))
                                                    .as[LikelihoodPoint]

    val likelihood: Double = likelihoodData.map(x => x.mu * scala.math.log(x.annotationsLikelihood)).reduce(_ + _)

    //Obtains improvement in likelihood to know if more iterations are necessary
    val improvement = likelihood - model.likelihood
    

    model(likelihood=likelihood, improvement=improvement) 
  }

}

