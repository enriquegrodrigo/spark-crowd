
package com.enriquegrodrigo.spark.crowd.methods

import com.enriquegrodrigo.spark.crowd.types._
import com.enriquegrodrigo.spark.crowd.utils.Functions

import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.optimization._
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Vector,Vectors}

import scala.util.Random

/**
 *  Provides functions for transforming an annotation dataset into 
 *  a standard label dataset using the RaykarBinary algorithm 
 *
 *  This algorithm only works with [[com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation]] datasets
 *
 *  @example
 *  {{{
 *    result: RaykarBinaryModel  = RaykarBinary(dataset)
 *  }}}
 *  @author enrique.grodrigo
 *  @version 0.1 
 *  @see Raykar, Vikas C., et al. "Learning from crowds." Journal of Machine
 *  Learning Research 11.Apr (2010): 1297-1322.
 *  
 */
object RaykarBinary {

  /****************************************************/
  /****************** CASE CLASSES ********************/
  /****************************************************/

  /**
  * Case class for the RaykarBinary partial model
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  case class RaykarBinaryPartialModel(dataset: DataFrame, annotatorData: Dataset[BinaryAnnotation], 
                                      mu: Dataset[BinarySoftLabel], dataStatistics: Dataset[RaykarBinaryStatistics],
                                      params: Broadcast[RaykarBinaryParams], logLikelihood: Double, 
                                      improvement: Double, nAnnotators: Int, nFeatures: Int) {
  
    def modify(nDataset: DataFrame =dataset, 
        nAnnotatorData: Dataset[BinaryAnnotation] =annotatorData, 
        nMu: Dataset[BinarySoftLabel] =mu, 
        nDataStatistics: Dataset[RaykarBinaryStatistics] = dataStatistics, 
        nParams: Broadcast[RaykarBinaryParams] =params, 
        nLogLikelihood: Double =logLikelihood, 
        nImprovement: Double =improvement, 
        nNAnnotators: Int =nAnnotators, 
        nNFeatures: Int =nFeatures) = 
          new RaykarBinaryPartialModel(nDataset, nAnnotatorData, nMu, nDataStatistics, 
            nParams, nLogLikelihood, nImprovement, nNAnnotators, nNFeatures)
  }

  /**
  * Estimation of a y b for an example
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  case class RaykarBinaryStatistics(example: Long, a: Double, b: Double)

  /**
  * Case class for storing RaykarBinary parameters
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  case class RaykarBinaryParams(alpha: Array[Double], beta: Array[Double], w: Array[Double], 
                                    a: Array[Array[Double]], b: Array[Array[Double]], wp: Array[Array[Double]])

  /**
  * Case class that stores annotations with class probability estimation 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  case class RaykarBinaryPartial(example: Long, annotator: Int, value: Int, mu: Double)
  
  /**
  * Stores the logistic predictions 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  case class LogisticPrediction(example: Long, p: Double) 

  /**
  * Stores annotators parameters 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  case class AnnotatorParameters(example: Long, a: Double, b: Double) 

  /**
  * Stores the parameters for the label estimation
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  case class FullParameters(example: Long, p:Double, a: Double, b: Double) 

  /**
  * Stores the ground truth estimation
  * @author enrique.grodrigo
  * @version 0.1 
  */
  case class MuEstimation(example: Long, mu:Double) 

  /**
  * Stores the parameters with the estimation of the ground truth label 
  * @author enrique.grodrigo
  * @version 0.1 
  */
  case class ParameterWithEstimation(example: Long, mu:Double, a: Double, b: Double, p: Double) 

  /**
  * Stores the value of an annotator parameter 
  * @author enrique.grodrigo
  * @version 0.1 
  */
  case class ParamValue(annotator: Long, value:Double) 

  /**
  * Stores data for parameter calculation 
  * @author enrique.grodrigo
  * @version 0.1 
  */
  case class ParamCalc(annotator: Long, num: Double, denom:Double) 

  /**
  * Stores partial estimations of a and b in the statistics aggregator 
  * @author enrique.grodrigo
  * @version 0.1 
  */
  case class RaykarBinaryStatisticsAggregatorBuffer(a: Double, b: Double)

  /****************************************************/
  /****************** AGGREGATORS ********************/
  /****************************************************/

  /**
  * Aggregator for obtaining a and b estimation for each example
  *
  * @author enrique.grodrigo
  * @version 0.1 
  */
  class RaykarBinaryStatisticsAggregator(params: Broadcast[RaykarBinaryParams]) 
    extends Aggregator[RaykarBinaryPartial, RaykarBinaryStatisticsAggregatorBuffer, (Double,Double)] {

    def zero: RaykarBinaryStatisticsAggregatorBuffer = RaykarBinaryStatisticsAggregatorBuffer(1,1) //Binary
    
    def reduce(b: RaykarBinaryStatisticsAggregatorBuffer, a: RaykarBinaryPartial) : RaykarBinaryStatisticsAggregatorBuffer = {
      val alphaValue = params.value.alpha(a.annotator)
      val alphaTerm = if (a.value == 1) alphaValue else 1-alphaValue
      val betaValue = params.value.beta(a.annotator)
      val betaTerm = if (a.value == 0) betaValue else 1-betaValue 
      RaykarBinaryStatisticsAggregatorBuffer(b.a * alphaTerm, b.b * betaTerm)
    }
  
    def merge(b1: RaykarBinaryStatisticsAggregatorBuffer, b2: RaykarBinaryStatisticsAggregatorBuffer) : RaykarBinaryStatisticsAggregatorBuffer = { 
      RaykarBinaryStatisticsAggregatorBuffer(b1.a * b2.a, b1.b*b2.b)
    }
  
    def finish(reduction: RaykarBinaryStatisticsAggregatorBuffer) = {
      (reduction.a,reduction.b)
    }
  
    def bufferEncoder: Encoder[RaykarBinaryStatisticsAggregatorBuffer] = Encoders.product[RaykarBinaryStatisticsAggregatorBuffer]
  
    def outputEncoder: Encoder[(Double,Double)] = Encoders.product[(Double,Double)]
  }

  /****************************************************/
  /******************** GRADIENT **********************/
  /****************************************************/

  /**
  * Computes the logistic function for a data point 
  * @author enrique.grodrigo
  * @version 0.1 
  */
  def computeSigmoid(x: Array[Double], w: Array[Double]): Double = {
      val vectMult = x.zip(w).map{case (x,w) =>  x*w}
      Functions.sigmoid(vectMult.sum)
  }

  /**
  * Computes the negative likelihood of a point (loss)
  * @author enrique.grodrigo
  * @version 0.1 
  */
  def computePointLoss(mui: Double, pi: Double, ai: Double, bi: Double): Double = {
    val mulaipi = ai*pi
    val mulbipi = bi*(1-pi)
    -(Functions.prodlog(mui,mulaipi) + Functions.prodlog((1-mui),mulbipi))
  }  

  /**
  * Matrix multiplication (TODO: improving using libraries)
  * @author enrique.grodrigo
  * @version 0.1 
  */
  def matMult (mat: Array[Array[Double]], v: Array[Double]): Array[Double] = {
    mat.map(mv => mv.zip(v).map{ case (x,y) => x*y }.reduce(_ + _))
  }

  /**
  * Computes the gradient for the SGD algorithm 
  * @author enrique.grodrigo
  * @version 0.1 
  */
  class RaykarBinaryGradient(params: Broadcast[RaykarBinaryParams]) extends Gradient {

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
  * @author enrique.grodrigo
  * @version 0.1 
  */
  class RaykarBinaryUpdater(priors: Broadcast[RaykarBinaryParams]) extends Updater {
    def compute(weightsOld:Vector, gradient: Vector, stepSize: Double, iter: Int, regParam: Double) = {

      val regTerm = matMult(priors.value.wp, weightsOld.toArray) //Regularization with prior weights

      val stepS = stepSize/scala.math.sqrt(iter) //Atenuates step size

      //Full update
      val fullGradient = gradient.toArray.zip(regTerm).map{case (g,t) => g - t}
      val newWeights = weightsOld.toArray.zip(fullGradient).map{ case (wold,fg) => wold + stepS*fg }
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
  *  @param dataset the dataset with feature vectors.
  *  @param annDataset the dataset with the annotations.
  *  @param emIters number of iterations for the EM algorithm
  *  @param emThreshold logLikelihood variability threshold for the EM algorithm
  *  @param gradIters maximum number of iterations for the GradientDescent algorithm
  *  @param gradThreshold threshold for the log likelihood variability for the gradient descent algorithm
  *  @param gradLearning learning rate for the gradient descent algorithm 
  *  @param a_prior prior (Beta distribution hyperparameters) for the estimation
  *  of the probability that an annotator correctly classifias positive instances 
  *  @param b_prior prior (Beta distribution hyperparameters) for the estimation
  *  of the probability that an annotator correctly classifias negative instances 
  *  @param w_prior prior for the weights of the logistic regression model
  *  @return [[com.enriquegrodrigo.spark.crowd.types.RaykarBinaryModel]]
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def apply(dataset: DataFrame, annDataset: Dataset[BinaryAnnotation], eMIters: Int = 3, 
            eMThreshold: Double = 0.001,  gradIters: Int = 100, 
            gradThreshold: Double = 0.1, gradLearning: Double = 0.1, 
            a_prior: Option[Array[Array[Double]]]= None, 
            b_prior: Option[Array[Array[Double]]]= None,
            w_prior: Option[Array[Array[Double]]]= None): RaykarBinaryModel = {
    import dataset.sparkSession.implicits._
    val datasetFixed = dataset.withColumn("comenriquegrodrigotempindependent", lit(1))
    val initialModel = initialization(datasetFixed, annDataset, a_prior, b_prior, w_prior)
    val secondModel = step(gradIters, gradThreshold, gradLearning)(initialModel,0)
    val fixed = secondModel.modify(nImprovement=1)
    val l = Stream.range(1,eMIters).scanLeft(fixed)(step(gradIters, gradThreshold, gradLearning))
                                    .takeWhile( (model) => model.improvement > eMThreshold )
                                    .last
    val preparedDataset = l.mu.select($"example", $"value").distinct()
    new RaykarBinaryModel(preparedDataset.as[BinarySoftLabel], l.params.value.alpha, l.params.value.beta, l.params.value.w, l.logLikelihood)
  }
  
  /**
  *  Initialize the parameters.  
  *  First ground truth estimation is done using the majority voting algorithm
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def initialization(dataset: DataFrame, annotatorData: Dataset[BinaryAnnotation], 
                      a_prior: Option[Array[Array[Double]]], b_prior: Option[Array[Array[Double]]],
                      w_prior: Option[Array[Array[Double]]]): RaykarBinaryPartialModel = {
    val sc = dataset.sparkSession.sparkContext
    import dataset.sparkSession.implicits._
    val annCached = annotatorData.cache() 
    val datasetCached = dataset.cache() 
    val nFeatures = datasetCached.take(1)(0).length - 1 //example
    val nAnnotators = annCached.select($"annotator").distinct().count().toInt
    val ap = a_prior match {
      case Some(arr) => arr 
      case None => Array.fill(nAnnotators,2)(2.0) 
    }
    val bp = b_prior match {
      case Some(arr) => arr 
      case None => Array.fill(nAnnotators,2)(2.0) 
    }
    val wp: Array[Array[Double]] = w_prior match {
      case Some(arr) => arr 
      case None => Array.tabulate(nFeatures,nFeatures){ case (x,y) => if (x == y) (if (x==0) 1 else 0) else 0 } 
    }
    val mu = MajorityVoting.transformSoftBinary(annCached)
    val placeholderStatistics = Seq(RaykarBinaryStatistics(0,0,0)).toDS()
    RaykarBinaryPartialModel(dataset, //Training data
      annotatorData, //Annotation data 
      mu, //Ground truth estimation 
      placeholderStatistics, //Parameters a and b for each example 
      sc.broadcast(
        new RaykarBinaryParams(Array.fill(nAnnotators)(-1), //Alpha
                                Array.fill(nAnnotators)(-1), //Beta
                                Array.fill(nFeatures)(-1), //Logistic weights
                                ap, bp, wp //Alpha, beta and weight priors
        )
      ), 
      0, //Neg-loglikelihood
      0, //Improvement 
      nAnnotators.toInt, //Number of annotators 
      nFeatures.toInt //Number of features in the training data
    ) 
  }

  /**
  *  M step of the EM algorithm.  
  *  
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def mStep(model: RaykarBinaryPartialModel, gradIters: Int, gradThreshold: Double, gradLearning: Double): RaykarBinaryPartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val sc = model.dataset.sparkSession.sparkContext

    /**
     * Prepares row to be used by the SGD algorithm
     */
    def convertRowRDD(r: Row, names: Array[String]): (Double,Vector) = {
      val s: Seq[Double] = r.toSeq.map( x => 
          x match {
            case d: Double => d
            case d: Int => d.toDouble
      })

      val a_index = names.indexOf("comenriquegrodrigotempa")
      val a_val = s(a_index)
      val b_index = names.indexOf("comenriquegrodrigotempb")
      val b_val = s(b_index)
      val mu_index = names.indexOf("comenriquegrodrigotempmu")
      val mu_val = s(mu_index)
      val indep_index = names.indexOf("comenriquegrodrigotempindependent")
      val indep_val = s(indep_index)
      val featureVector = s.zip(names).filter( { case (x,name) => name != "comenriquegrodrigotempa" && 
                                                                  name != "comenriquegrodrigotempb" && 
                                                                  name != "comenriquegrodrigotempindependent" && 
                                                                  name != "comenriquegrodrigotempmu"}) 
                                      .map( { case (x,name) => x })
      val nAr = Array(a_val, b_val, indep_val) ++ featureVector
      val vect = Vectors.dense(nAr)
      val result = (mu_val, vect)
      result
    }


    //Annotations dataset with ground truth estimation
    val joinedData = model.annotatorData.joinWith(model.mu, model.annotatorData.col("example") === model.mu.col("example"))
                                        .as[(BinaryAnnotation,BinarySoftLabel)]
                                        .map(x => RaykarBinaryPartial(x._1.example, x._1.annotator.toInt, x._1.value, x._2.value))
                                        .as[RaykarBinaryPartial]
                                        .cache()

    //Annotator alpha estimation (reliability of predicting positive cases)
    val p = model.params
    val alpha = Array.ofDim[Double](model.nAnnotators.toInt)
    val denomsalpha = joinedData.groupBy(col("annotator"))
                     .agg(sum(col("mu")) as "denom")
    val numsalpha = joinedData.groupBy(col("annotator"))
                   .agg(sum(col("mu") * col("value")) as "num")
    val alphad = numsalpha.as("n").join(denomsalpha.as("d"), 
                      $"n.annotator" === $"d.annotator")
                  .select(col("n.annotator") as "annotator", col("num"), col("denom"))
                  .as[ParamCalc]
                  .map{case ParamCalc(ann,num,denom) => ParamValue(ann, (num + p.value.a(ann.toInt)(0) - 1)/(denom + p.value.a(ann.toInt).sum - 2)) }
    alphad.collect.foreach((pv: ParamValue) => alpha(pv.annotator.toInt) = pv.value)
      
    //Annotator beta estimation (reliability of predicting negative cases)
    val beta = Array.ofDim[Double](model.nAnnotators.toInt)
    val denomsbeta = joinedData.groupBy("annotator")
                     .agg(sum(lit(1) - col("mu")) as "denom")
    val numsbeta = joinedData.groupBy(col("annotator"))
                   .agg(sum((lit(1)-col("mu")) *(lit(1)-col("value"))) as "num")
    val betad = numsbeta.as("n").join(denomsbeta.as("d"), 
                      $"n.annotator" === $"d.annotator")
                  .select(col("n.annotator") as "annotator", col("num"), col("denom"))
                  .as[ParamCalc]
                  .map{case ParamCalc(ann,num,denom) => ParamValue(ann,(num + p.value.b(ann.toInt)(0) - 1)/(denom + p.value.b(ann.toInt).sum - 2))}
    betad.collect().foreach((pv: ParamValue) => beta(pv.annotator.toInt) = pv.value)

    //Saving parameters for the model and broadcasting them
    val annParam = sc.broadcast(RaykarBinaryParams(alpha=alpha, beta=beta, w=model.params.value.w, 
                                      model.params.value.a, model.params.value.b, model.params.value.wp))

    //Obtains a and b for each example
    val aggregator = new RaykarBinaryStatisticsAggregator(annParam)
    val dataStatistics = joinedData.groupByKey(_.example)
                                   .agg(aggregator.toColumn)
                                   .map(x => RaykarBinaryStatistics(x._1, x._2._1, x._2._2))
                                   
    //Renames a and b, joining with full training data.
    val statsFixed = dataStatistics.toDF().withColumnRenamed("a", "comenriquegrodrigotempa")
                                          .withColumnRenamed("b", "comenriquegrodrigotempb")
    val withPar = model.dataset.as('d).join(statsFixed, "example")  

    // Renames mu column and adds it to full data
    val withMuRenamed = model.mu.toDF().withColumnRenamed("value","comenriquegrodrigotempmu")
    val withMu = withPar.join(withMuRenamed,"example").drop("example") 
    val colNames = withMu.columns

    //Prepares data for SGT
    val d1 = withMu.map(x => convertRowRDD(x,colNames))
    val finalData = d1.as[(Double,Vector)]
    val optiData = finalData.rdd 

    //Stochastic gradient descent process
    val grad = new RaykarBinaryGradient(annParam)
    val updater = new RaykarBinaryUpdater(annParam)
    val rand = new Random(0) //First weight estimation is random
    val initialWeights = Vectors.dense(Array.tabulate(model.nFeatures)(x => rand.nextDouble())) 
    val opt = GradientDescent.runMiniBatchSGD(optiData,grad,updater,gradLearning,gradIters,0,1,initialWeights,gradThreshold)._1
    val optWeights = opt.toArray

    //Saving results in the partial model
    val param = sc.broadcast(RaykarBinaryParams(alpha=alpha, beta=beta, w=optWeights, 
                              model.params.value.a, model.params.value.b, model.params.value.wp))
    model.modify(nDataStatistics=dataStatistics.cache(), nParams=param)
  }

  /**
  *  Obtains the logistic prediction for a data point.  
  *  
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def computeP(params : Broadcast[RaykarBinaryParams])(r: Row): LogisticPrediction = {
    val w = params.value.w
    //Converts number data to double
    val s: Seq[Double] = r.toSeq.map( x => 
          x match {
            case d: Double => d
            case d: Int => d.toDouble
            case d: Long => d.toDouble
    })
    val exampleId = s.head.toLong
    val x = s.tail.toArray
    LogisticPrediction(exampleId, computeSigmoid(x,w.toArray))
  }

  /**
  *  Estimates the ground truth.
  *  
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def estimateMu(params: FullParameters): BinarySoftLabel = {
    val a: Double = params.a
    val b: Double = params.b 
    val p: Double = params.p 
    BinarySoftLabel(params.example,(a * p)/(a*p + b*(1-p)))
  }

  /**
  *  E step for the EM algorithm.
  *  
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def eStep(model: RaykarBinaryPartialModel): RaykarBinaryPartialModel = {

    import model.dataset.sparkSession.implicits._ 

    val p = model.dataset.map(computeP(model.params)).as[LogisticPrediction]

    val allParams = p.joinWith(model.dataStatistics, p.col("example") === model.dataStatistics.col("example"))
                     .as[(LogisticPrediction, RaykarBinaryStatistics)]
                     .map( x => FullParameters(x._1.example, x._1.p, x._2.a, x._2.b)) 
                     .as[FullParameters]

    val mu = allParams.map(estimateMu(_)).as[BinarySoftLabel]

    model.modify(nMu = mu)
  }

  /**
  *  Log likelihood calculation.
  *  
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def logLikelihood(model: RaykarBinaryPartialModel): RaykarBinaryPartialModel = {
    import model.dataset.sparkSession.implicits._ 

    val p = model.dataset.map(computeP(model.params)).as[LogisticPrediction]

    val allParams = p.joinWith(model.dataStatistics, p.col("example") === model.dataStatistics.col("example"))
                     .as[(LogisticPrediction, RaykarBinaryStatistics)]
                     .map( x => FullParameters(x._1.example, x._1.p, x._2.a, x._2.b) ) 
                     .as[FullParameters]
    val temp = model.mu.joinWith(allParams, model.mu.col("example") === allParams.col("example"))
                              .as[(BinarySoftLabel,FullParameters)]
                              .map(x => ParameterWithEstimation(x._1.example,x._1.value,x._2.a, x._2.b, x._2.p))
    val logLikelihood = temp.as[ParameterWithEstimation].map( { case ParameterWithEstimation(example,mu,a,b,p) => computePointLoss(mu,a,b,p) } ).reduce(_ + _)
    model.modify(nLogLikelihood=logLikelihood, nImprovement=(model.logLikelihood-logLikelihood))
  }

  /**
  *  Full EM iteration.
  *  
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def step(gradIters: Int, gradThreshold: Double, gradLearning: Double)(model: RaykarBinaryPartialModel, i: Int): RaykarBinaryPartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val m = mStep(model, gradIters, gradThreshold, gradLearning)
    val e = eStep(m)
    val result = logLikelihood(e)
    result
  }

}

