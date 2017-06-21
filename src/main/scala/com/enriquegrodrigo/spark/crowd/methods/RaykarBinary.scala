
package com.enriquegrodrigo.spark.crowd.methods

import com.enriquegrodrigo.spark.crowd.types.RaykarBinaryPartial
import com.enriquegrodrigo.spark.crowd.types.RaykarBinaryPartialModel
import com.enriquegrodrigo.spark.crowd.types.RaykarBinaryModel
import com.enriquegrodrigo.spark.crowd.types.RaykarBinaryParams
import com.enriquegrodrigo.spark.crowd.types.BinarySoftLabel
import com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation
import com.enriquegrodrigo.spark.crowd.types.BinaryLabel
import com.enriquegrodrigo.spark.crowd.types.RaykarBinaryStatistics
import com.enriquegrodrigo.spark.crowd.aggregators.RaykarBinaryStatisticsAggregator 
import com.enriquegrodrigo.spark.crowd.utils.Functions

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions.{lit,sum,col}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.optimization.Gradient
import org.apache.spark.mllib.optimization.SimpleUpdater
import org.apache.spark.mllib.optimization.LogisticGradient
import org.apache.spark.mllib.optimization.Updater
import org.apache.spark.mllib.optimization.GradientDescent
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors

import scala.util.Random
import scala.math._

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

  /**
  * Stores the logistic predictions 
  */
  private[crowd] case class LogisticPrediction(example: Long, p: Double) 

  /**
  * Stores annotators parameters 
  */
  private[crowd] case class AnnotatorParameters(example: Long, a: Double, b: Double) 

  /**
  * Stores the parameters for the label estimation
  */
  private[crowd] case class FullParameters(example: Long, p:Double, a: Double, b: Double) 

  /**
  * Stores the ground truth estimation
  */
  private[crowd] case class MuEstimation(example: Long, mu:Double) 

  /**
  * Stores the parameters with the estimation of the ground truth label 
  */
  private[crowd] case class ParameterWithEstimation(example: Long, mu:Double, a: Double, b: Double, p: Double) 

  /**
  * Stores the value of an annotator parameter 
  */
  private[crowd] case class ParamValue(annotator: Long, value:Double) 

  /**
  * Stores data for parameter calculation 
  */
  private[crowd] case class ParamCalc(annotator: Long, num: Double, denom:Double) 


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
  private[crowd] class RaykarBinaryGradient(params: Broadcast[RaykarBinaryParams]) extends Gradient {

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
  private[crowd] class RaykarBinaryUpdater(priors: Broadcast[RaykarBinaryParams]) extends Updater {
    def compute(weightsOld:Vector, gradient: Vector, stepSize: Double, iter: Int, regParam: Double) = {

      val regTerm = matMult(priors.value.wp, weightsOld.toArray) //Regularization with prior weights

      val stepS = stepSize/sqrt(iter) //Atenuates step size

      //Full update
      val fullGradient = gradient.toArray.zip(regTerm).map{case (g,t) => g - t}
      val newWeights = weightsOld.toArray.zip(fullGradient).map{ case (wold,fg) => wold + stepS*fg }
      val newVector = Vectors.dense(newWeights)
      (newVector, 0) //Second parameter is not used
    }
  }

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
  def apply(dataset: DataFrame, annDataset: Dataset[BinaryAnnotation], eMIters: Int = 1, 
            eMThreshold: Double = 0.001,  gradIters: Int = 100, 
            gradThreshold: Double = 0.1, gradLearning: Double = 0.1, 
            a_prior: Option[Array[Array[Double]]]= None, 
            b_prior: Option[Array[Array[Double]]]= None,
            w_prior: Option[Array[Array[Double]]]= None): RaykarBinaryModel = {
    import dataset.sparkSession.implicits._
    val datasetFixed = dataset.withColumn("com.enriquegrodrigo.temp.independent", lit(1))
    val initialModel = initialization(datasetFixed, annDataset, a_prior, b_prior, w_prior)
    val secondModel = step(gradIters, gradThreshold, gradLearning)(initialModel,0)
    val fixed = secondModel.modify(nImprovement=1)
    val l = Stream.range(1,eMIters).scanLeft(fixed)(step(gradIters, gradThreshold, gradLearning))
                                    .takeWhile( (model) => model.improvement > eMThreshold )
                                    .last
    val preparedDataset = l.mu.select($"example", $"value").distinct()
    new RaykarBinaryModel(preparedDataset.as[BinarySoftLabel], l.params.value, l.logLikelihood)
  }
  
  /**
  *  Initialize the parameters.  
  *  First ground truth estimation is done using the majority voting algorithm
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def initialization(dataset: DataFrame, annotatorData: Dataset[BinaryAnnotation], 
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
                                Array.fill(nAnnotators)(-1), //Alpha
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
  private[crowd] def mStep(model: RaykarBinaryPartialModel, gradIters: Int, gradThreshold: Double, gradLearning: Double): RaykarBinaryPartialModel = {
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

      val a_index = names.indexOf("com.enriquegrodrigo.temp.a")
      val a_val = s(a_index)
      val b_index = names.indexOf("com.enriquegrodrigo.temp.b")
      val b_val = s(b_index)
      val mu_index = names.indexOf("com.enriquegrodrigo.temp.mu")
      val mu_val = s(mu_index)
      val indep_index = names.indexOf("com.enriquegrodrigo.temp.independent")
      val indep_val = s(indep_index)
      val featureVector = s.zip(names).filter( { case (x,name) => name != "com.enriquegrodrigo.temp.a" && 
                                                                  name != "com.enriquegrodrigo.temp.b" && 
                                                                  name != "com.enriquegrodrigo.temp.independent" && 
                                                                  name != "com.enriquegrodrigo.temp.mu"}) 
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
                      numsalpha.col("annotator") === denomsalpha.col("annotator"))
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
                      numsbeta.col("annotator") === denomsbeta.col("annotator"))
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
    val statsFixed = dataStatistics.toDF().withColumnRenamed("a", "com.enriquegrodrigo.temp.a")
                                          .withColumnRenamed("b", "com.enriquegrodrigo.temp.b")
    val withPar = model.dataset.as('d).join(statsFixed, "example")  

    // Renames mu column and adds it to full data
    val withMuRenamed = model.mu.toDF().withColumnRenamed("value","com.enriquegrodrigo.temp.mu")
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
    model.modify(nDataStatistics=dataStatistics, nParams=param)
  }

  /**
  *  Obtains the logistic prediction for a data point.  
  *  
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def computeP(params : Broadcast[RaykarBinaryParams])(r: Row): LogisticPrediction = {
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
  private[crowd] def estimateMu(params: FullParameters): BinarySoftLabel = {
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
  private[crowd] def eStep(model: RaykarBinaryPartialModel): RaykarBinaryPartialModel = {

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
  private[crowd] def logLikelihood(model: RaykarBinaryPartialModel): RaykarBinaryPartialModel = {
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
  private[crowd] def step(gradIters: Int, gradThreshold: Double, gradLearning: Double)(model: RaykarBinaryPartialModel, i: Int): RaykarBinaryPartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val m = mStep(model, gradIters, gradThreshold, gradLearning)
    val e = eStep(m)
    val result = logLikelihood(e)
    result
  }

}

