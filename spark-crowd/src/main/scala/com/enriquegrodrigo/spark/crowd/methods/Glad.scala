
package com.enriquegrodrigo.spark.crowd.methods

import com.enriquegrodrigo.spark.crowd.types._
import com.enriquegrodrigo.spark.crowd.utils.Functions

import org.apache.spark.sql._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.expressions.Aggregator

import scala.util.Random
import scala.math.{sqrt, exp}

/**
 *  Provides functions for transforming an annotation dataset into 
 *  a standard label dataset using the Glad algorithm.
 *
 *  This algorithm only works with [[types.BinaryAnnotation]] datasets.
 *
 *  The algorithm returns a [[types.GladModel]], with information about 
 *  the class true label estimation, the annotator precision, the 
 *  instances difficulty and the log-likilihood of the model.
 *
 *  @example
 *  {{{
 *   import com.enriquegrodrigo.spark.crowd.methods.Glad
 *   import com.enriquegrodrigo.spark.crowd.types._
 *   
 *   sc.setCheckpointDir("checkpoint")
 *   
 *   val annFile = "data/binary-ann.parquet"
 *   
 *   val annData = spark.read.parquet(annFile).as[BinaryAnnotation] 
 *   
 *   //Applying the learning algorithm
 *   val mode = Glad(annData)
 *   
 *   //Get MulticlassLabel with the class predictions
 *   val pred = mode.getMu().as[BinarySoftLabel] 
 *   
 *   //Annotator precision matrices
 *   val annprec = mode.getAnnotatorPrecision()
 *   
 *   //Annotator precision matrices
 *   val annprec = mode.getInstanceDifficulty()
 *   
 *   //Annotator likelihood 
 *   val like = mode.getLogLikelihood()
 *  }}}
 *  @see Whitehill, Jacob, et al. "Whose vote should count more: Optimal
 *  integration of labels from labelers of unknown expertise." Advances in
 *  neural information processing systems. 2009.
 */
object Glad {

  /****************************************************/
  /****************** CASE CLASSES ********************/
  /****************************************************/

  private[spark] case class GladPartialModel(dataset: Dataset[GladPartial], params: Broadcast[GladParams], 
                                  logLikelihood: Double, improvement: Double, 
                                  nAnnotators: Int) {

  def modify(nDataset: Dataset[GladPartial] =dataset, 
      nParams: Broadcast[GladParams] =params, 
      nLogLikelihood: Double =logLikelihood, 
      nImprovement: Double =improvement, 
      nNAnnotators: Int =nAnnotators) = 
        new GladPartialModel(nDataset, nParams, 
                                                nLogLikelihood, 
                                                nImprovement, 
                                                nNAnnotators)
  }

  /**
  * Class that storage the reliability for an annotator
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] case class GladAlphas(annotator: Long, alpha: Double)

  /**
  *  Glad Annotator precision estimation 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] case class GladAnnotatorPrecision(annotator: Long, alpha: Int)

  /**
  *  Case class for storing Glad model parameters
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] case class GladParams(alpha: Array[Double], w: Array[Double])

  /**
  *  Case class for storing annotations with class estimations and beta  
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] case class GladPartial(example: Long, annotator: Int, value: Int, est: Double, beta: Double)

  /**
  *  Buffer for the alpha aggregator 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] case class GladAlphaAggregatorBuffer(agg: Double, alpha: Double)

  /**
  *  Buffer for the beta aggregator 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] case class GladBetaAggregatorBuffer(agg: Double, beta: Double)

  /**
  *  Buffer for the E step aggregator 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] case class GladEAggregatorBuffer(aggVect: scala.collection.Seq[Double])

  /**
  *  Buffer for the likelihood aggregator 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] case class GladLogLikelihoodAggregatorBuffer(agg: Double, classProb: Double)

  /****************************************************/
  /****************** AGGREGATORS ********************/
  /****************************************************/

  /**
  *  Aggregator for the precision of annotators 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] class GladAlphaAggregator(params: Broadcast[GladParams], learningRate: Double) 
    extends Aggregator[GladPartial, GladAlphaAggregatorBuffer, Double] {

    def zero: GladAlphaAggregatorBuffer = GladAlphaAggregatorBuffer(0,-1) 
    
    def reduce(b: GladAlphaAggregatorBuffer, a: GladPartial) : GladAlphaAggregatorBuffer = {
      val alpha = params.value.alpha
      val al = alpha(a.annotator)
      val bet = a.beta
      val aest = a.est
      val sigmoidValue = Functions.sigmoid(alpha(a.annotator)*a.beta)
      val p = if (a.value == 1) a.est else (1-a.est)
      val term = (p - sigmoidValue)*bet
      GladAlphaAggregatorBuffer(b.agg + term, al)
    }
  
    def merge(b1: GladAlphaAggregatorBuffer, b2: GladAlphaAggregatorBuffer) : GladAlphaAggregatorBuffer = { 
      GladAlphaAggregatorBuffer(b1.agg + b2.agg, if (b1.alpha == -1) b2.alpha else b1.alpha )
    }
  
    def finish(reduction: GladAlphaAggregatorBuffer) = {
      reduction.alpha + learningRate * reduction.agg
    }
  
    def bufferEncoder: Encoder[GladAlphaAggregatorBuffer] = Encoders.product[GladAlphaAggregatorBuffer]
  
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }

  /**
  *  Aggregator for the difficulty of each example
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] class GladBetaAggregator(params: Broadcast[GladParams], learningRate: Double) 
    extends Aggregator[GladPartial, GladBetaAggregatorBuffer, Double]{

    def zero: GladBetaAggregatorBuffer = GladBetaAggregatorBuffer(0,-1) 
    
    def reduce(b: GladBetaAggregatorBuffer, a: GladPartial) : GladBetaAggregatorBuffer = {
      val alpha = params.value.alpha
      val al = alpha(a.annotator)
      val bet = a.beta
      val aest = a.est
      val sigmoidValue = Functions.sigmoid(alpha(a.annotator)*a.beta)
      val p = if (a.value == 1) a.est else (1-a.est)
      val term = (p - sigmoidValue)*alpha(a.annotator)
      GladBetaAggregatorBuffer(b.agg + term, a.beta)
    }
  
    def merge(b1: GladBetaAggregatorBuffer, b2: GladBetaAggregatorBuffer) : GladBetaAggregatorBuffer = { 
      GladBetaAggregatorBuffer(b1.agg + b2.agg, if (b1.beta == -1) b2.beta else b1.beta)
    }
  
    def finish(reduction: GladBetaAggregatorBuffer) = {
      reduction.beta + learningRate * reduction.agg
    }
  
    def bufferEncoder: Encoder[GladBetaAggregatorBuffer] = Encoders.product[GladBetaAggregatorBuffer]
  
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }

  /**
  *  Aggregator for the E step of the EM algorithm
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] class GladEAggregator(params: Broadcast[GladParams]) 
    extends Aggregator[GladPartial, GladEAggregatorBuffer, Double]{
  
    def zero: GladEAggregatorBuffer = GladEAggregatorBuffer(Vector.fill(2)(1)) //Binary
    
    def reduce(b: GladEAggregatorBuffer, a: GladPartial) : GladEAggregatorBuffer = {
      val alpha = params.value.alpha
      val sigmoidValue = Functions.sigmoid(alpha(a.annotator)*a.beta)
      val p0 = if (a.value == 0) sigmoidValue else (1 - sigmoidValue)
      val p1 = if (a.value == 1) sigmoidValue else (1 - sigmoidValue) 
      GladEAggregatorBuffer(Vector(Functions.logLim(p0) + b.aggVect(0), Functions.logLim(p1) + b.aggVect(1)))
    }
  
    def merge(b1: GladEAggregatorBuffer, b2: GladEAggregatorBuffer) : GladEAggregatorBuffer = { 
      GladEAggregatorBuffer(b1.aggVect.zip(b2.aggVect).map(x => x._1 + x._2))
    }
  
    def finish(reduction: GladEAggregatorBuffer) = {
      val w = params.value.w
      val negative = exp(reduction.aggVect(0) + Functions.logLim(w(0)))
      val positive = exp(reduction.aggVect(1) + Functions.logLim(w(1)))
      val norm = negative + positive
      positive/norm
    }
  
    def bufferEncoder: Encoder[GladEAggregatorBuffer] = Encoders.product[GladEAggregatorBuffer]
  
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }

  /**
  *  Aggregator for the Likelihood estimation of the EM algorithm
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] class GladLogLikelihoodAggregator(params: Broadcast[GladParams]) 
    extends Aggregator[GladPartial, GladLogLikelihoodAggregatorBuffer, Double]{

    def zero: GladLogLikelihoodAggregatorBuffer = GladLogLikelihoodAggregatorBuffer(0,-1)
  
    def reduce(b: GladLogLikelihoodAggregatorBuffer, a: GladPartial) : GladLogLikelihoodAggregatorBuffer = {
      val alphaVal = params.value.alpha(a.annotator.toInt)
      val betaVal = a.beta
      val sig = Functions.sigmoid(alphaVal*betaVal) 
      val p0 = 1-a.est
      val p1 = a.est
      val k0 = if (a.value == 0) sig else 1-sig 
      val k1 = if (a.value == 1) sig else 1-sig 
      GladLogLikelihoodAggregatorBuffer(b.agg + Functions.prodlog(p0,k0) 
                                            + Functions.prodlog(p1,k1), p1) 
    }
  
    def merge(b1: GladLogLikelihoodAggregatorBuffer, b2: GladLogLikelihoodAggregatorBuffer) : GladLogLikelihoodAggregatorBuffer = { 
      GladLogLikelihoodAggregatorBuffer(b1.agg + b2.agg, if (b1.classProb == -1) b2.classProb else b1.classProb)
    }
  
    def finish(reduction: GladLogLikelihoodAggregatorBuffer) =  {
      val agg = reduction.agg
      val w0 = params.value.w(0)
      val w1 = params.value.w(1)
      val lastVal = reduction.agg + Functions.prodlog((1-reduction.classProb),params.value.w(0)) + 
                        Functions.prodlog(reduction.classProb,params.value.w(1))
      lastVal
    }
  
  
    def bufferEncoder: Encoder[GladLogLikelihoodAggregatorBuffer] = Encoders.product[GladLogLikelihoodAggregatorBuffer]
  
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }

  /****************************************************/
  /******************** METHODS **********************/
  /****************************************************/

  /**
  *  Apply the Glad Algorithm.
  *
  *  @param dataset The dataset (spark Dataset of type [[types.BinaryAnnotation]] over which the algorithm will execute.
  *  @param eMIters Number of iterations for the EM algorithm
  *  @param eMThreshold LogLikelihood variability threshold for the EM algorithm
  *  @param gradIters Maximum number of iterations for the GradientDescent algorithm
  *  @param gradThreshold Threshold for the log likelihood variability for the gradient descent algorithm
  *  @param gradLearningRate Learning rate for the gradient descent algorithm 
  *  @param alphaPrior First value for all alpha parameters 
  *  @param betaPrior First value for all beta parameters 
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  def apply(dataset: Dataset[BinaryAnnotation], eMIters: Int = 5, eMThreshold: Double = 0.1, 
            gradIters: Int = 30, gradThreshold: Double = 0.5, gradLearningRate: Double=0.01,
            alphaPrior: Double = 1, betaPrior: Double = 10): GladModel = {
    import dataset.sparkSession.implicits._
    val initialModel = initialization(dataset, alphaPrior, betaPrior)
    val secondModel = step(gradIters,gradThreshold,gradLearningRate)(initialModel,0)
    val fixed = secondModel.modify(nImprovement=1)
    val l = Stream.range(2,eMIters).scanLeft(fixed)(step(gradIters,gradThreshold,gradLearningRate))
                                    .takeWhile( (model) => model.improvement > eMThreshold )
                                    .last
    val preparedDataset = l.dataset.select($"example", $"est" as "value").distinct()
    val difficulties = l.dataset.select($"example", $"beta").as[GladInstanceDifficulty].distinct
    new GladModel(preparedDataset.as[BinarySoftLabel], //Ground truth estimate
                        l.params.value.alpha, //Model parameters 
                        difficulties, //Difficulty for each example 
                        l.logLikelihood)
  }

  /**
  *  The E step from the EM algorithm
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] def eStep(model: GladPartialModel): GladPartialModel = {
    import model.dataset.sparkSession.implicits._ 

    //Label estimation
    val aggregator = new GladEAggregator(model.params)
    val newPrediction = model.dataset.groupByKey(_.example)
                                      .agg(aggregator.toColumn)
                                      .map(x => BinarySoftLabel(x._1, x._2))
    //Add it to the annotation dataset
    val nData= model.dataset.joinWith(newPrediction, model.dataset.col("example") === newPrediction.col("example"))
                               .as[(GladPartial,BinarySoftLabel)]
                               .map(x => GladPartial(x._1.example, x._1.annotator, x._1.value, x._2.value, x._1.beta))
                               .as[GladPartial]
                               .cache()

    model.modify(nDataset=nData)
  }
  
  /**
  *  The M step from the EM algorithm
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] def mStep(model: GladPartialModel, maxGradIters: Int, thresholdGrad: Double, learningRate: Double): GladPartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val sc = model.dataset.sparkSession.sparkContext

    /**
    * Take the old model and update the instance difficulty estimation. 
    * Uses a atenuation of the learning rate to facilitate convergence of the parameters.
    */
    def updateBeta(oldModel: GladPartialModel, iter: Int): GladPartialModel = {

      //New beta estimation
      val betaAgg = new GladBetaAggregator(oldModel.params,learningRate/sqrt(iter))
      val newBeta = oldModel.dataset.groupByKey(_.example)
                                    .agg(betaAgg.toColumn)
                                    .map(x => BinarySoftLabel(x._1,x._2))

      //Add the estimation to the annotation dataset
      val dataset = oldModel.dataset.joinWith(newBeta, oldModel.dataset.col("example") === newBeta.col("example"))
                                    .map(x => GladPartial(x._1.example,x._1.annotator,x._1.value, x._1.est, x._2.value))
                                    .as[GladPartial]
                                    .cache()
      //Checkpointing every 10th iteration of the Gradient descent algorithm
      oldModel.modify(nDataset = if (iter%10 == 0) dataset.checkpoint() else dataset )
    }

    
    /**
    * Annotator reliability estimation. 
    * Uses a atenuation of the learning rate to facilitate convergence of the parameters.
    */
    def obtainAlpha(oldModel: GladPartialModel, iter: Int): Array[Double] = {
      val oldAlpha = oldModel.params.value.alpha
      val alphaAgg = new GladAlphaAggregator(oldModel.params,learningRate/sqrt(iter))
      val alpha = Array.ofDim[Double](oldModel.nAnnotators)
      val alphasDataset =  oldModel.dataset.groupByKey(_.annotator)
                                           .agg(alphaAgg.toColumn)
                                           .map(x => GladAlphas(x._1, x._2))
                                           .as[GladAlphas]

      //Creates a local array with the reliability values
      alphasDataset.collect.foreach{case GladAlphas(annotator,al) => alpha(annotator.toInt) = al}
      alpha 
    }

    /**
    * Class weight estimation. 
    */
    def obtainWeights(oldModel: GladPartialModel): Array[Double] = {
      val est = oldModel.dataset.select($"example",$"est" as "value").distinct.as[BinarySoftLabel].cache()
      val n: Double= est.count
      val s: Double= est.map(_.value).reduce(_ + _)
      val p1 = s/n
      val p0 = 1-p1
      est.unpersist()
      Array(p0,p1)
    }

    /**
    * Broadcast parameters and make the data ready for the next gradient iteration. 
    */
    def mergeUpdates(updatedBeta: GladPartialModel, nAlpha: Array[Double],  nWeights: Array[Double]) = {
      val sc = model.dataset.sparkSession.sparkContext
      val params = sc.broadcast(GladParams(nAlpha, nWeights))
      updatedBeta.modify(nParams=params)
    }

    /**
    * Gradient descent iteration 
    */
    def update(oldModel: GladPartialModel, iter: Int): (GladPartialModel,Double) = {
      val nAlpha =  obtainAlpha(oldModel, iter) 
      val nWeights = obtainWeights(oldModel) 
      val updatedBeta = updateBeta(oldModel, iter) 
      val nModel = mergeUpdates(updatedBeta,nAlpha,nWeights) 
      val likeModel = logLikelihood(nModel) 
      val improvement = oldModel.logLikelihood - likeModel.logLikelihood 
      (nModel.modify(nLogLikelihood = likeModel.logLikelihood),improvement) 
    }

    val newModel: GladPartialModel = update(model,1)._1
    //Gradient loop
    val lastModel = Stream.range(2,maxGradIters).scanLeft((newModel,1.0))((x,i) => update(x._1, i))
                                    .takeWhile( (model) => model._2 > thresholdGrad )
                                    .last
    lastModel._1
  }

  /**
  *  Full step of the EM algorithm 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] def step(maxGradIters: Int, thresholdGrad: Double, learningRate: Double)(model: GladPartialModel, i: Int): GladPartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val m = mStep(model,maxGradIters,thresholdGrad,learningRate)
    val e = eStep(m)
    val result = logLikelihood(e)
    result.modify(nDataset = result.dataset.checkpoint)
  }

  /**
  *  Neg-logLikelihood calculation 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] def logLikelihood(model: GladPartialModel): GladPartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val aggregator = new GladLogLikelihoodAggregator(model.params)
    val logLikelihood = model.dataset.groupByKey(_.example).agg(aggregator.toColumn).reduce((x,y) => (x._1, x._2 + y._2))._2
    model.modify(nLogLikelihood=(-logLikelihood), nImprovement=(model.logLikelihood + logLikelihood ))
  }

  /**
  *  Initialization of the parameters of the algorithm. 
  *  The first label estimation is done using majority voting
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[spark] def initialization(dataset: Dataset[BinaryAnnotation], alphaPrior: Double, betaPrior: Double): GladPartialModel = {
    val sc = dataset.sparkSession.sparkContext
    import dataset.sparkSession.implicits._
    val datasetCached = dataset.cache() 
    val nAnnotators = datasetCached.select($"annotator").distinct().count()
    val anns = MajorityVoting.transformSoftBinary(datasetCached)
    val joinedDataset = datasetCached.joinWith(anns, datasetCached.col("example") === anns.col("example"))
                               .as[(BinaryAnnotation,BinarySoftLabel)]
                               .map(x => GladPartial(x._1.example, x._1.annotator.toInt, x._1.value, x._2.value, betaPrior))
                               .as[GladPartial]
    val partialDataset = joinedDataset
                                .select($"example", $"annotator", $"value", $"est", $"beta")
                                .as[GladPartial]
                                .cache()
    datasetCached.unpersist()
    val r = new Random(0)
    val alphaInit = Array.fill(nAnnotators.toInt)(alphaPrior)
    new GladPartialModel(partialDataset, //Annotation dataset 
                                sc.broadcast(new GladParams(alphaInit, //Reliability of annotators (placeholder)
                                  Array.fill(2)(10)) //Class weights (placeholder) 
                                ), 
                                0, //Neg-loglikelihood
                                0, //Neg-loglikelihood 
                                nAnnotators.toInt) //Number of annotators 
  }
}

