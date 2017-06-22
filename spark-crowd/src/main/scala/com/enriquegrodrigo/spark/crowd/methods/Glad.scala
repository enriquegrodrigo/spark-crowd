
package com.enriquegrodrigo.spark.crowd.methods

import com.enriquegrodrigo.spark.crowd.types.GladPartial
import com.enriquegrodrigo.spark.crowd.types.GladPartialModel
import com.enriquegrodrigo.spark.crowd.types.GladModel
import com.enriquegrodrigo.spark.crowd.types.GladParams
import com.enriquegrodrigo.spark.crowd.types.BinarySoftLabel
import com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation
import com.enriquegrodrigo.spark.crowd.types.BinaryLabel
import com.enriquegrodrigo.spark.crowd.aggregators.GladEAggregator 
import com.enriquegrodrigo.spark.crowd.aggregators.GladLogLikelihoodAggregator 
import com.enriquegrodrigo.spark.crowd.aggregators.GladBetaAggregator 
import com.enriquegrodrigo.spark.crowd.aggregators.GladAlphaAggregator 
import com.enriquegrodrigo.spark.crowd.utils.Functions

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Column
import org.apache.spark.broadcast.Broadcast

import scala.util.Random
import scala.math.sqrt

/**
 *  Provides functions for transforming an annotation dataset into 
 *  a standard label dataset using the Glad algorithm 
 *
 *  This algorithm only works with [[com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation]] datasets
 *
 *  @example
 *  {{{
 *    result: GladModel = Glad(dataset)
 *  }}}
 *  @see Whitehill, Jacob, et al. "Whose vote should count more: Optimal
 *  integration of labels from labelers of unknown expertise." Advances in
 *  neural information processing systems. 2009.
 */
object Glad {

  /**
  * Class that storage the reliability for an annotator
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] case class GladAlphas(annotator: Long, alpha: Double)

  /**
  *  Apply the Glad Algorithm.
  *
  *  @param dataset The dataset over which the algorithm will execute.
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
  def apply(dataset: Dataset[BinaryAnnotation], eMIters: Int = 1, eMThreshold: Double = 0.001, 
            gradIters: Int = 3, gradThreshold: Double = 0.5, gradLearningRate: Double=0.1,
            alphaPrior: Double = 10, betaPrior: Double = 10): GladModel = {
    import dataset.sparkSession.implicits._
    val initialModel = initialization(dataset, alphaPrior, betaPrior)
    val secondModel = step(gradIters,gradThreshold,gradLearningRate)(initialModel,0)
    val fixed = secondModel.modify(nImprovement=1)
    val l = Stream.range(2,eMIters).scanLeft(fixed)(step(gradIters,gradThreshold,gradLearningRate))
                                    .takeWhile( (model) => model.improvement > eMThreshold )
                                    .last
    val preparedDataset = l.dataset.select($"example", $"est" as "value").distinct()
    new GladModel(preparedDataset.as[BinarySoftLabel], //Ground truth estimate
                        l.params.value, //Model parameters 
                        l.logLikelihood)
  }

  /**
  *  The E step from the EM algorithm
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  private[crowd] def eStep(model: GladPartialModel): GladPartialModel = {
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
  private[crowd] def mStep(model: GladPartialModel, maxGradIters: Int, thresholdGrad: Double, learningRate: Double): GladPartialModel = {
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
  private[crowd] def step(maxGradIters: Int, thresholdGrad: Double, learningRate: Double)(model: GladPartialModel, i: Int): GladPartialModel = {
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
  private[crowd] def logLikelihood(model: GladPartialModel): GladPartialModel = {
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
  private[crowd] def initialization(dataset: Dataset[BinaryAnnotation], alphaPrior: Double, betaPrior: Double): GladPartialModel = {
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

