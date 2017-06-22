
package com.enriquegrodrigo.spark.crowd.methods

import com.enriquegrodrigo.spark.crowd.types.DawidSkenePartial
import com.enriquegrodrigo.spark.crowd.types.DawidSkenePartialModel
import com.enriquegrodrigo.spark.crowd.types.DawidSkeneModel
import com.enriquegrodrigo.spark.crowd.types.DawidSkeneParams
import com.enriquegrodrigo.spark.crowd.types.MulticlassAnnotation
import com.enriquegrodrigo.spark.crowd.types.MulticlassLabel
import com.enriquegrodrigo.spark.crowd.aggregators.DawidSkeneEAggregator 
import com.enriquegrodrigo.spark.crowd.aggregators.DawidSkeneLogLikelihoodAggregator 

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions._
import org.apache.spark.broadcast.Broadcast

/**
 *  Provides functions for transforming an annotation dataset into 
 *  a standard label dataset using the DawidSkene algorithm 
 *
 *  This algorithm only works with [[com.enriquegrodrigo.spark.crowd.types.MulticlassAnnotation]] datasets
 *
 *  @example
 *  {{{
 *    result: DawidSkeneModel = DawidSkene(dataset)
 *  }}}
 *  @author enrique.grodrigo
 *  @version 0.1 
 *  @see Dawid, Alexander Philip, and Allan M. Skene. "Maximum likelihood
 *  estimation of observer error-rates using the EM algorithm." Applied
 *  statistics (1979): 20-28.
 */
object DawidSkene {

  /**
   *  Case class for saving the annotator accuracy parameters. 
   *  This class stores the probability that an {{annotator}} 
   *  would classify an example of class {{j}} as class {{l}}.
   *
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private case class PiValue(annotator: Long, j: Integer, l:Integer, pi: Double)

  /**
   *  Case class for saving class weights. For each class, it stores the estimated
   *  probability of appearance of the class
   *  
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private case class WValue(c: Integer, p: Double)

  /**
   *  Applies learning algorithm.
   *
   *  @param dataset The dataset over which the algorithm will execute
   *  @param eMIters Number of iterations for the EM algorithm
   *  @param eMThreshold LogLikelihood variability threshold for the EM algorithm
   *  @return [[com.enriquegrodrigo.spark.crowd.types.DawidSkeneModel]]
   *
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  def apply(dataset: Dataset[MulticlassAnnotation], eMIters: Int = 10, eMThreshold: Double = 0.001): 
      DawidSkeneModel = {
    import dataset.sparkSession.implicits._
    val initialModel = initialization(dataset)
    val secondModel = step(initialModel,0)
    val fixed = secondModel.modify(nImprovement=1)

    //EM algorithm loop
    val l = Stream.range(1,eMIters).scanLeft(fixed)(step)
                                    .takeWhile( (model) => model.improvement > eMThreshold )
                                    .last

    val preparedDataset = l.dataset.select($"example", $"est" as "value").distinct() //Ground truth

    new DawidSkeneModel(preparedDataset.as[MulticlassLabel], //Ground truth
                        l.params.value, //Model parameters (pi, for the reliability matrix and w for the class weights) 
                        l.logLikelihood //Neg log-likelihood of the model
                       )
  }

  /**
   *  Applies the E Step of the EM algorithm.
   *
   *  @param model the partial DawidSkene model ([com.enriquegrodrigo.spark.crowd.types.DawidSkenePartialModel])
   *  @return [[com.enriquegrodrigo.spark.crowd.types.DawidSkenePartialModel]]   
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] def eStep(model: DawidSkenePartialModel): DawidSkenePartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val aggregator = new DawidSkeneEAggregator(model.params, model.nClasses)
    //Obtains the new estimation of the ground truth for each example
    val newPrediction = model.dataset.groupByKey(_.example)
                                      .agg(aggregator.toColumn)
                                      .map(x => MulticlassLabel(x._1, x._2))
    //Adds the prediction to the annotation dataset
    val nData= model.dataset.joinWith(newPrediction, model.dataset.col("example") === newPrediction.col("example"))
                               .as[(MulticlassAnnotation,MulticlassLabel)]
                               .map(x => DawidSkenePartial(x._1.example, x._1.annotator, x._1.value, x._2.value))
                               .as[DawidSkenePartial]
                               .cache()
    model.modify(nDataset=nData)
  }

  /**
   *  Applies the M Step of the EM algorithm.
   *
   *  @param model the partial DawidSkene model ([com.enriquegrodrigo.spark.crowd.types.DawidSkenePartialModel])
   *  @return [[com.enriquegrodrigo.spark.crowd.types.DawidSkenePartialModel]]   
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] def mStep(model: DawidSkenePartialModel): DawidSkenePartialModel = {
    import model.dataset.sparkSession.implicits._

    val sc = model.dataset.sparkSession.sparkContext
    val data = model.dataset
    val nClasses = model.nClasses
    val nAnnotators = model.nAnnotators
    val pi = Array.ofDim[Double](nAnnotators.toInt,nClasses,nClasses)
    val w = Array.ofDim[Double](nClasses)

    //Estimation of annotator confusión matrices
    val denoms = data.groupBy("annotator", "est")
                     .agg(count("example") as "denom")
    val nums = data.groupBy(col("annotator"), col("est"), col("value"))
                   .agg(count("example") as "num")
    val pis = nums.as("n").join(denoms.as("d"), 
                      nums.col("annotator") === denoms.col("annotator") &&  
                        nums.col("est") === denoms.col("est"))
        .select(col("n.annotator"), col("n.est") as "j", col("n.value") as "l", 
                    (col("num") + 1)/(col("denom") + nClasses) as "pi")
        .as[PiValue]
        .collect

    //Assigns the value to the annotator matrix (not distributed) Size: O(A*C^2)
    pis.foreach((pv: PiValue) => pi(pv.annotator.toInt)(pv.j)(pv.l) = pv.pi)

    //Estimation of prior class probabilities (class estimation changes on each iteration)
    
    //Denominator (number of examples)
    val nExam = data.agg(countDistinct("example")).collect()(0).getLong(0)

    /** 
     *  For each class we obtain the number of distinct examples and divide it by the 
     *  total number of examples in the dataset.
     */
    val ws = data.groupBy(col("est") as "c").agg((countDistinct("example")/nExam) as "p")
                 .as[WValue]
                 .collect

    //Assigns the value to the weights vector (not distributed) Size: O(C)
    ws.foreach{ case WValue(c,p) => w(c) = p}

    val params = sc.broadcast(DawidSkeneParams(pi, w))
    
    model.modify(nParams=params)
  }

  /**
   *  A full EM algorithm step with negative log-likelihood calculation.
   *
   *  @param model the partial DawidSkene model ([com.enriquegrodrigo.spark.crowd.types.DawidSkenePartialModel])
   *  @param i step number
   *  @return [[com.enriquegrodrigo.spark.crowd.types.DawidSkenePartialModel]]   
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] def step(model: DawidSkenePartialModel, i: Int): DawidSkenePartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val m = mStep(model)
    val e = eStep(m)
    val result = logLikelihood(e)
    result
  }

  /**
   *  Log likelihood calculation. 
   *
   *  @param model the partial DawidSkene model ([com.enriquegrodrigo.spark.crowd.types.DawidSkenePartialModel])
   *  @return [[com.enriquegrodrigo.spark.crowd.types.DawidSkenePartialModel]]   
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] def logLikelihood(model: DawidSkenePartialModel): DawidSkenePartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val aggregator = new DawidSkeneLogLikelihoodAggregator(model.params)
    val logLikelihood = model.dataset.groupByKey(_.example).agg(aggregator.toColumn).reduce((x,y) => (x._1, x._2 + y._2))._2
    model.modify(nLogLikelihood=(-logLikelihood), nImprovement=(model.logLikelihood+logLikelihood))
  }

  /**
   *  Initialization of the parameters for the algorithm. 
   *
   *  @param dataset the dataset of [[com.enriquegrodrigo.spark.crowd.types.MulticlassAnnotation]]
   *  @return [[com.enriquegrodrigo.spark.crowd.types.DawidSkenePartialModel]]   
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  private[crowd] def initialization(dataset: Dataset[MulticlassAnnotation]): DawidSkenePartialModel = {
    val sc = dataset.sparkSession.sparkContext
    import dataset.sparkSession.implicits._
    val datasetCached = dataset.cache() 

    //Number of classes
    val classes: Int = datasetCached.select($"value").distinct().count().toInt

    //Number of annotators
    val nAnnotators = datasetCached.select($"annotator").distinct().count()

    //First estimation using majority voting 
    val anns = MajorityVoting.transformMulticlass(datasetCached)
    
    //Adds the class estimation to the annotations 
    val joinedDataset = datasetCached.joinWith(anns, datasetCached.col("example") === anns.col("example"))
                               .as[(MulticlassAnnotation,MulticlassLabel)]
                               .map(x => DawidSkenePartial(x._1.example, x._1.annotator, x._1.value, x._2.value))
                               .as[DawidSkenePartial]
    val partialDataset = joinedDataset
                                .select($"example", $"annotator", $"value", $"est")
                                .as[DawidSkenePartial]
                                .cache()

    new DawidSkenePartialModel(partialDataset, 
                                sc.broadcast(
                                  new DawidSkeneParams(Array.ofDim[Double](nAnnotators.toInt, classes, classes), //Reliability Matrix for annotators
                                  Array.ofDim[Double](classes)) //Class weights
                                ),
                                0, //Neg Log-likelihood
                                0, //Improvement in likelihood 
                                classes, //Number of classes 
                                nAnnotators //Number of annotators
                              )  
  }
}

