
package com.enriquegrodrigo.spark.crowd.methods

import com.enriquegrodrigo.spark.crowd.types._

import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.broadcast.Broadcast

import scala.math.{log => mathLog}


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

  /****************************************************/
  /****************** CASE CLASSES ********************/
  /****************************************************/

  /**
  *  Partial model shared through EM iterations 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  case class DawidSkenePartialModel(dataset: Dataset[DawidSkenePartial], params: Broadcast[DawidSkeneParams], 
                                  logLikelihood: Double, improvement: Double, nClasses: Int, 
                                  nAnnotators: Long) {

    def modify(nDataset: Dataset[DawidSkenePartial] =dataset, 
        nParams: Broadcast[DawidSkeneParams] =params, 
        nLogLikelihood: Double =logLikelihood, 
        nImprovement: Double =improvement, 
        nNClasses: Int =nClasses, 
        nNAnnotators: Long =nAnnotators) = 
          new DawidSkenePartialModel(nDataset, nParams, 
                                                  nLogLikelihood, 
                                                  nImprovement, 
                                                  nNClasses, 
                                                  nNAnnotators)
  }

  /**
  * Case class storing the annotator precision and the class weights 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  case class DawidSkeneParams(pi: Array[Array[Array[Double]]], w: Array[Double])

 /**
  *  Stores examples with class estimation 
  *
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  case class DawidSkenePartial(example: Long, annotator: Long, value: Int, est: Int)

  /**
   *  Case class for saving the annotator accuracy parameters. 
   *  This class stores the probability that an {{annotator}} 
   *  would classify an example of class {{j}} as class {{l}}.
   *
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  case class PiValue(annotator: Long, j: Integer, l:Integer, pi: Double)

  /**
   *  Case class for saving class weights. For each class, it stores the estimated
   *  probability of appearance of the class
   *  
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  case class WValue(c: Integer, p: Double)

  /**
  *  Buffer for the E step aggregator for DawidSkene Method 
  *  
  *  
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  case class DawidSkeneAggregatorBuffer(aggVect: scala.collection.Seq[Double])

  /**
  *  Buffer for the LogLikelihood calculation of the DawidSkene method 
  *  
  *  
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  case class DawidSkeneLogLikelihoodAggregatorBuffer(agg: Double, predClass: Int)

  /****************************************************/
  /****************** AGGREGATORS ********************/
  /****************************************************/

  /**
  *  Aggregator for the ground truth estimation of the E step  
  *  
  *  
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  class DawidSkeneEAggregator(params: Broadcast[DawidSkeneParams], nClasses: Int) 
    extends Aggregator[DawidSkenePartial, DawidSkeneAggregatorBuffer, Int]{
  
    def zero: DawidSkeneAggregatorBuffer = DawidSkeneAggregatorBuffer(Vector.fill(nClasses)(1))
    
    def reduce(b: DawidSkeneAggregatorBuffer, a: DawidSkenePartial) : DawidSkeneAggregatorBuffer = {
      val pi = params.value.pi 
      val classCondi = Vector.range(0,nClasses).map( c => pi(a.annotator.toInt)(c)(a.value))
      val newVect = classCondi.zip(b.aggVect).map(x => x._1 * x._2)
      DawidSkeneAggregatorBuffer(newVect) 
    }
  
    def merge(b1: DawidSkeneAggregatorBuffer, b2: DawidSkeneAggregatorBuffer) : DawidSkeneAggregatorBuffer = { 
      val buf = DawidSkeneAggregatorBuffer(b1.aggVect.zip(b2.aggVect).map(x => x._1 * x._2))
      buf
    }
  
    def finish(reduction: DawidSkeneAggregatorBuffer) = {
      val result = reduction.aggVect.zipWithIndex.maxBy(x => x._1*params.value.w(x._2))._2
      result
    }
  
    def bufferEncoder: Encoder[DawidSkeneAggregatorBuffer] = Encoders.product[DawidSkeneAggregatorBuffer]
  
    def outputEncoder: Encoder[Int] = Encoders.scalaInt
  }
  
  /**
  *  Aggregator for the likelihood calculation of the DawidSkene Method 
  *  
  *  
  *  @author enrique.grodrigo
  *  @version 0.1 
  */
  class DawidSkeneLogLikelihoodAggregator(params: Broadcast[DawidSkeneParams]) 
    extends Aggregator[DawidSkenePartial, DawidSkeneLogLikelihoodAggregatorBuffer, Double]{

    def sumKey(map: Map[Int,Double], pair: (Int,Double)) = {
        val key = pair._1
        val value = pair._2
        val new_value = map.get(key) match {
          case Some(x) => x + value
          case None => value 
        }
        map.updated(key, new_value)
    }
  
    def zero: DawidSkeneLogLikelihoodAggregatorBuffer = DawidSkeneLogLikelihoodAggregatorBuffer(0, -1)
  
    def reduce(b: DawidSkeneLogLikelihoodAggregatorBuffer, a: DawidSkenePartial) : DawidSkeneLogLikelihoodAggregatorBuffer = {
      val pival = params.value.pi(a.annotator.toInt)(a.est)(a.value)
      DawidSkeneLogLikelihoodAggregatorBuffer(b.agg + mathLog(pival), a.est) 
    }
  
    def merge(b1: DawidSkeneLogLikelihoodAggregatorBuffer, b2: DawidSkeneLogLikelihoodAggregatorBuffer) : DawidSkeneLogLikelihoodAggregatorBuffer = { 
      DawidSkeneLogLikelihoodAggregatorBuffer(b1.agg + b2.agg, if (b1.predClass == -1) b2.predClass else b1.predClass) 
    }
  
    def finish(reduction: DawidSkeneLogLikelihoodAggregatorBuffer) =  {
      reduction.agg + mathLog(params.value.w(reduction.predClass))
    }
  
  
    def bufferEncoder: Encoder[DawidSkeneLogLikelihoodAggregatorBuffer] = Encoders.product[DawidSkeneLogLikelihoodAggregatorBuffer]
  
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }


  /****************************************************/
  /******************** METHODS **********************/
  /****************************************************/

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
                        l.params.value.pi, //Model parameters (pi, for the reliability matrix and w for the class weights) 
                        l.logLikelihood //Neg log-likelihood of the model
                       )
  }

  /**
   *  Applies the E Step of the EM algorithm.
   *
   *  @param model the partial DawidSkene model (DawidSkenePartialModel)
   *  @return DawidSkenePartialModel   
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  def eStep(model: DawidSkenePartialModel): DawidSkenePartialModel = {
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
   *  @param model the partial DawidSkene model (DawidSkenePartialModel)
   *  @return DawidSkenePartialModel   
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  def mStep(model: DawidSkenePartialModel): DawidSkenePartialModel = {
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
   *  @param model the partial DawidSkene model (DawidSkenePartialModel)
   *  @param i step number
   *  @return DawidSkenePartialModel   
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  def step(model: DawidSkenePartialModel, i: Int): DawidSkenePartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val m = mStep(model)
    val e = eStep(m)
    val result = logLikelihood(e)
    result
  }

  /**
   *  Log likelihood calculation. 
   *
   *  @param model the partial DawidSkene model (DawidSkenePartialModel)
   *  @return DawidSkenePartialModel   
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  def logLikelihood(model: DawidSkenePartialModel): DawidSkenePartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val aggregator = new DawidSkeneLogLikelihoodAggregator(model.params)
    val logLikelihood = model.dataset.groupByKey(_.example).agg(aggregator.toColumn).reduce((x,y) => (x._1, x._2 + y._2))._2
    model.modify(nLogLikelihood=(-logLikelihood), nImprovement=(model.logLikelihood+logLikelihood))
  }

  /**
   *  Initialization of the parameters for the algorithm. 
   *
   *  @param dataset the dataset of MulticlassAnnotation
   *  @return DawidSkenePartialModel   
   *  @author enrique.grodrigo
   *  @version 0.1 
   */
  def initialization(dataset: Dataset[MulticlassAnnotation]): DawidSkenePartialModel = {
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

