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
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.broadcast.Broadcast

import scala.math.{log => mathLog}


/**
 *  Provides functions for transforming an annotation dataset into 
 *  a standard label dataset using the DawidSkene algorithm.
 *
 *  This algorithm only works with [[types.MulticlassAnnotation]] datasets although one
 *  can easily use it for [[types.BinaryAnnotation]] through Spark Dataset ``as`` method
 *
 *  It returns a [[types.DawidSkeneModel]] with information about the estimation of the 
 *  true class, as well as the annotator quality and the log-likelihood obtained by the model.
 *
 *  @example
 *  {{{
 *    import com.enriquegrodrigo.spark.crowd.methods.DawidSkene
 *    import com.enriquegrodrigo.spark.crowd.types._
 *    
 *    val exampleFile = "data/multi-ann.parquet"
 *    
 *    val exampleData = spark.read.parquet(exampleFile).as[MulticlassAnnotation] 
 *    
 *    //Applying the learning algorithm
 *    val mode = DawidSkene(exampleData)
 *    
 *    //Get MulticlassLabel with the class predictions
 *    val pred = mode.getMu().as[MulticlassLabel] 
 *    
 *    //Annotator precision matrices
 *    val annprec = mode.getAnnotatorPrecision()
 *    
 *    //Annotator likelihood 
 *    val like = mode.getLogLikelihood()
 *  }}}
 *  @author enrique.grodrigo
 *  @version 0.1.3
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
  *  @version 0.1.3 
  */
  private[crowd] case class DawidSkenePartialModel(dataset: Dataset[DawidSkenePartial], params: Broadcast[DawidSkeneParams], 
                                  annotatorCombinations: Dataset[AnnotatorCombination], logLikelihood: Double, improvement: Double, 
                                  nClasses: Int, nAnnotators: Long) {

    def modify(nDataset: Dataset[DawidSkenePartial] =dataset, 
        nParams: Broadcast[DawidSkeneParams] =params, 
        annotatorCombinations: Dataset[AnnotatorCombination]=annotatorCombinations, 
        nLogLikelihood: Double=logLikelihood,
        nImprovement: Double =improvement, 
        nNClasses: Int =nClasses, 
        nNAnnotators: Long =nAnnotators) = 
          new DawidSkenePartialModel(nDataset, nParams, annotatorCombinations,
                                                  nLogLikelihood, 
                                                  nImprovement, 
                                                  nNClasses, 
                                                  nNAnnotators)
  }

  /**
  * Case class storing the annotator precision and the class weights 
  *
  *  @author enrique.grodrigo
  *  @version 0.1.3
  */
  private[crowd] case class DawidSkeneParams(pi: Array[Array[Array[Double]]], w: Array[Double])

 /**
  *  Stores examples with class estimation 
  *
  *  @author enrique.grodrigo
  *  @version 0.1.3 
  */
  private[crowd] case class DawidSkenePartial(example: Long, annotator: Long, value: Int, est: Int)

  /**
   *  Case class for saving the annotator combinations (to take care of corner empty cases). 
   *
   *  @author enrique.grodrigo
   *  @version 0.1.3 
   */
  private[crowd] case class AnnotatorCombination(annotator: Long, j: Integer, l:Integer)


  /**
   *  Case class for saving the annotator accuracy parameters. 
   *  This class stores the probability that an {{annotator}} 
   *  would classify an example of class {{j}} as class {{l}}.
   *
   *  @author enrique.grodrigo
   *  @version 0.1.3 
   */
  private[crowd] case class PiValue(annotator: Long, j: Integer, l:Integer, pi: Double)

  /**
   *  Case class for saving class weights. For each class, it stores the estimated
   *  probability of appearance of the class
   *  
   *  @author enrique.grodrigo
   *  @version 0.1.3 
   */
  private[crowd] case class WValue(c: Integer, p: Double)

  /**
  *  Buffer for the E step aggregator for DawidSkene Method 
  *  
  *  
  *  @author enrique.grodrigo
  *  @version 0.1.3 
  */
  private[crowd] case class DawidSkeneAggregatorBuffer(aggVect: scala.collection.Seq[Double])

  /**
  *  Buffer for the LogLikelihood calculation of the DawidSkene method 
  *  
  *  
  *  @author enrique.grodrigo
  *  @version 0.1.3 
  */
  private[crowd] case class DawidSkeneLogLikelihoodAggregatorBuffer(agg: Double, predClass: Int)

  /****************************************************/
  /****************** AGGREGATORS ********************/
  /****************************************************/

  /**
  *  Aggregator for the ground truth estimation of the E step  
  *  
  *  
  *  @author enrique.grodrigo
  *  @version 0.1.3 
  */
  private[crowd] class DawidSkeneEAggregator(params: Broadcast[DawidSkeneParams], nClasses: Int) 
    extends Aggregator[DawidSkenePartial, DawidSkeneAggregatorBuffer, Int]{
  
    def zero: DawidSkeneAggregatorBuffer = DawidSkeneAggregatorBuffer(Vector.fill(nClasses)(1))
    
    def reduce(b: DawidSkeneAggregatorBuffer, a: DawidSkenePartial) : DawidSkeneAggregatorBuffer = {
      val pi = params.value.pi 
      //Obtains the class conditional probabilities for an annotation
      val classCondi = Vector.range(0,nClasses).map( c => pi(a.annotator.toInt)(c)(a.value))
      //Accumulates them in the buffer for the example
      val newVect = classCondi.zip(b.aggVect).map(x => x._1 * x._2)
      DawidSkeneAggregatorBuffer(newVect) 
    }
  
    def merge(b1: DawidSkeneAggregatorBuffer, b2: DawidSkeneAggregatorBuffer) : DawidSkeneAggregatorBuffer = { 
      //Accumulates through multiplications the class conditional probabilities for an example
      val buf = DawidSkeneAggregatorBuffer(b1.aggVect.zip(b2.aggVect).map(x => x._1 * x._2))
      buf
    }
  
    def finish(reduction: DawidSkeneAggregatorBuffer) = {
      //In the buffer, one has the numerator one of the terms of bayes rule (supossing annotations are 
      //independent given the class). To obtain the numerator we use the class weight (p(c) * prod p(an|c)) 
      //and then take the class that makes max the expression
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
  *  @version 0.1.3 
  */
  private[crowd] class DawidSkeneLogLikelihoodAggregator(params: Broadcast[DawidSkeneParams]) 
    extends Aggregator[DawidSkenePartial, DawidSkeneLogLikelihoodAggregatorBuffer, Double]{

    def zero: DawidSkeneLogLikelihoodAggregatorBuffer = DawidSkeneLogLikelihoodAggregatorBuffer(0, -1)
  
    def reduce(b: DawidSkeneLogLikelihoodAggregatorBuffer, a: DawidSkenePartial) : DawidSkeneLogLikelihoodAggregatorBuffer = {
      //Obtains the likelihood of an annotation and accumulates on the buffer
      val pival = params.value.pi(a.annotator.toInt)(a.est)(a.value)
      DawidSkeneLogLikelihoodAggregatorBuffer(b.agg + mathLog(pival), a.est) 
    }
  
    def merge(b1: DawidSkeneLogLikelihoodAggregatorBuffer, b2: DawidSkeneLogLikelihoodAggregatorBuffer) : DawidSkeneLogLikelihoodAggregatorBuffer = { 
      //Accumulates log-likelihood of annotations
      DawidSkeneLogLikelihoodAggregatorBuffer(b1.agg + b2.agg, if (b1.predClass == -1) b2.predClass else b1.predClass) 
    }
  
    def finish(reduction: DawidSkeneLogLikelihoodAggregatorBuffer) =  {
      //Accumulates likelihood of the example 
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
   *  @param dataset The dataset over which the algorithm will execute (spark Dataset of type [[types.MulticlassAnnotation]]
   *  @param eMIters Number of iterations for the EM algorith
   *  @param eMThreshold LogLikelihood variability threshold for the EM algorithm
   *  @return [[types.DawidSkeneModel]]
   *
   *  @author enrique.grodrigo
   *  @version 0.1.3 
   */
  def apply(dataset: Dataset[MulticlassAnnotation], eMIters: Int = 10, eMThreshold: Double = 0.001): 
      DawidSkeneModel = {
    import dataset.sparkSession.implicits._
    val initialModel = initialization(dataset)
    val secondModel = step(initialModel,0)
    val fixed = secondModel.modify(nImprovement=1)

    //EM algorithm loop (done as a lazy stream, to stop when needed)
    val l = Stream.range(1,eMIters).scanLeft(fixed)(step)
                                    .takeWhile( (model) => model.improvement > eMThreshold )
                                    .last

    //Prepares ground truth
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
   *  @version 0.1.3 
   */
  private[crowd] def eStep(model: DawidSkenePartialModel): DawidSkenePartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val aggregator = new DawidSkeneEAggregator(model.params, model.nClasses)
    //Obtains the new estimation of the ground truth for each example
    val newPrediction = model.dataset.groupByKey(_.example)
                                      .agg(aggregator.toColumn)
                                      .map(x => MulticlassLabel(x._1, x._2))
    //Adds the prediction to the annotation dataset
    val nData= model.dataset.alias("d").joinWith(newPrediction.alias("p"), $"d.example" === $"p.example")
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
   *  @version 0.1.3 
   */
  private[crowd] def mStep(model: DawidSkenePartialModel): DawidSkenePartialModel = {
    import model.dataset.sparkSession.implicits._

    val sc = model.dataset.sparkSession.sparkContext
    val data = model.dataset
    val nClasses = model.nClasses
    val nAnnotators = model.nAnnotators
    //Matrix with annotator precision
    val pi = Array.fill[Double](nAnnotators.toInt,nClasses,nClasses)(0.0) //Case where annotator never classfied as a class
    val w = Array.ofDim[Double](nClasses)

    //Estimation of annotator confusion matrices
    val denoms = data.groupBy("annotator", "est")
                     .agg(count("example") as "denom")

    data.groupBy("annotator").count().filter($"annotator" === 0)

    val numjoined = data.alias("d").join(model.annotatorCombinations.alias("c"),
                                        $"d.annotator" === $"c.annotator" &&
                                        $"d.est" === $"c.j" &&
                                        $"d.value" === $"c.l", 
                                       "right_outer")
                                  .select($"c.annotator" as "annotator", 
                                            $"c.j" as "j",
                                            $"c.l" as "l",
                                            $"d.example" as "example")

    val nums = numjoined.groupBy($"annotator", $"j", $"l")
                   .agg(sum(when($"example".isNull,0).otherwise(1)) as "num")
    val pisd= nums.as("n").join(denoms.as("d"), 
                        ($"n.annotator" === $"d.annotator") &&  
                        ($"n.j" === $"d.est"))
                  .select($"n.annotator", $"n.j" as "j", $"n.l" as "l", 
                              ((col("n.num") + 1)/(col("d.denom") + nClasses)) as "pi")
                  .as[PiValue]

    val pis = pisd.collect

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
   *  @version 0.1.3
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
   *  @param model the partial DawidSkene model (DawidSkenePartialModel)
   *  @return DawidSkenePartialModel   
   *  @author enrique.grodrigo
   *  @version 0.1.3 
   */
  private[crowd] def logLikelihood(model: DawidSkenePartialModel): DawidSkenePartialModel = {
    import model.dataset.sparkSession.implicits._ 
    val aggregator = new DawidSkeneLogLikelihoodAggregator(model.params)
    //Sums the per example log-likelihood to obtain the final one
    val logLikelihood = model.dataset.groupByKey(_.example).agg(aggregator.toColumn).reduce((x,y) => (x._1, x._2 + y._2))._2
    model.modify(nLogLikelihood=(-logLikelihood), nImprovement=(model.logLikelihood+logLikelihood))
  }

  /**
   *  Initialization of the parameters for the algorithm. 
   *
   *  @param dataset the dataset of MulticlassAnnotation
   *  @return DawidSkenePartialModel   
   *  @author enrique.grodrigo
   *  @version 0.1.3 
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

    //Annotator-Class-Class combinations 
    val combinations = dataset.map(_.annotator)
                              .distinct
                              .withColumnRenamed("value", "annotator")
                              .withColumn("j", explode(array((0 until classes).map(lit): _*)))
                              .withColumn("l", explode(array((0 until classes).map(lit): _*)))
                              .as[AnnotatorCombination]

    
    //Adds the class estimation to the annotations 
    val joinedDataset = datasetCached.alias("dc").joinWith(anns.alias("an"), $"dc.example" === $"an.example")
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
                                combinations,
                                0, //Neg Log-likelihood
                                0, //Improvement in likelihood 
                                classes, //Number of classes 
                                nAnnotators //Number of annotators
                              )  
  }
}

