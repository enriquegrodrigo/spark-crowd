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

import com.enriquegrodrigo.spark.crowd.methods.ClusteredGlad
import com.enriquegrodrigo.spark.crowd.methods.Glad
import com.enriquegrodrigo.spark.crowd.methods.DawidSkene
import com.enriquegrodrigo.spark.crowd.types._
import org.apache.spark.sql._
import org.apache.log4j.Logger
import org.apache.log4j.Level

Logger.getLogger("org").setLevel(Level.OFF)
Logger.getLogger("akka").setLevel(Level.OFF)

def time[R](block: => R): (R,Double) = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    (result,(t1-t0)/1000000000)
}

def accuracy(result: Dataset[MulticlassLabel], expected: Dataset[MulticlassLabel]) = {

  def correctExample(x: (MulticlassLabel,MulticlassLabel)): Boolean = x._1.value == x._2.value 

  import result.sparkSession.implicits._
  val joined = result.joinWith(expected, result.col("example") === expected.col("example"))
                     .as[(MulticlassLabel,MulticlassLabel)]
                     .cache()
  val correct: Double = joined.filter( correctExample(_) ).count()
  val all: Double = joined.count()
  correct/all
}

sc.setCheckpointDir("checkpoint")

/*
val fileList = Array( ("data/AdultContent/ann.parquet", "data/AdultContent/gt.parquet"),
                      ("data/Affective-Anger/ann.parquet", "data/Affective-Anger/gt.parquet"),
                      ("data/Affective-Disgust/ann.parquet", "data/Affective-Disgust/gt.parquet"),
                      ("data/Affective-Fear/ann.parquet", "data/Affective-Fear/gt.parquet"),
                      ("data/Affective-Joy/ann.parquet", "data/Affective-Joy/gt.parquet"),
                      ("data/Affective-Sadness/ann.parquet", "data/Affective-Sadness/gt.parquet"),
                      ("data/Affective-Surprise/ann.parquet", "data/Affective-Surprise/gt.parquet"),
                      ("data/Affective-Valence/ann.parquet", "data/Affective-Valence/gt.parquet"),
                      ("data/BarzanMozafari/ann.parquet", "data/BarzanMozafari/gt.parquet"),
                      ("data/Bluebirds/ann.parquet", "data/Bluebirds/gt.parquet"),
                      ("data/Copyright/ann.parquet", "data/Copyright/gt.parquet"),
                      ("data/DocumentRelevance/ann.parquet", "data/DocumentRelevance/gt.parquet"),
                      ("data/Face/ann.parquet", "data/Face/gt.parquet"),
                      ("data/FactEvaluation/ann.parquet", "data/FactEvaluation/gt.parquet"),
                      ("data/MovieSentiment/ann.parquet", "data/MovieSentiment/gt.parquet"),
                      ("data/RTE/ann.parquet", "data/RTE/gt.parquet"),
                      ("data/SpamCF/ann.parquet", "data/SpamCF/gt.parquet"),
                      ("data/Temp/ann.parquet", "data/Temp/gt.parquet"),
                      ("data/TRECRF10/ann.parquet", "data/TRECRF10/gt.parquet"),
                      ("data/WeatherCrowdscale/ann.parquet", "data/WeatherCrowdscale/gt.parquet"),
                      ("data/WeatherSentiment/ann.parquet", "data/WeatherSentiment/gt.parquet"),
                      ("data/WordSim/ann.parquet", "data/WordSim/gt.parquet"))
*/
                    
val annFile = "data/Binary50000/ann.parquet"
val gtFile = "data/Binary50000/gt.parquet"

println(annFile)
println(gtFile)

//val annFile = "data/WeatherCrowdscale/ann.parquet"
//val gtFile = "data/WeatherCrowdscale/gt.parquet"

//CGlad execution
println("************************")
println("CGlad")


val annData = spark.read.parquet(annFile).as[BinaryAnnotation] 
val gt = spark.read.parquet(gtFile).select($"example", $"value").as[MulticlassLabel] 

//Applying the learning algorithm
val (mode,t) = time(ClusteredGlad(annData, rank=3, k=10))
println(s"Time: $t")

//Get MulticlassLabel with the class predictions
val pred = mode.getMu().as[BinarySoftLabel] 
val estimation = pred.map(x => MulticlassLabel(x.example, if(x.value>0.5) 1 else 0)).as[MulticlassLabel]
val acc = accuracy(estimation, gt)
println(s"Accuracy: $acc")


//Annotator precision matrices
val annprec = mode.getAnnotatorPrecision()
println(annprec)

//Annotator precision matrices
val instanceDif = mode.getInstanceDifficulty()
instanceDif.select("beta").distinct().show()

println("************************")
println("Glad")

val annData2 = spark.read.parquet(annFile).as[BinaryAnnotation] 
val gt2 = spark.read.parquet(gtFile).select($"example", $"value").as[MulticlassLabel] 

//Applying the learning algorithm
val (mode2,t2) = time(Glad(annData2))
println(s"Time: $t2")

//Get MulticlassLabel with the class predictions
val pred2 = mode2.getMu().as[BinarySoftLabel] 
val estimation2 = pred2.map(x => MulticlassLabel(x.example, if(x.value>0.5) 1 else 0)).as[MulticlassLabel]
val acc2 = accuracy(estimation2, gt2)
println(s"Accuracy: $acc2")


//Annotator precision matrices
val annprec2 = mode2.getAnnotatorPrecision()
println(annprec2)

//Annotator precision matrices
val instanceDif2 = mode2.getInstanceDifficulty()
instanceDif2.select("beta").distinct().show()

println("************************")
println("DawidSkene")

val annData3 = spark.read.parquet(annFile).as[MulticlassAnnotation] 
val gt3 = spark.read.parquet(gtFile).select($"example", $"value").as[MulticlassLabel] 

//Applying the learning algorithm
val (mode3,t3) = time(DawidSkene(annData3))
println(s"Time: $t3")

//Get MulticlassLabel with the class predictions
val pred3 = mode3.getMu().as[MulticlassLabel] 
val estimation3 = pred3.map(x => MulticlassLabel(x.example, if(x.value>0.5) 1 else 0)).as[MulticlassLabel]
val acc3 = accuracy(estimation3, gt3)
println(s"Accuracy: $acc3")


//Annotator precision matrices
val annprec3 = mode3.getAnnotatorPrecision()
println(annprec3)


