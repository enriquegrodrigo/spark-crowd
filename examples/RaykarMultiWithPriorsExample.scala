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

import com.enriquegrodrigo.spark.crowd.methods.RaykarMulti
import com.enriquegrodrigo.spark.crowd.types._

sc.setCheckpointDir("checkpoint")

val exampleFile = "data/multi-data.parquet"
val annFile = "data/multi-ann.parquet"

val exampleData = spark.read.parquet(exampleFile)
val annData = spark.read.parquet(annFile).as[MulticlassAnnotation] 

val nAnn = annData.map(_.annotator).distinct.count().toInt
val nClasses = annData.map(_.value).distinct.count().toInt

val annPriors = Array.fill[Double](nAnn, nClasses, nClasses)(2)

//Add more confidence to the first annotator in all classes
annPriors(0) = Array.tabulate[Double](nClasses, nClasses){case (i,j) => if (i==j) 1000 else 2}

//Add more less confidence to the second annotator in all classes
annPriors(1) = Array.tabulate[Double](nClasses, nClasses){case (i,j) => if (i!=j) 1000 else 2}

//Applying the learning algorithm
val mode = RaykarMulti(exampleData, annData, k_prior=Some(annPriors))

//Get MulticlassLabel with the class predictions
val pred = mode.getMu().as[MulticlassSoftProb] 

//Annotator precision matrices
val annprec = mode.getAnnotatorPrecision()


