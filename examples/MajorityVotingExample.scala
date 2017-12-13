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

import com.enriquegrodrigo.spark.crowd.methods.MajorityVoting
import com.enriquegrodrigo.spark.crowd.types._

val exampleFile = "data/binary-ann.parquet"
val exampleFileMulti = "data/multi-ann.parquet"
val exampleFileCont = "data/cont-ann.parquet"

val exampleDataBinary = spark.read.parquet(exampleFile).as[BinaryAnnotation] 
val exampleDataMulti = spark.read.parquet(exampleFileMulti).as[MulticlassAnnotation] 
val exampleDataCont = spark.read.parquet(exampleFileCont).as[RealAnnotation] 

//Applying the learning algorithm
//Binary class
val muBinary = MajorityVoting.transformBinary(exampleDataBinary)
val muBinaryProb = MajorityVoting.transformSoftBinary(exampleDataBinary)
//Multiclass
val muMulticlass = MajorityVoting.transformMulticlass(exampleDataMulti)
val muMulticlassProb = MajorityVoting.transformSoftMulti(exampleDataMulti)
//Continuous case
val muCont = MajorityVoting.transformReal(exampleDataCont)

