import com.enriquegrodrigo.spark.crowd.methods.RaykarBinary
import com.enriquegrodrigo.spark.crowd.types._

sc.setCheckpointDir("checkpoint")

val exampleFile = "data/binary-data.parquet"
val annFile = "data/binary-ann.parquet"

val exampleData = spark.read.parquet(exampleFile)
val annData = spark.read.parquet(annFile).as[BinaryAnnotation] 

val nAnn = annData.map(_.annotator).distinct.count().toInt

val a = Array.fill[Double](nAnn,2)(2.0) //Uniform prior
val b = Array.fill[Double](nAnn,2)(2.0) //Uniform prior

//Give first annotator more confidence
a(0)(0) += 1000 
b(0)(0) += 1000 

//Give second annotator less confidence
//Annotator 1
a(1)(1) += 1000 
b(1)(1) += 1000 


//Applying the learning algorithm
val mode = RaykarBinary(exampleData, annData, a_prior=Some(a), b_prior=Some(b))

//Get MulticlassLabel with the class predictions
val pred = mode.getMu().as[BinarySoftLabel] 

//Annotator precision matrices
val annprec = mode.getAnnotatorPrecision()

//Annotator likelihood 
val like = mode.getLogLikelihood()

