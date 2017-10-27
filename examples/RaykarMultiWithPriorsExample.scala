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

//Annotator likelihood 
val like = mode.getLogLikelihood()

