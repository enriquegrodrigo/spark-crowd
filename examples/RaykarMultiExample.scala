import com.enriquegrodrigo.spark.crowd.methods.RaykarMulti
import com.enriquegrodrigo.spark.crowd.types._

sc.setCheckpointDir("checkpoint")

val exampleFile = "data/multi-data.parquet"
val annFile = "data/multi-ann.parquet"

val exampleData = spark.read.parquet(exampleFile)
val annData = spark.read.parquet(annFile).as[MulticlassAnnotation] 

//Applying the learning algorithm
val mode = RaykarMulti(exampleData, annData)

//Get MulticlassLabel with the class predictions
val pred = mode.getMu().as[MulticlassSoftProb] 

//Annotator precision matrices
val annprec = mode.getAnnotatorPrecision()

//Annotator likelihood 
val like = mode.getLogLikelihood()

