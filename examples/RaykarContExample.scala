import com.enriquegrodrigo.spark.crowd.methods.RaykarCont
import com.enriquegrodrigo.spark.crowd.types._

sc.setCheckpointDir("checkpoint")

val exampleFile = "data/cont-data.parquet"
val annFile = "data/cont-ann.parquet"

val exampleData = spark.read.parquet(exampleFile)
val annData = spark.read.parquet(annFile).as[RealAnnotation] 

//Applying the learning algorithm
val mode = RaykarCont(exampleData, annData)

//Get MulticlassLabel with the class predictions
val pred = mode.getMu().as[RealLabel] 

//Annotator precision matrices
val annprec = mode.getAnnotatorPrecision()

//Annotator likelihood 
val like = mode.getLogLikelihood()

