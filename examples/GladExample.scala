import com.enriquegrodrigo.spark.crowd.methods.Glad
import com.enriquegrodrigo.spark.crowd.types._

sc.setCheckpointDir("checkpoint")

val annFile = "data/binary-ann.parquet"

val annData = spark.read.parquet(annFile).as[BinaryAnnotation] 

//Applying the learning algorithm
val mode = Glad(annData)

//Get MulticlassLabel with the class predictions
val pred = mode.getMu().as[BinarySoftLabel] 

//Annotator precision matrices
val annprec = mode.getAnnotatorPrecision()

//Annotator precision matrices
val annprec = mode.getInstanceDifficulty()

//Annotator likelihood 
val like = mode.getLogLikelihood()

