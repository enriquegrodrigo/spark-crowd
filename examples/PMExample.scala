import com.enriquegrodrigo.spark.crowd.methods.PM
import com.enriquegrodrigo.spark.crowd.types._

sc.setCheckpointDir("checkpoint")

val annFile = "examples/data/cont-ann.parquet"

val annData = spark.read.parquet(annFile).as[RealAnnotation]

//Applying the learning algorithm
val mode = PM(annData)

//Get MulticlassLabel with the class predictions
val pred = mode.getMu()

//Annotator precision matrices
val annprec = mode.getAnnotatorWeights()


