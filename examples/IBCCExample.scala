import com.enriquegrodrigo.spark.crowd.methods.IBCC
import com.enriquegrodrigo.spark.crowd.types._

sc.setCheckpointDir("checkpoint")

val annFile = "examples/data/multi-ann.parquet"

val annData = spark.read.parquet(annFile)

//Applying the learning algorithm
val mode = IBCC(annData.as[MulticlassAnnotation])

//Get MulticlassLabel with the class predictions
val pred = mode.getMu()

//Annotator precision matrices
val annprec = mode.getAnnotatorPrecision()

//Annotator precision matrices
val classPrior = mode.getClassPrior()

