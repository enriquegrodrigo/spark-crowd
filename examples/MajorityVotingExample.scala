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

