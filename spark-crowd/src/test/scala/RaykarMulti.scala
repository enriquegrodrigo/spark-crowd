import org.apache.log4j.Logger
import org.apache.log4j.Level
import collection.mutable.Stack
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import com.enriquegrodrigo.spark.crowd.methods.RaykarMulti
import com.enriquegrodrigo.spark.crowd.types._
import org.scalatest._
import org.scalatest.fixture
import org.scalactic.TolerantNumerics 
import java.io._

class RaykarMultiTest extends fixture.FlatSpec with Matchers {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(1e-4f)

  case class FixtureParam(spark: SparkSession)

  def withFixture(test: OneArgTest) = {
      val conf = new SparkConf().setAppName("Raykar Binary test").setMaster("local[*]")
      val spark = SparkSession.builder().config(conf).getOrCreate() 
      val sc = spark.sparkContext
      spark.sparkContext.setCheckpointDir("checkpoint")
      try super.withFixture(test.toNoArgTest(FixtureParam(spark))) finally {
        sc.stop()
      }
  }

  "RaykarMulti" should "obtain the expected result on test data" in { f => 
    val spark = f.spark
    val annotationsFile = getClass.getResource("/multi-ann.parquet").getPath
    val dataFile = getClass.getResource("/multi-data.parquet").getPath
    
    import spark.implicits._

    val annotations = spark.read.parquet(annotationsFile).as[MulticlassAnnotation] 
    val data = spark.read.parquet(dataFile) 
    val sc = spark.sparkContext
    val mode = RaykarMulti(data, annotations)
    val fis = mode.getMu().filter( x => x.example == 0 && x.clas == 0 ).collect()(0).prob
    assert(fis ===  1.0, "Result on the first example") 

    val fis2 = mode.getMu().filter( x => x.example == 1 && x.clas == 1 ).collect()(0).prob
    assert(fis2 ===  0.982154, "Result on the second example") 

    val fis3 = mode.getMu().filter( x => x.example == 4 && x.clas == 2 ).collect()(0).prob
    assert(fis3 ===  1.0, "Result on the fifth example") 

    val fis4 = mode.getLogLikelihood()
    assert(fis4 ===  -7309.94532, "LogLikelihood") 

    val fis5 = mode.getAnnotatorPrecision().filter( x => x.annotator==0 && x.c==0 && x.k==0 ).collect()(0).prob
    assert(fis5 ===  0.68751, "AnnotatorPrecision") 

    val fis6 = mode.getModelWeights(0)(1)
    assert(fis6 ===  -0.06403979, "First model weight") 
  }

}
