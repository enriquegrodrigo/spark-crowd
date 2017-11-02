import org.apache.log4j.Logger
import org.apache.log4j.Level
import collection.mutable.Stack
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import com.enriquegrodrigo.spark.crowd.methods.RaykarBinary
import com.enriquegrodrigo.spark.crowd.types._
import org.scalatest._
import org.scalatest.fixture
import org.scalactic.TolerantNumerics 
import java.io._

class RaykarBinaryTest extends fixture.FlatSpec with Matchers {

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

  "RaykarBinary" should "obtain the expected result in the test data" in { f => 
    val spark = f.spark
    val annotationsFile = getClass.getResource("/binary-ann.parquet").getPath
    val dataFile = getClass.getResource("/binary-data.parquet").getPath
    
    import spark.implicits._

    val annotations = spark.read.parquet(annotationsFile).as[BinaryAnnotation] 
    val data = spark.read.parquet(dataFile) 
    val sc = spark.sparkContext
    val mode = RaykarBinary(data, annotations)
    val fis = mode.getMu().filter(_.example == 1).collect()(0).value
    assert(fis ===  1.0, "Second example") 
    val fis2 = mode.getMu().filter(_.example == 5).collect()(0).value
    assert(fis2 ===  0.00068, "Sixth example") 
    val fis3 = mode.getAnnotatorPrecision()._1(0)
    assert(fis3 ===  0.8749, "First annotator alpha") 
    val fis4 = mode.getAnnotatorPrecision()._2(0)
    assert(fis4 ===  0.87752, "First annotator beta") 
    val fis5 = mode.getLogLikelihood()
    assert(fis5 ===  508152.016287, "LogLikelihood") 
    val fis6 = mode.getModelWeights()(1)
    assert(fis6 ===  0.2082767, "Model weights") 
  }


}
