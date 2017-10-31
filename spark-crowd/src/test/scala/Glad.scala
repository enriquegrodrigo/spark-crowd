import org.apache.log4j.Logger
import org.apache.log4j.Level
import collection.mutable.Stack
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import com.enriquegrodrigo.spark.crowd.methods.Glad
import com.enriquegrodrigo.spark.crowd.types._
import org.scalatest._
import org.scalatest.fixture
import org.scalactic.TolerantNumerics 
import java.io._

class GladTest extends fixture.FlatSpec with Matchers {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  case class FixtureParam(spark: SparkSession)

  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(1e-4f)

  def withFixture(test: OneArgTest) = {
      val conf = new SparkConf().setAppName("Glad test").setMaster("local[4]")
      val spark = SparkSession.builder().config(conf).getOrCreate() 
      val sc = spark.sparkContext
      spark.sparkContext.setCheckpointDir("checkpoint")
      try super.withFixture(test.toNoArgTest(FixtureParam(spark))) finally {
        sc.stop()
      }
  }

  "Glad" should "obtain the expected result in the binary data" in { f => 
    val exampleFile = getClass.getResource("/binary-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val exampleData = spark.read.parquet(exampleFile).as[BinaryAnnotation] 
    val mode = Glad(exampleData)
    val fis = mode.getMu().filter(_.example == 0).collect()(0).value
    assert(fis ===  1.0, "First example") 
    val fis2 = mode.getMu().filter(_.example == 1).collect()(0).value 
    assert(fis2 ===  1.0, "Second example") 
    val fis3 = mode.getMu().filter(_.example == 10).collect()(0).value
    assert(fis3 ===  0.0, "Eleventh example") 
    val fis4 = mode.getMu().filter(_.example == 2).collect()(0).value
    assert(fis4 ===  1.0, "Third example") 
    val fis5 = mode.getAnnotatorPrecision()(0)
    assert(fis5 ===  35.98048, "First annotator") 
    val fis6 = mode.getLogLikelihood()
    assert(fis6 ===  1348455.108297, "LogLikelihood") 
    val fis7 = mode.getInstanceDifficulty().filter(_.example==1).collect()(0).beta
    assert(fis7 ===  8.0204976, "First example difficulty") 
  }

}
