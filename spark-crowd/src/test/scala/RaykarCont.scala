import org.apache.log4j.Logger
import org.apache.log4j.Level
import collection.mutable.Stack
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import com.enriquegrodrigo.spark.crowd.methods.RaykarCont
import com.enriquegrodrigo.spark.crowd.types._
import org.scalatest._
import org.scalatest.fixture
import org.scalactic.TolerantNumerics 
import java.io._

class RaykarContTest extends fixture.FlatSpec with Matchers {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(1e-4f)

  case class FixtureParam(spark: SparkSession)

  def withFixture(test: OneArgTest) = {
      val conf = new SparkConf().setAppName("Raykar Binary test").setMaster("local[*]")
      val spark = SparkSession.builder().config(conf).getOrCreate() 
      val sc = spark.sparkContext
      try super.withFixture(test.toNoArgTest(FixtureParam(spark))) finally {
        sc.stop()
      }
  }

  "RaykarCont" should "obtain the expected result in the test data" in { f => 
    val spark = f.spark
    val annotationsFile = getClass.getResource("/cont-ann.parquet").getPath
    val dataFile = getClass.getResource("/cont-data.parquet").getPath
    
    import spark.implicits._

    val annotations = spark.read.parquet(annotationsFile).as[RealAnnotation] 
    val data = spark.read.parquet(dataFile) 
    val sc = spark.sparkContext
    val mode = RaykarCont(data, annotations)
    val fis = mode.getMu().filter(_.example == 1).collect()(0).value
    assert(fis ===  20.7012, "Second example") 
    val fis2 = mode.getMu().filter(_.example == 5).collect()(0).value
    assert(fis2 ===  0.00845, "Sixth example") 
    val fis3 = mode.getLogLikelihood()
    assert(fis3 ===  1.5613, "Square error") 
    val fis4 = mode.getAnnotatorPrecision().filter(_.annotator == 0).collect()(0).lambda
    assert(fis4 ===  0.2806, "First annotator precision") 
    val fis5 = mode.getModelWeights()(0)
    assert(fis5 ===  0.45575, "Model weights") 
  }

}
