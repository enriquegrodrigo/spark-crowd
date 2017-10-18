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

  "RaykarCont" should "obtain the expected result in the second example" in { f => 
    val spark = f.spark
    val annotationsFile = getClass.getResource("/cont-ann.parquet").getPath
    val dataFile = getClass.getResource("/cont-data.parquet").getPath
    
    import spark.implicits._

    val annotations = spark.read.parquet(annotationsFile).as[RealAnnotation] 
    val data = spark.read.parquet(dataFile) 
    val sc = spark.sparkContext
    val mode = RaykarCont(data, annotations)
    val fis = mode.getMu().filter(_.example == 1).collect()(0).value
    assert(fis ===  20.7012) 
  }

  "RaykarCont" should "obtain the expected result in the sixth example" in { f => 
    val spark = f.spark
    val annotationsFile = getClass.getResource("/cont-ann.parquet").getPath
    val dataFile = getClass.getResource("/cont-data.parquet").getPath
    
    import spark.implicits._

    val annotations = spark.read.parquet(annotationsFile).as[RealAnnotation] 
    val data = spark.read.parquet(dataFile) 
    val sc = spark.sparkContext
    val mode = RaykarCont(data, annotations)
    val fis = mode.getMu().filter(_.example == 5).collect()(0).value
    assert(fis ===  0.00845) 
  }



}
