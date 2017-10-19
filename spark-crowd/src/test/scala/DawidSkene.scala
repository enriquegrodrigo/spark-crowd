import org.apache.log4j.Logger
import org.apache.log4j.Level
import collection.mutable.Stack
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import com.enriquegrodrigo.spark.crowd.methods.DawidSkene
import com.enriquegrodrigo.spark.crowd.types._
import org.scalatest._
import org.scalatest.fixture
import org.scalactic.TolerantNumerics 
import java.io._

class DawidSkeneTest extends fixture.FlatSpec with Matchers {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(1e-4f)

  case class FixtureParam(spark: SparkSession)

  def withFixture(test: OneArgTest) = {
      val conf = new SparkConf().setAppName("Daiwd Skene test").setMaster("local[*]")
      val spark = SparkSession.builder().config(conf).getOrCreate() 
      val sc = spark.sparkContext
      try super.withFixture(test.toNoArgTest(FixtureParam(spark))) finally {
        sc.stop()
      }
  }

  "DawidSkene" should "obtain the expected result in the first example" in { f => 
    val exampleFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val exampleData = spark.read.parquet(exampleFile).as[MulticlassAnnotation] 
    val mode = DawidSkene(exampleData)
    val fis = mode.getMu().filter(_.example == 0).collect()(0)
    assert(fis.value ==  0) 
  }

  it should "obtain the expected result in the second example" in { f => 
    val exampleFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val exampleData = spark.read.parquet(exampleFile).as[MulticlassAnnotation] 
    val mode = DawidSkene(exampleData)
    val fis = mode.getMu().filter(_.example == 1).collect()(0)
    assert(fis.value ==  1) 
  }

  it should "obtain the expected result in the eleventh example" in { f => 
    val exampleFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val exampleData = spark.read.parquet(exampleFile).as[MulticlassAnnotation] 
    val mode = DawidSkene(exampleData)
    val fis = mode.getMu().filter(_.example == 10).collect()(0)
    assert(fis.value ==  0) 
  }

  it should "obtain the expected result in the thirty second example" in { f => 
    val exampleFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val exampleData = spark.read.parquet(exampleFile).as[MulticlassAnnotation] 
    val mode = DawidSkene(exampleData)
    val fis = mode.getMu().filter(_.example == 31).collect()(0)
    assert(fis.value ==  2) 
  }

  it should "obtain the expected result for the first annotator precision" in { f => 
    val exampleFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val exampleData = spark.read.parquet(exampleFile).as[MulticlassAnnotation] 
    val mode = DawidSkene(exampleData)
    val fis = mode.getAnnotatorPrecision()(0)(0)(0)
    assert(fis ===  0.6876) 
    val fis2 = mode.getAnnotatorPrecision()(0)(1)(1)
    assert(fis2 ===  0.8028) 
    val fis3 = mode.getAnnotatorPrecision()(0)(2)(2)
    assert(fis3 === 0.6866) 
  }

  it should "obtain the expected result for the likelihood" in { f => 
    val exampleFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val exampleData = spark.read.parquet(exampleFile).as[MulticlassAnnotation] 
    val mode = DawidSkene(exampleData)
    val fis = mode.getLogLikelihood()
    assert(fis ===  7265.6019) 
  }




}
