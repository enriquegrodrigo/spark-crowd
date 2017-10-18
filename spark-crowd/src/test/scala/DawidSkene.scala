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
import java.io._

class DawidSkeneTest extends fixture.FlatSpec with Matchers {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

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

}
