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

  "RaykarBinary" should "obtain the expected result in the second example" in { f => 
    val spark = f.spark
    val annotationsFile = getClass.getResource("/binary-ann.parquet").getPath
    val dataFile = getClass.getResource("/binary-data.parquet").getPath
    
    import spark.implicits._

    val annotations = spark.read.parquet(annotationsFile).as[BinaryAnnotation] 
    val data = spark.read.parquet(dataFile) 
    val sc = spark.sparkContext
    val mode = RaykarBinary(data, annotations)
    val fis = mode.getMu().filter(_.example == 1).collect()(0).value
    assert(fis ===  1.0) 
  }

  it should "obtain the expected result in the sixth example" in { f => 
    val spark = f.spark
    val annotationsFile = getClass.getResource("/binary-ann.parquet").getPath
    val dataFile = getClass.getResource("/binary-data.parquet").getPath
    
    import spark.implicits._

    val annotations = spark.read.parquet(annotationsFile).as[BinaryAnnotation] 
    val data = spark.read.parquet(dataFile) 
    val sc = spark.sparkContext
    val mode = RaykarBinary(data, annotations)
    val fis = mode.getMu().filter(_.example == 5).collect()(0).value
    assert(fis ===  0.01245) 
  }

  it should "obtain the expected result in the first annotator alpha" in { f => 
    val spark = f.spark
    val annotationsFile = getClass.getResource("/binary-ann.parquet").getPath
    val dataFile = getClass.getResource("/binary-data.parquet").getPath
    
    import spark.implicits._

    val annotations = spark.read.parquet(annotationsFile).as[BinaryAnnotation] 
    val data = spark.read.parquet(dataFile) 
    val sc = spark.sparkContext
    val mode = RaykarBinary(data, annotations)
    val fis = mode.getAnnotatorPrecision()._1(0)
    assert(fis ===  0.7915) 
  }

  it should "obtain the expected result in the first annotator beta" in { f => 
    val spark = f.spark
    val annotationsFile = getClass.getResource("/binary-ann.parquet").getPath
    val dataFile = getClass.getResource("/binary-data.parquet").getPath
    
    import spark.implicits._

    val annotations = spark.read.parquet(annotationsFile).as[BinaryAnnotation] 
    val data = spark.read.parquet(dataFile) 
    val sc = spark.sparkContext
    val mode = RaykarBinary(data, annotations)
    val fis = mode.getAnnotatorPrecision()._2(0)
    assert(fis ===  0.7855) 
  }


  it should "obtain the expected result in likelihood" in { f => 
    val spark = f.spark
    val annotationsFile = getClass.getResource("/binary-ann.parquet").getPath
    val dataFile = getClass.getResource("/binary-data.parquet").getPath
    
    import spark.implicits._

    val annotations = spark.read.parquet(annotationsFile).as[BinaryAnnotation] 
    val data = spark.read.parquet(dataFile) 
    val sc = spark.sparkContext
    val mode = RaykarBinary(data, annotations)
    val fis = mode.getLogLikelihood()
    assert(fis ===  517273.5181) 
  }

  it should "obtain the expected result in model weights" in { f => 
    val spark = f.spark
    val annotationsFile = getClass.getResource("/binary-ann.parquet").getPath
    val dataFile = getClass.getResource("/binary-data.parquet").getPath
    
    import spark.implicits._

    val annotations = spark.read.parquet(annotationsFile).as[BinaryAnnotation] 
    val data = spark.read.parquet(dataFile) 
    val sc = spark.sparkContext
    val mode = RaykarBinary(data, annotations)
    val fis = mode.getModelWeights()(1)
    assert(fis ===  0.20628) 
  }













}
