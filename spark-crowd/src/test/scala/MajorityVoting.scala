import org.apache.log4j.Logger
import org.apache.log4j.Level
import collection.mutable.Stack
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import com.enriquegrodrigo.spark.crowd.methods.MajorityVoting
import com.enriquegrodrigo.spark.crowd.types._
import org.scalatest._
import org.scalactic.TolerantNumerics 
import java.io._

class MajorityVotingTest extends fixture.FlatSpec with Matchers {


  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(1e-4f)

  case class FixtureParam(spark: SparkSession)

  def withFixture(test: OneArgTest) = {
      val conf = new SparkConf().setAppName("Majority voting test").setMaster("local")
      val spark = SparkSession.builder().config(conf).getOrCreate() 
      val sc = spark.sparkContext
      try super.withFixture(test.toNoArgTest(FixtureParam(spark))) finally {
        sc.stop()
      }
  }

  "MajorityVoting" should "obtain the expected result in the first element of the  binary file" in { f => 
    val binaryFile = getClass.getResource("/binary-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val binaryData = spark.read.parquet(binaryFile).as[BinaryAnnotation] 
    val processed = MajorityVoting.transformBinary(binaryData).collect()
    val fis = processed.filter(_.example == 0)(0)
    assert(fis.value ==  1) 
  }

  it should "obtain the expected result in the sixth element of the  binary file" in { f => 
    val binaryFile = getClass.getResource("/binary-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val binaryData = spark.read.parquet(binaryFile).as[BinaryAnnotation] 
    val processed = MajorityVoting.transformBinary(binaryData).collect()
    val fis = processed.filter(_.example == 5)(0)
    assert(fis.value ==  0) 
  }

   it should "obtain the expected probability in the third element of the binary file" in { f => 
    val binaryFile = getClass.getResource("/binary-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val binaryData = spark.read.parquet(binaryFile).as[BinaryAnnotation] 
    val processed = MajorityVoting.transformSoftBinary(binaryData).collect()
    val fis = processed.filter(_.example == 2)(0)
    assert(fis.value ===  0.9) 
  }
    
  it should "obtain the expected probability in the sixth element of the binary file" in { f => 
    val binaryFile = getClass.getResource("/binary-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val binaryData = spark.read.parquet(binaryFile).as[BinaryAnnotation] 
    val processed = MajorityVoting.transformSoftBinary(binaryData).collect()
    val fis = processed.filter(_.example == 5)(0)
    assert(fis.value ===  0.2) 
  }
  

  it should "obtain the expected result in the first element of the multiclass file" in { f => 
    val multiclassFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val multiclassData = spark.read.parquet(multiclassFile).as[MulticlassAnnotation] 
    val processed = MajorityVoting.transformMulticlass(multiclassData).collect()
    val fis = processed.filter(_.example == 0)(0)
    assert(fis.value ==  0) 
  }

  it should "obtain the expected result in the sixth element of the multiclass file" in { f => 
    val multiclassFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val multiclassData = spark.read.parquet(multiclassFile).as[MulticlassAnnotation] 
    val processed = MajorityVoting.transformMulticlass(multiclassData).collect()
    val fis = processed.filter(_.example == 5)(0)
    assert(fis.value ==  2) 
  }

  it should "obtain the expected result in the fourth element of the multiclass file" in { f => 
    val multiclassFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val multiclassData = spark.read.parquet(multiclassFile).as[MulticlassAnnotation] 
    val processed = MajorityVoting.transformMulticlass(multiclassData).collect()
    val fis = processed.filter(_.example == 3)(0)
    assert(fis.value ==  1) 
  }

  it should "obtain the expected result in the first element of the continuous file" in { f => 
    val contFile = getClass.getResource("/cont-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val contData = spark.read.parquet(contFile).as[RealAnnotation] 
    val processed = MajorityVoting.transformReal(contData).collect()
    val fis = processed.filter(_.example == 0)(0)
    assert(fis.value ===  21.1512) 
  }

  it should "obtain the expected result in the third element of the continuous file" in { f => 
    val contFile = getClass.getResource("/cont-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val contData = spark.read.parquet(contFile).as[RealAnnotation] 
    val processed = MajorityVoting.transformReal(contData).collect()
    val fis = processed.filter(_.example == 2)(0)
    assert(fis.value ===  -2.65438) 
  }


}
