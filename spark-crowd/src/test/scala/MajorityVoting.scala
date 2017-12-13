/*
 * MIT License 
 *
 * Copyright (c) 2017 Enrique González Rodrigo 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this 
 * software and associated documentation files (the "Software"), to deal in the Software 
 * without restriction, including without limitation the rights to use, copy, modify, 
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
 * persons to whom the Software is furnished to do so, subject to the following conditions: 
 *
 * The above copyright notice and this permission notice shall be included in all copies or 
 * substantial portions of the Software.  
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

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

  "MajorityVoting" should "obtain the expected results in the  binary file" in { f => 
    val binaryFile = getClass.getResource("/binary-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val binaryData = spark.read.parquet(binaryFile).as[BinaryAnnotation] 
    val processed = MajorityVoting.transformBinary(binaryData).collect()
    val fis = processed.filter(_.example == 0)(0)
    assert(fis.value ==  1, "First element") 
    val fis2 = processed.filter(_.example == 5)(0)
    assert(fis2.value ==  0, "Sixth element") 
  }

   it should "obtain the expected probabilities in the binary file" in { f => 
    val binaryFile = getClass.getResource("/binary-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val binaryData = spark.read.parquet(binaryFile).as[BinaryAnnotation] 
    val processed = MajorityVoting.transformSoftBinary(binaryData).collect()
    val fis = processed.filter(_.example == 2)(0)
    assert(fis.value ===  0.9, "Third element") 
    val fis2 = processed.filter(_.example == 5)(0)
    assert(fis2.value ===  0.2, "Sixth element") 
  }
    

  it should "obtain the expected result in the multiclass file" in { f => 
    val multiclassFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val multiclassData = spark.read.parquet(multiclassFile).as[MulticlassAnnotation] 
    val processed = MajorityVoting.transformMulticlass(multiclassData).collect()
    val fis = processed.filter(_.example == 0)(0)
    assert(fis.value ==  0, "First element") 
    val fis2 = processed.filter(_.example == 5)(0)
    assert(fis2.value ==  2, "Sixth element") 
    val fis3 = processed.filter(_.example == 3)(0)
    assert(fis3.value ==  1, "Fourth element") 
  }


  it should "obtain the expected probabilities multiclass file" in { f => 
    val multiclassFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val multiclassData = spark.read.parquet(multiclassFile).as[MulticlassAnnotation] 
    val processed = MajorityVoting.transformSoftMulti(multiclassData).collect()
    val fis = processed.filter(x => x.example == 5 && x.clas == 2)(0)
    assert(fis.prob ===  0.8, "Sixth element") 
    val fis2 = processed.filter(x => x.example == 3 && x.clas == 1)(0)
    assert(fis2.prob ===  0.9, "Fourth element") 
    val fis3 = processed.filter(x => x.example == 0 && x.clas == 0)(0)
    assert(fis3.prob ===  0.9, "First element") 
  }


  it should "obtain the expected result in the first element of the continuous file" in { f => 
    val contFile = getClass.getResource("/cont-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val contData = spark.read.parquet(contFile).as[RealAnnotation] 
    val processed = MajorityVoting.transformReal(contData).collect()
    val fis = processed.filter(_.example == 0)(0)
    assert(fis.value ===  21.1512, "First element") 
    val fis2 = processed.filter(_.example == 2)(0)
    assert(fis2.value ===  -2.65438, "Third element") 
  }

}
