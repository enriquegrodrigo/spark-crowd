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

  "DawidSkene" should "obtain the expected results in the test data" in { f => 
    val exampleFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val exampleData = spark.read.parquet(exampleFile).as[MulticlassAnnotation] 
    val mode = DawidSkene(exampleData, eMIters=1)
    val fis = mode.getMu().filter(_.example == 0).collect()(0)
    assert(fis.value ==  0, "First example") 
    val fis2 = mode.getMu().filter(_.example == 1).collect()(0)
    assert(fis2.value ==  1, "Second example") 
    val fis3 = mode.getMu().filter(_.example == 10).collect()(0)
    assert(fis3.value ==  0, "Eleventh example") 
    val fis4 = mode.getMu().filter(_.example == 31).collect()(0)
    assert(fis4.value ==  2, "Thirty second example") 
    val fis5 = mode.getAnnotatorPrecision()(0)(0)(0)
    assert(fis5 ===  0.6876, "Annotator precision 0,0,0") 
    val fis6 = mode.getAnnotatorPrecision()(0)(1)(1)
    assert(fis6 ===  0.8, "Annotator precision 0,1,1") 
    val fis7 = mode.getAnnotatorPrecision()(0)(2)(2)
    assert(fis7 === 0.690544, "Annotator precision 0,2,2") 

  }
}
