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
