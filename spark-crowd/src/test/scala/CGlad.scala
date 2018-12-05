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
import com.enriquegrodrigo.spark.crowd.methods.CGlad
import com.enriquegrodrigo.spark.crowd.types._
import org.scalatest._
import org.scalatest.fixture
import org.scalactic.TolerantNumerics 
import java.io._

class CGladTest extends fixture.FlatSpec with Matchers {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  case class FixtureParam(spark: SparkSession)

  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(1e-4f)

  def withFixture(test: OneArgTest) = {
      val conf = new SparkConf().setAppName("CGlad test").setMaster("local[4]")
      val spark = SparkSession.builder().config(conf).getOrCreate() 
      val sc = spark.sparkContext
      spark.sparkContext.setCheckpointDir("checkpoint")
      try super.withFixture(test.toNoArgTest(FixtureParam(spark))) finally {
        sc.stop()
      }
  }

  "CGlad" should "obtain the expected result in the binary data" in { f => 
    val exampleFile = getClass.getResource("/binary-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val exampleData = spark.read.parquet(exampleFile).as[BinaryAnnotation] 
    val mode = CGlad(exampleData, eMIters=3, gradIters=5)
    val fis = mode.getMu().filter(_.example == 0).collect()(0).value
    assert(fis ===  0.925586, "First example") 
    val fis2 = mode.getMu().filter(_.example == 1).collect()(0).value 
    assert(fis2 ===  0.925586, "Second example") 
    val fis3 = mode.getMu().filter(_.example == 10).collect()(0).value
    assert(fis3 ===  0.1837079, "Eleventh example") 
    val fis4 = mode.getMu().filter(_.example == 2).collect()(0).value
    assert(fis4 ===  0.882246, "Third example") 
    val fis5 = mode.getAnnotatorPrecision()(0)
    assert(fis5 ===  0.500566, "First annotator") 
    val fis7 = mode.getClusterDifficulty()(0)
    assert(fis7 ===  0.5001543, "First cluster difficulty") 
    val fis8 = mode.getRankData().filter(_.id==1).collect()(0).features(0)
    assert(fis8 ===  0.11677098, "First example rank data") 
    val fis9 = mode.getClusters().filter(_.example==1).collect()(0).cluster
    assert(fis9 ===  1, "First example cluster") 
  }

}
