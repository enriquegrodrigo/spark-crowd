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
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import com.enriquegrodrigo.spark.crowd.methods.IBCC
import com.enriquegrodrigo.spark.crowd.types._
import org.scalatest._
import org.scalatest.fixture
import org.scalactic.TolerantNumerics 
import java.io._

class IBCCTest extends fixture.FlatSpec with Matchers {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(1e-4f)

  case class FixtureParam(spark: SparkSession)

  def withFixture(test: OneArgTest) = {
      val conf = new SparkConf().setAppName("IBCC test").setMaster("local[*]")
      val spark = SparkSession.builder().config(conf).getOrCreate() 
      spark.sparkContext.setCheckpointDir("checkpoint")
      val sc = spark.sparkContext
      try super.withFixture(test.toNoArgTest(FixtureParam(spark))) finally {
        sc.stop()
      }
  }

  "IBCC" should "obtain the expected results in the test data" in { f => 
    val exampleFile = getClass.getResource("/multi-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val exampleData = spark.read.parquet(exampleFile).as[MulticlassAnnotation].toDF()
    val mode = IBCC(exampleData)
    val fis = mode.mu.where(col("example") === 0).where(col("class") ===0).collect()(0).getAs[Double](2)
    assert(fis === 0.9999, "First example. Class=0") 
    val fis2 = mode.mu.where(col("example") === 1).where(col("class") ===0).collect()(0).getAs[Double](2)
    assert(fis2 === 0.0, "Second example. Class=0") 
    val fis3 = mode.mu.where(col("example") === 1).where(col("class") ===1).collect()(0).getAs[Double](2)
    assert(fis3 === 0.9793, "Second example. Class=1") 
    val fis4 = mode.mu.where(col("example") === 3).where(col("class") ===1).collect()(0).getAs[Double](2)
    assert(fis4 === 0.9999, "Fourth example. Class=1") 
    val fis5 = mode.mu.where(col("example") === 6).where(col("class") ===1).collect()(0).getAs[Double](2)
    assert(fis5 === 0.0, "Seven example. Class=1") 
    val fis6 = mode.pi.where(col("annotator") === 0).where(col("c") === 0).where(col("k")===0).collect()(0).getAs[Double](3)
    assert(fis6 === 0.69216, "First Annotator. Class=0") 
    val fis7 = mode.pi.where(col("annotator") === 1).where(col("c") === 1).where(col("k")===1).collect()(0).getAs[Double](3)
    assert(fis7 === 0.79451, "Second Annotator. Class=1") 
    val fis8 = mode.pi.where(col("annotator") === 2).where(col("c") === 2).where(col("k")===2).collect()(0).getAs[Double](3)
    assert(fis8 === 0.729386, "Third Annotator. Class=2") 
    val fis9 = mode.p.where(col("class")===2).collect()(0)(1)
    assert(fis9 === 0.3466398592228416, "Class prior 2") 
  }
}
