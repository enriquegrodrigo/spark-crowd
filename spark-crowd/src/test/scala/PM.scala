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
import com.enriquegrodrigo.spark.crowd.methods.PM
import com.enriquegrodrigo.spark.crowd.types._
import org.scalatest._
import org.scalatest.fixture
import org.scalactic.TolerantNumerics 
import java.io._

class PMTest extends fixture.FlatSpec with Matchers {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(1e-4f)

  case class FixtureParam(spark: SparkSession)

  def withFixture(test: OneArgTest) = {
      val conf = new SparkConf().setAppName("PM test").setMaster("local[*]")
      val spark = SparkSession.builder().config(conf).getOrCreate() 
      spark.sparkContext.setCheckpointDir("checkpoint")
      val sc = spark.sparkContext
      try super.withFixture(test.toNoArgTest(FixtureParam(spark))) finally {
        sc.stop()
      }
  }

  "PM" should "obtain the expected results in the test data" in { f => 
    val exampleFile = getClass.getResource("/cont-ann.parquet").getPath
    val spark = f.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val exampleData = spark.read.parquet(exampleFile).toDF()
    val mode = PM(exampleData)
    val fis = mode.mu.where(col("example") === 0).collect()(0).getAs[Double](1)
    assert(fis === 21.145965, "First example.") 
    val fis2 = mode.mu.where(col("example") === 1).collect()(0).getAs[Double](1)
    assert(fis2 === 20.708624, "Second example.") 
    val fis3 = mode.mu.where(col("example") === 1).collect()(0).getAs[Double](1)
    assert(fis3 === 20.708624, "Second example.") 
    val fis4 = mode.mu.where(col("example") === 3).collect()(0).getAs[Double](1)
    assert(fis4 === 3.286017, "Fourth example.") 
    val fis5 = mode.mu.where(col("example") === 6).collect()(0).getAs[Double](1)
    assert(fis5 === 8.50105, "Seven example.") 
    val fis6 = mode.weights.where(col("annotator") === 0).collect()(0).getAs[Double](1)
    assert(fis6 === 2.312872, "First Annotator.") 
    val fis7 = mode.weights.where(col("annotator") === 1).collect()(0).getAs[Double](1)
    assert(fis7 === 2.310649, "Second Annotator.") 
    val fis8 = mode.weights.where(col("annotator") === 2).collect()(0).getAs[Double](1)
    assert(fis8 === 2.30007002, "Third Annotator.") 
  }
}
