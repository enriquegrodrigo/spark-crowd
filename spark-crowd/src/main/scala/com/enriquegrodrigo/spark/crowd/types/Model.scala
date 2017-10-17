
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql._

trait Model[A] {
  def getMu(): Dataset[A]
}
