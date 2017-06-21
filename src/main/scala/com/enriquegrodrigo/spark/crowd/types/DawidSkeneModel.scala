
package com.enriquegrodrigo.spark.crowd.types

import org.apache.spark.sql.Dataset

case class DawidSkeneModel(dataset: Dataset[MulticlassLabel], 
                                    params: DawidSkeneParams, 
                                    logLikelihood: Double) 


