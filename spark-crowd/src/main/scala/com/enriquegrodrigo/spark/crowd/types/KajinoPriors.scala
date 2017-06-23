
package com.enriquegrodrigo.spark.crowd.types

/**
 * Class for storing the Kajino method priors [[com.enriquegrodrigo.spark.crowd.methods.Kajino]].
 *
 * @param lambda value of the lambda parameter in the algorithm
 * @param eta value of the eta parameter in the algorithm
 * @author enrique.grodrigo
 * @version 0.1
 */
private[crowd] case class KajinoPriors(lambda: Double, eta: Double) 


