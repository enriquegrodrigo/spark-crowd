
.. _MajorityVoting: https://enriquegrodrigo.github.io/spark-crowd/_static/api/index.html#com.enriquegrodrigo.spark.crowd.methods.MajorityVoting$ 
.. _DawidSkene: https://enriquegrodrigo.github.io/spark-crowd/_static/api/index.html#com.enriquegrodrigo.spark.crowd.methods.DawidSkene$
.. _GLAD: https://enriquegrodrigo.github.io/spark-crowd/_static/api/index.html#com.enriquegrodrigo.spark.crowd.methods.Glad$
.. _CGLAD: https://enriquegrodrigo.github.io/spark-crowd/_static/api/index.html#com.enriquegrodrigo.spark.crowd.methods.CGlad$
.. _RaykarBinary: https://enriquegrodrigo.github.io/spark-crowd/_static/api/index.html#com.enriquegrodrigo.spark.crowd.methods.RaykarBinary$
.. _RaykarMulti: https://enriquegrodrigo.github.io/spark-crowd/_static/api/index.html#com.enriquegrodrigo.spark.crowd.methods.RaykarBinary$
.. _RaykarCont: https://enriquegrodrigo.github.io/spark-crowd/_static/api/index.html#com.enriquegrodrigo.spark.crowd.methods.RaykarBinary$
.. _IBCC: https://enriquegrodrigo.github.io/spark-crowd/_static/api/index.html#com.enriquegrodrigo.spark.crowd.methods.IBCC$
.. _PM: https://enriquegrodrigo.github.io/spark-crowd/_static/api/index.html#com.enriquegrodrigo.spark.crowd.methods.PM$
.. _PMTI: https://enriquegrodrigo.github.io/spark-crowd/_static/api/index.html#com.enriquegrodrigo.spark.crowd.methods.PMTI$
.. _CATD: https://enriquegrodrigo.github.io/spark-crowd/_static/api/index.html#com.enriquegrodrigo.spark.crowd.methods.CATD$
.. _JRSS: https://www.jstor.org/stable/2346806?seq=1#page_scan_tab_contents
.. _NIPS: https://papers.nips.cc/paper/3644-whose-vote-should-count-more-optimal-integration-of-labels-from-labelers-of-unknown-expertise
.. _IDEAL: https://aida.ii.uam.es/ideal2018/#!/main 
.. _JMLR: http://jmlr.csail.mit.edu/papers/v11/raykar10a.html 
.. _SIGMOD: https://dl.acm.org/citation.cfm?id=2588555.2610509
.. _VLDB:  http://www.vldb.org/pvldb/vol8/p425-li.pdf
.. _VLDB2: http://www.vldb.org/pvldb/vol10/p541-zheng.pdf
.. _AISTATS: http://proceedings.mlr.press/v22/kim12.html

.. _methods:

Methods
=======

You can find the methods implemented in the library below. All methods contain a link to its API where you 
can find more information. Besides, in the table column, you can find a link to the reference. 


.. table:: Methods implemented in spark-crowd

  +------------------------+--------------------+--------------------+--------------------+------------------+
  | Method                 |   Binary           | Multiclass         | Real               | Reference        | 
  +========================+====================+====================+====================+==================+
  | `MajorityVoting`_      | :math:`\surd`      |  :math:`\surd`     |  :math:`\surd`     |                  | 
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | `DawidSkene`_          |  :math:`\surd`     |  :math:`\surd`     |                    | `JRSS`_          |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | `IBCC`_                |  :math:`\surd`     |  :math:`\surd`     |                    | `AISTATS`_       |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | `GLAD`_                |  :math:`\surd`     |                    |                    | `NIPS`_          |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | `CGLAD`_               |  :math:`\surd`     |                    |                    | `IDEAL`_         |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | Raykar                 |  :math:`\surd`     |  :math:`\surd`     |  :math:`\surd`     | `JMLR`_          |
  |                        |  `RaykarBinary`_   |  `RaykarMulti`_    |  `RaykarCont`_     |                  |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | `CATD`_                |                    |                    |  :math:`\surd`     | `VLDB`_          |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | `PM`_                  |                    |                    |  :math:`\surd`     | `SIGMOD`_        |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | `PMTI`_                |                    |                    |  :math:`\surd`     | `VLDB2`_         |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 

Below, we provide a short summary of each method (see the references for details). 

MajorityVoting
----------------

With this, we refer to the mean for continuous target variables and the most frequent class (the mode) for the discrete 
case. Expressing this methods in terms of annotator accuracy, these methods suppose that all annotators have 
the same experience. Therefore, their contributions are weighted equally. Apart from the classical mean and 
most frequent class, the package also provides methods that return the proportion of each class value for each example. 
See the `API Docs <https://enriquegrodrigo.github.io/spark-crowd/_static/api/#package/>`_ for more information on these methods. 

DawidSkene
------------

This method estimates the accuracy of the annotators from the annotations themselves. For this, it uses the EM 
algorithm, starting from the most frequent class and improving the estimations through several iterations. The 
algorithm returns both the estimation of the ground truth and the accuracy of the annotators (a confusion 
matrix for each). This algorithm is a good alternative when looking for a simple way of aggregating annotations
without the assumption that all annotators are equally accurate. 

IBCC
------

This method is similar to the previous one but uses probabilistic estimations for the classes. For each example, 
the model returns probabilities for each class, so they can be useful in problems where a probability is needed. 
It shows a good compromise between the complexity of the model and its performance, as can be seen in 
`this paper <https://zhydhkcws.github.io/crowd_truth_inference/index.html>`_ or in our experimentation in section :ref:`comparison`.  

GLAD
---------

This method estimates both the accuracy of the annotators (one parameter per annotator) and the difficulty
of each example (a parameter for each instance), through EM algorithm and gradient descent. This computational complexity 
comes at a cost of a higher execution time in general. However, GLAD, and its enhancement, CGLAD, also implemented here, 
are the only two algorithms capable of estimating the difficulty of the instances.

CGLAD
---------

This method is an enhancement over the original GLAD algorithm to tackle bigger datasets more easily, using 
clustering techniques over the examples to reduce the number of parameters to be estimated. It follows a similar 
learning process as the GLAD algorithm


Raykar's algorithms
---------------------

We implement the three methods proposed in the paper Learning from crowds (see the `paper <http://jmlr.csail.mit.edu/papers/v11/raykar10a.html>`_ 
for details) for learning
from crowdsourced data when features are available. These methods use an annotations dataset and the features for 
each instance. The algorithms infer together a logistic model (or a regression model, for the continuous case), the 
ground truth and the quality of the annotators, which are all returned by the methods in our package. 

CATD
-----------------

This method estimates both the quality of the annotators (as a weight in the aggregation) and the ground truth 
for continuous target variables. From the annotations, it estimates which annotators provide better labels. Then,
it assigns more weight to them for the aggregation. In the package, only the continuous version is implemented, as other 
algorithms seem to work better in the discrete cases.(see `this paper <https://zhydhkcws.github.io/crowd_truth_inference/index.html>`_ for more information).

PM and PMTI
-----------------

They are methods for continuous target variables. We implement two versions, one following the formulas appearing 
in the original paper and the modification implemented in  `this package <https://zhydhkcws.github.io/crowd_truth_inference/index.html>`_. This modification obtains better results in our experimentation (see :ref:`comparison`).  



















