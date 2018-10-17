.. _MajorityVoting: https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.MajorityVoting$ 
.. _DawidSkene: https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.DawidSkene$
.. _GLAD: https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.GLAD$
.. _RaykarBinary: https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.RaykarBinary$
.. _RaykarMulti: https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.RaykarBinary$
.. _RaykarCont: https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.RaykarBinary$
.. _IBCC: https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.IBCC$
.. _PM: https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.PM$
.. _CATD: https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.CATD$
.. _JRSS: https://www.jstor.org/stable/2346806?seq=1#page_scan_tab_contents
.. _NIPS: https://papers.nips.cc/paper/3644-whose-vote-should-count-more-optimal-integration-of-labels-from-labelers-of-unknown-expertise
.. _JMLR: http://jmlr.csail.mit.edu/papers/v11/raykar10a.html 
.. _SIGMOD: https://dl.acm.org/citation.cfm?id=2588555.2610509
.. _VLDB:  http://www.vldb.org/pvldb/vol8/p425-li.pdf
.. _AISTATS: http://proceedings.mlr.press/v22/kim12.html

Methods
=======

You can find the methods implemented in this library below. All methods contain a link to its API where you 
can find more information. 

.. table:: Methods implemented in spark-crowd

  +------------------------+--------------------+--------------------+--------------------+------------------+
  | Method                 |   Binary           | Multiclass         | Real               | Reference        | 
  +========================+====================+====================+====================+==================+
  | `MajorityVoting`_      | :math:`\surd`      |  :math:`\surd`     |  :math:`\surd`     |                  | 
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | `DawidSkene`_          |  :math:`\surd`     |  :math:`\surd`     |                    | `JRSS`_          |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | `GLAD`_                |  :math:`\surd`     |                    |                    | `NIPS`_          |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | Raykar                 |  :math:`\surd`     |  :math:`\surd`     |  :math:`\surd`     | `JMLR`_          |
  |                        |  `RaykarBinary`_   |  `RaykarMulti`_    |  `RaykarCont`_     |                  |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | `IBCC`_                |  :math:`\surd`     |  :math:`\surd`     |                    | `AISTATS`_       |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | `CATD`_                |                    |                    |  :math:`\surd`     | `VLDB`_          |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 
  | `PM`_                  |                    |                    |  :math:`\surd`     | `SIGMOD`_        |
  +------------------------+--------------------+--------------------+--------------------+------------------+ 






