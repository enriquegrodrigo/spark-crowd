 
.. _quickstart:

Quick Start
=============

You can easily start using ``spark-crowd`` through our `docker <https://www.docker.com/>`_ image or through `spark-packages <https://spark-packages.org/>`_. 
See :ref:`installation`, for all installation alternatives (such as how to add the package as a dependency in your project). 

Start with our docker image
---------------------------

The quickest way to try out the package is through the 
`provided docker image <https://hub.docker.com/r/enriquegrodrigo/spark-crowd/>`_ with the latest version of 
our package, as you do not need to install any other software (apart from docker). 

.. code-block:: shell

  docker pull enriquegrodrigo/spark-crowd

Thanks to it, you can run the examples provided along with the
`package <https://github.com/enriquegrodrigo/spark-crowd>`_. For example, 
to run `DawidSkeneExample.scala` we can use:

.. code-block:: shell

  docker run --rm -it -v $(pwd)/:/home/work/project enriquegrodrigo/spark-crowd DawidSkeneExample.scala

You can also open a spark shell with the library preloaded. 

.. code-block:: shell

  docker run --rm -it -v $(pwd)/:/home/work/project enriquegrodrigo/spark-crowd 

By doing that, you can test you code directly. You will not benefit from the distributed execution of Apache Spark, 
but you are still able to use the algorithms with medium-sized datasets (since docker can use several cores in your 
machine). 



Start with `spark-packages` 
----------------------------------------

If you have an installation of `Apache Spark <https://spark.apache.org/>`_  you can open a `spark-shell` with 
our package pre-loaded using:

.. code-block:: shell

  spark-shell --packages com.enriquegrodrigo:spark-crowd_2.11:0.2.1

Likewise, you can submit an application to your cluster that uses `spark-crowd` using:

.. code-block:: shell

  spark-submit --packages com.enriquegrodrigo:spark-crowd_2.11:0.2.1 application.scala

To use this option you do not need to have a cluster of computers, you can also execute the code from 
your local machine because Apache Spark can be installed locally. For more information on how to install 
Apache Spark, please refer to its `homepage <https://spark.apache.org/>`_.

Basic usage
----------------

Once you have chosen a procedure to run the package, you have to import the method
that you want to use as well as the types for your data, as you can see below:   

.. code-block:: scala 

  import com.enriquegrodrigo.spark.crowd.methods.DawidSkene
  import com.enriquegrodrigo.spark.crowd.types.MulticlassAnnotation
  
  val exampleFile = "examples/data/multi-ann.parquet"
  
  val exampleData = spark.read.parquet(exampleFile).as[MulticlassAnnotation] 
  
  //Applying the learning algorithm
  val mode = DawidSkene(exampleData)
  
  //Get MulticlassLabel with the class predictions
  val pred = mode.getMu().as[MulticlassLabel] 
  
  //Annotator precision matrices
  val annprec = mode.getAnnotatorPrecision()

You can find a description of the code below:

#. First the method and the type are imported, in this case ``DawidSkene`` and ``MulticlassAnnotation``. The type 
   is needed as the package API only accepts typed datasets for the annotations.
#. Then the data file (provided with the package) is loaded. It contains annotations for different examples. As you 
   can see, the examples uses the method ``as`` to convert the Spark DataFrame in a typed Spark Dataset (with type
   MulticlassAnnotation). 
#. To execute the model and obtain the result, you can use the model name directly. 
   This function returns a ``DawidSkeneModel``, which includes several methods to obtain results from the algorithm.
#. The method  ``getMu`` returns the ground truth estimations made by the model. 
#. We use ``getAnnotatorPrecision`` to obtain the annotator quality calculated by the model. 

You can consult the models implemented in this package in :ref:`methods`, where you can find a link to the 
original article for each algorithm. 
