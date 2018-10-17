 
.. _quickstart:

Quick Start
=============

You can start quickly using our package through our `docker <https://www.docker.com/>`_ image or through `spark-packages <https://spark-packages.org/>`_. 
See :ref:`installation`, for all installation alternatives. 

Start with our docker image
---------------------------

The quickest way to try our package is using the `provided docker image <https://hub.docker.com/r/enriquegrodrigo/spark-crowd/>`_ image with the latest 
version of our package, as you won't need to install anything. 

.. code-block:: shell

  docker pull enriquegrodrigo/spark-crowd

With it you can run the examples provided along with the `package <https://github.com/enriquegrodrigo/spark-crowd>`_. For example, to run `DawidSkeneExample.scala`
we can use:

.. code-block:: shell

  docker run --rm -it -v $(pwd)/:/home/work/project enriquegrodrigo/spark-crowd DawidSkeneExample.scala

You can also open a spark shell with the library preloaded. 

.. code-block:: shell

  docker run --rm -it -v $(pwd)/:/home/work/project enriquegrodrigo/spark-crowd 

So you can test your code directly. 


Start with `spark-packages` 
----------------------------------------

If you have an installation of `Apache Spark <https://spark.apache.org/>`_  a you can open an `spark-shell` using:

.. code-block:: shell

  spark-shell --packages com.enriquegrodrigo:spark-crowd_2.11:0.1.5

Likewise, you can submit an application that uses `spark-crowd` using:

.. code-block:: shell

  spark-submit --packages com.enriquegrodrigo:spark-crowd_2.11:0.1.5 application.scala


Basic usage
----------------

Once you have chosen your preferred installation procedure, you only need to import the corresponding method
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

Let's go through each step of the code:

#. First we import the method, in this case `DawidSkene` and the annotations type (`MulticlassAnnotation`) that we will need 
   to load the data.
#. Then we load a data file (provided with the package) that contains annotations for different examples. We use the method `as` to 
   to convert the Spark DataFrame in a typed Spark Dataset (with type `MulticlassAnnotation`). 
#. To execute the model and obtain the result we use the model name directly. This function returns a `DawidSkeneModel`, that
   includes the methods several methods to obtain results from the algorithm.
#. We use the  `getMu` to obtain the ground truth estimations made by the model. 
#. We use `getAnnotatorPrecision` to obtain for the annotator precisions calculated by the model. 


