
.. _installation: 

Installation
==============

There are three alternatives to use the package in your own software: 

* Using the package directly from Spark Packages. 
* Adding it as a dependency to your project through Maven central.
* Compiling the source code and using the ``jar`` file.

Alternatively, if you just want to execute simple scala scripts locally, 
you can use the provided docker image as explained in :ref:`quickstart` 

Using Spark Packages
---------------------

The easiest way of using the package is through `Spark Packages <https://spark-packages.org/>`_, as you only need to add the package in the command line when running your 
application:

.. code-block:: shell

  spark-submit --packages com.enriquegrodrigo:spark-crowd_2.11:0.2.1 application.scala

You can also open a `spark-shell` using: 

.. code-block:: shell

  spark-shell --packages com.enriquegrodrigo:spark-crowd_2.11:0.2.1


Adding it as a dependency
--------------------------

In addition to Spark Packages, the library is also in Maven Central, so you can add it as a dependency in your scala project. 
For example, in *sbt* you can add the dependency as shown below:

.. code-block:: scala 

  libraryDependencies += "com.enriquegrodrigo" %% "spark-crowd" % "0.2.1"

This allows you to use the methods inside your Apache Spark projects. 

Compiling the source code
--------------------------

To build the package using *sbt* you can use the following command inside the spark-crowd folder:

.. code-block:: shell

  sbt package 

It generates a compiled ``jar`` file that you can add to your project. 


