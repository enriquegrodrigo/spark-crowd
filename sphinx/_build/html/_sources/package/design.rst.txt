Design and architechture
========================

The package design can be found in the figure below. 

.. image:: ../../img/package.png


Although, the library contains several folders, the only folders important for the users 
are the ``types`` folder, and the ``methods``. The other folders contain auxiliary
functions some of the methods. Concretely, in interesting to explore the data types, as 
they are key to understanding how the package works, as well as the common interface of 
the methods. 


Data types
------------

We provide types for annotations datasets and ground truth datasets, as they usually follow 
the same structure. These types are used in all the methods so you would need to convert 
your annotations dataset the correct format accepted by the algorithm. 

There are three types of annotations that we support for which we provide Scala case classes,
making it possible to detect errors at compile time when using the algorithms:

* ``BinaryAnnotation``: a Dataset of this type provides three columns, an example column, that 
  is the example for which the annotation is made, an annotator column, representing the 
  annotator that made the annotation and a value column, with the value of the annotation, that 
  can take value 0 or 1. 
* ``MulticlassAnnotation``: The difference form ``BinaryAnnotation`` is that the value column can 
  take more than two values, in the range from 0 to the total number of values. 
* ``RealAnnotation``: In this case, the value column can take any numeric value.


You can convert an annotation dataframe with columns example, annotator and value to a 
typed dataset easily with the following instruction:

.. code-block:: scala

  val typedData = untypedData.as[RealAnnotation]


In the case of labels, we provide 5 types of labels, 2 of which are probabilistic. The three non probabilistic 
types are: 

* ``BinaryLabel``: represents a dataset of example, value pairs where value is a binary value (0 or 1).
* ``MulticlassLabel``: where value can take more than two values. 
* ``RealLabel``: where value can take any numeric value. 

The probabilistic types are used by some algorithms, to provide more information about the confidence of each 
class value for an specific example. 

* ``BinarySoftLabel``: represents a dataset with two columns: example, and probability (prob). For each example, the probability 
  of positive is given.  
* ``MultiSoftLabel``: representas a dataset with three columns: example, class and probability (prob). For each example, there will be 
  several entries depending on the number of classes of the problem, with the probability estimate. 


Methods
---------

All methods implemented are in the ``methods`` package and are mostly independent of each other. There is only one exception to this, the 
use of the MajorityVoting algorithms, as most of the algorithms used these methods in the initialization step. Apart from that, all logic 
is implemented in their specific files.  This makes it easier to extend the package with new algorithms. Although independent, all algorithms have 
a similar interface, which facilitates its use. To execute an algorithm, the user normally needs to use the ``apply`` method of the model, as shown below

.. code-block:: scala

  ...
  val model = IBCC(annotations)
  ...

After the model completes its execution, a model object is returned, which will have information about the ground truth estimations and 
annotator's quality and instance difficulties. 

The only algorithm that do not follow this pattern is ``MajorityVoting``, which has methods for each of the class types and also to obtain 
probabilistic labels. See the API Docs for details. 





