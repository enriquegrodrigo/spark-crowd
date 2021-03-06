Examples
==========

In this page you can find examples for several of the algorithms in the library. 
You can find the data used for the examples in the Github repository. 

MajorityVoting
----------------

The example below shows how to use the MajorityVoting algorithm for estimating the ground truth for a binary target variable. 

.. code-block:: scala

  import com.enriquegrodrigo.spark.crowd.methods.MajorityVoting
  import com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation
  
  val exampleFile = "data/binary-ann.parquet"
  
  val exampleDataBinary = spark.read.parquet(exampleFile).as[BinaryAnnotation] 
  
  val muBinary = MajorityVoting.transformBinary(exampleDataBinary)

  muBinary.show()

The method returns a result similar to this one: 

.. code-block:: scala

    +-------+-----+
    |example|value|
    +-------+-----+
    |     26|    0|
    |     29|    1|
    |    474|    0|
    |    964|    1|
    |     65|    0|
    |    191|    0|
    |    418|    1|
    ....

MajorityVoting algorithms assume that all annotators are equally accurate, so they choose the 
most frequent annotation as the ground truth label. Because of this, they only return the ground 
truth for the problem. 

The data file in this example follow the format from the ``BinaryAnnotation`` type:

.. code-block:: scala 

  example, annotator, value
        0,         0,     1
        0,         1,     0
        0,         2,     1
        ...

In this example, we use a ``.parquet`` data file, which is usually a good option in terms of 
efficiency. However, we do not limit the types of files you can use, as long as they can be 
converted to typed datasets of ``BinaryAnnotation``, ``MulticlassAnnotation`` or ``RealAnnotation``.
However, algorithms will suppose that there are no missing examples or annotators. 

Specifically, MajorityVoting can make predictions both for discrete classes (``BinaryAnnotation`` and
``MulticlassAnnotation``) and continuous-valued target variables. (``RealAnnotation``). You can find 
information about these methods in the `API Docs <https://enriquegrodrigo.github.io/spark-crowd/_static/api/#package/>`_. 

DawidSkene
------------

This algorithm is one of the most recommended both for its simplicity and its good results generally. 


.. code-block:: scala
 
  import com.enriquegrodrigo.spark.crowd.methods.DawidSkene
  import com.enriquegrodrigo.spark.crowd.types.MulticlassAnnotation
  
  val exampleFile = "examples/data/multi-ann.parquet"
  
  val exampleData = spark.read.parquet(exampleFile).as[MulticlassAnnotation] 
  
  val mode = DawidSkene(exampleData, eMIters=10, emThreshold=0.001)
  
  val pred = mode.getMu().as[MulticlassLabel] 
  
  val annprec = mode.getAnnotatorPrecision()
   

In the implementation, two parameters are used for controlling the algorithm execution,
the maximum number of EM iterations and the threshold for the likelihood change.  The execution stops if the number of 
iterations reaches the established maximum or if the change in likelihood is less than the threshold. You do not need to 
provide these parameters, as they have default values. 

One executed, the model provides an estimation of the ground truth, and an estimation of 
the quality of each annotator, in the form of a confusion matrix. This information can be obtained as shown on the example.  


GLAD
-------

The GLAD algorithm is interesting as it provides both annotator accuracies and example difficulties obtained 
solely from the annotations. An example of how to use it can be found below.

.. code-block:: scala

  import com.enriquegrodrigo.spark.crowd.methods.Glad
  import com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation
  
  val annFile = "data/binary-ann.parquet"
  
  val annData = spark.read.parquet(annFile).as[BinaryAnnotation] 
  
  val mode = Glad(annData, 
                    eMIters=5, //Maximum number of iterations of EM algorithm
                    eMThreshold=0.1, //Threshold for likelihood changes
                    gradIters=30, //Gradient descent max number of iterations
                    gradTreshold=0.5, //Gradient descent threshold
                    gradLearningRate=0.01, //Gradient descent learning rate
                    alphaPrior=1, //Alpha first value (GLAD specific) 
                    betaPrior=1) //Beta first value (GLAD specific)
  
  val pred = mode.getMu().as[BinarySoftLabel] 
  
  val annprec = mode.getAnnotatorPrecision()
  
  val annprec = mode.getInstanceDifficulty()
  

This model as implemented in the library is only compatible with binary class problems. It has a 
higher number of free parameters in comparison with the previous algorithm, but we provided default 
values for all of them for convenience. The meaning of each of these parameters is commented in the 
example above, as it is on the `API Docs <https://enriquegrodrigo.github.io/spark-crowd/_static/api/#package/>`_. 
The annotator precision is given as a vector, with an entry for each annotator. The difficulty is given in the form of a DataFrame, returning 
a difficulty value for each example. For more information, you can consult the documentation and/or the paper. 


RaykarBinary, RaykarMulti and RaykarCont 
-----------------------------------------

We implement the three variants of this algorithm, two for discrete target variables (``RaykarBinary`` and 
``RaykarMulti``) and one for continuous variables (``RaykarCont``). 
These algorithms have in common that they are able to use features to estimate the ground truth 
and even learn a linear model. The model also is able to use prior information about annotators, 
which can be useful to add more confidence to certain annotators. The next example shows 
how to use this priors to indicate that the trust put in the first annotator is higher and 
that the second annotator is not reliable.

.. code-block:: scala

  import com.enriquegrodrigo.spark.crowd.methods.RaykarBinary
  import com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation
  
  val exampleFile = "data/binary-data.parquet"
  val annFile = "data/binary-ann.parquet"
  
  val exampleData = spark.read.parquet(exampleFile)
  val annData = spark.read.parquet(annFile).as[BinaryAnnotation] 
  
  //Preparing priors
  val nAnn = annData.map(_.annotator).distinct.count().toInt
  
  val a = Array.fill[Double](nAnn,2)(2.0) //Uniform prior
  val b = Array.fill[Double](nAnn,2)(2.0) //Uniform prior
  
  //Give first annotator more confidence
  a(0)(0) += 1000 
  b(0)(0) += 1000 
  
  //Give second annotator less confidence
  //Annotator 1
  a(1)(1) += 1000 
  b(1)(1) += 1000 
  
  
  //Applying the learning algorithm
  val mode = RaykarBinary(exampleData, annData, 
                            eMIters=5,
                            eMThreshold=0.001,
                            gradIters=100,
                            gradThreshold=0.1,
                            gradLearning=0.1
                            a_prior=Some(a), b_prior=Some(b))
  
  //Get MulticlassLabel with the class predictions
  val pred = mode.getMu().as[BinarySoftLabel] 
  
  //Annotator precision matrices
  val annprec = mode.getAnnotatorPrecision()


Apart form the features matrix and the priors, the meaning of the parameters is the same as in the previous examples. 
The priors are matrices of dimension A by 2, where A is the number of annotators. In each row we have the hyperparameters of a Beta distribution for each annotator.
The ``a_prior`` gives prior information about the ability of annotators to correctly classify a positive example. The 
``b_prior`` does the same thing but for the negative examples. More information about this method as well as the methods
for discrete and continuous target variables can be found in the `API Docs <https://enriquegrodrigo.github.io/spark-crowd/_static/api/#package/>`_. 


CATD
-------

This method allows to estimate continuous-value target variables from annotations.  


.. code-block:: scala

  import com.enriquegrodrigo.spark.crowd.methods.CATD
  import com.enriquegrodrigo.spark.crowd.types.RealAnnotation
  
  sc.setCheckpointDir("checkpoint")
  
  val annFile = "examples/data/cont-ann.parquet"
  
  val annData = spark.read.parquet(annFile).as[RealAnnotation]
  
  //Applying the learning algorithm
  val mode = CATD(annData, iterations=5,
                            threshold=0.1,
                            alpha=0.05)
  
  //Get MulticlassLabel with the class predictions
  val pred = mode.mu
  
  //Annotator precision matrices
  val annprec = mode.weights


It returns a model from which you can get the ground truth estimation and 
also the annotator weight used (more weight means a better annotator). 
The algorithm uses parameters such as ``iterations`` and ``threshold`` for 
controlling the execution, and also ``alpha``, which is a parameter of the model
(check the `API Docs <https://enriquegrodrigo.github.io/spark-crowd/_static/api/#package/>`_ for more information).





