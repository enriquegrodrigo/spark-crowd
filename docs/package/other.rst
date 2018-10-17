Comparison with other packages
==============================

There exists other packages implementing similar methods in other languages, but with 
different goals in mind. To our knowledge, there are 2 software packages with the goal 
of learning from crowdsourced data:

* `Ceka <http://ceka.sourceforge.net/>`_: it is a Java software package based on WEKA, with 
  a great number of methods that can be used to learn from crowdsource data. 
* `Truth inference in Crowdsourcing <https://zhydhkcws.github.io/crowd_truth_inference/index.html/>`_ makes available a collection
  of methods in Python to learn from crowdsourced data. 

Both are useful packages when dealing with crowdsourced data, with a focus on research. `spark-crowd` is different, in the sense that
not only is useful in research, but in production as well, providing tests for all of its methods with a high test coverage. Moreover, 
methods have been implemented with a focus on scalability, so it is useful in a wide variety of situations. We provide a 
comparison of the methods over a set of datasets next, taking into account both quality of the models and execution time. 

Data
-----

For this performance test we use simulated datasets of increasing size:

* **binary1-4**: simulated binary class datasets with 10K, 100K, 1M and 10M instances respectively. Each of them 
  has 10 simulated annotations per instance, and the ground truth for each example is known (but not used in the 
  learning process). The accuracy shown in the tables is obtained over this known ground truth. 
* **cont1-4**: simulated continuous target variable datasets, with 10k, 100k, 1M and 10M instances respectively. Each of them
  has 10 simulated annotations per instance, and the ground truth for each example is known (but not used in the 
  learning process). The Mean Absolute Error is obtained over this known ground truth.  
* **crowdscale**. A real multiclass dataset from the *Crowdsourcing at Scale* challenge. The data is comprised of 98979 instances, 
  evaluated by, at least, 5 annotators, for a total of 569375 answers. We only have ground truth for the 0.3% of the data, 
  which is used for evaluation. 

All datasets are available through this `link <>`_



CEKA
------

To compare our methods with Ceka, we used two of the main methods implemented in both packages, MajorityVoting and DawidSkene. Ceka and 
spark-crowd also implement GLAD and Raykar's algorithms. However, in Ceka, these algorithms are implemented using wrappers to other libraries. 
The library for the GLAD algorithm is not available on our platform, as it is given as an EXE Windows file, and the wrapper for Raykar's algorithms 
does not admit any configuration parameters. 

We provide the results of the execution of these methods in terms of accuracy (Acc) and time (in seconds). For our package, we also include 
the execution time for a cluster (tc) with 3 executor nodes of 10 cores and 30Gb of memory each. 

.. table:: Comparison with Ceka 

  +------------------------+-------------------------------------------------+---------------------------------------+
  |                        |   MajorityVoting                                | DawidSkene                            | 
  +------------------------+-------------------------+-----------------------+---------------+-----------------------+
  |                        |   Ceka                  | spark-crowd           | Ceka          | spark-crowd           |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | Method                 |   Acc   |     t1        |   Acc   |  t1  |  tc  | Acc   | t1    |   Acc   |  t1  |  tc  |
  +========================+=========+===============+=========+======+======+=======+=======+=========+======+======+
  | binary1                | 0.931   |     21        | 0.931   |  11  |   7  | 0.994 | 57    |  0.994  |  31  |  32  |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | binary2                | 0.936   |  15983        | 0.936   |  11  |   7  | 0.994 | 49259 |  0.994  | 181  |  43  |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | binary3                |   X     |     X         | 0.936   |  21  |   8  | X     | X     |  0.994  | 696  |  87  |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | binary4                |   X     |     X         | 0.936   |  84  |  42  | X     | X     | 0.994   | 1033 |  86  |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | crowdscale             |  0.88   |   10458       | 0.9     |  13  |  7   | 0.89  | 30999 | 0.9033  | 447  |  86  |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
 
Regarding accuracy, both packages achieve comparable results. However, regarding execution time, spark-crowd obtains 
significantly better results among all datasets especially on the bigger datasets, where it can solve problems that 
Ceka is not able to. You can see the speedup results in the table below.

.. table:: Speedup in comparison to Ceka 

  +------------------------+-------------------------------------+
  |                        |  MajorityVoting  |      DawidSkene  | 
  +------------------------+--------+---------+--------+---------+
  | Method                 |  t1    |  tc     |   t1   |     tc  |
  +========================+========+=========+========+=========+
  | binary1                | 1.86   |  2.93   |  1.84  |   1.78  |
  +------------------------+--------+---------+--------+---------+
  | binary2                |  1453  |  2283   |  272   |  1146   |
  +------------------------+--------+---------+--------+---------+
  | crowdscale             |  804   |  1494   |  69    |  360    |
  +------------------------+--------+---------+--------+---------+


We can see that spark-crowd obtains a high speedup in bigger datasets and performs 
slightly better in the smaller ones. 


Truth inference in crowdsourcing
----------------------------------

Now we compare spark-crowd with the methods available in this paper. Although the methods 
can certainly be used for to compare and try the algorithms, the integration of these 
methods into a large ecosystem will be very difficult, as the authors do not provide 
a software package structure. However, as it is an available package with a great number 
of methods, a comparison with them is needed. We will use the same datasets 
as the ones used in the previous comparison. In this case, we can compare a higher
number of models, as most of the methods are written in python. However, we were only able 
to execute the methods over datasets with binary or continuous target variables. As far as we 
know, the use of multiclass target variables seems to not be possible. Moreover, the use of 
feature sets is also restricted, although algorithms that should be capable of dealing with 
this kind of data are implemented, as is the case with the Raykar's methods. 

First, we compare the algorithms capable of learning from binary classes without feature sets. 
Inside this category, we will compare MajorityVoting, DawidSkene, GLAD and IBCC. We show the results
in terms of Accuracy (Acc) and time (in seconds) in the table below. 

.. table:: Comparative with Truth inference in Crowdsourcing package 

  +------------------------+-------------------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
  |                        |   MajorityVoting                                | DawidSkene                            | GLAD                                  | IBCC                                  | 
  +------------------------+-------------------------+-----------------------+---------------+-----------------------+---------------+-----------------------+---------------+-----------------------+
  |                        |   Truth-inf             | spark-crowd           | Truth-inf     | spark-crowd           | Truth-inf     | spark-crowd           | Truth-inf     | spark-crowd           |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+
  | Method                 |   Acc   |     t1        |   Acc   |  t1  |  tc  | Acc   | t1    |   Acc   |  t1  |  tc  | Acc   | t1    |   Acc   |  t1  |  tc  | Acc   | t1    |   Acc   |  t1  |  tc  |
  +========================+=========+===============+=========+======+======+=======+=======+=========+======+======+=======+=======+=========+======+======+=======+=======+=========+======+======+
  | binary1                | 0.931   |   1           | 0.931   |  11  |   7  | 0.994 | 12    |  0.994  |      |      | 0.994 | 1185  |  0.994  |      |      | 0.994 | 22    |  0.994  |      |      |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+
  | binary2                | 0.936   |   8           | 0.936   |  11  |   7  | 0.994 | 161   |  0.994  |      |      | 0.994 | 4168  |  0.994  |      |      | 0.994 | 372   |  0.994  |      |      |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+
  | binary3                | 0.936   |   112         | 0.936   |  21  |   8  | 0.994 | 1705  |  0.994  |      |      | X     | X     |  0.994  |      |      | 0.994 | 25764 |  0.994  |      |      |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+
  | binary4                |  0.936  |   2908        | 0.936   |  13  |  7   |   M   |   M   |  0.994  |      |      | X     | X     |  0.994  |      |      |   X   |   X   |    X    |      |      |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+


Next we analize methods that are able to learn from continuous target variables: MajorityVoting (mean), CATD and PM (with mean initialization). We show the results in terms of MAE (Mean absolute error) and time (in seconds).


.. table:: Comparative with Truth inference in Crowdsourcing package 

  +------------------------+-------------------------------------------------+---------------------------------------+---------------------------------------+
  |                        |   MajorityVoting (mean)                         | CATD                                  | PM                                    | 
  +------------------------+-------------------------+-----------------------+---------------+-----------------------+---------------+-----------------------+
  |                        |   Truth-inf             | spark-crowd           | Truth-inf     | spark-crowd           | Truth-inf     | spark-crowd           |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+
  | Method                 |   Acc   |     t1        |   Acc   |  t1  |  tc  | Acc   | t1    |   Acc   |  t1  |  tc  | Acc   | t1    |   Acc   |  t1  |  tc  |
  +========================+=========+===============+=========+======+======+=======+=======+=========+======+======+=======+=======+=========+======+======+
  | cont1                  | 1.234   |               | 1.234   |      |      | 0.324 |       |  0.324  |      |      | 0.495 |       |  0.495  |      |      |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+
  | cont2                  | 1.231   |               | 1.231   |      |      | 0.321 |       |  0.321  |      |      | 0.493 |       |  0.495  |      |      |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+
  | cont3                  | 1.231   |               | 1.231   |      |      |   X   |   X   |  0.322  |      |      | X     |       |  0.494  |      |      |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+
  | cont4                  | 1.231   |               | 1.231   |      |      |   X   |   X   |  0.322  |      |      | X     |       |  0.494  |      |      |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+-------+-------+---------+------+------+























