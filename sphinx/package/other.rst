.. _comparison:

Comparison with other packages
==============================

There exists other packages implementing similar methods in other languages, but with 
different goals in mind. To our knowledge, there are 2 software packages with the goal 
of learning from crowdsourced data:

* `Ceka <http://ceka.sourceforge.net/>`_: it is a Java software package based on WEKA, with 
  a great number of methods that can be used to learn from crowdsourced data. 
* `Truth inference in Crowdsourcing <https://zhydhkcws.github.io/crowd_truth_inference/index.html/>`_ makes available a collection
  of methods in Python to learn from crowdsourced data. 

Both are useful packages when dealing with crowdsourced data, with a focus mainly on research. Differently, `spark-crowd` is useful
not only in research, but also in production. It provides a clear usage interface as well as software tests for all of its methods
with a high tests coverage. Moreover, methods have been implemented with a focus on scalability, so it is 
useful in a wide variety of situations. A comparison of the methods over a set of datasets is provided in this section, taking 
into account both quality of the models and execution time. 

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

All datasets are available through this `link <https://www.dropbox.com/sh/odmhdf83latvezu/AAB6om3Oy7-waf-msIvk9yX6a?dl=0>`_



CEKA
------

To compare our methods with Ceka, we used two of the main methods implemented in both packages, MajorityVoting and DawidSkene. Ceka and 
``spark-crowd`` also implement GLAD and Raykar's algorithms. However, in Ceka, these algorithms are implemented using wrappers to other libraries. 
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
  | binary2                | 0.936   |  15983        | 0.936   |  11  |   7  | 0.994 | 49259 |  0.994  |  60  |  51  |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | binary3                |   X     |     X         | 0.936   |  21  |   8  | X     | X     |  0.994  | 111  |  69  |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | binary4                |   X     |     X         | 0.936   |  54  |  37  | X     | X     | 0.994   |      |      |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | crowdscale             |  0.88   |   10458       | 0.9     |  13  |  7   | 0.89  | 30999 | 0.9033  | 447  |  86  |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
 
Regarding accuracy, both packages achieve comparable results. However, regarding execution time, ``spark-crowd`` obtains 
significantly better results among all datasets, especially on the bigger datasets, where it can solve problems that 
Ceka is not able to. 


Truth inference in crowdsourcing
----------------------------------

Now we compare ``spark-crowd`` with the methods implemented by the authors. Although they 
can certainly be used to compare and test algorithms, the integration of these 
methods into a large ecosystem might be difficult, as the authors do not provide 
a software package structure. Nevertheless, as it is an available package with a great number 
of methods, a comparison with them is advisable. 

For the experimentation, the same datasets are used as well as the same environments. In this case, 
a higher number of models can be compared, as most of the methods are written in python. However, 
the methods can only be applied to binary or continuous target variables. As far as we know, the use of
multiclass target variables is not possible. Moreover, the use of feature information for Raykar's methods
it is also unsupported. 

First, we compare the algorithms capable of learning from binary classes. 
In this category, MajorityVoting, DawidSkene, GLAD and IBCC are compared. For each dataset, the results
in terms of Accuracy (Acc) and time (in seconds) are obtained. The table below shows the results for 
MajorityVoting and DawidSkene. Both packages obtain the same results in terms of 
accuracy. For the smaller datasets, the overhead imposed by parallelism makes Truth-inf a better choice, 
at least in terms of execution time. However, as the datasets increase, and especially, in the last two 
instances, the speedup obtained by our algorithm is notable. In the case of DawidSkene, the Truth-inf 
package is not able to complete the execution because of memory constraints in the largest dataset.  


.. table:: Comparative with Truth inference in Crowdsourcing package 

  +------------------------+-------------------------------------------------+---------------------------------------+
  |                        |   MajorityVoting                                | DawidSkene                            | 
  +------------------------+-------------------------+-----------------------+---------------+-----------------------+
  |                        |   Truth-inf             | spark-crowd           | Truth-inf     | spark-crowd           |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | Method                 |   Acc   |     t1        |   Acc   |  t1  |  tc  | Acc   | t1    |   Acc   |  t1  |  tc  |
  +========================+=========+===============+=========+======+======+=======+=======+=========+======+======+
  | binary1                | 0.931   |   1           | 0.931   |  11  |   7  | 0.994 | 12    |  0.994  | 31   | 32   |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | binary2                | 0.936   |   8           | 0.936   |  11  |   7  | 0.994 | 161   |  0.994  | 60   | 51   |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | binary3                | 0.936   |   112         | 0.936   |  21  |   8  | 0.994 | 1705  |  0.994  | 111  | 69   |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | binary4                |  0.936  |   2908        | 0.936   |  57  |  37  |   M   |   M   |  0.994  | 703  | 426  |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+


Next we show the results for GLAD and IBCC. As can be seen, both packages obtain similar results 
in terms of accuracy. Regarding execution time, they obtain comparable results 
in the two smaller datasets (with a slight speedup in ``binary2``) for the GLAD algorithm. However, 
for this algorithm, Truth-inf is not able to complete the execution for the two largest datasets. 
In the case of IBCC, the speedup starts to be noticeable from the second dataset on. It is also noticeable 
that Truth-Inf did not complete the execution for the last dataset. 



.. table:: Comparative with Truth inference in Crowdsourcing package (2)

  +------------------------+---------------------------------------+---------------------------------------+
  |                        | GLAD                                  | IBCC                                  |
  +------------------------+---------------+-----------------------+---------------+-----------------------+
  |                        | Truth-inf     | spark-crowd           | Truth-inf     | spark-crowd           |
  +------------------------+-------+-------+---------+------+------+-------+-------+---------+------+------+
  | Method                 | Acc   | t1    |   Acc   |  t1  |  tc  | Acc   | t1    |   Acc   |  t1  |  tc  |
  +========================+=======+=======+=========+======+======+=======+=======+=========+======+======+
  | binary1                | 0.994 | 1185  |  0.994  | 1568 | 1547 | 0.994 | 22    |  0.994  | 74   | 67   |
  +------------------------+-------+-------+---------+------+------+-------+-------+---------+------+------+
  | binary2                | 0.994 | 4168  |  0.994  | 2959 | 2051 | 0.994 | 372   |  0.994  | 97   | 76   |
  +------------------------+-------+-------+---------+------+------+-------+-------+---------+------+------+
  | binary3                | X     | X     |  0.491  | 600  | 226  | 0.994 | 25764 |  0.994  | 203  | 129  |
  +------------------------+-------+-------+---------+------+------+-------+-------+---------+------+------+
  | binary4                | X     | X     |  0.974  | 2407 | 1158 |   X   |   X   |    X    | 1529 | 823  |
  +------------------------+-------+-------+---------+------+------+-------+-------+---------+------+------+

Note that the performance of GLAD algorithm seems to degrade in the bigger datasets. 
This may be due to the ammount of parameters the algorithm needs to estimate. 
A way to improve the estimation goes through decreasing the learning rate, which 
makes the algorithm slower, as it needs a lot more iterations to obtain a good solution. This makes the algorithm 
unsuitable for several big data contexts.  To tackle this kind of problems, we developed and enhancement, CGLAD, which is 
included in this package  (See the last section of this page for results of other 
methods in the package, as well as this enhancement).

Next we analize methods that are able to learn from continuous target variables: MajorityVoting (mean), CATD and PM (with mean initialization). 
We show the results in terms of MAE (Mean absolute error) and time (in seconds). The 
results for MajorityVoting and CATD can be found in the table below. 


.. table:: Comparative with Truth inference in Crowdsourcing package on continuous target variables 

  +------------------------+-------------------------------------------------+---------------------------------------+
  |                        |   MajorityVoting (mean)                         | CATD                                  | 
  +------------------------+-------------------------+-----------------------+---------------+-----------------------+
  |                        |   Truth-inf             | spark-crowd           | Truth-inf     | spark-crowd           |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | Method                 |   Acc   |     t1        |   Acc   |  t1  |  tc  | Acc   | t1    |   Acc   |  t1  |  tc  |
  +========================+=========+===============+=========+======+======+=======+=======+=========+======+======+
  | cont1                  | 1.234   |      1        | 1.234   |  6   |  8   | 0.324 | 207   |  0.324  | 25   | 28   |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | cont2                  | 1.231   |      8        | 1.231   | 7    |  9   | 0.321 | 10429 |  0.321  | 26   | 24   |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | cont3                  | 1.231   |     74        | 1.231   | 12   | 13   |   X   |   X   |  0.322  | 42   | 38   |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+
  | cont4                  | 1.231   |    581        | 1.231   | 56   | 23   |   X   |   X   |  0.322  | 247  | 176  |
  +------------------------+---------+---------------+---------+------+------+-------+-------+---------+------+------+


As you can see in the table, both packages obtain similar results regarding MAE. Regarding execution time, 
the implementation of MajorityVoting from the Truth-inf package obtains good results, especially in the smallest
dataset. It is worth pointing out that, for the smallest datasets, the overhead imposed by parallelism makes the 
execution time of our package a little worse in comparison. However, as datasets increase in size, the speedup 
obtained by our package is notable, even in MajorityVoting, which is less complex computationally. 
Regarding CATD, Truth-inf seems not to be able to solve the 2 bigger problems 
in a reasonable time, however, they can be solved by our package in a small ammount of time. Even for the smaller 
datasets, our package obtains a high speedup in comparison to Truth-inf for CATD.

In the table below you can find the results for PM and PMTI algorithms. 



.. table:: Comparative with Truth inference in Crowdsourcing package on continuous target variables (2)

   +------------------------+---------------------------------------+---------------------------------------+
   |                        | PM                                    | PMTI                                  |
   +------------------------+---------------+-----------------------+---------------+-----------------------+
   |                        | Truth-inf     | spark-crowd           | Truth-inf     | spark-crowd           |
   +------------------------+-------+-------+---------+------+------+-------+-------+---------+------+------+
   | Method                 | Acc   | t1    |   Acc   |  t1  |  tc  | Acc   | t1    |   Acc   |  t1  |  tc  |
   +========================+=======+=======+=========+======+======+=======+=======+=========+======+======+
   | cont1                  | 0.495 | 77    |  0.495  | 57   | 51   | 0.388 | 139   |  0.388  | 68   |  61  |
   +------------------------+-------+-------+---------+------+------+-------+-------+---------+------+------+
   | cont2                  | 0.493 | 8079  |  0.495  | 76   | 57   | 0.386 | 14167 |  0.386  | 74   |  58  |
   +------------------------+-------+-------+---------+------+------+-------+-------+---------+------+------+
   | cont3                  | X     |  X    |  0.494  | 130  | 97   | X     |  X    |  0.387  | 143  |  98  |
   +------------------------+-------+-------+---------+------+------+-------+-------+---------+------+------+
   | cont4                  | X     |  X    |  0.494  | 769  | 421  | X     |  X    |  0.387  | 996  | 475  |
   +------------------------+-------+-------+---------+------+------+-------+-------+---------+------+------+

Although similar, the modification implemented in Truth-inf from the original algorithm seems to be more 
accurate. Even in the smallest sizes, our package obtains a slight speedup. However, as the datasets increase in 
size, our package is able to obtain a much higher speedup. 

Other methods
---------------

To complete our experimentation, next we focus on other methods implemented by our package that are not implemented 
by Ceka or Truth-Inf. These methods are the full implementation of the Raykar's algorithms (taking into account the 
features of the instances) and the enhancement over the GLAD algorithm. As a note, Truth-Inf implements a version of Raykar's 
algorithms that does not use the features of the instances. First, we show the results obtained by the Raykar's methods 
for discrete target variables. 

.. table:: Other methods implemented in spark-crowd. Raykar's methods for discrete target variables. 

   +------------------------+---------------------------------------+---------------------------------------+
   |                        | RaykarBinary                          | RaykarMulti                           |
   +------------------------+---------------------------------------+---------------------------------------+
   |                        | spark-crowd                           | spark-crowd                           |
   +------------------------+---------+------+----------------------+---------+------+----------------------+
   | Method                 |   Acc   |  t1  |  tc                  |   Acc   |  t1  |  tc                  |
   +========================+=========+======+======================+=========+======+======================+
   | binary1                |  0.994  |  65  | 63                   |  0.994  | 167  | 147                  |
   +------------------------+---------+------+----------------------+---------+------+----------------------+
   | binary2                |  0.994  | 92   | 74                   |  0.994  | 241  | 176                  |
   +------------------------+---------+------+----------------------+---------+------+----------------------+
   | binary3                |  0.994  | 181  | 190                  |  0.994  | 532  | 339                  |
   +------------------------+---------+------+----------------------+---------+------+----------------------+
   | binary4                |  0.994  | 1149 | 560                  |  0.994  | 4860 | 1196                 |
   +------------------------+---------+------+----------------------+---------+------+----------------------+


Next we show the Raykar method for tackling continous target variables. 


.. table:: Other methods implemented in spark-crowd. Raykar method for continuous target variables. 

   +------------------------+---------------------------------------+
   |                        | RaykarCont                            |
   +------------------------+---------------+-----------------------+
   |                        | spark-crowd                           |
   +------------------------+---------+------+----------------------+
   | Method                 |   Acc   |  t1  |  tc                  |
   +========================+=========+======+======================+
   | cont1                  |  0.994  | 31   | 32                   |
   +------------------------+---------+------+----------------------+
   | cont2                  |  0.994  | 60   | 51                   |
   +------------------------+---------+------+----------------------+
   | cont3                  |  0.994  | 111  | 69                   |
   +------------------------+---------+------+----------------------+
   | cont4                  |  0.994  | 703  | 426                  |
   +------------------------+---------+------+----------------------+

Finally, we show the results for the CGLAD algorithm. As you can see, it obtains similar results to the GLAD
algorithm in the smallest instances but it performs much better in the larger ones. Regarding execution time, CGLAD
obtains a high speedup in the cases where accuracy results for both algorithms are similar. 

.. table:: Other methods implemented in spark-crowd. CGlad, an enhancement over Glad algorithm. 

   +------------------------+-------------------------+
   |                        | CGlad                   |
   +------------------------+-------------------------+
   |                        | spark-crowd             |
   +------------------------+--------+-------+--------+
   | Method                 |   Acc  |  t1   |  tc    |
   +========================+========+=======+========+
   | binary1                |  0.994 | 128   | 128    |
   +------------------------+--------+-------+--------+
   | binary2                |  0.995 | 233   | 185    |
   +------------------------+--------+-------+--------+
   | binary3                |  0.995 | 1429  | 607    |
   +------------------------+--------+-------+--------+
   | binary4                |  0.995 | 17337 | 6190   |
   +------------------------+--------+-------+--------+
