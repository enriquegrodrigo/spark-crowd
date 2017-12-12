# Example datasets

All data for the examples is included in the folder data using the parquet data format. 
For each type of class (binary, multiclass and continuous) we a dataset with three 
separated components:
- *type*-data: the file containing a Dataframe of the form:

| example | f0   | f1   | f2   | ... |  f9  |
|:-------:|:----:|:----:|:----:|:---:|:----:|
| 0       | 0.84 | 0.75 | 0.42 | ... | 0.58 | 
| 1       | ...  | ...  | ...  | ... | ...  | 

- *type*-ann: a Dataset with type `BinaryAnnotation`, `MulticlassAnnotation` or `RealAnnotation` depending on
the dataset type.
- *type*-gt: The ground truth for the example. Normally we won't have this information, as the data is generated, 
we can use the ground truth for comparisons.  

For all the datasets, we have use 6 good annotators and 4 bad annotators. For each of the types, this means:

- **Binary**: 
  - A good annotator has 80% chance of classifying correctly either when the true class is 0 or 1:

  | Class | 0 | 1 |
  |:-----:|:-:|:-:|
  |   0   |0.8|0.2|
  |   1   |0.2|0.8|

  - A bad annotator has 20% chance of classifying correctly either when the true class is 0 or 1. 

  | Class | 0 | 1 |
  |:-----:|:-:|:-:|
  |   0   |0.2|0.8|
  |   1   |0.8|0.2|

- **Multiclass**: 
  - A good annotator has 80% chance of classifying correctly either when the true class is 0 or 1:

  | Class | 0 | 1 | 2 |
  |:-----:|:-:|:-:|:-:|
  |   0   |0.8|0.2|0.2|
  |   1   |0.2|0.8|0.2|
  |   1   |0.2|0.2|0.8|

  - A bad annotator has 20% chance of classifying correctly either when the true class is 0 or 1. 

  | Class | 0 | 1 | 2 |
  |:-----:|:-:|:-:|:-:|
  |   0   |0.2|0.4|0.4|
  |   1   |0.4|0.2|0.4|
  |   1   |0.4|0.4|0.2|

- **Continuous**: 
  - A good annotator classifies with a normal distribution with standard deviation of 1 and mean the true class. In precision terms (inverse of variance), this would be ~ 1.
  - A good annotator classifies with a normal distribution with standard deviation of 8 and mean the true class. In precision terms (inverse of variance), this would be 1/64 ~= 0.016.
