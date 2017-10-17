[![CircleCI](https://img.shields.io/circleci/project/github/enriquegrodrigo/spark-crowd.svg)](https://circleci.com/gh/enriquegrodrigo/spark-crowd)
[![Codecov](https://img.shields.io/codecov/c/github/enriquegrodrigo/spark-crowd.svg)](https://codecov.io/gh/enriquegrodrigo/spark-crowd)
 [![Maven Central](https://img.shields.io/maven-central/v/com.enriquegrodrigo/spark-crowd_2.11.svg)](https://mvnrepository.com/artifact/com.enriquegrodrigo/spark-crowd_2.11)
[![license](https://img.shields.io/github/license/enriquegrodrigo/spark-crowd.svg)](https://opensource.org/licenses/MIT)
# spark-crowd

A package for dealing with crowdsourced big data. 

Crowdsourced data introduces new problems that need to be addressed by 
Machine learning algorithms. This illustration exemplifies the main problems
of using this kind of data. 

![Crowdsourcing illustration](https://raw.githubusercontent.com/enriquegrodrigo/spark-crowd/master/img/illustration.png)



## Installation

The package uses [sbt](http://www.scala-sbt.org) for building the project, 
so we recommend installing this tool if you do not yet have it installed.

The simplest way to use the package is adding the next dependency directly 
into the `build.sbt` file of your project:

```scala
libraryDependencies += "com.enriquegrodrigo" %% "spark-crowd" % "0.1"
```

If this is not a possibility, you can compile the project and create a 
`.jar` file or you can publish the project to a local repository, as 
explained below.  

### Creating a `.jar` file and adding it to a new project

In the `spark-crowd` folder one should execute the command

    > sbt package 

to create a `.jar` file which will be in 
`target/scala-2.11/spark-crowd_2.11-0.1.jar` normally.

This `.jar` can be added to new projects using this library. In `sbt` one
can add `.jar` files to the `lib` folder.

### Publishing to a local repository

In the `spark-crowd` folder one should execute the command

    > sbt publish-local 

to publish the library to a local Ivy repository. One then can use the 
library adding the following line to the `build.sbt` file of a new
project:
```scala
    libraryDependencies += "com.enriquegrodrigo" %% "spark-crowd" % "0.1"
```


## Usage 

### Types

This package makes extensive use of Spark *DataFrame* an *Dataset* APIs. The last
uses typed rows which is beneficial for debugging purposes, among other things. 
As the annotations datasets normally have a fixed structure the package includes types
for three annotations datasets (binary, multiclass and real annotations), all of them 
with the following structure:

example | annotator | value 
--------|-----------|------
1 | 1| 0
1 | 2| 1 
2 | 2| 0
...|...|...

So the user needs to provide the annotations using this typed datasets to use the learning 
methods. This is usually simple if the user has all the information above in a Spark DataFrame:

```scala
import com.enriquegrodrigo.spark.crowd.types.BinaryAnnotation

val df = annotationDataFrame

val converted = fixed.map(x => BinaryAnnotation(x.getLong(0), x.getLong(1), x.getInt(2)))
                     .as[BinaryAnnotation]
```
The process is similar for the other types of annotation data. The `converted` Spark Dataset is ready to be use with the methods commented in the *Methods* subsection.

In the case of the feature dataset, the requisites are that:
 * Appart from the features, the data must have an `example` and a `class` columns
 * The example must be of type `Long`
 * Class must be of type `Integer` or `Double`, depending on the type of class (discrete or continuous)
 * All features must be of type `Double`

### Methods

The methods implemented as well as the type of annotations that they support are summarised 
in the following table:

Method | Binary | Multiclass | Real | Reference
:-----:|:------:|:----------:|:----:|:----------:
[MajorityVoting](https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.MajorityVoting$) | :white_check_mark: | :white_check_mark: | :white_check_mark: |  
[DawidSkene](https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.DawidSkene$) |:white_check_mark: | :white_check_mark: | | [JRSS](https://www.jstor.org/stable/2346806?seq=1#page_scan_tab_contents) 
[GLAD](https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.Glad$) | :white_check_mark: | | | [NIPS](https://papers.nips.cc/paper/3644-whose-vote-should-count-more-optimal-integration-of-labels-from-labelers-of-unknown-expertise)
[Raykar](https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.RaykarBinary$) | :white_check_mark: | :white_check_mark:| :white_check_mark:| [JMLR](http://jmlr.csail.mit.edu/papers/v11/raykar10a.html) 
[Kajino](https://enriquegrodrigo.github.io/spark-crowd/#com.enriquegrodrigo.spark.crowd.methods.Kajino$) | :white_check_mark: | | | [AAAI](https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/view/4919)

The algorithm name links to the documentation of the implemented method in our application, 
as well as to the publication where the algorithm was published. As an example, the 
following code shows how to use the `DawidSkene` method:

```scala
import com.enriquegrodrigo.spark.crowd.methods.DawidSkene

//Dataset of annotations
val df = annotationDataset.as[MulticlassAnnotation]

//Parameters for the method
val eMIters = 10
val eMThreshold = 0.01 

//Algorithm execution
result = DawidSkene(df,eMIters, eMThreshold) 
annotatorReliability = result.params.pi
groundTruth = result.dataset
```

The information obtained from each algorithm as well as about the parameters needed by them can 
be found in the [documentation](https://enriquegrodrigo.github.io/spark-crowd).

## Credits

Author:
 * Enrique G. Rodrigo

Contributors: 
 * Juan A. Aledo 
 * Jose A. Gamez

## License

MIT License

Copyright (c) 2017 Enrique González Rodrigo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
