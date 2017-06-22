# spark-crowd
The repository for spark-crowd, a package for dealing with crowdsourced big data. 

## Installation

The package uses [sbt](http://www.scala-sbt.org) for building the project, 
so the first step is installing this tool if it is not yet installed.

After that, one can create a `.jar` for adding to a new project or publish
to a local repository. 

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


















