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

    libraryDependencies += "com.enriquegrodrigo" %% "spark-crowd" % "0.1"








