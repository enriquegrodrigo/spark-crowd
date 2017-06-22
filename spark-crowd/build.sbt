lazy val root = (project in file(".")).settings(
    name := "spark-crowd",
    organization := "com.enriquegrodrigo",
    version := "0.1",
    scalaVersion := "2.11.8",
    isSnapshot := true,
    libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.1.0" % "provided",
    libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.1.0" % "provided",
    libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.1.0" % "provided",
    libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.0",
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % "test"
  )
