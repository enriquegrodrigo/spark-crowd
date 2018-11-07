lazy val root = (project in file(".")).settings(
    name := "spark-crowd",
    organization := "com.enriquegrodrigo",
    version := "0.2.1",
    scalaVersion := "2.11.8",
    pomIncludeRepository := { _ => false },
    publishTo := Some(
      if (isSnapshot.value)
        Opts.resolver.sonatypeSnapshots
      else
        Opts.resolver.sonatypeStaging
    ),
    licenses:=Seq("MIT" -> url("https://opensource.org/licenses/MIT")),
    homepage:=Some(url("https://github.com/enriquegrodrigo/spark-crowd")),
    scmInfo := Some(ScmInfo(url("https://github.com/enriquegrodrigo/spark-crowd"), "scm:git@github.com:enriquegrodrigo/spark-crowd.git")),
    developers := List(
      Developer(
        id = "enrique.grodrigo",
        name="Enrique Gonzalez Rodrigo",
        email="mail@enriquegrodrigo.com",
        url = url("https://github.com/enriquegrodrigo"))),
    publishMavenStyle := true,
    parallelExecution in Test := false,
    libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.3.0" % "provided",
    libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.3.0" % "provided",
    libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.3.0" % "provided",
    libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.0",
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % "test"
  )
