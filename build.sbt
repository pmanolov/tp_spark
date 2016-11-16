name := "tp_spark"
version := "1.0"
scalaVersion := "2.11.4"
organization := "paristech"
libraryDependencies ++= Seq(
  // Spark dependencies. Marked as provided because they must not be included in the uberjar
  "org.apache.spark" %% "spark-core" % "2.0.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "2.0.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "2.0.0" % "provided"
)


// A special option to exclude Scala itself form our assembly JAR, since Spark already bundles Scala.
assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
// Disable parallel execution because of spark-testing-base
parallelExecution in Test := false
// Configure the build to publish the assembly JAR
artifact in (Compile, assembly) := {
  val art = (artifact in (Compile, assembly)).value
  art.copy(`classifier` = Some("assembly"))
}
addArtifact(artifact in (Compile, assembly), assembly)



    