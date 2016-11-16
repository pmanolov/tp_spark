package com.sparkProject

import org.apache.spark.sql.SparkSession

/*import org.apache.spark.sql.functions */

/* Import org.apache.spark.sql.functions._ */


/* a faire importer librairie ML */


object Job {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      //.master("local")
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    import spark.implicits._


    /********************************************************************************
      *
      *        TP 1
      *
      *        - Set environment, InteliJ, submit jobs to Spark
      *        - Load local unstructured data
      *        - Word count , Map Reduce
      ********************************************************************************/



    // ----------------- word count ------------------------

    val df_wordCount = sc.textFile("/Users/papa/MS/spark-2.0.0-bin-hadoop2.7/README.md")
      .flatMap{case (line: String) => line.split(" ")}
      .map{case (word: String) => (word, 1)}
      .reduceByKey{case (i: Int, j: Int) => i + j}
      .toDF("word", "count")

    df_wordCount.orderBy($"count".desc).show()


    /********************************************************************************
      *
      *        TP 2 : début du projet
      *
      ********************************************************************************/
    /* charger un fichier csv */
    /*val myDF = sqlContext.csvFile("/Users/papa/MS/tp_spark/data/cumulative.csv", true, ',')
    */
    val df = spark
      .read // returns a DataFrameReader, giving access to methods “options” and “csv”
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .option("comment", "#") // All lines starting with # are ignored
      .csv("/Users/papa/MS/tp_spark/data/cumulative.csv")

    /*val df = spark.read.option("comment", "#").option("header",true).csv("/Users/papa/MS/tp_spark/data/cumulative.csv")
    df.colums*/
    println("/*******************************************************************************/")

    println("number of columns", df.columns.length) //df.columns returns an Array of columns names, and arrays have a method “length” returning their length
    println("number of rows", df.count)
    println("/*******************************************************************************/")
    df.show()
    println("/*******************************************************************************/")

    import org.apache.spark.sql.functions._
    val columns = df.columns.slice(10, 20) // df.columns returns an Array. In scala arrays have a method “slice” returning a slice of the array

    println("/*******************************************************************************/")
    println("Afficher le dataFrame sous forme de table.")
    df.select(columns.map(col): _*).show(50) //Afficher le dataFrame sous forme de table.

    println("/*******************************************************************************/")
    println("afficher le schema")
    df.printSchema()  // afficher le schema

    println("/*******************************************************************************/")
    println("Classification: Afficher le nombre d’éléments de chaque classe (colonne koi_disposition).")
    df.groupBy("koi_disposition").count().orderBy($"count".desc).show()




    println("/*******************************************************************************/")
    println("Conserver uniquement les lignes qui nous intéressent pour le modèle (koi_disposition = CONFIRMED ou FALSE POSITIVE")

    //Plusieurs syntaxes :
    /*val cdf = df.where("koi_disposition = 'CONFIRMED' or koi_disposition = 'FALSE POSITIVE'")

    OU
    val cdf = df.filter(df.col("koi_disposition") === "CONFIRMED" || df.col("koi_disposition") === "FALSE POSITIVE")

    OU
    val cdf = df.filter($"koi_disposition" === "CONFIRMED" || $"koi_disposition" === "FALSE POSITIVE")

    OU
    val df2 = df.filter(!($"koi_disposition" === "CANDIDATE"))
    */


    val df_filtered = df.where("koi_disposition = 'CONFIRMED' OR koi_disposition = 'FALSE POSITIVE'")
    df_filtered.groupBy("koi_disposition").count().orderBy($"count".desc).show()
    df_filtered.show()

    println("/*******************************************************************************/")
    println("Afficher le nombre d’éléments distincts dans la colonne “koi_eccen_err1”. ")

    df_filtered.groupBy("koi_eccen_err1").count().show()
    df_filtered.select("koi_eccen_err1").distinct().show()
    //df_filtered.show()

    println("/*******************************************************************************/")
    println("Enlever la colonne “koi_eccen_err1")

    df_filtered.drop("koi_eccen_err1")
    df_filtered.select(columns.map(col): _*).show(5)

    println("/*******************************************************************************/")
    println(" suppression des colonnes unitiles")

    val df_cleaned = df_filtered.drop("index","kepid","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec",
      "koi_sparprov","koi_trans_mod","koi_datalink_dvr","koi_datalink_dvs","koi_tce_delivname",
      "koi_parm_prov","koi_limbdark_mod","koi_fittype","koi_disp_prov","koi_comment","kepoi_name","kepler_name", "koi_vet_date","koi_pdisposition")

    println("/*******************************************************************************/")
    println(" suite .  ..")


    /* utiliser import org.apache.spark.sql.functions._   */

    val useless_column = df_cleaned.columns.filter{ case (column:String) =>
      df_cleaned.agg(countDistinct(column)).first().getLong(0) <= 1 }




    println("/*******************************************************************************/")
    println(" f. afficher des statistiques sur les colonnes du dataFrame" )
    df_cleaned.describe("koi_impact", "koi_duration").show


    println("/*******************************************************************************/")
    println(" g. remplacer toutes les valeurs manquantes par zéro ")
    val df_filled = df_cleaned.na.fill(0.0)


    println("/*******************************************************************************/")
    println(" 5 Joindre deux dataFrames ")
    println(" preparer les données")
    val df_labels = df_filled.select("rowid", "koi_disposition")
    val df_features = df_filled.drop("koi_disposition")



    println("/*******************************************************************************/")
    println(" a. joindre df_labels et df_features dans un seul dataFrame. ")


    val df_joined = df_features.join(df_labels, usingColumn = "rowid")




    println("/*******************************************************************************/")
    println(" Ajouter et manipuler des colonnes. ")

  //def udf_sum = udf((col1: Double, col2: Double) => col1 + col2)

 //   val df_newFeatures = df_joined
 //     .withColumn("koi_ror_min", udf_sum($"koi_ror", $"koi_ror_err2"))
 //     .withColumn("koi_ror_max", $"koi_ror" + $"koi_ror_err1")


    println("/*******************************************************************************/")
    println(" TP 4/5 ")

    // fichier nettoyé pour l'exo

    val df_final = spark.read.parquet("/Users/papa/MS/tp_spark/data/fichier.parquet")



  }


}
