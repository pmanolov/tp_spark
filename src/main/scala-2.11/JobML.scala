package com.sparkProject
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.{StringIndexer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator}
import org.apache.spark.ml.classification.LogisticRegression



import org.apache.spark.ml



/**
  *
  * Created by papa on 28/10/2016.
  */
object JobML {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      //.master("local")
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext
    import spark.implicits._

    println("/*******************************************************************************/")
    println("\tTP 4/5 Mastère Specialisé BigData 2016/2017")
    println("\t Plamen MANOLOV")
    println("/*******************************************************************************/")
    println("\t1) Mettre les données sous une forme utilisable par Spark.ML.")

    // fichier nettoyé pour l'exo
    val df = spark.read.parquet("/Users/papa/MS/tp_spark/data/cleanedDataFrame.parquet")
    println("\n/*******************************************************************************/")
    println("\t- fichier de données remonté en DF : cleanedDataFrame.parquet")
    println("\t- Nombre de lignes   : " + df.count)
    println("\t- Nombre de colonnes : " + df.columns.length)


    //Ne pas mettre les colonnes “koi_disposition” et “rowid” dans les features
    println("\n/*******************************************************************************/")
    println("\t- supression des colinnes rowid et koi_disposition")
    val cols = df.columns.filter(_ != "rowid").filter(_ != "koi_disposition")

    println("\n/*******************************************************************************/")
    println("\t- création de Vector Assembler ")
    val assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")

    println("\n/*******************************************************************************/")
    println("\t- transfomation du DF en Vector Assembler ")
    val df_vector = assembler.transform(df)
    //println(df_vector.select("features", "koi_disposition").first())


    println("\n/*******************************************************************************/")
    println("\t- centralisation et normalisation des données ")
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)
    val scalerModel = scaler.fit(df_vector)
    val scaledData = scalerModel.transform(df_vector)
    scaledData.drop("features")
    //println(scaledData.select("scaledFeatures", "koi_disposition").first())


    println("\n/*******************************************************************************/")
    println("\t- indexation des données ")
    val indexer = new StringIndexer()
      .setInputCol("koi_disposition")
      .setOutputCol("label")
      .fit(scaledData)
    val indexedScaledData = indexer.transform(scaledData)



    println("\n/*******************************************************************************/")
    println("\tMachine Learning")
    println("\n/*******************************************************************************/")
    println("\t- splitte des données en 'training' set et 'test' et")
    val Array(training, test) = indexedScaledData.randomSplit(Array(0.9, 0.1))


    println("\n/*******************************************************************************/")
    println("\t- entraînement du classifieur et réglage des hyper-paramètres de l’algorithme")


    println("\n/*******************************************************************************/")
    println("\t- preparation du model 'Logistic Regression'")
    val modelLR = new LogisticRegression()

    modelLR.setElasticNetParam(1.0)  // L1-norm regularization : LASSO
      .setLabelCol("label")
      .setStandardization(true)  // to scale each feature of the model
      .setFitIntercept(true)  // we want an affine regression (with false, it is a linear regression)
      .setTol(1.0e-5)  // stop criterion of the algorithm based on its convergence
      .setMaxIter(300)  // a security stop criterion to avoid infinite loops

    // Créer une grille de valeurs à tester pour les hyper-paramètres.
    println("\n/*******************************************************************************/")
    println("\t- création d'une Grid array de -6 à 1 avec pas de 0.5")
    val array = -6.0 to (0.0, 0.5) toArray
    val arrayLog = array.map(x => math.pow(10,x))
    val paramGrid = new ParamGridBuilder()
      .addGrid(modelLR.regParam, arrayLog)
      .build()

    // Creating the BinaryClassificationEvaluator
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")

    println("\n/*******************************************************************************/")
    println("\t- splitte des données en 70, 30")
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(modelLR)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    println("\n/*******************************************************************************/")
    println("\t- execution du modèle")
    val model = trainValidationSplit.fit(training)

    println("\n/*******************************************************************************/")
    println("\t- prediction sur les données test")
    val df_WithPredictions = model.transform(test).select("features", "label", "prediction")
    df_WithPredictions.show()
    df_WithPredictions.groupBy("label", "prediction").count.show()


    //Afficher un score permettant d’évaluer la pertinence du modèle sur les données test.
    println("\n/*******************************************************************************/")
    println("\t- Score Final : ")
    evaluator.setRawPredictionCol("prediction")
    println(evaluator.evaluate(df_WithPredictions))

    //Sauvegarder le modèle entraîné pour pouvoir le réutiliser plus tard
    println("\n/*******************************************************************************/")
    println("\t- export du modèle vers : /Users/papa/MS/tp_spark/data/model ")
    sc.parallelize(Seq(model), 1).saveAsObjectFile("/Users/papa/MS/tp_spark/data/model")


  }
}
