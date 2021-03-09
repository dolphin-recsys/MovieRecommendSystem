package com.recsys

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}

object Als {

    def main(args: Array[String]): Unit = {
        val conf: SparkConf = new SparkConf()
            .setMaster("local")
            .setAppName("CF")

        val spark: SparkSession = SparkSession.builder.config(conf).getOrCreate()
        val ratingResourcesPath = "/Users/kris/Learn/AIs/recommendationsystem/2020xiaoxiang/projectlast/MovieRecommendSystem/dataset/dataset1/user_movie.csv"
        val toInt: UserDefinedFunction = udf[Int, String]( _.toInt)
        val toFloat: UserDefinedFunction = udf[Double, String]( _.toFloat)
        val ratingSamples: DataFrame = spark.read.format("csv").option("header", "true").load(ratingResourcesPath)
            .withColumn("userIdInt", toInt(col("用户ID")))
            .withColumn("movieIdInt", toInt(col("电影ID")))
            .withColumn("ratingFloat", toFloat(col("评分")))

        val Array(training, test) = ratingSamples.randomSplit(Array(0.8, 0.2))

        // Build the recommendation model using ALS on the training data
        val als: ALS = new ALS()
            .setMaxIter(10)
            .setRegParam(0.01)
            .setUserCol("userIdInt")
            .setItemCol("movieIdInt")
            .setRatingCol("ratingFloat")

        val model: ALSModel = als.fit(training)

        // Evaluate the model by computing the RMSE on the test data
        // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
        model.setColdStartStrategy("drop")
        val predictions: DataFrame = model.transform(test)

        // print the latent factors of user and item
        model.itemFactors.show(10, truncate = false)
        model.userFactors.show(10, truncate = false)

        val evaluator: RegressionEvaluator = new RegressionEvaluator()
            .setMetricName("rmse")
            .setLabelCol("ratingFloat")
            .setPredictionCol("prediction")
        // evaluate the model on predictions
        val rmse: Double = evaluator.evaluate(predictions)

        println(s"Root-mean-square error = $rmse")

        // Generate top 10 movie recommendations for each user
        val userRecs: DataFrame = model.recommendForAllUsers(10)
        // Generate top 10 user recommendations for each movie
        val movieRecs: DataFrame = model.recommendForAllItems(10)

        // Generate top 10 movie recommendations for a specified set of users
        val users: Dataset[Row] = ratingSamples.select(als.getUserCol).distinct().limit(3)

        val userSubsetRecs: DataFrame = model.recommendForUserSubset(users, 10)

        // Generate top 10 user recommendations for a specified set of movies
        val movies: Dataset[Row] = ratingSamples.select(als.getItemCol).distinct().limit(3)
        val movieSubSetRecs: DataFrame = model.recommendForItemSubset(movies, 10)

        // print the result
        userRecs.show(false)
        movieRecs.show(false)
        userSubsetRecs.show(false)
        movieSubSetRecs.show(false)

        val paramGrid: Array[ParamMap] = new ParamGridBuilder()
            .addGrid(als.regParam, Array(0.01))
            .build()

        val cv: CrossValidator = new CrossValidator()
            .setEstimator(als)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(10)

        val cvModel = cv.fit(test)

        val avgMetrics: Array[Double] = cvModel.avgMetrics
        print(avgMetrics)

        spark.stop()
    }

}
