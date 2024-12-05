import os
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def predict_model(model_path, validation_dataset_path, feature_columns):
    spark = SparkSession.builder \
        .appName("WineQualityModelPrediction") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", os.environ['AWS_ACCESS_KEY_ID']) \
        .config("spark.hadoop.fs.s3a.secret.key", os.environ['AWS_SECRET_ACCESS_KEY']) \
        .config("spark.hadoop.fs.s3a.session.token", os.environ['AWS_SESSION_TOKEN']) \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider") \
        .getOrCreate()

    rfc_model = RandomForestClassificationModel.load(model_path)

    validation_df = spark.read.csv(validation_dataset_path, header=True, inferSchema=True)

    vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    validation_df = vector_assembler.transform(validation_df)

    predictions = rfc_model.transform(validation_df)

    model_evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_score = model_evaluator.evaluate(predictions)

    print("F1 Score:", f1_score)

    spark.stop()

if __name__ == "__main__":
    s3_path_model = "s3a://awsdatasetsbucket/trained/predict_model"
    validation_s3_path = "s3a://awsdatasetsbucket/Cleaned_ValidationDataset.csv"

    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides',
                       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    predict_model(s3_path_model, validation_s3_path, feature_columns)
