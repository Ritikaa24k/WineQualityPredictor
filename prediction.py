import os
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

class WineQualityPredictor:
    def __init__(self, model_path, validation_data_path, feature_columns):
        self.model_path = model_path
        self.validation_data_path = validation_data_path
        self.feature_columns = feature_columns
        self.spark = self._initialize_spark_session()
        self.model = None
        self.validation_df = None
        self.predictions = None

    def _initialize_spark_session(self):
        return SparkSession.builder \
            .appName("WineQualityPrediction") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.access.key", os.environ['AWS_ACCESS_KEY_ID']) \
            .config("spark.hadoop.fs.s3a.secret.key", os.environ['AWS_SECRET_ACCESS_KEY']) \
            .config("spark.hadoop.fs.s3a.session.token", os.environ['AWS_SESSION_TOKEN']) \
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider") \
            .getOrCreate()

    def load_model(self):
        self.model = RandomForestClassificationModel.load(self.model_path)

    def load_validation_data(self):
        self.validation_df = self.spark.read.csv(
            self.validation_data_path, header=True, inferSchema=True)

    def prepare_features(self):
        assembler = VectorAssembler(
            inputCols=self.feature_columns, outputCol="features")
        self.validation_df = assembler.transform(self.validation_df)

    def make_predictions(self):
        self.predictions = self.model.transform(self.validation_df)

    def evaluate_model(self):
        evaluator = MulticlassClassificationEvaluator(
            labelCol="quality", predictionCol="prediction", metricName="f1")
        f1_score = evaluator.evaluate(self.predictions)
        print(f"F1 Score: {f1_score}")

    def stop_spark_session(self):
        self.spark.stop()

    def run(self):
        self.load_model()
        self.load_validation_data()
        self.prepare_features()
        self.make_predictions()
        self.evaluate_model()
        self.stop_spark_session()

if __name__ == "__main__":
    # Define the S3 paths and feature columns
    model_s3_path = "s3a://awsdatasetsbucket/trained/model"
    validation_s3_path = "s3a://awsdatasetsbucket/Cleaned_ValidationDataset.csv"
    features = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'chlorides',
        'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    ]

    # Create an instance of the predictor and run the prediction pipeline
    predictor = WineQualityPredictor(model_s3_path, validation_s3_path, features)
    predictor.run()