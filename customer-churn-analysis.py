from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# File path to save all outputs
output_file = "model_outputs.txt"

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges").cast("double")))

    categorical_cols = ["gender", "PhoneService", "InternetService"]
    indexers = [StringIndexer(inputCol=col, outputCol=col + "Index") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=col + "Index", outputCol=col + "Vec") for col in categorical_cols]

    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
    
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    encoded_features = [col + "Vec" for col in categorical_cols]
    feature_cols = numeric_cols + encoded_features

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    from pyspark.ml import Pipeline
    stages = indexers + encoders + [label_indexer, assembler]
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df)
    final_df = model.transform(df)
    return final_df.select("features", "label")

# Task 2: Train and Evaluate Logistic Regression Model
def train_logistic_regression_model(df, output_file):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression()
    model = lr.fit(train_df)
    predictions = model.transform(test_df)
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    with open(output_file, "a") as f:
        f.write("=== Logistic Regression ===\n")
        f.write(f"AUC: {auc:.4f}\n\n")

# Task 3: Feature Selection using Chi-Square Test
def feature_selection(df, output_file):
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    selected_df = selector.fit(df).transform(df)

    with open(output_file, "a") as f:
        f.write("=== Feature Selection (Chi-Square) ===\n")
        f.write("Top 5 selected features (first 5 rows):\n")
        for row in selected_df.select("selectedFeatures", "label").take(5):
            f.write(f"{row}\n")
        f.write("\n")

# Task 4: Hyperparameter Tuning and Model Comparison
def tune_and_compare_models(df, output_file):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GBTClassifier": GBTClassifier()
    }

    param_grids = {
        "LogisticRegression": ParamGridBuilder().addGrid(models["LogisticRegression"].regParam, [0.01, 0.1]).build(),
        "DecisionTree": ParamGridBuilder().addGrid(models["DecisionTree"].maxDepth, [3, 5]).build(),
        "RandomForest": ParamGridBuilder().addGrid(models["RandomForest"].numTrees, [10, 20]).build(),
        "GBTClassifier": ParamGridBuilder().addGrid(models["GBTClassifier"].maxIter, [10, 20]).build(),
    }

    best_auc = 0.0
    best_model_name = ""
    best_model = None

    with open(output_file, "a") as f:
        f.write("=== Model Tuning and Comparison ===\n")

        for name, model in models.items():
            cv = CrossValidator(estimator=model,
                                estimatorParamMaps=param_grids[name],
                                evaluator=evaluator,
                                numFolds=5)
            cv_model = cv.fit(train_df)
            auc = evaluator.evaluate(cv_model.transform(test_df))
            f.write(f"{name} AUC: {auc:.4f}\n")
            if auc > best_auc:
                best_auc = auc
                best_model_name = name
                best_model = cv_model.bestModel

        f.write(f"Best model: {best_model_name} with AUC = {best_auc:.4f}\n\n")

# Clear the output file if it already exists and add the header
with open(output_file, "w") as f:
    f.write("Customer Churn Modeling Report\n")
    f.write("==============================\n\n")

# Execute tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df, output_file)
feature_selection(preprocessed_df, output_file)
tune_and_compare_models(preprocessed_df, output_file)
