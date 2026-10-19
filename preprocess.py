

from pathlib import Path



from pyspark.sql import SparkSession

from pyspark.sql import functions as F

from pyspark.sql.types import NumericType



from pyspark.ml import Pipeline

from pyspark.ml.feature import (

    Imputer,

    StringIndexer,

    OneHotEncoder,

    VectorAssembler,

    StandardScaler,

)





# --------------------------------------------------

# 1. PROJECT PATHS AND CONFIGURATION

# --------------------------------------------------



BASE_DIR = Path(__file__).resolve().parent



DATA_DIR = BASE_DIR / "data"

TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.parquet"

TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.parquet"



OUTPUT_DIR = DATA_DIR / "processed_parquet"

TRAIN_OUTPUT = OUTPUT_DIR / "train"

TEST_OUTPUT = OUTPUT_DIR / "test"



MODEL_DIR = BASE_DIR / "saved_models" / "preprocessing_pipeline"



CATEGORICAL_CANDIDATES = ["proto", "service", "state"]



# These columns must never become model input features.

EXCLUDED_FEATURES = {

    "id",

    "label",

    "attack_cat",

    "is_attack",

    "attack_type",

}





def spark_path(path: Path) -> str:

    """Convert a local path to a file URI for Spark."""

    return path.resolve().as_uri()





# --------------------------------------------------

# 2. CLEAN DATA AND CREATE TARGET COLUMNS

# --------------------------------------------------



def clean_dataframe(df, dataset_name: str):

    """Clean one dataset without learning preprocessing statistics."""



    required_columns = {"label", "attack_cat"}

    missing = required_columns - set(df.columns)



    if missing:

        raise ValueError(

            f"{dataset_name} is missing required columns: {sorted(missing)}"

        )



    # Standardize categorical feature values.

    for column_name in CATEGORICAL_CANDIDATES:

        if column_name in df.columns:

            value = F.lower(

                F.trim(F.col(column_name).cast("string"))

            )



            df = df.withColumn(

                column_name,

                F.when(

                    value.isNull() | (value == ""),

                    F.lit("unknown"),

                ).otherwise(value),

            )



    # Clean the attack-category target separately.

    attack_category = F.lower(

        F.trim(F.col("attack_cat").cast("string"))

    )



    df = df.withColumn(

        "attack_cat",

        F.when(

            attack_category.isNull() | (attack_category == ""),

            F.lit(None).cast("string"),

        ).otherwise(attack_category),

    )



    # Remove null, NaN and infinite values from numeric columns.

    # Replace them with null first; Imputer will handle numeric nulls.

    numeric_fields = [

        field.name

        for field in df.schema.fields

        if isinstance(field.dataType, NumericType)

        and field.name != "label"

    ]



    for column_name in numeric_fields:

        value = F.col(column_name).cast("double")



        invalid = (

            value.isNull()

            | F.isnan(value)

            | (value == float("inf"))

            | (value == float("-inf"))

        )



        df = df.withColumn(

            column_name,

            F.when(

                invalid,

                F.lit(None).cast("double"),

            ).otherwise(value),

        )



    # Stage 1 target: 0 = normal, 1 = attack.

    df = df.withColumn(

        "is_attack",

        F.col("label").cast("int"),

    )



    # Validate binary labels.

    invalid_labels = df.filter(

        F.col("is_attack").isNull()

        | (~F.col("is_attack").isin(0, 1))

    )



    if invalid_labels.limit(1).count() > 0:

        raise ValueError(

            f"{dataset_name} contains missing or invalid binary labels."

        )



    # Stage 2 target: the attack category.

    # Normal traffic receives "Normal"; missing attack labels

    # on malicious rows receive "Unknown".

    category = F.col("attack_cat")

    category_lower = F.lower(category)



    df = df.withColumn(

        "attack_type",

        F.when(

            F.col("is_attack") == 0,

            F.lit("Normal"),

        )

        .when(

            category.isNull()

            | (category == "")

            | category_lower.isin("normal", "nan", "none"),

            F.lit("Unknown"),

        )

        .when(

            category_lower == "dos",

            F.lit("DoS"),

        )

        .otherwise(F.initcap(category)),

    )



    # Remove exact duplicate records within this dataset.

    df = df.dropDuplicates()



    print(f"{dataset_name}: cleaning completed.")

    return df





# --------------------------------------------------

# 3. BUILD AND FIT THE PREPROCESSING PIPELINE

# --------------------------------------------------



def run_preprocessing():



    # Fail early if the expected local datasets are absent.

    for path in (TRAIN_PATH, TEST_PATH):

        if not path.is_file():

            raise FileNotFoundError(

                f"Dataset not found: {path}\n"

                "Place the original Parquet file inside the data folder."

            )



    spark = (

        SparkSession.builder

        .appName("NetworkIntrusionDetectionPreprocessing")

        .master("local[*]")

        .config("spark.driver.memory", "4g")

        .config("spark.sql.shuffle.partitions", "8")

        .config(

            "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version",

            "1",

        )

        .getOrCreate()

    )



    spark.sparkContext.setLogLevel("ERROR")



    try:

        print("Spark session initialized successfully.")



        # Read the existing Parquet files directly with Spark.

        train_raw = spark.read.parquet(spark_path(TRAIN_PATH))

        test_raw = spark.read.parquet(spark_path(TEST_PATH))



        # Both datasets must have the columns needed by the pipeline.

        missing_test_columns = set(train_raw.columns) - set(test_raw.columns)



        if missing_test_columns:

            raise ValueError(

                "Testing data is missing training columns: "

                f"{sorted(missing_test_columns)}"

            )



        # Use the training schema as the reference.

        test_raw = test_raw.select(*train_raw.columns)



        print("Training and testing datasets loaded.")



        # Clean the splits independently; never fit on combined data.

        train_clean = clean_dataframe(train_raw, "Training dataset")

        test_clean = clean_dataframe(test_raw, "Testing dataset")



        # Discover numerical features from the training schema.

        numeric_cols = [

            field.name

            for field in train_clean.schema.fields

            if isinstance(field.dataType, NumericType)

            and field.name not in EXCLUDED_FEATURES

            and field.name not in CATEGORICAL_CANDIDATES

        ]



        categorical_cols = [

            column_name

            for column_name in CATEGORICAL_CANDIDATES

            if column_name in train_clean.columns

        ]



        if not numeric_cols:

            raise ValueError("No numerical input features were found.")



        if not categorical_cols:

            raise ValueError("No categorical input features were found.")



        print(f"Numerical feature count: {len(numeric_cols)}")

        print(f"Categorical features: {categorical_cols}")



        stages = []



        # Impute missing numeric values using medians learned

        # exclusively from the training dataset.

        imputed_cols = [

            f"{column_name}__imputed"

            for column_name in numeric_cols

        ]



        imputer = Imputer(

            inputCols=numeric_cols,

            outputCols=imputed_cols,

            strategy="median",

        )



        stages.append(imputer)



        # Convert categorical strings into numeric indices.

        indexed_cols = []



        for column_name in categorical_cols:

            indexed_name = f"{column_name}__idx"



            indexer = StringIndexer(

                inputCol=column_name,

                outputCol=indexed_name,

                handleInvalid="keep",

            )



            stages.append(indexer)

            indexed_cols.append(indexed_name)



        # One-hot encode categories to avoid treating category IDs

        # as if they had an ordered numerical relationship.

        encoded_cols = [

            f"{column_name}__ohe"

            for column_name in categorical_cols

        ]



        encoder = OneHotEncoder(

            inputCols=indexed_cols,

            outputCols=encoded_cols,

            handleInvalid="keep",

            dropLast=False,

        )



        stages.append(encoder)



        # Combine numerical and categorical vectors into one feature vector.

        assembler = VectorAssembler(

            inputCols=imputed_cols + encoded_cols,

            outputCol="raw_features",

            handleInvalid="error",

        )



        stages.append(assembler)



        # Standardize numerical scales without centering sparse vectors.

        scaler = StandardScaler(

            inputCol="raw_features",

            outputCol="features",

            withStd=True,

            withMean=False,

        )



        stages.append(scaler)



        pipeline = Pipeline(stages=stages)



        print("Fitting preprocessing pipeline on training data only...")



        # Critical: fit ONLY on the training dataset.

        pipeline_model = pipeline.fit(train_clean)



        print("Training preprocessing pipeline fitted.")



        # Apply the same fitted transformations to both splits.

        train_processed = pipeline_model.transform(train_clean)

        test_processed = pipeline_model.transform(test_clean)



        # Keep the agreed interface for downstream team members.

        # Preserve the record ID when available for traceability.

        output_columns = []



        if "id" in train_processed.columns:

            output_columns.append("id")



        output_columns.extend(

            ["features", "is_attack", "attack_type"]

        )



        train_processed = train_processed.select(*output_columns)

        test_processed = test_processed.select(*output_columns)



        # Write using Spark; do not collect the entire dataset in Pandas.

        train_processed.write.mode("overwrite").parquet(

            spark_path(TRAIN_OUTPUT)

        )



        test_processed.write.mode("overwrite").parquet(

            spark_path(TEST_OUTPUT)

        )



        # Save the fitted preprocessing model for reuse.

        pipeline_model.write().overwrite().save(

            spark_path(MODEL_DIR)

        )



        print("\nPREPROCESSING COMPLETED SUCCESSFULLY")

        print(f"Processed training data: {TRAIN_OUTPUT}")

        print(f"Processed testing data:  {TEST_OUTPUT}")

        print(f"Saved preprocessing model: {MODEL_DIR}")



        print("\nTraining output schema:")

        train_processed.printSchema()



    finally:

        spark.stop()





if __name__ == "__main__":

    run_preprocessing()
