# Databricks notebook source
# MAGIC %pip install mlflow

# COMMAND ----------

# MAGIC %sh
# MAGIC # Download flights.csv to /tmp
# MAGIC wget -O /tmp/flights.csv "https://raw.githubusercontent.com/Rufinadcosta/Flight-Delay-Prediction-Trend-Analysis-with-Databricks/main/flights.csv"
# MAGIC
# MAGIC # Download weather.csv to /tmp
# MAGIC wget -O /tmp/weather.csv "https://raw.githubusercontent.com/Rufinadcosta/Flight-Delay-Prediction-Trend-Analysis-with-Databricks/main/weather.csv"
# MAGIC

# COMMAND ----------

# Copy from /tmp to DBFS
dbutils.fs.cp("file:/tmp/flights.csv", "dbfs:/FileStore/tables/flights.csv")
dbutils.fs.cp("file:/tmp/weather.csv", "dbfs:/FileStore/tables/weather.csv")


# COMMAND ----------

df_flights = spark.read.csv("/FileStore/tables/flights.csv", header=True, inferSchema=True)
df_weather = spark.read.csv("/FileStore/tables/weather.csv", header=True, inferSchema=True)

display(df_flights)
display(df_weather)


# COMMAND ----------

import json
import random
from datetime import datetime, timedelta

# Clickstream config
pages = ["home", "search_results", "flight_details", "checkout", "confirmation"]
actions = ["view", "click", "filter_applied", "sort_applied", "add_to_cart", "book"]
airports = ["JFK", "LAX", "ORD", "ATL", "DFW", "DEN"]
flight_numbers = [f"AA{random.randint(100, 999)}" for _ in range(100)]

records = []
base_time = datetime(2025, 5, 20, 8, 0, 0)

# Generate 5000 records
for i in range(5000):
    session_id = f"sess_{random.randint(1000, 9999)}"
    user_id = f"user_{random.randint(100, 999)}"
    timestamp = base_time + timedelta(seconds=random.randint(0, 360000))
    page = random.choice(pages)
    action = random.choice(actions)
    duration = random.randint(1, 120)
    origin = random.choice(airports)
    destination = random.choice([a for a in airports if a != origin])
    flight = random.choice(flight_numbers)

    record = {
        "session_id": session_id,
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "page": page,
        "action": action,
        "duration_seconds": duration,
        "flight_selected": flight,
        "origin": origin,
        "destination": destination
    }
    records.append(record)

# Save to temp local file in JSON lines format
local_path = "/tmp/clickstream_data.json"
with open(local_path, "w") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")

# Upload to DBFS
dbfs_path = "dbfs:/FileStore/clickstream_data.json"
dbutils.fs.cp(f"file:{local_path}", dbfs_path)



# COMMAND ----------

df_clickstream = spark.read.json("dbfs:/FileStore/clickstream_data.json")
df_clickstream.display()



# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/tables/"))


# COMMAND ----------

# MAGIC %md
# MAGIC 1. Data Ingestion (ETL Part)
# MAGIC Datasets:
# MAGIC
# MAGIC airline_data.csv – On-time performance
# MAGIC
# MAGIC weather_data.csv – Weather by airport
# MAGIC
# MAGIC Simulated clickstream_data.json – Booking site behavior (optional)
# MAGIC
# MAGIC Storage Target:
# MAGIC
# MAGIC Upload to Azure Blob Storage or DBFS
# MAGIC
# MAGIC Save raw data as Delta tables

# COMMAND ----------

#Data Ingestion (ETL Part)
#Save raw data as Delta tables
# Airline and Weather
df_flights = spark.read.csv("/FileStore/tables/flights.csv", header=True, inferSchema=True)
df_weather = spark.read.csv("/FileStore/tables/weather.csv", header=True, inferSchema=True)

# Optional Clickstream
df_clickstream = spark.read.json("dbfs:/FileStore/clickstream_data.json")

# Store raw data as Delta
df_flights.write.format("delta").mode("overwrite").save("/delta/raw/flights")
df_weather.write.format("delta").mode("overwrite").save("/delta/raw/weather")


# COMMAND ----------

# MAGIC %md
# MAGIC 2. Data Cleaning & Transformation
# MAGIC Tasks:
# MAGIC
# MAGIC Join airline + weather datasets on airport/date
# MAGIC
# MAGIC Handle nulls, encode categorical vars
# MAGIC
# MAGIC Create features:
# MAGIC
# MAGIC Day of week
# MAGIC
# MAGIC Part of day (morning/evening/night)
# MAGIC
# MAGIC Holiday indicator

# COMMAND ----------

from pyspark.sql.functions import col, dayofweek, hour, when, to_timestamp, lpad, concat, substring, lit

# Load data
flights = spark.read.format("delta").load("/delta/raw/flights")
weather = spark.read.format("delta").load("/delta/raw/weather")

# Format date
flights = flights.withColumn("FL_DATE", to_timestamp("FL_DATE", "yyyy-MM-dd"))

# Rename for join compatibility
weather = weather.withColumnRenamed("DATE", "FL_DATE") \
                 .withColumnRenamed("AIRPORT", "ORIGIN")

# Join
joined_df = flights.join(weather, on=["FL_DATE", "ORIGIN"], how="left")

# Convert CRS_DEP_TIME to timestamp
dep_time_padded = lpad(col("CRS_DEP_TIME").cast("string"), 4, "0")
dep_time_string = concat(
    col("FL_DATE").cast("date").cast("string"), 
    lit(" "), 
    substring(dep_time_padded, 1, 2), 
    lit(":"), 
    substring(dep_time_padded, 3, 2)
)
dep_time_ts = to_timestamp(dep_time_string, "yyyy-MM-dd HH:mm")

# Feature engineering
df = joined_df.withColumn("DepTimeTS", dep_time_ts) \
              .withColumn("Hour", hour("DepTimeTS")) \
              .withColumn("DayOfWeek", dayofweek("FL_DATE")) \
              .withColumn("PartOfDay", when(col("Hour") < 6, "Night")
                                         .when(col("Hour") < 12, "Morning")
                                         .when(col("Hour") < 18, "Afternoon")
                                         .otherwise("Evening"))

# Save
df.write.format("delta").mode("overwrite").save("/delta/cleaned/flights_weather")





# COMMAND ----------

# MAGIC %md
# MAGIC Validating if the data is correct

# COMMAND ----------

df.printSchema()
print(f"Total records: {df.count()}")


# COMMAND ----------

from pyspark.sql.functions import col, count, when

# Separate numeric and non-numeric columns for correct validation
null_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])

null_counts.show(truncate=False)



# COMMAND ----------

df.select("PartOfDay").distinct().show()
df.select("OP_CARRIER").distinct().show()
df.select("WEATHER_COND").distinct().show()


# COMMAND ----------

df.select("DEP_DELAY", "ARR_DELAY", "DISTANCE", "TEMP_C").describe().show()


# COMMAND ----------

df.select("CRS_DEP_TIME", "DepTimeTS", "Hour").show(5, truncate=False)


# COMMAND ----------

df.selectExpr("min(FL_DATE)", "max(FL_DATE)").show()


# COMMAND ----------

# MAGIC %md
# MAGIC 3. Exploratory Data Analysis (EDA)
# MAGIC Visual Analysis:
# MAGIC
# MAGIC Top delayed airlines/airports
# MAGIC
# MAGIC Delay trends by month/hour
# MAGIC
# MAGIC Use Databricks display() or matplotlib for plots

# COMMAND ----------

df = spark.read.format("delta").load("/delta/cleaned/flights_weather")

# Top delayed airlines
df.groupBy("OP_CARRIER").avg("ARR_DELAY").orderBy("avg(ARR_DELAY)", ascending=False).show()

# Hourly delay trend
from pyspark.sql.functions import hour, avg
df.groupBy(hour("DepTimeTS").alias("Hour")) \
  .agg(avg("ARR_DELAY").alias("avg_ARR_DELAY")) \
  .orderBy("Hour") \
  .show()


# COMMAND ----------

# MAGIC %md
# MAGIC 4. Machine Learning with MLlib
# MAGIC Goal: Predict if ARR_DELAY > 15

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark

# Label
df = df.withColumn("Delayed", when(col("ARR_DELAY") > 15, 1).otherwise(0))

# Preprocessing
indexer = StringIndexer(inputCol="PartOfDay", outputCol="PartOfDay_Idx")
features = ["DEP_DELAY", "DISTANCE", "DayOfWeek", "PartOfDay_Idx"]
assembler = VectorAssembler(inputCols=features, outputCol="features")

lr = LogisticRegression(labelCol="Delayed", featuresCol="features")

pipeline = Pipeline(stages=[indexer, assembler, lr])

train, test = df.randomSplit([0.8, 0.2], seed=42)

with mlflow.start_run():
    model = pipeline.fit(train)
    predictions = model.transform(test)

    evaluator = BinaryClassificationEvaluator(labelCol="Delayed")
    auc = evaluator.evaluate(predictions)

    mlflow.log_metric("AUC", auc)
    mlflow.spark.log_model(model, "flight_delay_model")


# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE delta.`/delta/streaming/flights` (
# MAGIC   FL_DATE timestamp,
# MAGIC   OP_CARRIER string,
# MAGIC   ARR_DELAY double
# MAGIC   -- Add other columns as needed, no trailing comma on last column
# MAGIC )
# MAGIC USING delta
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 5. Real-Time/Batch Simulation (Optional Bonus)
# MAGIC Simulate micro-batch streaming from a folder

# COMMAND ----------

stream_df = spark.readStream \
    .format("delta") \
    .load("/delta/streaming/flights")  # Simulated append in background

agg_df = stream_df.groupBy("OP_CARRIER").avg("ARR_DELAY")

query = agg_df.writeStream.outputMode("complete").format("console").start()
