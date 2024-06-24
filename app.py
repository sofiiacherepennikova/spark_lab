import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import time
import psutil
import matplotlib.pyplot as plt
 
def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

time_data = []
memory_data = []
 
spark = (
    SparkSession
    .builder
    .appName('ny_taxi')
    .getOrCreate()
)
 
print("Reading data from HDFS")
start_time = time.time()
data = spark.read.csv("hdfs://namenode:9001/data/ny_taxi/combined_ny_taxi.csv", inferSchema=True, header=True)
data.show(5)
time_to_read = time.time() - start_time
time_data.append(time_to_read)
memory_data.append(log_memory_usage())
print(f"Time to read data: {time_to_read:.2f} seconds")
print(f"Memory usage: {memory_data[-1]:.2f} MB")

print("Preparing data for the model")
feature_columns = [col for col in data.columns if col != 'trip_duration']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).select("features", "trip_duration")

train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
print(f"Training data: {train_data.count()} rows, Test data: {test_data.count()} rows")

print("Training the linear regression model")
lr = LinearRegression(labelCol="trip_duration", featuresCol="features")
start_time = time.time()
lr_model = lr.fit(train_data)
time_to_train = time.time() - start_time
time_data.append(time_to_train)
memory_data.append(log_memory_usage())
print(f"Time to train the model: {time_to_train:.2f} seconds")
print(f"Memory usage: {memory_data[-1]:.2f} MB")

print("Evaluating the model on test data")
start_time = time.time()
test_results = lr_model.evaluate(test_data)
time_to_evaluate = time.time() - start_time
time_data.append(time_to_evaluate)
memory_data.append(log_memory_usage())
print(f"Time to evaluate the model: {time_to_evaluate:.2f} seconds")
print(f"RMSE: {test_results.rootMeanSquaredError}, R2: {test_results.r2}")
print(f"Memory usage: {memory_data[-1]:.2f} MB")
 
spark.stop()

plt.figure(figsize=(12, 5))
 
plt.subplot(1, 2, 1)
plt.hist(time_data, bins=10, color='blue', edgecolor='black')
plt.title('Histogram of Time Distribution')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
 
plt.subplot(1, 2, 2)
plt.hist(memory_data, bins=10, color='green', edgecolor='black')
plt.title('Histogram of RAM Distribution')
plt.xlabel('Memory Usage (MB)')
plt.ylabel('Frequency')
 
plt.tight_layout()
plt.savefig("/app/performance_histograms.png")
plt.show()
 