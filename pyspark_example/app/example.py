# your_script.py
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("PySpark Example").getOrCreate()

# Create a simple DataFrame
data = [("John", 30), ("Jane", 25), ("Sam", 35)]
df = spark.createDataFrame(data, ["Name", "Age"])

# Show the DataFrame
df.show()

# Stop the Spark session
spark.stop()