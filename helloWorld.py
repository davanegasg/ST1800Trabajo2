from pyspark.sql import SparkSession

# Inicializar SparkSession
spark = SparkSession.builder \
    .appName("Test PySpark") \
    .getOrCreate()

# Test simple
df = spark.createDataFrame([("Hello, world!",)], ["text"])
df.show()

# Detener SparkSession
spark.stop()