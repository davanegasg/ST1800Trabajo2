from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.ml.feature import Tokenizer, CountVectorizer, PCA
from pyspark.ml.linalg import Vectors, VectorUDT

# Inicializar SparkSession
spark = SparkSession.builder \
    .appName("Twitter Data Processing") \
    .getOrCreate()

# Ajustar el nivel de registro para minimizar las advertencias
spark.sparkContext.setLogLevel("ERROR")

# Cargar datos
df = spark.read.csv('climateTwitterData.csv', header=True, inferSchema=True)

# Seleccionar sólo la columna de texto para procesamiento
df = df.select("text")

# Crear un objeto Tokenizer
tokenizer = Tokenizer(inputCol="text", outputCol="words")

# Aplicar tokenización
df_tokenized = tokenizer.transform(df)

# Instanciar el CountVectorizer
cv = CountVectorizer(inputCol="words", outputCol="features")

# Ajustar el modelo y transformar los datos
model = cv.fit(df_tokenized)
df_features = model.transform(df_tokenized)

# Función para convertir un vector disperso a un vector denso
to_dense = udf(lambda v: Vectors.dense(v.toArray()), VectorUDT())

# Aplicar la conversión de vectores dispersos a densos
df_features = df_features.withColumn("features_dense", to_dense("features"))

# Instanciar y configurar PCA para utilizar vectores densos
pca = PCA(k=100, inputCol="features_dense", outputCol="pcaFeatures")

# Ajustar y transformar los datos con PCA
model_pca = pca.fit(df_features)
df_pca = model_pca.transform(df_features)

# Mostrar resultados
df_pca.select("pcaFeatures").show()

# Detener SparkSession
spark.stop()
