# 03_pca_kmeans.py

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

# =====================================================
# Spark Session
# =====================================================

spark = SparkSession.builder \
    .appName("PCA KMeans Pendidikan") \
    .getOrCreate()

# =====================================================
# Read Gold Layer
# =====================================================

gold = spark.read.parquet(
    "/gold/pendidikan_pemerataan"
)

# =====================================================
# Feature Selection
# =====================================================

features = [
    "APK_SD",
    "APK_SMP",
    "APM_SD",
    "APM_SMP",
    "Guru_SD_per_1000_Murid",
    "Guru_SMP_per_1000_Murid",
    "Lab_SD_per_100_Sekolah",
    "Lab_SMP_per_100_Sekolah",
    "Murid_SD_per_Sekolah",
    "Murid_SMP_per_Sekolah"
]

# =====================================================
# Vector Assembler
# =====================================================

assembler = VectorAssembler(
    inputCols=features,
    outputCol="features"
)

data = assembler.transform(gold)

# =====================================================
# Standard Scaler
# =====================================================

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=True
)

scaler_model = scaler.fit(data)

scaled_data = scaler_model.transform(data)

# =====================================================
# PCA
# =====================================================

pca = PCA(
    k=2,
    inputCol="scaled_features",
    outputCol="pca_features"
)

pca_model = pca.fit(scaled_data)

pca_data = pca_model.transform(scaled_data)

# =====================================================
# K-Means
# =====================================================

kmeans = KMeans(
    k=3,
    seed=42,
    featuresCol="pca_features"
)

model = kmeans.fit(pca_data)

clustered = model.transform(pca_data)

# =====================================================
# Simpan Hasil Cluster
# =====================================================

hasil_cluster = clustered.select(
    "Provinsi",
    "prediction"
)

hasil_cluster.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/gold/hasil_cluster")

# =====================================================
# Hasil Cluster Lengkap
# =====================================================

hasil_cluster_lengkap = clustered.select(
    "Provinsi",
    "prediction"
).join(
    gold,
    "Provinsi"
)

hasil_cluster_lengkap.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/gold/hasil_cluster_lengkap")

# =====================================================
# PCA Visualization Dataset
# =====================================================

get_pc1 = udf(
    lambda v: float(v[0]),
    DoubleType()
)

get_pc2 = udf(
    lambda v: float(v[1]),
    DoubleType()
)

pca_vis = clustered \
.withColumn("PC1", get_pc1("pca_features")) \
.withColumn("PC2", get_pc2("pca_features"))

pca_vis.select(
    "Provinsi",
    "prediction",
    "PC1",
    "PC2"
).write \
.mode("overwrite") \
.option("header", True) \
.csv("/gold/pca_visualisasi")

# =====================================================
# Output
# =====================================================

print("Jumlah Provinsi:", clustered.count())

clustered.groupBy(
    "prediction"
).count().show()

clustered.select(
    "Provinsi",
    "prediction"
).show(38, False)
```
