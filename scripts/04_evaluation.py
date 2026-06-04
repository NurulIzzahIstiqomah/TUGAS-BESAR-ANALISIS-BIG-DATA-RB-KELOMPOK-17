# 04_evaluation.py

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# =====================================================
# Spark Session
# =====================================================

spark = SparkSession.builder \
    .appName("Evaluasi Clustering Pendidikan") \
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

scaled_data = scaler.fit(data).transform(data)

# =====================================================
# PCA
# =====================================================

pca = PCA(
    k=2,
    inputCol="scaled_features",
    outputCol="pca_features"
)

pca_data = pca.fit(scaled_data).transform(scaled_data)

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
# Silhouette Score
# =====================================================

evaluator = ClusteringEvaluator(
    predictionCol="prediction",
    featuresCol="pca_features",
    metricName="silhouette"
)

silhouette = evaluator.evaluate(clustered)

print("\n===================================")
print("SILHOUETTE SCORE")
print("===================================")
print(silhouette)

# =====================================================
# Jumlah Anggota Cluster
# =====================================================

print("\n===================================")
print("JUMLAH PROVINSI PER CLUSTER")
print("===================================")

clustered.groupBy(
    "prediction"
).count().show()

# =====================================================
# Karakteristik Cluster
# =====================================================

print("\n===================================")
print("RATA-RATA INDIKATOR PER CLUSTER")
print("===================================")

clustered.groupBy("prediction").avg(
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
).show(truncate=False)

# =====================================================
# Daftar Provinsi Tiap Cluster
# =====================================================

print("\n===================================")
print("HASIL CLUSTERING")
print("===================================")

clustered.select(
    "Provinsi",
    "prediction"
).orderBy(
    "prediction",
    "Provinsi"
).show(50, False)

# =====================================================
# Centroid Cluster
# =====================================================

print("\n===================================")
print("CENTROID CLUSTER")
print("===================================")

for i, center in enumerate(model.clusterCenters()):
    print(f"Cluster {i}:")
    print(center)
    print()
```
