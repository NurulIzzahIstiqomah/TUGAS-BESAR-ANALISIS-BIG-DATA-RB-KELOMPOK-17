# 02_silver_to_gold.py

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# =====================================================
# Spark Session
# =====================================================

spark = SparkSession.builder \
    .appName("Silver to Gold Pendidikan") \
    .getOrCreate()

# =====================================================
# Read Silver Layer
# =====================================================

silver = spark.read.parquet(
    "/silver/pendidikan"
)

# =====================================================
# Feature Engineering
# =====================================================

gold = silver \
.withColumn(
    "Guru_SD_per_1000_Murid",
    (col("Guru_SD") / col("Murid_SD")) * 1000
) \
.withColumn(
    "Guru_SMP_per_1000_Murid",
    (col("Guru_SMP") / col("Murid_SMP")) * 1000
) \
.withColumn(
    "Lab_SD_per_100_Sekolah",
    (col("Lab_SD") / col("Sekolah_SD")) * 100
) \
.withColumn(
    "Lab_SMP_per_100_Sekolah",
    (col("Lab_SMP") / col("Sekolah_SMP")) * 100
) \
.withColumn(
    "Murid_SD_per_Sekolah",
    col("Murid_SD") / col("Sekolah_SD")
) \
.withColumn(
    "Murid_SMP_per_Sekolah",
    col("Murid_SMP") / col("Sekolah_SMP")
)

# =====================================================
# Save Gold Layer
# =====================================================

gold.write \
    .mode("overwrite") \
    .parquet("/gold/pendidikan_pemerataan")

gold.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/gold/pendidikan_pemerataan_csv")

print("Gold Layer berhasil dibuat")
print("Jumlah Provinsi:", gold.count())

gold.select(
    "Provinsi",
    "Guru_SD_per_1000_Murid",
    "Guru_SMP_per_1000_Murid",
    "Lab_SD_per_100_Sekolah",
    "Lab_SMP_per_100_Sekolah"
).show(10, False)
```
