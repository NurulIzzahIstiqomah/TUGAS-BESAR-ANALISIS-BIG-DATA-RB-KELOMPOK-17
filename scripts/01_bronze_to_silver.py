# 01_bronze_to_silver.py

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, upper, trim

# =====================================================
# Spark Session
# =====================================================

spark = SparkSession.builder \
    .appName("Bronze to Silver Pendidikan") \
    .getOrCreate()

# =====================================================
# Read Bronze Layer
# =====================================================

apk = spark.read.csv(
    "/bronze/apk.csv",
    header=True,
    inferSchema=True
)

apm = spark.read.csv(
    "/bronze/apm.csv",
    header=True,
    inferSchema=True
)

sd = spark.read.csv(
    "/bronze/sd.csv",
    header=True,
    inferSchema=True
)

smp = spark.read.csv(
    "/bronze/smp.csv",
    header=True,
    inferSchema=True
)

# =====================================================
# Read Laboratorium SD
# =====================================================

lab_sd_raw = spark.read.csv(
    "/bronze/lab_sd.csv",
    header=True,
    inferSchema=True
)

# =====================================================
# Read Laboratorium SMP
# =====================================================

lab_smp_raw = spark.read.csv(
    "/bronze/lab_smp.csv",
    header=False,
    inferSchema=True
)

# =====================================================
# Cleaning LAB SMP
# Menghapus header tambahan
# =====================================================

lab_smp_clean = lab_smp_raw.filter(
    col("_c0") == 2024
)

lab_smp_clean = lab_smp_clean.toDF(
    "Periode",
    "Wilayah",
    "Kode_Wilayah",
    "Kota_Kab",
    "Kode_Kota_Kab",
    "Status_Sekolah",
    "Baik",
    "Rusak_Ringan",
    "Rusak_Sedang",
    "Rusak_Berat",
    "Total_Rusak",
    "Jumlah"
)

# =====================================================
# Cleaning LAB SD
# =====================================================

lab_sd = lab_sd_raw.withColumnRenamed(
    "Status Sekolah",
    "Status_Sekolah"
)

# =====================================================
# Buang Baris Total
# (Negeri dan Swasta)
# =====================================================

lab_sd = lab_sd.filter(
    col("Status_Sekolah") != "Negeri dan Swasta"
)

lab_smp_clean = lab_smp_clean.filter(
    col("Status_Sekolah") != "Negeri dan Swasta"
)

# =====================================================
# Agregasi Laboratorium per Provinsi
# =====================================================

lab_sd_prov = lab_sd.groupBy(
    "Wilayah"
).agg(
    sum("Jumlah").alias("Lab_SD")
)

lab_smp_prov = lab_smp_clean.groupBy(
    "Wilayah"
).agg(
    sum("Jumlah").alias("Lab_SMP")
)

# =====================================================
# Standarisasi Nama Provinsi
# =====================================================

mapping = {
    "PROV. ACEH": "ACEH",
    "PROV. SUMATERA UTARA": "SUMATERA UTARA",
    "PROV. SUMATERA BARAT": "SUMATERA BARAT",
    "PROV. RIAU": "RIAU",
    "PROV. JAMBI": "JAMBI",
    "PROV. SUMATERA SELATAN": "SUMATERA SELATAN",
    "PROV. BENGKULU": "BENGKULU",
    "PROV. LAMPUNG": "LAMPUNG",
    "PROV. KEPULAUAN BANGKA BELITUNG": "KEP. BANGKA BELITUNG",
    "PROV. KEPULAUAN RIAU": "KEP. RIAU",
    "PROV. D.K.I. JAKARTA": "DKI JAKARTA",
    "PROV. D.I. YOGYAKARTA": "DI YOGYAKARTA",
    "PROV. JAWA BARAT": "JAWA BARAT",
    "PROV. JAWA TENGAH": "JAWA TENGAH",
    "PROV. JAWA TIMUR": "JAWA TIMUR",
    "PROV. BANTEN": "BANTEN",
    "PROV. BALI": "BALI",
    "PROV. NUSA TENGGARA BARAT": "NUSA TENGGARA BARAT",
    "PROV. NUSA TENGGARA TIMUR": "NUSA TENGGARA TIMUR",
    "PROV. KALIMANTAN BARAT": "KALIMANTAN BARAT",
    "PROV. KALIMANTAN TENGAH": "KALIMANTAN TENGAH",
    "PROV. KALIMANTAN SELATAN": "KALIMANTAN SELATAN",
    "PROV. KALIMANTAN TIMUR": "KALIMANTAN TIMUR",
    "PROV. KALIMANTAN UTARA": "KALIMANTAN UTARA",
    "PROV. SULAWESI UTARA": "SULAWESI UTARA",
    "PROV. SULAWESI TENGAH": "SULAWESI TENGAH",
    "PROV. SULAWESI SELATAN": "SULAWESI SELATAN",
    "PROV. SULAWESI TENGGARA": "SULAWESI TENGGARA",
    "PROV. GORONTALO": "GORONTALO",
    "PROV. SULAWESI BARAT": "SULAWESI BARAT",
    "PROV. MALUKU": "MALUKU",
    "PROV. MALUKU UTARA": "MALUKU UTARA",
    "PROV. PAPUA BARAT": "PAPUA BARAT",
    "PROV. PAPUA BARAT DAYA": "PAPUA BARAT DAYA",
    "PROV. PAPUA": "PAPUA",
    "PROV. PAPUA SELATAN": "PAPUA SELATAN",
    "PROV. PAPUA TENGAH": "PAPUA TENGAH",
    "PROV. PAPUA PEGUNUNGAN": "PAPUA PEGUNUNGAN"
}

for old_name, new_name in mapping.items():
    lab_sd_prov = lab_sd_prov.replace(
        old_name,
        new_name,
        subset=["Wilayah"]
    )

    lab_smp_prov = lab_smp_prov.replace(
        old_name,
        new_name,
        subset=["Wilayah"]
    )

lab_sd_prov = lab_sd_prov.withColumnRenamed(
    "Wilayah",
    "Provinsi"
)

lab_smp_prov = lab_smp_prov.withColumnRenamed(
    "Wilayah",
    "Provinsi"
)

# =====================================================
# Standarisasi Kolom Provinsi Dataset Lain
# =====================================================

apk = apk.withColumn(
    "Provinsi",
    upper(trim(col("Provinsi")))
)

apm = apm.withColumn(
    "Provinsi",
    upper(trim(col("Provinsi")))
)

# =====================================================
# Join Semua Dataset
# =====================================================

silver = apk \
    .join(apm, "Provinsi") \
    .join(sd, "Provinsi") \
    .join(smp, "Provinsi") \
    .join(lab_sd_prov, "Provinsi") \
    .join(lab_smp_prov, "Provinsi")

# =====================================================
# Save Silver Layer
# =====================================================

silver.write \
    .mode("overwrite") \
    .parquet("/silver/pendidikan")

silver.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/silver/pendidikan_csv")

print("Silver Layer berhasil dibuat")
print("Jumlah Provinsi:", silver.count())
```

