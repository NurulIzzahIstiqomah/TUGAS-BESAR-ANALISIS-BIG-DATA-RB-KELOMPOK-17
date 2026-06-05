## Implementasi Big Data Ecosystem untuk Analisis Pemerataan Pendidikan di Indonesia Menggunakan Apache Spark

### Kelompok 17 - RB

#### Anggota

- Nurul Izzah Istiqomah (123450054)
- Tesalonika Hutajulu (123450033)
- Mia Al Musdari (123450068)
- Hanifah Inaya Sani (123450123)

## Deskripsi 
Proyek ini mengimplementasikan Big Data Ecosystem untuk menganalisis pemerataan
pendidikan di seluruh provinsi Indonesia menggunakan Apache Spark. Pipeline data
dibangun dengan pendekatan Medallion Architecture (Bronze → Silver → Gold), mulai
dari pengumpulan data mentah, pembersihan, integrasi, hingga feature engineering.
Analisis dilakukan menggunakan metode PCA dan K-Means Clustering untuk
mengelompokkan 38 provinsi berdasarkan kualitas pemerataan pendidikannya, 
mencakup indikator APK, APM, rasio guru, rasio laboratorium, dan rasio murid per
sekolah pada jenjang SD dan SMP. Hasil clustering menunjukkan bahwa sebagian besar
wilayah Papua masih memiliki tingkat pemerataan pendidikan yang rendah dibandingkan
provinsi lainnya, dengan Silhouette Score sebesar 0.6506 yang menandakan kualitas
pengelompokan yang cukup baik.


## Dataset

Dataset yang digunakan meliputi:
- Angka Partisipasi Kasar (APK)
- Angka Partisipasi Murni (APM)
- Jumlah Sekolah SD dan SMP
- Jumlah Guru SD dan SMP
- Jumlah Murid SD dan SMP
- Jumlah Laboratorium IPA SD dan SMP

## Arsitektur Data
Bronze Layer → Silver Layer → Gold Layer → PCA → K-Means Clustering

## Metode
1. Data Cleaning
Melakukan pembersihan data dengan mengidentifikasi dan menangani missing values, data duplikat, serta inkonsistensi format untuk memastikan kualitas data yang optimal sebelum proses analisis.
2. Data Integration
Menggabungkan data dari berbagai sumber menjadi satu dataset yang terstruktur dan konsisten sehingga seluruh variabel dapat dianalisis secara terpadu.
3. Feature Engineering
Melakukan pemilihan dan pengolahan fitur yang relevan untuk meningkatkan representasi informasi serta membantu model dalam mengenali pola yang lebih baik.
4. Standardization
Menerapkan Z-Score Standardization untuk menyamakan skala seluruh variabel sehingga tidak ada fitur yang mendominasi proses clustering akibat perbedaan rentang nilai.
5. Principal Component Analysis (PCA)
Mengurangi dimensi data dengan mentransformasikan variabel asli ke dalam komponen utama yang mampu mempertahankan sebagian besar variasi informasi sekaligus mengurangi kompleksitas data.
6. K-Means Clustering
Mengelompokkan 38 provinsi ke dalam 3 cluster berdasarkan kemiripan karakteristik menggunakan algoritma K-Means pada data hasil transformasi PCA.
7. Model Evaluation (Silhouette Score)
Evaluasi kualitas clustering dilakukan menggunakan Silhouette Score dan menghasilkan nilai 0.6506, yang menunjukkan bahwa cluster yang terbentuk memiliki tingkat kohesi yang baik dan pemisahan antar cluster yang cukup jelas.

## Hasil

- Jumlah Cluster: 3
- Cluster 0: 15 Provinsi
- Cluster 1: 3 Provinsi
- Cluster 2: 20 Provinsi
- Silhouette Score: 0.6506
