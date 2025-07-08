# 🎓 Aplikasi Pengelompokan Siswa (Student Clustering Application)

## 📋 Deskripsi

Aplikasi web interaktif untuk mengelompokkan siswa berdasarkan karakteristik belajar menggunakan algoritma K-Means clustering. Aplikasi ini membantu guru dan administrator sekolah untuk menganalisis performa siswa dan memberikan rekomendasi yang tepat.

## ✨ Fitur Utama

### 📊 Analisis Cluster

- **Pengelompokan Otomatis**: Mengelompokkan siswa menjadi 5 cluster berdasarkan karakteristik belajar
- **Visualisasi Data**: Grafik dan chart interaktif untuk memahami distribusi cluster
- **Analisis Detail**: Deskripsi lengkap setiap cluster dengan karakteristik, kekuatan, dan rekomendasi
- **Statistik Real-time**: Informasi statistik dataset yang diperbarui secara real-time

### 🎯 Prediksi Cluster

- **Form Input Interaktif**: Form yang mudah digunakan untuk memasukkan data siswa baru
- **Prediksi Real-time**: Memprediksi cluster yang sesuai untuk siswa baru
- **Analisis Komprehensif**: Memberikan analisis mendalam tentang karakteristik siswa
- **Rekomendasi Personal**: Saran yang disesuaikan dengan karakteristik siswa

### 🎨 Interface yang User-Friendly

- **Bahasa Indonesia**: Interface lengkap dalam bahasa Indonesia
- **Design Responsif**: Tampilan yang responsif dan mudah dinavigasi
- **Sidebar Informatif**: Panel samping dengan informasi dataset dan filter
- **Color Coding**: Penggunaan warna untuk membedakan tingkat performa

## 🏗️ Arsitektur Sistem

### Teknologi yang Digunakan

- **Frontend**: Streamlit (Python web framework)
- **Machine Learning**: Scikit-learn (K-Means clustering)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly (interactive charts)
- **Data Preprocessing**: StandardScaler, LabelEncoder

### Struktur Data

Aplikasi menggunakan dataset dengan fitur-fitur berikut:

- **Data Akademik**: Jam belajar, kehadiran, nilai ujian
- **Data Latar Belakang**: Keterlibatan orang tua, akses sumber belajar, motivasi
- **Data Tambahan**: Kegiatan ekstrakurikuler, kualitas guru, pengaruh teman

## 🚀 Cara Instalasi

### Prasyarat

- Python 3.7 atau lebih baru
- pip (Python package manager)

### Langkah Instalasi

1. **Clone atau download repository ini**

2. **Jalankan file instalasi otomatis**:

   ```bash
   # Windows
   pip install -r requirements.txt

   # Atau manual installation
   pip install streamlit pandas numpy scikit-learn plotly
   ```

3. **Pastikan file dataset.csv ada di folder yang sama**

4. **Jalankan aplikasi**:

   ```bash
   # Windows
   run_app.bat

   # Atau manual
   streamlit run main.py
   ```

5. **Buka browser** dan akses `http://localhost:8501`

## 📖 Cara Penggunaan

### 1. Tab Analisis Cluster

- **Lihat Distribusi**: Melihat jumlah siswa di setiap cluster
- **Analisis Detail**: Membaca deskripsi lengkap setiap cluster
- **Filter Data**: Menggunakan filter di sidebar untuk melihat data tertentu
- **Download Data**: Mengunduh data lengkap untuk analisis lebih lanjut

### 2. Tab Prediksi Cluster

- **Masukkan Data**: Isi form dengan data siswa yang ingin diprediksi
- **Data Akademik**: Jam belajar, kehadiran, nilai sebelumnya
- **Data Latar Belakang**: Keterlibatan orang tua, motivasi, akses sumber
- **Informasi Tambahan**: Kegiatan ekstrakurikuler, kualitas guru, dll
- **Klik Prediksi**: Dapatkan hasil prediksi dan rekomendasi

### 3. Sidebar Features

- **Informasi Dataset**: Total siswa, jumlah fitur
- **Statistik Data**: Nilai rata-rata, jam belajar, kehadiran
- **Distribusi Performa**: Breakdown berdasarkan tingkat performa
- **Filter Cepat**: Filter berdasarkan performa, jam belajar, kehadiran
- **Aksi Cepat**: Download data, lihat statistik detail
- **Bantuan**: Panduan penggunaan dan informasi cluster

## 🎯 Klasifikasi Cluster

### Cluster 1 🟢 - Siswa Berprestasi Tinggi

- **Karakteristik**: Nilai tinggi, jam belajar konsisten, kehadiran baik
- **Rekomendasi**: Pertahankan performa, kembangkan leadership, ikuti program pengayaan

### Cluster 2 🟡 - Siswa Konsisten dan Stabil

- **Karakteristik**: Performa stabil, motivasi cukup, dukungan keluarga memadai
- **Rekomendasi**: Tingkatkan target nilai, kembangkan strategi belajar, bangun kepercayaan diri

### Cluster 3 🟠 - Siswa Rata-rata dengan Potensi

- **Karakteristik**: Performa menengah, potensi berkembang, perlu bimbingan
- **Rekomendasi**: Tingkatkan jam belajar, ikuti bimbingan tambahan, buat jadwal teratur

### Cluster 4 🔴 - Siswa Perlu Perhatian Khusus

- **Karakteristik**: Performa di bawah rata-rata, motivasi rendah, perlu bimbingan khusus
- **Rekomendasi**: Program remedial intensif, tingkatkan kehadiran, bimbingan personal

### Cluster 5 🔴 - Siswa Berisiko Tinggi

- **Karakteristik**: Performa rendah, motivasi sangat rendah, perlu intervensi
- **Rekomendasi**: Intervensi segera, program intensif, konseling psikologis

## 📊 Algoritma dan Metodologi

### K-Means Clustering

- **Jumlah Cluster**: 5 cluster (optimal berdasarkan analisis data)
- **Features**: Semua fitur kecuali Exam_Score (target variable)
- **Preprocessing**: StandardScaler untuk normalisasi data
- **Encoding**: LabelEncoder untuk variabel kategorikal

### Klasifikasi Performa

- **Berprestasi Tinggi**: ≥ 72
- **Di Atas Rata-rata**: 68-71
- **Rata-rata**: 64-67
- **Di Bawah Rata-rata**: 60-63
- **Perlu Bantuan**: < 60

### Penilaian Risiko

Berdasarkan kombinasi:

- Nilai ujian
- Persentase kehadiran
- Jam belajar per minggu

## 🔧 Konfigurasi dan Kustomisasi

### Mengubah Jumlah Cluster

```python
# Di fungsi perform_clustering()
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
```

### Menambah Fitur Baru

1. Tambahkan kolom di dataset.csv
2. Update feature_columns di load_and_preprocess_data()
3. Tambahkan input field di form prediksi

### Mengubah Threshold Klasifikasi

```python
# Di fungsi classify_performance()
if score >= 72:  # Ubah threshold sesuai kebutuhan
    return "Berprestasi Tinggi", "🟢"
```

## 📁 Struktur File

```
SELESAI/
├── main.py                 # File utama aplikasi
├── dataset.csv            # Dataset siswa
├── requirements.txt       # Dependencies Python
├── install_requirements.bat  # Script instalasi Windows
├── run_app.bat           # Script menjalankan aplikasi
└── README.md             # Dokumentasi ini
```

## 🐛 Troubleshooting

### Error: "No module named 'streamlit'"

**Solusi**: Jalankan `install.bat` atau `pip install streamlit`

### Error: "File 'dataset.csv' not found"

**Solusi**: Pastikan file dataset.csv ada di folder yang sama dengan main.py

### Error: "LabelEncoder not fitted"

**Solusi**: Pastikan semua nilai kategorikal di form sesuai dengan yang ada di dataset

### Aplikasi tidak bisa diakses

**Solusi**:

1. Pastikan port 8501 tidak digunakan aplikasi lain
2. Cek firewall settings
3. Coba akses `http://127.0.0.1:8501`

## 🤝 Kontribusi

Untuk berkontribusi pada pengembangan aplikasi ini:

1. Fork repository
2. Buat branch fitur baru
3. Commit perubahan
4. Push ke branch
5. Buat Pull Request

## 📄 Lisensi

Aplikasi ini dikembangkan untuk tujuan pendidikan dan dapat digunakan secara bebas.

## 👥 Tim Pengembang

Aplikasi ini dikembangkan sebagai proyek analisis data untuk pengelompokan siswa berdasarkan karakteristik belajar.

---

**🎓 Semoga aplikasi ini membantu dalam menganalisis dan meningkatkan performa siswa!**
