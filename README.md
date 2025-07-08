# 🎓 Student Behavior Clustering Analysis

Aplikasi web untuk mengelompokkan siswa berdasarkan perilaku belajar menggunakan algoritma K-Means Clustering.

## 📋 Deskripsi

Aplikasi ini menganalisis data siswa berdasarkan berbagai faktor seperti:

- Jam belajar
- Kehadiran
- Keterlibatan orang tua
- Akses ke sumber belajar
- Aktivitas ekstrakurikuler
- Jam tidur
- Nilai sebelumnya
- Level motivasi
- Dan faktor lainnya

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

### 3. Buka Browser

Aplikasi akan terbuka di `http://localhost:8501`

## 📊 Fitur Aplikasi

### 🔍 Analisis Data

- **Auto-detection K optimal**: Menggunakan metode elbow untuk menemukan jumlah cluster terbaik
- **Manual K selection**: Pilihan manual untuk jumlah cluster
- **Data preprocessing**: Normalisasi dan encoding data kategorikal

### 📈 Visualisasi

- **PCA Visualization**: Visualisasi 2D menggunakan Principal Component Analysis
- **Cluster Distribution**: Distribusi siswa di setiap cluster
- **Feature Heatmap**: Heatmap pentingnya fitur per cluster
- **Elbow Plot**: Grafik untuk menentukan K optimal

### 📋 Hasil Analisis

- **Cluster Characteristics**: Karakteristik detail setiap cluster
- **Performance Metrics**: Metrik performa model
- **Download Results**: Download hasil clustering dalam format CSV

## 🎯 Interpretasi Cluster

Aplikasi akan mengelompokkan siswa menjadi beberapa kategori:

### 🟢 High Performers

- Siswa dengan performa akademik tinggi
- Nilai rata-rata > 75
- Karakteristik: Belajar rajin, kehadiran tinggi

### 🟡 Average Performers

- Siswa dengan performa akademik sedang
- Nilai rata-rata 65-75
- Karakteristik: Performa standar, perlu motivasi

### 🔴 Need Support

- Siswa yang membutuhkan dukungan akademik
- Nilai rata-rata < 65
- Karakteristik: Perlu bimbingan khusus

## 📁 Struktur File

```
├── main.py              # Aplikasi Streamlit utama
├── requirements.txt    # Dependencies Python
├── dataset.csv         # Dataset siswa
└── README.md          # Dokumentasi ini
```

## 🛠️ Teknologi yang Digunakan

- **Streamlit**: Framework web untuk dashboard
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical computing

## 📊 Dataset Format

Dataset harus memiliki kolom-kolom berikut:

- `Hours_Studied`: Jam belajar
- `Attendance`: Kehadiran (%)
- `Parental_Involvement`: Keterlibatan orang tua
- `Access_to_Resources`: Akses ke sumber belajar
- `Extracurricular_Activities`: Aktivitas ekstrakurikuler
- `Sleep_Hours`: Jam tidur
- `Previous_Scores`: Nilai sebelumnya
- `Motivation_Level`: Level motivasi
- `Internet_Access`: Akses internet
- `Tutoring_Sessions`: Sesi bimbingan
- `Family_Income`: Pendapatan keluarga
- `Teacher_Quality`: Kualitas guru
- `School_Type`: Jenis sekolah
- `Peer_Influence`: Pengaruh teman sebaya
- `Physical_Activity`: Aktivitas fisik
- `Learning_Disabilities`: Disabilitas belajar
- `Parental_Education_Level`: Level pendidikan orang tua
- `Distance_from_Home`: Jarak dari rumah
- `Gender`: Jenis kelamin
- `Exam_Score`: Nilai ujian (target)

## 🤝 Kontribusi

Silakan berkontribusi dengan:

1. Fork repository
2. Buat feature branch
3. Commit changes
4. Push ke branch
5. Buat Pull Request

## 📝 Lisensi

Project ini dibuat untuk tujuan edukasi dan analisis data siswa.
