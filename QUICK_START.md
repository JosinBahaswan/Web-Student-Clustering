# 🚀 Quick Start Guide - Student Behavior Clustering

## ⚡ Cara Cepat Menjalankan Aplikasi

### 🎯 **Pilihan 1: Double-click (Paling Mudah)**

```
Double-click file: run_app.bat
```

### 🎯 **Pilihan 2: Command Line**

```bash
streamlit run app.py
```

### 🎯 **Pilihan 3: Python Script**

```bash
python run_app.py
```

## 📊 **Apa yang Akan Anda Dapatkan**

### 🎓 **Dashboard Interaktif**

- **Auto-detection K optimal** - Aplikasi otomatis menemukan jumlah cluster terbaik
- **Manual K selection** - Pilihan manual untuk jumlah cluster
- **Real-time analysis** - Analisis langsung dengan visualisasi interaktif

### 📈 **Visualisasi Lengkap**

1. **🎯 Basic Visualizations**

   - PCA 2D Scatter Plot
   - Cluster Distribution Chart

2. **🚀 Advanced Visualizations**

   - 3D PCA Scatter Plot
   - t-SNE Visualization
   - Radar Chart (Feature importance)
   - Box Plots (Key features by cluster)
   - Correlation Heatmaps

3. **📊 Analysis Tools**
   - Elbow Method Plot
   - Performance Comparison
   - Detailed Cluster Summaries

### 🎯 **Hasil Clustering**

Aplikasi akan mengelompokkan siswa menjadi:

- **🟢 High Performers** (>75): Siswa berprestasi tinggi
- **🟡 Average Performers** (65-75): Siswa dengan performa sedang
- **🔴 Need Support** (<65): Siswa yang butuh dukungan

### 💾 **Output yang Tersedia**

- **Interactive Dashboard** dengan semua visualisasi
- **Download CSV** dengan hasil clustering
- **Detailed Reports** per cluster
- **Recommendations** untuk setiap kelompok siswa

## 🔧 **Troubleshooting**

### ❌ **Error: "Module not found"**

```bash
pip install -r requirements.txt
```

### ❌ **Error: "Port already in use"**

```bash
streamlit run app.py --server.port 8502
```

### ❌ **Error: "Dataset not found"**

Pastikan file `dataset.csv` ada di folder yang sama dengan `app.py`

## 📱 **Akses Aplikasi**

Setelah menjalankan, buka browser di:

```
http://localhost:8501
```

## 🎉 **Selamat Menganalisis!**

Aplikasi sudah siap digunakan dengan:

- ✅ Dataset: 6,607 siswa
- ✅ Features: 20 variabel
- ✅ Visualisasi: 8+ jenis chart
- ✅ Analysis: Complete clustering workflow

**Happy Clustering! 🎓📊**
