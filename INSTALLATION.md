# 🚀 Installation Guide - Student Behavior Clustering

## 📋 Prerequisites

Sebelum menjalankan aplikasi, pastikan Anda memiliki:

- **Python 3.8 atau lebih baru**
- **pip** (Python package installer)
- **Git** (opsional, untuk clone repository)

## 🔧 Cara Install

### Method 1: Menggunakan Batch File (Windows)

1. **Double-click** file `run_app.bat`
2. Script akan otomatis:
   - Mengecek Python
   - Install dependencies
   - Menjalankan aplikasi

### Method 2: Manual Installation

#### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Step 2: Jalankan Aplikasi

```bash
streamlit run app.py
```

#### Step 3: Buka Browser

Aplikasi akan terbuka di `http://localhost:8501`

### Method 3: Menggunakan Python Script

```bash
python run_app.py
```

## 📦 Dependencies yang Diinstall

Aplikasi ini menggunakan library berikut:

- **streamlit==1.28.1** - Framework web untuk dashboard
- **pandas==2.1.3** - Data manipulation dan analysis
- **numpy==1.24.3** - Numerical computing
- **scikit-learn==1.3.2** - Machine learning algorithms
- **plotly==5.17.0** - Interactive visualizations
- **matplotlib==3.8.2** - Basic plotting
- **seaborn==0.13.0** - Statistical data visualization

## 🔍 Troubleshooting

### Error: "Module not found"

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Error: "Port already in use"

```bash
streamlit run app.py --server.port 8502
```

### Error: "Dataset not found"

Pastikan file `dataset.csv` berada di folder yang sama dengan `app.py`

### Error: "Python not found"

1. Download Python dari https://python.org
2. Install dengan opsi "Add to PATH"
3. Restart terminal/command prompt

## 🐍 Virtual Environment (Recommended)

Untuk isolasi dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## 📊 Dataset Requirements

File `dataset.csv` harus memiliki kolom-kolom berikut:

| Kolom                      | Tipe Data   | Deskripsi                      |
| -------------------------- | ----------- | ------------------------------ |
| Hours_Studied              | Numeric     | Jam belajar per minggu         |
| Attendance                 | Numeric     | Persentase kehadiran           |
| Parental_Involvement       | Categorical | Low/Medium/High                |
| Access_to_Resources        | Categorical | Low/Medium/High                |
| Extracurricular_Activities | Categorical | Yes/No                         |
| Sleep_Hours                | Numeric     | Jam tidur per hari             |
| Previous_Scores            | Numeric     | Nilai sebelumnya               |
| Motivation_Level           | Categorical | Low/Medium/High                |
| Internet_Access            | Categorical | Yes/No                         |
| Tutoring_Sessions          | Numeric     | Jumlah sesi bimbingan          |
| Family_Income              | Categorical | Low/Medium/High                |
| Teacher_Quality            | Categorical | Low/Medium/High                |
| School_Type                | Categorical | Public/Private                 |
| Peer_Influence             | Categorical | Positive/Negative/Neutral      |
| Physical_Activity          | Numeric     | Jam aktivitas fisik per minggu |
| Learning_Disabilities      | Categorical | Yes/No                         |
| Parental_Education_Level   | Categorical | Education level                |
| Distance_from_Home         | Categorical | Near/Moderate/Far              |
| Gender                     | Categorical | Male/Female                    |
| Exam_Score                 | Numeric     | Nilai ujian (target)           |

## 🎯 Quick Start

1. **Download** semua file project
2. **Double-click** `run_app.bat` (Windows) atau jalankan `python run_app.py`
3. **Tunggu** aplikasi terbuka di browser
4. **Pilih** metode clustering (Auto atau Manual)
5. **Analisis** hasil clustering
6. **Download** hasil dalam format CSV

## 📞 Support

Jika mengalami masalah:

1. Cek error message di terminal
2. Pastikan semua dependencies terinstall
3. Pastikan dataset format benar
4. Restart aplikasi jika perlu

## 🔄 Update

Untuk update aplikasi:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```
