# 🎓 Student Clustering Web App - Simple Version

## 📱 **Aplikasi Web Sederhana untuk Clustering Siswa**

Aplikasi web interaktif dengan form input untuk prediksi cluster dan berbagai metode pengelompokan siswa.

## 🚀 **Cara Menjalankan**

```bash
streamlit run simple_web_app.py
```

Aplikasi akan terbuka di: `http://localhost:8504`

## 📊 **Fitur Utama**

### 1. **📊 Clustering Analysis**

- Analisis clustering dengan 5 kelompok siswa
- Visualisasi distribusi cluster
- Analisis detail setiap cluster
- Performance level classification

### 2. **🎯 Cluster Prediction**

- **Form input** untuk data siswa baru
- **Prediksi cluster** berdasarkan karakteristik
- **Performance assessment** otomatis
- **Learning style classification**
- **Risk level assessment**
- **Personalized recommendations**

### 3. **📈 Alternative Grouping Methods**

- **Learning Style Groups** (4 kategori)
- **Risk Level Groups** (4 kategori)
- **Intervention Priority** (4 kategori)
- **Study Pattern Groups** (4 kategori)

### 4. **📋 About Section**

- Informasi tentang aplikasi
- Dataset description
- Technology stack

## 🎯 **Form Input Fields**

### 📚 **Academic Data:**

- Hours Studied per Week (0-40)
- Attendance (%) (0-100)
- Previous Scores (0-100)
- Sleep Hours per Day (4-12)
- Tutoring Sessions (0-10)

### 🏠 **Background Data:**

- Parental Involvement (Low/Medium/High)
- Access to Resources (Low/Medium/High)
- Motivation Level (Low/Medium/High)
- Internet Access (Yes/No)
- Family Income (Low/Medium/High)

### 📋 **Additional Information:**

- Extracurricular Activities (Yes/No)
- Teacher Quality (Low/Medium/High)
- Peer Influence (Positive/Neutral/Negative)
- Physical Activity (hours/week)
- Learning Disabilities (Yes/No)
- School Type (Public/Private)

## 🎯 **Output Prediction**

### 📊 **Cluster Prediction:**

- Predicted cluster number
- Similar students count
- Average performance metrics

### 📈 **Performance Assessment:**

- Performance Level (High/Above Average/Average/Below Average/Need Support)
- Learning Style (Dedicated/Consistent/Moderate/Struggling)
- Risk Level (High/Medium/Low/No Risk)

### 💡 **Personalized Recommendations:**

- Academic intervention suggestions
- Study habit improvements
- Attendance recommendations
- Tutoring suggestions
- Goal setting advice

## 🎨 **Alternative Grouping Methods**

### 📚 **Learning Style Groups:**

- 🟢 **Dedicated Learners**: High study hours + high attendance + high motivation
- 🟡 **Consistent Learners**: Good study habits + consistent attendance
- 🟠 **Moderate Learners**: Moderate effort in studies
- 🔴 **Struggling Learners**: Need significant support

### ⚠️ **Risk Level Groups:**

- 🔴 **High Risk**: Low scores + low attendance + low study hours
- 🟠 **Medium Risk**: Moderate academic challenges
- 🟡 **Low Risk**: Minor issues that need attention
- 🟢 **No Risk**: Good academic performance

### 🎯 **Intervention Priority:**

- 🔴 **Immediate Intervention**: Urgent academic support needed
- 🟠 **Regular Monitoring**: Consistent tracking required
- 🟡 **Light Support**: Minor assistance needed
- 🟢 **Self-Sufficient**: Independent learners

### 📊 **Study Pattern Groups:**

- 🟢 **High Study + High Attendance**: Excellent study habits
- 🟡 **High Study + Low Attendance**: Good effort but attendance issues
- 🟠 **Low Study + High Attendance**: Present but not studying enough
- 🔴 **Low Study + Low Attendance**: Major intervention needed

## 🎯 **Contoh Penggunaan**

### **Scenario 1: Siswa Baru**

1. Masukkan data siswa di form
2. Klik "Predict Cluster"
3. Lihat hasil prediksi cluster
4. Baca recommendations yang diberikan

### **Scenario 2: Analisis Kelompok**

1. Buka tab "Clustering Analysis"
2. Lihat distribusi 5 cluster
3. Analisis karakteristik setiap cluster
4. Bandingkan performance levels

### **Scenario 3: Alternative Grouping**

1. Buka tab "Alternative Grouping"
2. Lihat berbagai cara mengelompokkan siswa
3. Pilih metode yang sesuai kebutuhan
4. Terapkan untuk analisis

## 🛠️ **Technology Stack**

- **Frontend**: Streamlit
- **Backend**: Python
- **ML**: Scikit-learn (K-Means)
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

## 📊 **Dataset Info**

- **Total Students**: 6,607
- **Features**: 20 variables
- **Clusters**: 5 optimal clusters
- **Performance Range**: 55-101

## 🎉 **Keunggulan Aplikasi**

✅ **Simple & User-Friendly**: Interface yang mudah digunakan
✅ **Interactive Form**: Input data siswa dengan mudah
✅ **Multiple Grouping Methods**: Berbagai cara mengelompokkan
✅ **Personalized Recommendations**: Saran yang disesuaikan
✅ **Real-time Prediction**: Prediksi cluster instan
✅ **Visual Analytics**: Grafik dan visualisasi yang informatif

## 🚀 **Quick Start**

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run application:**

   ```bash
   streamlit run simple_web_app.py
   ```

3. **Open browser:**

   ```
   http://localhost:8504
   ```

4. **Start using:**
   - Explore clustering analysis
   - Try cluster prediction
   - Check alternative grouping methods

**Happy Clustering! 🎓📊**
