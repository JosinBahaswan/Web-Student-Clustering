import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Aplikasi Pengelompokan Siswa",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .cluster-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
    }
    .form-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid #dee2e6;
    }
    .cluster-description {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #ffc107;
    }
    .conclusion-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess dataset"""
    try:
        df = pd.read_csv('dataset.csv', sep=';')
        
        # Handle missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('Tidak Diketahui')
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        # Encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        # Select features for clustering
        feature_columns = [col for col in df.columns if col not in ['Exam_Score']]
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[feature_columns])
        
        return df, features_scaled, feature_columns, label_encoders, scaler
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

def perform_clustering(features_scaled, n_clusters=5):
    """Perform K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    return kmeans, cluster_labels

def get_cluster_color(cluster_id):
    """Get color for each cluster based on cluster ID"""
    colors = {
        0: "🟢",  # Cluster 1: Siswa Berprestasi Tinggi - Hijau
        1: "🟡",  # Cluster 2: Siswa Konsisten - Kuning
        2: "🟠",  # Cluster 3: Siswa Rata-rata - Oranye
        3: "🔴",  # Cluster 4: Siswa Perlu Perhatian - Merah
        4: "🔴"   # Cluster 5: Siswa Berisiko Tinggi - Merah
    }
    return colors.get(cluster_id, "⚪")

def classify_performance(score):
    """Classify performance based on score"""
    if score >= 72:
        return "Berprestasi Tinggi", "🟢"
    elif score >= 68:
        return "Di Atas Rata-rata", "🟢"
    elif score >= 64:
        return "Rata-rata", "🟡"
    elif score >= 60:
        return "Di Bawah Rata-rata", "🟠"
    else:
        return "Perlu Bantuan", "🔴"

def get_cluster_description(cluster_id, avg_score, avg_hours, avg_attendance, student_count):
    """Get detailed description for each cluster"""
    descriptions = {
        0: {
            "name": "Siswa Berprestasi Tinggi",
            "description": "Kelompok siswa dengan performa akademik terbaik dan konsisten",
            "characteristics": [
                f"Nilai ujian rata-rata {avg_score:.1f} (sangat tinggi)",
                f"Jam belajar rata-rata {avg_hours:.1f} jam/minggu (sangat konsisten)",
                f"Kehadiran rata-rata {avg_attendance:.1f}% (sangat baik)",
                "Motivasi belajar sangat tinggi dan stabil",
                "Dukungan keluarga dan lingkungan optimal",
                "Kemampuan self-directed learning yang kuat"
            ],
            "strengths": [
                "Kemampuan akademik yang sangat kuat",
                "Disiplin dan konsistensi dalam belajar",
                "Kehadiran yang sangat teratur",
                "Motivasi intrinsik yang tinggi",
                "Dukungan lingkungan yang sangat baik",
                "Kemampuan leadership dan mentoring"
            ],
            "recommendations": [
                "Pertahankan performa yang sudah sangat baik",
                "Kembangkan kemampuan leadership dan mentoring",
                "Bantu dan bimbing teman yang membutuhkan",
                "Ikuti program pengayaan dan kompetisi akademik",
                "Persiapkan untuk jenjang pendidikan berikutnya",
                "Jadilah role model bagi siswa lain"
            ]
        },
        1: {
            "name": "Siswa Konsisten dan Stabil",
            "description": "Kelompok siswa dengan performa yang stabil dan memiliki potensi untuk berkembang",
            "characteristics": [
                f"Nilai ujian rata-rata {avg_score:.1f} (di atas rata-rata)",
                f"Jam belajar rata-rata {avg_hours:.1f} jam/minggu (teratur)",
                f"Kehadiran rata-rata {avg_attendance:.1f}% (baik)",
                "Motivasi belajar cukup dan stabil",
                "Dukungan keluarga memadai",
                "Kemampuan belajar yang konsisten"
            ],
            "strengths": [
                "Konsistensi dalam belajar dan kehadiran",
                "Motivasi yang stabil dan dapat diandalkan",
                "Kemampuan akademik yang baik",
                "Kesediaan untuk belajar dan berkembang",
                "Dukungan keluarga yang memadai",
                "Potensi untuk menjadi lebih baik"
            ],
            "recommendations": [
                "Tingkatkan target nilai secara bertahap",
                "Kembangkan strategi belajar yang lebih efektif",
                "Ikuti kegiatan ekstrakurikuler yang menantang",
                "Bangun kepercayaan diri dan ambisi",
                "Jadilah role model untuk teman sebaya",
                "Persiapkan untuk tantangan akademik yang lebih tinggi"
            ]
        },
        2: {
            "name": "Siswa Rata-rata dengan Potensi",
            "description": "Kelompok siswa dengan performa menengah yang memiliki potensi untuk berkembang dengan bimbingan yang tepat",
            "characteristics": [
                f"Nilai ujian rata-rata {avg_score:.1f} (rata-rata)",
                f"Jam belajar rata-rata {avg_hours:.1f} jam/minggu (bervariasi)",
                f"Kehadiran rata-rata {avg_attendance:.1f}% (cukup baik)",
                "Motivasi belajar sedang dan perlu dorongan",
                "Dukungan keluarga bervariasi",
                "Kemampuan belajar yang masih bisa dikembangkan"
            ],
            "strengths": [
                "Potensi akademik yang masih bisa dikembangkan",
                "Kehadiran yang cukup baik",
                "Kemampuan dasar yang memadai",
                "Kesediaan untuk belajar dan berubah",
                "Dukungan dari sistem pendidikan",
                "Motivasi yang bisa ditingkatkan"
            ],
            "recommendations": [
                "Tingkatkan jam belajar secara bertahap dan teratur",
                "Ikuti bimbingan belajar tambahan yang sesuai",
                "Tingkatkan motivasi diri dan kepercayaan diri",
                "Buat jadwal belajar yang teratur dan realistis",
                "Minta dukungan dari guru dan keluarga",
                "Fokus pada pengembangan kebiasaan belajar yang baik"
            ]
        },
        3: {
            "name": "Siswa Perlu Perhatian Khusus",
            "description": "Kelompok siswa yang membutuhkan perhatian dan bimbingan khusus untuk meningkatkan performa akademik",
            "characteristics": [
                f"Nilai ujian rata-rata {avg_score:.1f} (di bawah rata-rata)",
                f"Jam belajar rata-rata {avg_hours:.1f} jam/minggu (kurang optimal)",
                f"Kehadiran rata-rata {avg_attendance:.1f}% (perlu ditingkatkan)",
                "Motivasi belajar rendah dan tidak stabil",
                "Dukungan keluarga terbatas atau tidak optimal",
                "Kemampuan belajar yang membutuhkan bimbingan khusus"
            ],
            "strengths": [
                "Potensi yang masih ada dan bisa dikembangkan",
                "Kesediaan untuk berubah dan berkembang",
                "Dukungan dari sekolah dan sistem pendidikan",
                "Kemampuan dasar yang bisa dikembangkan",
                "Motivasi yang bisa ditingkatkan dengan pendekatan yang tepat",
                "Kesempatan untuk berubah dengan program yang sesuai"
            ],
            "recommendations": [
                "Ikuti program remedial intensif dan terstruktur",
                "Tingkatkan kehadiran di sekolah secara signifikan",
                "Dapatkan bimbingan belajar khusus yang personal",
                "Tingkatkan motivasi belajar dengan pendekatan yang menarik",
                "Libatkan orang tua dalam proses belajar secara aktif",
                "Buat rencana pembelajaran yang realistis dan terukur"
            ]
        },
        4: {
            "name": "Siswa Berisiko Tinggi - Perlu Intervensi",
            "description": "Kelompok siswa yang membutuhkan intervensi segera dan program khusus untuk mengatasi masalah akademik",
            "characteristics": [
                f"Nilai ujian rata-rata {avg_score:.1f} (rendah)",
                f"Jam belajar rata-rata {avg_hours:.1f} jam/minggu (sangat kurang)",
                f"Kehadiran rata-rata {avg_attendance:.1f}% (tidak teratur)",
                "Motivasi belajar sangat rendah dan tidak stabil",
                "Dukungan keluarga minimal atau tidak ada",
                "Kemampuan belajar yang membutuhkan intervensi khusus"
            ],
            "strengths": [
                "Potensi yang masih bisa dikembangkan dengan program yang tepat",
                "Dukungan dari sistem pendidikan dan sekolah",
                "Kesempatan untuk berubah dengan intervensi yang intensif",
                "Program khusus dari sekolah yang tersedia",
                "Kemampuan dasar yang bisa dikembangkan",
                "Motivasi yang bisa dibangun dengan pendekatan yang tepat"
            ],
            "recommendations": [
                "Intervensi akademik segera dan intensif",
                "Program bimbingan belajar yang sangat personal",
                "Konseling psikologis untuk motivasi dan kepercayaan diri",
                "Pendekatan personal dari guru dan konselor",
                "Libatkan semua pihak (sekolah, keluarga, masyarakat)",
                "Buat program pemulihan yang komprehensif dan terukur"
            ]
        }
    }
    
    return descriptions.get(cluster_id, {
        "name": "Cluster Tidak Diketahui",
        "description": "Deskripsi tidak tersedia",
        "characteristics": [],
        "strengths": [],
        "recommendations": []
    })

def classify_learning_style(hours_studied, attendance, motivation_level):
    """Classify learning style based on study patterns"""
    if hours_studied > 25 and attendance > 85 and motivation_level > 1:
        return "Pembelajar Berdedikasi", "🟢", "Siswa yang sangat rajin dan termotivasi"
    elif hours_studied > 20 and attendance > 80:
        return "Pembelajar Konsisten", "🟡", "Siswa yang konsisten dalam belajar"
    elif hours_studied > 15 and attendance > 75:
        return "Pembelajar Sedang", "🟠", "Siswa dengan usaha sedang"
    else:
        return "Pembelajar Bermasalah", "🔴", "Siswa yang perlu bantuan"

def classify_risk_level(score, attendance, hours_studied):
    """Classify academic risk level"""
    if score < 60 or attendance < 70 or hours_studied < 15:
        return "Risiko Tinggi", "🔴", "Risiko tinggi - perlu intervensi segera"
    elif score < 65 or attendance < 80 or hours_studied < 18:
        return "Risiko Sedang", "🟠", "Risiko sedang - perlu monitoring"
    elif score < 70 or attendance < 85:
        return "Risiko Rendah", "🟡", "Risiko rendah - perlu sedikit dukungan"
    else:
        return "Tidak Berisiko", "🟢", "Tidak ada risiko - performa baik"

def predict_cluster(student_data, kmeans_model, scaler, feature_columns):
    """Predict cluster for new student data"""
    # Create DataFrame with student data
    df_new = pd.DataFrame([student_data])
    
    # Scale the features
    features_scaled = scaler.transform(df_new[feature_columns])
    
    # Predict cluster
    cluster = kmeans_model.predict(features_scaled)[0]
    
    return cluster

def get_prediction_conclusion(cluster_id, student_data, avg_score, avg_hours, avg_attendance):
    """Get conclusion for cluster prediction"""
    
    # Helper function to evaluate study hours
    def evaluate_study_hours(hours):
        if hours >= 30:
            return f"Jam belajar ({hours} jam/minggu) menunjukkan dedikasi yang sangat tinggi"
        elif hours >= 25:
            return f"Jam belajar ({hours} jam/minggu) menunjukkan dedikasi yang baik"
        elif hours >= 20:
            return f"Jam belajar ({hours} jam/minggu) menunjukkan usaha yang baik"
        elif hours >= 15:
            return f"Jam belajar ({hours} jam/minggu) masih bisa ditingkatkan"
        elif hours >= 10:
            return f"Jam belajar ({hours} jam/minggu) perlu ditingkatkan"
        else:
            return f"Jam belajar ({hours} jam/minggu) sangat kurang"
    
    # Helper function to evaluate attendance
    def evaluate_attendance(attendance):
        if attendance >= 95:
            return f"Kehadiran ({attendance}%) sangat baik"
        elif attendance >= 90:
            return f"Kehadiran ({attendance}%) menunjukkan komitmen yang tinggi"
        elif attendance >= 85:
            return f"Kehadiran ({attendance}%) menunjukkan tanggung jawab"
        elif attendance >= 80:
            return f"Kehadiran ({attendance}%) menunjukkan komitmen dasar"
        elif attendance >= 75:
            return f"Kehadiran ({attendance}%) perlu diperbaiki"
        else:
            return f"Kehadiran ({attendance}%) tidak teratur"
    
    conclusions = {
        0: {
            "title": "🎯 Siswa Berpotensi Unggul",
            "description": "Berdasarkan data yang dimasukkan, siswa ini memiliki karakteristik yang mirip dengan kelompok siswa berprestasi tinggi.",
            "analysis": [
                evaluate_study_hours(student_data['Hours_Studied']),
                evaluate_attendance(student_data['Attendance']),
                "Motivasi belajar yang tinggi mendukung performa akademik"
            ],
            "expectations": [
                "Diharapkan dapat mencapai nilai di atas 70",
                "Dapat menjadi role model bagi teman-teman",
                "Potensial untuk mengikuti program pengayaan",
                "Siap untuk jenjang pendidikan berikutnya"
            ],
            "next_steps": [
                "Pertahankan kebiasaan belajar yang baik",
                "Kembangkan kemampuan leadership",
                "Ikuti kegiatan ekstrakurikuler yang menantang",
                "Persiapkan untuk kompetisi akademik"
            ]
        },
        1: {
            "title": "📚 Siswa Konsisten dan Stabil",
            "description": "Siswa ini menunjukkan karakteristik yang konsisten dan memiliki potensi untuk berkembang lebih baik.",
            "analysis": [
                evaluate_study_hours(student_data['Hours_Studied']),
                evaluate_attendance(student_data['Attendance']),
                "Motivasi belajar yang cukup mendukung perkembangan"
            ],
            "expectations": [
                "Diharapkan dapat mencapai nilai 65-75",
                "Dapat meningkatkan performa secara bertahap",
                "Potensial untuk menjadi siswa yang lebih baik",
                "Dapat membantu teman yang membutuhkan"
            ],
            "next_steps": [
                "Tingkatkan target nilai secara bertahap",
                "Kembangkan strategi belajar yang lebih efektif",
                "Bangun kepercayaan diri",
                "Ikuti bimbingan belajar untuk pengayaan"
            ]
        },
        2: {
            "title": "🔄 Siswa dengan Potensi Pengembangan",
            "description": "Siswa ini memiliki potensi yang baik namun membutuhkan dorongan dan strategi yang tepat untuk berkembang.",
            "analysis": [
                evaluate_study_hours(student_data['Hours_Studied']),
                evaluate_attendance(student_data['Attendance']),
                "Motivasi belajar perlu ditingkatkan untuk hasil yang lebih baik"
            ],
            "expectations": [
                "Diharapkan dapat mencapai nilai 60-70",
                "Dapat meningkatkan performa dengan bimbingan yang tepat",
                "Potensial untuk menjadi siswa yang lebih baik",
                "Membutuhkan dukungan dari guru dan keluarga"
            ],
            "next_steps": [
                "Tingkatkan jam belajar secara bertahap",
                "Ikuti bimbingan belajar tambahan",
                "Buat jadwal belajar yang teratur",
                "Tingkatkan motivasi diri"
            ]
        },
        3: {
            "title": "⚠️ Siswa Perlu Perhatian Khusus",
            "description": "Siswa ini membutuhkan perhatian dan bimbingan khusus untuk meningkatkan performa akademiknya.",
            "analysis": [
                evaluate_study_hours(student_data['Hours_Studied']),
                evaluate_attendance(student_data['Attendance']),
                "Motivasi belajar perlu ditingkatkan secara signifikan"
            ],
            "expectations": [
                "Diharapkan dapat mencapai nilai 55-65",
                "Membutuhkan program remedial",
                "Perlu monitoring intensif",
                "Dapat berkembang dengan dukungan yang tepat"
            ],
            "next_steps": [
                "Ikuti program remedial intensif",
                "Tingkatkan kehadiran di sekolah",
                "Dapatkan bimbingan belajar khusus",
                "Libatkan orang tua dalam proses belajar"
            ]
        },
        4: {
            "title": "🚨 Siswa Berisiko Tinggi - Perlu Intervensi",
            "description": "Siswa ini membutuhkan intervensi segera dan program khusus untuk mengatasi masalah akademiknya.",
            "analysis": [
                evaluate_study_hours(student_data['Hours_Studied']),
                evaluate_attendance(student_data['Attendance']),
                "Motivasi belajar sangat rendah dan perlu ditingkatkan"
            ],
            "expectations": [
                "Diharapkan dapat mencapai nilai minimal 50",
                "Membutuhkan program intervensi khusus",
                "Perlu monitoring sangat intensif",
                "Dapat berkembang dengan program yang tepat"
            ],
            "next_steps": [
                "Intervensi akademik segera",
                "Program bimbingan intensif",
                "Konseling psikologis",
                "Pendekatan personal dari guru",
                "Libatkan semua pihak (sekolah, keluarga, masyarakat)"
            ]
        }
    }
    
    return conclusions.get(cluster_id, {
        "title": "Cluster Tidak Diketahui",
        "description": "Kesimpulan tidak tersedia",
        "analysis": [],
        "expectations": [],
        "next_steps": []
    })

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Aplikasi Pengelompokan Siswa</h1>', unsafe_allow_html=True)
    st.markdown("### Aplikasi analisis cluster untuk mengelompokkan siswa berdasarkan karakteristik belajar")
    
    # Load data
    with st.spinner("Memuat dataset..."):
        df, features_scaled, feature_columns, label_encoders, scaler = load_and_preprocess_data()
    
    if df is None:
        st.error("Gagal memuat dataset!")
        return
    
    # Sidebar
    st.sidebar.header("📊 Informasi Dataset")
    st.sidebar.write(f"**Total Siswa:** {len(df)}")
    st.sidebar.write(f"**Fitur:** {len(df.columns)}")
    
    # Data Statistics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Statistik Data")
    
    # Calculate basic statistics
    avg_exam_score = df['Exam_Score'].mean()
    avg_hours_studied = df['Hours_Studied'].mean()
    avg_attendance = df['Attendance'].mean()
    
    st.sidebar.metric("Nilai Rata-rata", f"{avg_exam_score:.1f}")
    st.sidebar.metric("Jam Belajar Rata-rata", f"{avg_hours_studied:.1f} jam/minggu")
    st.sidebar.metric("Kehadiran Rata-rata", f"{avg_attendance:.1f}%")
    
    # Performance Distribution
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Distribusi Performa")
    
    high_performers = len(df[df['Exam_Score'] >= 72])
    above_avg = len(df[(df['Exam_Score'] >= 68) & (df['Exam_Score'] < 72)])
    average = len(df[(df['Exam_Score'] >= 64) & (df['Exam_Score'] < 68)])
    below_avg = len(df[(df['Exam_Score'] >= 60) & (df['Exam_Score'] < 64)])
    need_help = len(df[df['Exam_Score'] < 60])
    
    st.sidebar.write(f"🟢 Berprestasi Tinggi: {high_performers}")
    st.sidebar.write(f"🟢 Di Atas Rata-rata: {above_avg}")
    st.sidebar.write(f"🟡 Rata-rata: {average}")
    st.sidebar.write(f"🟠 Di Bawah Rata-rata: {below_avg}")
    st.sidebar.write(f"🔴 Perlu Bantuan: {need_help}")
    
    # Quick Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filter Cepat")
    
    # Filter by performance level
    performance_filter = st.sidebar.selectbox(
        "Filter Berdasarkan Performa",
        ["Semua", "Berprestasi Tinggi", "Di Atas Rata-rata", "Rata-rata", "Di Bawah Rata-rata", "Perlu Bantuan"]
    )
    
    # Filter by study hours
    study_hours_filter = st.sidebar.slider(
        "Filter Jam Belajar (jam/minggu)",
        min_value=int(df['Hours_Studied'].min()),
        max_value=int(df['Hours_Studied'].max()),
        value=(int(df['Hours_Studied'].min()), int(df['Hours_Studied'].max()))
    )
    
    # Filter by attendance
    attendance_filter = st.sidebar.slider(
        "Filter Kehadiran (%)",
        min_value=int(df['Attendance'].min()),
        max_value=int(df['Attendance'].max()),
        value=(int(df['Attendance'].min()), int(df['Attendance'].max()))
    )
    
    # Quick Actions
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Aksi Cepat")
    
    if st.sidebar.button("📊 Download Data Lengkap"):
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name="student_clustering_data.csv",
            mime="text/csv"
        )
    
    if st.sidebar.button("📈 Lihat Statistik Detail"):
        st.sidebar.write("**Statistik Detail:**")
        st.sidebar.write(f"• Nilai Tertinggi: {df['Exam_Score'].max():.1f}")
        st.sidebar.write(f"• Nilai Terendah: {df['Exam_Score'].min():.1f}")
        st.sidebar.write(f"• Standar Deviasi: {df['Exam_Score'].std():.1f}")
        st.sidebar.write(f"• Jam Belajar Tertinggi: {df['Hours_Studied'].max():.1f}")
        st.sidebar.write(f"• Kehadiran Tertinggi: {df['Attendance'].max():.1f}%")
    
    # Help Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("❓ Bantuan")
    
    with st.sidebar.expander("💡 Cara Menggunakan"):
        st.write("""
        1. **Tab Analisis Cluster**: Lihat pengelompokan semua siswa
        2. **Tab Prediksi Cluster**: Masukkan data siswa baru untuk prediksi
        3. **Filter**: Gunakan filter di sidebar untuk melihat data tertentu
        4. **Download**: Unduh data lengkap untuk analisis lebih lanjut
        """)
    
    with st.sidebar.expander("🎯 Tentang Cluster"):
        st.write("""
        • **Cluster 1** 🟢: Siswa Berprestasi Tinggi
        • **Cluster 2** 🟡: Siswa Konsisten dan Stabil
        • **Cluster 3** 🟠: Siswa Rata-rata dengan Potensi
        • **Cluster 4** 🔴: Siswa Perlu Perhatian Khusus
        • **Cluster 5** 🔴: Siswa Berisiko Tinggi
        """)
    
    # Main content
    tab1, tab2 = st.tabs(["📊 Analisis Cluster", "🎯 Prediksi Cluster"])
    
    with tab1:
        st.header("📊 Analisis Cluster Semua Data")
        
        # Perform clustering
        kmeans_model, cluster_labels = perform_clustering(features_scaled)
        
        # Add clusters to dataframe
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = cluster_labels
        
        # Display cluster distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Distribusi Cluster")
            cluster_counts = np.bincount(cluster_labels)
            cluster_df = pd.DataFrame({
                'Cluster': [f'Cluster {i+1}' for i in range(len(cluster_counts))],
                'Siswa': cluster_counts,
                'Persentase': cluster_counts / len(df) * 100
            })
            st.dataframe(cluster_df.round(2))
        
        with col2:
            # Cluster distribution chart
            fig = px.bar(
                cluster_df, 
                x='Cluster', 
                y='Siswa',
                title="Jumlah Siswa per Cluster",
                color='Cluster'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed cluster analysis
        st.subheader("🔍 Analisis Detail Setiap Cluster")
        
        for cluster_id in range(len(cluster_counts)):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
            
            avg_score = cluster_data['Exam_Score'].mean()
            avg_hours = cluster_data['Hours_Studied'].mean()
            avg_attendance = cluster_data['Attendance'].mean()
            
            performance_level, _ = classify_performance(avg_score)
            cluster_color = get_cluster_color(cluster_id)
            cluster_info = get_cluster_description(cluster_id, avg_score, avg_hours, avg_attendance, len(cluster_data))
            
            st.markdown(f"""
            <div class="cluster-box">
                <h3>{cluster_color} Cluster {cluster_id + 1}: {cluster_info['name']}</h3>
                <p><strong>Jumlah Siswa:</strong> {len(cluster_data)} ({len(cluster_data)/len(df)*100:.1f}%)</p>
                <p><strong>Nilai Rata-rata:</strong> {avg_score:.1f} ({performance_level})</p>
                <p><strong>Jam Belajar:</strong> {avg_hours:.1f} jam/minggu</p>
                <p><strong>Kehadiran:</strong> {avg_attendance:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Cluster description
            st.markdown(f"""
            <div class="cluster-description">
                <h4>📝 Deskripsi Cluster {cluster_id + 1}</h4>
                <p><strong>{cluster_info['description']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("🎯 Karakteristik:")
            for char in cluster_info['characteristics']:
                st.write(f"• {char}")
            
            st.subheader("💪 Kekuatan:")
            for strength in cluster_info['strengths']:
                st.write(f"• {strength}")
            
            st.subheader("💡 Rekomendasi:")
            for rec in cluster_info['recommendations']:
                st.write(f"• {rec}")
            
            st.markdown("---")  # Separator between clusters
    
    with tab2:
        st.header("🎯 Prediksi Cluster Siswa")
        st.markdown("Masukkan data siswa untuk memprediksi cluster yang sesuai dan mendapatkan kesimpulan")
        
        # Form for prediction
        with st.container():
            st.markdown('<div class="form-container">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📚 Data Akademik")
                hours_studied = st.slider("Jam Belajar per Minggu", 0, 40, 20, help="Masukkan jumlah jam belajar dalam seminggu")
                attendance = st.slider("Kehadiran (%)", 0, 100, 80, help="Masukkan persentase kehadiran di sekolah")
                previous_scores = st.slider("Nilai Sebelumnya", 0, 100, 70, help="Masukkan nilai ujian sebelumnya")
                sleep_hours = st.slider("Jam Tidur per Hari", 4, 12, 7, help="Masukkan jumlah jam tidur per hari")
                tutoring_sessions = st.slider("Sesi Bimbingan Belajar", 0, 10, 2, help="Masukkan jumlah sesi bimbingan belajar")
            
            with col2:
                st.subheader("🏠 Data Latar Belakang")
                parental_involvement = st.selectbox("Keterlibatan Orang Tua", ["Low", "Medium", "High"], help="Pilih tingkat keterlibatan orang tua dalam pendidikan")
                access_resources = st.selectbox("Akses ke Sumber Belajar", ["Low", "Medium", "High"], help="Pilih tingkat akses ke sumber belajar")
                motivation_level = st.selectbox("Tingkat Motivasi", ["Low", "Medium", "High"], help="Pilih tingkat motivasi belajar")
                internet_access = st.selectbox("Akses Internet", ["Yes", "No"], help="Pilih ketersediaan akses internet")
                family_income = st.selectbox("Pendapatan Keluarga", ["Low", "Medium", "High"], help="Pilih tingkat pendapatan keluarga")
            
            # Additional features
            st.subheader("📋 Informasi Tambahan")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                extracurricular = st.selectbox("Kegiatan Ekstrakurikuler", ["Yes", "No"], help="Pilih apakah mengikuti kegiatan ekstrakurikuler")
                teacher_quality = st.selectbox("Kualitas Guru", ["Low", "Medium", "High"], help="Pilih penilaian terhadap kualitas guru")
            
            with col4:
                peer_influence = st.selectbox("Pengaruh Teman", ["Positive", "Neutral", "Negative"], help="Pilih pengaruh teman terhadap belajar")
                physical_activity = st.slider("Aktivitas Fisik (jam/minggu)", 0, 10, 3, help="Masukkan jam aktivitas fisik per minggu")
            
            with col5:
                learning_disabilities = st.selectbox("Kesulitan Belajar", ["Yes", "No"], help="Pilih apakah memiliki kesulitan belajar")
                school_type = st.selectbox("Jenis Sekolah", ["Public", "Private"], help="Pilih jenis sekolah")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction button
        if st.button("🔮 Prediksi Cluster", type="primary"):
            with st.spinner("Memprediksi cluster..."):
                # Prepare student data
                student_data = {
                    'Hours_Studied': hours_studied,
                    'Attendance': attendance,
                    'Parental_Involvement': parental_involvement,
                    'Access_to_Resources': access_resources,
                    'Extracurricular_Activities': extracurricular,
                    'Sleep_Hours': sleep_hours,
                    'Previous_Scores': previous_scores,
                    'Motivation_Level': motivation_level,
                    'Internet_Access': internet_access,
                    'Tutoring_Sessions': tutoring_sessions,
                    'Family_Income': family_income,
                    'Teacher_Quality': teacher_quality,
                    'School_Type': school_type,
                    'Peer_Influence': peer_influence,
                    'Physical_Activity': physical_activity,
                    'Learning_Disabilities': learning_disabilities,
                    'Parental_Education_Level': 'High School',  # Default value
                    'Distance_from_Home': 'Near',  # Default value
                    'Gender': 'Male'  # Default value
                }
                
                # Encode categorical variables
                for col in ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
                           'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                           'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
                           'Parental_Education_Level', 'Distance_from_Home', 'Gender']:
                    if col in label_encoders:
                        student_data[col] = label_encoders[col].transform([student_data[col]])[0]
                
                # Predict cluster
                predicted_cluster = predict_cluster(student_data, kmeans_model, scaler, feature_columns)
                
                # Get cluster info
                cluster_data = df_with_clusters[df_with_clusters['Cluster'] == predicted_cluster]
                avg_score = cluster_data['Exam_Score'].mean()
                avg_hours = cluster_data['Hours_Studied'].mean()
                avg_attendance = cluster_data['Attendance'].mean()
                
                performance_level, _ = classify_performance(avg_score)
                learning_style, style_color, style_desc = classify_learning_style(hours_studied, attendance, student_data['Motivation_Level'])
                risk_level, risk_color, risk_desc = classify_risk_level(previous_scores, attendance, hours_studied)
                
                # Get conclusion
                conclusion = get_prediction_conclusion(predicted_cluster, student_data, avg_score, avg_hours, avg_attendance)
                
                # Display results
                st.success(f"🎯 **Prediksi Selesai!**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 Cluster yang Diprediksi</h4>
                        <h3>Cluster {predicted_cluster + 1}</h3>
                        <p>Mirip dengan {len(cluster_data)} siswa</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 Tingkat Performa</h4>
                        <h3>{risk_color} {risk_level}</h3>
                        <p>Nilai Rata-rata: {avg_score:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📚 Gaya Belajar</h4>
                        <h3>{style_color} {learning_style}</h3>
                        <p>{style_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk assessment
                st.markdown(f"""
                <div class="cluster-box">
                    <h4>⚠️ Penilaian Risiko</h4>
                    <h3>{risk_color} {risk_level}</h3>
                    <p>{risk_desc}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed conclusion
                st.markdown(f"""
                <div class="conclusion-box">
                    <h3>{conclusion['title']}</h3>
                    <p><strong>{conclusion['description']}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("📊 Analisis Data:")
                for analysis in conclusion['analysis']:
                    st.write(f"• {analysis}")
                
                st.subheader("🎯 Harapan:")
                for expectation in conclusion['expectations']:
                    st.write(f"• {expectation}")
                
                st.subheader("📋 Langkah Selanjutnya:")
                for step in conclusion['next_steps']:
                    st.write(f"• {step}")

if __name__ == "__main__":
    main() 