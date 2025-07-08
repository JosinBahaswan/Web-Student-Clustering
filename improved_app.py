import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Student Behavior Clustering - Realistic Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .cluster-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .conclusion-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv('dataset.csv', sep=';')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess data for clustering"""
    df_processed = df.copy()
    
    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].fillna('Unknown')
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Handle categorical variables
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    # Select features for clustering
    feature_columns = [col for col in df_processed.columns if col not in ['Exam_Score']]
    
    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_processed[feature_columns])
    
    return df_processed, features_scaled, feature_columns, label_encoders, scaler

def find_optimal_k(features_scaled, max_k=10):
    """Find optimal number of clusters using elbow method"""
    inertias = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertias.append(kmeans.inertia_)
    
    return K_range, inertias

def perform_clustering(features_scaled, n_clusters):
    """Perform K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(features_scaled)
    return kmeans, cluster_labels

def classify_performance_realistic(avg_score):
    """Classify performance level based on realistic data distribution"""
    if avg_score >= 72:
        return "High Performers", "🟢", "#28a745"
    elif avg_score >= 68:
        return "Above Average", "🟢", "#28a745"
    elif avg_score >= 64:
        return "Average Performers", "🟡", "#ffc107"
    elif avg_score >= 60:
        return "Below Average", "🟠", "#fd7e14"
    else:
        return "Need Support", "🔴", "#dc3545"

def analyze_data_distribution(df):
    """Analyze the distribution of exam scores"""
    st.subheader("📊 Data Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic statistics
        st.markdown("**📈 Basic Statistics:**")
        stats_df = pd.DataFrame({
            'Metric': ['Min', 'Max', 'Mean', 'Median', 'Std Dev'],
            'Value': [
                df['Exam_Score'].min(),
                df['Exam_Score'].max(),
                f"{df['Exam_Score'].mean():.2f}",
                f"{df['Exam_Score'].median():.2f}",
                f"{df['Exam_Score'].std():.2f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True)
    
    with col2:
        # Percentiles
        st.markdown("**📊 Percentiles:**")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentiles_df = pd.DataFrame({
            'Percentile': [f"{p}%" for p in percentiles],
            'Score': [df['Exam_Score'].quantile(p/100) for p in percentiles]
        })
        st.dataframe(percentiles_df, hide_index=True)
    
    # Score distribution chart
    fig_dist = px.histogram(
        df, 
        x='Exam_Score', 
        nbins=30,
        title="Distribution of Exam Scores",
        labels={'Exam_Score': 'Exam Score', 'count': 'Number of Students'}
    )
    
    # Add vertical lines for classification boundaries
    fig_dist.add_vline(x=72, line_dash="dash", line_color="green", annotation_text="High Performers (≥72)")
    fig_dist.add_vline(x=68, line_dash="dash", line_color="lightgreen", annotation_text="Above Average (≥68)")
    fig_dist.add_vline(x=64, line_dash="dash", line_color="yellow", annotation_text="Average (≥64)")
    fig_dist.add_vline(x=60, line_dash="dash", line_color="orange", annotation_text="Below Average (≥60)")
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Classification explanation
    st.markdown("""
    <div class="warning-box">
    <h4>💡 Why All Clusters Were 'Average Performers'?</h4>
    <p><strong>Problem:</strong> The original classification used unrealistic thresholds:</p>
    <ul>
        <li>High Performers: >75 (only 1.2% of students)</li>
        <li>Average: 65-75 (51.4% of students)</li>
        <li>Need Support: <65 (48.6% of students)</li>
    </ul>
    <p><strong>Solution:</strong> Using realistic thresholds based on data distribution:</p>
    <ul>
        <li>🟢 High Performers: ≥72 (top 10%)</li>
        <li>🟢 Above Average: 68-71 (good performers)</li>
        <li>🟡 Average: 64-67 (typical students)</li>
        <li>🟠 Below Average: 60-63 (struggling)</li>
        <li>🔴 Need Support: <60 (need intervention)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def create_cluster_summary_realistic(df, cluster_labels):
    """Create detailed cluster summary with realistic classification"""
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    summaries = []
    
    for cluster_id in range(len(df_with_clusters['Cluster'].unique())):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        
        # Calculate key metrics
        avg_hours = cluster_data['Hours_Studied'].mean()
        avg_attendance = cluster_data['Attendance'].mean()
        avg_scores = cluster_data['Exam_Score'].mean()
        avg_previous = cluster_data['Previous_Scores'].mean()
        
        # Determine performance level with realistic classification
        performance_level, color, color_code = classify_performance_realistic(avg_scores)
        
        # Generate recommendations based on performance level
        if performance_level in ["High Performers", "Above Average"]:
            recommendations = [
                "Maintain excellent study habits and routines",
                "Consider advanced courses or enrichment programs",
                "Mentor other students in study groups",
                "Participate in academic competitions",
                "Set higher academic goals"
            ]
        elif performance_level == "Average Performers":
            recommendations = [
                "Increase study hours gradually",
                "Improve attendance if below 80%",
                "Seek additional tutoring if needed",
                "Set specific academic goals",
                "Join study groups with high performers"
            ]
        elif performance_level == "Below Average":
            recommendations = [
                "Significant increase in study hours needed",
                "Regular attendance improvement required",
                "Consider intensive tutoring programs",
                "Parental involvement enhancement",
                "Regular progress monitoring"
            ]
        else:  # Need Support
            recommendations = [
                "Immediate intervention required",
                "Intensive tutoring programs",
                "Daily attendance monitoring",
                "Parent-teacher collaboration",
                "Regular counseling sessions"
            ]
        
        # Calculate additional characteristics
        high_attendance = (cluster_data['Attendance'] > 80).sum() / len(cluster_data) * 100
        high_study_hours = (cluster_data['Hours_Studied'] > 20).sum() / len(cluster_data) * 100
        
        summary = {
            'cluster_id': cluster_id + 1,
            'size': len(cluster_data),
            'avg_hours': avg_hours,
            'avg_attendance': avg_attendance,
            'avg_scores': avg_scores,
            'avg_previous': avg_previous,
            'performance_level': performance_level,
            'color': color,
            'color_code': color_code,
            'recommendations': recommendations,
            'high_attendance_pct': high_attendance,
            'high_study_hours_pct': high_study_hours
        }
        
        summaries.append(summary)
    
    return summaries

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Student Behavior Clustering - Realistic Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### Analisis clustering dengan klasifikasi yang realistis berdasarkan distribusi data")
    
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load dataset. Please check if 'dataset.csv' exists in the current directory.")
        return
    
    # Display basic dataset info
    st.sidebar.header("📊 Dataset Information")
    st.sidebar.write(f"**Total Students:** {len(df)}")
    st.sidebar.write(f"**Features:** {len(df.columns)}")
    
    # Analyze data distribution
    analyze_data_distribution(df)
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        df_processed, features_scaled, feature_columns, label_encoders, scaler = preprocess_data(df)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Clustering Parameters")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Choose clustering method:",
        ["Auto-detect optimal K", "Manual K selection"]
    )
    
    if method == "Auto-detect optimal K":
        max_k = st.sidebar.slider("Maximum K to test:", 2, 15, 10)
        
        # Find optimal K
        with st.spinner("Finding optimal number of clusters..."):
            K_range, inertias = find_optimal_k(features_scaled, max_k)
        
        # Plot elbow curve
        fig_elbow = px.line(
            x=list(K_range), 
            y=inertias,
            title="Elbow Method for Optimal K",
            labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'}
        )
        fig_elbow.add_vline(x=K_range[np.argmin(np.gradient(inertias))], line_dash="dash", line_color="red")
        
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        # Auto-select optimal K
        optimal_k = K_range[np.argmin(np.gradient(inertias))]
        st.success(f"🎯 **Optimal number of clusters detected: {optimal_k}**")
        
        n_clusters = optimal_k
        
    else:
        n_clusters = st.sidebar.slider("Number of clusters (K):", 2, 10, 5)
    
    # Perform clustering
    with st.spinner(f"Performing clustering with {n_clusters} clusters..."):
        kmeans_model, cluster_labels = perform_clustering(features_scaled, n_clusters)
    
    # Display clustering results
    st.header("📈 Clustering Results with Realistic Classification")
    
    # Show cluster distribution
    cluster_counts = np.bincount(cluster_labels)
    st.subheader("📊 Cluster Distribution")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Number of Clusters", len(np.unique(cluster_labels)))
    with col3:
        st.metric("Average Cluster Size", f"{len(df) // len(np.unique(cluster_labels))}")
    
    # Create visualizations
    st.subheader("🎯 Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        # PCA 2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(df_processed[feature_columns])
        
        fig_pca = px.scatter(
            x=features_2d[:, 0], 
            y=features_2d[:, 1],
            color=cluster_labels,
            title="Student Clusters (PCA 2D)",
            labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_pca, use_container_width=True)
    
    with col2:
        # Cluster distribution
        cluster_df = pd.DataFrame({
            'Cluster': [f'Cluster {i+1}' for i in range(len(cluster_counts))],
            'Size': cluster_counts,
            'Percentage': cluster_counts / len(df) * 100
        })
        
        fig_dist = px.bar(
            x=cluster_df['Cluster'],
            y=cluster_df['Size'],
            title="Distribution of Students Across Clusters",
            labels={'x': 'Cluster', 'y': 'Number of Students'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Detailed Cluster Analysis with Realistic Classification
    st.header("🔍 Detailed Cluster Analysis")
    
    # Create cluster summaries with realistic classification
    cluster_summaries = create_cluster_summary_realistic(df, cluster_labels)
    
    # Display each cluster
    for summary in cluster_summaries:
        st.markdown(f"""
        <div class="cluster-info" style="border-left-color: {summary['color_code']};">
            <h3>{summary['color']} Cluster {summary['cluster_id']} - {summary['performance_level']}</h3>
            <p><strong>Size:</strong> {summary['size']} students ({summary['size']/len(df)*100:.1f}%)</p>
            <p><strong>Average Study Hours:</strong> {summary['avg_hours']:.1f} hours</p>
            <p><strong>Average Attendance:</strong> {summary['avg_attendance']:.1f}%</p>
            <p><strong>Average Exam Score:</strong> {summary['avg_scores']:.1f}</p>
            <p><strong>Average Previous Score:</strong> {summary['avg_previous']:.1f}</p>
            <p><strong>High Attendance Rate:</strong> {summary['high_attendance_pct']:.1f}%</p>
            <p><strong>High Study Hours Rate:</strong> {summary['high_study_hours_pct']:.1f}%</p>
            
            <h4>🎯 Recommendations:</h4>
            <ul>
        """, unsafe_allow_html=True)
        
        for rec in summary['recommendations']:
            st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # Performance Distribution
    st.header("📊 Performance Distribution")
    
    # Calculate performance distribution
    performance_counts = {}
    for summary in cluster_summaries:
        level = summary['performance_level']
        if level not in performance_counts:
            performance_counts[level] = 0
        performance_counts[level] += summary['size']
    
    # Create performance distribution chart
    perf_df = pd.DataFrame([
        {'Performance Level': level, 'Students': count, 'Percentage': count/len(df)*100}
        for level, count in performance_counts.items()
    ])
    
    fig_perf = px.pie(
        perf_df, 
        values='Students', 
        names='Performance Level',
        title="Distribution of Students by Performance Level (Realistic Classification)",
        color_discrete_map={
            'High Performers': '#28a745',
            'Above Average': '#20c997',
            'Average Performers': '#ffc107',
            'Below Average': '#fd7e14',
            'Need Support': '#dc3545'
        }
    )
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # Key findings
    st.header("🔍 Key Findings")
    
    # Find best and worst performing clusters
    best_cluster = max(cluster_summaries, key=lambda x: x['avg_scores'])
    worst_cluster = min(cluster_summaries, key=lambda x: x['avg_scores'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="conclusion-box">
        <h4>🏆 Best Performing Cluster</h4>
        <p><strong>{best_cluster['color']} Cluster {best_cluster['cluster_id']}</strong></p>
        <p>📊 Average Score: {best_cluster['avg_scores']:.1f}</p>
        <p>📚 Study Hours: {best_cluster['avg_hours']:.1f}</p>
        <p>📅 Attendance: {best_cluster['avg_attendance']:.1f}%</p>
        <p>👥 Students: {best_cluster['size']}</p>
        <p>🎯 Level: {best_cluster['performance_level']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="conclusion-box">
        <h4>⚠️ Cluster Needing Most Support</h4>
        <p><strong>{worst_cluster['color']} Cluster {worst_cluster['cluster_id']}</strong></p>
        <p>📊 Average Score: {worst_cluster['avg_scores']:.1f}</p>
        <p>📚 Study Hours: {worst_cluster['avg_hours']:.1f}</p>
        <p>📅 Attendance: {worst_cluster['avg_attendance']:.1f}%</p>
        <p>👥 Students: {worst_cluster['size']}</p>
        <p>🎯 Level: {worst_cluster['performance_level']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Strategic recommendations
    st.header("🎯 Strategic Recommendations")
    
    st.markdown("""
    <div class="conclusion-box">
    <h4>📋 Overall Strategic Recommendations:</h4>
    <ol>
        <li><strong>Targeted Intervention Programs:</strong> Implement specific programs for each cluster based on their realistic performance levels</li>
        <li><strong>Mentorship Programs:</strong> Pair high-performing students with those needing support</li>
        <li><strong>Study Group Formation:</strong> Create study groups based on cluster characteristics</li>
        <li><strong>Progress Monitoring:</strong> Regular tracking of student progress across clusters</li>
        <li><strong>Parent-Teacher Collaboration:</strong> Cluster-specific parent-teacher conferences</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Download results
    st.header("💾 Download Results")
    
    # Create downloadable dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Add cluster descriptions with realistic classification
    for summary in cluster_summaries:
        mask = df_with_clusters['Cluster'] == (summary['cluster_id'] - 1)
        df_with_clusters.loc[mask, 'Cluster_Description'] = f"{summary['performance_level']} - {summary['color']}"
        df_with_clusters.loc[mask, 'Performance_Level'] = summary['performance_level']
    
    # Download button
    csv = df_with_clusters.to_csv(index=False)
    st.download_button(
        label="📥 Download Realistic Clustering Results (CSV)",
        data=csv,
        file_name=f"realistic_student_clustering_results_{n_clusters}_clusters.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎓 Student Behavior Clustering - Realistic Analysis | Built with Streamlit & Scikit-learn</p>
        <p>📊 Analysis completed with realistic classification based on actual data distribution</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 