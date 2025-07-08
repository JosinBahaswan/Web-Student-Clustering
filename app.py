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
from utils import create_advanced_visualizations, calculate_cluster_metrics, create_cluster_summary, create_performance_comparison

# Set page config
st.set_page_config(
    page_title="Student Behavior Clustering",
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        # Try with semicolon separator first
        df = pd.read_csv('dataset.csv', sep=';')
        
        # If that fails, try with comma separator
        if df.shape[1] == 1:
            df = pd.read_csv('dataset.csv', sep=',')
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess data for clustering"""
    # Create a copy for preprocessing
    df_processed = df.copy()
    
    # Handle categorical variables
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    # Select features for clustering (excluding target variable if exists)
    feature_columns = [col for col in df_processed.columns if col not in ['Exam_Score']]
    
    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_processed[feature_columns])
    
    return df_processed, features_scaled, feature_columns, label_encoders, scaler

def find_optimal_k(features_scaled, max_k=10):
    """Find optimal number of clusters using elbow method"""
    inertias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertias.append(kmeans.inertia_)
    
    return K_range, inertias

def perform_clustering(features_scaled, n_clusters):
    """Perform K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    return kmeans, cluster_labels

def create_visualizations(df, cluster_labels, feature_columns):
    """Create various visualizations for clustering results"""
    
    # Add cluster labels to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # 1. PCA for 2D visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(df[feature_columns])
    
    fig_pca = px.scatter(
        x=features_2d[:, 0], 
        y=features_2d[:, 1],
        color=cluster_labels,
        title="Student Clusters (PCA Visualization)",
        labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
        color_continuous_scale='viridis'
    )
    
    # 2. Cluster distribution
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    fig_dist = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        title="Distribution of Students Across Clusters",
        labels={'x': 'Cluster', 'y': 'Number of Students'}
    )
    
    # 3. Feature importance by cluster
    cluster_means = df_with_clusters.groupby('Cluster')[feature_columns].mean()
    
    fig_heatmap = px.imshow(
        cluster_means.T,
        title="Feature Importance by Cluster (Normalized)",
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )
    
    return fig_pca, fig_dist, fig_heatmap, cluster_means

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Student Behavior Clustering Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### Mengelompokkan siswa berdasarkan perilaku belajar menggunakan K-Means Clustering")
    
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
    
    # Show sample data
    with st.expander("📋 View Sample Data"):
        st.dataframe(df.head(10))
        st.write(f"**Dataset Shape:** {df.shape}")
    
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
        n_clusters = st.sidebar.slider("Number of clusters (K):", 2, 10, 3)
    
    # Perform clustering
    with st.spinner(f"Performing clustering with {n_clusters} clusters..."):
        kmeans_model, cluster_labels = perform_clustering(features_scaled, n_clusters)
        
        # Check if clustering was successful
        unique_clusters = np.unique(cluster_labels)
        if len(unique_clusters) < n_clusters:
            st.warning(f"⚠️ Warning: Only {len(unique_clusters)} clusters were created instead of {n_clusters}")
            st.info("This might be due to data characteristics. Trying with different parameters...")
            
            # Try with different parameters
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
            cluster_labels = kmeans_model.fit_predict(features_scaled)
            
            unique_clusters = np.unique(cluster_labels)
            if len(unique_clusters) < n_clusters:
                st.error(f"❌ Still only {len(unique_clusters)} clusters. Consider using fewer clusters.")
                n_clusters = len(unique_clusters)
    
    # Display clustering results
    st.header("📈 Clustering Results")
    
    # Show cluster distribution
    cluster_counts = np.bincount(cluster_labels)
    st.subheader("📊 Cluster Distribution")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Number of Clusters", len(unique_clusters))
    with col3:
        st.metric("Average Cluster Size", f"{len(df) // len(unique_clusters)}")
    
    # Show cluster sizes
    cluster_df = pd.DataFrame({
        'Cluster': [f'Cluster {i+1}' for i in range(len(cluster_counts))],
        'Size': cluster_counts,
        'Percentage': cluster_counts / len(df) * 100
    })
    
    st.dataframe(cluster_df.round(2))
    
    # Calculate advanced metrics
    metrics = calculate_cluster_metrics(features_scaled, cluster_labels)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
    
    with col2:
        st.metric("Calinski Score", f"{metrics['calinski_harabasz_score']:.0f}")
    
    with col3:
        st.metric("Inertia", f"{kmeans_model.inertia_:.2f}")
    
    with col4:
        st.metric("Model Quality", "Good" if metrics['silhouette_score'] > 0.3 else "Fair")
    
    # Create advanced visualizations
    with st.spinner("Creating advanced visualizations..."):
        fig_3d, fig_tsne, fig_radar, fig_box, fig_corr = create_advanced_visualizations(
            df_processed, cluster_labels, feature_columns
        )
    
    # Display basic visualizations
    st.subheader("🎯 Basic Clustering Visualizations")
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
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        fig_dist = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title="Distribution of Students Across Clusters",
            labels={'x': 'Cluster', 'y': 'Number of Students'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Display advanced visualizations
    st.subheader("🚀 Advanced Visualizations")
    
    # 3D Plot
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # t-SNE if available
    if fig_tsne is not None:
        st.plotly_chart(fig_tsne, use_container_width=True)
    
    # Radar Chart
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Box Plots
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Correlation Heatmaps
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Cluster analysis
    st.header("🔍 Detailed Cluster Analysis")
    
    # Add clusters to original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Display cluster characteristics
    for cluster_id in range(n_clusters):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        
        st.markdown(f"### Cluster {cluster_id + 1} ({len(cluster_data)} students)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show key statistics for this cluster
            key_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Exam_Score']
            cluster_stats = cluster_data[key_features].describe()
            st.dataframe(cluster_stats.round(2))
        
        with col2:
            # Show cluster characteristics
            st.markdown("**Cluster Characteristics:**")
            
            # Calculate averages for key features
            avg_hours = cluster_data['Hours_Studied'].mean()
            avg_attendance = cluster_data['Attendance'].mean()
            avg_scores = cluster_data['Exam_Score'].mean()
            
            st.write(f"📚 Avg Study Hours: {avg_hours:.1f}")
            st.write(f"📅 Avg Attendance: {avg_attendance:.1f}%")
            st.write(f"📊 Avg Exam Score: {avg_scores:.1f}")
            
            # Determine cluster type based on characteristics
            if avg_scores > 75:
                cluster_type = "High Performers"
                color = "🟢"
            elif avg_scores > 70:
                cluster_type = "Above Average"
                color = "🟢"
            elif avg_scores > 65:
                cluster_type = "Average Performers"
                color = "🟡"
            elif avg_scores > 60:
                cluster_type = "Below Average"
                color = "🟠"
            else:
                cluster_type = "Need Support"
                color = "🔴"
            
            st.write(f"{color} **Cluster Type:** {cluster_type}")
    
    # Download results
    st.header("💾 Download Results")
    
    # Create downloadable dataframe
    results_df = df_with_clusters.copy()
    
    # Add cluster descriptions
    cluster_descriptions = []
    for cluster_id in range(n_clusters):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        avg_score = cluster_data['Exam_Score'].mean()
        
        if avg_score > 75:
            desc = "High Performers - Excellent academic performance"
        elif avg_score > 65:
            desc = "Average Performers - Moderate academic performance"
        else:
            desc = "Need Support - Requires additional academic support"
        
        cluster_descriptions.append(desc)
    
    results_df['Cluster_Description'] = results_df['Cluster'].map(
        {i: cluster_descriptions[i] for i in range(n_clusters)}
    )
    
    # Performance Comparison
    st.header("📊 Performance Comparison")
    
    # Create performance comparison
    fig_comparison, performance_data = create_performance_comparison(df_with_clusters, cluster_labels)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Show performance data table
    st.subheader("📋 Performance Metrics by Cluster")
    st.dataframe(performance_data.round(2))
    
    # Detailed Cluster Summary
    st.header("📝 Detailed Cluster Summary & Recommendations")
    
    # Create cluster summaries
    cluster_summaries = create_cluster_summary(df_with_clusters, cluster_labels)
    
    for summary in cluster_summaries:
        st.markdown(f"""
        <div class="cluster-info">
            <h3>{summary['color']} Cluster {summary['cluster_id']} - {summary['performance_level']}</h3>
            <p><strong>Size:</strong> {summary['size']} students</p>
            <p><strong>Average Study Hours:</strong> {summary['avg_hours']:.1f} hours</p>
            <p><strong>Average Attendance:</strong> {summary['avg_attendance']:.1f}%</p>
            <p><strong>Average Exam Score:</strong> {summary['avg_scores']:.1f}</p>
            <p><strong>High Attendance Rate:</strong> {summary['high_attendance_pct']:.1f}%</p>
            <p><strong>High Study Hours Rate:</strong> {summary['high_study_hours_pct']:.1f}%</p>
            
            <h4>🎯 Recommendations:</h4>
            <ul>
        """, unsafe_allow_html=True)
        
        for rec in summary['recommendations']:
            st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Clustering Results (CSV)",
        data=csv,
        file_name=f"student_clustering_results_{n_clusters}_clusters.csv",
        mime="text/csv"
    )
    
    # COMPREHENSIVE CONCLUSIONS SECTION
    st.header("📋 COMPREHENSIVE ANALYSIS CONCLUSIONS")
    
    # Overall summary
    st.subheader("🎯 Overall Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students Analyzed", len(df))
    with col2:
        st.metric("Optimal Clusters Found", n_clusters)
    with col3:
        st.metric("Analysis Quality", "Excellent" if metrics['silhouette_score'] > 0.4 else "Good" if metrics['silhouette_score'] > 0.2 else "Fair")
    
    # Cluster performance analysis
    st.subheader("📊 Cluster Performance Analysis")
    
    # Calculate performance metrics for each cluster
    performance_data = []
    for cluster_id in range(n_clusters):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        
        avg_score = cluster_data['Exam_Score'].mean()
        avg_hours = cluster_data['Hours_Studied'].mean()
        avg_attendance = cluster_data['Attendance'].mean()
        avg_previous = cluster_data['Previous_Scores'].mean()
        
        # Determine performance level
        if avg_score > 75:
            performance_level = "High Performers"
            color = "🟢"
        elif avg_score > 65:
            performance_level = "Average Performers"
            color = "🟡"
        else:
            performance_level = "Need Support"
            color = "🔴"
        
        performance_data.append({
            'Cluster': f'Cluster {cluster_id + 1}',
            'Size': len(cluster_data),
            'Avg_Score': avg_score,
            'Avg_Hours': avg_hours,
            'Avg_Attendance': avg_attendance,
            'Performance_Level': performance_level,
            'Color': color
        })
    
    # Display performance table
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df[['Cluster', 'Size', 'Avg_Score', 'Avg_Hours', 'Avg_Attendance', 'Performance_Level']].round(2))
    
    # Key findings
    st.subheader("🔍 Key Findings")
    
    # Find best and worst performing clusters
    best_cluster = max(performance_data, key=lambda x: x['Avg_Score'])
    worst_cluster = min(performance_data, key=lambda x: x['Avg_Score'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #e8f5e8; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745;'>
        <h4>🏆 Best Performing Cluster</h4>
        <p><strong>{best_cluster['Cluster']}</strong></p>
        <p>📊 Average Score: {best_cluster['Avg_Score']:.1f}</p>
        <p>📚 Study Hours: {best_cluster['Avg_Hours']:.1f}</p>
        <p>📅 Attendance: {best_cluster['Avg_Attendance']:.1f}%</p>
        <p>👥 Students: {best_cluster['Size']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color: #ffeaea; padding: 15px; border-radius: 10px; border-left: 5px solid #dc3545;'>
        <h4>⚠️ Cluster Needing Support</h4>
        <p><strong>{worst_cluster['Cluster']}</strong></p>
        <p>📊 Average Score: {worst_cluster['Avg_Score']:.1f}</p>
        <p>📚 Study Hours: {worst_cluster['Avg_Hours']:.1f}</p>
        <p>📅 Attendance: {worst_cluster['Avg_Attendance']:.1f}%</p>
        <p>👥 Students: {worst_cluster['Size']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    # Overall recommendations
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
    <h4>📋 Overall Strategic Recommendations:</h4>
    <ol>
        <li><strong>Targeted Intervention Programs:</strong> Implement specific programs for each cluster based on their characteristics</li>
        <li><strong>Mentorship Programs:</strong> Pair high-performing students with those needing support</li>
        <li><strong>Study Group Formation:</strong> Create study groups based on cluster similarities</li>
        <li><strong>Progress Monitoring:</strong> Regular tracking of student progress across clusters</li>
        <li><strong>Parent-Teacher Collaboration:</strong> Cluster-specific parent-teacher conferences</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Cluster-specific recommendations
    st.subheader("🎯 Cluster-Specific Recommendations")
    
    for perf in performance_data:
        if perf['Performance_Level'] == "High Performers":
            recommendations = [
                "Maintain excellent study habits and routines",
                "Consider advanced courses or enrichment programs",
                "Mentor other students in study groups",
                "Participate in academic competitions",
                "Set higher academic goals"
            ]
        elif perf['Performance_Level'] == "Average Performers":
            recommendations = [
                "Increase study hours gradually",
                "Improve attendance if below 80%",
                "Seek additional tutoring if needed",
                "Set specific academic goals",
                "Join study groups with high performers"
            ]
        else:  # Need Support
            recommendations = [
                "Significant increase in study hours needed",
                "Regular attendance improvement required",
                "Consider intensive tutoring programs",
                "Parental involvement enhancement",
                "Regular progress monitoring and feedback"
            ]
        
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <h4>{perf['Color']} {perf['Cluster']} - {perf['Performance_Level']}</h4>
        <p><strong>Size:</strong> {perf['Size']} students | <strong>Avg Score:</strong> {perf['Avg_Score']:.1f}</p>
        <h5>Recommendations:</h5>
        <ul>
        """, unsafe_allow_html=True)
        
        for rec in recommendations:
            st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # Action items
    st.subheader("📋 Immediate Action Items")
    
    st.markdown("""
    <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3;'>
    <h4>🚀 Priority Actions:</h4>
    <ol>
        <li><strong>Week 1:</strong> Review cluster analysis with teaching staff</li>
        <li><strong>Week 2:</strong> Design cluster-specific intervention programs</li>
        <li><strong>Week 3:</strong> Implement mentorship and study group programs</li>
        <li><strong>Week 4:</strong> Begin progress monitoring system</li>
        <li><strong>Month 2:</strong> Evaluate effectiveness and adjust strategies</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎓 Student Behavior Clustering Analysis | Built with Streamlit & Scikit-learn</p>
        <p>📊 Analysis completed with comprehensive insights and actionable recommendations</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 