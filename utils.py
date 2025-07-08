import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE

def create_advanced_visualizations(df, cluster_labels, feature_columns):
    """
    Create advanced visualizations for clustering analysis
    """
    
    # Add cluster labels to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Check number of features for PCA
    n_features = len(feature_columns)
    n_samples = len(df)
    
    # 1. 3D Scatter Plot using PCA (only if we have enough features)
    fig_3d = None
    if n_features >= 3 and n_samples >= 3:
        try:
            from sklearn.decomposition import PCA
            pca_3d = PCA(n_components=min(3, n_features))
            features_3d = pca_3d.fit_transform(df[feature_columns])
            
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=features_3d[:, 0],
                y=features_3d[:, 1] if features_3d.shape[1] > 1 else [0] * len(features_3d),
                z=features_3d[:, 2] if features_3d.shape[1] > 2 else [0] * len(features_3d),
                mode='markers',
                marker=dict(
                    size=8,
                    color=cluster_labels,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[f'Student {i+1}, Cluster {c+1}' for i, c in enumerate(cluster_labels)],
                hovertemplate='<b>%{text}</b><br>' +
                              'PC1: %{x:.2f}<br>' +
                              'PC2: %{y:.2f}<br>' +
                              'PC3: %{z:.2f}<extra></extra>'
            )])
            
            fig_3d.update_layout(
                title="3D Student Clusters (PCA)",
                scene=dict(
                    xaxis_title="Principal Component 1",
                    yaxis_title="Principal Component 2",
                    zaxis_title="Principal Component 3"
                ),
                width=800,
                height=600
            )
        except Exception as e:
            st.warning(f"Could not create 3D visualization: {e}")
            fig_3d = None
    
    # 2. t-SNE Visualization
    fig_tsne = None
    if n_features >= 2 and n_samples >= 2:
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
            features_tsne = tsne.fit_transform(df[feature_columns])
            
            fig_tsne = px.scatter(
                x=features_tsne[:, 0],
                y=features_tsne[:, 1],
                color=cluster_labels,
                title="Student Clusters (t-SNE Visualization)",
                labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
                color_continuous_scale='viridis'
            )
        except Exception as e:
            st.warning(f"Could not create t-SNE visualization: {e}")
            fig_tsne = None
    
    # 3. Feature Importance Radar Chart
    fig_radar = None
    if n_features >= 3:
        try:
            cluster_means = df_with_clusters.groupby('Cluster')[feature_columns].mean()
            
            # Normalize features for radar chart
            cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
            
            fig_radar = go.Figure()
            
            for cluster_id in range(len(cluster_means)):
                fig_radar.add_trace(go.Scatterpolar(
                    r=cluster_means_normalized.iloc[cluster_id].values,
                    theta=feature_columns,
                    fill='toself',
                    name=f'Cluster {cluster_id + 1}',
                    line_color=px.colors.qualitative.Set1[cluster_id % len(px.colors.qualitative.Set1)]
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Feature Importance by Cluster (Radar Chart)"
            )
        except Exception as e:
            st.warning(f"Could not create radar chart: {e}")
            fig_radar = None
    
    # 4. Box Plots for Key Features
    fig_box = None
    try:
        # Select key features that exist in the dataset
        available_key_features = []
        for feature in ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Exam_Score']:
            if feature in feature_columns:
                available_key_features.append(feature)
        
        if len(available_key_features) >= 1:
            n_plots = len(available_key_features)
            cols = min(2, n_plots)
            rows = (n_plots + cols - 1) // cols
            
            fig_box = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=available_key_features,
                specs=[[{"type": "box"}] * cols] * rows
            )
            
            for i, feature in enumerate(available_key_features):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                for cluster_id in range(len(df_with_clusters['Cluster'].unique())):
                    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id][feature]
                    fig_box.add_trace(
                        go.Box(
                            y=cluster_data,
                            name=f'Cluster {cluster_id + 1}',
                            marker_color=px.colors.qualitative.Set1[cluster_id % len(px.colors.qualitative.Set1)]
                        ),
                        row=row, col=col
                    )
            
            fig_box.update_layout(
                height=300 * rows,
                title_text="Distribution of Key Features by Cluster",
                showlegend=False
            )
    except Exception as e:
        st.warning(f"Could not create box plots: {e}")
        fig_box = None
    
    # 5. Correlation Heatmap for each cluster
    fig_corr = None
    try:
        n_clusters = len(df_with_clusters['Cluster'].unique())
        if n_clusters >= 1 and n_features >= 2:
            fig_corr = make_subplots(
                rows=1, cols=n_clusters,
                subplot_titles=[f'Cluster {i+1}' for i in range(n_clusters)],
                specs=[[{"type": "heatmap"}] * n_clusters]
            )
            
            for i, cluster_id in enumerate(range(n_clusters)):
                cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id][feature_columns]
                if len(cluster_data) > 1:  # Need at least 2 samples for correlation
                    corr_matrix = cluster_data.corr()
                    
                    fig_corr.add_trace(
                        go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale='RdBu_r',
                            zmid=0,
                            showscale=(i == n_clusters - 1)
                        ),
                        row=1, col=i+1
                    )
            
            fig_corr.update_layout(
                height=500,
                title_text="Feature Correlation by Cluster"
            )
    except Exception as e:
        st.warning(f"Could not create correlation heatmaps: {e}")
        fig_corr = None
    
    return fig_3d, fig_tsne, fig_radar, fig_box, fig_corr

def calculate_cluster_metrics(features_scaled, cluster_labels):
    """
    Calculate clustering quality metrics
    """
    try:
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    except:
        silhouette_avg = 0
    
    try:
        calinski_score = calinski_harabasz_score(features_scaled, cluster_labels)
    except:
        calinski_score = 0
    
    return {
        'silhouette_score': silhouette_avg,
        'calinski_harabasz_score': calinski_score
    }

def create_cluster_summary(df, cluster_labels):
    """
    Create detailed summary for each cluster
    """
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
        
        # Determine cluster characteristics
        if avg_scores > 75:
            performance_level = "High Performers"
            color = "🟢"
            recommendations = [
                "Maintain excellent study habits",
                "Continue with current learning strategies",
                "Consider advanced courses or enrichment programs"
            ]
        elif avg_scores > 65:
            performance_level = "Average Performers"
            color = "🟡"
            recommendations = [
                "Increase study hours gradually",
                "Improve attendance if below 80%",
                "Seek additional tutoring if needed"
            ]
        else:
            performance_level = "Need Support"
            color = "🔴"
            recommendations = [
                "Significant increase in study hours needed",
                "Regular attendance improvement required",
                "Consider intensive tutoring programs",
                "Parental involvement enhancement"
            ]
        
        # Calculate other characteristics
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
            'recommendations': recommendations,
            'high_attendance_pct': high_attendance,
            'high_study_hours_pct': high_study_hours
        }
        
        summaries.append(summary)
    
    return summaries

def create_performance_comparison(df, cluster_labels):
    """
    Create performance comparison charts
    """
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Performance comparison
    performance_data = df_with_clusters.groupby('Cluster').agg({
        'Hours_Studied': 'mean',
        'Attendance': 'mean',
        'Previous_Scores': 'mean',
        'Exam_Score': 'mean'
    }).round(2)
    
    # Create comparison bar chart
    fig_comparison = go.Figure()
    
    metrics = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Exam_Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        fig_comparison.add_trace(go.Bar(
            name=metric,
            x=[f'Cluster {j+1}' for j in range(len(performance_data))],
            y=performance_data[metric],
            marker_color=colors[i]
        ))
    
    fig_comparison.update_layout(
        title="Performance Comparison Across Clusters",
        xaxis_title="Cluster",
        yaxis_title="Average Score",
        barmode='group'
    )
    
    return fig_comparison, performance_data 