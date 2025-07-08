#!/usr/bin/env python3
"""
Debug script untuk mengatasi masalah clustering dan menambahkan kesimpulan
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset():
    """Analisis dataset untuk memahami masalah clustering"""
    print("🔍 ANALISIS DATASET")
    print("=" * 50)
    
    # Load dataset
    df = pd.read_csv('dataset.csv', sep=';')
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Check data types
    print("\n📈 Data Types:")
    print(df.dtypes)
    
    # Check for missing values
    print("\n❓ Missing Values:")
    print(df.isnull().sum())
    
    # Check unique values in categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\n🏷️ Categorical columns: {list(categorical_cols)}")
    
    for col in categorical_cols:
        print(f"\n{col}: {df[col].unique()}")
    
    return df

def preprocess_data(df):
    """Preprocess data dengan penanganan yang lebih baik"""
    print("\n🛠️ PREPROCESSING DATA")
    print("=" * 50)
    
    df_processed = df.copy()
    
    # Handle missing values
    print("🔧 Handling missing values...")
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].fillna('Unknown')
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Handle categorical variables
    print("🔧 Encoding categorical variables...")
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"   {col}: {len(le.classes_)} categories")
    
    # Select features for clustering
    feature_columns = [col for col in df_processed.columns if col not in ['Exam_Score']]
    print(f"\n📊 Features for clustering: {len(feature_columns)}")
    print(f"📋 Feature names: {feature_columns}")
    
    # Scale features
    print("🔧 Scaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_processed[feature_columns])
    
    print(f"✅ Preprocessing complete!")
    print(f"📊 Scaled features shape: {features_scaled.shape}")
    
    return df_processed, features_scaled, feature_columns, label_encoders, scaler

def test_clustering(features_scaled, n_clusters_range=range(2, 11)):
    """Test clustering dengan berbagai jumlah cluster"""
    print(f"\n🧪 TESTING CLUSTERING")
    print("=" * 50)
    
    results = {}
    
    for k in n_clusters_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Check if all clusters are used
            unique_clusters = np.unique(cluster_labels)
            cluster_counts = np.bincount(cluster_labels)
            
            results[k] = {
                'inertia': kmeans.inertia_,
                'cluster_counts': cluster_counts,
                'unique_clusters': len(unique_clusters),
                'min_cluster_size': cluster_counts.min(),
                'max_cluster_size': cluster_counts.max()
            }
            
            print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Clusters={len(unique_clusters)}, "
                  f"Min={cluster_counts.min()}, Max={cluster_counts.max()}")
            
        except Exception as e:
            print(f"K={k}: Error - {e}")
    
    return results

def find_optimal_k(results):
    """Temukan K optimal menggunakan elbow method"""
    print(f"\n🎯 FINDING OPTIMAL K")
    print("=" * 50)
    
    k_values = list(results.keys())
    inertias = [results[k]['inertia'] for k in k_values]
    
    # Calculate elbow point
    if len(inertias) > 1:
        # Calculate second derivative
        second_derivative = np.gradient(np.gradient(inertias))
        elbow_idx = np.argmax(second_derivative)
        optimal_k = k_values[elbow_idx]
    else:
        optimal_k = k_values[0]
    
    print(f"📊 K values tested: {k_values}")
    print(f"📊 Inertias: {[f'{i:.2f}' for i in inertias]}")
    print(f"🎯 Optimal K: {optimal_k}")
    
    return optimal_k, k_values, inertias

def create_detailed_analysis(df, cluster_labels, feature_columns):
    """Buat analisis detail dan kesimpulan"""
    print(f"\n📊 DETAILED CLUSTER ANALYSIS")
    print("=" * 50)
    
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Analyze each cluster
    cluster_summaries = []
    
    for cluster_id in range(len(np.unique(cluster_labels))):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        
        print(f"\n🔍 CLUSTER {cluster_id + 1} ({len(cluster_data)} students)")
        print("-" * 30)
        
        # Key metrics
        avg_hours = cluster_data['Hours_Studied'].mean()
        avg_attendance = cluster_data['Attendance'].mean()
        avg_scores = cluster_data['Exam_Score'].mean()
        avg_previous = cluster_data['Previous_Scores'].mean()
        
        print(f"📚 Average Study Hours: {avg_hours:.1f}")
        print(f"📅 Average Attendance: {avg_attendance:.1f}%")
        print(f"📊 Average Exam Score: {avg_scores:.1f}")
        print(f"📈 Average Previous Score: {avg_previous:.1f}")
        
        # Determine cluster characteristics
        if avg_scores > 75:
            performance_level = "High Performers"
            color = "🟢"
            recommendations = [
                "Maintain excellent study habits",
                "Continue with current learning strategies",
                "Consider advanced courses or enrichment programs",
                "Mentor other students"
            ]
        elif avg_scores > 65:
            performance_level = "Average Performers"
            color = "🟡"
            recommendations = [
                "Increase study hours gradually",
                "Improve attendance if below 80%",
                "Seek additional tutoring if needed",
                "Set specific academic goals"
            ]
        else:
            performance_level = "Need Support"
            color = "🔴"
            recommendations = [
                "Significant increase in study hours needed",
                "Regular attendance improvement required",
                "Consider intensive tutoring programs",
                "Parental involvement enhancement",
                "Regular progress monitoring"
            ]
        
        # Additional characteristics
        high_attendance = (cluster_data['Attendance'] > 80).sum() / len(cluster_data) * 100
        high_study_hours = (cluster_data['Hours_Studied'] > 20).sum() / len(cluster_data) * 100
        
        print(f"{color} Performance Level: {performance_level}")
        print(f"📊 High Attendance Rate: {high_attendance:.1f}%")
        print(f"📚 High Study Hours Rate: {high_study_hours:.1f}%")
        
        # Store summary
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
        
        cluster_summaries.append(summary)
        
        print("🎯 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    return cluster_summaries

def generate_overall_conclusions(cluster_summaries, df):
    """Generate kesimpulan keseluruhan"""
    print(f"\n📋 OVERALL CONCLUSIONS")
    print("=" * 50)
    
    total_students = len(df)
    
    print(f"📊 TOTAL ANALYSIS SUMMARY")
    print(f"   Total Students: {total_students}")
    print(f"   Number of Clusters: {len(cluster_summaries)}")
    
    # Performance distribution
    high_performers = sum(1 for s in cluster_summaries if s['performance_level'] == "High Performers")
    avg_performers = sum(1 for s in cluster_summaries if s['performance_level'] == "Average Performers")
    need_support = sum(1 for s in cluster_summaries if s['performance_level'] == "Need Support")
    
    print(f"\n🎯 PERFORMANCE DISTRIBUTION")
    print(f"   🟢 High Performers: {high_performers} clusters")
    print(f"   🟡 Average Performers: {avg_performers} clusters")
    print(f"   🔴 Need Support: {need_support} clusters")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS")
    
    # Find best performing cluster
    best_cluster = max(cluster_summaries, key=lambda x: x['avg_scores'])
    print(f"   🏆 Best Performing Cluster: Cluster {best_cluster['cluster_id']}")
    print(f"      Average Score: {best_cluster['avg_scores']:.1f}")
    print(f"      Study Hours: {best_cluster['avg_hours']:.1f}")
    print(f"      Attendance: {best_cluster['avg_attendance']:.1f}%")
    
    # Find cluster needing most support
    worst_cluster = min(cluster_summaries, key=lambda x: x['avg_scores'])
    print(f"   ⚠️ Cluster Needing Most Support: Cluster {worst_cluster['cluster_id']}")
    print(f"      Average Score: {worst_cluster['avg_scores']:.1f}")
    print(f"      Study Hours: {worst_cluster['avg_hours']:.1f}")
    print(f"      Attendance: {worst_cluster['avg_attendance']:.1f}%")
    
    # Overall recommendations
    print(f"\n🎯 OVERALL RECOMMENDATIONS")
    print("   1. Implement targeted intervention programs for low-performing clusters")
    print("   2. Develop mentorship programs pairing high and low performers")
    print("   3. Create study groups based on cluster characteristics")
    print("   4. Regular monitoring of student progress across clusters")
    print("   5. Parent-teacher conferences focused on cluster-specific strategies")
    
    return cluster_summaries

def main():
    print("🎓 STUDENT BEHAVIOR CLUSTERING - DEBUG & ANALYSIS")
    print("=" * 60)
    
    # Analyze dataset
    df = analyze_dataset()
    
    # Preprocess data
    df_processed, features_scaled, feature_columns, label_encoders, scaler = preprocess_data(df)
    
    # Test clustering
    results = test_clustering(features_scaled)
    
    # Find optimal K
    optimal_k, k_values, inertias = find_optimal_k(results)
    
    # Perform final clustering
    print(f"\n🎯 PERFORMING FINAL CLUSTERING WITH K={optimal_k}")
    print("=" * 50)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    print(f"✅ Clustering successful!")
    print(f"📊 Cluster distribution: {np.bincount(cluster_labels)}")
    print(f"📊 Inertia: {kmeans.inertia_:.2f}")
    
    # Create detailed analysis
    cluster_summaries = create_detailed_analysis(df, cluster_labels, feature_columns)
    
    # Generate overall conclusions
    generate_overall_conclusions(cluster_summaries, df)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("=" * 50)
    print("📁 Results saved in memory")
    print("🚀 Ready to integrate with Streamlit app")

if __name__ == "__main__":
    main() 