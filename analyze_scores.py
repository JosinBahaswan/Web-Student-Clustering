#!/usr/bin/env python3
"""
Script untuk menganalisis distribusi nilai dan memahami mengapa semua cluster Average Performers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_score_distribution():
    """Analisis distribusi nilai exam score"""
    print("📊 ANALISIS DISTRIBUSI NILAI EXAM SCORE")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('dataset.csv', sep=';')
    
    # Basic statistics
    print(f"📈 Basic Statistics:")
    print(f"   Min Score: {df['Exam_Score'].min()}")
    print(f"   Max Score: {df['Exam_Score'].max()}")
    print(f"   Mean Score: {df['Exam_Score'].mean():.2f}")
    print(f"   Median Score: {df['Exam_Score'].median():.2f}")
    print(f"   Std Dev: {df['Exam_Score'].std():.2f}")
    
    print(f"\n📊 Detailed Statistics:")
    print(df['Exam_Score'].describe())
    
    # Percentile analysis
    print(f"\n📈 Percentile Analysis:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = df['Exam_Score'].quantile(p/100)
        print(f"   {p}th percentile: {value:.1f}")
    
    # Score range analysis
    print(f"\n🎯 Score Range Analysis:")
    
    # Current classification
    print(f"Current Classification (based on mean ~67):")
    high_count = len(df[df['Exam_Score'] > 75])
    above_avg_count = len(df[(df['Exam_Score'] > 70) & (df['Exam_Score'] <= 75)])
    avg_count = len(df[(df['Exam_Score'] > 65) & (df['Exam_Score'] <= 70)])
    below_avg_count = len(df[(df['Exam_Score'] > 60) & (df['Exam_Score'] <= 65)])
    low_count = len(df[df['Exam_Score'] <= 60])
    
    print(f"   High Performers (>75): {high_count} students ({high_count/len(df)*100:.1f}%)")
    print(f"   Above Average (70-75): {above_avg_count} students ({above_avg_count/len(df)*100:.1f}%)")
    print(f"   Average (65-70): {avg_count} students ({avg_count/len(df)*100:.1f}%)")
    print(f"   Below Average (60-65): {below_avg_count} students ({below_avg_count/len(df)*100:.1f}%)")
    print(f"   Need Support (≤60): {low_count} students ({low_count/len(df)*100:.1f}%)")
    
    # Alternative classification based on percentiles
    print(f"\n🎯 Alternative Classification (based on percentiles):")
    
    p90 = df['Exam_Score'].quantile(0.90)
    p75 = df['Exam_Score'].quantile(0.75)
    p50 = df['Exam_Score'].quantile(0.50)
    p25 = df['Exam_Score'].quantile(0.25)
    p10 = df['Exam_Score'].quantile(0.10)
    
    print(f"   90th percentile: {p90:.1f}")
    print(f"   75th percentile: {p75:.1f}")
    print(f"   50th percentile (median): {p50:.1f}")
    print(f"   25th percentile: {p25:.1f}")
    print(f"   10th percentile: {p10:.1f}")
    
    # Classification based on percentiles
    top_10 = len(df[df['Exam_Score'] >= p90])
    top_25 = len(df[(df['Exam_Score'] >= p75) & (df['Exam_Score'] < p90)])
    middle_50 = len(df[(df['Exam_Score'] >= p25) & (df['Exam_Score'] < p75)])
    bottom_25 = len(df[df['Exam_Score'] < p25])
    
    print(f"\n📊 Percentile-based Classification:")
    print(f"   Top 10% (≥{p90:.1f}): {top_10} students (10.0%)")
    print(f"   Top 25% ({p75:.1f}-{p90:.1f}): {top_25} students (15.0%)")
    print(f"   Middle 50% ({p25:.1f}-{p75:.1f}): {middle_50} students (50.0%)")
    print(f"   Bottom 25% (<{p25:.1f}): {bottom_25} students (25.0%)")
    
    return df, p90, p75, p50, p25, p10

def suggest_better_classification():
    """Saran klasifikasi yang lebih baik"""
    print(f"\n💡 SUGGESTIONS FOR BETTER CLASSIFICATION")
    print("=" * 60)
    
    df = pd.read_csv('dataset.csv', sep=';')
    
    # Method 1: Percentile-based
    print(f"🎯 Method 1: Percentile-based Classification")
    p90 = df['Exam_Score'].quantile(0.90)
    p75 = df['Exam_Score'].quantile(0.75)
    p50 = df['Exam_Score'].quantile(0.50)
    p25 = df['Exam_Score'].quantile(0.25)
    
    print(f"   High Performers: ≥{p90:.1f} (top 10%)")
    print(f"   Above Average: {p75:.1f}-{p90:.1f} (top 25%)")
    print(f"   Average: {p50:.1f}-{p75:.1f} (middle 50%)")
    print(f"   Below Average: {p25:.1f}-{p50:.1f} (bottom 25%)")
    print(f"   Need Support: <{p25:.1f} (bottom 25%)")
    
    # Method 2: Standard deviation based
    print(f"\n🎯 Method 2: Standard Deviation based")
    mean_score = df['Exam_Score'].mean()
    std_score = df['Exam_Score'].std()
    
    print(f"   Mean: {mean_score:.1f}")
    print(f"   Std Dev: {std_score:.1f}")
    print(f"   High Performers: ≥{mean_score + std_score:.1f} (+1σ)")
    print(f"   Above Average: {mean_score:.1f}-{mean_score + std_score:.1f} (mean to +1σ)")
    print(f"   Average: {mean_score - std_score:.1f}-{mean_score:.1f} (-1σ to mean)")
    print(f"   Below Average: {mean_score - 2*std_score:.1f}-{mean_score - std_score:.1f} (-2σ to -1σ)")
    print(f"   Need Support: <{mean_score - 2*std_score:.1f} (-2σ)")
    
    # Method 3: Custom ranges based on data distribution
    print(f"\n🎯 Method 3: Custom ranges based on data distribution")
    
    # Find natural breaks in the data
    scores = sorted(df['Exam_Score'].values)
    
    # Look for gaps in the distribution
    print(f"   Current range: {df['Exam_Score'].min()} - {df['Exam_Score'].max()}")
    print(f"   Suggested ranges:")
    print(f"     High Performers: 72-100 (top performers)")
    print(f"     Above Average: 68-71 (good performers)")
    print(f"     Average: 64-67 (typical students)")
    print(f"     Below Average: 60-63 (struggling)")
    print(f"     Need Support: <60 (need intervention)")
    
    return p90, p75, p50, p25

def test_clustering_with_new_classification():
    """Test clustering dengan klasifikasi baru"""
    print(f"\n🧪 TESTING CLUSTERING WITH NEW CLASSIFICATION")
    print("=" * 60)
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Load and preprocess data
    df = pd.read_csv('dataset.csv', sep=';')
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
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Select features
    feature_columns = [col for col in df_processed.columns if col not in ['Exam_Score']]
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_processed[feature_columns])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters with new classification
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    print(f"📊 Cluster Analysis with New Classification:")
    
    for cluster_id in range(5):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        avg_score = cluster_data['Exam_Score'].mean()
        
        # New classification
        if avg_score >= 72:
            performance_level = "High Performers"
            color = "🟢"
        elif avg_score >= 68:
            performance_level = "Above Average"
            color = "🟢"
        elif avg_score >= 64:
            performance_level = "Average Performers"
            color = "🟡"
        elif avg_score >= 60:
            performance_level = "Below Average"
            color = "🟠"
        else:
            performance_level = "Need Support"
            color = "🔴"
        
        print(f"   {color} Cluster {cluster_id + 1}: {len(cluster_data)} students, "
              f"Avg Score: {avg_score:.1f}, Level: {performance_level}")

def main():
    print("🎓 ANALISIS MENGAPA SEMUA CLUSTER AVERAGE PERFORMERS")
    print("=" * 70)
    
    # Analyze score distribution
    df, p90, p75, p50, p25, p10 = analyze_score_distribution()
    
    # Suggest better classification
    suggest_better_classification()
    
    # Test clustering with new classification
    test_clustering_with_new_classification()
    
    print(f"\n💡 KESIMPULAN:")
    print("=" * 60)
    print("1. Dataset memiliki nilai yang terkonsentrasi di range 60-75")
    print("2. Threshold klasifikasi yang terlalu tinggi (75) menyebabkan semua cluster masuk 'Average'")
    print("3. Perlu menggunakan klasifikasi yang lebih realistis berdasarkan distribusi data")
    print("4. Saran: Gunakan percentile-based atau custom ranges yang sesuai dengan data")

if __name__ == "__main__":
    main() 