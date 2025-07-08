#!/usr/bin/env python3
"""
Test script untuk memverifikasi aplikasi clustering berfungsi dengan benar
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import os

def test_data_loading():
    """Test loading dataset"""
    print("🧪 Testing data loading...")
    try:
        df = pd.read_csv('dataset.csv', sep=';')
        print(f"✅ Dataset loaded successfully: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def test_preprocessing(df):
    """Test data preprocessing"""
    print("\n🧪 Testing data preprocessing...")
    try:
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
        
        print(f"✅ Preprocessing successful")
        print(f"📊 Features: {len(feature_columns)}")
        print(f"📊 Scaled features shape: {features_scaled.shape}")
        
        return df_processed, features_scaled, feature_columns, label_encoders, scaler
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return None, None, None, None, None

def test_clustering(features_scaled, n_clusters=3):
    """Test clustering"""
    print(f"\n🧪 Testing clustering with {n_clusters} clusters...")
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        print(f"✅ Clustering successful")
        print(f"📊 Cluster distribution: {np.bincount(cluster_labels)}")
        print(f"📊 Inertia: {kmeans.inertia_:.2f}")
        
        return kmeans, cluster_labels
    except Exception as e:
        print(f"❌ Error in clustering: {e}")
        return None, None

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'sklearn', 
        'plotly', 'matplotlib', 'seaborn'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            return False
    
    return True

def main():
    print("🎓 Student Behavior Clustering - Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Some required packages are missing!")
        print("Please install: pip install -r requirements.txt")
        return
    
    # Test data loading
    df = test_data_loading()
    if df is None:
        return
    
    # Test preprocessing
    df_processed, features_scaled, feature_columns, label_encoders, scaler = test_preprocessing(df)
    if df_processed is None:
        return
    
    # Test clustering
    kmeans, cluster_labels = test_clustering(features_scaled)
    if kmeans is None:
        return
    
    print("\n🎉 All tests passed! Application should work correctly.")
    print("\n📋 Summary:")
    print(f"   - Dataset: {df.shape[0]} students, {df.shape[1]} features")
    print(f"   - Processed features: {len(feature_columns)}")
    print(f"   - Clusters: {len(np.unique(cluster_labels))}")
    print(f"   - Model quality (inertia): {kmeans.inertia_:.2f}")
    
    print("\n🚀 Ready to run: streamlit run app.py")

if __name__ == "__main__":
    main() 