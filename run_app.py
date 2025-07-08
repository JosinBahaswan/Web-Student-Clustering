#!/usr/bin/env python3
"""
Runner script untuk aplikasi Student Behavior Clustering
"""

import subprocess
import sys
import os

def main():
    print("🎓 Starting Student Behavior Clustering Application...")
    print("=" * 50)
    
    # Check if requirements are installed
    try:
        import streamlit
        import pandas
        import sklearn
        import plotly
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return
    
    # Check if dataset exists
    if not os.path.exists('dataset.csv'):
        print("❌ dataset.csv not found in current directory")
        print("Please ensure dataset.csv is in the same directory as this script")
        return
    
    print("✅ Dataset found")
    print("🚀 Launching Streamlit application...")
    print("📱 The app will open in your browser at http://localhost:8501")
    print("=" * 50)
    
    # Run streamlit app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main() 