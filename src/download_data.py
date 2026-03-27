"""
Data Download Helper for Telemetry Anomaly Detection Datasets
"""

import os
import subprocess
from pathlib import Path


def print_download_instructions():
    """
    Print download instructions for all supported datasets.
    """
    print("=" * 70)
    print("TELEMETRY ANOMALY DETECTION DATASETS - DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    
    print("\n1. NASA SMAP & MSL Dataset")
    print("-" * 40)
    print("   Source: Kaggle")
    print("   URL: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
    print("   Download method:")
    print("   a) Manual download:")
    print("      1. Go to the URL above")
    print("      2. Click 'Download' button")
    print("      3. Extract the zip file")
    print("      4. Place contents in: data/")
    print("   b) Kaggle CLI (requires API key):")
    print("      kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
    print("      unzip nasa-anomaly-detection-dataset-smap-msl.zip -d data/")
    print("\n   Expected structure:")
    print("   data/")
    print("   +-- train/           # .npy files for each channel")
    print("   +-- test/            # .npy files for each channel")
    print("   +-- labeled_anomalies.csv")
    
    print("\n\n2. OPS-SAT Satellite Telemetry")
    print("-" * 40)
    print("   Source: Zenodo")
    print("   URL: https://zenodo.org/record/8144166")
    print("   Download method:")
    print("      1. Go to the URL above")
    print("      2. Download the dataset files")
    print("      3. Extract and place in: data/opssat/")
    
    print("\n\n3. ADAPT Dataset (NASA)")
    print("-" * 40)
    print("   Source: NASA Open Data Portal")
    print("   URL: https://data.nasa.gov/dataset/adapt-dataset")
    print("   Download method:")
    print("      1. Go to the URL above")
    print("      2. Click 'Export' -> 'CSV' or 'Download'")
    print("      3. Place files in: data/adapt/")
    
    print("\n\n4. Space Shuttle Main Propulsion System")
    print("-" * 40)
    print("   Source: NASA Data Catalog")
    print("   URL: https://catalog.data.gov/dataset/space-shuttle-main-propulsion-system-anomaly-detection-a-case-study")
    print("   Download method:")
    print("      1. Go to the URL above")
    print("      2. Find and download the dataset")
    print("      3. Place files in: data/space_shuttle/")
    
    print("\n\n5. Satellite Telemetry (GitHub)")
    print("-" * 40)
    print("   Source: GitHub")
    print("   URL: https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection")
    print("   Download method:")
    print("      git clone https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection.git")
    print("      cp -r Satellite-Telemetry-Anomaly-Detection/data/* data/")
    
    print("\n" + "=" * 70)
    print("After downloading, run the visualization notebook:")
    print("  jupyter notebook notebooks/analysis.ipynb")
    print("=" * 70)


def setup_kaggle_cli():
    """
    Setup Kaggle CLI for automated downloads.
    Requires Kaggle API credentials.
    """
    print("\nSetting up Kaggle CLI...")
    print("\n1. Go to: https://www.kaggle.com/settings")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Save kaggle.json to: ~/.kaggle/kaggle.json")
    print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
    print("\nThen you can download with:")
    print("  kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl")


def download_nasa_smap_msl():
    """
    Download NASA SMAP & MSL dataset using Kaggle CLI.
    """
    try:
        import kaggle
        print("Downloading NASA SMAP & MSL dataset...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
            "-p", "data"
        ], check=True)
        print("Download complete. Extract the zip file.")
    except ImportError:
        print("Kaggle CLI not installed. Install with: pip install kaggle")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from the URL above.")


if __name__ == "__main__":
    print_download_instructions()
    
    print("\n\nWould you like to setup Kaggle CLI for automated downloads? (y/n)")
    response = input("> ").strip().lower()
    if response == 'y':
        setup_kaggle_cli()