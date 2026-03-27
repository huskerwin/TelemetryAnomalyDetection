# Telemetry Anomaly Detection

Tools for visualizing and analyzing spacecraft telemetry anomaly detection datasets.

## Project Structure

```
TelemetryAnomalyDetection/
├── data/           # Place your dataset files here
├── src/            # Source code
├── notebooks/      # Jupyter notebooks
├── plots/          # Generated plots
└── README.md
```

## Supported Datasets

Place your downloaded datasets in the `data/` folder:

### 1. NASA SMAP & MSL
- Source: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
- Files: `.npy` files in `data/train/` and `data/test/` folders
- Labels: `labeled_anomalies.csv`

### 2. OPS-SAT
- Source: https://zenodo.org/record/8144166
- Files: CSV telemetry files

### 3. ADAPT (NASA)
- Source: https://data.nasa.gov/dataset/adapt-dataset
- Files: Sensor data with fault injections

### 4. Space Shuttle
- Source: https://catalog.data.gov/dataset/space-shuttle-main-propulsion-system-anomaly-detection-a-case-study
- Files: Engine telemetry data

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Visualize Data
```bash
python src/visualize.py --dataset nasa_smap_msl
```

### Run Jupyter Notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```