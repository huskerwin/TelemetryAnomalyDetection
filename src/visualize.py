"""
Telemetry Anomaly Detection Visualization Tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path
from datetime import datetime, timedelta


class TelemetryVisualizer:
    """
    Visualization tools for spacecraft telemetry anomaly detection.
    """
    
    def __init__(self, output_dir="plots"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    def load_nasa_smap_msl(self, data_dir):
        """
        Load NASA SMAP & MSL dataset (.npy files).
        
        Args:
            data_dir: Path to data directory containing train/ and test/ folders
            
        Returns:
            Dictionary with train_data, test_data, and labels
        """
        data_dir = Path(data_dir)
        result = {"train": {}, "test": {}, "labels": None}
        
        # Load labeled anomalies
        label_file = data_dir / "labeled_anomalies.csv"
        if label_file.exists():
            result["labels"] = pd.read_csv(label_file)
            print(f"Loaded {len(result['labels'])} labeled anomaly records")
        
        # Load train data
        train_dir = data_dir / "train"
        if train_dir.exists():
            for npy_file in train_dir.glob("*.npy"):
                channel_id = npy_file.stem
                result["train"][channel_id] = np.load(npy_file)
            print(f"Loaded {len(result['train'])} training channels")
        
        # Load test data
        test_dir = data_dir / "test"
        if test_dir.exists():
            for npy_file in test_dir.glob("*.npy"):
                channel_id = npy_file.stem
                result["test"][channel_id] = np.load(npy_file)
            print(f"Loaded {len(result['test'])} test channels")
        
        return result
    
    def load_csv_data(self, data_dir):
        """
        Load CSV-based telemetry data (for OPS-SAT, ADAPT, etc.).
        
        Args:
            data_dir: Path to directory containing CSV files
            
        Returns:
            Dictionary of DataFrames
        """
        data_dir = Path(data_dir)
        datasets = {}
        
        for csv_file in data_dir.glob("*.csv"):
            name = csv_file.stem
            datasets[name] = pd.read_csv(csv_file)
            print(f"Loaded {name}: {datasets[name].shape}")
        
        return datasets
    
    def plot_channel_overview(self, data, channel_id, labels_df=None, save=True):
        """
        Plot overview of a single telemetry channel.
        
        Args:
            data: Dictionary with train/test arrays or DataFrame
            channel_id: Channel identifier
            labels_df: DataFrame with anomaly labels
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Get train and test data
        if isinstance(data, dict) and "train" in data and "test" in data:
            train_data = data["train"].get(channel_id)
            test_data = data["test"].get(channel_id)
            
            if train_data is None or test_data is None:
                print(f"Channel {channel_id} not found")
                return
            
            # Handle multi-dimensional data
            if train_data.ndim > 1:
                train_values = train_data[:, 0]  # First column is telemetry
                test_values = test_data[:, 0]
            else:
                train_values = train_data
                test_values = test_data
            
            # Get anomaly type for title
            anomaly_type = ""
            if labels_df is not None:
                id_col = 'chan_id' if 'chan_id' in labels_df.columns else 'channel_id'
                if id_col in labels_df.columns:
                    channel_labels = labels_df[labels_df[id_col] == channel_id]
                    if not channel_labels.empty and 'class' in channel_labels.columns:
                        class_val = str(channel_labels.iloc[0]['class']).lower()
                        if 'point' in class_val and 'contextual' not in class_val:
                            anomaly_type = " (Point Anomaly)"
                        elif 'contextual' in class_val and 'point' not in class_val:
                            anomaly_type = " (Contextual Anomaly)"
                        elif 'point' in class_val and 'contextual' in class_val:
                            anomaly_type = " (Mixed Anomaly)"
            
            # Plot training data
            axes[0].plot(train_values, color=self.colors[0], alpha=0.8, linewidth=0.8)
            axes[0].set_title(f'Training Data - {channel_id}', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Value')
            axes[0].grid(True, alpha=0.3)
            
            # Plot test data
            axes[1].plot(test_values, color=self.colors[1], alpha=0.8, linewidth=0.8)
            
            # Add anomaly labels if available
            if labels_df is not None:
                # Handle different column names (channel_id or chan_id)
                id_col = 'chan_id' if 'chan_id' in labels_df.columns else 'channel_id'
                if id_col in labels_df.columns:
                    channel_labels = labels_df[labels_df[id_col] == channel_id]
                    if not channel_labels.empty:
                        for _, row in channel_labels.iterrows():
                            try:
                                anomaly_seq = eval(row['anomaly_sequences']) if isinstance(row['anomaly_sequences'], str) else row['anomaly_sequences']
                                # Get anomaly class
                                class_val = str(row.get('class', '')).lower()
                                for start, end in anomaly_seq:
                                    # Different colors for point vs contextual
                                    if 'point' in class_val and 'contextual' not in class_val:
                                        axes[1].axvspan(start, end, alpha=0.4, color='#FF6B6B', label='Point Anomaly')
                                    elif 'contextual' in class_val and 'point' not in class_val:
                                        axes[1].axvspan(start, end, alpha=0.4, color='#4ECDC4', label='Contextual Anomaly')
                                    else:
                                        axes[1].axvspan(start, end, alpha=0.4, color='#FFD93D', label='Mixed Anomaly')
                            except:
                                pass
            
            axes[1].set_title(f'Test Data - {channel_id}{anomaly_type}', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Value')
            # Remove duplicate labels
            handles, labels = axes[1].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[1].legend(by_label.values(), by_label.keys(), loc='upper right')
            axes[1].grid(True, alpha=0.3)
            
            # Plot combined distribution
            combined = np.concatenate([train_values, test_values])
            axes[2].hist(train_values, bins=50, alpha=0.6, color=self.colors[0], label='Train', density=True)
            axes[2].hist(test_values, bins=50, alpha=0.6, color=self.colors[1], label='Test', density=True)
            axes[2].set_title('Value Distribution', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Value')
            axes[2].set_ylabel('Density')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"channel_{channel_id}_overview.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def plot_multiple_channels(self, data, channel_ids, max_channels=5, save=True):
        """
        Plot multiple telemetry channels in one figure.
        
        Args:
            data: Dictionary with train/test arrays
            channel_ids: List of channel IDs to plot
            max_channels: Maximum number of channels to plot
            save: Whether to save the plot
        """
        n_channels = min(len(channel_ids), max_channels)
        
        fig, axes = plt.subplots(n_channels, 1, figsize=(14, 4*n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes]
        
        for i, channel_id in enumerate(channel_ids[:n_channels]):
            train_data = data["train"].get(channel_id)
            test_data = data["test"].get(channel_id)
            
            if train_data is None or test_data is None:
                continue
            
            train_values = train_data[:, 0] if train_data.ndim > 1 else train_data
            test_values = test_data[:, 0] if test_data.ndim > 1 else test_data
            
            combined = np.concatenate([train_values, test_values])
            
            axes[i].plot(combined, color=self.colors[i % len(self.colors)], alpha=0.8, linewidth=0.8)
            axes[i].axvline(x=len(train_values), color='gray', linestyle='--', alpha=0.5, label='Train/Test Split')
            axes[i].set_ylabel(f'{channel_id}', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            
            if i == 0:
                axes[i].set_title('Multiple Telemetry Channels', fontsize=12, fontweight='bold')
            if i == n_channels - 1:
                axes[i].set_xlabel('Time Step')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"multiple_channels.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_anomaly_statistics(self, labels_df, save=True):
        """
        Plot statistics about anomalies in the dataset.
        
        Args:
            labels_df: DataFrame with anomaly labels
            save: Whether to save the plot
        """
        if labels_df is None or labels_df.empty:
            print("No label data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Anomalies by spacecraft
        if 'spacecraft' in labels_df.columns:
            spacecraft_counts = labels_df['spacecraft'].value_counts()
            axes[0, 0].bar(spacecraft_counts.index, spacecraft_counts.values, color=self.colors[:len(spacecraft_counts)])
            axes[0, 0].set_title('Anomalies by Spacecraft', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Anomaly types (point vs contextual)
        if 'class' in labels_df.columns:
            class_counts = labels_df['class'].value_counts()
            axes[0, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                          colors=self.colors[:len(class_counts)])
            axes[0, 1].set_title('Anomaly Types', fontsize=12, fontweight='bold')
        
        # 3. Channel types distribution
        if 'channel_id' in labels_df.columns:
            channel_types = labels_df['channel_id'].str[0].value_counts()
            axes[1, 0].bar(channel_types.index, channel_types.values, color=self.colors[:len(channel_types)])
            axes[1, 0].set_title('Anomalies by Channel Type', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Channel Type (First Letter)')
            axes[1, 0].set_ylabel('Count')
        
        # 4. Number of values distribution
        if 'num_values' in labels_df.columns:
            axes[1, 1].hist(labels_df['num_values'], bins=30, color=self.colors[0], alpha=0.8, edgecolor='white')
            axes[1, 1].set_title('Distribution of Channel Lengths', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Number of Values')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_xscale('log')
        
        plt.suptitle('Anomaly Detection Dataset Statistics', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "anomaly_statistics.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, data, channel_ids=None, max_channels=10, save=True):
        """
        Plot correlation matrix between channels.
        
        Args:
            data: Dictionary with train/test arrays
            channel_ids: List of channel IDs (uses first max_channels if None)
            max_channels: Maximum channels to include
            save: Whether to save the plot
        """
        if channel_ids is None:
            channel_ids = list(data["train"].keys())[:max_channels]
        
        # Create DataFrame with all channels
        combined_data = {}
        for channel_id in channel_ids:
            train_data = data["train"].get(channel_id)
            if train_data is not None:
                values = train_data[:, 0] if train_data.ndim > 1 else train_data
                combined_data[channel_id] = values
        
        # Ensure all arrays have the same length
        min_len = min(len(v) for v in combined_data.values())
        for key in combined_data:
            combined_data[key] = combined_data[key][:min_len]
        
        df = pd.DataFrame(combined_data)
        corr = df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Channel Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "correlation_matrix.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_time_series_features(self, data, channel_id, window_size=50, save=True):
        """
        Plot time series features (rolling statistics, frequency analysis).
        
        Args:
            data: Dictionary with train/test arrays
            channel_id: Channel ID to analyze
            window_size: Window size for rolling statistics
            save: Whether to save the plot
        """
        train_data = data["train"].get(channel_id)
        test_data = data["test"].get(channel_id)
        
        if train_data is None or test_data is None:
            print(f"Channel {channel_id} not found")
            return
        
        train_values = train_data[:, 0] if train_data.ndim > 1 else train_data
        test_values = test_data[:, 0] if test_data.ndim > 1 else test_data
        combined = np.concatenate([train_values, test_values])
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame({'value': combined})
        df['rolling_mean'] = df['value'].rolling(window=window_size, center=True).mean()
        df['rolling_std'] = df['value'].rolling(window=window_size, center=True).std()
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Original signal
        axes[0].plot(df['value'], color=self.colors[0], alpha=0.7, linewidth=0.8)
        axes[0].axvline(x=len(train_values), color='gray', linestyle='--', alpha=0.5, label='Train/Test Split')
        axes[0].set_title(f'Time Series Analysis - {channel_id}', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Rolling statistics
        axes[1].plot(df['rolling_mean'], color=self.colors[1], linewidth=1.5, label='Rolling Mean')
        axes[1].fill_between(df.index, df['rolling_mean'] - df['rolling_std'], 
                           df['rolling_mean'] + df['rolling_std'], alpha=0.2, color=self.colors[1])
        axes[1].set_ylabel('Rolling Statistics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Frequency analysis (FFT)
        fft = np.fft.fft(combined)
        freqs = np.fft.fftfreq(len(combined))
        axes[2].plot(freqs[:len(freqs)//2], np.abs(fft)[:len(fft)//2], color=self.colors[2])
        axes[2].set_xlabel('Frequency')
        axes[2].set_ylabel('Magnitude')
        axes[2].set_title('Frequency Spectrum', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # Autocorrelation
        autocorr = np.correlate(combined - np.mean(combined), combined - np.mean(combined), mode='full')
        autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
        axes[3].plot(autocorr[:200], color=self.colors[3])
        axes[3].set_xlabel('Lag')
        axes[3].set_ylabel('Autocorrelation')
        axes[3].set_title('Autocorrelation Function', fontsize=10)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"ts_features_{channel_id}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def generate_summary_report(self, data, labels_df=None):
        """
        Generate a summary report of the dataset.
        
        Args:
            data: Dictionary with train/test arrays
            labels_df: DataFrame with anomaly labels
        """
        print("=" * 60)
        print("TELEMETRY DATASET SUMMARY")
        print("=" * 60)
        
        print(f"\nTraining channels: {len(data.get('train', {}))}")
        print(f"Test channels: {len(data.get('test', {}))}")
        
        if labels_df is not None:
            print(f"\nTotal anomalies: {len(labels_df)}")
            if 'spacecraft' in labels_df.columns:
                print(f"Spacecraft: {', '.join(labels_df['spacecraft'].unique())}")
            if 'class' in labels_df.columns:
                print(f"Anomaly types: {', '.join(labels_df['class'].unique())}")
        
        # Channel statistics
        if data.get('train'):
            sample_channel = list(data['train'].keys())[0]
            sample_data = data['train'][sample_channel]
            print(f"\nSample channel ({sample_channel}):")
            print(f"  Shape: {sample_data.shape}")
            print(f"  Min: {np.min(sample_data):.4f}")
            print(f"  Max: {np.max(sample_data):.4f}")
            print(f"  Mean: {np.mean(sample_data):.4f}")
            print(f"  Std: {np.std(sample_data):.4f}")
        
        print("\n" + "=" * 60)


def main():
    """
    Example usage of the visualizer.
    """
    visualizer = TelemetryVisualizer()
    
    # Check for data
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("No data directory found. Please create a 'data' directory and add your datasets.")
        return
    
    # List available data files
    print("\nAvailable data files:")
    for path in data_dir.rglob("*"):
        if path.is_file():
            print(f"  {path.relative_to(data_dir)}")
    
    # Try to load NASA SMAP & MSL data
    if (data_dir / "train").exists() and (data_dir / "test").exists():
        print("\nLoading NASA SMAP & MSL data...")
        data = visualizer.load_nasa_smap_msl(data_dir)
        
        if data['train']:
            # Get first channel
            channel_id = list(data['train'].keys())[0]
            print(f"\nVisualizing channel: {channel_id}")
            
            # Generate plots
            visualizer.plot_channel_overview(data, channel_id, data['labels'])
            visualizer.plot_anomaly_statistics(data['labels'])
            visualizer.plot_correlation_matrix(data)
            visualizer.generate_summary_report(data, data['labels'])
    
    # Try to load CSV data
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        print("\nFound CSV files:")
        for csv_file in csv_files:
            print(f"  Loading {csv_file.name}...")
            df = pd.read_csv(csv_file)
            print(f"    Shape: {df.shape}")
            print(f"    Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()