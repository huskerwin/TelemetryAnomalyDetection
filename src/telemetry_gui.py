"""
Telemetry Anomaly Detection GUI Viewer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import os


class TelemetryViewer:
    """GUI application for viewing telemetry anomaly detection data."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Telemetry Anomaly Detection Viewer")
        self.root.geometry("1400x900")
        
        # Data storage
        self.train_data = {}
        self.test_data = {}
        self.labels_df = None
        self.data_dir = None
        
        # Colors
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI components."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top frame - controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Data directory selection
        ttk.Label(control_frame, text="Data Directory:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.dir_var = tk.StringVar(value="")
        self.dir_entry = ttk.Entry(control_frame, textvariable=self.dir_var, width=50)
        self.dir_entry.grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Browse", command=self.browse_directory).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Load Data", command=self.load_data).grid(row=0, column=3, padx=5)
        
        # Channel selection
        ttk.Label(control_frame, text="Channel:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(control_frame, textvariable=self.channel_var, state="readonly", width=30)
        self.channel_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        self.channel_combo.bind("<<ComboboxSelected>>", self.on_channel_select)
        
        # View type selection
        ttk.Label(control_frame, text="View:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.view_var = tk.StringVar(value="Overview")
        view_options = ["Overview", "Training Only", "Test Only", "Distribution", "Time Series Features", "Rolling Stats"]
        self.view_combo = ttk.Combobox(control_frame, textvariable=self.view_var, values=view_options, state="readonly", width=20)
        self.view_combo.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        self.view_combo.bind("<<ComboboxSelected>>", self.on_channel_select)
        
        # Feature selection for multi-dimensional data
        ttk.Label(control_frame, text="Feature:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.feature_var = tk.StringVar(value="0")
        self.feature_spin = ttk.Spinbox(control_frame, from_=0, to=24, textvariable=self.feature_var, width=5)
        self.feature_spin.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        self.feature_spin.bind("<Return>", self.on_channel_select)
        
        # Filter by anomaly type
        ttk.Label(control_frame, text="Filter Type:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        self.filter_var = tk.StringVar(value="All")
        filter_options = ["All", "Point", "Contextual", "Mixed"]
        self.filter_combo = ttk.Combobox(control_frame, textvariable=self.filter_var, values=filter_options, state="readonly", width=15)
        self.filter_combo.grid(row=2, column=3, padx=5, pady=5, sticky=tk.W)
        self.filter_combo.bind("<<ComboboxSelected>>", self.filter_channels)
        
        # Info label
        self.info_var = tk.StringVar(value="Load a data directory to begin")
        ttk.Label(control_frame, textvariable=self.info_var, foreground="gray").grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=5, pady=5)
        
        # Canvas frame for matplotlib
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Navigation toolbar
        toolbar_frame = ttk.Frame(canvas_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, pady=(5, 0))
    
    def browse_directory(self):
        """Open directory browser."""
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            self.dir_var.set(directory)
    
    def load_data(self):
        """Load telemetry data from selected directory."""
        directory = self.dir_var.get()
        if not directory:
            messagebox.showwarning("Warning", "Please select a data directory")
            return
        
        self.data_dir = Path(directory)
        self.status_var.set("Loading data...")
        self.root.update()
        
        try:
            # Check for train/test subdirectories (NASA SMAP & MSL format)
            train_dir = self.data_dir / "train"
            test_dir = self.data_dir / "test"
            
            if train_dir.exists() and test_dir.exists():
                # Load .npy files
                self.train_data = {}
                self.test_data = {}
                
                for npy_file in train_dir.glob("*.npy"):
                    channel_id = npy_file.stem
                    self.train_data[channel_id] = np.load(npy_file)
                
                for npy_file in test_dir.glob("*.npy"):
                    channel_id = npy_file.stem
                    self.test_data[channel_id] = np.load(npy_file)
                
                # Load labels if available
                label_file = self.data_dir / "labeled_anomalies.csv"
                if label_file.exists():
                    self.labels_df = pd.read_csv(label_file)
                    self.parse_anomaly_types()
                
                # Update channel combo
                channels = sorted(self.train_data.keys())
                self.channel_combo['values'] = channels
                self.all_channels = channels
                if channels:
                    self.channel_combo.current(0)
                
                self.info_var.set(f"Loaded {len(channels)} channels from {self.data_dir.name}")
                self.status_var.set(f"Data loaded: {len(channels)} channels")
                self.on_channel_select()
            
            else:
                # Try loading CSV files
                csv_files = list(self.data_dir.glob("*.csv"))
                if csv_files:
                    self.load_csv_data(csv_files[0])
                else:
                    messagebox.showerror("Error", "No valid data found. Expected train/ and test/ directories or CSV files.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Error loading data")
    
    def load_csv_data(self, csv_file):
        """Load CSV telemetry data."""
        df = pd.read_csv(csv_file)
        self.train_data = {}
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.train_data[col] = df[col].values
        
        # Update UI
        channels = sorted(self.train_data.keys())
        self.channel_combo['values'] = channels
        self.all_channels = channels
        if channels:
            self.channel_combo.current(0)
        
        self.info_var.set(f"Loaded {len(channels)} columns from {csv_file.name}")
        self.status_var.set(f"Data loaded: {len(channels)} columns")
        self.on_channel_select()
    
    def parse_anomaly_types(self):
        """Parse anomaly types from labels."""
        if self.labels_df is None:
            return
        
        def get_type(s):
            if pd.isna(s):
                return 'unknown'
            s = str(s).lower()
            if 'point' in s and 'contextual' not in s:
                return 'point'
            elif 'contextual' in s and 'point' not in s:
                return 'contextual'
            elif 'point' in s and 'contextual' in s:
                return 'mixed'
            return 'other'
        
        id_col = 'chan_id' if 'chan_id' in self.labels_df.columns else 'channel_id'
        if 'class' in self.labels_df.columns:
            self.labels_df['anomaly_type'] = self.labels_df['class'].apply(get_type)
    
    def filter_channels(self, event=None):
        """Filter channels by anomaly type."""
        filter_type = self.filter_var.get()
        
        if filter_type == "All" or self.labels_df is None:
            self.channel_combo['values'] = self.all_channels
        else:
            filter_type_lower = filter_type.lower()
            filtered = self.labels_df[self.labels_df['anomaly_type'] == filter_type_lower]['chan_id'].tolist()
            self.channel_combo['values'] = sorted(filtered)
        
        if self.channel_combo['values']:
            self.channel_combo.current(0)
            self.on_channel_select()
    
    def get_anomaly_type(self, channel_id):
        """Get anomaly type for a channel."""
        if self.labels_df is None:
            return ""
        
        id_col = 'chan_id' if 'chan_id' in self.labels_df.columns else 'channel_id'
        channel_labels = self.labels_df[self.labels_df[id_col] == channel_id]
        
        if not channel_labels.empty and 'class' in channel_labels.columns:
            class_val = str(channel_labels.iloc[0]['class']).lower()
            if 'point' in class_val and 'contextual' not in class_val:
                return " (Point Anomaly)"
            elif 'contextual' in class_val and 'point' not in class_val:
                return " (Contextual Anomaly)"
            elif 'point' in class_val and 'contextual' in class_val:
                return " (Mixed Anomaly)"
        return ""
    
    def get_anomaly_sequences(self, channel_id):
        """Get anomaly sequences for a channel."""
        if self.labels_df is None:
            return []
        
        id_col = 'chan_id' if 'chan_id' in self.labels_df.columns else 'channel_id'
        channel_labels = self.labels_df[self.labels_df[id_col] == channel_id]
        
        if not channel_labels.empty:
            for _, row in channel_labels.iterrows():
                try:
                    seq = row.get('anomaly_sequences', [])
                    if isinstance(seq, str):
                        seq = eval(seq)
                    return seq
                except:
                    pass
        return []
    
    def get_feature_index(self):
        """Get selected feature index."""
        try:
            return int(self.feature_var.get())
        except:
            return 0
    
    def on_channel_select(self, event=None):
        """Handle channel selection."""
        channel_id = self.channel_var.get()
        if not channel_id:
            return
        
        view_type = self.view_var.get()
        feature_idx = self.get_feature_index()
        
        self.fig.clear()
        
        try:
            if view_type == "Overview":
                self.plot_overview(channel_id, feature_idx)
            elif view_type == "Training Only":
                self.plot_training(channel_id, feature_idx)
            elif view_type == "Test Only":
                self.plot_test(channel_id, feature_idx)
            elif view_type == "Distribution":
                self.plot_distribution(channel_id, feature_idx)
            elif view_type == "Time Series Features":
                self.plot_time_series_features(channel_id, feature_idx)
            elif view_type == "Rolling Stats":
                self.plot_rolling_stats(channel_id, feature_idx)
            
            self.canvas.draw()
            self.status_var.set(f"Viewing: {channel_id} - {view_type}")
        
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
    
    def plot_overview(self, channel_id, feature_idx):
        """Plot channel overview."""
        train_data = self.train_data.get(channel_id)
        test_data = self.test_data.get(channel_id)
        
        if train_data is None:
            return
        
        # Get feature data
        if train_data.ndim > 1:
            train_values = train_data[:, feature_idx]
            test_values = test_data[:, feature_idx] if test_data is not None else None
        else:
            train_values = train_data
            test_values = test_data
        
        anomaly_type = self.get_anomaly_type(channel_id)
        anomaly_seqs = self.get_anomaly_sequences(channel_id)
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.5])
        
        # Training data
        ax1 = self.fig.add_subplot(gs[0])
        ax1.plot(train_values, color=self.colors[0], alpha=0.8, linewidth=0.8)
        ax1.set_title(f'Training Data - {channel_id}', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Test data with anomalies
        if test_values is not None:
            ax2 = self.fig.add_subplot(gs[1])
            ax2.plot(test_values, color=self.colors[1], alpha=0.8, linewidth=0.8)
            
            # Add anomaly regions
            for seq in anomaly_seqs:
                try:
                    start, end = seq
                    class_val = str(self.labels_df[self.labels_df['chan_id'] == channel_id].iloc[0].get('class', '')).lower()
                    if 'point' in class_val and 'contextual' not in class_val:
                        color = '#FF6B6B'
                        label = 'Point Anomaly'
                    elif 'contextual' in class_val and 'point' not in class_val:
                        color = '#4ECDC4'
                        label = 'Contextual Anomaly'
                    else:
                        color = '#FFD93D'
                        label = 'Mixed Anomaly'
                    ax2.axvspan(start, end, alpha=0.4, color=color, label=label)
                except:
                    pass
            
            ax2.set_title(f'Test Data - {channel_id}{anomaly_type}', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Value')
            handles, labels = ax2.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax2.legend(by_label.values(), by_label.keys(), loc='upper right')
            ax2.grid(True, alpha=0.3)
        
        # Distribution
        ax3 = self.fig.add_subplot(gs[2])
        ax3.hist(train_values, bins=50, alpha=0.6, color=self.colors[0], label='Train', density=True)
        if test_values is not None:
            ax3.hist(test_values, bins=50, alpha=0.6, color=self.colors[1], label='Test', density=True)
        ax3.set_title('Value Distribution', fontsize=10)
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
    
    def plot_training(self, channel_id, feature_idx):
        """Plot training data only."""
        train_data = self.train_data.get(channel_id)
        if train_data is None:
            return
        
        values = train_data[:, feature_idx] if train_data.ndim > 1 else train_data
        
        ax = self.fig.add_subplot(111)
        ax.plot(values, color=self.colors[0], alpha=0.8, linewidth=0.8)
        ax.set_title(f'Training Data - {channel_id}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
    
    def plot_test(self, channel_id, feature_idx):
        """Plot test data with anomalies."""
        test_data = self.test_data.get(channel_id)
        if test_data is None:
            return
        
        values = test_data[:, feature_idx] if test_data.ndim > 1 else test_data
        anomaly_type = self.get_anomaly_type(channel_id)
        anomaly_seqs = self.get_anomaly_sequences(channel_id)
        
        ax = self.fig.add_subplot(111)
        ax.plot(values, color=self.colors[1], alpha=0.8, linewidth=0.8)
        
        # Add anomaly regions
        for seq in anomaly_seqs:
            try:
                start, end = seq
                class_val = str(self.labels_df[self.labels_df['chan_id'] == channel_id].iloc[0].get('class', '')).lower()
                if 'point' in class_val and 'contextual' not in class_val:
                    color = '#FF6B6B'
                elif 'contextual' in class_val and 'point' not in class_val:
                    color = '#4ECDC4'
                else:
                    color = '#FFD93D'
                ax.axvspan(start, end, alpha=0.4, color=color)
            except:
                pass
        
        ax.set_title(f'Test Data - {channel_id}{anomaly_type}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
    
    def plot_distribution(self, channel_id, feature_idx):
        """Plot value distribution."""
        train_data = self.train_data.get(channel_id)
        test_data = self.test_data.get(channel_id)
        
        if train_data is None:
            return
        
        train_values = train_data[:, feature_idx] if train_data.ndim > 1 else train_data
        test_values = test_data[:, feature_idx] if test_data is not None and test_data.ndim > 1 else test_data
        
        ax = self.fig.add_subplot(111)
        ax.hist(train_values, bins=50, alpha=0.6, color=self.colors[0], label='Train', density=True)
        if test_values is not None:
            ax.hist(test_values, bins=50, alpha=0.6, color=self.colors[1], label='Test', density=True)
        
        ax.set_title(f'Value Distribution - {channel_id}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Train: mean={np.mean(train_values):.3f}, std={np.std(train_values):.3f}"
        if test_values is not None:
            stats_text += f"\nTest: mean={np.mean(test_values):.3f}, std={np.std(test_values):.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.fig.tight_layout()
    
    def plot_time_series_features(self, channel_id, feature_idx):
        """Plot time series analysis features."""
        train_data = self.train_data.get(channel_id)
        test_data = self.test_data.get(channel_id)
        
        if train_data is None:
            return
        
        train_values = train_data[:, feature_idx] if train_data.ndim > 1 else train_data
        test_values = test_data[:, feature_idx] if test_data is not None and test_data.ndim > 1 else test_data
        
        if test_values is not None:
            combined = np.concatenate([train_values, test_values])
        else:
            combined = train_values
        
        gs = self.fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])
        
        # Original signal
        ax1 = self.fig.add_subplot(gs[0])
        ax1.plot(combined, color=self.colors[0], alpha=0.8, linewidth=0.8)
        if test_values is not None:
            ax1.axvline(x=len(train_values), color='gray', linestyle='--', alpha=0.5, label='Train/Test Split')
        ax1.set_title(f'Time Series Analysis - {channel_id}', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rolling statistics
        window = 50
        rolling_mean = pd.Series(combined).rolling(window=window, center=True).mean()
        rolling_std = pd.Series(combined).rolling(window=window, center=True).std()
        
        ax2 = self.fig.add_subplot(gs[1])
        ax2.plot(rolling_mean, color=self.colors[1], linewidth=1.5, label='Rolling Mean')
        ax2.fill_between(range(len(combined)), rolling_mean - rolling_std, 
                        rolling_mean + rolling_std, alpha=0.2, color=self.colors[1])
        ax2.set_ylabel('Rolling Stats')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # FFT
        ax3 = self.fig.add_subplot(gs[2])
        fft = np.fft.fft(combined)
        freqs = np.fft.fftfreq(len(combined))
        ax3.plot(freqs[:len(freqs)//2], np.abs(fft)[:len(fft)//2], color=self.colors[2])
        ax3.set_xlabel('Frequency')
        ax3.set_ylabel('Magnitude')
        ax3.set_title('Frequency Spectrum', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Autocorrelation
        ax4 = self.fig.add_subplot(gs[3])
        autocorr = np.correlate(combined - np.mean(combined), combined - np.mean(combined), mode='full')
        autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
        ax4.plot(autocorr[:200], color=self.colors[3])
        ax4.set_xlabel('Lag')
        ax4.set_ylabel('Autocorrelation')
        ax4.set_title('Autocorrelation Function', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
    
    def plot_rolling_stats(self, channel_id, feature_idx):
        """Plot rolling statistics."""
        train_data = self.train_data.get(channel_id)
        test_data = self.test_data.get(channel_id)
        
        if train_data is None:
            return
        
        train_values = train_data[:, feature_idx] if train_data.ndim > 1 else train_data
        test_values = test_data[:, feature_idx] if test_data is not None and test_data.ndim > 1 else test_data
        
        if test_values is not None:
            combined = np.concatenate([train_values, test_values])
        else:
            combined = train_values
        
        windows = [20, 50, 100]
        
        gs = self.fig.add_gridspec(len(windows) + 1, 1)
        
        # Original signal
        ax0 = self.fig.add_subplot(gs[0])
        ax0.plot(combined, color=self.colors[0], alpha=0.6, linewidth=0.5)
        if test_values is not None:
            ax0.axvline(x=len(train_values), color='gray', linestyle='--', alpha=0.5)
        ax0.set_title(f'Rolling Statistics - {channel_id}', fontsize=12, fontweight='bold')
        ax0.set_ylabel('Value')
        ax0.grid(True, alpha=0.3)
        
        # Rolling stats for different windows
        for i, window in enumerate(windows):
            ax = self.fig.add_subplot(gs[i + 1])
            rolling_mean = pd.Series(combined).rolling(window=window, center=True).mean()
            rolling_std = pd.Series(combined).rolling(window=window, center=True).std()
            
            ax.plot(rolling_mean, color=self.colors[i + 1], linewidth=1.5, label=f'Window={window}')
            ax.fill_between(range(len(combined)), rolling_mean - rolling_std, 
                           rolling_mean + rolling_std, alpha=0.2, color=self.colors[i + 1])
            ax.set_ylabel(f'W={window}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = TelemetryViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()