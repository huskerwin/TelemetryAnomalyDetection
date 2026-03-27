"""
Telemetry Anomaly Detection GUI Viewer - Modern UI
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path


# True dark color scheme
COLORS = {
    'bg_dark': '#0a0a0a',
    'bg_medium': '#111111',
    'bg_light': '#1a1a1a',
    'bg_lighter': '#242424',
    'border': '#2a2a2a',
    'accent': '#00d4ff',
    'accent_hover': '#33ddff',
    'text': '#e8e8e8',
    'text_secondary': '#666666',
    'success': '#00ff88',
    'warning': '#ffaa00',
    'point_anomaly': '#ff4444',
    'contextual_anomaly': '#00ff88',
    'mixed_anomaly': '#ffaa00',
    'chart_cyan': '#00d4ff',
    'chart_pink': '#ff6b9d',
    'chart_orange': '#ff9f43',
    'chart_red': '#ff4444',
    'chart_green': '#00ff88',
    'chart_purple': '#a855f7',
}

# Matplotlib colors for plots
CHART_COLORS = [
    COLORS['chart_cyan'],
    COLORS['chart_pink'],
    COLORS['chart_orange'],
    COLORS['chart_red'],
    COLORS['chart_green'],
]


class ModernButton(tk.Canvas):
    """Custom modern-looking button with hover effects."""
    
    def __init__(self, parent, text="", command=None, width=120, height=36,
                 bg_color=COLORS['accent'], hover_color=COLORS['accent_hover'],
                 text_color=COLORS['text'], font=('Segoe UI', 10), **kwargs):
        super().__init__(parent, width=width, height=height, bg=COLORS['bg_medium'],
                        highlightthickness=0, **kwargs)
        
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = font
        self.width = width
        self.height = height
        
        self._draw_button(text)
        
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Button-1>', self._on_click)
    
    def _draw_button(self, text, bg=None):
        """Draw the button."""
        if bg is None:
            bg = self.bg_color
        
        self.delete('all')
        
        # Draw rounded rectangle (simulated with arcs)
        r = 8
        x1, y1 = 2, 2
        x2, y2 = self.width - 4, self.height - 4
        
        self.create_arc(x1, y1, x1 + 2*r, y1 + 2*r, start=90, extent=90,
                       fill=bg, outline=bg)
        self.create_arc(x2 - 2*r, y1, x2, y1 + 2*r, start=0, extent=90,
                       fill=bg, outline=bg)
        self.create_arc(x1, y2 - 2*r, x1 + 2*r, y2, start=180, extent=90,
                       fill=bg, outline=bg)
        self.create_arc(x2 - 2*r, y2 - 2*r, x2, y2, start=270, extent=90,
                       fill=bg, outline=bg)
        
        # Fill middle
        self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=bg, outline=bg)
        self.create_rectangle(x1, y1 + r, x2, y2 - r, fill=bg, outline=bg)
        
        # Draw text
        self.create_text(self.width/2, self.height/2, text=text,
                        fill=self.text_color, font=self.font)
    
    def _on_enter(self, event):
        self._draw_button(self.itemcget(3, 'text'), self.hover_color)
    
    def _on_leave(self, event):
        self._draw_button(self.itemcget(3, 'text'), self.bg_color)
    
    def _on_click(self, event):
        if self.command:
            self.command()
    
    def config_text(self, text):
        """Update button text."""
        self._draw_button(text)


class TelemetryViewer:
    """Modern GUI application for viewing telemetry anomaly detection data."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Telemetry Anomaly Detection Viewer")
        self.root.geometry("1500x950")
        self.root.configure(bg=COLORS['bg_dark'])
        
        # Data storage
        self.train_data = {}
        self.test_data = {}
        self.labels_df = None
        self.data_dir = None
        self.all_channels = []
        
        # Configure styles
        self._configure_styles()
        
        # Setup GUI
        self.setup_gui()
        
        # Configure matplotlib for dark theme
        self._configure_matplotlib()
    
    def _configure_styles(self):
        """Configure ttk styles for modern look."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('.', background=COLORS['bg_dark'], foreground=COLORS['text'])
        
        # Frame styles
        style.configure('TFrame', background=COLORS['bg_dark'])
        style.configure('Card.TFrame', background=COLORS['bg_medium'])
        
        # Label styles
        style.configure('TLabel', background=COLORS['bg_dark'], foreground=COLORS['text'],
                       font=('Segoe UI', 10))
        style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'),
                       foreground=COLORS['text'])
        style.configure('Subheader.TLabel', font=('Segoe UI', 11),
                       foreground=COLORS['text_secondary'])
        style.configure('Card.TLabel', background=COLORS['bg_medium'],
                       foreground=COLORS['text'])
        style.configure('Accent.TLabel', foreground=COLORS['accent'])
        style.configure('Success.TLabel', foreground=COLORS['success'])
        
        # Entry styles
        style.configure('TEntry', fieldbackground=COLORS['bg_light'],
                       foreground=COLORS['text'], insertcolor=COLORS['text'])
        
        # Combobox styles
        style.configure('TCombobox', 
                       fieldbackground=COLORS['bg_light'],
                       background=COLORS['bg_lighter'],
                       foreground=COLORS['text'],
                       arrowcolor=COLORS['accent'],
                       selectbackground=COLORS['accent'],
                       selectforeground=COLORS['text'],
                       insertcolor=COLORS['text'])
        style.map('TCombobox',
                 fieldbackground=[('readonly', COLORS['bg_light'])],
                 selectbackground=[('readonly', COLORS['accent'])],
                 selectforeground=[('readonly', COLORS['text'])],
                 foreground=[('readonly', COLORS['text'])])
        
        # Spinbox styles
        style.configure('TSpinbox', fieldbackground=COLORS['bg_light'],
                       foreground=COLORS['text'])
        
        # LabelFrame styles
        style.configure('TLabelframe', background=COLORS['bg_medium'],
                       foreground=COLORS['text'])
        style.configure('TLabelframe.Label', background=COLORS['bg_medium'],
                       foreground=COLORS['accent'], font=('Segoe UI', 10, 'bold'))
        
        # Status bar
        style.configure('Status.TLabel', background=COLORS['bg_light'],
                       foreground=COLORS['text_secondary'], padding=(10, 5))
    
    def _configure_matplotlib(self):
        """Configure matplotlib for dark theme."""
        plt.style.use('dark_background')
        matplotlib.rcParams.update({
            'figure.facecolor': COLORS['bg_medium'],
            'axes.facecolor': COLORS['bg_light'],
            'axes.edgecolor': COLORS['text_secondary'],
            'axes.labelcolor': COLORS['text'],
            'text.color': COLORS['text'],
            'xtick.color': COLORS['text_secondary'],
            'ytick.color': COLORS['text_secondary'],
            'grid.color': COLORS['bg_dark'],
            'grid.alpha': 0.3,
        })
    
    def setup_gui(self):
        """Setup the GUI components with modern layout."""
        # Main container
        main_container = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header
        self._create_header(main_container)
        
        # Content area with sidebar and plot
        content_frame = tk.Frame(main_container, bg=COLORS['bg_dark'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0))
        
        # Sidebar
        self._create_sidebar(content_frame)
        
        # Plot area
        self._create_plot_area(content_frame)
        
        # Status bar
        self._create_status_bar(main_container)
    
    def _create_header(self, parent):
        """Create modern header."""
        header_frame = tk.Frame(parent, bg=COLORS['bg_medium'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(header_frame, text="Telemetry Anomaly Detection",
                              font=('Segoe UI', 18, 'bold'), fg=COLORS['text'],
                              bg=COLORS['bg_medium'])
        title_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Subtitle
        subtitle_label = tk.Label(header_frame, text="Spacecraft Telemetry Visualizer",
                                 font=('Segoe UI', 11), fg=COLORS['text_secondary'],
                                 bg=COLORS['bg_medium'])
        subtitle_label.pack(side=tk.LEFT, padx=(0, 20), pady=15)
        
        # Channel count badge
        self.channel_count_var = tk.StringVar(value="0 channels")
        count_label = tk.Label(header_frame, textvariable=self.channel_count_var,
                              font=('Segoe UI', 10), fg=COLORS['success'],
                              bg=COLORS['bg_light'], padx=15, pady=5)
        count_label.pack(side=tk.RIGHT, padx=20, pady=15)
    
    def _create_sidebar(self, parent):
        """Create modern sidebar with controls."""
        sidebar = tk.Frame(parent, bg=COLORS['bg_medium'], width=320)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        sidebar.pack_propagate(False)
        
        # Sidebar title
        tk.Label(sidebar, text="Controls", font=('Segoe UI', 14, 'bold'),
                fg=COLORS['text'], bg=COLORS['bg_medium']).pack(pady=(20, 15), padx=15, anchor='w')
        
        # Directory selection section
        self._create_section(sidebar, "Data Directory")
        
        dir_frame = tk.Frame(sidebar, bg=COLORS['bg_medium'])
        dir_frame.pack(fill=tk.X, padx=15, pady=(5, 15))
        
        self.dir_var = tk.StringVar(value="")
        dir_entry = tk.Entry(dir_frame, textvariable=self.dir_var,
                            bg=COLORS['bg_light'], fg=COLORS['text'],
                            insertbackground=COLORS['text'], font=('Segoe UI', 9),
                            relief='flat', bd=0)
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        
        browse_btn = ModernButton(dir_frame, text="Browse", command=self.browse_directory,
                                 width=80, height=32, bg_color=COLORS['bg_light'],
                                 hover_color=COLORS['accent'])
        browse_btn.pack(side=tk.LEFT)
        
        load_btn = ModernButton(sidebar, text="Load Data", command=self.load_data,
                               width=280, height=40)
        load_btn.pack(padx=15, pady=(0, 20))
        
        # Channel selection section
        self._create_section(sidebar, "Channel Selection")
        
        channel_frame = tk.Frame(sidebar, bg=COLORS['bg_medium'])
        channel_frame.pack(fill=tk.X, padx=15, pady=(5, 15))
        
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(channel_frame, textvariable=self.channel_var,
                                         state="readonly", font=('Segoe UI', 10))
        self.channel_combo.pack(fill=tk.X, ipady=5)
        self.channel_combo.bind("<<ComboboxSelected>>", self.on_channel_select)
        
        # View selection section
        self._create_section(sidebar, "View Type")
        
        view_frame = tk.Frame(sidebar, bg=COLORS['bg_medium'])
        view_frame.pack(fill=tk.X, padx=15, pady=(5, 15))
        
        self.view_var = tk.StringVar(value="Overview")
        view_options = ["Overview", "Training Only", "Test Only", "Distribution",
                       "Time Series Features", "Rolling Stats"]
        self.view_combo = ttk.Combobox(view_frame, textvariable=self.view_var,
                                      values=view_options, state="readonly",
                                      font=('Segoe UI', 10))
        self.view_combo.pack(fill=tk.X, ipady=5)
        self.view_combo.bind("<<ComboboxSelected>>", self.on_channel_select)
        
        # Feature selection section
        self._create_section(sidebar, "Feature Index")
        
        feature_frame = tk.Frame(sidebar, bg=COLORS['bg_medium'])
        feature_frame.pack(fill=tk.X, padx=15, pady=(5, 15))
        
        self.feature_var = tk.StringVar(value="0")
        self.feature_spin = ttk.Spinbox(feature_frame, from_=0, to=24,
                                       textvariable=self.feature_var, width=5,
                                       font=('Segoe UI', 10))
        self.feature_spin.pack(side=tk.LEFT, ipady=5)
        self.feature_spin.bind("<Return>", self.on_channel_select)
        
        # Filter section
        self._create_section(sidebar, "Filter by Anomaly Type")
        
        filter_frame = tk.Frame(sidebar, bg=COLORS['bg_medium'])
        filter_frame.pack(fill=tk.X, padx=15, pady=(5, 15))
        
        self.filter_var = tk.StringVar(value="All")
        filter_options = ["All", "Point", "Contextual", "Mixed"]
        self.filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var,
                                        values=filter_options, state="readonly",
                                        font=('Segoe UI', 10))
        self.filter_combo.pack(fill=tk.X, ipady=5)
        self.filter_combo.bind("<<ComboboxSelected>>", self.filter_channels)
        
        # Info section
        self._create_section(sidebar, "Information")
        
        info_frame = tk.Frame(sidebar, bg=COLORS['bg_medium'])
        info_frame.pack(fill=tk.X, padx=15, pady=(5, 20))
        
        self.info_var = tk.StringVar(value="Load a data directory to begin")
        info_label = tk.Label(info_frame, textvariable=self.info_var,
                             font=('Segoe UI', 9), fg=COLORS['text_secondary'],
                             bg=COLORS['bg_medium'], wraplength=270, justify='left')
        info_label.pack(anchor='w')
        
        # Legend section
        self._create_section(sidebar, "Legend")
        
        legend_frame = tk.Frame(sidebar, bg=COLORS['bg_medium'])
        legend_frame.pack(fill=tk.X, padx=15, pady=(5, 20))
        
        self._create_legend_item(legend_frame, COLORS['point_anomaly'], "Point Anomaly")
        self._create_legend_item(legend_frame, COLORS['contextual_anomaly'], "Contextual Anomaly")
        self._create_legend_item(legend_frame, COLORS['mixed_anomaly'], "Mixed Anomaly")
    
    def _create_section(self, parent, title):
        """Create a section header."""
        tk.Label(parent, text=title, font=('Segoe UI', 10, 'bold'),
                fg=COLORS['accent'], bg=COLORS['bg_medium']).pack(pady=(15, 0), padx=15, anchor='w')
    
    def _create_legend_item(self, parent, color, text):
        """Create a legend item."""
        frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        frame.pack(fill=tk.X, pady=2)
        
        color_box = tk.Canvas(frame, width=16, height=16, bg=COLORS['bg_medium'],
                             highlightthickness=0)
        color_box.pack(side=tk.LEFT, padx=(0, 10))
        color_box.create_rectangle(0, 0, 16, 16, fill=color, outline=color)
        
        tk.Label(frame, text=text, font=('Segoe UI', 9), fg=COLORS['text'],
                bg=COLORS['bg_medium']).pack(side=tk.LEFT)
    
    def _create_plot_area(self, parent):
        """Create the plot area."""
        plot_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 7), dpi=100, facecolor=COLORS['bg_medium'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Toolbar
        toolbar_frame = tk.Frame(plot_frame, bg=COLORS['bg_dark'])
        toolbar_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Style toolbar
        for widget in toolbar_frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.configure(bg=COLORS['bg_light'], fg=COLORS['text'],
                                activebackground=COLORS['accent'],
                                activeforeground=COLORS['text'])
    
    def _create_status_bar(self, parent):
        """Create status bar."""
        status_frame = tk.Frame(parent, bg=COLORS['bg_light'], height=30)
        status_frame.pack(fill=tk.X, pady=(15, 0))
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               font=('Segoe UI', 9), fg=COLORS['text_secondary'],
                               bg=COLORS['bg_light'], anchor='w')
        status_label.pack(side=tk.LEFT, padx=15, pady=5)
    
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
            # Check if user selected train/ or test/ directory - use parent instead
            if self.data_dir.name in ['train', 'test']:
                self.data_dir = self.data_dir.parent
                self.dir_var.set(str(self.data_dir))
            
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
                self.all_channels = sorted(self.train_data.keys())
                self.channel_combo['values'] = self.all_channels
                if self.all_channels:
                    self.channel_combo.current(0)
                
                self.channel_count_var.set(f"{len(self.all_channels)} channels loaded")
                self.info_var.set(f"Data loaded from: {self.data_dir.name}")
                self.status_var.set(f"Loaded {len(self.all_channels)} channels successfully")
                self.on_channel_select()
            
            # Check if directory contains .npy files directly
            elif list(self.data_dir.glob("*.npy")):
                self.train_data = {}
                self.test_data = {}
                
                for npy_file in self.data_dir.glob("*.npy"):
                    channel_id = npy_file.stem
                    data = np.load(npy_file)
                    split_idx = int(0.8 * len(data))
                    self.train_data[channel_id] = data[:split_idx]
                    self.test_data[channel_id] = data[split_idx:]
                
                self.all_channels = sorted(self.train_data.keys())
                self.channel_combo['values'] = self.all_channels
                if self.all_channels:
                    self.channel_combo.current(0)
                
                self.channel_count_var.set(f"{len(self.all_channels)} channels loaded")
                self.info_var.set(f"Data loaded from: {self.data_dir.name}")
                self.status_var.set(f"Loaded {len(self.all_channels)} channels successfully")
                self.on_channel_select()
            
            else:
                # Try loading CSV files
                csv_files = list(self.data_dir.glob("*.csv"))
                if csv_files:
                    self.load_csv_data(csv_files[0])
                else:
                    messagebox.showerror("Error",
                        "No valid data found.\n\nExpected:\n"
                        "- train/ and test/ directories with .npy files\n"
                        "- .npy files in selected directory\n"
                        "- .csv files in selected directory")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
            self.status_var.set("Error loading data")
    
    def load_csv_data(self, csv_file):
        """Load CSV telemetry data."""
        df = pd.read_csv(csv_file)
        self.train_data = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.train_data[col] = df[col].values
        
        self.all_channels = sorted(self.train_data.keys())
        self.channel_combo['values'] = self.all_channels
        if self.all_channels:
            self.channel_combo.current(0)
        
        self.channel_count_var.set(f"{len(self.all_channels)} columns loaded")
        self.info_var.set(f"Loaded: {csv_file.name}")
        self.status_var.set(f"Loaded {len(self.all_channels)} columns successfully")
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
        
        if train_data.ndim > 1:
            train_values = train_data[:, feature_idx]
            test_values = test_data[:, feature_idx] if test_data is not None else None
        else:
            train_values = train_data
            test_values = test_data
        
        anomaly_type = self.get_anomaly_type(channel_id)
        anomaly_seqs = self.get_anomaly_sequences(channel_id)
        
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.5], hspace=0.4)
        
        # Training data
        ax1 = self.fig.add_subplot(gs[0])
        ax1.plot(train_values, color=CHART_COLORS[0], alpha=0.9, linewidth=0.8)
        ax1.set_title(f'Training Data - {channel_id}', fontsize=12, fontweight='bold',
                     color=COLORS['text'], pad=10)
        ax1.set_ylabel('Value', fontsize=9)
        ax1.tick_params(labelsize=8)
        ax1.grid(True, alpha=0.2, color=COLORS['text_secondary'])
        ax1.set_facecolor(COLORS['bg_light'])
        
        # Test data with anomalies
        if test_values is not None:
            ax2 = self.fig.add_subplot(gs[1])
            ax2.plot(test_values, color=CHART_COLORS[1], alpha=0.9, linewidth=0.8)
            
            # Add anomaly regions
            for seq in anomaly_seqs:
                try:
                    start, end = seq
                    class_val = str(self.labels_df[self.labels_df['chan_id'] == channel_id].iloc[0].get('class', '')).lower()
                    if 'point' in class_val and 'contextual' not in class_val:
                        color = COLORS['point_anomaly']
                        label = 'Point Anomaly'
                    elif 'contextual' in class_val and 'point' not in class_val:
                        color = COLORS['contextual_anomaly']
                        label = 'Contextual Anomaly'
                    else:
                        color = COLORS['mixed_anomaly']
                        label = 'Mixed Anomaly'
                    ax2.axvspan(start, end, alpha=0.5, color=color, label=label)
                except:
                    pass
            
            ax2.set_title(f'Test Data - {channel_id}{anomaly_type}', fontsize=12,
                         fontweight='bold', color=COLORS['text'], pad=10)
            ax2.set_ylabel('Value', fontsize=9)
            ax2.tick_params(labelsize=8)
            ax2.grid(True, alpha=0.2, color=COLORS['text_secondary'])
            ax2.set_facecolor(COLORS['bg_light'])
            
            handles, labels = ax2.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax2.legend(by_label.values(), by_label.keys(), loc='upper right',
                          fontsize=8, facecolor=COLORS['bg_medium'],
                          edgecolor=COLORS['text_secondary'])
        
        # Distribution
        ax3 = self.fig.add_subplot(gs[2])
        ax3.hist(train_values, bins=50, alpha=0.7, color=CHART_COLORS[0], label='Train', density=True)
        if test_values is not None:
            ax3.hist(test_values, bins=50, alpha=0.7, color=CHART_COLORS[1], label='Test', density=True)
        ax3.set_title('Value Distribution', fontsize=10, color=COLORS['text'], pad=5)
        ax3.set_xlabel('Value', fontsize=9)
        ax3.set_ylabel('Density', fontsize=9)
        ax3.tick_params(labelsize=8)
        ax3.legend(fontsize=8, facecolor=COLORS['bg_medium'], edgecolor=COLORS['text_secondary'])
        ax3.grid(True, alpha=0.2, color=COLORS['text_secondary'])
        ax3.set_facecolor(COLORS['bg_light'])
    
    def plot_training(self, channel_id, feature_idx):
        """Plot training data only."""
        train_data = self.train_data.get(channel_id)
        if train_data is None:
            return
        
        values = train_data[:, feature_idx] if train_data.ndim > 1 else train_data
        
        ax = self.fig.add_subplot(111)
        ax.plot(values, color=CHART_COLORS[0], alpha=0.9, linewidth=0.8)
        ax.set_title(f'Training Data - {channel_id}', fontsize=14, fontweight='bold',
                    color=COLORS['text'])
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.2, color=COLORS['text_secondary'])
        ax.set_facecolor(COLORS['bg_light'])
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
        ax.plot(values, color=CHART_COLORS[1], alpha=0.9, linewidth=0.8)
        
        for seq in anomaly_seqs:
            try:
                start, end = seq
                class_val = str(self.labels_df[self.labels_df['chan_id'] == channel_id].iloc[0].get('class', '')).lower()
                if 'point' in class_val and 'contextual' not in class_val:
                    color = COLORS['point_anomaly']
                elif 'contextual' in class_val and 'point' not in class_val:
                    color = COLORS['contextual_anomaly']
                else:
                    color = COLORS['mixed_anomaly']
                ax.axvspan(start, end, alpha=0.5, color=color)
            except:
                pass
        
        ax.set_title(f'Test Data - {channel_id}{anomaly_type}', fontsize=14,
                    fontweight='bold', color=COLORS['text'])
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.2, color=COLORS['text_secondary'])
        ax.set_facecolor(COLORS['bg_light'])
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
        ax.hist(train_values, bins=50, alpha=0.7, color=CHART_COLORS[0], label='Train', density=True)
        if test_values is not None:
            ax.hist(test_values, bins=50, alpha=0.7, color=CHART_COLORS[1], label='Test', density=True)
        
        ax.set_title(f'Value Distribution - {channel_id}', fontsize=14,
                    fontweight='bold', color=COLORS['text'])
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9, facecolor=COLORS['bg_medium'], edgecolor=COLORS['text_secondary'])
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.2, color=COLORS['text_secondary'])
        ax.set_facecolor(COLORS['bg_light'])
        
        stats_text = f"Train: mean={np.mean(train_values):.3f}, std={np.std(train_values):.3f}"
        if test_values is not None:
            stats_text += f"\nTest: mean={np.mean(test_values):.3f}, std={np.std(test_values):.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=COLORS['bg_medium'],
                edgecolor=COLORS['text_secondary'], alpha=0.9))
        
        self.fig.tight_layout()
    
    def plot_time_series_features(self, channel_id, feature_idx):
        """Plot time series analysis features."""
        train_data = self.train_data.get(channel_id)
        test_data = self.test_data.get(channel_id)
        
        if train_data is None:
            return
        
        train_values = train_data[:, feature_idx] if train_data.ndim > 1 else train_data
        test_values = test_data[:, feature_idx] if test_data is not None and test_data.ndim > 1 else test_data
        
        combined = np.concatenate([train_values, test_values]) if test_values is not None else train_values
        
        gs = self.fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.5)
        
        # Original signal
        ax1 = self.fig.add_subplot(gs[0])
        ax1.plot(combined, color=CHART_COLORS[0], alpha=0.9, linewidth=0.8)
        if test_values is not None:
            ax1.axvline(x=len(train_values), color=COLORS['accent'], linestyle='--',
                       alpha=0.7, label='Train/Test Split')
        ax1.set_title(f'Time Series Analysis - {channel_id}', fontsize=12,
                     fontweight='bold', color=COLORS['text'])
        ax1.set_ylabel('Value', fontsize=9)
        ax1.legend(fontsize=8, facecolor=COLORS['bg_medium'], edgecolor=COLORS['text_secondary'])
        ax1.tick_params(labelsize=8)
        ax1.grid(True, alpha=0.2)
        ax1.set_facecolor(COLORS['bg_light'])
        
        # Rolling statistics
        window = 50
        rolling_mean = pd.Series(combined).rolling(window=window, center=True).mean()
        rolling_std = pd.Series(combined).rolling(window=window, center=True).std()
        
        ax2 = self.fig.add_subplot(gs[1])
        ax2.plot(rolling_mean, color=CHART_COLORS[1], linewidth=1.5, label='Rolling Mean')
        ax2.fill_between(range(len(combined)), rolling_mean - rolling_std,
                        rolling_mean + rolling_std, alpha=0.3, color=CHART_COLORS[1])
        ax2.set_ylabel('Rolling Stats', fontsize=9)
        ax2.legend(fontsize=8, facecolor=COLORS['bg_medium'], edgecolor=COLORS['text_secondary'])
        ax2.tick_params(labelsize=8)
        ax2.grid(True, alpha=0.2)
        ax2.set_facecolor(COLORS['bg_light'])
        
        # FFT
        ax3 = self.fig.add_subplot(gs[2])
        fft = np.fft.fft(combined)
        freqs = np.fft.fftfreq(len(combined))
        ax3.plot(freqs[:len(freqs)//2], np.abs(fft)[:len(fft)//2], color=CHART_COLORS[2])
        ax3.set_xlabel('Frequency', fontsize=9)
        ax3.set_ylabel('Magnitude', fontsize=9)
        ax3.set_title('Frequency Spectrum', fontsize=10, color=COLORS['text'])
        ax3.tick_params(labelsize=8)
        ax3.grid(True, alpha=0.2)
        ax3.set_facecolor(COLORS['bg_light'])
        
        # Autocorrelation
        ax4 = self.fig.add_subplot(gs[3])
        autocorr = np.correlate(combined - np.mean(combined), combined - np.mean(combined), mode='full')
        autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
        ax4.plot(autocorr[:200], color=CHART_COLORS[3])
        ax4.set_xlabel('Lag', fontsize=9)
        ax4.set_ylabel('Autocorrelation', fontsize=9)
        ax4.set_title('Autocorrelation Function', fontsize=10, color=COLORS['text'])
        ax4.tick_params(labelsize=8)
        ax4.grid(True, alpha=0.2)
        ax4.set_facecolor(COLORS['bg_light'])
    
    def plot_rolling_stats(self, channel_id, feature_idx):
        """Plot rolling statistics."""
        train_data = self.train_data.get(channel_id)
        test_data = self.test_data.get(channel_id)
        
        if train_data is None:
            return
        
        train_values = train_data[:, feature_idx] if train_data.ndim > 1 else train_data
        test_values = test_data[:, feature_idx] if test_data is not None and test_data.ndim > 1 else test_data
        
        combined = np.concatenate([train_values, test_values]) if test_values is not None else train_values
        
        windows = [20, 50, 100]
        gs = self.fig.add_gridspec(len(windows) + 1, 1, hspace=0.5)
        
        # Original signal
        ax0 = self.fig.add_subplot(gs[0])
        ax0.plot(combined, color=CHART_COLORS[0], alpha=0.7, linewidth=0.5)
        if test_values is not None:
            ax0.axvline(x=len(train_values), color=COLORS['accent'], linestyle='--', alpha=0.7)
        ax0.set_title(f'Rolling Statistics - {channel_id}', fontsize=12,
                     fontweight='bold', color=COLORS['text'])
        ax0.set_ylabel('Value', fontsize=9)
        ax0.tick_params(labelsize=8)
        ax0.grid(True, alpha=0.2)
        ax0.set_facecolor(COLORS['bg_light'])
        
        # Rolling stats for different windows
        for i, window in enumerate(windows):
            ax = self.fig.add_subplot(gs[i + 1])
            rolling_mean = pd.Series(combined).rolling(window=window, center=True).mean()
            rolling_std = pd.Series(combined).rolling(window=window, center=True).std()
            
            ax.plot(rolling_mean, color=CHART_COLORS[i + 1], linewidth=1.5, label=f'Window={window}')
            ax.fill_between(range(len(combined)), rolling_mean - rolling_std,
                           rolling_mean + rolling_std, alpha=0.3, color=CHART_COLORS[i + 1])
            ax.set_ylabel(f'W={window}', fontsize=9)
            ax.legend(fontsize=8, facecolor=COLORS['bg_medium'], edgecolor=COLORS['text_secondary'])
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.2)
            ax.set_facecolor(COLORS['bg_light'])


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    
    # Set window icon (if available)
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    app = TelemetryViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()