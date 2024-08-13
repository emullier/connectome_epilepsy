import tkinter as tk
from tkinter import ttk
import json

class ConfigGUI:
    def __init__(self, root, run_processing, config_defaults):
        self.root = root
        self.run_processing = run_processing
        self.config_defaults = config_defaults
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        
        self.parameters_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.parameters_tab, text='Parameters')
        
        # Create GUI elements for parameters
        self.create_parameters_section()

    def create_parameters_section(self):
        """Create GUI elements for parameters."""
        # Parameters Section
        ttk.Label(self.parameters_tab, text="Output Directory:").grid(row=0, column=0, sticky='w')
        self.output_dir = tk.StringVar(value=self.config_defaults.get('Parameters', {}).get('output_dir', ''))
        ttk.Entry(self.parameters_tab, textvariable=self.output_dir).grid(row=0, column=1, sticky='ew')

        ttk.Label(self.parameters_tab, text="Info Path:").grid(row=1, column=0, sticky='w')
        self.info_path = tk.StringVar(value=self.config_defaults.get('Parameters', {}).get('info_path', ''))
        ttk.Entry(self.parameters_tab, textvariable=self.info_path).grid(row=1, column=1, sticky='ew')

        ttk.Label(self.parameters_tab, text="Data Directory:").grid(row=2, column=0, sticky='w')
        self.data_dir = tk.StringVar(value=self.config_defaults.get('Parameters', {}).get('data_dir', ''))
        ttk.Entry(self.parameters_tab, textvariable=self.data_dir).grid(row=2, column=1, sticky='ew')

        ttk.Label(self.parameters_tab, text="Scale:").grid(row=3, column=0, sticky='w')
        self.scale = tk.IntVar(value=self.config_defaults.get('Parameters', {}).get('scale', 2))
        ttk.Entry(self.parameters_tab, textvariable=self.scale).grid(row=3, column=1, sticky='ew')

        ttk.Label(self.parameters_tab, text="Metric:").grid(row=4, column=0, sticky='w')
        self.metric = tk.StringVar(value=self.config_defaults.get('Parameters', {}).get('metric', 'shore_mean_gfa'))
        ttk.Entry(self.parameters_tab, textvariable=self.metric).grid(row=4, column=1, sticky='ew')

        # Filters Section
        ttk.Label(self.parameters_tab, text="Filters:").grid(row=5, column=0, sticky='w', pady="5")
        filters_frame = ttk.Frame(self.parameters_tab)
        filters_frame.grid(row=6, column=0, columnspan=2, sticky='ew')

        self.filter_vars = {}
        self.filter_options = {}
        filters = self.config_defaults.get('Parameters', {}).get('filters', [])
        self.selected_filters = {}

        for i, filter_name in enumerate(filters):
            ttk.Label(filters_frame, text=filter_name).grid(row=i, column=0, sticky='w')
            filter_values = self.config_defaults.get('Parameters', {}).get(filter_name, [])
            # Ensure values are strings
            filter_values_str = [str(val) for val in filter_values]
            self.selected_filters[filter_name] = tk.StringVar(value=', '.join(filter_values_str))
            filter_combobox = ttk.Combobox(filters_frame, textvariable=self.selected_filters[filter_name], values=filter_values_str, state='normal')
            filter_combobox.grid(row=i, column=1, sticky='ew')

    def create_processing_section(self):
        """Create GUI elements for processing."""
        # Processing Section
        ttk.Label(self.processing_tab, text="Processing:").grid(row=0, column=0, sticky='w', pady="5")
        processing_frame = ttk.Frame(self.processing_tab)
        processing_frame.grid(row=1, column=0, columnspan=2, sticky='ew')

        self.processing_vars = {}
        self.processing_paths = {}
        processing_config = self.config_defaults.get('Parameters', {}).get('processing', {})

        for i, (proc_name, proc_path) in enumerate(processing_config.items()):
            self.processing_vars[proc_name] = tk.IntVar(value=1)
            self.processing_paths[proc_name] = tk.StringVar(value=proc_path)
            ttk.Checkbutton(processing_frame, text=proc_name, variable=self.processing_vars[proc_name]).grid(row=i, column=0, sticky='w')
            ttk.Entry(processing_frame, textvariable=self.processing_paths[proc_name]).grid(row=i, column=1, sticky='ew')

    def create_bias_section(self):
        """Create GUI elements for BIAS parameters."""
        # BIAS Section
        ttk.Label(self.bias_tab, text="BIAS Analysis:").grid(row=0, column=0, sticky='w', pady="5")
        bias_frame = ttk.Frame(self.bias_tab)
        bias_frame.grid(row=1, column=0, columnspan=2, sticky='ew')

        self.bias_analysis = tk.BooleanVar(value=self.config_defaults.get('BIAS', {}).get('perform_analysis', False))
        ttk.Checkbutton(bias_frame, text="Perform Bias Analysis", variable=self.bias_analysis).grid(row=0, column=0, sticky='w')

        ttk.Label(bias_frame, text="Dimensionality Reduction Method:").grid(row=1, column=0, sticky='w')
        self.dim_reduc = tk.StringVar(value=self.config_defaults.get('BIAS', {}).get('DimReduc', 'PCA'))
        reduction_options = ['PCA', 'Other']  # Add other options as necessary
        ttk.OptionMenu(bias_frame, self.dim_reduc, *reduction_options).grid(row=1, column=1, sticky='ew')

        ttk.Label(bias_frame, text="Colors Key:").grid(row=2, column=0, sticky='w')
        self.colors_key = tk.StringVar(value=self.config_defaults.get('BIAS', {}).get('colors_key', ''))
        ttk.Entry(bias_frame, textvariable=self.colors_key).grid(row=2, column=1, sticky='ew')

    def create_consensus_section(self):
        """Create GUI elements for CONSENSUS parameters."""
        # CONSENSUS Section
        ttk.Label(self.consensus_tab, text="CONSENSUS Analysis:").grid(row=0, column=0, sticky='w', pady="5")
        consensus_frame = ttk.Frame(self.consensus_tab)
        consensus_frame.grid(row=1, column=0, columnspan=2, sticky='ew')

        self.consensus_analysis = tk.BooleanVar(value=self.config_defaults.get('CONSENSUS', {}).get('perform_analysis', False))
        ttk.Checkbutton(consensus_frame, text="Perform Consensus Analysis", variable=self.consensus_analysis).grid(row=0, column=0, sticky='w')

        ttk.Label(consensus_frame, text="Number of Bins:").grid(row=1, column=0, sticky='w')
        self.nbins = tk.IntVar(value=self.config_defaults.get('CONSENSUS', {}).get('nbins', 10))
        ttk.Entry(consensus_frame, textvariable=self.nbins).grid(row=1, column=1, sticky='ew')

    def create_harmonics_section(self):
        """Create GUI elements for HARMONICS parameters."""
        # HARMONICS Section
        ttk.Label(self.harmonics_tab, text="HARMONICS Analysis:").grid(row=0, column=0, sticky='w', pady="5")
        harmonics_frame = ttk.Frame(self.harmonics_tab)
        harmonics_frame.grid(row=1, column=0, columnspan=2, sticky='ew')

        self.generate_harmonics = tk.BooleanVar(value=self.config_defaults.get('HARMONICS', {}).get('generate_harmonics', False))
        ttk.Checkbutton(harmonics_frame, text="Generate Harmonics", variable=self.generate_harmonics).grid(row=0, column=0, sticky='w')

        self.compare_harmonics = tk.BooleanVar(value=self.config_defaults.get('HARMONICS', {}).get('compare_harmonics', False))
        ttk.Checkbutton(harmonics_frame, text="Compare Harmonics", variable=self.compare_harmonics).grid(row=1, column=0, sticky='w')

        self.consensus_harmonics = tk.BooleanVar(value=self.config_defaults.get('HARMONICS', {}).get('consensus', False))
        ttk.Checkbutton(harmonics_frame, text="Harmonics Consensus", variable=self.consensus_harmonics).grid(row=2, column=0, sticky='w')

    def create_indvscons_section(self):
        """Create GUI elements for INDvsCONS parameters."""
        # INDvsCONS Section
        ttk.Label(self.indvscons_tab, text="INDvsCONS Analysis:").grid(row=0, column=0, sticky='w', pady="5")
        indvscons_frame = ttk.Frame(self.indvscons_tab)
        indvscons_frame.grid(row=1, column=0, columnspan=2, sticky='ew')

        self.indvscons_analysis = tk.BooleanVar(value=self.config_defaults.get('INDvsCONS', {}).get('perform_analysis', False))
        ttk.Checkbutton(indvscons_frame, text="Perform INDvsCONS Analysis", variable=self.indvscons_analysis).grid(row=0, column=0, sticky='w')

    def run_script(self):
        # Collect parameters from the GUI
        filters_selected = {name: value.get() for name, value in self.selected_filters.items() if value.get()}
        processing_selected = {name: path.get() for name, path in self.processing_paths.items() if self.processing_vars[name].get() == 1}

        config = {
            "Parameters": {
                "output_dir": self.output_dir.get(),
                "info_path": self.info_path.get(),
                "data_dir": self.data_dir.get(),
                "scale": self.scale.get(),
                "metric": self.metric.get(),
                "filters": filters_selected,
                "processing": processing_selected,
                "load_dataset": self.load_dataset.get(),
            },
            "BIAS": {
                "perform_analysis": self.bias_analysis.get(),
                "DimReduc": self.dim_reduc.get(),
                "colors_key": self.colors_key.get(),
            },
            "CONSENSUS": {
                "perform_analysis": self.consensus_analysis.get(),
                "nbins": self.nbins.get(),
            },
            "HARMONICS": {
                "generate_harmonics": self.generate_harmonics.get(),
                "compare_harmonics": self.compare_harmonics.get(),
                "consensus": self.consensus_harmonics.get(),
            },
            "INDvsCONS": {
                "perform_analysis": self.indvscons_analysis.get(),
            },
        }

        # Call the main processing function with the config
        self.run_processing_callback(config)
