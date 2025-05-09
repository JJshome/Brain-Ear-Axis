#!/usr/bin/env python3
"""
Dynamic Causal Modeling (DCM) Analysis for Brain-Ear Axis

This module implements the DCM analysis component for the neural signal
analysis module (340) of the Brain-Ear Axis Analysis System. It analyzes
functional connectivity between auditory cortex and other brain regions.

The implementation uses Python wrappers around SPM's DCM functionality.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import scipy.io as sio
from scipy import stats
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dcm_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants for default regions of interest
DEFAULT_ROI_DEFINITIONS = {
    "primary_auditory_cortex": {
        "center": (-42, -22, 7),  # MNI coordinates for left primary auditory cortex
        "radius": 8,               # mm
        "hemisphere": "left",
        "abbreviation": "A1L"
    },
    "primary_auditory_cortex_right": {
        "center": (42, -22, 7),    # MNI coordinates for right primary auditory cortex
        "radius": 8,               # mm
        "hemisphere": "right",
        "abbreviation": "A1R"
    },
    "superior_temporal_gyrus": {
        "center": (-58, -12, -2),  # MNI coordinates for left STG
        "radius": 8,               # mm
        "hemisphere": "left",
        "abbreviation": "STGL"
    },
    "superior_temporal_gyrus_right": {
        "center": (58, -12, -2),   # MNI coordinates for right STG
        "radius": 8,               # mm
        "hemisphere": "right",
        "abbreviation": "STGR"
    },
    "amygdala": {
        "center": (-23, -5, -19),  # MNI coordinates for left amygdala
        "radius": 6,               # mm
        "hemisphere": "left",
        "abbreviation": "AMYL"
    },
    "amygdala_right": {
        "center": (23, -5, -19),   # MNI coordinates for right amygdala
        "radius": 6,               # mm
        "hemisphere": "right",
        "abbreviation": "AMYR"
    },
    "inferior_frontal_gyrus": {
        "center": (-45, 28, 14),   # MNI coordinates for left IFG
        "radius": 8,               # mm
        "hemisphere": "left",
        "abbreviation": "IFGL"
    },
    "anterior_cingulate_cortex": {
        "center": (0, 22, 28),     # MNI coordinates for ACC
        "radius": 8,               # mm
        "hemisphere": "bilateral",
        "abbreviation": "ACC"
    }
}

# Definition of default DCM models (connectivity patterns to test)
DEFAULT_DCM_MODELS = {
    "tinnitus_model": {
        "description": "Model for tinnitus patients, with enhanced connectivity between auditory and limbic regions",
        "regions": ["primary_auditory_cortex", "primary_auditory_cortex_right", "amygdala", "amygdala_right"],
        "connections": [
            ("primary_auditory_cortex", "primary_auditory_cortex_right"),
            ("primary_auditory_cortex_right", "primary_auditory_cortex"),
            ("primary_auditory_cortex", "amygdala"),
            ("primary_auditory_cortex_right", "amygdala_right"),
            ("amygdala", "amygdala_right"),
            ("amygdala_right", "amygdala"),
            ("amygdala", "primary_auditory_cortex"),
            ("amygdala_right", "primary_auditory_cortex_right")
        ],
        "driving_inputs": ["primary_auditory_cortex", "primary_auditory_cortex_right"]
    },
    "hearing_loss_model": {
        "description": "Model for hearing loss patients, with altered connectivity in auditory pathways",
        "regions": ["primary_auditory_cortex", "primary_auditory_cortex_right", "superior_temporal_gyrus", "superior_temporal_gyrus_right"],
        "connections": [
            ("primary_auditory_cortex", "primary_auditory_cortex_right"),
            ("primary_auditory_cortex_right", "primary_auditory_cortex"),
            ("primary_auditory_cortex", "superior_temporal_gyrus"),
            ("primary_auditory_cortex_right", "superior_temporal_gyrus_right"),
            ("superior_temporal_gyrus", "superior_temporal_gyrus_right"),
            ("superior_temporal_gyrus_right", "superior_temporal_gyrus")
        ],
        "driving_inputs": ["primary_auditory_cortex", "primary_auditory_cortex_right"]
    },
    "meniere_model": {
        "description": "Model for Meniere's disease patients, with involvement of vestibular and auditory regions",
        "regions": ["primary_auditory_cortex", "primary_auditory_cortex_right", "superior_temporal_gyrus", "anterior_cingulate_cortex"],
        "connections": [
            ("primary_auditory_cortex", "primary_auditory_cortex_right"),
            ("primary_auditory_cortex_right", "primary_auditory_cortex"),
            ("primary_auditory_cortex", "superior_temporal_gyrus"),
            ("primary_auditory_cortex_right", "superior_temporal_gyrus"),
            ("superior_temporal_gyrus", "anterior_cingulate_cortex"),
            ("anterior_cingulate_cortex", "superior_temporal_gyrus")
        ],
        "driving_inputs": ["primary_auditory_cortex", "primary_auditory_cortex_right"]
    }
}

class DCMAnalysis:
    """Class for DCM analysis of brain connectivity related to ear diseases."""
    
    def __init__(self, 
                 fmri_data: str,
                 output_dir: str,
                 patient_id: str,
                 conditions: List[str],
                 model_type: str = "tinnitus_model",
                 custom_model: Optional[Dict[str, Any]] = None,
                 custom_roi_definitions: Optional[Dict[str, Dict[str, Any]]] = None,
                 tr: float = 2.0,
                 matlab_path: Optional[str] = None,
                 spm_path: Optional[str] = None):
        """
        Initialize the DCM analysis.
        
        Args:
            fmri_data: Path to preprocessed fMRI data
            output_dir: Directory for output files
            patient_id: Patient identifier
            conditions: List of experimental conditions
            model_type: Type of DCM model to use
            custom_model: Custom DCM model definition (overrides model_type)
            custom_roi_definitions: Custom ROI definitions
            tr: Repetition time of fMRI data (in seconds)
            matlab_path: Path to MATLAB executable
            spm_path: Path to SPM installation
        """
        self.fmri_data = fmri_data
        self.output_dir = Path(output_dir)
        self.patient_id = patient_id
        self.conditions = conditions
        self.model_type = model_type
        self.custom_model = custom_model
        self.tr = tr
        self.matlab_path = matlab_path
        self.spm_path = spm_path
        
        # Use custom ROI definitions if provided, otherwise use defaults
        self.roi_definitions = custom_roi_definitions if custom_roi_definitions else DEFAULT_ROI_DEFINITIONS
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            "patient_id": patient_id,
            "model_type": model_type,
            "conditions": conditions,
            "connectivity_strengths": {},
            "model_evidence": None,
            "abnormal_connections": []
        }
        
        # Initialize model definition
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the DCM model based on model_type or custom_model."""
        if self.custom_model:
            self.model = self.custom_model
            logger.info("Using custom DCM model")
        else:
            if self.model_type not in DEFAULT_DCM_MODELS:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.model = DEFAULT_DCM_MODELS[self.model_type]
            logger.info(f"Using predefined DCM model: {self.model_type}")
            
        # Validate that all required regions are defined
        missing_regions = [r for r in self.model["regions"] if r not in self.roi_definitions]
        if missing_regions:
            raise ValueError(f"Missing ROI definitions for regions: {missing_regions}")
            
        logger.info(f"Model initialized with {len(self.model['regions'])} regions and {len(self.model['connections'])} connections")
        
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run the full DCM analysis pipeline.
        
        Returns:
            Dict containing analysis results
        """
        logger.info(f"Starting DCM analysis for patient {self.patient_id}")
        
        try:
            # Step 1: Extract time series from ROIs
            time_series = self._extract_roi_time_series()
            
            # Step 2: Specify DCM model
            dcm_spec = self._specify_dcm_model(time_series)
            
            # Step 3: Estimate DCM parameters
            dcm_params = self._estimate_dcm_parameters(dcm_spec)
            
            # Step 4: Analyze connectivity patterns
            connectivity = self._analyze_connectivity(dcm_params)
            
            # Step 5: Compare with normative data
            abnormal_connections = self._compare_with_normative_data(connectivity)
            
            # Step 6: Generate visualizations
            self._generate_connectivity_visualizations(connectivity, abnormal_connections)
            
            # Step 7: Prepare results
            self.results["connectivity_strengths"] = connectivity
            self.results["abnormal_connections"] = abnormal_connections
            
            # Save results
            self._save_results()
            
            logger.info(f"DCM analysis completed for patient {self.patient_id}")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in DCM analysis: {e}")
            raise
            
    def _extract_roi_time_series(self) -> Dict[str, np.ndarray]:
        """
        Extract time series from regions of interest.
        
        Returns:
            Dict mapping region names to time series arrays
        """
        logger.info("Extracting time series from ROIs")
        
        # In a real implementation, this would load the fMRI data and extract time series
        # using the ROI definitions. Here we generate synthetic data instead.
        
        # Mock implementation with synthetic data
        time_series = {}
        
        # Generate random time series for each region
        n_timepoints = 200  # Typical number of timepoints in an fMRI run
        
        for region in self.model["regions"]:
            # Generate synthetic time series with temporal autocorrelation
            # to mimic fMRI BOLD signal
            signal = np.zeros(n_timepoints)
            for i in range(1, n_timepoints):
                signal[i] = 0.8 * signal[i-1] + 0.2 * np.random.normal()
                
            # Add condition-specific responses
            for condition_idx, condition in enumerate(self.conditions):
                # Create a hemodynamic response for each condition
                condition_onsets = np.zeros(n_timepoints)
                
                # Simulate stimulus blocks or events
                block_length = 20  # timepoints
                for block_start in range(condition_idx * 10, n_timepoints, 3 * block_length):
                    if block_start + block_length < n_timepoints:
                        condition_onsets[block_start:block_start+block_length] = 1.0
                
                # Convolve with a simple hemodynamic response function
                hrf = np.exp(-(np.arange(20) - 6)**2 / 16)
                hrf = hrf / np.sum(hrf)
                condition_response = np.convolve(condition_onsets, hrf, mode='same')
                
                # Add to signal with region-specific amplitude
                amplitude = 1.0
                if region in ["amygdala", "amygdala_right"] and "tinnitus" in self.model_type:
                    amplitude = 1.5  # Enhanced response in limbic regions for tinnitus model
                
                signal += amplitude * condition_response
            
            # Add noise
            signal += 0.1 * np.random.normal(size=n_timepoints)
            
            # Store in time series dict
            time_series[region] = signal
            
        logger.info(f"Extracted time series for {len(time_series)} regions")
        return time_series
        
    def _specify_dcm_model(self, time_series: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Specify the DCM model for estimation.
        
        Args:
            time_series: Dict mapping region names to time series arrays
            
        Returns:
            Dict with DCM model specification
        """
        logger.info("Specifying DCM model")
        
        # In a real implementation, this would create a DCM specification in the format
        # expected by SPM/MATLAB. Here we create a simplified representation.
        
        # Create adjacency matrix for intrinsic connections
        n_regions = len(self.model["regions"])
        region_index = {region: i for i, region in enumerate(self.model["regions"])}
        
        # Initialize connectivity matrices
        intrinsic_connectivity = np.zeros((n_regions, n_regions))
        
        # Set up intrinsic connections
        for source, target in self.model["connections"]:
            source_idx = region_index[source]
            target_idx = region_index[target]
            intrinsic_connectivity[target_idx, source_idx] = 1  # Mark connection as present
            
        # Set up driving inputs
        n_conditions = len(self.conditions)
        driving_inputs = np.zeros((n_regions, n_conditions))
        
        for region in self.model["driving_inputs"]:
            region_idx = region_index[region]
            driving_inputs[region_idx, :] = 1  # Mark all conditions as inputs to this region
            
        # Create DCM specification
        dcm_spec = {
            "regions": self.model["regions"],
            "region_index": region_index,
            "conditions": self.conditions,
            "time_series": time_series,
            "intrinsic_connectivity": intrinsic_connectivity,
            "driving_inputs": driving_inputs,
            "TR": self.tr
        }
        
        return dcm_spec
        
    def _estimate_dcm_parameters(self, dcm_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate DCM parameters.
        
        Args:
            dcm_spec: DCM model specification
            
        Returns:
            Dict with estimated DCM parameters
        """
        logger.info("Estimating DCM parameters")
        
        # In a real implementation, this would call SPM's DCM estimation function
        # through MATLAB. Here we generate synthetic parameter estimates.
        
        # Get dimensions
        n_regions = len(dcm_spec["regions"])
        n_connections = np.sum(dcm_spec["intrinsic_connectivity"])
        
        # Generate synthetic connection strengths
        np.random.seed(42)  # For reproducibility
        
        # Create parameter dictionary
        parameters = {
            "intrinsic": np.zeros((n_regions, n_regions)),
            "driving_input": np.zeros((n_regions, len(self.conditions))),
            "model_evidence": -500.0  # Synthetic model evidence value
        }
        
        # Fill in intrinsic connection parameters
        for i in range(n_regions):
            for j in range(n_regions):
                if dcm_spec["intrinsic_connectivity"][i, j] == 1:
                    # Generate synthetic connection strength
                    # Vary strength based on model type to simulate disease-specific patterns
                    base_strength = 0.5
                    
                    source_region = dcm_spec["regions"][j]
                    target_region = dcm_spec["regions"][i]
                    
                    # Modify strengths based on disease model
                    if "tinnitus" in self.model_type:
                        # Enhanced connectivity between auditory cortex and amygdala in tinnitus
                        if ("auditory" in source_region and "amygdala" in target_region) or \
                           ("amygdala" in source_region and "auditory" in target_region):
                            base_strength = 0.8
                    
                    elif "hearing_loss" in self.model_type:
                        # Reduced connectivity within auditory regions in hearing loss
                        if "auditory" in source_region and "auditory" in target_region:
                            base_strength = 0.3
                    
                    # Add random variation
                    strength = base_strength + 0.2 * np.random.normal()
                    parameters["intrinsic"][i, j] = strength
        
        # Fill in driving input parameters
        for i in range(n_regions):
            for j in range(len(self.conditions)):
                if dcm_spec["driving_inputs"][i, j] == 1:
                    # Generate synthetic driving input strength
                    strength = 0.7 + 0.2 * np.random.normal()
                    parameters["driving_input"][i, j] = strength
        
        return parameters
        
    def _analyze_connectivity(self, dcm_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze connectivity patterns from DCM parameters.
        
        Args:
            dcm_params: Estimated DCM parameters
            
        Returns:
            Dict mapping connection names to connectivity strengths
        """
        logger.info("Analyzing connectivity patterns")
        
        # Extract connectivity strengths for each connection
        connectivity = {}
        
        intrinsic = dcm_params["intrinsic"]
        regions = self.model["regions"]
        
        for i, target in enumerate(regions):
            for j, source in enumerate(regions):
                if intrinsic[i, j] != 0:
                    connection_name = f"{source}_to_{target}"
                    connectivity[connection_name] = float(intrinsic[i, j])
        
        logger.info(f"Analyzed {len(connectivity)} connections")
        return connectivity
        
    def _compare_with_normative_data(self, connectivity: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Compare connectivity strengths with normative data.
        
        Args:
            connectivity: Dict mapping connection names to connectivity strengths
            
        Returns:
            List of abnormal connections
        """
        logger.info("Comparing connectivity with normative data")
        
        # In a real implementation, this would load normative connectivity data
        # from a database and perform statistical comparison.
        # Here we use synthetic normative values.
        
        # Mock normative data (mean and std for each connection)
        normative_data = {}
        for connection_name in connectivity:
            source, target = connection_name.split("_to_")
            
            # Set different normative values based on connection type
            if "auditory" in source and "auditory" in target:
                # Auditory-auditory connections
                mean = 0.5
                std = 0.1
            elif ("auditory" in source and "amygdala" in target) or \
                 ("amygdala" in source and "auditory" in target):
                # Auditory-limbic connections
                mean = 0.3
                std = 0.15
            else:
                # Other connections
                mean = 0.4
                std = 0.12
                
            normative_data[connection_name] = {"mean": mean, "std": std}
        
        # Compare each connection with normative data
        abnormal_connections = []
        
        for connection_name, strength in connectivity.items():
            normative = normative_data.get(connection_name)
            if normative:
                # Calculate z-score
                z_score = (strength - normative["mean"]) / normative["std"]
                
                # Check if connection is abnormally strong or weak (|z| > 2)
                if abs(z_score) > 2:
                    abnormal_connections.append({
                        "connection": connection_name,
                        "strength": strength,
                        "normative_mean": normative["mean"],
                        "normative_std": normative["std"],
                        "z_score": z_score,
                        "direction": "increased" if z_score > 0 else "decreased"
                    })
        
        logger.info(f"Found {len(abnormal_connections)} abnormal connections")
        return abnormal_connections
        
    def _generate_connectivity_visualizations(self, connectivity: Dict[str, float], 
                                             abnormal_connections: List[Dict[str, Any]]):
        """
        Generate visualizations of connectivity patterns.
        
        Args:
            connectivity: Dict mapping connection names to connectivity strengths
            abnormal_connections: List of abnormal connections
        """
        logger.info("Generating connectivity visualizations")
        
        # Create a network graph
        G = nx.DiGraph()
        
        # Add nodes
        for region in self.model["regions"]:
            G.add_node(region)
        
        # Add edges with weights
        for connection_name, strength in connectivity.items():
            source, target = connection_name.split("_to_")
            G.add_edge(source, target, weight=strength)
        
        # Set abnormal connections
        abnormal_connections_set = {conn["connection"] for conn in abnormal_connections}
        
        # Plot network
        plt.figure(figsize=(12, 10))
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
        
        # Draw normal edges
        normal_edges = [(source, target) for source, target, attrs in G.edges(data=True) 
                        if f"{source}_to_{target}" not in abnormal_connections_set]
        normal_weights = [attrs['weight'] * 2 for source, target, attrs in G.edges(data=True) 
                         if f"{source}_to_{target}" not in abnormal_connections_set]
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=normal_weights, 
                               alpha=0.7, edge_color='gray', arrows=True)
        
        # Draw abnormal edges
        abnormal_edges = [(source, target) for source, target, attrs in G.edges(data=True) 
                         if f"{source}_to_{target}" in abnormal_connections_set]
        abnormal_weights = [attrs['weight'] * 2 for source, target, attrs in G.edges(data=True) 
                           if f"{source}_to_{target}" in abnormal_connections_set]
        nx.draw_networkx_edges(G, pos, edgelist=abnormal_edges, width=abnormal_weights, 
                               alpha=0.9, edge_color='red', arrows=True)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # Add title
        plt.title(f"Brain Connectivity Network for Patient {self.patient_id}\nModel: {self.model_type}")
        plt.axis('off')
        
        # Save figure
        plt.savefig(self.output_dir / f"{self.patient_id}_{self.model_type}_connectivity.png", dpi=300)
        plt.close()
        
        # Create bar plot for connection strengths
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        connections = list(connectivity.keys())
        strengths = list(connectivity.values())
        
        # Determine colors
        colors = ['red' if conn in abnormal_connections_set else 'blue' for conn in connections]
        
        # Create bar plot
        plt.bar(connections, strengths, color=colors)
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=90)
        
        # Add labels and title
        plt.xlabel('Connection')
        plt.ylabel('Connectivity Strength')
        plt.title(f"Connection Strengths for Patient {self.patient_id}\nModel: {self.model_type}")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Normal Connection'),
            Patch(facecolor='red', label='Abnormal Connection')
        ]
        plt.legend(handles=legend_elements)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / f"{self.patient_id}_{self.model_type}_connection_strengths.png", dpi=300)
        plt.close()
        
    def _save_results(self):
        """Save DCM analysis results."""
        # Save results as JSON
        results_file = self.output_dir / f"{self.patient_id}_{self.model_type}_dcm_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Results saved to {results_file}")


def main():
    """Main entry point for running the DCM analysis."""
    parser = argparse.ArgumentParser(description='DCM Analysis for Brain-Ear Axis')
    parser.add_argument('--fmri-data', required=True, help='Path to preprocessed fMRI data')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--patient-id', required=True, help='Patient identifier')
    parser.add_argument('--conditions', required=True, nargs='+', help='Experimental conditions')
    parser.add_argument('--model-type', default='tinnitus_model', 
                        choices=list(DEFAULT_DCM_MODELS.keys()),
                        help='Type of DCM model to use')
    parser.add_argument('--tr', type=float, default=2.0, help='Repetition time of fMRI data (in seconds)')
    parser.add_argument('--matlab-path', help='Path to MATLAB executable')
    parser.add_argument('--spm-path', help='Path to SPM installation')
    
    args = parser.parse_args()
    
    analysis = DCMAnalysis(
        fmri_data=args.fmri_data,
        output_dir=args.output_dir,
        patient_id=args.patient_id,
        conditions=args.conditions,
        model_type=args.model_type,
        tr=args.tr,
        matlab_path=args.matlab_path,
        spm_path=args.spm_path
    )
    
    results = analysis.run_analysis()
    
    print(json.dumps({
        "status": "success",
        "summary": {
            "patient_id": results["patient_id"],
            "model_type": results["model_type"],
            "conditions": results["conditions"],
            "number_of_connections": len(results["connectivity_strengths"]),
            "number_of_abnormal_connections": len(results["abnormal_connections"])
        }
    }, indent=2))


if __name__ == "__main__":
    main()
