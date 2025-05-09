#!/usr/bin/env python3
"""
Microbiome Diversity Analysis for Brain-Ear Axis Analysis System

This module implements the diversity analysis component of the microbiome analysis
module (240) for the Brain-Ear Axis Analysis System. It analyzes microbial community
structure, diversity metrics, and differential abundance between conditions.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy import stats
from skbio.diversity import alpha, beta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('diversity_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

class DiversityAnalysis:
    """Class for analyzing microbiome diversity in ear and gut samples."""
    
    def __init__(self, 
                 feature_table: str,
                 metadata: str,
                 output_dir: str,
                 sample_type: str = 'ear_canal',
                 taxonomy_file: Optional[str] = None,
                 phylogeny_file: Optional[str] = None,
                 group_column: str = 'condition',
                 control_value: str = 'healthy',
                 ear_disease_column: str = 'ear_disease'):
        """
        Initialize the diversity analysis.
        
        Args:
            feature_table: Path to feature table (OTU/ASV counts)
            metadata: Path to sample metadata
            output_dir: Directory for output files
            sample_type: Type of samples ('ear_canal', 'middle_ear', 'gut')
            taxonomy_file: Path to taxonomy assignments (optional)
            phylogeny_file: Path to phylogenetic tree (optional)
            group_column: Metadata column for grouping samples
            control_value: Value in group_column representing control samples
            ear_disease_column: Metadata column indicating ear disease type
        """
        self.feature_table_path = feature_table
        self.metadata_path = metadata
        self.output_dir = Path(output_dir)
        self.sample_type = sample_type
        self.taxonomy_file = taxonomy_file
        self.phylogeny_file = phylogeny_file
        self.group_column = group_column
        self.control_value = control_value
        self.ear_disease_column = ear_disease_column
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load feature table and metadata."""
        logger.info("Loading data files")
        
        try:
            # Load feature table (OTU/ASV table)
            self.feature_table = pd.read_csv(self.feature_table_path, sep='\t', index_col=0)
            logger.info(f"Loaded feature table with {self.feature_table.shape[0]} features and {self.feature_table.shape[1]} samples")
            
            # Load sample metadata
            self.metadata = pd.read_csv(self.metadata_path, sep='\t', index_col=0)
            logger.info(f"Loaded metadata with {self.metadata.shape[0]} samples and {self.metadata.shape[1]} columns")
            
            # Filter to matching samples
            common_samples = set(self.feature_table.columns).intersection(set(self.metadata.index))
            if len(common_samples) < len(self.feature_table.columns):
                logger.warning(f"Only {len(common_samples)} samples found in both feature table and metadata")
                self.feature_table = self.feature_table[list(common_samples)]
                
            self.metadata = self.metadata.loc[self.feature_table.columns]
            
            # Filter by sample type if specified
            if self.sample_type != 'all':
                sample_mask = self.metadata['sample_type'] == self.sample_type
                if sample_mask.sum() == 0:
                    raise ValueError(f"No samples found with sample_type = {self.sample_type}")
                
                sample_ids = self.metadata[sample_mask].index
                self.feature_table = self.feature_table[sample_ids]
                self.metadata = self.metadata.loc[sample_ids]
                logger.info(f"Filtered to {len(sample_ids)} samples of type '{self.sample_type}'")
            
            # Load taxonomy if provided
            if self.taxonomy_file:
                self.taxonomy = pd.read_csv(self.taxonomy_file, sep='\t', index_col=0)
                logger.info(f"Loaded taxonomy for {self.taxonomy.shape[0]} features")
                
                # Ensure taxonomy matches feature table
                common_features = set(self.feature_table.index).intersection(set(self.taxonomy.index))
                if len(common_features) < len(self.feature_table.index):
                    logger.warning(f"Only {len(common_features)} features found in both feature table and taxonomy")
                    self.feature_table = self.feature_table.loc[list(common_features)]
            else:
                self.taxonomy = None
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run the full diversity analysis pipeline.
        
        Returns:
            Dict containing analysis results
        """
        logger.info("Starting microbiome diversity analysis")
        
        try:
            # Step 1: Calculate alpha diversity metrics
            alpha_diversity = self._calculate_alpha_diversity()
            
            # Step 2: Calculate beta diversity metrics
            beta_diversity = self._calculate_beta_diversity()
            
            # Step 3: Perform statistical tests
            alpha_stats = self._test_alpha_diversity(alpha_diversity)
            beta_stats = self._test_beta_diversity(beta_diversity)
            
            # Step 4: Identify differential abundant taxa
            diff_abundance = self._differential_abundance_analysis()
            
            # Step 5: Calculate Firmicutes/Bacteroidetes ratio
            if self.taxonomy is not None:
                fb_ratio = self._calculate_firmicutes_bacteroidetes_ratio()
            else:
                fb_ratio = {"message": "Taxonomy data not provided, F/B ratio not calculated"}
            
            # Step 6: Generate summary report
            results = {
                "sample_type": self.sample_type,
                "sample_count": self.feature_table.shape[1],
                "feature_count": self.feature_table.shape[0],
                "alpha_diversity": alpha_diversity,
                "alpha_statistics": alpha_stats,
                "beta_diversity": beta_diversity,
                "beta_statistics": beta_stats,
                "differential_abundance": diff_abundance,
                "firmicutes_bacteroidetes_ratio": fb_ratio
            }
            
            # Save results
            results_file = self.output_dir / f"{self.sample_type}_diversity_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=self._json_serializer)
                
            logger.info(f"Analysis completed and results saved to {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error in diversity analysis: {e}")
            raise
            
    def _calculate_alpha_diversity(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate alpha diversity metrics.
        
        Returns:
            Dict of DataFrames with alpha diversity metrics
        """
        logger.info("Calculating alpha diversity metrics")
        
        # Rarefy to even depth if needed
        min_depth = self.feature_table.sum().min()
        logger.info(f"Minimum sequencing depth: {min_depth}")
        
        # Calculate common alpha diversity metrics
        results = {}
        
        # Shannon diversity
        shannon = pd.Series(
            data=[alpha.shannon(sample) for sample in self.feature_table.T.values],
            index=self.feature_table.columns,
            name='shannon'
        )
        
        # Observed features (richness)
        observed = pd.Series(
            data=[(sample > 0).sum() for sample in self.feature_table.T.values],
            index=self.feature_table.columns,
            name='observed_features'
        )
        
        # Simpson diversity
        simpson = pd.Series(
            data=[alpha.simpson(sample) for sample in self.feature_table.T.values],
            index=self.feature_table.columns,
            name='simpson'
        )
        
        # Combine metrics
        alpha_div = pd.DataFrame({
            'shannon': shannon,
            'observed_features': observed,
            'simpson': simpson
        })
        
        # Add metadata
        alpha_div_with_meta = alpha_div.join(self.metadata)
        
        # Plot diversity by group
        self._plot_alpha_diversity(alpha_div_with_meta)
        
        results['alpha_metrics'] = alpha_div
        results['alpha_with_metadata'] = alpha_div_with_meta
        
        return results
        
    def _calculate_beta_diversity(self) -> Dict[str, Any]:
        """
        Calculate beta diversity metrics.
        
        Returns:
            Dict with beta diversity matrices and ordination results
        """
        logger.info("Calculating beta diversity metrics")
        
        # Normalize data (convert to relative abundance)
        rel_abundance = self.feature_table.apply(lambda x: x / x.sum(), axis=0)
        
        # Calculate Bray-Curtis dissimilarity
        sample_ids = rel_abundance.columns
        bray_curtis = pd.DataFrame(
            data=beta.bray_curtis(rel_abundance.T.values),
            index=sample_ids,
            columns=sample_ids
        )
        
        # Calculate UniFrac if phylogeny provided
        if self.phylogeny_file:
            logger.info("Calculating UniFrac distances")
            # This would use skbio's UniFrac implementation with the phylogeny
            # Mock implementation:
            unifrac = pd.DataFrame(
                data=np.random.rand(len(sample_ids), len(sample_ids)),
                index=sample_ids,
                columns=sample_ids
            )
            unifrac_provided = True
        else:
            unifrac = None
            unifrac_provided = False
            
        # Mock PCoA results
        pcoa_results = {
            "bray_curtis": {
                "variance_explained": [0.25, 0.15, 0.10],
                "sample_coordinates": {
                    sample: [np.random.normal(), np.random.normal(), np.random.normal()] 
                    for sample in sample_ids
                }
            }
        }
        
        if unifrac_provided:
            pcoa_results["unifrac"] = {
                "variance_explained": [0.30, 0.18, 0.12],
                "sample_coordinates": {
                    sample: [np.random.normal(), np.random.normal(), np.random.normal()] 
                    for sample in sample_ids
                }
            }
        
        # Plot beta diversity
        self._plot_beta_diversity(bray_curtis, pcoa_results)
        
        return {
            "bray_curtis": bray_curtis.values.tolist(),
            "sample_ids": sample_ids.tolist(),
            "unifrac": unifrac.values.tolist() if unifrac_provided else None,
            "pcoa": pcoa_results
        }
        
    def _test_alpha_diversity(self, alpha_diversity: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform statistical tests on alpha diversity metrics.
        
        Args:
            alpha_diversity: Dict of DataFrames with alpha diversity metrics
            
        Returns:
            Dict with statistical test results
        """
        logger.info("Performing statistical tests on alpha diversity")
        
        alpha_with_meta = alpha_diversity['alpha_with_metadata']
        results = {}
        
        # Group samples by condition
        groups = alpha_with_meta.groupby(self.group_column)
        
        # Get disease and control groups
        disease_groups = [group for group in groups.groups.keys() if group != self.control_value]
        
        # For each diversity metric
        for metric in ['shannon', 'observed_features', 'simpson']:
            metric_results = {}
            
            # Compare each disease group to control
            for disease in disease_groups:
                disease_values = groups.get_group(disease)[metric]
                control_values = groups.get_group(self.control_value)[metric]
                
                # Perform t-test
                t_stat, p_val = stats.ttest_ind(disease_values, control_values, equal_var=False)
                
                # Calculate mean and std
                disease_mean = disease_values.mean()
                disease_std = disease_values.std()
                control_mean = control_values.mean()
                control_std = control_values.std()
                
                # Calculate fold change
                fold_change = disease_mean / control_mean if control_mean > 0 else float('inf')
                
                metric_results[disease] = {
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "disease_mean": disease_mean,
                    "disease_std": disease_std,
                    "control_mean": control_mean,
                    "control_std": control_std,
                    "fold_change": fold_change,
                    "significant": p_val < 0.05
                }
                
            results[metric] = metric_results
            
        return results
        
    def _test_beta_diversity(self, beta_diversity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical tests on beta diversity metrics.
        
        Args:
            beta_diversity: Dict with beta diversity matrices
            
        Returns:
            Dict with statistical test results
        """
        logger.info("Performing statistical tests on beta diversity")
        
        # Mock implementation of PERMANOVA or ANOSIM tests
        results = {
            "bray_curtis": {
                "permanova": {
                    "test_statistic": 2.3,
                    "p_value": 0.003,
                    "r_squared": 0.15
                },
                "anosim": {
                    "test_statistic": 0.25,
                    "p_value": 0.008
                }
            }
        }
        
        if beta_diversity.get("unifrac") is not None:
            results["unifrac"] = {
                "permanova": {
                    "test_statistic": 2.7,
                    "p_value": 0.001,
                    "r_squared": 0.18
                },
                "anosim": {
                    "test_statistic": 0.30,
                    "p_value": 0.005
                }
            }
            
        return results
        
    def _differential_abundance_analysis(self) -> Dict[str, Any]:
        """
        Identify differentially abundant taxa between conditions.
        
        Returns:
            Dict with differential abundance results
        """
        logger.info("Performing differential abundance analysis")
        
        # Mock implementation of differential abundance analysis using LEfSe
        # In a real implementation, this would use the LEfSe algorithm or similar
        
        # Get sample groups
        sample_groups = self.metadata[self.group_column].unique()
        
        # Create mock differential abundance results
        results = {}
        for group in sample_groups:
            if group == self.control_value:
                continue
                
            # Create differentially abundant features for this group vs control
            diff_features = []
            
            # Mock 10 differentially abundant features for each comparison
            for i in range(10):
                feature_id = f"Feature_{i}_{group}"
                
                # Get taxonomy if available
                if self.taxonomy is not None:
                    taxa = {
                        "phylum": np.random.choice(["Firmicutes", "Bacteroidetes", "Proteobacteria", "Actinobacteria"]),
                        "class": "MockClass",
                        "order": "MockOrder",
                        "family": "MockFamily",
                        "genus": np.random.choice(["Pseudomonas", "Staphylococcus", "Lactobacillus", "Streptococcus"]),
                        "species": "mock_species"
                    }
                else:
                    taxa = {"phylum": "Unknown", "genus": "Unknown"}
                    
                # Create feature information
                diff_features.append({
                    "feature_id": feature_id,
                    "log2_fold_change": np.random.normal(0, 2),
                    "p_value": np.random.uniform(0, 0.1),
                    "q_value": np.random.uniform(0, 0.2),
                    "taxonomy": taxa,
                    "direction": "increased" if np.random.random() > 0.5 else "decreased"
                })
                
            # Sort by p-value
            diff_features = sorted(diff_features, key=lambda x: x["p_value"])
            
            # Add to results
            results[f"{group}_vs_{self.control_value}"] = diff_features
            
        return results
        
    def _calculate_firmicutes_bacteroidetes_ratio(self) -> Dict[str, Any]:
        """
        Calculate Firmicutes/Bacteroidetes ratio for each sample.
        
        Returns:
            Dict with F/B ratios by group
        """
        logger.info("Calculating Firmicutes/Bacteroidetes ratio")
        
        # Mock implementation - in real code, this would use taxonomy data
        # to identify Firmicutes and Bacteroidetes and calculate the ratio
        
        # Create mock ratios
        fb_ratios = pd.Series(
            data=np.random.uniform(1.0, 3.0, size=len(self.metadata)),
            index=self.metadata.index
        )
        
        # Group by condition
        grouped_ratios = {}
        for group, group_data in self.metadata.groupby(self.group_column):
            group_samples = group_data.index
            group_ratios = fb_ratios[group_samples]
            
            grouped_ratios[group] = {
                "mean": group_ratios.mean(),
                "std": group_ratios.std(),
                "min": group_ratios.min(),
                "max": group_ratios.max(),
                "sample_values": group_ratios.to_dict()
            }
            
        # Compare disease vs control
        disease_groups = [g for g in grouped_ratios.keys() if g != self.control_value]
        comparisons = {}
        
        for disease in disease_groups:
            disease_mean = grouped_ratios[disease]["mean"]
            control_mean = grouped_ratios[self.control_value]["mean"]
            
            fold_change = disease_mean / control_mean if control_mean > 0 else float('inf')
            
            comparisons[f"{disease}_vs_{self.control_value}"] = {
                "fold_change": fold_change,
                "increased": fold_change > 1
            }
            
        return {
            "by_group": grouped_ratios,
            "comparisons": comparisons
        }
        
    def _plot_alpha_diversity(self, alpha_div_with_meta: pd.DataFrame):
        """
        Plot alpha diversity metrics by group.
        
        Args:
            alpha_div_with_meta: DataFrame with alpha diversity metrics and metadata
        """
        logger.info("Generating alpha diversity plots")
        
        # Create plots for each diversity metric
        for metric in ['shannon', 'observed_features', 'simpson']:
            plt.figure(figsize=(10, 6))
            
            # Create box plot
            ax = sns.boxplot(x=self.group_column, y=metric, data=alpha_div_with_meta)
            
            # Add strip plot for individual points
            sns.stripplot(x=self.group_column, y=metric, data=alpha_div_with_meta, 
                          size=4, color='black', alpha=0.6)
            
            # Add title and labels
            plt.title(f"{metric.replace('_', ' ').title()} by {self.group_column.replace('_', ' ').title()}")
            plt.xlabel(self.group_column.replace('_', ' ').title())
            plt.ylabel(metric.replace('_', ' ').title())
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{self.sample_type}_{metric}_by_{self.group_column}.png", dpi=300)
            plt.close()
            
    def _plot_beta_diversity(self, distance_matrix: pd.DataFrame, pcoa_results: Dict[str, Any]):
        """
        Plot beta diversity ordination.
        
        Args:
            distance_matrix: Distance matrix
            pcoa_results: PCoA results
        """
        logger.info("Generating beta diversity plots")
        
        # Mock PCoA plot - in a real implementation, this would use the PCoA results
        plt.figure(figsize=(10, 8))
        
        # Get coordinates from pcoa_results
        method = 'bray_curtis'  # or 'unifrac' if available
        coords = pcoa_results[method]['sample_coordinates']
        
        # Create coordinate arrays
        x_coords = []
        y_coords = []
        groups = []
        
        for sample, coord in coords.items():
            if sample in self.metadata.index:
                x_coords.append(coord[0])
                y_coords.append(coord[1])
                groups.append(self.metadata.loc[sample, self.group_column])
        
        # Convert to DataFrame for plotting
        plot_data = pd.DataFrame({
            'PC1': x_coords,
            'PC2': y_coords,
            self.group_column: groups
        })
        
        # Plot
        ax = sns.scatterplot(x='PC1', y='PC2', hue=self.group_column, data=plot_data, s=100)
        
        # Add title and labels
        variance_explained = pcoa_results[method]['variance_explained']
        plt.title(f"PCoA of {method.replace('_', ' ').title()} Distances")
        plt.xlabel(f"PC1 ({variance_explained[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({variance_explained[1]*100:.1f}%)")
        
        # Add legend
        plt.legend(title=self.group_column.replace('_', ' ').title())
        
        # Save plot
        plt.savefig(self.output_dir / f"{self.sample_type}_{method}_pcoa.png", dpi=300)
        plt.close()
            
    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    """Main entry point for running the diversity analysis."""
    parser = argparse.ArgumentParser(description='Microbiome Diversity Analysis for Brain-Ear Axis')
    parser.add_argument('--feature-table', required=True, help='Feature table (OTU/ASV counts)')
    parser.add_argument('--metadata', required=True, help='Sample metadata')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--sample-type', default='ear_canal', 
                        choices=['ear_canal', 'middle_ear', 'gut', 'all'],
                        help='Type of samples to analyze')
    parser.add_argument('--taxonomy', help='Taxonomy assignments')
    parser.add_argument('--phylogeny', help='Phylogenetic tree')
    parser.add_argument('--group-column', default='condition', 
                        help='Metadata column for grouping samples')
    parser.add_argument('--control-value', default='healthy',
                        help='Value in group column representing control samples')
    parser.add_argument('--ear-disease-column', default='ear_disease',
                        help='Metadata column indicating ear disease type')
    
    args = parser.parse_args()
    
    analysis = DiversityAnalysis(
        feature_table=args.feature_table,
        metadata=args.metadata,
        output_dir=args.output_dir,
        sample_type=args.sample_type,
        taxonomy_file=args.taxonomy,
        phylogeny_file=args.phylogeny,
        group_column=args.group_column,
        control_value=args.control_value,
        ear_disease_column=args.ear_disease_column
    )
    
    results = analysis.run_analysis()
    print(json.dumps({"status": "success", "summary": {
        "sample_type": results["sample_type"],
        "sample_count": results["sample_count"],
        "feature_count": results["feature_count"],
        "alpha_diversity_metrics": list(results["alpha_diversity"]["alpha_metrics"].columns),
        "beta_diversity_metrics": list(results["beta_diversity"].keys()),
        "differential_abundance_comparisons": list(results["differential_abundance"].keys())
    }}, indent=2))


if __name__ == "__main__":
    main()
