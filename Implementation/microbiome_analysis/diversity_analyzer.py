#!/usr/bin/env python3
"""
Microbiome Diversity Analysis Module

This module implements methods for analyzing microbiome diversity,
including alpha diversity, beta diversity, and taxonomic composition analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import skbio.diversity as skdiv
from skbio.stats.ordination import pcoa

class MicrobiomeDiversityAnalyzer:
    """
    Class for analyzing microbial diversity in microbiome samples.
    """
    
    def __init__(self, output_dir="results/microbiome_diversity"):
        """
        Initialize the diversity analyzer.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.abundance_data = None
        self.metadata = None
        self.taxonomy = None
        self.distance_matrix = None
    
    def load_data(self, abundance_data, metadata=None, taxonomy=None):
        """
        Load microbiome abundance data and optional metadata.
        
        Parameters:
        -----------
        abundance_data : pandas.DataFrame
            OTU/ASV abundance table with samples as rows and taxa as columns
        metadata : pandas.DataFrame, optional
            Sample metadata with samples as rows
        taxonomy : pandas.DataFrame, optional
            Taxonomic classification with taxa as rows
            
        Returns:
        --------
        abundance_data : pandas.DataFrame
            Loaded abundance data
        """
        self.abundance_data = abundance_data
        self.metadata = metadata
        self.taxonomy = taxonomy
        
        print(f"Loaded abundance data with {abundance_data.shape[0]} samples and {abundance_data.shape[1]} taxa")
        if metadata is not None:
            print(f"Loaded metadata with {metadata.shape[0]} samples and {metadata.shape[1]} features")
        if taxonomy is not None:
            print(f"Loaded taxonomy with {taxonomy.shape[0]} taxa")
        
        return abundance_data
    
    def preprocess_data(self, min_sample_count=1000, min_taxa_prevalence=0.1, normalize=True):
        """
        Preprocess microbiome data by filtering and normalizing.
        
        Parameters:
        -----------
        min_sample_count : int
            Minimum total count per sample
        min_taxa_prevalence : float
            Minimum prevalence (proportion of samples) for taxa to be retained
        normalize : bool
            Whether to normalize to relative abundance
            
        Returns:
        --------
        filtered_data : pandas.DataFrame
            Preprocessed abundance data
        """
        if self.abundance_data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        # Make a copy of the data
        filtered_data = self.abundance_data.copy()
        
        # Filter samples with low counts
        sample_counts = filtered_data.sum(axis=1)
        low_count_samples = sample_counts[sample_counts < min_sample_count].index
        if len(low_count_samples) > 0:
            filtered_data = filtered_data.drop(low_count_samples)
            print(f"Removed {len(low_count_samples)} samples with count < {min_sample_count}")
        
        # Filter low-prevalence taxa
        taxa_prevalence = (filtered_data > 0).mean(axis=0)
        low_prev_taxa = taxa_prevalence[taxa_prevalence < min_taxa_prevalence].index
        if len(low_prev_taxa) > 0:
            filtered_data = filtered_data.drop(columns=low_prev_taxa)
            print(f"Removed {len(low_prev_taxa)} taxa with prevalence < {min_taxa_prevalence}")
        
        # Normalize to relative abundance if requested
        if normalize:
            row_sums = filtered_data.sum(axis=1)
            filtered_data = filtered_data.div(row_sums, axis=0) * 100  # Convert to percentage
            print("Normalized data to relative abundance (%)")
        
        self.abundance_data = filtered_data
        return filtered_data
    
    def compute_alpha_diversity(self, metrics=None):
        """
        Compute alpha diversity metrics for each sample.
        
        Parameters:
        -----------
        metrics : list, optional
            List of alpha diversity metrics to compute
            
        Returns:
        --------
        alpha_diversity : pandas.DataFrame
            DataFrame with alpha diversity metrics for each sample
        """
        if self.abundance_data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        if metrics is None:
            metrics = ['observed_otus', 'shannon', 'simpson', 'chao1', 'pielou_e']
        
        # Convert to counts if data is in relative abundance
        if self.abundance_data.max().max() <= 1:
            counts = self.abundance_data * 1000  # Scale to avoid floating point issues
        else:
            counts = self.abundance_data
        
        # Initialize results
        alpha_div = {}
        
        # Compute each metric
        for metric in metrics:
            if metric == 'observed_otus':
                alpha_div[metric] = (counts > 0).sum(axis=1)
            
            elif metric == 'shannon':
                def shannon_index(x):
                    x_norm = x / x.sum()
                    return -np.sum(x_norm * np.log(x_norm + 1e-10))
                
                alpha_div[metric] = counts.apply(shannon_index, axis=1)
            
            elif metric == 'simpson':
                def simpson_index(x):
                    x_norm = x / x.sum()
                    return 1 - np.sum(x_norm ** 2)
                
                alpha_div[metric] = counts.apply(simpson_index, axis=1)
            
            elif metric == 'chao1':
                def chao1_index(x):
                    x = x[x > 0]  # Only consider observed species
                    singletons = np.sum(x == 1)
                    doubletons = np.sum(x == 2)
                    if doubletons == 0:
                        # Avoid division by zero
                        return len(x) + singletons * (singletons - 1) / (2 * (doubletons + 1))
                    else:
                        return len(x) + singletons**2 / (2 * doubletons)
                
                alpha_div[metric] = counts.apply(chao1_index, axis=1)
            
            elif metric == 'pielou_e':
                def pielou_evenness(x):
                    x_norm = x / x.sum()
                    shannon = -np.sum(x_norm * np.log(x_norm + 1e-10))
                    s = np.sum(x > 0)  # Observed species richness
                    if s <= 1:
                        return 0  # Avoid division by zero or log(1)=0
                    return shannon / np.log(s)
                
                alpha_div[metric] = counts.apply(pielou_evenness, axis=1)
        
        # Create DataFrame from results
        alpha_diversity = pd.DataFrame(alpha_div, index=counts.index)
        
        # Add metadata if available
        if self.metadata is not None:
            # Ensure the indices match
            common_samples = alpha_diversity.index.intersection(self.metadata.index)
            if len(common_samples) > 0:
                # Merge with metadata
                alpha_diversity = pd.merge(
                    alpha_diversity, 
                    self.metadata, 
                    left_index=True, 
                    right_index=True
                )
                print(f"Added metadata columns to alpha diversity results")
        
        # Save to CSV
        alpha_diversity.to_csv(os.path.join(self.output_dir, "alpha_diversity.csv"))
        
        # Plot alpha diversity
        self._plot_alpha_diversity(alpha_diversity)
        
        return alpha_diversity
    
    def _plot_alpha_diversity(self, alpha_diversity):
        """
        Plot alpha diversity metrics.
        
        Parameters:
        -----------
        alpha_diversity : pandas.DataFrame
            DataFrame with alpha diversity metrics
        """
        # Get diversity metrics (exclude metadata columns)
        if self.metadata is not None:
            diversity_cols = [col for col in alpha_diversity.columns if col in 
                             ['observed_otus', 'shannon', 'simpson', 'chao1', 'pielou_e']]
        else:
            diversity_cols = alpha_diversity.columns
        
        # Distribution of diversity metrics
        for metric in diversity_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(alpha_diversity[metric], kde=True)
            plt.title(f"Distribution of {metric}")
            plt.xlabel(metric)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"alpha_diversity_{metric}_distribution.png"))
        
        # If metadata is available, create boxplots by groups
        if self.metadata is not None:
            # Look for categorical columns with a reasonable number of categories
            for col in self.metadata.columns:
                if alpha_diversity[col].dtype == 'object' or alpha_diversity[col].dtype.name == 'category':
                    n_categories = alpha_diversity[col].nunique()
                    if 2 <= n_categories <= 10:  # Only plot if 2-10 categories
                        for metric in diversity_cols:
                            plt.figure(figsize=(12, 6))
                            sns.boxplot(x=col, y=metric, data=alpha_diversity)
                            plt.title(f"{metric} by {col}")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.savefig(os.path.join(self.output_dir, f"alpha_diversity_{metric}_by_{col}.png"))
    
    def compute_beta_diversity(self, metrics=None, ordination=True):
        """
        Compute beta diversity metrics between samples.
        
        Parameters:
        -----------
        metrics : list, optional
            List of beta diversity metrics to compute
        ordination : bool
            Whether to perform ordination (PCoA)
            
        Returns:
        --------
        distance_matrices : dict
            Dictionary of distance matrices for each metric
        """
        if self.abundance_data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        if metrics is None:
            metrics = ['bray_curtis', 'jaccard', 'unweighted_unifrac']
        
        # Initialize results
        distance_matrices = {}
        
        # Convert to array for calculations
        abundance_array = self.abundance_data.values
        sample_ids = self.abundance_data.index
        
        # Compute each metric
        for metric in metrics:
            if metric == 'bray_curtis':
                # Bray-Curtis dissimilarity
                dist_matrix = skdiv.beta_diversity(
                    "braycurtis", 
                    abundance_array, 
                    ids=sample_ids
                )
            
            elif metric == 'jaccard':
                # Jaccard distance (presence/absence)
                binary_array = (abundance_array > 0).astype(float)
                dist_matrix = skdiv.beta_diversity(
                    "jaccard", 
                    binary_array, 
                    ids=sample_ids
                )
            
            elif metric == 'unweighted_unifrac':
                # UniFrac requires phylogenetic tree
                # This is a simplified version using Jaccard as proxy
                print("Warning: UniFrac requires phylogenetic tree data. Using Jaccard as proxy.")
                binary_array = (abundance_array > 0).astype(float)
                dist_matrix = skdiv.beta_diversity(
                    "jaccard", 
                    binary_array, 
                    ids=sample_ids
                )
            
            else:
                # Use skbio for general metrics
                try:
                    dist_matrix = skdiv.beta_diversity(
                        metric, 
                        abundance_array, 
                        ids=sample_ids
                    )
                except ValueError:
                    print(f"Metric {metric} not supported. Skipping.")
                    continue
            
            distance_matrices[metric] = dist_matrix
            
            # Save distance matrix
            dm_df = pd.DataFrame(
                dist_matrix.data, 
                index=dist_matrix.ids, 
                columns=dist_matrix.ids
            )
            dm_df.to_csv(os.path.join(self.output_dir, f"beta_diversity_{metric}.csv"))
            
            # Visualize distance matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(dm_df, cmap='viridis')
            plt.title(f"{metric} Distance Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"beta_diversity_{metric}_heatmap.png"))
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(squareform(dist_matrix.data), method='average')
            plt.figure(figsize=(12, 8))
            dendrogram(
                linkage_matrix, 
                labels=dist_matrix.ids, 
                leaf_rotation=90
            )
            plt.title(f"Hierarchical Clustering based on {metric}")
            plt.xlabel("Samples")
            plt.ylabel("Distance")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"beta_diversity_{metric}_dendrogram.png"))
            
            # Perform ordination if requested
            if ordination:
                self._perform_ordination(dist_matrix, metric)
        
        # Store the first distance matrix for other methods
        if len(distance_matrices) > 0:
            self.distance_matrix = next(iter(distance_matrices.values()))
        
        return distance_matrices
    
    def _perform_ordination(self, distance_matrix, metric_name):
        """
        Perform ordination (PCoA) on a distance matrix.
        
        Parameters:
        -----------
        distance_matrix : skbio.DistanceMatrix
            Distance matrix
        metric_name : str
            Name of the metric used
            
        Returns:
        --------
        pcoa_results : pandas.DataFrame
            PCoA coordinates for each sample
        """
        # Perform PCoA
        pcoa_results = pcoa(distance_matrix)
        
        # Extract results
        pcoa_df = pcoa_results.samples.copy()
        
        # Add sample IDs as index
        pcoa_df.index = distance_matrix.ids
        
        # Save PCoA results
        pcoa_df.to_csv(os.path.join(self.output_dir, f"pcoa_{metric_name}.csv"))
        
        # Get explained variance
        explained_var = pcoa_results.proportion_explained
        
        # Create the scatter plot (first two components)
        plt.figure(figsize=(10, 8))
        
        # Basic scatter plot
        plt.scatter(
            pcoa_df.iloc[:, 0],
            pcoa_df.iloc[:, 1],
            alpha=0.8,
            s=50
        )
        
        # Add sample labels
        for i, sample_id in enumerate(pcoa_df.index):
            plt.annotate(
                sample_id,
                (pcoa_df.iloc[i, 0], pcoa_df.iloc[i, 1]),
                fontsize=8
            )
        
        # Add axis labels with explained variance
        plt.xlabel(f"PC1 ({explained_var[0]:.2%})")
        plt.ylabel(f"PC2 ({explained_var[1]:.2%})")
        plt.title(f"PCoA of {metric_name} Distances")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"pcoa_{metric_name}_scatter.png"))
        
        # If metadata is available, create colored plots by groups
        if self.metadata is not None:
            # Merge with metadata
            merged_df = pd.merge(pcoa_df, self.metadata, left_index=True, right_index=True)
            
            # Look for categorical columns with a reasonable number of categories
            for col in self.metadata.columns:
                if self.metadata[col].dtype == 'object' or self.metadata[col].dtype.name == 'category':
                    n_categories = self.metadata[col].nunique()
                    if 2 <= n_categories <= 10:  # Only plot if 2-10 categories
                        plt.figure(figsize=(10, 8))
                        
                        # Create a scatter plot with colors by group
                        sns.scatterplot(
                            x=merged_df.iloc[:, 0],
                            y=merged_df.iloc[:, 1],
                            hue=merged_df[col],
                            palette='colorblind',
                            s=50,
                            alpha=0.8
                        )
                        
                        # Add axis labels with explained variance
                        plt.xlabel(f"PC1 ({explained_var[0]:.2%})")
                        plt.ylabel(f"PC2 ({explained_var[1]:.2%})")
                        plt.title(f"PCoA of {metric_name} Distances by {col}")
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.output_dir, f"pcoa_{metric_name}_by_{col}.png"))
        
        return pcoa_df
    
    def analyze_taxonomic_composition(self, level='phylum', top_n=10, plot_heatmap=True):
        """
        Analyze and visualize taxonomic composition.
        
        Parameters:
        -----------
        level : str
            Taxonomic level for aggregation ('phylum', 'class', 'order', 'family', 'genus', 'species')
        top_n : int
            Number of top taxa to display
        plot_heatmap : bool
            Whether to plot a heatmap of abundances
            
        Returns:
        --------
        taxa_abundance : pandas.DataFrame
            Abundance of taxa at the specified level
        """
        if self.abundance_data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        # If taxonomy information is available, use it for aggregation
        if self.taxonomy is not None and level in self.taxonomy.columns:
            # Transpose abundance data to have taxa as rows
            abundance_t = self.abundance_data.T
            
            # Merge with taxonomy information
            merged = pd.merge(
                abundance_t, 
                self.taxonomy[level], 
                left_index=True, 
                right_index=True,
                how='left'
            )
            
            # Fill missing values
            merged[level] = merged[level].fillna('Unknown')
            
            # Group by taxonomic level and sum abundance
            taxa_abundance = merged.groupby(level).sum().T
            
        else:
            # If taxonomy info not available, use the raw data
            print(f"Warning: Taxonomy information for level '{level}' not available.")
            
            # Assume column names contain taxonomic information
            if all(';' in col for col in self.abundance_data.columns):
                # Extract the specified level from taxonomy strings
                # Assuming format like "k__Bacteria; p__Firmicutes; c__Bacilli; ..."
                level_prefix = level[0].lower() + "__"
                
                taxa_dict = {}
                for col in self.abundance_data.columns:
                    # Find the specific level in the taxonomy string
                    taxa_parts = col.split(';')
                    level_part = next((p for p in taxa_parts if p.strip().startswith(level_prefix)), 'Unknown')
                    taxon = level_part.strip().replace(level_prefix, '')
                    
                    if taxon not in taxa_dict:
                        taxa_dict[taxon] = np.zeros(self.abundance_data.shape[0])
                    
                    taxa_dict[taxon] += self.abundance_data[col].values
                
                # Create DataFrame from dictionary
                taxa_abundance = pd.DataFrame(
                    taxa_dict, 
                    index=self.abundance_data.index
                )
            else:
                print("Cannot aggregate by taxonomy. Using original data.")
                taxa_abundance = self.abundance_data
        
        # Calculate relative abundance if not already
        if taxa_abundance.max().max() > 1:
            taxa_abundance = taxa_abundance.div(taxa_abundance.sum(axis=1), axis=0) * 100
        
        # Get top N taxa by mean abundance
        top_taxa = taxa_abundance.mean().nlargest(top_n).index
        
        # Calculate "Other" category for remaining taxa
        other_taxa = taxa_abundance.drop(columns=top_taxa).sum(axis=1)
        
        # Create a new DataFrame with top taxa and "Other"
        top_taxa_abundance = taxa_abundance[top_taxa].copy()
        top_taxa_abundance['Other'] = other_taxa
        
        # Save to CSV
        top_taxa_abundance.to_csv(os.path.join(self.output_dir, f"taxonomic_composition_{level}.csv"))
        
        # Plot taxonomic composition as stacked bar
        plt.figure(figsize=(12, 8))
        top_taxa_abundance.plot(kind='bar', stacked=True, colormap='tab20')
        plt.title(f"Taxonomic Composition at {level.capitalize()} Level")
        plt.xlabel("Sample")
        plt.ylabel("Relative Abundance (%)")
        plt.xticks(rotation=90)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"taxonomic_composition_{level}_barplot.png"))
        
        # Plot mean abundance across samples
        plt.figure(figsize=(10, 6))
        mean_abundance = top_taxa_abundance.mean()
        mean_abundance.sort_values(ascending=True).plot(kind='barh', colormap='tab20')
        plt.title(f"Mean Abundance at {level.capitalize()} Level")
        plt.xlabel("Mean Relative Abundance (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"taxonomic_composition_{level}_mean.png"))
        
        # Plot heatmap if requested
        if plot_heatmap:
            plt.figure(figsize=(12, 10))
            sns.clustermap(
                top_taxa_abundance, 
                method='average', 
                cmap='viridis', 
                z_score=0,  # Scale rows
                xticklabels=True,
                yticklabels=True
            )
            plt.title(f"Heatmap of {level.capitalize()} Abundance")
            plt.savefig(os.path.join(self.output_dir, f"taxonomic_composition_{level}_heatmap.png"))
        
        return taxa_abundance
    
    def perform_differential_abundance(self, group_col, min_prevalence=0.2, fold_change_threshold=2):
        """
        Perform differential abundance analysis between groups.
        
        Parameters:
        -----------
        group_col : str
            Column in metadata for grouping samples
        min_prevalence : float
            Minimum prevalence (proportion of samples) for taxa to be considered
        fold_change_threshold : float
            Minimum fold change for reporting
            
        Returns:
        --------
        results : pandas.DataFrame
            Differential abundance results
        """
        if self.abundance_data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        if self.metadata is None:
            raise ValueError("No metadata loaded. Cannot perform differential abundance analysis.")
        
        if group_col not in self.metadata.columns:
            raise ValueError(f"Group column '{group_col}' not found in metadata.")
        
        # Ensure metadata index matches abundance data index
        common_samples = self.abundance_data.index.intersection(self.metadata.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples between abundance data and metadata.")
        
        # Subset data to common samples
        abundance = self.abundance_data.loc[common_samples]
        metadata = self.metadata.loc[common_samples]
        
        # Get groups
        groups = metadata[group_col].dropna().unique()
        
        if len(groups) < 2:
            raise ValueError(f"Need at least 2 groups for differential abundance, found {len(groups)}.")
        
        if len(groups) == 2:
            # Binary comparison
            group1, group2 = groups
            
            # Get samples for each group
            group1_samples = metadata[metadata[group_col] == group1].index
            group2_samples = metadata[metadata[group_col] == group2].index
            
            # Subset abundance data
            group1_abundance = abundance.loc[group1_samples]
            group2_abundance = abundance.loc[group2_samples]
            
            # Filter low prevalence taxa
            prevalence = (abundance > 0).mean(axis=0)
            high_prev_taxa = prevalence[prevalence >= min_prevalence].index
            
            # Subset to high prevalence taxa
            group1_abundance = group1_abundance[high_prev_taxa]
            group2_abundance = group2_abundance[high_prev_taxa]
            
            # Calculate mean abundance for each group
            group1_mean = group1_abundance.mean(axis=0)
            group2_mean = group2_abundance.mean(axis=0)
            
            # Calculate fold change
            # Add a small pseudocount to avoid division by zero
            pseudocount = 0.01 * min(group1_mean.min(), group2_mean.min())
            fold_change = (group1_mean + pseudocount) / (group2_mean + pseudocount)
            log2_fold_change = np.log2(fold_change)
            
            # Perform statistical test (Wilcoxon rank-sum test)
            p_values = []
            for taxon in high_prev_taxa:
                try:
                    from scipy.stats import mannwhitneyu
                    stat, p = mannwhitneyu(group1_abundance[taxon], group2_abundance[taxon])
                    p_values.append(p)
                except:
                    p_values.append(1.0)  # Default to non-significant
            
            # Multiple testing correction
            from statsmodels.stats.multitest import multipletests
            adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]
            
            # Create results DataFrame
            results = pd.DataFrame({
                'taxon': high_prev_taxa,
                f'mean_{group1}': group1_mean.values,
                f'mean_{group2}': group2_mean.values,
                'fold_change': fold_change.values,
                'log2_fold_change': log2_fold_change.values,
                'p_value': p_values,
                'adjusted_p_value': adjusted_p_values
            })
            
            # Filter by significance and fold change
            significant = results['adjusted_p_value'] < 0.05
            high_fold_change = np.abs(results['log2_fold_change']) >= np.log2(fold_change_threshold)
            significant_taxa = results[significant & high_fold_change].sort_values('adjusted_p_value')
            
            # Save results
            results.to_csv(os.path.join(self.output_dir, f"differential_abundance_{group_col}.csv"), index=False)
            significant_taxa.to_csv(os.path.join(self.output_dir, f"significant_taxa_{group_col}.csv"), index=False)
            
            # Plot volcano plot
            plt.figure(figsize=(10, 8))
            
            # Plot non-significant points
            plt.scatter(
                results.loc[~(significant & high_fold_change), 'log2_fold_change'],
                -np.log10(results.loc[~(significant & high_fold_change), 'p_value']),
                alpha=0.5,
                color='grey',
                label='Not significant'
            )
            
            # Plot significant points
            plt.scatter(
                results.loc[significant & high_fold_change, 'log2_fold_change'],
                -np.log10(results.loc[significant & high_fold_change, 'p_value']),
                alpha=0.8,
                color='red',
                label='Significant'
            )
            
            # Add labels for top significant taxa
            for i, row in significant_taxa.head(10).iterrows():
                plt.annotate(
                    row['taxon'],
                    (row['log2_fold_change'], -np.log10(row['p_value'])),
                    fontsize=8
                )
            
            # Add threshold lines
            plt.axhline(-np.log10(0.05), linestyle='--', color='grey', alpha=0.5)
            plt.axvline(np.log2(fold_change_threshold), linestyle='--', color='grey', alpha=0.5)
            plt.axvline(-np.log2(fold_change_threshold), linestyle='--', color='grey', alpha=0.5)
            
            plt.xlabel(f'Log2 Fold Change ({group1} / {group2})')
            plt.ylabel('-Log10 p-value')
            plt.title(f'Differential Abundance: {group1} vs {group2}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"volcano_plot_{group_col}.png"))
            
            # Bar plot of top differentially abundant taxa
            if len(significant_taxa) > 0:
                top_taxa = significant_taxa.head(15)
                plt.figure(figsize=(12, 8))
                top_taxa.sort_values('log2_fold_change', ascending=True, inplace=True)
                bar_colors = ['blue' if fc < 0 else 'red' for fc in top_taxa['log2_fold_change']]
                plt.barh(top_taxa['taxon'], top_taxa['log2_fold_change'], color=bar_colors)
                plt.axvline(0, color='black', linestyle='-', alpha=0.3)
                plt.xlabel(f'Log2 Fold Change ({group1} / {group2})')
                plt.title(f'Top Differentially Abundant Taxa: {group1} vs {group2}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"top_diff_abundant_taxa_{group_col}.png"))
            
            return significant_taxa
        
        else:
            # Multiple group comparison
            print(f"Multiple group comparison with groups: {groups}")
            
            # Filter low prevalence taxa
            prevalence = (abundance > 0).mean(axis=0)
            high_prev_taxa = prevalence[prevalence >= min_prevalence].index
            
            # Subset to high prevalence taxa
            filtered_abundance = abundance[high_prev_taxa]
            
            # Placeholder for multiple group comparison
            # For now, just return the filtered data
            print("Multiple group comparison not fully implemented. Returning filtered data.")
            
            return pd.DataFrame({
                'taxon': high_prev_taxa,
                'prevalence': prevalence[high_prev_taxa].values
            })
    
    def perform_correlation_analysis(self, metadata_col, min_prevalence=0.2, correlation_method='spearman'):
        """
        Correlate taxon abundance with a continuous metadata variable.
        
        Parameters:
        -----------
        metadata_col : str
            Column in metadata to correlate with
        min_prevalence : float
            Minimum prevalence (proportion of samples) for taxa to be considered
        correlation_method : str
            Method for correlation: 'pearson', 'spearman', or 'kendall'
            
        Returns:
        --------
        correlation_results : pandas.DataFrame
            Correlation results for each taxon
        """
        if self.abundance_data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        if self.metadata is None:
            raise ValueError("No metadata loaded. Cannot perform correlation analysis.")
        
        if metadata_col not in self.metadata.columns:
            raise ValueError(f"Metadata column '{metadata_col}' not found.")
        
        # Ensure data is numeric
        if not np.issubdtype(self.metadata[metadata_col].dtype, np.number):
            raise ValueError(f"Metadata column '{metadata_col}' is not numeric.")
        
        # Ensure metadata index matches abundance data index
        common_samples = self.abundance_data.index.intersection(self.metadata.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples between abundance data and metadata.")
        
        # Subset data to common samples
        abundance = self.abundance_data.loc[common_samples]
        metadata = self.metadata.loc[common_samples]
        
        # Filter low prevalence taxa
        prevalence = (abundance > 0).mean(axis=0)
        high_prev_taxa = prevalence[prevalence >= min_prevalence].index
        filtered_abundance = abundance[high_prev_taxa]
        
        # Perform correlation for each taxon
        correlation_results = []
        
        for taxon in high_prev_taxa:
            # Get abundance values and metadata
            taxon_abundance = filtered_abundance[taxon]
            meta_values = metadata[metadata_col]
            
            # Remove samples with NaN values
            valid_idx = ~(np.isnan(taxon_abundance) | np.isnan(meta_values))
            if valid_idx.sum() < 5:  # Skip if too few valid samples
                continue
                
            taxon_abundance = taxon_abundance[valid_idx]
            meta_values = meta_values[valid_idx]
            
            # Calculate correlation
            if correlation_method == 'pearson':
                from scipy.stats import pearsonr
                corr, p_value = pearsonr(taxon_abundance, meta_values)
            elif correlation_method == 'spearman':
                from scipy.stats import spearmanr
                corr, p_value = spearmanr(taxon_abundance, meta_values)
            elif correlation_method == 'kendall':
                from scipy.stats import kendalltau
                corr, p_value = kendalltau(taxon_abundance, meta_values)
            else:
                raise ValueError(f"Correlation method '{correlation_method}' not supported.")
                
            # Add to results
            correlation_results.append({
                'taxon': taxon,
                'correlation': corr,
                'p_value': p_value,
                'method': correlation_method
            })
        
        if not correlation_results:
            print("No valid correlations found.")
            return pd.DataFrame()
            
        # Create DataFrame from results
        results_df = pd.DataFrame(correlation_results)
        
        # Multiple testing correction
        from statsmodels.stats.multitest import multipletests
        results_df['adjusted_p_value'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        
        # Sort by absolute correlation
        results_df['abs_correlation'] = np.abs(results_df['correlation'])
        results_df.sort_values('abs_correlation', ascending=False, inplace=True)
        results_df.drop('abs_correlation', axis=1, inplace=True)
        
        # Save results
        results_df.to_csv(os.path.join(self.output_dir, f"correlation_{metadata_col}_{correlation_method}.csv"), index=False)
        
        # Plot top correlations
        significant = results_df['adjusted_p_value'] < 0.05
        top_taxa = results_df[significant].head(15)
        
        if len(top_taxa) > 0:
            plt.figure(figsize=(12, 8))
            bar_colors = ['blue' if c < 0 else 'red' for c in top_taxa['correlation']]
            plt.barh(top_taxa['taxon'], top_taxa['correlation'], color=bar_colors)
            plt.axvline(0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel(f'Correlation ({correlation_method})')
            plt.title(f'Top Taxa Correlated with {metadata_col}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"top_correlated_taxa_{metadata_col}.png"))
            
            # Scatter plots for top 6 taxa
            top6_taxa = top_taxa.head(6)
            if len(top6_taxa) > 0:
                # Create a grid of scatter plots
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for i, (_, row) in enumerate(top6_taxa.iterrows()):
                    ax = axes[i]
                    taxon = row['taxon']
                    
                    taxon_abundance = filtered_abundance[taxon]
                    meta_values = metadata[metadata_col]
                    
                    # Remove samples with NaN values
                    valid_idx = ~(np.isnan(taxon_abundance) | np.isnan(meta_values))
                    taxon_abundance = taxon_abundance[valid_idx]
                    meta_values = meta_values[valid_idx]
                    
                    # Plot scatter
                    ax.scatter(taxon_abundance, meta_values, alpha=0.7)
                    
                    # Add regression line
                    from scipy import stats
                    slope, intercept, _, _, _ = stats.linregress(taxon_abundance, meta_values)
                    x_vals = np.linspace(taxon_abundance.min(), taxon_abundance.max(), 100)
                    y_vals = intercept + slope * x_vals
                    ax.plot(x_vals, y_vals, 'r')
                    
                    # Add correlation info
                    ax.set_title(f"{taxon}\nr={row['correlation']:.2f}, p={row['adjusted_p_value']:.3f}")
                    ax.set_xlabel("Abundance")
                    ax.set_ylabel(metadata_col)
                    
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"correlation_scatter_{metadata_col}.png"))
        
        return results_df
    
    def analyze_network(self, correlation_threshold=0.5, p_value_threshold=0.05, min_prevalence=0.3):
        """
        Create and analyze a correlation network of taxa.
        
        Parameters:
        -----------
        correlation_threshold : float
            Minimum absolute correlation to include in the network
        p_value_threshold : float
            Maximum p-value to consider a correlation significant
        min_prevalence : float
            Minimum prevalence (proportion of samples) for taxa to be considered
            
        Returns:
        --------
        network_stats : dict
            Network statistics
        """
        if self.abundance_data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX not found. Install it with 'pip install networkx'.")
            return None
        
        # Filter low prevalence taxa
        prevalence = (self.abundance_data > 0).mean(axis=0)
        high_prev_taxa = prevalence[prevalence >= min_prevalence].index
        filtered_abundance = self.abundance_data[high_prev_taxa]
        
        # Calculate correlation matrix
        from scipy.stats import spearmanr
        corr_matrix, p_matrix = spearmanr(filtered_abundance, axis=0)
        
        # Convert to DataFrames
        corr_df = pd.DataFrame(corr_matrix, index=high_prev_taxa, columns=high_prev_taxa)
        p_df = pd.DataFrame(p_matrix, index=high_prev_taxa, columns=high_prev_taxa)
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for taxon in high_prev_taxa:
            G.add_node(taxon, prevalence=prevalence[taxon])
        
        # Add edges (correlations)
        for i, taxon1 in enumerate(high_prev_taxa):
            for j, taxon2 in enumerate(high_prev_taxa):
                if i < j:  # Lower triangular matrix only
                    corr = corr_df.loc[taxon1, taxon2]
                    p_val = p_df.loc[taxon1, taxon2]
                    
                    if np.abs(corr) >= correlation_threshold and p_val <= p_value_threshold:
                        G.add_edge(taxon1, taxon2, weight=corr, p_value=p_val)
        
        # Calculate network statistics
        network_stats = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'connected_components': nx.number_connected_components(G)
        }
        
        # Calculate node centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Create a DataFrame with node statistics
        node_stats = pd.DataFrame({
            'node': list(G.nodes()),
            'degree': [G.degree(node) for node in G.nodes()],
            'degree_centrality': [degree_centrality[node] for node in G.nodes()],
            'betweenness_centrality': [betweenness_centrality[node] for node in G.nodes()],
            'closeness_centrality': [closeness_centrality[node] for node in G.nodes()],
            'prevalence': [G.nodes[node]['prevalence'] for node in G.nodes()]
        })
        
        # Save network statistics
        with open(os.path.join(self.output_dir, "network_stats.txt"), 'w') as f:
            for key, value in network_stats.items():
                f.write(f"{key}: {value}\n")
        
        node_stats.to_csv(os.path.join(self.output_dir, "node_stats.csv"), index=False)
        
        # Visualize network
        plt.figure(figsize=(12, 12))
        
        # Set node positions
        pos = nx.spring_layout(G, seed=42)
        
        # Set node sizes based on degree
        node_sizes = [50 + 20 * G.degree(node) for node in G.nodes()]
        
        # Set edge colors based on correlation sign
        edge_colors = ['red' if G[u][v]['weight'] < 0 else 'green' for u, v in G.edges()]
        
        # Set edge widths based on correlation strength
        edge_widths = [1 + 2 * np.abs(G[u][v]['weight']) for u, v in G.edges()]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=node_sizes, 
            node_color=list(node_stats['degree_centrality']),
            cmap=plt.cm.viridis,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, 
            width=edge_widths, 
            edge_color=edge_colors, 
            alpha=0.6
        )
        
        # Draw labels for nodes with high degree
        high_degree_nodes = [node for node, degree in G.degree() if degree > 2]
        nx.draw_networkx_labels(
            G, pos, 
            labels={node: node for node in high_degree_nodes},
            font_size=8
        )
        
        plt.title(f"Microbial Correlation Network (|r| ≥ {correlation_threshold}, p ≤ {p_value_threshold})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlation_network.png"))
        
        # Plot degree distribution
        plt.figure(figsize=(10, 6))
        degrees = [G.degree(node) for node in G.nodes()]
        plt.hist(degrees, bins=max(degrees), alpha=0.7)
        plt.xlabel("Node Degree")
        plt.ylabel("Frequency")
        plt.title("Degree Distribution")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "degree_distribution.png"))
        
        return network_stats, node_stats


def run_diversity_analysis(abundance_data, metadata=None, taxonomy=None, output_dir="results/microbiome_diversity"):
    """
    Run the complete microbiome diversity analysis workflow.
    
    Parameters:
    -----------
    abundance_data : pandas.DataFrame
        OTU/ASV abundance table with samples as rows and taxa as columns
    metadata : pandas.DataFrame, optional
        Sample metadata with samples as rows
    taxonomy : pandas.DataFrame, optional
        Taxonomic classification with taxa as rows
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    analyzer : MicrobiomeDiversityAnalyzer
        The diversity analyzer object
    """
    # Initialize diversity analyzer
    analyzer = MicrobiomeDiversityAnalyzer(output_dir=output_dir)
    
    # Load data
    analyzer.load_data(abundance_data, metadata, taxonomy)
    
    # Preprocess data
    analyzer.preprocess_data(
        min_sample_count=1000,
        min_taxa_prevalence=0.1,
        normalize=True
    )
    
    # Compute alpha diversity
    alpha_diversity = analyzer.compute_alpha_diversity()
    
    # Compute beta diversity
    beta_diversity = analyzer.compute_beta_diversity(metrics=['bray_curtis', 'jaccard'])
    
    # Analyze taxonomic composition at different levels
    for level in ['phylum', 'class', 'family', 'genus']:
        try:
            analyzer.analyze_taxonomic_composition(level=level, top_n=10)
        except Exception as e:
            print(f"Error analyzing {level} level: {str(e)}")
    
    # If metadata is available, perform additional analyses
    if metadata is not None:
        # Look for categorical columns with a reasonable number of categories
        for col in metadata.columns:
            if metadata[col].dtype == 'object' or metadata[col].dtype.name == 'category':
                n_categories = metadata[col].nunique()
                if 2 <= n_categories <= 10:
                    try:
                        analyzer.perform_differential_abundance(group_col=col)
                    except Exception as e:
                        print(f"Error in differential abundance for {col}: {str(e)}")
        
        # Look for numeric columns for correlation analysis
        for col in metadata.columns:
            if np.issubdtype(metadata[col].dtype, np.number):
                try:
                    analyzer.perform_correlation_analysis(metadata_col=col)
                except Exception as e:
                    print(f"Error in correlation analysis for {col}: {str(e)}")
    
    # Perform network analysis
    try:
        analyzer.analyze_network()
    except Exception as e:
        print(f"Error in network analysis: {str(e)}")
    
    print("Diversity analysis completed successfully!")
    print(f"Results saved to {output_dir}")
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("../..")  # Add project root to path
    
    from data.loaders import load_microbiome_data, load_metadata, load_taxonomy
    
    # Load example data
    abundance_data = load_microbiome_data("data/raw/microbiome_abundance.csv")
    metadata = load_metadata("data/raw/sample_metadata.csv")
    taxonomy = load_taxonomy("data/raw/taxonomy.csv")
    
    # Run analysis
    analyzer = run_diversity_analysis(
        abundance_data,
        metadata=metadata,
        taxonomy=taxonomy,
        output_dir="results/microbiome_diversity"
    )
