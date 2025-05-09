#!/usr/bin/env python3
"""
Integrated Data Processing Module

This module implements methods for integrating and analyzing multimodal data
from neural signals, microbiome, and functional connectivity in the brain-ear axis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cross_decomposition import CCA
import matplotlib.patches as mpatches

class IntegratedDataProcessor:
    """
    Class for integrating and analyzing multimodal data from the brain-ear axis.
    """
    
    def __init__(self, output_dir="results/integrated_analysis"):
        """
        Initialize the integrated data processor.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save integrated analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.neural_data = None
        self.microbiome_data = None
        self.connectivity_data = None
        self.metadata = None
        self.integrated_data = None
    
    def load_data(self, neural_data=None, microbiome_data=None, connectivity_data=None, metadata=None):
        """
        Load multimodal data for integrated analysis.
        
        Parameters:
        -----------
        neural_data : pandas.DataFrame, optional
            Neural signal features with samples as rows
        microbiome_data : pandas.DataFrame, optional
            Microbiome features with samples as rows
        connectivity_data : pandas.DataFrame, optional
            Connectivity features with samples as rows
        metadata : pandas.DataFrame, optional
            Sample metadata with samples as rows
            
        Returns:
        --------
        common_samples : list
            List of common samples across all datasets
        """
        self.neural_data = neural_data
        self.microbiome_data = microbiome_data
        self.connectivity_data = connectivity_data
        self.metadata = metadata
        
        # Print information about loaded data
        if neural_data is not None:
            print(f"Loaded neural data with {neural_data.shape[0]} samples and {neural_data.shape[1]} features")
        if microbiome_data is not None:
            print(f"Loaded microbiome data with {microbiome_data.shape[0]} samples and {microbiome_data.shape[1]} features")
        if connectivity_data is not None:
            print(f"Loaded connectivity data with {connectivity_data.shape[0]} samples and {connectivity_data.shape[1]} features")
        if metadata is not None:
            print(f"Loaded metadata with {metadata.shape[0]} samples and {metadata.shape[1]} features")
        
        # Find common samples across all datasets
        sample_sets = []
        if neural_data is not None:
            sample_sets.append(set(neural_data.index))
        if microbiome_data is not None:
            sample_sets.append(set(microbiome_data.index))
        if connectivity_data is not None:
            sample_sets.append(set(connectivity_data.index))
        if metadata is not None:
            sample_sets.append(set(metadata.index))
        
        if not sample_sets:
            return []
        
        common_samples = set.intersection(*sample_sets)
        print(f"Found {len(common_samples)} common samples across all datasets")
        
        return list(common_samples)
    
    def preprocess_data(self, scale=True, impute_missing=True, common_samples=None):
        """
        Preprocess multimodal data for integrated analysis.
        
        Parameters:
        -----------
        scale : bool
            Whether to scale features
        impute_missing : bool
            Whether to impute missing values
        common_samples : list, optional
            List of samples to keep in all datasets
            
        Returns:
        --------
        processed_data : dict
            Dictionary of preprocessed data
        """
        # Get common samples if not provided
        if common_samples is None:
            common_samples = self.load_data(
                self.neural_data, 
                self.microbiome_data, 
                self.connectivity_data, 
                self.metadata
            )
        
        if not common_samples:
            raise ValueError("No common samples found across datasets.")
        
        # Initialize dict for processed data
        processed_data = {}
        
        # Function to process a single dataframe
        def process_df(df, name):
            if df is None:
                return None
            
            # Subset to common samples
            df_common = df.loc[common_samples].copy()
            
            # Handle missing values if present
            if impute_missing and df_common.isnull().sum().sum() > 0:
                # Simple imputation - replace with column means
                df_common = df_common.fillna(df_common.mean())
                print(f"Imputed missing values in {name} data")
            
            # Scale features
            if scale:
                scaler = StandardScaler()
                # Keep only numeric columns for scaling
                numeric_cols = df_common.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df_common[numeric_cols] = scaler.fit_transform(df_common[numeric_cols])
                    print(f"Scaled numeric features in {name} data")
            
            return df_common
        
        # Process each dataset
        if self.neural_data is not None:
            processed_data['neural'] = process_df(self.neural_data, 'neural')
        
        if self.microbiome_data is not None:
            processed_data['microbiome'] = process_df(self.microbiome_data, 'microbiome')
        
        if self.connectivity_data is not None:
            processed_data['connectivity'] = process_df(self.connectivity_data, 'connectivity')
        
        if self.metadata is not None:
            # For metadata, we don't need to scale
            processed_data['metadata'] = self.metadata.loc[common_samples].copy()
        
        # Update instance variables
        self.neural_data = processed_data.get('neural')
        self.microbiome_data = processed_data.get('microbiome')
        self.connectivity_data = processed_data.get('connectivity')
        self.metadata = processed_data.get('metadata')
        
        return processed_data
    
    def integrate_data(self, method='concatenate', n_components=None):
        """
        Integrate multimodal data using different methods.
        
        Parameters:
        -----------
        method : str
            Integration method: 'concatenate', 'pca', or 'cca'
        n_components : int, optional
            Number of components for dimensionality reduction
            
        Returns:
        --------
        integrated_data : pandas.DataFrame
            Integrated data with samples as rows
        """
        # Get list of available datasets
        datasets = []
        dataset_names = []
        if self.neural_data is not None:
            datasets.append(self.neural_data.select_dtypes(include=[np.number]))
            dataset_names.append('neural')
        if self.microbiome_data is not None:
            datasets.append(self.microbiome_data.select_dtypes(include=[np.number]))
            dataset_names.append('microbiome')
        if self.connectivity_data is not None:
            datasets.append(self.connectivity_data.select_dtypes(include=[np.number]))
            dataset_names.append('connectivity')
        
        if len(datasets) < 1:
            raise ValueError("At least one dataset is required for integration.")
        
        # Ensure all datasets have the same samples and are sorted
        common_samples = set.intersection(*[set(ds.index) for ds in datasets])
        if not common_samples:
            raise ValueError("No common samples found across datasets.")
        
        common_samples = sorted(list(common_samples))
        datasets = [ds.loc[common_samples] for ds in datasets]
        
        # Integration methods
        if method == 'concatenate':
            # Simple concatenation of features
            df_merged = pd.concat(datasets, axis=1)
            
            # Rename duplicate columns if present
            df_merged.columns = pd.Index([
                f"{col}_{i}" if col in df_merged.columns[:i] else col
                for i, col in enumerate(df_merged.columns)
            ])
            
            self.integrated_data = df_merged
            print(f"Integrated data by concatenation: {df_merged.shape[0]} samples, {df_merged.shape[1]} features")
            
        elif method == 'pca':
            # Apply PCA to each dataset first
            pca_results = []
            
            for i, df in enumerate(datasets):
                if n_components is None:
                    # Determine number of components to explain 90% variance
                    pca = PCA(n_components=0.9)
                else:
                    # Use specified number of components
                    pca = PCA(n_components=min(n_components, df.shape[1], df.shape[0]))
                
                pca_result = pca.fit_transform(df)
                pca_df = pd.DataFrame(
                    pca_result,
                    index=df.index,
                    columns=[f"{dataset_names[i]}_PC{j+1}" for j in range(pca_result.shape[1])]
                )
                
                pca_results.append(pca_df)
                print(f"Applied PCA to {dataset_names[i]} data: {pca_df.shape[1]} components, "
                     f"{sum(pca.explained_variance_ratio_)*100:.1f}% variance explained")
            
            # Concatenate PCA results
            df_merged = pd.concat(pca_results, axis=1)
            self.integrated_data = df_merged
            print(f"Integrated data by PCA: {df_merged.shape[0]} samples, {df_merged.shape[1]} features")
            
        elif method == 'cca':
            if len(datasets) != 2:
                raise ValueError("CCA method requires exactly 2 datasets.")
            
            # Apply CCA to two datasets
            X = datasets[0].values
            Y = datasets[1].values
            
            if n_components is None:
                # Determine number of components as min(n_features1, n_features2, n_samples)
                n_components = min(X.shape[1], Y.shape[1], X.shape[0])
            else:
                n_components = min(n_components, X.shape[1], Y.shape[1], X.shape[0])
            
            cca = CCA(n_components=n_components)
            X_c, Y_c = cca.fit_transform(X, Y)
            
            # Create dataframes for canonical variables
            X_df = pd.DataFrame(
                X_c,
                index=datasets[0].index,
                columns=[f"{dataset_names[0]}_CV{j+1}" for j in range(X_c.shape[1])]
            )
            
            Y_df = pd.DataFrame(
                Y_c,
                index=datasets[1].index,
                columns=[f"{dataset_names[1]}_CV{j+1}" for j in range(Y_c.shape[1])]
            )
            
            # Combine canonical variables
            df_merged = pd.concat([X_df, Y_df], axis=1)
            self.integrated_data = df_merged
            print(f"Integrated data by CCA: {df_merged.shape[0]} samples, {df_merged.shape[1]} features")
            
        else:
            raise ValueError(f"Integration method '{method}' not supported. Use 'concatenate', 'pca', or 'cca'.")
        
        # Save integrated data
        self.integrated_data.to_csv(os.path.join(self.output_dir, f"integrated_data_{method}.csv"))
        
        return self.integrated_data
    
    def reduce_dimensions(self, method='pca', n_components=2):
        """
        Reduce the dimensionality of integrated data for visualization.
        
        Parameters:
        -----------
        method : str
            Dimensionality reduction method: 'pca' or 'tsne'
        n_components : int
            Number of components to retain
            
        Returns:
        --------
        reduced_data : pandas.DataFrame
            Reduced data with samples as rows
        """
        if self.integrated_data is None:
            raise ValueError("Integrated data not available. Run integrate_data first.")
        
        # Get numeric data for dimensionality reduction
        numeric_data = self.integrated_data.select_dtypes(include=[np.number])
        
        if method == 'pca':
            # Apply PCA
            pca = PCA(n_components=min(n_components, numeric_data.shape[1], numeric_data.shape[0]))
            reduced = pca.fit_transform(numeric_data)
            
            # Create dataframe
            reduced_df = pd.DataFrame(
                reduced,
                index=numeric_data.index,
                columns=[f"PC{i+1}" for i in range(reduced.shape[1])]
            )
            
            # Add variance explained
            for i, var in enumerate(pca.explained_variance_ratio_):
                print(f"PC{i+1}: {var*100:.2f}% variance explained")
            
            # Plot variance explained
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Explained Variance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "pca_explained_variance.png"))
            
        elif method == 'tsne':
            # Apply t-SNE
            tsne = TSNE(n_components=n_components, random_state=42)
            reduced = tsne.fit_transform(numeric_data)
            
            # Create dataframe
            reduced_df = pd.DataFrame(
                reduced,
                index=numeric_data.index,
                columns=[f"TSNE{i+1}" for i in range(reduced.shape[1])]
            )
            
            print("Applied t-SNE dimensionality reduction")
            
        else:
            raise ValueError(f"Dimensionality reduction method '{method}' not supported. Use 'pca' or 'tsne'.")
        
        # Save reduced data
        reduced_df.to_csv(os.path.join(self.output_dir, f"reduced_data_{method}.csv"))
        
        # Visualize reduced data
        if n_components == 2:
            self._plot_reduced_data(reduced_df, method)
        
        return reduced_df
    
    def _plot_reduced_data(self, reduced_df, method):
        """
        Visualize reduced data.
        
        Parameters:
        -----------
        reduced_df : pandas.DataFrame
            Reduced data with 2 components
        method : str
            Method used for dimensionality reduction
        """
        # Basic scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(
            reduced_df.iloc[:, 0],
            reduced_df.iloc[:, 1],
            alpha=0.8,
            s=50
        )
        
        # Add sample labels
        for i, sample in enumerate(reduced_df.index):
            plt.annotate(
                sample,
                (reduced_df.iloc[i, 0], reduced_df.iloc[i, 1]),
                fontsize=8
            )
        
        # Set axis labels
        if method == 'pca':
            plt.xlabel('PC1')
            plt.ylabel('PC2')
        else:
            plt.xlabel('TSNE1')
            plt.ylabel('TSNE2')
        
        plt.title(f'2D {method.upper()} Projection of Integrated Data')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{method}_projection.png"))
        
        # If metadata is available, create colored plots by groups
        if self.metadata is not None:
            # Merge with metadata
            merged_df = pd.merge(reduced_df, self.metadata, left_index=True, right_index=True)
            
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
                        
                        # Set axis labels
                        if method == 'pca':
                            plt.xlabel('PC1')
                            plt.ylabel('PC2')
                        else:
                            plt.xlabel('TSNE1')
                            plt.ylabel('TSNE2')
                        
                        plt.title(f'{method.upper()} Projection by {col}')
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.output_dir, f"{method}_projection_by_{col}.png"))
    
    def perform_clustering(self, n_clusters=None, method='kmeans'):
        """
        Perform clustering on integrated data.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters
        method : str
            Clustering method: 'kmeans' or 'hierarchical'
            
        Returns:
        --------
        clusters : pandas.Series
            Cluster assignments for each sample
        """
        if self.integrated_data is None:
            raise ValueError("Integrated data not available. Run integrate_data first.")
        
        # Get numeric data for clustering
        numeric_data = self.integrated_data.select_dtypes(include=[np.number])
        
        # Determine number of clusters if not provided
        if n_clusters is None:
            # Simple heuristic: square root of number of samples
            n_clusters = min(int(np.sqrt(numeric_data.shape[0])), 10)
        
        if method == 'kmeans':
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(numeric_data)
            
            # Create series with cluster assignments
            clusters_series = pd.Series(
                clusters,
                index=numeric_data.index,
                name='cluster'
            )
            
            print(f"Performed K-means clustering with {n_clusters} clusters")
            
            # Visualize clusters with reduced dimensions
            reduced_df = self.reduce_dimensions(method='pca', n_components=2)
            merged_df = pd.merge(reduced_df, clusters_series, left_index=True, right_index=True)
            
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                x=merged_df.iloc[:, 0],
                y=merged_df.iloc[:, 1],
                hue=merged_df['cluster'],
                palette='tab10',
                s=50,
                alpha=0.8
            )
            
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title(f'K-means Clustering (k={n_clusters})')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"kmeans_clusters_{n_clusters}.png"))
            
            # Save cluster assignments
            clusters_series.to_csv(os.path.join(self.output_dir, f"kmeans_clusters_{n_clusters}.csv"))
            
        elif method == 'hierarchical':
            # Apply hierarchical clustering
            linkage_matrix = linkage(numeric_data, method='ward')
            
            # Plot dendrogram
            plt.figure(figsize=(12, 8))
            dendrogram(
                linkage_matrix,
                labels=numeric_data.index,
                leaf_rotation=90
            )
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Samples')
            plt.ylabel('Distance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "hierarchical_clustering.png"))
            
            # Get cluster assignments for the desired number of clusters
            from scipy.cluster.hierarchy import fcluster
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Create series with cluster assignments
            clusters_series = pd.Series(
                clusters,
                index=numeric_data.index,
                name='cluster'
            )
            
            print(f"Performed hierarchical clustering with {n_clusters} clusters")
            
            # Visualize clusters with reduced dimensions
            reduced_df = self.reduce_dimensions(method='pca', n_components=2)
            merged_df = pd.merge(reduced_df, clusters_series, left_index=True, right_index=True)
            
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                x=merged_df.iloc[:, 0],
                y=merged_df.iloc[:, 1],
                hue=merged_df['cluster'],
                palette='tab10',
                s=50,
                alpha=0.8
            )
            
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title(f'Hierarchical Clustering (k={n_clusters})')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"hierarchical_clusters_{n_clusters}.png"))
            
            # Save cluster assignments
            clusters_series.to_csv(os.path.join(self.output_dir, f"hierarchical_clusters_{n_clusters}.csv"))
            
        else:
            raise ValueError(f"Clustering method '{method}' not supported. Use 'kmeans' or 'hierarchical'.")
        
        return clusters_series
    
    def compute_cross_correlations(self, method='spearman', sig_threshold=0.05):
        """
        Compute correlations between variables across datasets.
        
        Parameters:
        -----------
        method : str
            Correlation method: 'spearman' or 'pearson'
        sig_threshold : float
            Significance threshold for correlations
            
        Returns:
        --------
        correlation_results : pandas.DataFrame
            Significant correlations between datasets
        """
        # Get list of available datasets
        datasets = []
        dataset_names = []
        if self.neural_data is not None:
            datasets.append(self.neural_data.select_dtypes(include=[np.number]))
            dataset_names.append('neural')
        if self.microbiome_data is not None:
            datasets.append(self.microbiome_data.select_dtypes(include=[np.number]))
            dataset_names.append('microbiome')
        if self.connectivity_data is not None:
            datasets.append(self.connectivity_data.select_dtypes(include=[np.number]))
            dataset_names.append('connectivity')
        
        if len(datasets) < 2:
            raise ValueError("At least two datasets are required for cross-correlation analysis.")
        
        # Ensure all datasets have the same samples and are sorted
        common_samples = set.intersection(*[set(ds.index) for ds in datasets])
        if not common_samples:
            raise ValueError("No common samples found across datasets.")
        
        common_samples = sorted(list(common_samples))
        datasets = [ds.loc[common_samples] for ds in datasets]
        
        # Initialize results list
        correlation_results = []
        
        # Compute correlations between all pairs of datasets
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                print(f"Computing correlations between {dataset_names[i]} and {dataset_names[j]}...")
                
                # Get feature names
                features_i = datasets[i].columns
                features_j = datasets[j].columns
                
                # Compute correlations for each pair of features
                for feat_i in features_i:
                    for feat_j in features_j:
                        # Get data
                        x = datasets[i][feat_i].values
                        y = datasets[j][feat_j].values
                        
                        # Compute correlation
                        if method == 'spearman':
                            corr, p_value = spearmanr(x, y)
                        elif method == 'pearson':
                            corr, p_value = pearsonr(x, y)
                        else:
                            raise ValueError(f"Correlation method '{method}' not supported. Use 'spearman' or 'pearson'.")
                        
                        # Add to results if significant
                        if p_value <= sig_threshold:
                            correlation_results.append({
                                'dataset1': dataset_names[i],
                                'feature1': feat_i,
                                'dataset2': dataset_names[j],
                                'feature2': feat_j,
                                'correlation': corr,
                                'p_value': p_value,
                                'method': method
                            })
        
        if not correlation_results:
            print("No significant correlations found.")
            return pd.DataFrame()
        
        # Create DataFrame from results
        results_df = pd.DataFrame(correlation_results)
        
        # Sort by absolute correlation
        results_df['abs_correlation'] = np.abs(results_df['correlation'])
        results_df.sort_values('abs_correlation', ascending=False, inplace=True)
        results_df.drop('abs_correlation', axis=1, inplace=True)
        
        # Save results
        results_df.to_csv(os.path.join(self.output_dir, f"cross_correlations_{method}.csv"), index=False)
        
        # Plot top correlations
        top_corr = results_df.head(20)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(
            [f"{row['feature1']} ({row['dataset1']}) vs\n{row['feature2']} ({row['dataset2']})" 
             for _, row in top_corr.iterrows()],
            top_corr['correlation'],
            color=[plt.cm.coolwarm(0.8 * (0.5 + corr/2)) for corr in top_corr['correlation']]
        )
        
        plt.axvline(0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel(f'{method.capitalize()} Correlation')
        plt.title(f'Top {len(top_corr)} Cross-Dataset Correlations')
        plt.grid(True, axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"top_cross_correlations_{method}.png"))
        
        # Create a heatmap for selected features
        if len(top_corr) > 0:
            # Select top features from each dataset
            top_features = {}
            for dataset in dataset_names:
                # Get top features for this dataset
                dataset_features = set(
                    list(top_corr[top_corr['dataset1'] == dataset]['feature1']) + 
                    list(top_corr[top_corr['dataset2'] == dataset]['feature2'])
                )
                if dataset_features:
                    top_features[dataset] = list(dataset_features)[:5]  # Limit to top 5
            
            # Create merged dataset with top features only
            selected_data = {}
            for i, dataset in enumerate(dataset_names):
                if dataset in top_features and top_features[dataset]:
                    selected_data[dataset] = datasets[i][top_features[dataset]]
            
            if len(selected_data) >= 2:
                # Merge selected features
                merged_top = pd.concat([df for df in selected_data.values()], axis=1)
                
                # Compute correlation matrix
                corr_matrix = merged_top.corr(method=method)
                
                # Create heatmap
                plt.figure(figsize=(12, 10))
                mask = np.zeros_like(corr_matrix, dtype=bool)
                np.fill_diagonal(mask, True)  # Mask diagonal
                
                sns.heatmap(
                    corr_matrix,
                    mask=mask,
                    cmap='coolwarm',
                    center=0,
                    annot=True,
                    fmt='.2f',
                    linewidths=0.5,
                    square=True
                )
                
                # Add colored rectangles to identify datasets
                ax = plt.gca()
                
                # Define colors for datasets
                colors = plt.cm.tab10.colors
                
                # Create patches for legend
                patches = []
                
                # Track current position
                pos = 0
                for i, (dataset, features) in enumerate(selected_data.items()):
                    n_features = len(features)
                    
                    # Draw rectangle around the dataset's features
                    ax.add_patch(plt.Rectangle(
                        (pos, pos),
                        n_features, n_features,
                        fill=False, 
                        edgecolor=colors[i % len(colors)],
                        lw=2
                    ))
                    
                    # Add to legend
                    patches.append(mpatches.Patch(
                        color=colors[i % len(colors)],
                        label=dataset
                    ))
                    
                    pos += n_features
                
                plt.legend(
                    handles=patches,
                    loc='upper right',
                    title='Datasets'
                )
                
                plt.title('Cross-Dataset Correlation Heatmap')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"cross_correlation_heatmap_{method}.png"))
        
        return results_df
    
    def analyze_group_differences(self, group_col, sig_threshold=0.05):
        """
        Analyze differences in integrated features between groups.
        
        Parameters:
        -----------
        group_col : str
            Column in metadata for grouping samples
        sig_threshold : float
            Significance threshold for statistical tests
            
        Returns:
        --------
        results : pandas.DataFrame
            Results of group difference analysis
        """
        if self.integrated_data is None:
            raise ValueError("Integrated data not available. Run integrate_data first.")
        
        if self.metadata is None:
            raise ValueError("Metadata not available. Group difference analysis requires metadata.")
        
        if group_col not in self.metadata.columns:
            raise ValueError(f"Group column '{group_col}' not found in metadata.")
        
        # Get group values
        groups = self.metadata[group_col].dropna().unique()
        
        if len(groups) < 2:
            raise ValueError(f"Need at least 2 groups for analysis, found {len(groups)}.")
        
        # Merge integrated data with group information
        merged_data = pd.merge(
            self.integrated_data,
            self.metadata[[group_col]],
            left_index=True,
            right_index=True
        )
        
        # Get numeric features
        numeric_features = self.integrated_data.select_dtypes(include=[np.number]).columns
        
        # Initialize results list
        results = []
        
        # Perform statistical tests for each feature
        for feature in numeric_features:
            # First check if data is normally distributed
            from scipy.stats import shapiro
            try:
                _, shapiro_p = shapiro(merged_data[feature])
                is_normal = shapiro_p > 0.05
            except:
                # If shapiro test fails, assume non-normal
                is_normal = False
            
            # Choose appropriate test
            if len(groups) == 2:
                # Two groups: t-test or Mann-Whitney U
                group1, group2 = groups
                data1 = merged_data[merged_data[group_col] == group1][feature]
                data2 = merged_data[merged_data[group_col] == group2][feature]
                
                if is_normal:
                    # Use t-test
                    from scipy.stats import ttest_ind
                    stat, p_value = ttest_ind(data1, data2)
                    test_name = 't-test'
                else:
                    # Use Mann-Whitney U
                    from scipy.stats import mannwhitneyu
                    stat, p_value = mannwhitneyu(data1, data2)
                    test_name = 'Mann-Whitney U'
                
                # Add to results if significant
                if p_value <= sig_threshold:
                    results.append({
                        'feature': feature,
                        'test': test_name,
                        'p_value': p_value,
                        'statistic': stat,
                        f'mean_{group1}': data1.mean(),
                        f'mean_{group2}': data2.mean(),
                        f'std_{group1}': data1.std(),
                        f'std_{group2}': data2.std()
                    })
            else:
                # Multiple groups: ANOVA or Kruskal-Wallis
                if is_normal:
                    # Use ANOVA
                    from scipy.stats import f_oneway
                    
                    # Prepare data for ANOVA
                    group_data = [merged_data[merged_data[group_col] == group][feature] for group in groups]
                    
                    # Run ANOVA
                    stat, p_value = f_oneway(*group_data)
                    test_name = 'ANOVA'
                else:
                    # Use Kruskal-Wallis
                    from scipy.stats import kruskal
                    
                    # Prepare data for Kruskal-Wallis
                    group_data = [merged_data[merged_data[group_col] == group][feature] for group in groups]
                    
                    # Run Kruskal-Wallis
                    stat, p_value = kruskal(*group_data)
                    test_name = 'Kruskal-Wallis'
                
                # Add to results if significant
                if p_value <= sig_threshold:
                    result_dict = {
                        'feature': feature,
                        'test': test_name,
                        'p_value': p_value,
                        'statistic': stat
                    }
                    
                    # Add group means and stds
                    for group in groups:
                        data = merged_data[merged_data[group_col] == group][feature]
                        result_dict[f'mean_{group}'] = data.mean()
                        result_dict[f'std_{group}'] = data.std()
                    
                    results.append(result_dict)
        
        if not results:
            print("No significant differences found between groups.")
            return pd.DataFrame()
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        
        # Apply multiple testing correction
        from statsmodels.stats.multitest import multipletests
        results_df['adjusted_p_value'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        
        # Sort by adjusted p-value
        results_df.sort_values('adjusted_p_value', inplace=True)
        
        # Save results
        results_df.to_csv(os.path.join(self.output_dir, f"group_differences_{group_col}.csv"), index=False)
        
        # Plot top different features
        # Limit to top 15 features
        top_diff = results_df.head(15)
        
        if len(top_diff) > 0:
            for _, row in top_diff.iterrows():
                feature = row['feature']
                
                plt.figure(figsize=(10, 6))
                sns.boxplot(
                    x=group_col,
                    y=feature,
                    data=merged_data
                )
                
                plt.title(f"{feature} by {group_col}\n{row['test']}, p={row['adjusted_p_value']:.3f}")
                plt.ylabel(feature)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"diff_{group_col}_{feature.replace('/', '_')}.png"))
        
        return results_df


def run_integrated_analysis(neural_data=None, microbiome_data=None, connectivity_data=None, 
                          metadata=None, output_dir="results/integrated_analysis"):
    """
    Run the complete integrated data analysis workflow.
    
    Parameters:
    -----------
    neural_data : pandas.DataFrame, optional
        Neural signal features with samples as rows
    microbiome_data : pandas.DataFrame, optional
        Microbiome features with samples as rows
    connectivity_data : pandas.DataFrame, optional
        Connectivity features with samples as rows
    metadata : pandas.DataFrame, optional
        Sample metadata with samples as rows
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    processor : IntegratedDataProcessor
        The integrated data processor object
    """
    # Initialize integrated data processor
    processor = IntegratedDataProcessor(output_dir=output_dir)
    
    # Load data
    common_samples = processor.load_data(
        neural_data=neural_data, 
        microbiome_data=microbiome_data, 
        connectivity_data=connectivity_data, 
        metadata=metadata
    )
    
    if not common_samples:
        raise ValueError("No common samples found across datasets.")
    
    # Preprocess data
    processor.preprocess_data(scale=True, impute_missing=True, common_samples=common_samples)
    
    # Integrate data
    processor.integrate_data(method='pca', n_components=10)
    
    # Reduce dimensions for visualization
    processor.reduce_dimensions(method='pca', n_components=2)
    processor.reduce_dimensions(method='tsne', n_components=2)
    
    # Perform clustering
    processor.perform_clustering(n_clusters=None, method='kmeans')
    
    # Compute cross-correlations
    processor.compute_cross_correlations(method='spearman')
    
    # If metadata is available, analyze group differences
    if metadata is not None:
        # Look for categorical columns with a reasonable number of categories
        for col in metadata.columns:
            if metadata[col].dtype == 'object' or metadata[col].dtype.name == 'category':
                n_categories = metadata[col].nunique()
                if 2 <= n_categories <= 10:  # Only analyze if 2-10 categories
                    try:
                        processor.analyze_group_differences(group_col=col)
                    except Exception as e:
                        print(f"Error in group difference analysis for {col}: {str(e)}")
    
    print("Integrated analysis completed successfully!")
    print(f"Results saved to {output_dir}")
    
    return processor


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("../..")  # Add project root to path
    
    from data.loaders import (
        load_neural_features, 
        load_microbiome_features, 
        load_connectivity_features, 
        load_metadata
    )
    
    # Load example data
    neural_data = load_neural_features("data/processed/neural_features.csv")
    microbiome_data = load_microbiome_features("data/processed/microbiome_features.csv")
    connectivity_data = load_connectivity_features("data/processed/connectivity_features.csv")
    metadata = load_metadata("data/raw/sample_metadata.csv")
    
    # Run integrated analysis
    processor = run_integrated_analysis(
        neural_data=neural_data,
        microbiome_data=microbiome_data,
        connectivity_data=connectivity_data,
        metadata=metadata,
        output_dir="results/integrated_analysis"
    )
