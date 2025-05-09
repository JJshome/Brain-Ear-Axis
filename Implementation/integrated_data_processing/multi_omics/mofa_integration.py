#!/usr/bin/env python3
"""
Multi-Omics Integration Analysis for Brain-Ear Axis

This module implements Multi-Omics Factor Analysis (MOFA) for integrating
multiple omics data types (e.g., microbiome, transcriptomics, proteomics) 
to identify joint factors that explain variations across datasets.

References:
- Argelaguet, R. et al. (2018). Multi-Omics Factor Analysis—a framework for 
  unsupervised integration of multi-omics data sets. Molecular Systems Biology, 14(6), e8124.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import mofapy2
from mofapy2.run.entry_point import entry_point

class MOFAIntegration:
    """
    Multi-Omics Factor Analysis implementation for integrating multiple omics datasets.
    """
    
    def __init__(self, output_dir="results/mofa_integration"):
        """
        Initialize the MOFA integration class.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save MOFA results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model = None
        self.data = {}
        self.factors = None
        self.weights = {}
        
    def prepare_data(self, omics_data_dict):
        """
        Prepare multiple omics datasets for MOFA analysis.
        
        Parameters:
        -----------
        omics_data_dict : dict
            Dictionary with keys as omics types and values as pandas DataFrames
            Example: {'microbiome': microbiome_df, 'transcriptome': transcriptome_df}
        """
        # Standardize each omics dataset
        self.data = {}
        for omics_type, df in omics_data_dict.items():
            # Ensure data is numeric
            numeric_df = df.select_dtypes(include=[np.number])
            
            # Handle missing values - MOFA can handle missingness, but we'll impute simple cases
            if numeric_df.isna().sum().sum() > 0:
                # Simple imputation - replace with feature means
                numeric_df = numeric_df.fillna(numeric_df.mean())
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            scaled_df = pd.DataFrame(
                scaled_data, 
                index=numeric_df.index, 
                columns=numeric_df.columns
            )
            
            self.data[omics_type] = scaled_df
            
        print(f"Prepared {len(self.data)} omics datasets for MOFA analysis")
        return self.data
    
    def train_model(self, n_factors=10, seed=42, convergence_mode="fast"):
        """
        Train the MOFA model on the prepared data.
        
        Parameters:
        -----------
        n_factors : int
            Number of factors to learn
        seed : int
            Random seed for reproducibility
        convergence_mode : str
            MOFA convergence mode ('fast', 'medium', 'slow')
        
        Returns:
        --------
        model : mofapy2 model
            Trained MOFA model
        """
        # Create MOFA entry point
        ent = entry_point()
        
        # Set data options
        data_options = {
            "scale_views": False,  # We've already scaled the data
        }
        
        # Set model options
        model_options = {
            "num_factors": n_factors,
            "likelihoods": ["gaussian"] * len(self.data),
            "spikeslab_weights": True
        }
        
        # Set training options
        training_options = {
            "maxiter": 1000, 
            "convergence_mode": convergence_mode,
            "seed": seed,
            "dropR2": 0.01,
            "verbose": True
        }
        
        # Prepare data for MOFA
        mofa_data = {}
        for i, (omics_type, df) in enumerate(self.data.items()):
            mofa_data[omics_type] = [df.values]
        
        # Set data
        ent.set_data_options(data_options)
        ent.set_data_matrix(mofa_data)
        
        # Set model and training options
        ent.set_model_options(model_options)
        ent.set_train_options(training_options)
        
        # Train the model
        print("Training MOFA model...")
        ent.build()
        ent.run()
        
        # Get the trained model
        self.model = ent.model
        
        # Extract factors and weights
        self.factors = self.model.getExpectations()["Z"]["E"]
        
        for i, view_name in enumerate(self.data.keys()):
            self.weights[view_name] = self.model.getExpectations()["W"][i]["E"]
        
        # Save model
        self.model.save(os.path.join(self.output_dir, "mofa_model.hdf5"))
        
        print(f"MOFA model trained with {n_factors} factors")
        return self.model
    
    def analyze_factor_variance(self):
        """
        Analyze and visualize the variance explained by MOFA factors.
        
        Returns:
        --------
        r2_total : numpy.ndarray
            Total variance explained by each factor
        r2_per_dataset : pandas.DataFrame
            Variance explained by each factor for each dataset
        """
        if self.model is None:
            raise ValueError("Model not trained. Run train_model first.")
        
        # Get variance explained (R2) per factor
        r2_per_factor = self.model.calculate_variance_explained()
        
        # Total variance explained per factor
        r2_total = r2_per_factor["r2_total"]
        
        # Variance explained per dataset and factor
        r2_per_dataset = pd.DataFrame(r2_per_factor["r2_per_factor"])
        r2_per_dataset.index = list(self.data.keys())
        
        # Plot total variance explained
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(r2_total) + 1), r2_total)
        plt.xlabel('Factor')
        plt.ylabel('Variance explained')
        plt.title('Total variance explained by each MOFA factor')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "total_variance_explained.png"))
        
        # Plot variance explained per dataset
        plt.figure(figsize=(12, 8))
        r2_per_dataset.T.plot(kind='bar', stacked=False)
        plt.xlabel('Factor')
        plt.ylabel('Variance explained')
        plt.title('Variance explained per dataset and factor')
        plt.legend(title='Dataset')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "variance_explained_per_dataset.png"))
        
        return r2_total, r2_per_dataset
    
    def visualize_factor_loadings(self, factor_index=0, n_features=20):
        """
        Visualize the top features contributing to a specific factor.
        
        Parameters:
        -----------
        factor_index : int
            The index of the factor to visualize
        n_features : int
            Number of top features to show
            
        Returns:
        --------
        loadings_dfs : dict
            Dictionary of DataFrames with feature loadings for each omics type
        """
        if self.model is None:
            raise ValueError("Model not trained. Run train_model first.")
        
        loadings_dfs = {}
        
        for view_name, weight_matrix in self.weights.items():
            # Get feature names from the original data
            feature_names = self.data[view_name].columns
            
            # Get the loadings for the selected factor
            factor_loadings = weight_matrix[:, factor_index]
            
            # Create a dataframe with feature names and loadings
            loadings_df = pd.DataFrame({
                'feature': feature_names,
                'loading': factor_loadings
            })
            
            # Sort by absolute loading values
            loadings_df['abs_loading'] = abs(loadings_df['loading'])
            loadings_df = loadings_df.sort_values('abs_loading', ascending=False).head(n_features)
            loadings_df = loadings_df.drop('abs_loading', axis=1)
            
            loadings_dfs[view_name] = loadings_df
            
            # Visualize the top features
            plt.figure(figsize=(10, 8))
            sns.barplot(x='loading', y='feature', data=loadings_df.sort_values('loading'))
            plt.title(f'Top {n_features} features for Factor {factor_index+1} in {view_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"factor{factor_index+1}_{view_name}_loadings.png"))
        
        return loadings_dfs
    
    def plot_factor_scatter(self, factor_x=0, factor_y=1, color_by=None):
        """
        Create scatter plot of samples in the factor space.
        
        Parameters:
        -----------
        factor_x : int
            Factor for x-axis
        factor_y : int
            Factor for y-axis
        color_by : pd.Series or np.array, optional
            Values to color points by (e.g., sample groups or phenotype)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The scatter plot figure
        """
        if self.model is None:
            raise ValueError("Model not trained. Run train_model first.")
        
        # Get sample names from first omics dataset
        sample_names = list(next(iter(self.data.values())).index)
        
        # Create a dataframe with factor values
        factor_df = pd.DataFrame(
            self.factors,
            index=sample_names,
            columns=[f"Factor_{i+1}" for i in range(self.factors.shape[1])]
        )
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if color_by is not None:
            scatter = ax.scatter(
                factor_df.iloc[:, factor_x],
                factor_df.iloc[:, factor_y],
                c=color_by,
                cmap='viridis',
                alpha=0.8
            )
            plt.colorbar(scatter, ax=ax, label='Group')
        else:
            ax.scatter(
                factor_df.iloc[:, factor_x],
                factor_df.iloc[:, factor_y],
                alpha=0.8
            )
        
        ax.set_xlabel(f"Factor {factor_x+1}")
        ax.set_ylabel(f"Factor {factor_y+1}")
        ax.set_title(f"MOFA Factors {factor_x+1} vs {factor_y+1}")
        
        for i, sample in enumerate(sample_names):
            ax.annotate(
                sample,
                (factor_df.iloc[i, factor_x], factor_df.iloc[i, factor_y]),
                fontsize=8
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"factor{factor_x+1}_vs_factor{factor_y+1}_scatter.png"))
        
        return fig
    
    def get_factor_values(self):
        """
        Get the factor values for all samples.
        
        Returns:
        --------
        factor_df : pandas.DataFrame
            DataFrame with factor values for all samples
        """
        if self.model is None:
            raise ValueError("Model not trained. Run train_model first.")
        
        # Get sample names from first omics dataset
        sample_names = list(next(iter(self.data.values())).index)
        
        # Create a dataframe with factor values
        factor_df = pd.DataFrame(
            self.factors,
            index=sample_names,
            columns=[f"Factor_{i+1}" for i in range(self.factors.shape[1])]
        )
        
        return factor_df
    
    def correlate_factors_with_metadata(self, metadata_df):
        """
        Correlate MOFA factors with sample metadata.
        
        Parameters:
        -----------
        metadata_df : pandas.DataFrame
            DataFrame with sample metadata
            
        Returns:
        --------
        correlation_df : pandas.DataFrame
            DataFrame with correlation values
        """
        if self.model is None:
            raise ValueError("Model not trained. Run train_model first.")
        
        # Get factor values
        factor_df = self.get_factor_values()
        
        # Ensure metadata and factor dataframes have matching indices
        common_samples = factor_df.index.intersection(metadata_df.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples found between factors and metadata")
        
        factor_df = factor_df.loc[common_samples]
        metadata_df = metadata_df.loc[common_samples]
        
        # Select only numeric columns from metadata
        numeric_metadata = metadata_df.select_dtypes(include=[np.number])
        
        if numeric_metadata.shape[1] == 0:
            print("No numeric metadata columns found for correlation analysis")
            return None
        
        # Calculate correlations
        correlation_df = pd.DataFrame(
            index=factor_df.columns,
            columns=numeric_metadata.columns
        )
        
        for factor in factor_df.columns:
            for meta_col in numeric_metadata.columns:
                correlation = np.corrcoef(
                    factor_df[factor].values,
                    numeric_metadata[meta_col].values
                )[0, 1]
                correlation_df.loc[factor, meta_col] = correlation
        
        # Visualize correlations
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_df,
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f'
        )
        plt.title('Correlation between MOFA factors and metadata')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "factor_metadata_correlation.png"))
        
        return correlation_df


def run_mofa_integration(omics_data_dict, metadata=None, n_factors=10, output_dir="results/mofa_integration"):
    """
    Run the complete MOFA integration workflow.
    
    Parameters:
    -----------
    omics_data_dict : dict
        Dictionary with keys as omics types and values as pandas DataFrames
    metadata : pandas.DataFrame, optional
        Sample metadata for correlation analysis
    n_factors : int
        Number of factors to learn
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    mofa : MOFAIntegration
        The trained MOFA integration object
    """
    # Initialize MOFA integration
    mofa = MOFAIntegration(output_dir=output_dir)
    
    # Prepare data
    mofa.prepare_data(omics_data_dict)
    
    # Train model
    mofa.train_model(n_factors=n_factors)
    
    # Analyze variance explained
    r2_total, r2_per_dataset = mofa.analyze_factor_variance()
    
    # Visualize top factors
    for i in range(min(3, n_factors)):
        mofa.visualize_factor_loadings(factor_index=i)
    
    # Plot factor scatter
    mofa.plot_factor_scatter(0, 1)
    
    # Correlate with metadata if provided
    if metadata is not None:
        mofa.correlate_factors_with_metadata(metadata)
    
    return mofa


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("../../..")  # Add project root to path
    
    from data.loaders import load_microbiome_data, load_transcriptome_data
    
    # Load example datasets
    microbiome_data = load_microbiome_data("data/raw/microbiome_abundance.csv")
    transcriptome_data = load_transcriptome_data("data/raw/gene_expression.csv")
    
    # Example metadata
    metadata = pd.read_csv("data/raw/sample_metadata.csv", index_col=0)
    
    # Prepare omics dictionary
    omics_dict = {
        "microbiome": microbiome_data,
        "transcriptome": transcriptome_data
    }
    
    # Run MOFA integration
    mofa = run_mofa_integration(
        omics_dict,
        metadata=metadata,
        n_factors=8,
        output_dir="results/mofa_integration"
    )
    
    print("MOFA integration completed successfully!")
