#!/usr/bin/env python3
"""
Brain-Ear Connectivity Analysis Module

This module implements methods for analyzing connectivity between brain regions
and ear-related structures using functional and structural connectivity measures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from nilearn import connectome
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score

class BrainEarConnectivity:
    """
    Class for analyzing and visualizing connectivity between brain regions and ear-related structures.
    """
    
    def __init__(self, output_dir="results/connectivity"):
        """
        Initialize the connectivity analysis class.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save connectivity results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.connectivity_matrix = None
        self.region_names = None
        self.graph = None
        
    def compute_functional_connectivity(self, time_series_data, method='correlation', 
                                       region_names=None, threshold=0.3):
        """
        Compute functional connectivity matrix from time series data.
        
        Parameters:
        -----------
        time_series_data : numpy.ndarray or pandas.DataFrame
            Time series data with shape (n_samples, n_regions)
        method : str
            Connectivity metric: 'correlation', 'partial_correlation', 'tangent', or 'mutual_info'
        region_names : list, optional
            Names of brain and ear regions
        threshold : float
            Threshold for connectivity values (absolute value)
            
        Returns:
        --------
        connectivity_matrix : numpy.ndarray
            Functional connectivity matrix
        """
        if isinstance(time_series_data, pd.DataFrame):
            if region_names is None:
                region_names = time_series_data.columns.tolist()
            time_series_data = time_series_data.values
        
        self.region_names = region_names
        
        if method in ['correlation', 'partial_correlation', 'tangent']:
            # Using nilearn's ConnectivityMeasure
            conn_measure = connectome.ConnectivityMeasure(kind=method)
            connectivity_matrix = conn_measure.fit_transform([time_series_data])[0]
        
        elif method == 'mutual_info':
            # Manual computation of mutual information
            n_regions = time_series_data.shape[1]
            connectivity_matrix = np.zeros((n_regions, n_regions))
            
            for i in range(n_regions):
                for j in range(n_regions):
                    if i != j:
                        # Discrete bins for mutual information
                        x_bins = pd.qcut(time_series_data[:, i], 10, labels=False, duplicates='drop')
                        y_bins = pd.qcut(time_series_data[:, j], 10, labels=False, duplicates='drop')
                        mi = mutual_info_score(x_bins, y_bins)
                        connectivity_matrix[i, j] = mi
        else:
            raise ValueError(f"Method {method} not supported. Use 'correlation', 'partial_correlation', 'tangent', or 'mutual_info'")
        
        # Apply threshold
        if threshold is not None:
            if method in ['correlation', 'partial_correlation', 'tangent']:
                connectivity_matrix[np.abs(connectivity_matrix) < threshold] = 0
            else:
                # For methods like mutual info where values are positive
                connectivity_matrix[connectivity_matrix < threshold] = 0
        
        self.connectivity_matrix = connectivity_matrix
        
        # Convert to graph
        self._create_graph()
        
        return connectivity_matrix
    
    def compute_structural_connectivity(self, structural_data, method='weighted', 
                                       region_names=None, threshold=0.3):
        """
        Compute structural connectivity matrix from tract data.
        
        Parameters:
        -----------
        structural_data : numpy.ndarray or pandas.DataFrame
            Structural connectivity data with shape (n_regions, n_regions)
        method : str
            Method for processing: 'weighted', 'binary', or 'normalized'
        region_names : list, optional
            Names of brain and ear regions
        threshold : float
            Threshold for connectivity values
            
        Returns:
        --------
        connectivity_matrix : numpy.ndarray
            Structural connectivity matrix
        """
        if isinstance(structural_data, pd.DataFrame):
            if region_names is None:
                region_names = structural_data.columns.tolist()
            structural_data = structural_data.values
        
        self.region_names = region_names
        
        # Process according to the specified method
        if method == 'binary':
            connectivity_matrix = (structural_data > threshold).astype(float)
        
        elif method == 'normalized':
            # Normalize by row maximum
            connectivity_matrix = np.zeros_like(structural_data)
            for i in range(structural_data.shape[0]):
                row_max = np.max(structural_data[i, :])
                if row_max > 0:
                    connectivity_matrix[i, :] = structural_data[i, :] / row_max
            
            # Apply threshold after normalization
            connectivity_matrix[connectivity_matrix < threshold] = 0
        
        elif method == 'weighted':
            connectivity_matrix = structural_data.copy()
            connectivity_matrix[connectivity_matrix < threshold] = 0
        
        else:
            raise ValueError(f"Method {method} not supported. Use 'weighted', 'binary', or 'normalized'")
        
        self.connectivity_matrix = connectivity_matrix
        
        # Convert to graph
        self._create_graph()
        
        return connectivity_matrix
    
    def _create_graph(self):
        """
        Create a NetworkX graph from the connectivity matrix.
        
        Returns:
        --------
        G : networkx.Graph
            NetworkX graph representation of the connectivity
        """
        if self.connectivity_matrix is None:
            raise ValueError("Connectivity matrix not computed. Run compute_functional_connectivity or compute_structural_connectivity first.")
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes with region names
        n_regions = self.connectivity_matrix.shape[0]
        
        if self.region_names is None:
            self.region_names = [f"Region_{i}" for i in range(n_regions)]
        
        # Add nodes
        G.add_nodes_from([(i, {"name": self.region_names[i]}) for i in range(n_regions)])
        
        # Add edges with weights
        for i in range(n_regions):
            for j in range(i+1, n_regions):  # Only upper triangle to avoid duplicates
                if self.connectivity_matrix[i, j] != 0:
                    G.add_edge(i, j, weight=self.connectivity_matrix[i, j])
        
        self.graph = G
        return G
    
    def visualize_connectivity_matrix(self, title=None, cmap='coolwarm', ear_indices=None):
        """
        Visualize the connectivity matrix as a heatmap.
        
        Parameters:
        -----------
        title : str, optional
            Title for the plot
        cmap : str
            Colormap for the heatmap
        ear_indices : list, optional
            Indices of ear-related regions to highlight
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The heatmap figure
        """
        if self.connectivity_matrix is None:
            raise ValueError("Connectivity matrix not computed. Run compute_functional_connectivity or compute_structural_connectivity first.")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a mask for the upper triangle
        mask = np.zeros_like(self.connectivity_matrix, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        
        # Plot the heatmap
        if cmap == 'coolwarm':
            # For correlation-like values centered at 0
            vmax = np.max(np.abs(self.connectivity_matrix))
            vmin = -vmax
        else:
            # For positive-only values
            vmin = 0
            vmax = np.max(self.connectivity_matrix)
        
        sns.heatmap(
            self.connectivity_matrix,
            mask=mask,
            cmap=cmap,
            square=True,
            linewidths=.5,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            xticklabels=self.region_names,
            yticklabels=self.region_names,
            ax=ax
        )
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Connectivity Matrix")
        
        # Highlight ear-related regions if specified
        if ear_indices is not None:
            for idx in ear_indices:
                ax.get_xticklabels()[idx].set_color("red")
                ax.get_yticklabels()[idx].set_color("red")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "connectivity_matrix.png"))
        
        return fig
    
    def visualize_connectivity_graph(self, layout='spring', node_size_factor=300, 
                                    edge_width_factor=2, ear_color='red', 
                                    brain_color='blue', ear_indices=None):
        """
        Visualize the connectivity as a graph.
        
        Parameters:
        -----------
        layout : str
            Layout algorithm: 'spring', 'circular', 'kamada_kawai', 'spectral'
        node_size_factor : float
            Factor to scale node sizes
        edge_width_factor : float
            Factor to scale edge widths
        ear_color : str
            Color for ear-related nodes
        brain_color : str
            Color for brain-related nodes
        ear_indices : list, optional
            Indices of ear-related regions
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The graph visualization figure
        """
        if self.graph is None:
            raise ValueError("Graph not created. Run compute_functional_connectivity or compute_structural_connectivity first.")
        
        G = self.graph
        
        # Compute node degrees for node sizes
        node_degrees = dict(G.degree())
        
        # Normalize node sizes by degree
        node_sizes = [node_size_factor * (1 + deg) for node, deg in node_degrees.items()]
        
        # Determine node colors
        if ear_indices is not None:
            node_colors = [ear_color if i in ear_indices else brain_color for i in range(G.number_of_nodes())]
        else:
            node_colors = [brain_color] * G.number_of_nodes()
        
        # Determine edge widths by weight
        edge_weights = [edge_width_factor * G[u][v]['weight'] for u, v in G.edges()]
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            raise ValueError(f"Layout {layout} not supported. Use 'spring', 'circular', 'kamada_kawai', or 'spectral'")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray', ax=ax)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, labels={i: region for i, region in enumerate(self.region_names)}, 
                               font_size=8, font_color='black', ax=ax)
        
        ax.set_title("Brain-Ear Connectivity Graph")
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"connectivity_graph_{layout}.png"))
        
        return fig
    
    def compute_centrality_measures(self):
        """
        Compute centrality measures for all nodes in the graph.
        
        Returns:
        --------
        centrality_df : pandas.DataFrame
            DataFrame with centrality measures for each region
        """
        if self.graph is None:
            raise ValueError("Graph not created. Run compute_functional_connectivity or compute_structural_connectivity first.")
        
        G = self.graph
        
        # Compute various centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        
        # Create a DataFrame
        centrality_df = pd.DataFrame({
            'region': self.region_names,
            'degree_centrality': [degree_centrality[i] for i in range(len(self.region_names))],
            'betweenness_centrality': [betweenness_centrality[i] for i in range(len(self.region_names))],
            'closeness_centrality': [closeness_centrality[i] for i in range(len(self.region_names))],
            'eigenvector_centrality': [eigenvector_centrality[i] for i in range(len(self.region_names))]
        })
        
        # Save to CSV
        centrality_df.to_csv(os.path.join(self.output_dir, "centrality_measures.csv"), index=False)
        
        return centrality_df
    
    def visualize_centrality(self, centrality_type='degree_centrality', ear_indices=None):
        """
        Visualize centrality measures for all regions.
        
        Parameters:
        -----------
        centrality_type : str
            Type of centrality: 'degree_centrality', 'betweenness_centrality', 
                               'closeness_centrality', or 'eigenvector_centrality'
        ear_indices : list, optional
            Indices of ear-related regions to highlight
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The centrality visualization figure
        """
        # Compute centrality if not already done
        try:
            centrality_df = self.compute_centrality_measures()
        except FileNotFoundError:
            # If centrality file already exists, load it
            centrality_df = pd.read_csv(os.path.join(self.output_dir, "centrality_measures.csv"))
        
        # Sort by the specified centrality
        sorted_df = centrality_df.sort_values(centrality_type, ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Determine colors
        if ear_indices is not None:
            colors = ['red' if i in ear_indices else 'blue' for i in sorted_df.index]
        else:
            colors = ['blue'] * len(sorted_df)
        
        # Create bar plot
        sns.barplot(
            x='region', 
            y=centrality_type, 
            data=sorted_df,
            palette=colors,
            ax=ax
        )
        
        ax.set_title(f"{centrality_type.replace('_', ' ').title()}")
        ax.set_xlabel("Region")
        ax.set_ylabel(centrality_type.replace('_', ' ').title())
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{centrality_type}.png"))
        
        return fig
    
    def compute_community_structure(self, resolution=1.0):
        """
        Detect communities in the connectivity graph.
        
        Parameters:
        -----------
        resolution : float
            Resolution parameter for community detection
            
        Returns:
        --------
        communities : dict
            Dictionary mapping nodes to community IDs
        """
        if self.graph is None:
            raise ValueError("Graph not created. Run compute_functional_connectivity or compute_structural_connectivity first.")
        
        # Use Louvain method for community detection
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(self.graph, resolution=resolution)
        except ImportError:
            print("community package not found. Using NetworkX's native community detection.")
            # Alternative using NetworkX
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(self.graph)
            # Convert to dictionary format
            community_dict = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_dict[node] = i
            communities = community_dict
        
        # Create a dataframe
        community_df = pd.DataFrame({
            'region': self.region_names,
            'community': [communities[i] for i in range(len(self.region_names))]
        })
        
        # Save to CSV
        community_df.to_csv(os.path.join(self.output_dir, "community_structure.csv"), index=False)
        
        return communities
    
    def visualize_communities(self, ear_indices=None):
        """
        Visualize community structure in the graph.
        
        Parameters:
        -----------
        ear_indices : list, optional
            Indices of ear-related regions to highlight with markers
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The community visualization figure
        """
        # Detect communities
        communities = self.compute_community_structure()
        
        G = self.graph
        
        # Get positions
        pos = nx.spring_layout(G, seed=42)
        
        # Create a colormap
        unique_communities = set(communities.values())
        color_map = plt.cm.get_cmap('tab10', len(unique_communities))
        
        # Assign colors to communities
        node_colors = [color_map(communities[i]) for i in range(G.number_of_nodes())]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color=node_colors, 
            node_size=300, 
            alpha=0.8, 
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, 
            width=[G[u][v]['weight'] * 2 for u, v in G.edges()], 
            alpha=0.5, 
            edge_color='gray', 
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos, 
            labels={i: region for i, region in enumerate(self.region_names)}, 
            font_size=8, 
            font_color='black', 
            ax=ax
        )
        
        # Highlight ear-related nodes if specified
        if ear_indices is not None:
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=ear_indices, 
                node_color='white', 
                node_shape='*', 
                node_size=500, 
                alpha=1.0, 
                linewidths=2, 
                edgecolors='black', 
                ax=ax
            )
        
        ax.set_title("Brain-Ear Connectivity Communities")
        ax.axis('off')
        
        # Add a legend for communities
        for comm_id in unique_communities:
            ax.scatter([], [], color=color_map(comm_id), label=f'Community {comm_id}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "community_structure.png"))
        
        return fig
    
    def analyze_ear_brain_connectivity(self, ear_indices, statistical_test='permutation'):
        """
        Analyze connectivity between ear and brain regions.
        
        Parameters:
        -----------
        ear_indices : list
            Indices of ear-related regions
        statistical_test : str
            Type of statistical test: 'permutation' or 't_test'
            
        Returns:
        --------
        results_df : pandas.DataFrame
            DataFrame with connectivity analysis results
        """
        if self.connectivity_matrix is None:
            raise ValueError("Connectivity matrix not computed. Run compute_functional_connectivity or compute_structural_connectivity first.")
        
        # Identify brain indices (all indices not in ear_indices)
        brain_indices = [i for i in range(self.connectivity_matrix.shape[0]) if i not in ear_indices]
        
        # Extract connectivity values between ear and brain regions
        ear_brain_connectivity = []
        for ear_idx in ear_indices:
            for brain_idx in brain_indices:
                ear_brain_connectivity.append({
                    'ear_region': self.region_names[ear_idx],
                    'brain_region': self.region_names[brain_idx],
                    'connectivity': self.connectivity_matrix[ear_idx, brain_idx]
                })
        
        # Create a dataframe
        connectivity_df = pd.DataFrame(ear_brain_connectivity)
        
        # Compute statistics
        if statistical_test == 'permutation':
            # Permutation test for significance
            n_permutations = 1000
            observed_mean = np.mean(connectivity_df['connectivity'])
            
            # Permutation distribution
            perm_means = []
            for _ in range(n_permutations):
                # Randomly permute connectivity values
                perm_connectivity = np.random.permutation(self.connectivity_matrix.flatten())
                perm_connectivity = perm_connectivity.reshape(self.connectivity_matrix.shape)
                
                # Extract ear-brain connectivity from permuted matrix
                perm_ear_brain = []
                for ear_idx in ear_indices:
                    for brain_idx in brain_indices:
                        perm_ear_brain.append(perm_connectivity[ear_idx, brain_idx])
                
                perm_means.append(np.mean(perm_ear_brain))
            
            # Compute p-value
            p_value = np.mean(np.array(perm_means) >= observed_mean)
            
        elif statistical_test == 't_test':
            # t-test against the null hypothesis of zero connectivity
            t_stat, p_value = stats.ttest_1samp(connectivity_df['connectivity'], 0)
        
        else:
            raise ValueError(f"Statistical test {statistical_test} not supported. Use 'permutation' or 't_test'")
        
        # Add significance information
        connectivity_df['significant'] = connectivity_df['connectivity'] > np.percentile(connectivity_df['connectivity'], 75)
        
        # Save to CSV
        connectivity_df.to_csv(os.path.join(self.output_dir, "ear_brain_connectivity.csv"), index=False)
        
        # Create summary statistics
        summary_stats = {
            'mean_connectivity': observed_mean,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'test_type': statistical_test
        }
        
        # Save summary statistics
        with open(os.path.join(self.output_dir, "connectivity_summary.txt"), 'w') as f:
            for key, value in summary_stats.items():
                f.write(f"{key}: {value}\n")
        
        # Visualize ear-brain connectivity
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            self.connectivity_matrix[ear_indices][:, brain_indices],
            cmap='coolwarm',
            annot=True,
            fmt=".2f",
            xticklabels=[self.region_names[i] for i in brain_indices],
            yticklabels=[self.region_names[i] for i in ear_indices]
        )
        plt.title("Ear-Brain Connectivity")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ear_brain_connectivity.png"))
        
        return connectivity_df, summary_stats


def run_connectivity_analysis(time_series_data, region_names, ear_region_names, 
                             method='correlation', output_dir="results/connectivity"):
    """
    Run the complete connectivity analysis workflow.
    
    Parameters:
    -----------
    time_series_data : numpy.ndarray or pandas.DataFrame
        Time series data with shape (n_samples, n_regions)
    region_names : list
        Names of all regions
    ear_region_names : list
        Names of ear-related regions
    method : str
        Connectivity metric: 'correlation', 'partial_correlation', 'tangent', or 'mutual_info'
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    connectivity : BrainEarConnectivity
        The connectivity analysis object
    """
    # Initialize connectivity analysis
    connectivity = BrainEarConnectivity(output_dir=output_dir)
    
    # Compute connectivity
    connectivity.compute_functional_connectivity(
        time_series_data,
        method=method,
        region_names=region_names
    )
    
    # Find indices of ear regions
    ear_indices = [region_names.index(ear_region) for ear_region in ear_region_names]
    
    # Visualize connectivity matrix
    connectivity.visualize_connectivity_matrix(
        title=f"Brain-Ear {method.replace('_', ' ').title()} Connectivity",
        ear_indices=ear_indices
    )
    
    # Visualize connectivity graph
    connectivity.visualize_connectivity_graph(ear_indices=ear_indices)
    
    # Compute and visualize centrality
    for centrality_type in ['degree_centrality', 'betweenness_centrality', 
                          'closeness_centrality', 'eigenvector_centrality']:
        connectivity.visualize_centrality(centrality_type, ear_indices)
    
    # Visualize community structure
    connectivity.visualize_communities(ear_indices)
    
    # Analyze ear-brain connectivity
    connectivity_df, summary_stats = connectivity.analyze_ear_brain_connectivity(ear_indices)
    
    print("Connectivity analysis completed successfully!")
    print(f"Results saved to {output_dir}")
    
    return connectivity


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("../..")  # Add project root to path
    
    from data.loaders import load_functional_data
    
    # Load example data
    data = load_functional_data("data/raw/time_series.csv")
    
    # Example region names
    region_names = [
        "Auditory_Cortex_L", "Auditory_Cortex_R",
        "Cochlear_Nucleus_L", "Cochlear_Nucleus_R", 
        "Superior_Olivary_Complex_L", "Superior_Olivary_Complex_R",
        "Inferior_Colliculus_L", "Inferior_Colliculus_R",
        "Medial_Geniculate_Body_L", "Medial_Geniculate_Body_R",
        "Prefrontal_Cortex_L", "Prefrontal_Cortex_R",
        "Hippocampus_L", "Hippocampus_R",
        "Amygdala_L", "Amygdala_R",
        "Cochlea_L", "Cochlea_R",
        "Vestibular_System_L", "Vestibular_System_R"
    ]
    
    # Ear-related regions
    ear_region_names = [
        "Cochlea_L", "Cochlea_R",
        "Vestibular_System_L", "Vestibular_System_R",
        "Cochlear_Nucleus_L", "Cochlear_Nucleus_R"
    ]
    
    # Run analysis
    connectivity = run_connectivity_analysis(
        data,
        region_names=region_names,
        ear_region_names=ear_region_names,
        method='correlation',
        output_dir="results/brain_ear_connectivity"
    )
