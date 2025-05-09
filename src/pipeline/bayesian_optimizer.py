"""
Bayesian Optimizer for hyperparameter tuning.

This module provides functionality for Bayesian optimization of model hyperparameters,
which is more efficient than grid or random search for expensive models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from bayes_opt import BayesianOptimization
import time
import mlflow
import os


class BayesianHyperparameterOptimizer:
    """
    Perform Bayesian optimization for hyperparameter tuning.
    """
    
    def __init__(self, model_builder, param_space, scoring='accuracy', cv=5, n_iter=25, 
                 random_state=42, verbose=2, problem_type='classification'):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            model_builder: Function that builds a model given hyperparameters,
                           or a scikit-learn pipeline
            param_space (dict): Dictionary of parameter ranges to optimize
            scoring (str): Scoring metric to optimize
            cv (int): Number of cross-validation folds
            n_iter (int): Number of optimization iterations
            random_state (int): Random seed
            verbose (int): Verbosity level
            problem_type (str): 'classification' or 'regression'
        """
        self.model_builder = model_builder
        self.param_space = param_space
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.problem_type = problem_type
        
        self.best_params = None
        self.best_score = None
        self.optimizer = None
        self.results = []
        self.X = None
        self.y = None
        self.use_mlflow = False
    
    def _objective_sklearn(self, **params):
        """
        Objective function for scikit-learn models.
        
        Args:
            **params: Model parameters
            
        Returns:
            float: Cross-validation score
        """
        # Convert params to appropriate types
        parsed_params = self._parse_params(params)
        
        # Build model
        if isinstance(self.model_builder, Pipeline):
            model = clone(self.model_builder)
            # Set params for the last step if pipeline
            step_name = model.steps[-1][0]
            model_params = {f"{step_name}__{k}": v for k, v in parsed_params.items()}
            model.set_params(**model_params)
        else:
            model = self.model_builder(**parsed_params)
        
        # Evaluate
        start_time = time.time()
        scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring=self.scoring)
        end_time = time.time()
        
        mean_score = scores.mean()
        
        # Log results
        self.results.append({
            "params": parsed_params,
            "mean_score": mean_score,
            "std_score": scores.std(),
            "scores": scores.tolist(),
            "time": end_time - start_time
        })
        
        if self.verbose > 0:
            print(f"Score: {mean_score:.4f} | Params: {parsed_params} | Time: {end_time - start_time:.2f}s")
        
        # Log to MLflow if enabled
        if self.use_mlflow:
            with mlflow.start_run(run_name=f"bo_iter_{len(self.results)}", nested=True):
                # Log parameters
                for key, value in parsed_params.items():
                    mlflow.log_param(key, value)
                
                # Log metrics
                mlflow.log_metric(f"mean_{self.scoring}", mean_score)
                mlflow.log_metric(f"std_{self.scoring}", scores.std())
                mlflow.log_metric("training_time", end_time - start_time)
                
                # Log individual scores
                for i, score in enumerate(scores):
                    mlflow.log_metric(f"fold_{i+1}_{self.scoring}", score)
        
        return mean_score
    
    def _objective_keras(self, **params):
        """
        Objective function for Keras models.
        
        Args:
            **params: Model parameters
            
        Returns:
            float: Cross-validation score
        """
        # Convert params to appropriate types
        parsed_params = self._parse_params(params)
        
        # Extract training params
        epochs = parsed_params.pop('epochs', 10)
        batch_size = parsed_params.pop('batch_size', 32)
        
        # Create a Keras wrapper
        model_fn = lambda: self.model_builder(**parsed_params)
        
        if self.problem_type == 'classification':
            keras_wrapper = KerasClassifier(build_fn=model_fn, epochs=epochs, 
                                           batch_size=batch_size, verbose=0)
        else:
            keras_wrapper = KerasRegressor(build_fn=model_fn, epochs=epochs, 
                                          batch_size=batch_size, verbose=0)
        
        # Evaluate
        start_time = time.time()
        scores = cross_val_score(keras_wrapper, self.X, self.y, cv=self.cv, scoring=self.scoring)
        end_time = time.time()
        
        mean_score = scores.mean()
        
        # Log results
        self.results.append({
            "params": {**parsed_params, 'epochs': epochs, 'batch_size': batch_size},
            "mean_score": mean_score,
            "std_score": scores.std(),
            "scores": scores.tolist(),
            "time": end_time - start_time
        })
        
        if self.verbose > 0:
            print(f"Score: {mean_score:.4f} | Params: {parsed_params} | Time: {end_time - start_time:.2f}s")
        
        # Log to MLflow if enabled
        if self.use_mlflow:
            with mlflow.start_run(run_name=f"bo_iter_{len(self.results)}", nested=True):
                # Log parameters
                for key, value in parsed_params.items():
                    mlflow.log_param(key, value)
                mlflow.log_param('epochs', epochs)
                mlflow.log_param('batch_size', batch_size)
                
                # Log metrics
                mlflow.log_metric(f"mean_{self.scoring}", mean_score)
                mlflow.log_metric(f"std_{self.scoring}", scores.std())
                mlflow.log_metric("training_time", end_time - start_time)
                
                # Log individual scores
                for i, score in enumerate(scores):
                    mlflow.log_metric(f"fold_{i+1}_{self.scoring}", score)
        
        return mean_score
    
    def _parse_params(self, params):
        """
        Parse parameters based on their types.
        
        Args:
            params (dict): Raw parameters
            
        Returns:
            dict: Parsed parameters
        """
        parsed = {}
        for key, value in params.items():
            if key in self.param_space and isinstance(self.param_space[key], dict):
                param_type = self.param_space[key].get("type")
                
                if param_type == "int":
                    parsed[key] = int(value)
                elif param_type == "categorical":
                    categories = self.param_space[key].get("categories", [])
                    idx = min(int(value), len(categories) - 1)
                    parsed[key] = categories[idx]
                else:
                    parsed[key] = value
            else:
                parsed[key] = value
        
        return parsed
    
    def optimize(self, X, y, use_mlflow=False, experiment_name=None):
        """
        Run Bayesian optimization.
        
        Args:
            X: Input features
            y: Target variable
            use_mlflow (bool): Whether to log to MLflow
            experiment_name (str): MLflow experiment name
            
        Returns:
            tuple: (best_params, best_score)
        """
        self.X = X
        self.y = y
        self.use_mlflow = use_mlflow
        
        if use_mlflow:
            if experiment_name is None:
                experiment_name = "bayesian_optimization"
            
            try:
                mlflow.create_experiment(experiment_name)
            except:
                pass
            
            mlflow.set_experiment(experiment_name)
        
        # Prepare bounds for BayesianOptimization
        pbounds = {}
        for param, bounds in self.param_space.items():
            if isinstance(bounds, dict):
                if bounds.get("type") == "categorical":
                    categories = bounds.get("categories", [])
                    # Represent categorical variables as indices
                    pbounds[param] = (0, len(categories) - 0.9999)
                else:
                    pbounds[param] = (bounds.get("min", 0), bounds.get("max", 1))
            else:
                pbounds[param] = bounds
        
        # Select the appropriate objective function
        if isinstance(self.model_builder, type) and issubclass(self.model_builder, tf.keras.Model):
            objective = self._objective_keras
        else:
            objective = self._objective_sklearn
        
        # Initialize and run the optimizer
        with mlflow.start_run(run_name="bayesian_optimization") if use_mlflow else self._null_context():
            self.optimizer = BayesianOptimization(
                f=objective,
                pbounds=pbounds,
                verbose=self.verbose,
                random_state=self.random_state
            )
            
            self.optimizer.maximize(
                init_points=5,
                n_iter=self.n_iter
            )
            
            # Process results
            self.best_params = self.optimizer.max['params']
            self.best_score = self.optimizer.max['target']
            
            # Convert parameters to their actual types
            self.best_params = self._parse_params(self.best_params)
            
            if use_mlflow:
                # Log best parameters and score
                for key, value in self.best_params.items():
                    mlflow.log_param(f"best_{key}", value)
                
                mlflow.log_metric(f"best_{self.scoring}", self.best_score)
                
                # Log optimization process visualization
                self._log_optimization_plot()
        
        return self.best_params, self.best_score
    
    def _log_optimization_plot(self):
        """Create and log plots of the optimization process."""
        if len(self.results) < 2:
            return
        
        # Extract data
        iterations = list(range(1, len(self.results) + 1))
        scores = [result["mean_score"] for result in self.results]
        best_scores = np.maximum.accumulate(scores)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot 1: Optimization progress
        plt.subplot(1, 2, 1)
        plt.plot(iterations, scores, 'o-', label='Score')
        plt.plot(iterations, best_scores, 'r--', label='Best score')
        plt.xlabel('Iteration')
        plt.ylabel(self.scoring.capitalize())
        plt.title('Bayesian Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Parameter importance
        plt.subplot(1, 2, 2)
        
        # Get parameter names
        param_names = list(self.best_params.keys())
        
        # Calculate correlation between parameters and scores
        correlations = []
        for param in param_names:
            param_values = []
            for result in self.results:
                # Handle categorical parameters
                param_value = result["params"].get(param)
                if isinstance(param_value, str):
                    # If the parameter is categorical, use the index in categories
                    if param in self.param_space and self.param_space[param].get("type") == "categorical":
                        categories = self.param_space[param].get("categories", [])
                        param_value = categories.index(param_value) if param_value in categories else 0
                
                param_values.append(param_value)
            
            # Calculate correlation if possible
            if len(set(param_values)) > 1:
                corr = np.corrcoef(param_values, [result["mean_score"] for result in self.results])[0, 1]
                correlations.append((param, abs(corr)))
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Plot correlations
        params = [item[0] for item in correlations]
        corr_values = [item[1] for item in correlations]
        
        plt.barh(params, corr_values)
        plt.xlabel('Correlation with Score')
        plt.title('Parameter Importance')
        plt.tight_layout()
        
        # Save plot
        plot_path = "/tmp/optimization_progress.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log plot
        mlflow.log_artifact(plot_path, "plots")
    
    def get_results_dataframe(self):
        """
        Get optimization results as a DataFrame.
        
        Returns:
            pd.DataFrame: Results DataFrame
        """
        # Flatten results for DataFrame
        flat_results = []
        for i, result in enumerate(self.results):
            row = {
                "iteration": i + 1,
                "mean_score": result["mean_score"],
                "std_score": result["std_score"],
                "time": result["time"]
            }
            # Add parameters
            for param, value in result["params"].items():
                row[f"param_{param}"] = value
            
            flat_results.append(row)
        
        return pd.DataFrame(flat_results)
    
    def plot_optimization_progress(self, figsize=(12, 8)):
        """
        Plot optimization progress.
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if len(self.results) < 2:
            raise ValueError("Not enough optimization iterations to plot")
        
        # Extract data
        iterations = list(range(1, len(self.results) + 1))
        scores = [result["mean_score"] for result in self.results]
        best_scores = np.maximum.accumulate(scores)
        times = [result["time"] for result in self.results]
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Optimization progress
        axs[0, 0].plot(iterations, scores, 'o-', label='Score')
        axs[0, 0].plot(iterations, best_scores, 'r--', label='Best score')
        axs[0, 0].set_xlabel('Iteration')
        axs[0, 0].set_ylabel(self.scoring.capitalize())
        axs[0, 0].set_title('Optimization Progress')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Score distribution
        axs[0, 1].hist(scores, bins=10, alpha=0.7)
        axs[0, 1].axvline(self.best_score, color='r', linestyle='--', label=f'Best score: {self.best_score:.4f}')
        axs[0, 1].set_xlabel(self.scoring.capitalize())
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_title('Score Distribution')
        axs[0, 1].legend()
        
        # Plot 3: Evaluation time
        axs[1, 0].plot(iterations, times, 'o-')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Time (seconds)')
        axs[1, 0].set_title('Evaluation Time')
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Parameter importance
        # Get parameter names
        param_names = list(self.best_params.keys())
        
        # Calculate correlation between parameters and scores
        correlations = []
        for param in param_names:
            param_values = []
            for result in self.results:
                # Handle categorical parameters
                param_value = result["params"].get(param)
                if isinstance(param_value, str):
                    # If the parameter is categorical, use the index in categories
                    if param in self.param_space and self.param_space[param].get("type") == "categorical":
                        categories = self.param_space[param].get("categories", [])
                        param_value = categories.index(param_value) if param_value in categories else 0
                
                param_values.append(param_value)
            
            # Calculate correlation if possible
            if len(set(param_values)) > 1:
                corr = np.corrcoef(param_values, [result["mean_score"] for result in self.results])[0, 1]
                correlations.append((param, abs(corr)))
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Plot correlations
        params = [item[0] for item in correlations]
        corr_values = [item[1] for item in correlations]
        
        axs[1, 1].barh(params, corr_values)
        axs[1, 1].set_xlabel('Correlation with Score')
        axs[1, 1].set_title('Parameter Importance')
        
        fig.tight_layout()
        return fig
    
    def save_results(self, path):
        """
        Save optimization results to file.
        
        Args:
            path (str): Path to save results
        """
        results = {
            "param_space": self.param_space,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "iterations": self.results,
            "config": {
                "scoring": self.scoring,
                "cv": self.cv,
                "n_iter": self.n_iter,
                "random_state": self.random_state
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save results
        np.save(path, results)
    
    @classmethod
    def load_results(cls, path):
        """
        Load optimization results from file.
        
        Args:
            path (str): Path to results file
            
        Returns:
            dict: Optimization results
        """
        return np.load(path, allow_pickle=True).item()
    
    def _null_context(self):
        """Context manager that does nothing."""
        class NullContext:
            def __enter__(self):
                return None
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        return NullContext()
