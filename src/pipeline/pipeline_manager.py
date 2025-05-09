"""
Pipeline Manager for reproducible research workflows.

This module provides a framework for creating, executing, and tracking
analysis pipelines in a reproducible manner.
"""

import os
import json
import pickle
import hashlib
import datetime
import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
import mlflow
import mlflow.sklearn
import mlflow.tensorflow

class ReproduciblePipeline:
    """
    A framework for creating and tracking reproducible analysis pipelines.
    
    This class allows for:
    - Creating and executing analysis pipelines
    - Tracking all inputs, parameters, and outputs
    - Saving snapshots of data and models
    - Integration with MLflow for experiment tracking
    """
    
    def __init__(self, name, base_dir="./pipelines", use_mlflow=True, experiment_name=None):
        """
        Initialize the pipeline.
        
        Args:
            name (str): Name of the pipeline
            base_dir (str): Base directory for storing pipeline artifacts
            use_mlflow (bool): Whether to use MLflow for tracking
            experiment_name (str): Name of MLflow experiment (defaults to pipeline name)
        """
        self.name = name
        self.base_dir = base_dir
        self.pipeline = None
        self.config = {}
        self.data_hash = None
        self.pipeline_hash = None
        self.results = {}
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name or name
        self.run_id = None
        
        # Set up logging
        self.logger = logging.getLogger(f"pipeline.{name}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Create directory structure
        self.pipeline_dir = os.path.join(base_dir, name)
        os.makedirs(self.pipeline_dir, exist_ok=True)
        os.makedirs(os.path.join(self.pipeline_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.pipeline_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.pipeline_dir, "data_snapshots"), exist_ok=True)
        os.makedirs(os.path.join(self.pipeline_dir, "logs"), exist_ok=True)
        
        # Set up MLflow
        if use_mlflow:
            self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        # Try to create experiment (will succeed or return existing experiment)
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        
        self.logger.info(f"Using MLflow experiment '{self.experiment_name}' (ID: {self.experiment_id})")
    
    def set_config(self, config):
        """
        Set configuration for the pipeline.
        
        Args:
            config (dict): Configuration parameters
            
        Returns:
            self: For method chaining
        """
        self.config = config
        self._save_config()
        
        if self.use_mlflow and self.run_id:
            # Log parameters to MLflow
            for key, value in config.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)
                else:
                    try:
                        # Try to convert to JSON
                        mlflow.log_param(f"{key}_type", str(type(value)))
                        
                        # For dictionaries and lists, log them as separate parameters
                        if isinstance(value, dict):
                            for k, v in value.items():
                                if isinstance(v, (int, float, str, bool)):
                                    mlflow.log_param(f"{key}.{k}", v)
                        elif isinstance(value, list) and len(value) < 10:
                            for i, v in enumerate(value):
                                if isinstance(v, (int, float, str, bool)):
                                    mlflow.log_param(f"{key}.{i}", v)
                    except:
                        self.logger.warning(f"Could not log parameter {key} to MLflow")
        
        return self
    
    def _save_config(self):
        """Save configuration to disk."""
        config_path = os.path.join(self.pipeline_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _compute_data_hash(self, X, y=None):
        """
        Compute hash of input data for reproducibility tracking.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            str: Hash of the data
        """
        # Convert X to bytes
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            data_bytes = X.to_csv().encode()
        elif isinstance(X, np.ndarray):
            data_bytes = X.tobytes()
        else:
            data_bytes = str(X).encode()
        
        # Add y to bytes if provided
        if y is not None:
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                data_bytes += y.to_csv().encode()
            elif isinstance(y, np.ndarray):
                data_bytes += y.tobytes()
            else:
                data_bytes += str(y).encode()
        
        # Compute hash
        return hashlib.sha256(data_bytes).hexdigest()
    
    def build_pipeline(self, steps):
        """
        Build a scikit-learn pipeline with the given steps.
        
        Args:
            steps: List of (name, transform) tuples
            
        Returns:
            self: For method chaining
        """
        self.pipeline = Pipeline(steps)
        
        # Compute pipeline hash based on steps
        pipeline_str = str(steps)
        self.pipeline_hash = hashlib.sha256(pipeline_str.encode()).hexdigest()
        
        self.logger.info(f"Built pipeline with hash {self.pipeline_hash}")
        
        return self
    
    def save_data_snapshot(self, X, y=None, metadata=None):
        """
        Save a snapshot of the data for reproducibility.
        
        Args:
            X: Input features
            y: Target variable
            metadata: Additional metadata about the data
            
        Returns:
            self: For method chaining
        """
        self.data_hash = self._compute_data_hash(X, y)
        
        snapshot_dir = os.path.join(self.pipeline_dir, "data_snapshots", self.data_hash)
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Save data
        self.logger.info(f"Saving data snapshot with hash {self.data_hash}")
        
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X.to_csv(os.path.join(snapshot_dir, "X.csv"), index=False)
        elif isinstance(X, np.ndarray):
            np.save(os.path.join(snapshot_dir, "X.npy"), X)
        
        if y is not None:
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y.to_csv(os.path.join(snapshot_dir, "y.csv"), index=False)
            elif isinstance(y, np.ndarray):
                np.save(os.path.join(snapshot_dir, "y.npy"), y)
        
        # Save metadata
        if metadata is not None:
            with open(os.path.join(snapshot_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Log to MLflow
        if self.use_mlflow and self.run_id:
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(f"data.{key}", value)
            
            mlflow.log_param("data_hash", self.data_hash)
        
        return self
    
    def start_run(self, run_name=None):
        """
        Start a new MLflow run.
        
        Args:
            run_name (str): Name for the run
            
        Returns:
            self: For method chaining
        """
        if self.use_mlflow:
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name or f"{self.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.run_id = run.info.run_id
            self.logger.info(f"Started MLflow run with ID {self.run_id}")
            
            # Log basic info
            mlflow.log_param("pipeline_name", self.name)
            if self.pipeline_hash:
                mlflow.log_param("pipeline_hash", self.pipeline_hash)
            if self.data_hash:
                mlflow.log_param("data_hash", self.data_hash)
        
        return self
    
    def end_run(self):
        """
        End the current MLflow run.
        
        Returns:
            self: For method chaining
        """
        if self.use_mlflow and self.run_id:
            mlflow.end_run()
            self.logger.info(f"Ended MLflow run with ID {self.run_id}")
            self.run_id = None
        
        return self
    
    def fit(self, X, y=None):
        """
        Fit the pipeline and record results.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            self: For method chaining
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline first.")
        
        # Compute data hash if not already done
        if self.data_hash is None:
            self.data_hash = self._compute_data_hash(X, y)
        
        # Start MLflow run if not already started
        if self.use_mlflow and not self.run_id:
            self.start_run(run_name=f"fit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Fit the pipeline
        self.logger.info("Fitting pipeline...")
        start_time = datetime.datetime.now()
        self.pipeline.fit(X, y)
        end_time = datetime.datetime.now()
        fit_time = (end_time - start_time).total_seconds()
        
        # Save fitted model
        model_path = os.path.join(
            self.pipeline_dir, "models", 
            f"{self.pipeline_hash}_{self.data_hash}.joblib"
        )
        pickle.dump(self.pipeline, open(model_path, 'wb'))
        
        # Record fitting metadata
        self.results["fit"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "data_hash": self.data_hash,
            "pipeline_hash": self.pipeline_hash,
            "model_path": model_path,
            "fit_time": fit_time
        }
        
        # Log to MLflow
        if self.use_mlflow and self.run_id:
            mlflow.log_metric("fit_time", fit_time)
            mlflow.sklearn.log_model(self.pipeline, "model")
            
            # Log model details if available
            try:
                params = self.pipeline.get_params()
                for key, value in params.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(f"model.{key}", value)
            except:
                self.logger.warning("Could not log model parameters to MLflow")
        
        # Save results
        self._save_results()
        
        self.logger.info(f"Pipeline fitted in {fit_time:.2f} seconds")
        
        return self
    
    def evaluate(self, X, y, metrics=None, cv=5):
        """
        Evaluate the pipeline with cross-validation.
        
        Args:
            X: Input features
            y: Target variable
            metrics (list): List of metric names
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Evaluation results
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline first.")
        
        # Default metrics
        if metrics is None:
            metrics = ['accuracy']
        
        # Start MLflow run if not already started
        if self.use_mlflow and not self.run_id:
            self.start_run(run_name=f"evaluate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Evaluate with cross-validation
        self.logger.info(f"Evaluating pipeline with {cv}-fold cross-validation...")
        
        eval_results = {}
        for metric in metrics:
            start_time = datetime.datetime.now()
            scores = cross_val_score(
                self.pipeline, X, y, 
                cv=cv, scoring=metric
            )
            end_time = datetime.datetime.now()
            eval_time = (end_time - start_time).total_seconds()
            
            eval_results[metric] = {
                "mean": scores.mean(),
                "std": scores.std(),
                "scores": scores.tolist(),
                "evaluation_time": eval_time
            }
            
            # Log to MLflow
            if self.use_mlflow and self.run_id:
                mlflow.log_metric(f"cv_mean_{metric}", scores.mean())
                mlflow.log_metric(f"cv_std_{metric}", scores.std())
                for i, score in enumerate(scores):
                    mlflow.log_metric(f"cv_fold_{i+1}_{metric}", score)
                mlflow.log_metric(f"evaluation_time_{metric}", eval_time)
        
        # Record evaluation metadata
        self.results["evaluation"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "data_hash": self._compute_data_hash(X, y),
            "metrics": eval_results,
            "cv": cv
        }
        
        # Save results
        self._save_results()
        
        self.logger.info(f"Evaluation complete: {', '.join([f'{metric}={eval_results[metric]['mean']:.4f}±{eval_results[metric]['std']:.4f}' for metric in metrics])}")
        
        return eval_results
    
    def optimize_hyperparameters(self, X, y, param_grid, cv=5, scoring=None, n_jobs=-1):
        """
        Optimize hyperparameters using grid search.
        
        Args:
            X: Input features
            y: Target variable
            param_grid (dict): Grid of parameters to search
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            n_jobs (int): Number of parallel jobs
            
        Returns:
            tuple: (best_params, best_score)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline first.")
        
        # Start MLflow run if not already started
        if self.use_mlflow and not self.run_id:
            self.start_run(run_name=f"optimize_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Create grid search
        self.logger.info("Starting hyperparameter optimization...")
        
        grid_search = GridSearchCV(
            self.pipeline, param_grid, 
            cv=cv, scoring=scoring,
            verbose=1, n_jobs=n_jobs,
            return_train_score=True
        )
        
        # Fit grid search
        start_time = datetime.datetime.now()
        grid_search.fit(X, y)
        end_time = datetime.datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        # Get best pipeline
        self.pipeline = grid_search.best_estimator_
        
        # Update pipeline hash
        pipeline_str = str(self.pipeline.get_params())
        self.pipeline_hash = hashlib.sha256(pipeline_str.encode()).hexdigest()
        
        # Save optimized model
        model_path = os.path.join(
            self.pipeline_dir, "models", 
            f"{self.pipeline_hash}_{self.data_hash}_optimized.joblib"
        )
        pickle.dump(self.pipeline, open(model_path, 'wb'))
        
        # Log to MLflow
        if self.use_mlflow and self.run_id:
            # Log best parameters
            for key, value in grid_search.best_params_.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(f"best_{key}", value)
            
            # Log best score
            mlflow.log_metric("best_score", grid_search.best_score_)
            mlflow.log_metric("optimization_time", optimization_time)
            
            # Log best model
            mlflow.sklearn.log_model(self.pipeline, "best_model")
            
            # Log CV results
            cv_results = pd.DataFrame(grid_search.cv_results_)
            cv_results_path = os.path.join(self.pipeline_dir, "results", f"cv_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            cv_results.to_csv(cv_results_path, index=False)
            mlflow.log_artifact(cv_results_path, "cv_results")
        
        # Record optimization metadata
        self.results["optimization"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "data_hash": self.data_hash,
            "pipeline_hash": self.pipeline_hash,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "model_path": model_path,
            "optimization_time": optimization_time
        }
        
        # Save results
        self._save_results()
        
        self.logger.info(f"Hyperparameter optimization complete in {optimization_time:.2f} seconds")
        self.logger.info(f"Best score: {grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_params_, grid_search.best_score_
    
    def predict(self, X):
        """
        Make predictions using the pipeline.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline first.")
        
        self.logger.info("Making predictions...")
        return self.pipeline.predict(X)
    
    def _save_results(self):
        """Save results to disk."""
        results_path = os.path.join(
            self.pipeline_dir, "results", 
            f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Also save the latest results
        latest_results_path = os.path.join(self.pipeline_dir, "results", "latest_results.json")
        with open(latest_results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
    @classmethod
    def load(cls, name, base_dir="./pipelines", snapshot=None):
        """
        Load a previously saved pipeline.
        
        Args:
            name (str): Name of the pipeline
            base_dir (str): Base directory where pipelines are stored
            snapshot (str): Optional model snapshot to load
            
        Returns:
            ReproduciblePipeline: Loaded pipeline
        """
        pipeline_dir = os.path.join(base_dir, name)
        
        if not os.path.exists(pipeline_dir):
            raise ValueError(f"Pipeline {name} not found in {base_dir}")
        
        # Create pipeline instance
        pipeline_obj = cls(name, base_dir)
        
        # Load config
        config_path = os.path.join(pipeline_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                pipeline_obj.config = json.load(f)
        
        # Load results
        results_path = os.path.join(pipeline_dir, "results", "latest_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                pipeline_obj.results = json.load(f)
        
        # Load model
        if snapshot is None and "fit" in pipeline_obj.results:
            model_path = pipeline_obj.results["fit"]["model_path"]
            pipeline_obj.pipeline = pickle.load(open(model_path, 'rb'))
            pipeline_obj.data_hash = pipeline_obj.results["fit"]["data_hash"]
            pipeline_obj.pipeline_hash = pipeline_obj.results["fit"]["pipeline_hash"]
        elif snapshot is not None:
            model_path = os.path.join(pipeline_dir, "models", f"{snapshot}.joblib")
            if os.path.exists(model_path):
                pipeline_obj.pipeline = pickle.load(open(model_path, 'rb'))
            else:
                raise ValueError(f"Model snapshot {snapshot} not found")
        
        return pipeline_obj


class MLflowTracker:
    """
    Utility class for tracking experiments with MLflow.
    """
    
    def __init__(self, experiment_name, tracking_uri=None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name (str): Name of the MLflow experiment
            tracking_uri (str): URI for MLflow tracking server
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        self.run_id = None
    
    def start_run(self, run_name=None):
        """
        Start a new MLflow run.
        
        Args:
            run_name (str): Name for the run
            
        Returns:
            str: Run ID
        """
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.run_id = run.info.run_id
        return self.run_id
    
    def end_run(self):
        """End the current run."""
        if self.run_id:
            mlflow.end_run()
            self.run_id = None
    
    def log_sklearn_model(self, model, params, metrics):
        """
        Log a scikit-learn model to MLflow.
        
        Args:
            model: scikit-learn model
            params (dict): Parameters
            metrics (dict): Metrics
            
        Returns:
            str: Run ID
        """
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            return run.info.run_id
    
    def log_tensorflow_model(self, model, params, metrics):
        """
        Log a TensorFlow model to MLflow.
        
        Args:
            model: TensorFlow model
            params (dict): Parameters
            metrics (dict): Metrics
            
        Returns:
            str: Run ID
        """
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log model
            mlflow.tensorflow.log_model(model, "model")
            
            return run.info.run_id
    
    def get_run_details(self, run_id=None):
        """
        Get details of a run.
        
        Args:
            run_id (str): Run ID (uses current run if None)
            
        Returns:
            dict: Run details
        """
        run_id = run_id or self.run_id
        if not run_id:
            raise ValueError("No run ID provided and no current run")
        
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        return {
            "run_id": run_id,
            "status": run.info.status,
            "start_time": datetime.datetime.fromtimestamp(run.info.start_time / 1000.0),
            "end_time": datetime.datetime.fromtimestamp(run.info.end_time / 1000.0) if run.info.end_time else None,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags
        }
    
    def list_runs(self, n=10):
        """
        List recent runs for the experiment.
        
        Args:
            n (int): Number of runs to list
            
        Returns:
            list: Run details
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            max_results=n
        )
        
        return runs
    
    def get_best_run(self, metric, ascending=False):
        """
        Get the best run based on a metric.
        
        Args:
            metric (str): Metric name
            ascending (bool): Whether to sort in ascending order
            
        Returns:
            str: Run ID of best run
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"metrics.{metric} IS NOT NULL",
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )
        
        if len(runs) == 0:
            return None
        
        return runs.iloc[0].run_id
    
    def get_model(self, run_id):
        """
        Load a model from a run.
        
        Args:
            run_id (str): Run ID
            
        Returns:
            Model object
        """
        try:
            return mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        except:
            try:
                return mlflow.tensorflow.load_model(f"runs:/{run_id}/model")
            except:
                raise ValueError(f"Could not load model for run {run_id}")
    
    def compare_runs(self, run_ids, metric_names=None):
        """
        Compare runs based on metrics.
        
        Args:
            run_ids (list): List of run IDs
            metric_names (list): List of metric names to compare
            
        Returns:
            pd.DataFrame: Comparison of runs
        """
        client = mlflow.tracking.MlflowClient()
        
        comparison = {}
        for run_id in run_ids:
            run = client.get_run(run_id)
            run_data = {
                "params": run.data.params,
                "metrics": run.data.metrics
            }
            comparison[run_id] = run_data
        
        if metric_names is not None:
            # Create a dataframe for comparing specific metrics
            metric_data = []
            for run_id, data in comparison.items():
                metrics = {metric: data["metrics"].get(metric, None) for metric in metric_names}
                metrics["run_id"] = run_id
                metric_data.append(metrics)
            
            return pd.DataFrame(metric_data)
        
        return comparison
