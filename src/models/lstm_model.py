"""
Brain-Ear LSTM Model for Time Series Analysis

This module implements LSTM-based neural network models for analyzing 
temporal patterns in neural signal data related to auditory processing.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import numpy as np
import os
import pickle

class BrainEarLSTM:
    """
    LSTM model for analyzing brain-ear related time series data.
    
    This model can be used for various tasks like anomaly detection,
    classification, or prediction of time series data from EEG, MEG,
    or auditory brainstem responses.
    """
    def __init__(self, input_shape, output_size, lstm_units=(128, 64)):
        """
        Initialize the LSTM model.
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            output_size (int): Size of output layer (e.g., number of classes)
            lstm_units (tuple): Number of units in LSTM layers
        """
        self.input_shape = input_shape
        self.output_size = output_size
        
        # Build model
        self.model = Sequential([
            LSTM(lstm_units[0], return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(lstm_units[1]),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(output_size, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, validation_data=None, epochs=50, batch_size=32, 
              callbacks=None):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            validation_data: Tuple of (X_val, y_val) for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            
        Returns:
            History object containing training metrics
        """
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=10, 
                    restore_best_weights=True
                )
            ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data
            
        Returns:
            Model predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        loss, accuracy = self.model.evaluate(X_test, y_test)
        predictions = self.predict(X_test)
        
        results = {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': predictions
        }
        
        return results
    
    def detect_anomalies(self, X, threshold=0.9):
        """
        Detect anomalies in time series data.
        
        Args:
            X: Input time series data
            threshold: Anomaly detection threshold
            
        Returns:
            Boolean array indicating anomalies
        """
        # Predict on data
        predictions = self.predict(X)
        
        # Calculate prediction confidence
        max_probs = np.max(predictions, axis=1)
        
        # Mark samples below threshold as anomalies
        anomalies = max_probs < threshold
        
        return anomalies
    
    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path: Path where model will be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.save(path)
        
        # Save additional attributes
        with open(f"{path}_attributes.pkl", 'wb') as f:
            pickle.dump({
                'input_shape': self.input_shape,
                'output_size': self.output_size
            }, f)
    
    @classmethod
    def load(cls, path):
        """
        Load a previously saved model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded BrainEarLSTM model
        """
        # Load model attributes
        with open(f"{path}_attributes.pkl", 'rb') as f:
            attributes = pickle.load(f)
        
        # Create instance without building model
        instance = cls.__new__(cls)
        instance.input_shape = attributes['input_shape']
        instance.output_size = attributes['output_size']
        
        # Load Keras model
        instance.model = tf.keras.models.load_model(path)
        
        return instance


class MultiChannelLSTM(BrainEarLSTM):
    """
    Specialized LSTM model for multi-channel neural or auditory data.
    """
    
    def __init__(self, input_shape, output_size, lstm_units=(128, 64), channel_attention=True):
        """
        Initialize the multi-channel LSTM model.
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, channels, features)
            output_size (int): Size of output layer
            lstm_units (tuple): Number of units in LSTM layers
            channel_attention (bool): Whether to use channel attention mechanism
        """
        self.input_shape = input_shape
        self.output_size = output_size
        self.channel_attention = channel_attention
        
        # Build custom model with channel attention if requested
        inputs = tf.keras.Input(shape=input_shape)
        
        if channel_attention:
            # Reshape for channel attention
            # (batch, timesteps, channels, features) -> (batch, timesteps, channels*features)
            reshape_layer = tf.keras.layers.Reshape(
                (input_shape[0], input_shape[1] * input_shape[2])
            )(inputs)
            
            # Apply channel attention
            channel_attention_layer = self._build_channel_attention(reshape_layer, input_shape)
            
            # LSTM layers
            x = LSTM(lstm_units[0], return_sequences=True)(channel_attention_layer)
        else:
            # Reshape for regular LSTM
            reshape_layer = tf.keras.layers.Reshape(
                (input_shape[0], input_shape[1] * input_shape[2])
            )(inputs)
            
            # LSTM layers
            x = LSTM(lstm_units[0], return_sequences=True)(reshape_layer)
        
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = LSTM(lstm_units[1])(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(output_size, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def _build_channel_attention(self, x, input_shape):
        """
        Build channel attention mechanism.
        
        Args:
            x: Input tensor
            input_shape: Shape of input data
            
        Returns:
            Tensor with channel attention applied
        """
        # Compute channel attention weights
        channels = input_shape[1]
        
        # Reshape to separate channels
        reshaped = tf.keras.layers.Reshape(
            (input_shape[0], channels, input_shape[2])
        )(x)
        
        # Global average pooling along time dimension
        pooled = tf.keras.layers.GlobalAveragePooling1D()(reshaped)
        
        # Calculate channel weights
        channel_weights = tf.keras.layers.Dense(channels, activation='sigmoid')(pooled)
        
        # Reshape weights for broadcasting
        weights_reshaped = tf.keras.layers.Reshape((1, channels, 1))(channel_weights)
        
        # Apply weights to channels
        weighted = tf.multiply(reshaped, weights_reshaped)
        
        # Reshape back to original shape for LSTM
        return tf.keras.layers.Reshape(
            (input_shape[0], channels * input_shape[2])
        )(weighted)
    
    def preprocess_data(self, X, normalize=True):
        """
        Preprocess multi-channel data for the model.
        
        Args:
            X: Input data with shape (samples, timesteps, channels)
            normalize: Whether to normalize each channel
            
        Returns:
            Preprocessed data ready for the model
        """
        if normalize:
            # Normalize each channel independently
            X_norm = np.zeros_like(X)
            for i in range(X.shape[0]):  # For each sample
                for j in range(X.shape[2]):  # For each channel
                    # Min-max normalization
                    channel_data = X[i, :, j]
                    min_val = np.min(channel_data)
                    max_val = np.max(channel_data)
                    
                    if max_val > min_val:  # Avoid division by zero
                        X_norm[i, :, j] = (channel_data - min_val) / (max_val - min_val)
                    else:
                        X_norm[i, :, j] = 0  # Constant channel
            
            return X_norm
        
        return X
