"""
Brain-Ear Transformer Model for Time Series Analysis

This module implements Transformer-based neural network models specifically designed
for analyzing complex temporal patterns in auditory and neural data.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.models import Model
import numpy as np
import os
import pickle


class TransformerBlock(tf.keras.layers.Layer):
    """
    Transformer block for sequence modeling.
    
    This implements a standard transformer block with multi-head attention
    and feed-forward network components.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        """
        Initialize transformer block.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward dimension
            rate (float): Dropout rate
        """
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        """
        Forward pass through transformer block.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        # Self-attention with residual connection and layer normalization
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network with residual connection and layer normalization
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TimeSeriesTransformer(tf.keras.Model):
    """
    Transformer model specifically designed for time series data analysis.
    
    This model applies transformer architecture to analyze temporal patterns
    in neural and auditory data.
    """
    def __init__(self, 
                 input_shape, 
                 output_size, 
                 head_size=256, 
                 num_heads=4, 
                 ff_dim=512,
                 num_transformer_blocks=4, 
                 mlp_units=[128], 
                 dropout=0.1,
                 use_time2vec=True):
        """
        Initialize the Transformer model.
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            output_size (int): Size of output layer (e.g., number of classes)
            head_size (int): Size of each attention head
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward network dimension
            num_transformer_blocks (int): Number of transformer blocks
            mlp_units (list): Units in final MLP layers
            dropout (float): Dropout rate
            use_time2vec (bool): Whether to use Time2Vec encoding
        """
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_shape = input_shape
        self.output_size = output_size
        self.use_time2vec = use_time2vec
        
        # Time2Vec layer for better time feature representation
        if use_time2vec:
            self.time2vec_layer = Time2Vec(kernel_size=head_size)
            # Adjust head_size if using Time2Vec
            adjusted_head_size = head_size
        else:
            adjusted_head_size = head_size
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(adjusted_head_size, num_heads, ff_dim, dropout) 
            for _ in range(num_transformer_blocks)
        ]
        
        # Global average pooling
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()
        
        # Final MLP layers
        self.mlp_layers = []
        for dim in mlp_units:
            self.mlp_layers.append(Dense(dim, activation="relu"))
            self.mlp_layers.append(Dropout(dropout))
        
        # Output layer
        self.output_layer = Dense(output_size, activation="softmax")
    
    def call(self, inputs, training=False):
        """
        Forward pass through the transformer model.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        # Apply Time2Vec if specified
        if self.use_time2vec:
            x = self.time2vec_layer(inputs)
        else:
            x = inputs
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        # Apply global average pooling
        x = self.global_avg_pooling(x)
        
        # Apply MLP layers
        for layer in self.mlp_layers:
            x = layer(x, training=training)
        
        # Apply output layer
        return self.output_layer(x)
    
    def build_model(self):
        """
        Build and compile the model.
        
        Returns:
            Compiled Keras model
        """
        # Define input
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Get output from call method
        outputs = self.call(inputs)
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer="adam", 
            loss="categorical_crossentropy", 
            metrics=["accuracy"]
        )
        
        return model
    
    def train(self, X_train, y_train, validation_data=None, epochs=50, batch_size=32, callbacks=None):
        """
        Train the transformer model.
        
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
        # Build and compile model if not already done
        model = self.build_model()
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=10, 
                    restore_best_weights=True
                )
            ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Update weights from the trained model
        self.set_weights(model.get_weights())
        
        return history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data
            
        Returns:
            Model predictions
        """
        # Build model for prediction if needed
        model = self.build_model()
        
        return model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Build model for evaluation if needed
        model = self.build_model()
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Return evaluation metrics
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': predictions
        }
    
    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path: Path where model will be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Build model
        model = self.build_model()
        
        # Save Keras model
        model.save(path)
        
        # Save additional attributes
        with open(f"{path}_attributes.pkl", 'wb') as f:
            pickle.dump({
                'input_shape': self.input_shape,
                'output_size': self.output_size,
                'use_time2vec': self.use_time2vec
            }, f)
    
    @classmethod
    def load(cls, path):
        """
        Load a previously saved model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded TimeSeriesTransformer model
        """
        # Load model attributes
        with open(f"{path}_attributes.pkl", 'rb') as f:
            attributes = pickle.load(f)
        
        # Create instance
        instance = cls(
            input_shape=attributes['input_shape'],
            output_size=attributes['output_size'],
            use_time2vec=attributes['use_time2vec']
        )
        
        # Load Keras model
        keras_model = tf.keras.models.load_model(
            path,
            custom_objects={
                'Time2Vec': Time2Vec,
                'TransformerBlock': TransformerBlock
            }
        )
        
        # Set weights from loaded model
        instance.set_weights(keras_model.get_weights())
        
        return instance


class Time2Vec(tf.keras.layers.Layer):
    """
    Time2Vec layer for better time feature representation.
    
    Implementation based on the paper "Time2Vec: Learning a Vector Representation of Time"
    https://arxiv.org/abs/1907.05321
    """
    def __init__(self, kernel_size=1):
        """
        Initialize Time2Vec layer.
        
        Args:
            kernel_size: Dimension of the time embedding
        """
        super(Time2Vec, self).__init__()
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        """
        Build Time2Vec layer.
        
        Args:
            input_shape: Shape of input tensor
        """
        # Linear time feature
        self.wb = self.add_weight(
            shape=(1,),
            initializer="uniform",
            trainable=True,
            name="wb"
        )
        self.bb = self.add_weight(
            shape=(1,),
            initializer="uniform",
            trainable=True,
            name="bb"
        )
        
        # Periodic time features
        self.wa = self.add_weight(
            shape=(1, self.kernel_size),
            initializer="uniform",
            trainable=True,
            name="wa"
        )
        self.ba = self.add_weight(
            shape=(1, self.kernel_size),
            initializer="uniform",
            trainable=True,
            name="ba"
        )
        
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs):
        """
        Forward pass of Time2Vec.
        
        Args:
            inputs: Input tensor of shape [batch, timesteps, features]
            
        Returns:
            Time embedded tensor
        """
        # Apply Time2Vec to each feature independently
        
        # Get feature dimension
        features = inputs.shape[-1]
        timesteps = inputs.shape[-2]
        
        # Reshape inputs for processing
        # [batch, timesteps, features] -> [batch, timesteps * features]
        reshaped_inputs = tf.reshape(inputs, [-1, timesteps * features])
        
        # Initialize output tensor
        batch_size = tf.shape(inputs)[0]
        time_encoded = []
        
        # Process each feature
        for i in range(features):
            # Extract feature across all timesteps
            feature = inputs[:, :, i]
            
            # Linear feature: wb * t + bb
            linear = feature * self.wb + self.bb
            
            # Periodic features: sin(wa * t + ba)
            periodic = tf.sin(tf.matmul(tf.expand_dims(feature, -1), self.wa) + self.ba)
            
            # Concatenate linear and periodic features
            feature_encoded = tf.concat([tf.expand_dims(linear, -1), periodic], axis=-1)
            
            # Add to list of encoded features
            time_encoded.append(feature_encoded)
        
        # Combine all encoded features
        # Each feature now has shape [batch, timesteps, kernel_size + 1]
        # We want to maintain the original feature structure but expand the feature dimension
        
        # Concatenate along feature dimension
        # Result: [batch, timesteps, features * (kernel_size + 1)]
        result = tf.concat(time_encoded, axis=-1)
        
        # Reshape to match expected output shape
        # [batch, timesteps, features * (kernel_size + 1)]
        return result
