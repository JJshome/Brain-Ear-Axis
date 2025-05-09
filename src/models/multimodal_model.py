"""
Multi-Modal Integration Model for Brain-Ear Axis Analysis

This module implements models for integrating data from multiple modalities
including neural signals, microbiome data, and auditory measurements.
"""

import tensorflow as tf
import numpy as np
import os
import pickle


class MultiModalIntegration(tf.keras.Model):
    """
    Model for integrating multi-modal data from neural, microbiome, and auditory sources.
    
    This model uses separate encoders for each modality and combines them using
    various fusion strategies.
    """
    
    def __init__(self, 
                 neural_shape, 
                 microbiome_shape, 
                 auditory_shape, 
                 output_size,
                 fusion_strategy='attention',
                 dropout_rate=0.3):
        """
        Initialize the multi-modal integration model.
        
        Args:
            neural_shape (tuple): Shape of neural input data
            microbiome_shape (tuple): Shape of microbiome input data
            auditory_shape (tuple): Shape of auditory input data
            output_size (int): Size of output layer
            fusion_strategy (str): Fusion strategy ('concatenate', 'attention', or 'weighted')
            dropout_rate (float): Dropout rate for regularization
        """
        super(MultiModalIntegration, self).__init__()
        
        self.neural_shape = neural_shape
        self.microbiome_shape = microbiome_shape
        self.auditory_shape = auditory_shape
        self.output_size = output_size
        self.fusion_strategy = fusion_strategy
        self.dropout_rate = dropout_rate
        
        # Build neural encoder
        self.neural_encoder = self._build_neural_encoder()
        
        # Build microbiome encoder
        self.microbiome_encoder = self._build_microbiome_encoder()
        
        # Build auditory encoder
        self.auditory_encoder = self._build_auditory_encoder()
        
        # Build fusion layer
        if fusion_strategy == 'attention':
            self.fusion_layer = self._build_attention_fusion()
        elif fusion_strategy == 'weighted':
            self.fusion_layer = self._build_weighted_fusion()
        else:  # Default to concatenate
            self.fusion_layer = self._build_concatenate_fusion()
        
        # Build classifier
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(output_size, activation='softmax')
        ])
    
    def _build_neural_encoder(self):
        """
        Build encoder for neural data.
        
        Returns:
            Neural encoder model
        """
        # Check if input is time series
        if len(self.neural_shape) > 1:
            # Use LSTM for time series data
            return tf.keras.Sequential([
                tf.keras.layers.Input(shape=self.neural_shape),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(16, activation='relu')
            ])
        else:
            # Use Dense layers for non-time series data
            return tf.keras.Sequential([
                tf.keras.layers.Input(shape=self.neural_shape),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu')
            ])
    
    def _build_microbiome_encoder(self):
        """
        Build encoder for microbiome data.
        
        Returns:
            Microbiome encoder model
        """
        # Microbiome data is typically high-dimensional feature vectors
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.microbiome_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu')
        ])
    
    def _build_auditory_encoder(self):
        """
        Build encoder for auditory data.
        
        Returns:
            Auditory encoder model
        """
        # Check if input is time series
        if len(self.auditory_shape) > 1:
            # Use CNN for spectral data or time series
            return tf.keras.Sequential([
                tf.keras.layers.Input(shape=self.auditory_shape),
                tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu')
            ])
        else:
            # Use Dense layers for feature vectors
            return tf.keras.Sequential([
                tf.keras.layers.Input(shape=self.auditory_shape),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu')
            ])
    
    def _build_attention_fusion(self):
        """
        Build attention-based fusion layer.
        
        Returns:
            Attention fusion layer
        """
        class AttentionFusion(tf.keras.layers.Layer):
            def __init__(self, attention_units=32):
                super(AttentionFusion, self).__init__()
                self.attention_units = attention_units
                
                # Attention layers
                self.W1 = tf.keras.layers.Dense(attention_units)
                self.W2 = tf.keras.layers.Dense(attention_units)
                self.W3 = tf.keras.layers.Dense(attention_units)
                self.V = tf.keras.layers.Dense(1)
            
            def call(self, inputs):
                # Inputs is a list of [neural, microbiome, auditory]
                neural, microbiome, auditory = inputs
                
                # Calculate attention scores
                neural_score = self.V(tf.nn.tanh(self.W1(neural)))
                microbiome_score = self.V(tf.nn.tanh(self.W2(microbiome)))
                auditory_score = self.V(tf.nn.tanh(self.W3(auditory)))
                
                # Calculate attention weights
                attention_weights = tf.nn.softmax(
                    tf.concat([neural_score, microbiome_score, auditory_score], axis=1),
                    axis=1
                )
                
                # Split attention weights
                neural_weight, microbiome_weight, auditory_weight = tf.split(
                    attention_weights, 
                    num_or_size_splits=3, 
                    axis=1
                )
                
                # Apply attention weights
                neural_weighted = neural * neural_weight
                microbiome_weighted = microbiome * microbiome_weight
                auditory_weighted = auditory * auditory_weight
                
                # Concatenate weighted features
                return tf.concat([neural_weighted, microbiome_weighted, auditory_weighted], axis=1)
        
        return AttentionFusion()
    
    def _build_weighted_fusion(self):
        """
        Build weighted fusion layer with learnable weights.
        
        Returns:
            Weighted fusion layer
        """
        class WeightedFusion(tf.keras.layers.Layer):
            def __init__(self):
                super(WeightedFusion, self).__init__()
                self.neural_weight = self.add_weight(
                    shape=(), 
                    initializer=tf.keras.initializers.Constant(0.33),
                    trainable=True,
                    name='neural_weight'
                )
                self.microbiome_weight = self.add_weight(
                    shape=(), 
                    initializer=tf.keras.initializers.Constant(0.33),
                    trainable=True,
                    name='microbiome_weight'
                )
                self.auditory_weight = self.add_weight(
                    shape=(), 
                    initializer=tf.keras.initializers.Constant(0.33),
                    trainable=True,
                    name='auditory_weight'
                )
            
            def call(self, inputs):
                # Inputs is a list of [neural, microbiome, auditory]
                neural, microbiome, auditory = inputs
                
                # Get normalized weights (sum to 1)
                total = self.neural_weight + self.microbiome_weight + self.auditory_weight
                norm_neural_weight = self.neural_weight / total
                norm_microbiome_weight = self.microbiome_weight / total
                norm_auditory_weight = self.auditory_weight / total
                
                # Apply weights
                neural_weighted = neural * norm_neural_weight
                microbiome_weighted = microbiome * norm_microbiome_weight
                auditory_weighted = auditory * norm_auditory_weight
                
                # Concatenate weighted features
                return tf.concat([neural_weighted, microbiome_weighted, auditory_weighted], axis=1)
        
        return WeightedFusion()
    
    def _build_concatenate_fusion(self):
        """
        Build simple concatenation fusion layer.
        
        Returns:
            Concatenation fusion layer
        """
        class ConcatenateFusion(tf.keras.layers.Layer):
            def call(self, inputs):
                # Inputs is a list of [neural, microbiome, auditory]
                return tf.concat(inputs, axis=1)
        
        return ConcatenateFusion()
    
    def call(self, inputs):
        """
        Forward pass through the model.
        
        Args:
            inputs: List of [neural_input, microbiome_input, auditory_input]
            
        Returns:
            Model output
        """
        neural_input, microbiome_input, auditory_input = inputs
        
        # Encode each modality
        neural_encoded = self.neural_encoder(neural_input)
        microbiome_encoded = self.microbiome_encoder(microbiome_input)
        auditory_encoded = self.auditory_encoder(auditory_input)
        
        # Fuse modalities
        fused = self.fusion_layer([neural_encoded, microbiome_encoded, auditory_encoded])
        
        # Classify
        output = self.classifier(fused)
        
        return output
    
    def build_model(self):
        """
        Build and compile the full model.
        
        Returns:
            Compiled Keras model
        """
        # Define inputs
        neural_input = tf.keras.Input(shape=self.neural_shape)
        microbiome_input = tf.keras.Input(shape=self.microbiome_shape)
        auditory_input = tf.keras.Input(shape=self.auditory_shape)
        
        # Get output from call method
        output = self.call([neural_input, microbiome_input, auditory_input])
        
        # Create model
        model = tf.keras.Model(
            inputs=[neural_input, microbiome_input, auditory_input],
            outputs=output
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, neural_data, microbiome_data, auditory_data, labels, 
              validation_data=None, epochs=50, batch_size=32, callbacks=None):
        """
        Train the multi-modal model.
        
        Args:
            neural_data: Neural input data
            microbiome_data: Microbiome input data
            auditory_data: Auditory input data
            labels: Target labels
            validation_data: Tuple of (X_val, y_val) for validation
                where X_val is a list of [neural_val, microbiome_val, auditory_val]
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            
        Returns:
            History object containing training metrics
        """
        # Build and compile model
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
            x=[neural_data, microbiome_data, auditory_data],
            y=labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Update weights from trained model
        self.set_weights(model.get_weights())
        
        return history
    
    def predict(self, neural_data, microbiome_data, auditory_data):
        """
        Make predictions using the trained model.
        
        Args:
            neural_data: Neural input data
            microbiome_data: Microbiome input data
            auditory_data: Auditory input data
            
        Returns:
            Model predictions
        """
        # Build model for prediction
        model = self.build_model()
        
        # Make predictions
        return model.predict([neural_data, microbiome_data, auditory_data])
    
    def evaluate(self, neural_data, microbiome_data, auditory_data, labels):
        """
        Evaluate the model performance.
        
        Args:
            neural_data: Neural input data
            microbiome_data: Microbiome input data
            auditory_data: Auditory input data
            labels: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Build model for evaluation
        model = self.build_model()
        
        # Evaluate model
        loss, accuracy = model.evaluate(
            [neural_data, microbiome_data, auditory_data],
            labels
        )
        
        # Make predictions
        predictions = model.predict([neural_data, microbiome_data, auditory_data])
        
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
                'neural_shape': self.neural_shape,
                'microbiome_shape': self.microbiome_shape,
                'auditory_shape': self.auditory_shape,
                'output_size': self.output_size,
                'fusion_strategy': self.fusion_strategy,
                'dropout_rate': self.dropout_rate
            }, f)
    
    @classmethod
    def load(cls, path):
        """
        Load a previously saved model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded MultiModalIntegration model
        """
        # Load model attributes
        with open(f"{path}_attributes.pkl", 'rb') as f:
            attributes = pickle.load(f)
        
        # Create instance
        instance = cls(
            neural_shape=attributes['neural_shape'],
            microbiome_shape=attributes['microbiome_shape'],
            auditory_shape=attributes['auditory_shape'],
            output_size=attributes['output_size'],
            fusion_strategy=attributes['fusion_strategy'],
            dropout_rate=attributes['dropout_rate']
        )
        
        # Load Keras model
        keras_model = tf.keras.models.load_model(path)
        
        # Set weights from loaded model
        instance.set_weights(keras_model.get_weights())
        
        return instance


class SelfSupervisedFeatureExtractor:
    """
    Self-supervised feature extractor for unlabeled data.
    
    This class implements autoencoder-based feature extraction to learn
    representations from unlabeled data.
    """
    
    def __init__(self, input_shape, latent_dim=32):
        """
        Initialize the feature extractor.
        
        Args:
            input_shape (tuple): Shape of input data
            latent_dim (int): Dimension of latent space
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.autoencoder = None
        self.encoder = None
        
    def build_autoencoder(self):
        """
        Build autoencoder model for feature extraction.
        
        Returns:
            Tuple of (encoder, autoencoder) models
        """
        # Check if input is time series
        if len(self.input_shape) > 1:
            # Build convolutional autoencoder for time series
            return self._build_conv_autoencoder()
        else:
            # Build dense autoencoder for feature vectors
            return self._build_dense_autoencoder()
    
    def _build_conv_autoencoder(self):
        """
        Build convolutional autoencoder for time series data.
        
        Returns:
            Tuple of (encoder, autoencoder) models
        """
        # Encoder
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        
        # Convolutional layers for encoder
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(input_layer)
        x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
        x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
        
        # Flatten and dense for bottleneck
        flatten_shape = x.shape[1:]
        x = tf.keras.layers.Flatten()(x)
        encoded = tf.keras.layers.Dense(self.latent_dim, activation='relu')(x)
        
        # Build encoder model
        encoder = tf.keras.Model(input_layer, encoded, name='encoder')
        
        # Decoder
        decoder_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(tf.math.reduce_prod(flatten_shape))(decoder_input)
        x = tf.keras.layers.Reshape(flatten_shape)(x)
        
        # Convolutional layers for decoder
        x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling1D(2)(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling1D(2)(x)
        decoded = tf.keras.layers.Conv1D(self.input_shape[-1], 3, activation='sigmoid', padding='same')(x)
        
        # Build decoder model
        decoder = tf.keras.Model(decoder_input, decoded, name='decoder')
        
        # Build autoencoder by connecting encoder and decoder
        autoencoder_output = decoder(encoder(input_layer))
        autoencoder = tf.keras.Model(input_layer, autoencoder_output, name='autoencoder')
        
        # Compile autoencoder
        autoencoder.compile(optimizer='adam', loss='mse')
        
        self.encoder = encoder
        self.autoencoder = autoencoder
        
        return encoder, autoencoder
    
    def _build_dense_autoencoder(self):
        """
        Build dense autoencoder for feature vector data.
        
        Returns:
            Tuple of (encoder, autoencoder) models
        """
        # Encoder
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        encoded = tf.keras.layers.Dense(self.latent_dim, activation='relu')(x)
        
        # Build encoder model
        encoder = tf.keras.Model(input_layer, encoded, name='encoder')
        
        # Decoder
        decoder_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(decoder_input)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        decoded = tf.keras.layers.Dense(self.input_shape[0], activation='sigmoid')(x)
        
        # Build decoder model
        decoder = tf.keras.Model(decoder_input, decoded, name='decoder')
        
        # Build autoencoder by connecting encoder and decoder
        autoencoder_output = decoder(encoder(input_layer))
        autoencoder = tf.keras.Model(input_layer, autoencoder_output, name='autoencoder')
        
        # Compile autoencoder
        autoencoder.compile(optimizer='adam', loss='mse')
        
        self.encoder = encoder
        self.autoencoder = autoencoder
        
        return encoder, autoencoder
    
    def train(self, data, validation_data=None, epochs=100, batch_size=32, callbacks=None):
        """
        Train the autoencoder.
        
        Args:
            data: Training data
            validation_data: Validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        # Build autoencoder if not already built
        if self.autoencoder is None:
            self.build_autoencoder()
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data is not None else 'loss', 
                    patience=10, 
                    restore_best_weights=True
                )
            ]
        
        # Prepare validation data
        if validation_data is not None:
            validation_data = (validation_data, validation_data)
        
        # Train autoencoder
        history = self.autoencoder.fit(
            data, data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        return history
    
    def extract_features(self, data):
        """
        Extract features using the trained encoder.
        
        Args:
            data: Input data
            
        Returns:
            Extracted features
        """
        if self.encoder is None:
            raise ValueError("Encoder not built. Call build_autoencoder or train first.")
        
        return self.encoder.predict(data)
    
    def save(self, path):
        """
        Save the models to disk.
        
        Args:
            path: Base path where models will be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save models
        if self.encoder is not None:
            self.encoder.save(f"{path}_encoder")
        
        if self.autoencoder is not None:
            self.autoencoder.save(f"{path}_autoencoder")
        
        # Save attributes
        with open(f"{path}_attributes.pkl", 'wb') as f:
            pickle.dump({
                'input_shape': self.input_shape,
                'latent_dim': self.latent_dim
            }, f)
    
    @classmethod
    def load(cls, path):
        """
        Load a previously saved model.
        
        Args:
            path: Base path to the saved models
            
        Returns:
            Loaded SelfSupervisedFeatureExtractor
        """
        # Load attributes
        with open(f"{path}_attributes.pkl", 'rb') as f:
            attributes = pickle.load(f)
        
        # Create instance
        instance = cls(
            input_shape=attributes['input_shape'],
            latent_dim=attributes['latent_dim']
        )
        
        # Load models
        if os.path.exists(f"{path}_encoder"):
            instance.encoder = tf.keras.models.load_model(f"{path}_encoder")
        
        if os.path.exists(f"{path}_autoencoder"):
            instance.autoencoder = tf.keras.models.load_model(f"{path}_autoencoder")
        
        return instance
