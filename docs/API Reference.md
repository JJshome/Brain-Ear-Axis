# Brain-Ear Axis Platform: API Reference

## Overview
This document provides comprehensive documentation for the Brain-Ear Axis API, enabling programmatic access to all platform capabilities.

## Authentication
```python
import brainear

# Initialize with API key
client = brainear.Client(api_key="your_api_key")

# Verify authentication
status = client.verify_credentials()
```

## Data Management

### Loading Data
```python
# Load neural data
neural_dataset = client.load_neural_data(
    filepath="path/to/eeg_data.edf",
    patient_id="P12345",
    format="edf"
)

# Load microbiome data
microbiome_dataset = client.load_microbiome_data(
    filepath="path/to/microbiome_data.biom",
    patient_id="P12345",
    format="biom"
)

# Load auditory data
auditory_dataset = client.load_auditory_data(
    filepath="path/to/auditory_tests.csv",
    patient_id="P12345"
)
```

### Data Preprocessing
```python
# Preprocess neural data
preprocessed_neural = client.preprocess_neural(
    dataset=neural_dataset,
    filter_range=(0.5, 50),  # Hz
    remove_artifacts=True,
    reference="average"
)

# Preprocess microbiome data
preprocessed_microbiome = client.preprocess_microbiome(
    dataset=microbiome_dataset,
    normalization="relative_abundance",
    min_abundance=0.01
)
```

## Analysis Methods

### Connectivity Analysis
```python
# Calculate brain connectivity metrics
connectivity = client.analyze_connectivity(
    neural_data=preprocessed_neural,
    method="phase_locking_value",
    frequency_band=(8, 12),  # Alpha band
    regions=["temporal", "frontal"]
)

# Visualize connectivity
client.plot_connectivity(
    connectivity=connectivity,
    plot_type="circular",
    save_path="connectivity_results.png"
)
```

### Microbiome-Neural Integration
```python
# Correlation analysis
correlation = client.correlate_microbiome_neural(
    microbiome_data=preprocessed_microbiome,
    neural_data=preprocessed_neural,
    method="spearman",
    multiple_testing_correction="fdr"
)

# Network analysis
network = client.build_network(
    correlation=correlation,
    threshold=0.6,
    community_detection="louvain"
)
```

### Auditory Processing
```python
# Auditory response analysis
auditory_response = client.analyze_auditory_response(
    neural_data=preprocessed_neural,
    auditory_data=auditory_dataset,
    stimuli_times=[100, 200, 300],  # ms
    epoch_window=(-100, 500)  # ms
)

# Classification of responses
classification = client.classify_responses(
    response_data=auditory_response,
    classifier="random_forest",
    cross_validation=5
)
```

## Results and Reporting

### Statistical Analysis
```python
# Statistical testing
statistics = client.run_statistics(
    data=correlation,
    test="permutation",
    iterations=1000,
    alpha=0.05
)

# Effect size calculation
effect_sizes = client.calculate_effect_size(
    data=classification,
    method="cohen_d"
)
```

### Visualization
```python
# Create interactive dashboard
dashboard = client.create_dashboard(
    neural_data=preprocessed_neural,
    microbiome_data=preprocessed_microbiome,
    connectivity=connectivity,
    correlation=correlation,
    save_path="patient_dashboard.html"
)
```

### Export Results
```python
# Export to various formats
client.export_results(
    results={
        "neural": preprocessed_neural,
        "microbiome": preprocessed_microbiome,
        "connectivity": connectivity,
        "correlation": correlation,
        "statistics": statistics
    },
    format="json",
    filepath="analysis_results.json"
)
```

## Error Handling
```python
try:
    result = client.analyze_connectivity(
        neural_data=preprocessed_neural,
        method="invalid_method"
    )
except brainear.InvalidMethodError as e:
    print(f"Error: {e}")
    # Handle the error appropriately
```

## Batch Processing
```python
# Process multiple patients
results = client.batch_process(
    patient_ids=["P12345", "P12346", "P12347"],
    pipeline="standard_analysis",
    parallel=True,
    max_workers=4
)
```

## Advanced Configuration
```python
# Set global configuration
client.configure(
    compute_backend="gpu",
    temp_directory="/path/to/temp",
    logging_level="INFO",
    cache_size=2048  # MB
)
```
