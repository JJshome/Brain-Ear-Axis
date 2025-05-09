# Brain-Ear Axis Platform: Example Workflows

This document provides step-by-step workflows for common research and clinical scenarios using the Brain-Ear Axis platform.

## Research Workflows

### Workflow 1: Basic Neural-Microbiome Association Study

#### Step 1: Data Collection and Import
```python
import brainear as bea

# Initialize client
client = bea.Client(project_name="neural_microbiome_study")

# Import neural data
neural_data = client.import_data(
    data_type="neural",
    source_files=["eeg_recordings/*.edf"],
    subject_metadata="subject_info.csv"
)

# Import microbiome data
microbiome_data = client.import_data(
    data_type="microbiome",
    source_files=["microbiome_samples/*.fastq"],
    taxonomy_database="silva_138_1",
    subject_metadata="subject_info.csv"
)
```

#### Step 2: Data Preprocessing
```python
# Preprocess neural data
neural_processed = client.preprocess_neural(
    data=neural_data,
    steps=[
        "filter_bandpass(0.5, 45)",
        "remove_artifacts",
        "epoching(event_markers, -0.2, 1.0)",
        "baseline_correction(-0.2, 0)"
    ]
)

# Preprocess microbiome data
microbiome_processed = client.preprocess_microbiome(
    data=microbiome_data,
    steps=[
        "remove_low_quality_reads(q_score=20)",
        "taxonomic_classification",
        "normalize_abundance",
        "diversity_metrics"
    ]
)
```

#### Step 3: Analysis
```python
# Correlation analysis
correlation_results = client.analyze_correlation(
    dataset_x=neural_processed.spectral_power(bands=["alpha", "beta", "gamma"]),
    dataset_y=microbiome_processed.abundance_table(level="genus"),
    method="spearman",
    correction="fdr_bh"
)

# Network analysis
network = client.create_network(
    correlation_matrix=correlation_results.matrix,
    threshold=0.6,
    layout="force_directed"
)
```

#### Step 4: Visualization and Results
```python
# Create visualization dashboard
dashboard = client.create_dashboard([
    bea.plot.correlation_heatmap(correlation_results),
    bea.plot.network_graph(network),
    bea.plot.diversity_boxplot(microbiome_processed),
    bea.plot.brain_connectivity(neural_processed)
])

# Export results
client.export_results(
    results={
        "correlation": correlation_results,
        "network": network,
        "neural_metrics": neural_processed.get_metrics(),
        "microbiome_metrics": microbiome_processed.get_metrics()
    },
    format="xlsx",
    filename="neural_microbiome_study_results.xlsx"
)
```

### Workflow 2: Auditory Response and Microbial Influence

#### Step 1: Setup Project and Import Data
```python
project = bea.Project(
    name="auditory_microbiome_project",
    description="Investigating microbial influence on auditory processing",
    investigators=["Dr. Jane Smith", "Dr. John Doe"]
)

# Import data
auditory_data = project.import_auditory_data(
    evoked_potentials="auditory_erp/*.fif",
    behavioral_tests="hearing_tests/*.csv"
)

microbiome_data = project.import_microbiome_data(
    sequencing_data="gut_samples/*.biom",
    metadata="patient_metadata.xlsx"
)
```

#### Step 2: Auditory Processing
```python
# Process auditory ERPs
auditory_processed = project.process_auditory(
    data=auditory_data,
    erp_components=["P1", "N1", "P2", "P300"],
    regions_of_interest=["temporal_left", "temporal_right", "frontal"]
)

# Analyze hearing metrics
hearing_metrics = project.analyze_hearing(
    data=auditory_data.behavioral,
    tests=["pure_tone", "speech_recognition", "gap_detection"],
    normalize=True
)
```

#### Step 3: Integrated Analysis
```python
# Group patients by microbiome profile
patient_groups = project.cluster_subjects(
    data=microbiome_data,
    method="kmeans",
    n_clusters=3,
    features="abundance_top_50_genera"
)

# Compare auditory responses between microbiome clusters
comparison = project.compare_groups(
    groups=patient_groups,
    metrics=[
        auditory_processed.components,
        hearing_metrics.scores
    ],
    statistical_tests=["anova", "tukey_hsd"]
)
```

#### Step 4: Machine Learning Prediction
```python
# Train prediction model
prediction_model = project.train_model(
    predictors=microbiome_data.get_features(top_n=100),
    target=auditory_processed.get_component("P300").amplitude,
    algorithm="gradient_boosting",
    cross_validation=5,
    hyperparameter_tuning=True
)

# Evaluate model
model_performance = prediction_model.evaluate(
    metrics=["mse", "r2", "explained_variance"]
)

# Generate feature importance
feature_importance = prediction_model.get_feature_importance()
```

#### Step 5: Results and Report
```python
# Visualization
figures = [
    project.plot.erp_comparison(auditory_processed, groups=patient_groups),
    project.plot.microbiome_composition(microbiome_data, groups=patient_groups),
    project.plot.feature_importance(feature_importance, top_n=20),
    project.plot.prediction_actual(prediction_model)
]

# Generate report
report = project.generate_report(
    title="Microbiome Influence on Auditory Processing",
    sections=[
        "methodology",
        "patient_demographics",
        "microbiome_profiles",
        "auditory_responses",
        "statistical_analysis",
        "prediction_model"
    ],
    figures=figures,
    output_format="pdf"
)
```

## Clinical Workflows

### Workflow 3: Clinical Assessment and Monitoring

#### Step 1: Patient Onboarding
```python
# Create new patient record
patient = bea.Patient(
    id="PAT001",
    demographics={
        "age": 42,
        "sex": "F",
        "medical_history": ["tinnitus", "mild hearing loss"]
    }
)

# Initial assessment
initial_assessment = bea.ClinicalAssessment(
    patient=patient,
    assessment_type="initial",
    date="2025-01-15"
)

# Collect baseline measurements
baseline_data = initial_assessment.collect_data(
    modules=[
        "auditory_testing",
        "eeg_resting_state",
        "microbiome_sampling"
    ]
)
```

#### Step 2: Analysis Pipeline
```python
# Process through clinical pipeline
results = bea.clinical_pipeline(
    patient_data=baseline_data,
    analysis_modules=[
        "hearing_profile",
        "neural_signature",
        "microbiome_profile",
        "integrative_assessment"
    ]
)

# Risk assessment
risk_profile = bea.risk_assessment(
    patient=patient,
    analysis_results=results,
    risk_models=["hearing_deterioration", "tinnitus_severity"]
)
```

#### Step 3: Treatment Recommendation
```python
# Generate personalized recommendations
recommendations = bea.generate_recommendations(
    patient=patient,
    risk_profile=risk_profile,
    treatment_database="auditory_treatments_v2",
    personalization_factors=["age", "comorbidities", "microbiome_profile"]
)

# Create treatment plan
treatment_plan = bea.TreatmentPlan(
    patient=patient,
    recommendations=recommendations,
    start_date="2025-01-22",
    duration_weeks=12,
    monitoring_frequency="biweekly"
)
```

#### Step 4: Monitoring and Follow-up
```python
# Schedule follow-up assessments
follow_ups = treatment_plan.schedule_follow_ups(
    assessment_types=["hearing", "microbiome", "neural"],
    intervals=[2, 4, 8, 12]  # weeks
)

# Perform follow-up assessment (at week 4)
follow_up_assessment = bea.ClinicalAssessment(
    patient=patient,
    assessment_type="follow_up",
    date="2025-02-19",
    reference_assessment=initial_assessment
)

follow_up_data = follow_up_assessment.collect_data(
    modules=[
        "auditory_testing",
        "eeg_resting_state",
        "microbiome_sampling"
    ]
)

# Compare with baseline
comparison = bea.compare_assessments(
    baseline=baseline_data,
    follow_up=follow_up_data,
    metrics=["hearing_thresholds", "neural_connectivity", "microbiome_diversity"]
)
```

#### Step 5: Treatment Adjustment
```python
# Evaluate treatment effectiveness
effectiveness = bea.evaluate_treatment(
    treatment_plan=treatment_plan,
    assessment_comparison=comparison,
    threshold_improvement=0.15  # 15% improvement required
)

# Adjust treatment if necessary
if not effectiveness.meets_threshold:
    adjusted_plan = bea.adjust_treatment(
        current_plan=treatment_plan,
        assessment_results=follow_up_data,
        adjustment_strategy="increase_intensity"
    )
```
