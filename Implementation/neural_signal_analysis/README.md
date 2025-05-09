# Neural Signal Analysis Module (300)

This module implements the neural signal analysis component of the Brain-Ear Axis Analysis System. It focuses on analyzing brain activity in response to auditory stimuli to understand the functional connections between the brain and ear.

![Neural Signal Analysis](https://raw.githubusercontent.com/JJshome/Brain-Ear-Axis/main/examples/connectivity_visualization.svg)

## Components

### 1. Auditory Stimulus Presentation Module (310)
- Generation of various auditory stimuli (pure tones, complex sounds)
- Frequency and intensity calibration
- Stimulus timing and sequencing
- Adaptation for specific ear conditions (tinnitus matching, hearing threshold testing)

### 2. fMRI Data Acquisition Module (320)
- High-resolution functional MRI protocols for auditory cortex imaging
- Integration with 7T MRI scanner systems
- Preprocessing of functional and structural MRI data
- Noise cancellation for scanner acoustic interference

### 3. EEG Data Acquisition Module (330)
- High-density EEG recording during auditory tasks
- Real-time EEG signal processing
- Artifact removal and signal enhancement
- Event-related potential (ERP) extraction

### 4. Brain-Ear Axis Activation Pattern Analysis Module (340)
- Dynamic Causal Modeling (DCM) for functional connectivity analysis
- Functional network analysis between auditory and non-auditory regions
- Detection of abnormal activation patterns associated with ear diseases
- Temporal analysis of neural responses to auditory stimuli

## Key Capabilities

1. **Auditory Network Mapping**: Identification and characterization of brain networks involved in processing auditory information, with focus on connections to ear pathologies.

2. **Functional Connectivity Analysis**: Quantification of functional connectivity between auditory cortex and other brain regions, including limbic system for emotional processing of auditory stimuli.

3. **Neural Correlates of Ear Diseases**: Characterization of brain activity patterns associated with specific ear diseases (tinnitus, hearing loss, Meniere's disease).

4. **Brain-Ear Axis Evaluation**: Assessment of bidirectional communication between brain and ear through neural signal analysis.

## Directory Structure

```
neural_signal_analysis/
├── auditory_stimulus/       # Auditory stimulus generation and presentation
├── fmri_processing/         # fMRI data acquisition and processing
├── eeg_processing/          # EEG data acquisition and processing
├── connectivity_analysis/   # Brain connectivity and network analysis
├── data/                    # Reference data and models
│   ├── templates/           # Brain templates and atlases
│   ├── normative/           # Normative connectivity data
│   └── ear_disease_patterns/# Neural patterns associated with ear diseases
└── tests/                   # Module tests
```

## Technologies

- Python for data processing and analysis
- MATLAB for stimulus generation and specialized neuroimaging analysis
- SPM (Statistical Parametric Mapping) for fMRI analysis
- EEGLAB/MNE-Python for EEG processing
- DCM (Dynamic Causal Modeling) for effective connectivity analysis
- TensorFlow/PyTorch for deep learning-based pattern recognition
- Docker/Singularity for containerization
