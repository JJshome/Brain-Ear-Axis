/**
 * BrainEarDashboard.jsx
 * 
 * Main dashboard component for the Brain-Ear Axis platform that integrates
 * all visualization components and provides a complete user interface for
 * data exploration and analysis.
 */

import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Tabs, Tab, Alert, Button } from 'react-bootstrap';
import ConnectivityNetwork from '../visualization/connectivity_network';
import TimeSeries from '../visualization/time_series';
import CorrelationMatrix from '../visualization/correlation_matrix';
import CollaborationPanel from './CollaborationPanel';
import PatientSelector from './PatientSelector';
import AnalysisControls from './AnalysisControls';
import ResultsSummary from './ResultsSummary';

// Mock API service - would be replaced with real API calls in production
const mockApiService = {
  getPatients: () => {
    return Promise.resolve([
      { id: 'patient-001', name: 'John Doe', age: 45, gender: 'Male' },
      { id: 'patient-002', name: 'Jane Smith', age: 32, gender: 'Female' },
      { id: 'patient-003', name: 'Robert Johnson', age: 58, gender: 'Male' }
    ]);
  },
  
  getNeuralData: (patientId) => {
    // Generate mock time series data
    const timePoints = Array.from({ length: 500 }, (_, i) => i * 0.01);
    const channels = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'];
    
    const values = {};
    channels.forEach(channel => {
      // Generate different patterns for each channel
      const amplitude = Math.random() * 30 + 10;
      const frequency = Math.random() * 5 + 1;
      const phase = Math.random() * Math.PI * 2;
      
      values[channel] = timePoints.map(t => 
        amplitude * Math.sin(2 * Math.PI * frequency * t + phase) + 
        Math.random() * 5 - 2.5  // Add some noise
      );
    });
    
    return Promise.resolve({
      timePoints,
      channels,
      values,
      metadata: {
        recordingDate: '2025-03-01',
        samplingRate: 100,
        protocol: 'Resting State EEG'
      }
    });
  },
  
  getMicrobiomeData: (patientId) => {
    // Generate mock microbiome data
    const taxa = [
      'Akkermansia muciniphila',
      'Bacteroides fragilis',
      'Bifidobacterium longum',
      'Escherichia coli',
      'Faecalibacterium prausnitzii',
      'Lactobacillus reuteri',
      'Prevotella copri',
      'Roseburia intestinalis'
    ];
    
    const abundances = taxa.map(() => Math.random());
    const sum = abundances.reduce((a, b) => a + b, 0);
    const normalizedAbundances = abundances.map(a => a / sum);
    
    return Promise.resolve({
      taxa,
      abundances: normalizedAbundances,
      metadata: {
        collectionDate: '2025-02-28',
        sequencingMethod: '16S rRNA',
        diversityIndices: {
          shannon: Math.random() * 3 + 2,
          simpson: Math.random() * 0.5 + 0.5
        }
      }
    });
  },
  
  getConnectivityData: (patientId) => {
    // Generate mock connectivity network data
    const regions = [
      'Frontal_L', 'Frontal_R', 
      'Temporal_L', 'Temporal_R', 
      'Parietal_L', 'Parietal_R', 
      'Occipital_L', 'Occipital_R',
      'Cerebellum', 'Hippocampus',
      'Auditory_Cortex_L', 'Auditory_Cortex_R'
    ];
    
    const nodes = regions.map(region => ({
      id: region,
      name: region.replace('_', ' '),
      weight: Math.random() * 5 + 1,
      color: region.includes('Auditory') ? '#ff7f0e' : '#1f77b4'
    }));
    
    const links = [];
    
    // Create links with random correlation values
    for (let i = 0; i < regions.length; i++) {
      for (let j = i + 1; j < regions.length; j++) {
        // Generate a correlation value between -1 and 1
        // Make auditory regions have stronger connections
        let value;
        if (regions[i].includes('Auditory') || regions[j].includes('Auditory')) {
          value = Math.random() * 1.2 - 0.2; // -0.2 to 1.0, biased toward positive
        } else {
          value = Math.random() * 2 - 1; // -1 to 1, uniform
        }
        
        links.push({
          source: regions[i],
          target: regions[j],
          value: parseFloat(value.toFixed(3))
        });
      }
    }
    
    return Promise.resolve({
      nodes,
      links,
      metadata: {
        analysisDate: '2025-03-02',
        method: 'Phase Locking Value',
        frequencyBand: 'Alpha (8-12 Hz)'
      }
    });
  },
  
  getCorrelationData: (patientId) => {
    // Generate mock correlation data between neural, microbiome, and auditory features
    
    // Define features
    const neuralFeatures = ['Alpha_Power', 'Beta_Power', 'Theta_Power', 'Delta_Power', 'Gamma_Power'];
    const microbiomeFeatures = ['Akkermansia', 'Bacteroides', 'Bifidobacterium', 'Lactobacillus', 'Prevotella'];
    const auditoryFeatures = ['Hearing_Threshold', 'Speech_Recognition', 'Auditory_ERP_P300', 'ABR_Wave_I', 'ABR_Wave_V'];
    
    const allFeatures = [
      ...neuralFeatures.map(name => ({ name, modality: 'neural', cluster: 1 })),
      ...microbiomeFeatures.map(name => ({ name, modality: 'microbiome', cluster: 2 })),
      ...auditoryFeatures.map(name => ({ name, modality: 'auditory', cluster: 3 }))
    ];
    
    // Generate correlations
    const correlations = [];
    
    // Add correlations between all pairs of features
    for (let i = 0; i < allFeatures.length; i++) {
      for (let j = i + 1; j < allFeatures.length; j++) {
        // Generate correlation value
        // Bias for stronger correlations within same modality
        let value;
        
        if (allFeatures[i].modality === allFeatures[j].modality) {
          value = Math.random() * 0.8 + 0.2; // 0.2 to 1.0, strong positive bias
        } 
        // Bias for stronger neural-auditory correlations
        else if (
          (allFeatures[i].modality === 'neural' && allFeatures[j].modality === 'auditory') ||
          (allFeatures[i].modality === 'auditory' && allFeatures[j].modality === 'neural')
        ) {
          value = Math.random() * 0.7 + 0.1; // 0.1 to 0.8, moderate positive bias
        }
        // Bias for neural-microbiome correlations
        else if (
          (allFeatures[i].modality === 'neural' && allFeatures[j].modality === 'microbiome') ||
          (allFeatures[i].modality === 'microbiome' && allFeatures[j].modality === 'neural')
        ) {
          value = Math.random() * 0.6 - 0.3; // -0.3 to 0.3, both positive and negative
        }
        // Bias for microbiome-auditory correlations
        else {
          value = Math.random() * 0.5 - 0.2; // -0.2 to 0.3, slight positive bias
        }
        
        // Add p-value
        const pvalue = Math.random() * 0.1;
        
        correlations.push({
          source: allFeatures[i].name,
          target: allFeatures[j].name,
          value: parseFloat(value.toFixed(3)),
          pvalue: parseFloat(pvalue.toFixed(3))
        });
      }
    }
    
    return Promise.resolve({
      features: allFeatures,
      correlations,
      metadata: {
        analysisDate: '2025-03-03',
        method: 'Spearman Correlation',
        multipleTestingCorrection: 'FDR'
      }
    });
  },
  
  getAnalysisResults: (patientId) => {
    // Generate mock analysis results
    return Promise.resolve({
      summary: {
        neuralFindings: 'Increased alpha power in temporal regions',
        microbiomeFindings: 'High abundance of Akkermansia muciniphila',
        auditoryFindings: 'Mild high-frequency hearing loss',
        correlationalFindings: 'Strong association between microbiome diversity and auditory processing'
      },
      statistics: {
        neuralMetrics: {
          alphaPower: { value: 8.3, percentile: 65, status: 'normal' },
          betaPower: { value: 5.2, percentile: 45, status: 'normal' },
          thetaPower: { value: 4.1, percentile: 40, status: 'normal' }
        },
        microbiomeMetrics: {
          diversity: { value: 3.2, percentile: 75, status: 'high' },
          firmicutesBacteroidetesRatio: { value: 1.8, percentile: 60, status: 'normal' }
        },
        auditoryMetrics: {
          hearingThreshold: { value: 35, percentile: 30, status: 'mild_loss' },
          speechRecognition: { value: 85, percentile: 50, status: 'normal' }
        }
      },
      recommendations: [
        'Continue monitoring hearing function',
        'Consider probiotic supplementation',
        'Follow-up in 6 months'
      ]
    });
  },
  
  runAnalysis: (patientId, options) => {
    // Mock running an analysis
    return new Promise(resolve => {
      setTimeout(() => {
        resolve(mockApiService.getAnalysisResults(patientId));
      }, 2000); // Simulate a delay
    });
  }
};

// Main dashboard component
const BrainEarDashboard = () => {
  const [loading, setLoading] = useState(true);
  const [patient, setPatient] = useState(null);
  const [patients, setPatients] = useState([]);
  const [neuralData, setNeuralData] = useState(null);
  const [microbiomeData, setMicrobiomeData] = useState(null);
  const [connectivityData, setConnectivityData] = useState(null);
  const [correlationData, setCorrelationData] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [error, setError] = useState(null);
  
  // Load patient list on component mount
  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await mockApiService.getPatients();
        setPatients(response);
        // Auto-select first patient
        if (response.length > 0) {
          setPatient(response[0]);
        }
      } catch (error) {
        setError('Error loading patients: ' + error.message);
      }
    };
    
    fetchPatients();
  }, []);
  
  // Load data when patient changes
  useEffect(() => {
    if (!patient) return;
    
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Fetch all data types in parallel
        const [neuralResponse, microbiomeResponse, connectivityResponse, correlationResponse, resultsResponse] = 
          await Promise.all([
            mockApiService.getNeuralData(patient.id),
            mockApiService.getMicrobiomeData(patient.id),
            mockApiService.getConnectivityData(patient.id),
            mockApiService.getCorrelationData(patient.id),
            mockApiService.getAnalysisResults(patient.id)
          ]);
        
        setNeuralData(neuralResponse);
        setMicrobiomeData(microbiomeResponse);
        setConnectivityData(connectivityResponse);
        setCorrelationData(correlationResponse);
        setAnalysisResults(resultsResponse);
      } catch (error) {
        setError('Error loading data: ' + error.message);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [patient]);
  
  // Handle patient change
  const handlePatientChange = (patientId) => {
    const selectedPatient = patients.find(p => p.id === patientId);
    setPatient(selectedPatient);
  };
  
  // Handle analysis submission
  const handleAnalysisSubmit = async (options) => {
    if (!patient) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await mockApiService.runAnalysis(patient.id, options);
      setAnalysisResults(response);
      // Switch to results tab
      setActiveTab('results');
    } catch (error) {
      setError('Error running analysis: ' + error.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle filter changes from collaboration panel
  const handleCollaborationFilterChange = (filter) => {
    // This would apply filters across the dashboard
    console.log('Applying filter from collaboration:', filter);
  };
  
  return (
    <Container fluid className="py-4">
      <Row className="mb-4">
        <Col>
          <h1 className="display-5">Brain-Ear Axis Platform</h1>
          <p className="lead">Integrated analysis of neural, microbiome, and auditory data</p>
        </Col>
      </Row>
      
      {error && (
        <Row className="mb-4">
          <Col>
            <Alert variant="danger" onClose={() => setError(null)} dismissible>
              {error}
            </Alert>
          </Col>
        </Row>
      )}
      
      <Row className="mb-4">
        <Col md={4}>
          <PatientSelector 
            patients={patients}
            selectedPatient={patient}
            onPatientChange={handlePatientChange}
          />
        </Col>
        <Col md={8}>
          <AnalysisControls 
            onSubmit={handleAnalysisSubmit}
            disabled={loading || !patient}
          />
        </Col>
      </Row>
      
      {loading ? (
        <Row>
          <Col className="text-center py-5">
            <div className="spinner-border" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
            <p className="mt-3">Loading data...</p>
          </Col>
        </Row>
      ) : patient ? (
        <Row>
          <Col md={9}>
            <Card>
              <Card.Header>
                <Tabs
                  activeKey={activeTab}
                  onSelect={(key) => setActiveTab(key)}
                  className="mb-0"
                >
                  <Tab eventKey="overview" title="Overview">
                  </Tab>
                  <Tab eventKey="neural" title="Neural Analysis">
                  </Tab>
                  <Tab eventKey="microbiome" title="Microbiome Analysis">
                  </Tab>
                  <Tab eventKey="auditory" title="Auditory Analysis">
                  </Tab>
                  <Tab eventKey="correlation" title="Correlation Analysis">
                  </Tab>
                  <Tab eventKey="results" title="Results">
                  </Tab>
                </Tabs>
              </Card.Header>
              <Card.Body>
                {activeTab === 'overview' && (
                  <Row>
                    <Col md={6} className="mb-4">
                      <Card className="h-100">
                        <Card.Header>Neural Activity</Card.Header>
                        <Card.Body>
                          {neuralData && (
                            <TimeSeries 
                              data={neuralData} 
                              height={300}
                              defaultVisualization="line"
                            />
                          )}
                        </Card.Body>
                      </Card>
                    </Col>
                    
                    <Col md={6} className="mb-4">
                      <Card className="h-100">
                        <Card.Header>Brain-Ear Connectivity</Card.Header>
                        <Card.Body>
                          {connectivityData && (
                            <ConnectivityNetwork 
                              data={connectivityData} 
                              height={300}
                              showLabels={true}
                            />
                          )}
                        </Card.Body>
                      </Card>
                    </Col>
                    
                    <Col md={12}>
                      <Card>
                        <Card.Header>Neural-Microbiome-Auditory Correlations</Card.Header>
                        <Card.Body>
                          {correlationData && (
                            <CorrelationMatrix 
                              data={correlationData} 
                              height={500}
                              defaultGrouping="modality"
                            />
                          )}
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                )}
                
                {activeTab === 'neural' && (
                  <Row>
                    <Col md={12}>
                      <div className="pb-3">
                        <h4>Neural Time Series Analysis</h4>
                        <p>
                          Visualize and analyze neural time series data, including EEG/MEG recordings 
                          and derived measures.
                        </p>
                      </div>
                      
                      {neuralData && (
                        <TimeSeries 
                          data={neuralData} 
                          height={600}
                          defaultVisualization="line"
                        />
                      )}
                    </Col>
                  </Row>
                )}
                
                {activeTab === 'correlation' && (
                  <Row>
                    <Col md={12}>
                      <div className="pb-3">
                        <h4>Multi-Modal Correlation Analysis</h4>
                        <p>
                          Explore correlations between neural, microbiome, and auditory features to 
                          discover potential connections and patterns.
                        </p>
                      </div>
                      
                      {correlationData && (
                        <CorrelationMatrix 
                          data={correlationData} 
                          height={700}
                          defaultGrouping="modality"
                        />
                      )}
                    </Col>
                  </Row>
                )}
                
                {activeTab === 'results' && (
                  <Row>
                    <Col md={12}>
                      <div className="pb-3">
                        <h4>Analysis Results</h4>
                        <p>
                          Summary of integrated analysis findings for neural, microbiome, and auditory data.
                        </p>
                      </div>
                      
                      {analysisResults && (
                        <ResultsSummary 
                          data={analysisResults}
                        />
                      )}
                    </Col>
                  </Row>
                )}
              </Card.Body>
            </Card>
          </Col>
          
          <Col md={3}>
            <CollaborationPanel 
              patientId={patient.id}
              onFilterChange={handleCollaborationFilterChange}
            />
          </Col>
        </Row>
      ) : (
        <Row>
          <Col className="text-center py-5">
            <Alert variant="info">
              Please select a patient to view data
            </Alert>
          </Col>
        </Row>
      )}
      
      <Row className="mt-4">
        <Col>
          <footer className="border-top pt-3 text-muted small">
            <p>© 2025 Brain-Ear Axis Platform | <a href="#">Documentation</a> | <a href="#">Support</a></p>
          </footer>
        </Col>
      </Row>
    </Container>
  );
};

export default BrainEarDashboard;
