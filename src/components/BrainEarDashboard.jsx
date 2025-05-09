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
    // Generate mock connectivity data
    const regions = [
      'Temporal_L', 'Temporal_R', 
      'Frontal_L', 'Frontal_R', 
      'Parietal_L', 'Parietal_R', 
      'Occipital_L', 'Occipital_R',
      'Insula_L', 'Insula_R',
      'Cingulate_Anterior', 'Cingulate_Posterior'
    ];
    
    // Create nodes
    const nodes = regions.map((region, i) => ({
      id: region,
      name: region.replace('_', ' '),
      weight: Math.random() * 3 + 1,
      color: region.includes('Temporal') ? '#e41a1c' : 
             region.includes('Frontal') ? '#377eb8' : 
             region.includes('Parietal') ? '#4daf4a' : 
             region.includes('Occipital') ? '#984ea3' :
             region.includes('Insula') ? '#ff7f00' : 
             '#a65628'
    }));
    
    // Create links (connections between regions)
    const links = [];
    for (let i = 0; i < regions.length; i++) {
      for (let j = i + 1; j < regions.length; j++) {
        // Create connections with different strengths
        // Higher probability of connection for regions in the same hemisphere or area
        const sameHemisphere = 
          (regions[i].includes('_L') && regions[j].includes('_L')) || 
          (regions[i].includes('_R') && regions[j].includes('_R'));
        
        const sameArea = 
          regions[i].split('_')[0] === regions[j].split('_')[0];
        
        const connectionProbability = sameArea ? 0.9 : sameHemisphere ? 0.7 : 0.3;
        
        if (Math.random() < connectionProbability) {
          // Create connection with random strength
          links.push({
            source: regions[i],
            target: regions[j],
            value: (Math.random() * 2 - 1) * (sameArea ? 0.8 : 0.5)  // Stronger for same area
          });
        }
      }
    }
    
    return Promise.resolve({ nodes, links });
  },
  
  getCorrelationData: (patientId) => {
    // Generate mock correlation data between neural, microbiome, and auditory features
    
    // Define features from different modalities
    const features = [
      // Neural features
      { name: 'Alpha_Temporal_L', modality: 'neural', description: 'Alpha power in left temporal region', cluster: 1 },
      { name: 'Alpha_Temporal_R', modality: 'neural', description: 'Alpha power in right temporal region', cluster: 1 },
      { name: 'Beta_Frontal_L', modality: 'neural', description: 'Beta power in left frontal region', cluster: 2 },
      { name: 'Beta_Frontal_R', modality: 'neural', description: 'Beta power in right frontal region', cluster: 2 },
      { name: 'Theta_Frontal', modality: 'neural', description: 'Theta power in frontal region', cluster: 3 },
      { name: 'Gamma_Temporal', modality: 'neural', description: 'Gamma power in temporal region', cluster: 3 },
      
      // Microbiome features
      { name: 'Akkermansia', modality: 'microbiome', description: 'Abundance of Akkermansia muciniphila', cluster: 4 },
      { name: 'Bacteroides', modality: 'microbiome', description: 'Abundance of Bacteroides fragilis', cluster: 4 },
      { name: 'Bifidobacterium', modality: 'microbiome', description: 'Abundance of Bifidobacterium longum', cluster: 4 },
      { name: 'Shannon_Index', modality: 'microbiome', description: 'Shannon diversity index', cluster: 5 },
      { name: 'Firmicutes_Bacteroidetes', modality: 'microbiome', description: 'Firmicutes to Bacteroidetes ratio', cluster: 5 },
      
      // Auditory features
      { name: 'PTA_Left', modality: 'auditory', description: 'Pure Tone Average, left ear', cluster: 6 },
      { name: 'PTA_Right', modality: 'auditory', description: 'Pure Tone Average, right ear', cluster: 6 },
      { name: 'SRT_Left', modality: 'auditory', description: 'Speech Reception Threshold, left ear', cluster: 6 },
      { name: 'SRT_Right', modality: 'auditory', description: 'Speech Reception Threshold, right ear', cluster: 6 },
      { name: 'ABR_Wave_I', modality: 'auditory', description: 'Auditory Brainstem Response, Wave I', cluster: 7 },
      { name: 'ABR_Wave_V', modality: 'auditory', description: 'Auditory Brainstem Response, Wave V', cluster: 7 }
    ];
    
    // Generate correlations between features
    const correlations = [];
    
    for (let i = 0; i < features.length; i++) {
      for (let j = i + 1; j < features.length; j++) {
        // Higher probability of correlation for features in the same modality or cluster
        const sameModality = features[i].modality === features[j].modality;
        const sameCluster = features[i].cluster === features[j].cluster;
        
        const correlationProbability = sameCluster ? 0.9 : sameModality ? 0.7 : 0.4;
        
        if (Math.random() < correlationProbability) {
          // Create correlation with random value
          const correlation = {
            source: features[i].name,
            target: features[j].name,
            value: (Math.random() * 2 - 1) * (sameCluster ? 0.8 : sameModality ? 0.6 : 0.4),
            pvalue: Math.random() * 0.1  // Small p-values for significant correlations
          };
          
          correlations.push(correlation);
        }
      }
    }
    
    return Promise.resolve({ features, correlations });
  },
  
  getAnalysisResults: (patientId) => {
    // Generate mock analysis results
    return Promise.resolve({
      status: 'completed',
      date: '2025-03-05',
      summary: {
        neuralFindings: 'Increased alpha power in temporal regions',
        microbiomeFindings: 'Elevated Akkermansia abundance, associated with gut health',
        auditoryFindings: 'Mild hearing loss in high frequencies',
        correlations: [
          'Strong correlation between Alpha_Temporal and ABR_Wave_V (r=0.76, p<0.01)',
          'Moderate correlation between Akkermansia and Theta_Frontal (r=0.42, p<0.05)',
          'Inverse correlation between PTA scores and Bifidobacterium (r=-0.38, p<0.05)'
        ]
      },
      recommendations: [
        'Consider follow-up auditory testing in 3 months',
        'Potential benefit from probiotics to support gut-brain axis',
        'Recommend detailed audiological assessment'
      ]
    });
  },
  
  runAnalysis: (patientId, options) => {
    // Simulate running an analysis
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          status: 'completed',
          date: new Date().toISOString(),
          summary: {
            neuralFindings: 'Custom analysis found abnormal beta oscillations',
            microbiomeFindings: 'Reduced diversity in gut microbiome composition',
            auditoryFindings: 'Asymmetry in auditory processing between left and right',
            correlations: [
              'New significant correlation between Beta_Frontal and PTA scores (r=0.58, p<0.01)',
              'Relationship between Shannon diversity and ABR latencies (r=-0.44, p<0.05)'
            ]
          },
          recommendations: [
            'Additional neuroimaging recommended',
            'Consider auditory training exercises',
            'Dietary modifications to improve gut microbiome diversity'
          ]
        });
      }, 2000);  // Simulate 2 seconds of processing time
    });
  }
};

// Main Dashboard Component
const BrainEarDashboard = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [patient, setPatient] = useState(null);
  const [patients, setPatients] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');
  
  // Data states
  const [neuralData, setNeuralData] = useState(null);
  const [microbiomeData, setMicrobiomeData] = useState(null);
  const [connectivityData, setConnectivityData] = useState(null);
  const [correlationData, setCorrelationData] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  
  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // Load patients on component mount
  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const patientsData = await mockApiService.getPatients();
        setPatients(patientsData);
        
        // Auto-select first patient
        if (patientsData.length > 0) {
          setPatient(patientsData[0]);
        }
        
        setLoading(false);
      } catch (err) {
        setError('Failed to load patients data');
        setLoading(false);
      }
    };
    
    fetchPatients();
  }, []);
  
  // Load patient data when patient changes
  useEffect(() => {
    if (!patient) return;
    
    const fetchPatientData = async () => {
      setLoading(true);
      try {
        // Fetch all data types in parallel
        const [
          neuralResponse, 
          microbiomeResponse, 
          connectivityResponse, 
          correlationResponse,
          analysisResultsResponse
        ] = await Promise.all([
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
        setAnalysisResults(analysisResultsResponse);
        setLoading(false);
      } catch (err) {
        setError('Failed to load patient data');
        setLoading(false);
      }
    };
    
    fetchPatientData();
  }, [patient]);
  
  // Handle patient change
  const handlePatientChange = (patientId) => {
    const selectedPatient = patients.find(p => p.id === patientId);
    setPatient(selectedPatient);
  };
  
  // Run analysis
  const handleRunAnalysis = async (options) => {
    if (!patient) return;
    
    setIsAnalyzing(true);
    try {
      const results = await mockApiService.runAnalysis(patient.id, options);
      setAnalysisResults(results);
      setActiveTab('results');
    } catch (err) {
      setError('Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  // Handle collaboration events
  const handleAnnotationAdd = (annotation) => {
    // Would typically update the analysis results or add a marker in the visualizations
    console.log('Annotation added:', annotation);
  };
  
  return (
    <Container fluid className="brain-ear-dashboard">
      <Row className="mb-4">
        <Col md={8}>
          <h1 className="mb-3">Brain-Ear Axis Analysis Platform</h1>
        </Col>
        <Col md={4} className="text-end">
          <PatientSelector 
            patients={patients}
            selectedPatient={patient}
            onPatientChange={handlePatientChange}
          />
        </Col>
      </Row>
      
      {error && (
        <Alert variant="danger" onClose={() => setError(null)} dismissible>
          {error}
        </Alert>
      )}
      
      {loading ? (
        <div className="text-center my-5">
          <div className="spinner-border" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <p className="mt-2">Loading data...</p>
        </div>
      ) : patient ? (
        <Row>
          <Col md={9}>
            <Card className="mb-4">
              <Card.Header>
                <div className="d-flex justify-content-between align-items-center">
                  <h3 className="m-0">
                    {patient.name} <small className="text-muted">{patient.age}y, {patient.gender}</small>
                  </h3>
                  <AnalysisControls 
                    onSubmit={handleRunAnalysis}
                    disabled={isAnalyzing}
                    isLoading={isAnalyzing}
                  />
                </div>
              </Card.Header>
              
              <Card.Body>
                <Tabs
                  activeKey={activeTab}
                  onSelect={(key) => setActiveTab(key)}
                  className="mb-4"
                >
                  <Tab eventKey="overview" title="Overview">
                    <Row>
                      <Col md={6}>
                        <Card className="mb-4">
                          <Card.Header>Neural Activity</Card.Header>
                          <Card.Body style={{ height: '300px' }}>
                            {neuralData && (
                              <TimeSeries 
                                data={neuralData} 
                                height={250}
                                defaultVisualization="line"
                              />
                            )}
                          </Card.Body>
                        </Card>
                      </Col>
                      
                      <Col md={6}>
                        <Card className="mb-4">
                          <Card.Header>Brain-Ear Connectivity</Card.Header>
                          <Card.Body style={{ height: '300px' }}>
                            {connectivityData && (
                              <ConnectivityNetwork 
                                data={connectivityData} 
                                width={400}
                                height={250}
                              />
                            )}
                          </Card.Body>
                        </Card>
                      </Col>
                    </Row>
                    
                    <Card className="mb-4">
                      <Card.Header>Cross-Modal Correlations</Card.Header>
                      <Card.Body style={{ height: '400px' }}>
                        {correlationData && (
                          <CorrelationMatrix 
                            data={correlationData}
                            width={800}
                            height={350}
                          />
                        )}
                      </Card.Body>
                    </Card>
                  </Tab>
                  
                  <Tab eventKey="neural" title="Neural Data">
                    <Card>
                      <Card.Header>
                        <div className="d-flex justify-content-between align-items-center">
                          <h5 className="m-0">Neural Time Series</h5>
                          <div className="text-muted small">
                            {neuralData?.metadata?.protocol} | {neuralData?.metadata?.recordingDate}
                          </div>
                        </div>
                      </Card.Header>
                      <Card.Body style={{ height: '600px' }}>
                        {neuralData && (
                          <TimeSeries 
                            data={neuralData} 
                            height={550}
                            defaultVisualization="line"
                          />
                        )}
                      </Card.Body>
                    </Card>
                  </Tab>
                  
                  <Tab eventKey="connectivity" title="Connectivity">
                    <Card>
                      <Card.Header>
                        <h5 className="m-0">Brain Region Connectivity</h5>
                      </Card.Header>
                      <Card.Body style={{ height: '600px' }}>
                        {connectivityData && (
                          <ConnectivityNetwork 
                            data={connectivityData} 
                            width={800}
                            height={550}
                          />
                        )}
                      </Card.Body>
                    </Card>
                  </Tab>
                  
                  <Tab eventKey="correlation" title="Correlations">
                    <Card>
                      <Card.Header>
                        <h5 className="m-0">Multi-Modal Feature Correlations</h5>
                      </Card.Header>
                      <Card.Body style={{ height: '600px' }}>
                        {correlationData && (
                          <CorrelationMatrix 
                            data={correlationData}
                            width={800}
                            height={550}
                          />
                        )}
                      </Card.Body>
                    </Card>
                  </Tab>
                  
                  <Tab eventKey="results" title="Results">
                    <Card>
                      <Card.Header>
                        <div className="d-flex justify-content-between align-items-center">
                          <h5 className="m-0">Analysis Results</h5>
                          <div className="text-muted small">
                            {analysisResults?.date && new Date(analysisResults.date).toLocaleDateString()}
                          </div>
                        </div>
                      </Card.Header>
                      <Card.Body>
                        {analysisResults ? (
                          <ResultsSummary results={analysisResults} />
                        ) : (
                          <div className="text-center text-muted my-5">
                            <p>No analysis results available</p>
                            <Button 
                              variant="primary" 
                              onClick={() => setActiveTab('overview')}
                            >
                              Return to Overview
                            </Button>
                          </div>
                        )}
                      </Card.Body>
                    </Card>
                  </Tab>
                </Tabs>
              </Card.Body>
            </Card>
          </Col>
          
          <Col md={3}>
            <CollaborationPanel 
              patientId={patient.id} 
              onAnnotationAdd={handleAnnotationAdd}
            />
          </Col>
        </Row>
      ) : (
        <div className="text-center my-5">
          <Alert variant="info">
            Please select a patient to view data
          </Alert>
        </div>
      )}
    </Container>
  );
};

export default BrainEarDashboard;
