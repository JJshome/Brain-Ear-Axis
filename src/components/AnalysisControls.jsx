/**
 * AnalysisControls.jsx
 * 
 * A component for configuring and launching various analyses on the data.
 */

import React, { useState } from 'react';
import { Card, Form, Button, Row, Col, Accordion } from 'react-bootstrap';
import PropTypes from 'prop-types';

const AnalysisControls = ({ onSubmit, disabled }) => {
  const [options, setOptions] = useState({
    analysisType: 'comprehensive',
    includeNeural: true,
    includeMicrobiome: true,
    includeAuditory: true,
    correlationMethod: 'spearman',
    significanceThreshold: 0.05,
    multipleTestingCorrection: 'fdr',
    clusteringMethod: 'hierarchical',
    numberOfClusters: 3
  });
  
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setOptions({
      ...options,
      [name]: type === 'checkbox' ? checked : value
    });
  };
  
  const handleNumberChange = (e) => {
    const { name, value } = e.target;
    setOptions({
      ...options,
      [name]: parseInt(value, 10)
    });
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(options);
  };
  
  return (
    <Card>
      <Card.Header>
        <h5 className="mb-0">Analysis Controls</h5>
      </Card.Header>
      <Card.Body>
        <Form onSubmit={handleSubmit}>
          <Row className="mb-3">
            <Col md={6}>
              <Form.Group>
                <Form.Label>Analysis Type</Form.Label>
                <Form.Select
                  name="analysisType"
                  value={options.analysisType}
                  onChange={handleChange}
                  disabled={disabled}
                >
                  <option value="comprehensive">Comprehensive Analysis</option>
                  <option value="neural">Neural Analysis</option>
                  <option value="microbiome">Microbiome Analysis</option>
                  <option value="auditory">Auditory Analysis</option>
                  <option value="correlation">Correlation Analysis</option>
                </Form.Select>
              </Form.Group>
            </Col>
            <Col md={6}>
              <Form.Group>
                <Form.Label>Data Types to Include</Form.Label>
                <div className="d-flex">
                  <Form.Check
                    type="checkbox"
                    label="Neural"
                    name="includeNeural"
                    id="includeNeural"
                    checked={options.includeNeural}
                    onChange={handleChange}
                    disabled={disabled || options.analysisType === 'neural'}
                    className="me-3"
                  />
                  <Form.Check
                    type="checkbox"
                    label="Microbiome"
                    name="includeMicrobiome"
                    id="includeMicrobiome"
                    checked={options.includeMicrobiome}
                    onChange={handleChange}
                    disabled={disabled || options.analysisType === 'microbiome'}
                    className="me-3"
                  />
                  <Form.Check
                    type="checkbox"
                    label="Auditory"
                    name="includeAuditory"
                    id="includeAuditory"
                    checked={options.includeAuditory}
                    onChange={handleChange}
                    disabled={disabled || options.analysisType === 'auditory'}
                  />
                </div>
              </Form.Group>
            </Col>
          </Row>
          
          <Accordion className="mb-3">
            <Accordion.Item eventKey="0">
              <Accordion.Header>Advanced Settings</Accordion.Header>
              <Accordion.Body>
                <Row>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Correlation Method</Form.Label>
                      <Form.Select
                        name="correlationMethod"
                        value={options.correlationMethod}
                        onChange={handleChange}
                        disabled={disabled}
                      >
                        <option value="pearson">Pearson Correlation</option>
                        <option value="spearman">Spearman Rank Correlation</option>
                        <option value="kendall">Kendall's Tau</option>
                        <option value="distance">Distance Correlation</option>
                      </Form.Select>
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>
                        Significance Threshold (p-value): {options.significanceThreshold}
                      </Form.Label>
                      <Form.Range
                        name="significanceThreshold"
                        min={0.001}
                        max={0.1}
                        step={0.001}
                        value={options.significanceThreshold}
                        onChange={handleChange}
                        disabled={disabled}
                      />
                    </Form.Group>
                  </Col>
                  
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Multiple Testing Correction</Form.Label>
                      <Form.Select
                        name="multipleTestingCorrection"
                        value={options.multipleTestingCorrection}
                        onChange={handleChange}
                        disabled={disabled}
                      >
                        <option value="none">None</option>
                        <option value="bonferroni">Bonferroni</option>
                        <option value="fdr">False Discovery Rate (FDR)</option>
                        <option value="holm">Holm-Bonferroni</option>
                      </Form.Select>
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>Clustering Method</Form.Label>
                      <Form.Select
                        name="clusteringMethod"
                        value={options.clusteringMethod}
                        onChange={handleChange}
                        disabled={disabled}
                      >
                        <option value="hierarchical">Hierarchical Clustering</option>
                        <option value="kmeans">K-Means Clustering</option>
                        <option value="dbscan">DBSCAN</option>
                        <option value="spectral">Spectral Clustering</option>
                      </Form.Select>
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>Number of Clusters</Form.Label>
                      <Form.Control
                        type="number"
                        name="numberOfClusters"
                        value={options.numberOfClusters}
                        onChange={handleNumberChange}
                        min={2}
                        max={10}
                        disabled={disabled || options.clusteringMethod === 'dbscan'}
                      />
                    </Form.Group>
                  </Col>
                </Row>
              </Accordion.Body>
            </Accordion.Item>
          </Accordion>
          
          <div className="d-grid">
            <Button 
              type="submit" 
              variant="primary" 
              size="lg"
              disabled={disabled}
            >
              Run Analysis
            </Button>
          </div>
        </Form>
      </Card.Body>
      {disabled && (
        <Card.Footer className="text-muted">
          Please select a patient to enable analysis options
        </Card.Footer>
      )}
    </Card>
  );
};

AnalysisControls.propTypes = {
  onSubmit: PropTypes.func.isRequired,
  disabled: PropTypes.bool
};

AnalysisControls.defaultProps = {
  disabled: false
};

export default AnalysisControls;
