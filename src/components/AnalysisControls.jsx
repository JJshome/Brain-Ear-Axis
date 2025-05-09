/**
 * AnalysisControls.jsx
 * 
 * Component for configuring and running analyses with various options
 */

import React, { useState } from 'react';
import { Button, Modal, Form, Spinner } from 'react-bootstrap';
import PropTypes from 'prop-types';

const AnalysisControls = ({ onSubmit, disabled, isLoading }) => {
  const [showModal, setShowModal] = useState(false);
  const [options, setOptions] = useState({
    analysisType: 'comprehensive',
    includeNeural: true,
    includeMicrobiome: true,
    includeAuditory: true,
    advancedOptions: {
      normalizeData: true,
      multipleTestingCorrection: 'fdr',
      significanceThreshold: 0.05,
      bootstrapIterations: 1000
    }
  });
  
  const handleOpenModal = () => {
    setShowModal(true);
  };
  
  const handleCloseModal = () => {
    setShowModal(false);
  };
  
  const handleOptionChange = (e) => {
    const { name, value, type, checked } = e.target;
    setOptions({
      ...options,
      [name]: type === 'checkbox' ? checked : value
    });
  };
  
  const handleAdvancedOptionChange = (e) => {
    const { name, value, type, checked } = e.target;
    setOptions({
      ...options,
      advancedOptions: {
        ...options.advancedOptions,
        [name]: type === 'checkbox' ? checked : value
      }
    });
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(options);
    handleCloseModal();
  };
  
  return (
    <>
      <Button 
        variant="primary" 
        onClick={handleOpenModal}
        disabled={disabled}
      >
        {isLoading ? (
          <>
            <Spinner
              as="span"
              animation="border"
              size="sm"
              role="status"
              aria-hidden="true"
              className="me-2"
            />
            Running Analysis...
          </>
        ) : (
          'Run Analysis'
        )}
      </Button>
      
      <Modal show={showModal} onHide={handleCloseModal} size="lg">
        <Form onSubmit={handleSubmit}>
          <Modal.Header closeButton>
            <Modal.Title>Configure Analysis</Modal.Title>
          </Modal.Header>
          
          <Modal.Body>
            <Form.Group className="mb-3">
              <Form.Label>Analysis Type</Form.Label>
              <Form.Select 
                name="analysisType"
                value={options.analysisType}
                onChange={handleOptionChange}
              >
                <option value="comprehensive">Comprehensive Analysis</option>
                <option value="neural">Neural Analysis</option>
                <option value="microbiome">Microbiome Analysis</option>
                <option value="auditory">Auditory Analysis</option>
                <option value="correlation">Correlation Analysis</option>
                <option value="custom">Custom Analysis</option>
              </Form.Select>
              <Form.Text className="text-muted">
                Select the type of analysis to perform on the patient data
              </Form.Text>
            </Form.Group>
            
            <Form.Group className="mb-3">
              <Form.Label>Data Modalities</Form.Label>
              <div className="d-flex flex-column">
                <Form.Check 
                  type="checkbox"
                  id="include-neural"
                  label="Include Neural Data"
                  name="includeNeural"
                  checked={options.includeNeural}
                  onChange={handleOptionChange}
                />
                <Form.Check 
                  type="checkbox"
                  id="include-microbiome"
                  label="Include Microbiome Data"
                  name="includeMicrobiome"
                  checked={options.includeMicrobiome}
                  onChange={handleOptionChange}
                />
                <Form.Check 
                  type="checkbox"
                  id="include-auditory"
                  label="Include Auditory Data"
                  name="includeAuditory"
                  checked={options.includeAuditory}
                  onChange={handleOptionChange}
                />
              </div>
            </Form.Group>
            
            <h5 className="mt-4">Advanced Options</h5>
            <hr />
            
            <Form.Group className="mb-3">
              <Form.Check 
                type="checkbox"
                id="normalize-data"
                label="Normalize Data"
                name="normalizeData"
                checked={options.advancedOptions.normalizeData}
                onChange={handleAdvancedOptionChange}
              />
              <Form.Text className="text-muted">
                Apply normalization to all data modalities before analysis
              </Form.Text>
            </Form.Group>
            
            <Form.Group className="mb-3">
              <Form.Label>Multiple Testing Correction</Form.Label>
              <Form.Select 
                name="multipleTestingCorrection"
                value={options.advancedOptions.multipleTestingCorrection}
                onChange={handleAdvancedOptionChange}
              >
                <option value="fdr">False Discovery Rate (FDR)</option>
                <option value="bonferroni">Bonferroni Correction</option>
                <option value="holm">Holm-Bonferroni Method</option>
                <option value="none">No Correction</option>
              </Form.Select>
            </Form.Group>
            
            <Form.Group className="mb-3">
              <Form.Label>
                Significance Threshold (p-value): {options.advancedOptions.significanceThreshold}
              </Form.Label>
              <Form.Range 
                name="significanceThreshold"
                min={0.001}
                max={0.1}
                step={0.001}
                value={options.advancedOptions.significanceThreshold}
                onChange={handleAdvancedOptionChange}
              />
            </Form.Group>
            
            <Form.Group className="mb-3">
              <Form.Label>Bootstrap Iterations</Form.Label>
              <Form.Select 
                name="bootstrapIterations"
                value={options.advancedOptions.bootstrapIterations}
                onChange={handleAdvancedOptionChange}
              >
                <option value={100}>100 (Faster, less accurate)</option>
                <option value={500}>500</option>
                <option value={1000}>1000 (Recommended)</option>
                <option value={5000}>5000 (Slower, more accurate)</option>
                <option value={10000}>10000 (Very slow, highest accuracy)</option>
              </Form.Select>
            </Form.Group>
          </Modal.Body>
          
          <Modal.Footer>
            <Button variant="secondary" onClick={handleCloseModal}>
              Cancel
            </Button>
            <Button variant="primary" type="submit" disabled={isLoading}>
              {isLoading ? 'Running...' : 'Run Analysis'}
            </Button>
          </Modal.Footer>
        </Form>
      </Modal>
    </>
  );
};

AnalysisControls.propTypes = {
  onSubmit: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
  isLoading: PropTypes.bool
};

AnalysisControls.defaultProps = {
  disabled: false,
  isLoading: false
};

export default AnalysisControls;
