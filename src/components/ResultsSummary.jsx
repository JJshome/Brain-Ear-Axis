/**
 * ResultsSummary.jsx
 * 
 * Component for displaying analysis results in a structured and readable format
 */

import React from 'react';
import { Card, Row, Col, ListGroup, Badge } from 'react-bootstrap';
import PropTypes from 'prop-types';

const ResultsSummary = ({ results }) => {
  // Format date string
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    } catch (e) {
      return dateString;
    }
  };
  
  // Determine status badge color
  const getStatusBadge = (status) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return <Badge bg="success">Completed</Badge>;
      case 'in_progress':
        return <Badge bg="warning">In Progress</Badge>;
      case 'failed':
        return <Badge bg="danger">Failed</Badge>;
      default:
        return <Badge bg="secondary">{status}</Badge>;
    }
  };
  
  return (
    <div className="results-summary">
      <div className="mb-4">
        <h4>Analysis Status</h4>
        <Card>
          <Card.Body>
            <Row>
              <Col md={4}>
                <div className="text-muted">Status</div>
                <div>{getStatusBadge(results.status)}</div>
              </Col>
              <Col md={4}>
                <div className="text-muted">Date</div>
                <div>{formatDate(results.date)}</div>
              </Col>
              <Col md={4}>
                <div className="text-muted">Analysis ID</div>
                <div>{results.id || 'N/A'}</div>
              </Col>
            </Row>
          </Card.Body>
        </Card>
      </div>
      
      <h4>Findings</h4>
      <Row className="mb-4">
        <Col md={4}>
          <Card className="h-100">
            <Card.Header className="bg-primary text-white">
              Neural Findings
            </Card.Header>
            <Card.Body>
              <p>{results.summary?.neuralFindings || 'No findings available'}</p>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="h-100">
            <Card.Header className="bg-success text-white">
              Microbiome Findings
            </Card.Header>
            <Card.Body>
              <p>{results.summary?.microbiomeFindings || 'No findings available'}</p>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="h-100">
            <Card.Header className="bg-info text-white">
              Auditory Findings
            </Card.Header>
            <Card.Body>
              <p>{results.summary?.auditoryFindings || 'No findings available'}</p>
            </Card.Body>
          </Card>
        </Col>
      </Row>
      
      <h4>Correlations</h4>
      <Card className="mb-4">
        <Card.Body>
          {results.summary?.correlations && results.summary.correlations.length > 0 ? (
            <ListGroup>
              {results.summary.correlations.map((correlation, index) => (
                <ListGroup.Item key={index}>
                  {correlation}
                </ListGroup.Item>
              ))}
            </ListGroup>
          ) : (
            <p className="text-muted">No significant correlations found</p>
          )}
        </Card.Body>
      </Card>
      
      <h4>Recommendations</h4>
      <Card>
        <Card.Body>
          {results.recommendations && results.recommendations.length > 0 ? (
            <ListGroup>
              {results.recommendations.map((recommendation, index) => (
                <ListGroup.Item key={index}>
                  {recommendation}
                </ListGroup.Item>
              ))}
            </ListGroup>
          ) : (
            <p className="text-muted">No recommendations available</p>
          )}
        </Card.Body>
      </Card>
      
      {results.downloadables && (
        <div className="mt-4">
          <h4>Available Downloads</h4>
          <Card>
            <Card.Body>
              <div className="d-flex flex-wrap gap-2">
                {results.downloadables.map((item, index) => (
                  <a
                    key={index}
                    href={item.url}
                    className="btn btn-outline-primary"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {item.name}
                  </a>
                ))}
              </div>
            </Card.Body>
          </Card>
        </div>
      )}
    </div>
  );
};

ResultsSummary.propTypes = {
  results: PropTypes.shape({
    id: PropTypes.string,
    status: PropTypes.string.isRequired,
    date: PropTypes.string,
    summary: PropTypes.shape({
      neuralFindings: PropTypes.string,
      microbiomeFindings: PropTypes.string,
      auditoryFindings: PropTypes.string,
      correlations: PropTypes.arrayOf(PropTypes.string)
    }),
    recommendations: PropTypes.arrayOf(PropTypes.string),
    downloadables: PropTypes.arrayOf(
      PropTypes.shape({
        name: PropTypes.string.isRequired,
        url: PropTypes.string.isRequired
      })
    )
  }).isRequired
};

export default ResultsSummary;
