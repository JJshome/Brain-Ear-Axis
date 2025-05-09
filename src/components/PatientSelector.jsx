/**
 * PatientSelector.jsx
 * 
 * A component for selecting patients from a list with search and filtering capabilities.
 */

import React, { useState } from 'react';
import { Card, Form, ListGroup, Row, Col, Badge } from 'react-bootstrap';
import PropTypes from 'prop-types';

const PatientSelector = ({ patients, selectedPatient, onPatientChange }) => {
  const [searchTerm, setSearchTerm] = useState('');
  
  // Filter patients based on search term
  const filteredPatients = patients.filter(patient => 
    patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patient.id.toLowerCase().includes(searchTerm.toLowerCase())
  );
  
  return (
    <Card className="h-100">
      <Card.Header>
        <h5 className="mb-0">Select Patient</h5>
      </Card.Header>
      <Card.Body>
        <Form.Group className="mb-3">
          <Form.Control
            type="text"
            placeholder="Search patients..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </Form.Group>
        
        {filteredPatients.length === 0 ? (
          <div className="text-center py-4 text-muted">
            <p>No patients found</p>
          </div>
        ) : (
          <ListGroup>
            {filteredPatients.map(patient => (
              <ListGroup.Item 
                key={patient.id}
                action
                active={selectedPatient && selectedPatient.id === patient.id}
                onClick={() => onPatientChange(patient.id)}
              >
                <Row>
                  <Col xs={8}>
                    <div className="fw-bold">{patient.name}</div>
                    <div className="small text-muted">ID: {patient.id}</div>
                  </Col>
                  <Col xs={4} className="text-end">
                    <Badge bg="secondary">{patient.gender}</Badge>
                    <div className="small mt-1">{patient.age} years</div>
                  </Col>
                </Row>
              </ListGroup.Item>
            ))}
          </ListGroup>
        )}
      </Card.Body>
      <Card.Footer className="text-muted small">
        {filteredPatients.length} patients
      </Card.Footer>
    </Card>
  );
};

PatientSelector.propTypes = {
  patients: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      name: PropTypes.string.isRequired,
      age: PropTypes.number,
      gender: PropTypes.string
    })
  ).isRequired,
  selectedPatient: PropTypes.shape({
    id: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    age: PropTypes.number,
    gender: PropTypes.string
  }),
  onPatientChange: PropTypes.func.isRequired
};

export default PatientSelector;
