/**
 * PatientSelector.jsx
 * 
 * Component for selecting patients from a dropdown list
 */

import React from 'react';
import { Form } from 'react-bootstrap';
import PropTypes from 'prop-types';

const PatientSelector = ({ patients, selectedPatient, onPatientChange }) => {
  const handleChange = (e) => {
    onPatientChange(e.target.value);
  };
  
  return (
    <Form.Group className="mb-3">
      <Form.Label>Select Patient</Form.Label>
      <Form.Select 
        value={selectedPatient?.id || ''} 
        onChange={handleChange}
      >
        <option value="" disabled>Select a patient...</option>
        {patients.map(patient => (
          <option key={patient.id} value={patient.id}>
            {patient.name} ({patient.age}y, {patient.gender})
          </option>
        ))}
      </Form.Select>
    </Form.Group>
  );
};

PatientSelector.propTypes = {
  patients: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      name: PropTypes.string.isRequired,
      age: PropTypes.number.isRequired,
      gender: PropTypes.string.isRequired
    })
  ).isRequired,
  selectedPatient: PropTypes.shape({
    id: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    age: PropTypes.number.isRequired,
    gender: PropTypes.string.isRequired
  }),
  onPatientChange: PropTypes.func.isRequired
};

export default PatientSelector;
