/**
 * CorrelationMatrix.jsx
 * 
 * A React component for visualizing correlation matrices between features
 * from different modalities (neural, microbiome, auditory) with interactive
 * sorting, filtering, and zooming capabilities.
 */

import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import PropTypes from 'prop-types';

const CorrelationMatrix = ({ 
  data, 
  width = 800, 
  height = 600,
  defaultGrouping = 'modality'
}) => {
  const svgRef = useRef();
  const tooltipRef = useRef();
  const [threshold, setThreshold] = useState(0.2);
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [grouping, setGrouping] = useState(defaultGrouping);
  const [sortBy, setSortBy] = useState('name');
  
  useEffect(() => {
    if (!data || !svgRef.current) return;
    
    // Clear previous visualization
    d3.select(svgRef.current).selectAll("*").remove();
    
    // Create tooltip
    const tooltip = d3.select(tooltipRef.current);
    
    // Extract features and correlations
    const { features, correlations } = data;
    
    // Filter correlations based on threshold
    const filteredCorrelations = correlations.filter(d => 
      Math.abs(d.value) >= threshold && 
      (!selectedFeature || d.source === selectedFeature || d.target === selectedFeature)
    );
    
    // Group features based on grouping option
    let groupedFeatures = [...features];
    
    if (grouping === 'modality') {
      // Group by modality (neural, microbiome, auditory)
      groupedFeatures.sort((a, b) => {
        if (a.modality !== b.modality) {
          return a.modality.localeCompare(b.modality);
        }
        return a.name.localeCompare(b.name);
      });
    } else if (grouping === 'cluster') {
      // Group by correlation structure
      // This would typically involve some clustering algorithm
      // Here we'll just use a placeholder sorting
      groupedFeatures.sort((a, b) => a.cluster - b.cluster || a.name.localeCompare(b.name));
    } else {
      // Sort alphabetically by name
      if (sortBy === 'name') {
        groupedFeatures.sort((a, b) => a.name.localeCompare(b.name));
      } else if (sortBy === 'correlation') {
        // Sort by total correlation strength
        const strengthMap = new Map();
        
        // Calculate total correlation strength for each feature
        features.forEach(feature => {
          const strength = correlations
            .filter(d => d.source === feature.name || d.target === feature.name)
            .reduce((sum, d) => sum + Math.abs(d.value), 0);
          
          strengthMap.set(feature.name, strength);
        });
        
        groupedFeatures.sort((a, b) => 
          strengthMap.get(b.name) - strengthMap.get(a.name) || a.name.localeCompare(b.name)
        );
      }
    }
    
    // Extract feature names in order
    const featureNames = groupedFeatures.map(d => d.name);
    
    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);
    
    const margin = { top: 50, right: 50, bottom: 100, left: 100 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    const cellSize = Math.min(
      innerWidth / featureNames.length,
      innerHeight / featureNames.length
    );
    
    // Create scales
    const x = d3.scaleBand()
      .domain(featureNames)
      .range([0, featureNames.length * cellSize]);
    
    const y = d3.scaleBand()
      .domain(featureNames)
      .range([0, featureNames.length * cellSize]);
    
    const colorScale = d3.scaleLinear()
      .domain([-1, 0, 1])
      .range(['#d62728', '#fff', '#2ca02c']);
    
    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);
    
    // Add matrix cells
    const cells = g.selectAll('.cell')
      .data(filteredCorrelations)
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', d => x(d.source))
      .attr('y', d => y(d.target))
      .attr('width', cellSize)
      .attr('height', cellSize)
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', '#ddd')
      .attr('stroke-width', 0.5)
      .on('mouseover', function(event, d) {
        // Highlight row and column
        d3.selectAll('.cell')
          .style('opacity', c => 
            (c.source === d.source || c.target === d.source || 
             c.source === d.target || c.target === d.target) ? 1 : 0.3
          );
        
        // Highlight axis labels
        d3.selectAll('.x-axis text')
          .style('font-weight', text => 
            (text === d.source || text === d.target) ? 'bold' : 'normal'
          );
        
        d3.selectAll('.y-axis text')
          .style('font-weight', text => 
            (text === d.source || text === d.target) ? 'bold' : 'normal'
          );
        
        // Show tooltip
        tooltip
          .style('display', 'block')
          .style('left', `${event.pageX + 10}px`)
          .style('top', `${event.pageY - 30}px`)
          .html(`
            <strong>${d.source}</strong> × <strong>${d.target}</strong><br>
            Correlation: <strong>${d.value.toFixed(3)}</strong><br>
            p-value: <strong>${d.pvalue ? d.pvalue.toFixed(3) : 'N/A'}</strong>
          `);
      })
      .on('mouseout', function() {
        // Reset opacity
        d3.selectAll('.cell').style('opacity', 1);
        
        // Reset axis labels
        d3.selectAll('.x-axis text').style('font-weight', 'normal');
        d3.selectAll('.y-axis text').style('font-weight', 'normal');
        
        // Hide tooltip
        tooltip.style('display', 'none');
      })
      .on('click', function(event, d) {
        // Toggle feature selection
        if (selectedFeature === d.source) {
          setSelectedFeature(null);
        } else {
          setSelectedFeature(d.source);
        }
      });
    
    // Add x axis
    const xAxis = g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${featureNames.length * cellSize})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .attr('text-anchor', 'end')
      .attr('x', -8)
      .attr('y', 8)
      .text(d => d);
    
    // Add y axis
    const yAxis = g.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(y));
    
    // Highlight modalities with different colors if grouping by modality
    if (grouping === 'modality') {
      // Create a map of modality to color
      const modalityColors = {
        neural: '#f8f9fa',
        microbiome: '#e9ecef',
        auditory: '#dee2e6'
      };
      
      // Add background rectangles for modalities
      const modalities = [...new Set(groupedFeatures.map(d => d.modality))];
      
      modalities.forEach(modality => {
        const modalityFeatures = groupedFeatures.filter(d => d.modality === modality);
        const startIdx = featureNames.indexOf(modalityFeatures[0].name);
        const endIdx = featureNames.indexOf(modalityFeatures[modalityFeatures.length - 1].name);
        
        // Add background for x axis
        g.append('rect')
          .attr('x', x(featureNames[startIdx]))
          .attr('y', -10)
          .attr('width', (endIdx - startIdx + 1) * cellSize)
          .attr('height', 10)
          .attr('fill', modalityColors[modality]);
        
        // Add background for y axis
        g.append('rect')
          .attr('x', -10)
          .attr('y', y(featureNames[startIdx]))
          .attr('width', 10)
          .attr('height', (endIdx - startIdx + 1) * cellSize)
          .attr('fill', modalityColors[modality]);
        
        // Add modality label for x axis
        g.append('text')
          .attr('x', x(featureNames[startIdx]) + (endIdx - startIdx) * cellSize / 2)
          .attr('y', -15)
          .attr('text-anchor', 'middle')
          .style('font-weight', 'bold')
          .text(modality.charAt(0).toUpperCase() + modality.slice(1));
        
        // Add modality label for y axis
        g.append('text')
          .attr('transform', `translate(-15, ${y(featureNames[startIdx]) + (endIdx - startIdx) * cellSize / 2}) rotate(-90)`)
          .attr('text-anchor', 'middle')
          .style('font-weight', 'bold')
          .text(modality.charAt(0).toUpperCase() + modality.slice(1));
      });
    }
    
    // Add legend
    const legendWidth = 200;
    const legendHeight = 20;
    
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - margin.right - legendWidth}, ${margin.top - 30})`);
    
    // Create a gradient for the legend
    const defs = legend.append('defs');
    const linearGradient = defs.append('linearGradient')
      .attr('id', 'correlation-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '100%')
      .attr('y2', '0%');
    
    linearGradient.selectAll('stop')
      .data([
        {offset: '0%', color: colorScale(-1)},
        {offset: '50%', color: colorScale(0)},
        {offset: '100%', color: colorScale(1)}
      ])
      .enter().append('stop')
      .attr('offset', d => d.offset)
      .attr('stop-color', d => d.color);
    
    // Add the gradient rectangle
    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#correlation-gradient)');
    
    // Add legend axis
    const legendScale = d3.scaleLinear()
      .domain([-1, 0, 1])
      .range([0, legendWidth / 2, legendWidth]);
    
    const legendAxis = d3.axisBottom(legendScale)
      .tickValues([-1, -0.5, 0, 0.5, 1])
      .tickFormat(d3.format('.1f'));
    
    legend.append('g')
      .attr('transform', `translate(0, ${legendHeight})`)
      .call(legendAxis);
    
    // Add legend title
    legend.append('text')
      .attr('x', legendWidth / 2)
      .attr('y', -5)
      .attr('text-anchor', 'middle')
      .text('Correlation');
    
    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text('Feature Correlation Matrix');
    
  }, [data, threshold, selectedFeature, grouping, sortBy, width, height]);
  
  // Handle threshold change
  const handleThresholdChange = (e) => {
    setThreshold(parseFloat(e.target.value));
  };
  
  // Reset selection
  const handleReset = () => {
    setSelectedFeature(null);
    setThreshold(0.2);
  };
  
  return (
    <div className="correlation-matrix">
      <div className="controls mb-3">
        <div className="row">
          <div className="col-md-4">
            <label htmlFor="threshold-range" className="form-label">
              Correlation Threshold: {threshold.toFixed(2)}
            </label>
            <input 
              type="range" 
              className="form-range" 
              id="threshold-range"
              min="0" 
              max="1" 
              step="0.01" 
              value={threshold} 
              onChange={handleThresholdChange} 
            />
          </div>
          
          <div className="col-md-4">
            <label htmlFor="grouping-select" className="form-label">
              Group Features By
            </label>
            <select 
              id="grouping-select" 
              className="form-select"
              value={grouping}
              onChange={(e) => setGrouping(e.target.value)}
            >
              <option value="modality">Modality</option>
              <option value="cluster">Correlation Structure</option>
              <option value="custom">Custom Sorting</option>
            </select>
          </div>
          
          <div className="col-md-4">
            <label htmlFor="sorting-select" className="form-label">
              Sort Features By
            </label>
            <select 
              id="sorting-select" 
              className="form-select"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              disabled={grouping !== 'custom'}
            >
              <option value="name">Name</option>
              <option value="correlation">Correlation Strength</option>
            </select>
          </div>
        </div>
        
        {selectedFeature && (
          <div className="selected-info alert alert-info mt-2">
            <strong>Selected Feature:</strong> {selectedFeature}
            <button 
              type="button" 
              className="btn btn-outline-secondary btn-sm ms-2"
              onClick={handleReset}
            >
              Reset Selection
            </button>
          </div>
        )}
      </div>
      
      <div className="matrix-container position-relative">
        <svg ref={svgRef}></svg>
        <div 
          ref={tooltipRef} 
          className="tooltip" 
          style={{
            position: 'absolute',
            display: 'none',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            border: '1px solid #ddd',
            borderRadius: '4px',
            padding: '8px',
            pointerEvents: 'none',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            zIndex: 1000
          }}
        ></div>
      </div>
      
      <div className="instructions mt-3 small text-muted">
        <p>
          <strong>Instructions:</strong> Hover over cells to see detailed correlation values. 
          Click on a cell to highlight all correlations for a specific feature. 
          Adjust the threshold slider to filter out weaker correlations.
        </p>
      </div>
    </div>
  );
};

CorrelationMatrix.propTypes = {
  data: PropTypes.shape({
    features: PropTypes.arrayOf(PropTypes.shape({
      name: PropTypes.string.isRequired,
      modality: PropTypes.string.isRequired,
      description: PropTypes.string,
      cluster: PropTypes.number
    })).isRequired,
    correlations: PropTypes.arrayOf(PropTypes.shape({
      source: PropTypes.string.isRequired,
      target: PropTypes.string.isRequired,
      value: PropTypes.number.isRequired,
      pvalue: PropTypes.number
    })).isRequired
  }).isRequired,
  width: PropTypes.number,
  height: PropTypes.number,
  defaultGrouping: PropTypes.oneOf(['modality', 'cluster', 'custom'])
};

export default CorrelationMatrix;
