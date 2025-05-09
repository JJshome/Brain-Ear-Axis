/**
 * ConnectivityNetwork.jsx
 * 
 * A React component for visualizing brain-ear connectivity networks
 * using D3.js. This component provides interactive visualization of
 * neural connection patterns with filtering and selection capabilities.
 */

import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import PropTypes from 'prop-types';

const ConnectivityNetwork = ({ 
  data, 
  width = 600, 
  height = 500,
  showLabels = true,
  defaultThreshold = 0.2,
  defaultNetworkType = 'all'
}) => {
  const svgRef = useRef();
  const [threshold, setThreshold] = useState(defaultThreshold);
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [networkType, setNetworkType] = useState(defaultNetworkType);
  
  useEffect(() => {
    if (!data || !svgRef.current) return;
    
    // Clear previous visualization
    d3.select(svgRef.current).selectAll("*").remove();
    
    // Filter links based on threshold and network type
    const filteredLinks = data.links.filter(link => {
      if (Math.abs(link.value) < threshold) return false;
      if (networkType === 'positive' && link.value <= 0) return false;
      if (networkType === 'negative' && link.value >= 0) return false;
      if (selectedRegion && link.source !== selectedRegion && link.target !== selectedRegion) return false;
      return true;
    });
    
    // Create node set from filtered links
    const nodeSet = new Set();
    filteredLinks.forEach(link => {
      nodeSet.add(typeof link.source === 'object' ? link.source.id : link.source);
      nodeSet.add(typeof link.target === 'object' ? link.target.id : link.target);
    });
    
    // Filter nodes
    const filteredNodes = data.nodes.filter(node => nodeSet.has(node.id));
    
    // Color scale for links
    const linkColorScale = d3.scaleLinear()
      .domain([-1, 0, 1])
      .range(['#d62728', '#aaa', '#2ca02c']);
    
    // Create SVG container
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width / 2}, ${height / 2})`);
    
    // Create simulation
    const simulation = d3.forceSimulation(filteredNodes)
      .force('link', d3.forceLink(filteredLinks)
        .id(d => d.id)
        .distance(100)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(0, 0))
      .force('collision', d3.forceCollide().radius(40));
    
    // Create links
    const link = svg.append('g')
      .selectAll('line')
      .data(filteredLinks)
      .join('line')
      .attr('stroke', d => linkColorScale(d.value))
      .attr('stroke-width', d => Math.abs(d.value) * 5)
      .attr('stroke-opacity', 0.7);
    
    // Node groups
    const node = svg.append('g')
      .selectAll('.node')
      .data(filteredNodes)
      .join('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended)
      )
      .on('click', (event, d) => {
        setSelectedRegion(selectedRegion === d.id ? null : d.id);
      });
    
    // Node circles
    node.append('circle')
      .attr('r', d => 10 + (d.weight || 1) * 5)
      .attr('fill', d => d.color || '#69b3a2')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .append('title')
      .text(d => d.name || d.id);
    
    // Node labels
    if (showLabels) {
      node.append('text')
        .attr('dx', 15)
        .attr('dy', 4)
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .text(d => d.name || d.id);
    }
    
    // Update positions during simulation
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      
      node
        .attr('transform', d => `translate(${d.x}, ${d.y})`);
    });
    
    // Drag functions
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
    
    // Add zoom functionality
    const zoom = d3.zoom()
      .scaleExtent([0.5, 5])
      .on('zoom', (event) => {
        svg.attr('transform', event.transform);
      });
    
    d3.select(svgRef.current).call(zoom);
    
    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [data, threshold, selectedRegion, showLabels, networkType, width, height]);
  
  // Handle threshold change
  const handleThresholdChange = (e) => {
    setThreshold(parseFloat(e.target.value));
  };
  
  // Reset selection
  const handleReset = () => {
    setSelectedRegion(null);
    setThreshold(defaultThreshold);
    setNetworkType(defaultNetworkType);
  };
  
  return (
    <div className="connectivity-network">
      <div className="controls mb-3">
        <div className="row">
          <div className="col-md-6">
            <label htmlFor="threshold-range" className="form-label">
              Connection Threshold: {threshold.toFixed(2)}
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
          
          <div className="col-md-6">
            <div className="form-check form-switch mb-2">
              <input 
                className="form-check-input" 
                type="checkbox"
                id="show-labels"
                checked={showLabels}
                onChange={(e) => setShowLabels(e.target.checked)}
              />
              <label className="form-check-label" htmlFor="show-labels">
                Show Labels
              </label>
            </div>
            
            <div className="btn-group" role="group" aria-label="Connection Type">
              <button 
                type="button" 
                className={`btn btn-${networkType === 'all' ? 'primary' : 'outline-primary'}`}
                onClick={() => setNetworkType('all')}
              >
                All
              </button>
              <button 
                type="button" 
                className={`btn btn-${networkType === 'positive' ? 'primary' : 'outline-primary'}`}
                onClick={() => setNetworkType('positive')}
              >
                Positive
              </button>
              <button 
                type="button" 
                className={`btn btn-${networkType === 'negative' ? 'primary' : 'outline-primary'}`}
                onClick={() => setNetworkType('negative')}
              >
                Negative
              </button>
            </div>
          </div>
        </div>
        
        {selectedRegion && (
          <div className="selected-info alert alert-info mt-2">
            <strong>Selected Region:</strong> {selectedRegion}
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
      
      <div className="network-container">
        <svg ref={svgRef}></svg>
      </div>
      
      <div className="legend mt-3">
        <div className="d-flex justify-content-between">
          <div>
            <span className="legend-color" style={{ 
              display: 'inline-block', 
              width: '20px', 
              height: '10px', 
              backgroundColor: '#d62728',
              marginRight: '5px' 
            }}></span>
            <span>Negative Correlation</span>
          </div>
          <div>
            <span className="legend-color" style={{ 
              display: 'inline-block', 
              width: '20px', 
              height: '10px', 
              backgroundColor: '#aaa',
              marginRight: '5px' 
            }}></span>
            <span>No Correlation</span>
          </div>
          <div>
            <span className="legend-color" style={{ 
              display: 'inline-block', 
              width: '20px', 
              height: '10px', 
              backgroundColor: '#2ca02c',
              marginRight: '5px' 
            }}></span>
            <span>Positive Correlation</span>
          </div>
        </div>
      </div>
    </div>
  );
};

ConnectivityNetwork.propTypes = {
  data: PropTypes.shape({
    nodes: PropTypes.arrayOf(PropTypes.shape({
      id: PropTypes.string.isRequired,
      name: PropTypes.string,
      weight: PropTypes.number,
      color: PropTypes.string
    })).isRequired,
    links: PropTypes.arrayOf(PropTypes.shape({
      source: PropTypes.oneOfType([PropTypes.string, PropTypes.object]).isRequired,
      target: PropTypes.oneOfType([PropTypes.string, PropTypes.object]).isRequired,
      value: PropTypes.number.isRequired
    })).isRequired
  }).isRequired,
  width: PropTypes.number,
  height: PropTypes.number,
  showLabels: PropTypes.bool,
  defaultThreshold: PropTypes.number,
  defaultNetworkType: PropTypes.oneOf(['all', 'positive', 'negative'])
};

export default ConnectivityNetwork;
