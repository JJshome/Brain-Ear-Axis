/**
 * TimeSeries.jsx
 * 
 * A React component for visualizing time series data from neural, auditory,
 * or other temporal measurements with interactive controls and filtering.
 */

import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import PropTypes from 'prop-types';

const TimeSeries = ({ 
  data, 
  width = 800, 
  height = 400,
  defaultVisualization = 'line'
}) => {
  const svgRef = useRef();
  const tooltipRef = useRef();
  const [selectedChannels, setSelectedChannels] = useState([]);
  const [timeWindow, setTimeWindow] = useState([0, 100]); // percentage
  const [filter, setFilter] = useState('none');
  const [visualization, setVisualization] = useState(defaultVisualization);
  const [resample, setResample] = useState(false);
  
  // Filter options
  const filterOptions = [
    { value: 'none', label: 'No Filter' },
    { value: 'lowpass', label: 'Low Pass' },
    { value: 'highpass', label: 'High Pass' },
    { value: 'bandpass', label: 'Band Pass' }
  ];
  
  // Visualization options
  const visualizationOptions = [
    { value: 'line', label: 'Line Chart' },
    { value: 'heatmap', label: 'Heatmap' },
    { value: 'spectrogram', label: 'Spectrogram' }
  ];
  
  useEffect(() => {
    if (!data || !svgRef.current) return;
    
    // Clear previous visualization
    d3.select(svgRef.current).selectAll("*").remove();
    
    // Create tooltip
    const tooltip = d3.select(tooltipRef.current);
    
    // Prepare data
    // In a real implementation, you would apply filtering based on the filter state
    const filteredData = filterTimeSeries(data, filter);
    
    // Subset data based on selected channels
    const channelsToShow = selectedChannels.length > 0 ? 
      selectedChannels : 
      data.channels.slice(0, 5); // Default to first 5 channels if none selected
    
    // Compute time window in data points
    const totalPoints = data.timePoints.length;
    const startIndex = Math.floor(timeWindow[0] / 100 * totalPoints);
    const endIndex = Math.floor(timeWindow[1] / 100 * totalPoints);
    
    // Time points to show
    const timePointsToShow = data.timePoints.slice(startIndex, endIndex);
    
    // Format data for visualization
    const seriesData = channelsToShow.map(channel => {
      return {
        name: channel,
        values: data.values[channel].slice(startIndex, endIndex)
      };
    });
    
    // Create SVG container
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);
    
    const margin = { top: 20, right: 80, bottom: 30, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([timePointsToShow[0], timePointsToShow[timePointsToShow.length - 1]])
      .range([0, innerWidth]);
    
    // Find global min and max for y scale
    let minValue = d3.min(seriesData, d => d3.min(d.values));
    let maxValue = d3.max(seriesData, d => d3.max(d.values));
    
    // Add some padding to y scale
    const yPadding = (maxValue - minValue) * 0.1;
    
    const yScale = d3.scaleLinear()
      .domain([minValue - yPadding, maxValue + yPadding])
      .range([innerHeight, 0]);
    
    // Render based on visualization type
    if (visualization === 'line') {
      renderLineChart(g, seriesData, timePointsToShow, xScale, yScale, innerHeight, innerWidth, margin, tooltip);
    } else if (visualization === 'heatmap') {
      renderHeatmap(g, seriesData, timePointsToShow, xScale, innerHeight, innerWidth, tooltip);
    } else if (visualization === 'spectrogram') {
      renderSpectrogram(g, seriesData, timePointsToShow, xScale, innerHeight, innerWidth, tooltip);
    }
    
  }, [data, selectedChannels, timeWindow, filter, visualization, width, height, resample]);
  
  // Function to filter time series (placeholder - would be implemented with proper DSP in a real app)
  const filterTimeSeries = (data, filterType) => {
    // In a real implementation, apply proper digital signal processing
    // This is just a placeholder that returns the original data
    return data;
  };
  
  // Line chart renderer
  const renderLineChart = (g, seriesData, timePointsToShow, xScale, yScale, innerHeight, innerWidth, margin, tooltip) => {
    // Create color scale
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    
    // Line generator
    const line = d3.line()
      .x((d, i) => xScale(timePointsToShow[i]))
      .y(d => yScale(d));
    
    // Add lines
    seriesData.forEach((series, i) => {
      g.append('path')
        .datum(series.values)
        .attr('class', 'line')
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', colorScale(i))
        .attr('stroke-width', 2)
        .attr('stroke-linejoin', 'round')
        .attr('stroke-linecap', 'round');
    });
    
    // Add axes
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .ticks(5)
        .tickFormat(d => `${d.toFixed(2)}s`));
    
    g.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale));
    
    // Add axis labels
    g.append('text')
      .attr('transform', `translate(${innerWidth / 2}, ${innerHeight + margin.bottom - 5})`)
      .style('text-anchor', 'middle')
      .text('Time (s)');
    
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -margin.left + 15)
      .attr('x', -innerHeight / 2)
      .style('text-anchor', 'middle')
      .text('Amplitude (µV)');
    
    // Add legend
    const svg = d3.select(svgRef.current);
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - margin.right + 10}, ${margin.top})`);
    
    seriesData.forEach((series, i) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);
      
      legendItem.append('rect')
        .attr('width', 10)
        .attr('height', 10)
        .attr('fill', colorScale(i));
      
      legendItem.append('text')
        .attr('x', 15)
        .attr('y', 9)
        .text(series.name);
    });
    
    // Add hover effect
    const mouseG = g.append("g")
      .attr("class", "mouse-over-effects");
    
    mouseG.append("path")
      .attr("class", "mouse-line")
      .style("stroke", "#666")
      .style("stroke-width", "1px")
      .style("opacity", "0");
    
    const mousePerLine = mouseG.selectAll('.mouse-per-line')
      .data(seriesData)
      .enter()
      .append("g")
      .attr("class", "mouse-per-line");
    
    mousePerLine.append("circle")
      .attr("r", 6)
      .style("stroke", (d, i) => colorScale(i))
      .style("fill", "none")
      .style("stroke-width", "2px")
      .style("opacity", "0");
    
    mouseG.append('rect')
      .attr('width', innerWidth)
      .attr('height', innerHeight)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .on('mouseout', () => {
        d3.select(".mouse-line").style("opacity", "0");
        d3.selectAll(".mouse-per-line circle").style("opacity", "0");
        tooltip.style('display', 'none');
      })
      .on('mouseover', () => {
        d3.select(".mouse-line").style("opacity", "1");
        d3.selectAll(".mouse-per-line circle").style("opacity", "1");
        tooltip.style('display', 'block');
      })
      .on('mousemove', function(event) {
        const mouse = d3.pointer(event);
        
        d3.select(".mouse-line")
          .attr("d", () => `M${mouse[0]},${innerHeight} ${mouse[0]},0`);
        
        d3.selectAll(".mouse-per-line")
          .attr("transform", function(d, i) {
            const xDate = xScale.invert(mouse[0]);
            const bisect = d3.bisector(function(d, x) { return x - d; }).left;
            const idx = bisect(timePointsToShow, xDate);
            
            d3.select(this).select('circle')
              .attr("cx", xScale(timePointsToShow[idx]))
              .attr("cy", yScale(d.values[idx]));
            
            return `translate(${xScale(timePointsToShow[idx])}, ${yScale(d.values[idx])})`;
          });
        
        // Update tooltip
        tooltip
          .style('left', `${event.pageX + 10}px`)
          .style('top', `${event.pageY - 30}px`)
          .html(() => {
            const xDate = xScale.invert(mouse[0]);
            const bisect = d3.bisector(function(d, x) { return x - d; }).left;
            const idx = bisect(timePointsToShow, xDate);
            
            let tooltipHtml = `Time: ${timePointsToShow[idx].toFixed(3)}s<br>`;
            seriesData.forEach((d, i) => {
              tooltipHtml += `${d.name}: ${d.values[idx].toFixed(2)}µV<br>`;
            });
            
            return tooltipHtml;
          });
      });
  };
  
  // Heatmap renderer
  const renderHeatmap = (g, seriesData, timePointsToShow, xScale, innerHeight, innerWidth, tooltip) => {
    // Get all values for color scale
    const allValues = seriesData.flatMap(d => d.values);
    const colorExtent = [d3.min(allValues), d3.max(allValues)];
    
    // Color scale
    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain(colorExtent);
    
    // Cell size
    const cellHeight = innerHeight / seriesData.length;
    const cellWidth = innerWidth / timePointsToShow.length;
    
    // Draw heatmap cells
    seriesData.forEach((channel, channelIndex) => {
      channel.values.forEach((value, timeIndex) => {
        g.append('rect')
          .attr('x', xScale(timePointsToShow[timeIndex]) - cellWidth/2)
          .attr('y', channelIndex * cellHeight)
          .attr('width', cellWidth)
          .attr('height', cellHeight)
          .attr('fill', colorScale(value))
          .on('mouseover', (event) => {
            tooltip
              .style('display', 'block')
              .style('left', `${event.pageX + 10}px`)
              .style('top', `${event.pageY - 30}px`)
              .html(`Channel: ${channel.name}<br>Time: ${timePointsToShow[timeIndex].toFixed(3)}s<br>Value: ${value.toFixed(2)}µV`);
          })
          .on('mouseout', () => {
            tooltip.style('display', 'none');
          });
      });
      
      // Add channel labels
      g.append('text')
        .attr('x', -5)
        .attr('y', channelIndex * cellHeight + cellHeight / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .style('font-size', '12px')
        .text(channel.name);
    });
    
    // Add axes
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .ticks(5)
        .tickFormat(d => `${d.toFixed(2)}s`));
    
    // Add color legend
    const svg = d3.select(svgRef.current);
    const legendWidth = 20;
    const legendHeight = innerHeight;
    
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${innerWidth + 60}, ${20})`);
    
    const legendScale = d3.scaleLinear()
      .domain(colorExtent)
      .range([legendHeight, 0]);
    
    // Add gradient defs
    const defs = legend.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'heatmap-gradient')
      .attr('x1', '0%')
      .attr('y1', '100%')
      .attr('x2', '0%')
      .attr('y2', '0%');
    
    // Add gradient stops
    const stops = d3.range(0, 1.01, 0.1);
    stops.forEach(stop => {
      const value = colorExtent[0] + stop * (colorExtent[1] - colorExtent[0]);
      gradient.append('stop')
        .attr('offset', `${stop * 100}%`)
        .attr('stop-color', colorScale(value));
    });
    
    // Add legend rectangle
    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#heatmap-gradient)');
    
    // Add legend axis
    const legendAxis = d3.axisRight(legendScale)
      .ticks(5)
      .tickFormat(d => d.toFixed(1));
    
    legend.append('g')
      .attr('transform', `translate(${legendWidth}, 0)`)
      .call(legendAxis);
    
    // Add legend title
    legend.append('text')
      .attr('transform', `translate(${legendWidth/2}, ${-10})`)
      .style('text-anchor', 'middle')
      .text('µV');
  };
  
  // Spectrogram renderer (simplified version)
  const renderSpectrogram = (g, seriesData, timePointsToShow, xScale, innerHeight, innerWidth, tooltip) => {
    // This is a simplified spectrogram for demonstration
    // In a real app, you would calculate the frequency spectrum using FFT
    
    // For now, we'll just show a placeholder
    g.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .style('font-size', '16px')
      .text('Spectrogram view - Requires FFT calculation');
    
    // Add note about the placeholder
    g.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight / 2 + 30)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .style('font-size', '12px')
      .text('In a real implementation, this would show frequency over time');
  };
  
  // Toggle channel selection
  const handleChannelToggle = (channel) => {
    if (selectedChannels.includes(channel)) {
      setSelectedChannels(selectedChannels.filter(c => c !== channel));
    } else {
      setSelectedChannels([...selectedChannels, channel]);
    }
  };
  
  // Handle time window change
  const handleStartTimeChange = (e) => {
    const start = parseInt(e.target.value);
    if (start < timeWindow[1]) {
      setTimeWindow([start, timeWindow[1]]);
    }
  };
  
  const handleEndTimeChange = (e) => {
    const end = parseInt(e.target.value);
    if (end > timeWindow[0]) {
      setTimeWindow([timeWindow[0], end]);
    }
  };
  
  return (
    <div className="time-series-visualization">
      <div className="controls mb-3">
        <div className="row">
          <div className="col-md-4">
            <div className="form-group mb-3">
              <label htmlFor="filter-select">Filter</label>
              <select 
                className="form-control" 
                id="filter-select" 
                value={filter} 
                onChange={(e) => setFilter(e.target.value)}
              >
                {filterOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="col-md-4">
            <div className="form-group mb-3">
              <label htmlFor="visualization-select">Visualization Type</label>
              <select 
                className="form-control" 
                id="visualization-select" 
                value={visualization} 
                onChange={(e) => setVisualization(e.target.value)}
              >
                {visualizationOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="col-md-4">
            <div className="form-group mb-3">
              <label htmlFor="resample-switch">Resample</label>
              <div className="form-check form-switch">
                <input 
                  className="form-check-input" 
                  type="checkbox"
                  id="resample-switch"
                  checked={resample}
                  onChange={() => setResample(!resample)}
                />
                <label className="form-check-label" htmlFor="resample-switch">
                  Enable resampling
                </label>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mb-3">
          <label>Time Window: {timeWindow[0]}% - {timeWindow[1]}%</label>
          <div className="row">
            <div className="col-md-6">
              <label htmlFor="start-time" className="form-label">Start</label>
              <input 
                type="range" 
                className="form-range" 
                id="start-time"
                min="0" 
                max="99" 
                value={timeWindow[0]} 
                onChange={handleStartTimeChange} 
              />
            </div>
            <div className="col-md-6">
              <label htmlFor="end-time" className="form-label">End</label>
              <input 
                type="range" 
                className="form-range" 
                id="end-time"
                min="1" 
                max="100" 
                value={timeWindow[1]} 
                onChange={handleEndTimeChange} 
              />
            </div>
          </div>
        </div>
        
        {data && data.channels && (
          <div className="mb-3">
            <label>Channels</label>
            <div className="channel-selector d-flex flex-wrap">
              {data.channels.map((channel) => (
                <div key={channel} className="form-check me-3">
                  <input
                    className="form-check-input"
                    type="checkbox"
                    id={`channel-${channel}`}
                    checked={selectedChannels.includes(channel)}
                    onChange={() => handleChannelToggle(channel)}
                  />
                  <label className="form-check-label" htmlFor={`channel-${channel}`}>
                    {channel}
                  </label>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      <div className="visualization-container position-relative">
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
    </div>
  );
};

TimeSeries.propTypes = {
  data: PropTypes.shape({
    timePoints: PropTypes.arrayOf(PropTypes.number).isRequired,
    channels: PropTypes.arrayOf(PropTypes.string).isRequired,
    values: PropTypes.object.isRequired
  }).isRequired,
  width: PropTypes.number,
  height: PropTypes.number,
  defaultVisualization: PropTypes.oneOf(['line', 'heatmap', 'spectrogram'])
};

export default TimeSeries;
