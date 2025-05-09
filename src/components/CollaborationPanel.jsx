/**
 * CollaborationPanel.jsx
 * 
 * A React component for real-time collaboration between researchers
 * working on the same dataset or analysis. It provides chat, shared annotations,
 * and synchronized views of the data.
 */

import React, { useState, useEffect, useRef } from 'react';
import { Card, Form, Button, Badge, ListGroup, Tabs, Tab, Alert } from 'react-bootstrap';
import PropTypes from 'prop-types';

// This is a placeholder for the collaboration service that would be implemented in a real app
// In a real implementation, this would be a service using WebSockets or similar technology
const mockCollaborationService = {
  connected: false,
  roomId: null,
  userId: 'user-' + Math.random().toString(36).substr(2, 9),
  username: 'Anonymous',
  participants: [],
  messages: [],
  annotations: [],
  
  connect: function() {
    this.connected = true;
    return Promise.resolve();
  },
  
  disconnect: function() {
    this.connected = false;
    this.roomId = null;
    this.messages = [];
    this.participants = [];
    this.annotations = [];
  },
  
  joinRoom: function(roomId, username) {
    this.roomId = roomId;
    this.username = username;
    // Simulate other participants
    this.participants = [
      { userId: this.userId, username: username },
      { userId: 'user-123', username: 'Researcher 1' },
      { userId: 'user-456', username: 'Researcher 2' }
    ];
    return Promise.resolve(this.participants);
  },
  
  sendMessage: function(text) {
    const message = {
      id: 'msg-' + Math.random().toString(36).substr(2, 9),
      userId: this.userId,
      username: this.username,
      text: text,
      timestamp: new Date().toISOString()
    };
    this.messages.push(message);
    return Promise.resolve(message);
  },
  
  addAnnotation: function(annotation) {
    const newAnnotation = {
      id: 'ann-' + Math.random().toString(36).substr(2, 9),
      userId: this.userId,
      username: this.username,
      ...annotation,
      timestamp: new Date().toISOString()
    };
    this.annotations.push(newAnnotation);
    return Promise.resolve(newAnnotation);
  },
  
  deleteAnnotation: function(id) {
    this.annotations = this.annotations.filter(ann => ann.id !== id);
    return Promise.resolve({ id });
  },
  
  getMessages: function() {
    return this.messages;
  },
  
  getParticipants: function() {
    return Promise.resolve(this.participants);
  },
  
  getAnnotations: function() {
    return this.annotations;
  },
  
  onEvent: function(event, callback) {
    // This would set up event handlers in a real implementation
    // For now, it's just a placeholder
  }
};

const CollaborationPanel = ({ 
  patientId, 
  onFilterChange, 
  onSelectionChange,
  onAnnotationAdd,
  onAnnotationUpdate,
  onAnnotationDelete 
}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [room, setRoom] = useState(null);
  const [username, setUsername] = useState('');
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [participants, setParticipants] = useState([]);
  const [annotations, setAnnotations] = useState([]);
  const [newAnnotation, setNewAnnotation] = useState({ title: '', text: '' });
  const [isJoining, setIsJoining] = useState(false);
  const [error, setError] = useState(null);
  
  const chatContainerRef = useRef();
  
  // Set room ID based on patient ID
  useEffect(() => {
    if (patientId) {
      setRoom(`patient-${patientId}`);
    }
  }, [patientId]);
  
  // Load username from local storage
  useEffect(() => {
    const savedUsername = localStorage.getItem('username');
    if (savedUsername) {
      setUsername(savedUsername);
    }
  }, []);
  
  // Auto-scroll chat to bottom when messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);
  
  // Load messages and participants when connected and room is joined
  useEffect(() => {
    if (isConnected && room) {
      // Load initial messages
      setMessages(mockCollaborationService.getMessages());
      
      // Load initial annotations
      setAnnotations(mockCollaborationService.getAnnotations());
      
      // Setup event handlers for real-time updates
      // In a real app, this would set up WebSocket listeners
    }
  }, [isConnected, room]);
  
  // Connect to collaboration server
  const connect = async () => {
    if (isConnected) return;
    
    try {
      await mockCollaborationService.connect();
      setIsConnected(true);
      setError(null);
      
      // Save username to local storage
      localStorage.setItem('username', username);
    } catch (err) {
      setError(`Connection failed: ${err.message}`);
    }
  };
  
  // Join the room
  const joinRoom = async () => {
    if (!isConnected || !room || !username) return;
    
    setIsJoining(true);
    try {
      const participants = await mockCollaborationService.joinRoom(room, username);
      setParticipants(participants);
      
      // Add system message
      const joinMessage = {
        id: `system-${Date.now()}`,
        type: 'system',
        text: `You joined the room`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, joinMessage]);
      
      setError(null);
    } catch (err) {
      setError(`Failed to join room: ${err.message}`);
    } finally {
      setIsJoining(false);
    }
  };
  
  // Disconnect
  const disconnect = () => {
    mockCollaborationService.disconnect();
    setIsConnected(false);
    setMessages([]);
    setParticipants([]);
    setAnnotations([]);
  };
  
  // Send a chat message
  const sendMessage = (e) => {
    e.preventDefault();
    
    if (!message.trim()) return;
    
    mockCollaborationService.sendMessage(message)
      .then((newMessage) => {
        setMessages(prev => [...prev, newMessage]);
        setMessage('');
      })
      .catch(err => {
        setError(`Failed to send message: ${err.message}`);
      });
  };
  
  // Add an annotation
  const addAnnotation = (e) => {
    e.preventDefault();
    
    if (!newAnnotation.title.trim() || !newAnnotation.text.trim()) return;
    
    mockCollaborationService.addAnnotation(newAnnotation)
      .then((addedAnnotation) => {
        setAnnotations(prev => [...prev, addedAnnotation]);
        setNewAnnotation({ title: '', text: '' });
        
        // Notify parent component
        if (onAnnotationAdd) {
          onAnnotationAdd(addedAnnotation);
        }
        
        // Add system message
        const annotationMessage = {
          id: `system-${Date.now()}`,
          type: 'system',
          text: `You added an annotation: ${addedAnnotation.title}`,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, annotationMessage]);
      })
      .catch(err => {
        setError(`Failed to add annotation: ${err.message}`);
      });
  };
  
  // Delete an annotation
  const deleteAnnotation = (id) => {
    mockCollaborationService.deleteAnnotation(id)
      .then(() => {
        setAnnotations(prev => prev.filter(ann => ann.id !== id));
        
        // Notify parent component
        if (onAnnotationDelete) {
          onAnnotationDelete(id);
        }
        
        // Add system message
        const annotationMessage = {
          id: `system-${Date.now()}`,
          type: 'system',
          text: `You deleted an annotation`,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, annotationMessage]);
      })
      .catch(err => {
        setError(`Failed to delete annotation: ${err.message}`);
      });
  };
  
  // Format timestamp
  const formatTime = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  // Share current view with other participants
  const shareCurrentView = () => {
    // Add system message
    const shareMessage = {
      id: `system-${Date.now()}`,
      type: 'system',
      text: `You shared your current view with all participants`,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, shareMessage]);
    
    // In a real app, this would broadcast the current view to all participants
  };
  
  return (
    <Card className="collaboration-panel">
      <Card.Header>
        <div className="d-flex justify-content-between align-items-center">
          <h5 className="mb-0">Collaboration</h5>
          {isConnected ? (
            <Badge bg="success">Connected</Badge>
          ) : (
            <Badge bg="secondary">Disconnected</Badge>
          )}
        </div>
      </Card.Header>
      
      <Card.Body>
        {error && (
          <Alert variant="danger" onClose={() => setError(null)} dismissible>
            {error}
          </Alert>
        )}
        
        {!isConnected ? (
          <div className="connection-form">
            <Form.Group className="mb-3">
              <Form.Label>Your Name</Form.Label>
              <Form.Control 
                type="text" 
                value={username} 
                onChange={(e) => setUsername(e.target.value)} 
                placeholder="Enter your name"
              />
            </Form.Group>
            
            <Button 
              variant="primary" 
              onClick={connect} 
              disabled={!username}
            >
              Connect
            </Button>
          </div>
        ) : !room ? (
          <Alert variant="info">
            Please select a patient to join the collaboration room.
          </Alert>
        ) : (
          <>
            <Alert variant="info" className="mb-3">
              <div className="d-flex justify-content-between align-items-center">
                <div>
                  <strong>Room:</strong> {room} &nbsp;|&nbsp; 
                  <strong>Participants:</strong> {participants.length}
                </div>
                <Button 
                  variant="outline-primary" 
                  size="sm"
                  onClick={shareCurrentView}
                >
                  Share Current View
                </Button>
              </div>
            </Alert>
            
            <Tabs defaultActiveKey="chat" className="mb-3">
              <Tab eventKey="chat" title="Chat">
                <div 
                  className="chat-messages mb-3" 
                  ref={chatContainerRef}
                  style={{ 
                    height: '300px', 
                    overflowY: 'auto', 
                    border: '1px solid #dee2e6', 
                    borderRadius: '0.25rem',
                    padding: '0.5rem'
                  }}
                >
                  {messages.length === 0 ? (
                    <div className="text-center text-muted my-5">
                      No messages yet
                    </div>
                  ) : (
                    messages.map(msg => (
                      <div 
                        key={msg.id} 
                        className={`message ${msg.type === 'system' ? 'system-message' : msg.userId === mockCollaborationService.userId ? 'own-message' : 'other-message'}`}
                        style={{
                          marginBottom: '0.75rem',
                          padding: '0.5rem',
                          borderRadius: '0.25rem',
                          backgroundColor: msg.type === 'system' 
                            ? '#f8f9fa' 
                            : msg.userId === mockCollaborationService.userId 
                              ? '#e3f2fd' 
                              : '#f5f5f5'
                        }}
                      >
                        {msg.type !== 'system' && (
                          <div className="message-header" style={{ fontWeight: 'bold' }}>
                            {msg.username} <span className="text-muted" style={{ fontSize: '0.8rem' }}>{formatTime(msg.timestamp)}</span>
                          </div>
                        )}
                        <div>{msg.text}</div>
                      </div>
                    ))
                  )}
                </div>
                
                <Form onSubmit={sendMessage}>
                  <div className="input-group">
                    <Form.Control
                      type="text"
                      value={message}
                      onChange={(e) => setMessage(e.target.value)}
                      placeholder="Type a message..."
                    />
                    <Button type="submit" variant="primary">Send</Button>
                  </div>
                </Form>
              </Tab>
              
              <Tab eventKey="participants" title="Participants">
                <ListGroup style={{ height: '300px', overflowY: 'auto' }}>
                  {participants.length === 0 ? (
                    <ListGroup.Item className="text-center text-muted">
                      No participants
                    </ListGroup.Item>
                  ) : (
                    participants.map(participant => (
                      <ListGroup.Item 
                        key={participant.userId}
                        className={participant.userId === mockCollaborationService.userId ? 'bg-light' : ''}
                      >
                        {participant.username}
                        {participant.userId === mockCollaborationService.userId && (
                          <Badge bg="primary" className="ms-2">You</Badge>
                        )}
                      </ListGroup.Item>
                    ))
                  )}
                </ListGroup>
              </Tab>
              
              <Tab eventKey="annotations" title="Annotations">
                <div className="annotations-container mb-3" style={{ height: '200px', overflowY: 'auto' }}>
                  {annotations.length === 0 ? (
                    <div className="text-center text-muted my-5">
                      No annotations yet
                    </div>
                  ) : (
                    <ListGroup>
                      {annotations.map(annotation => (
                        <ListGroup.Item key={annotation.id}>
                          <div className="d-flex justify-content-between align-items-start">
                            <div>
                              <div className="fw-bold">{annotation.title}</div>
                              <div>{annotation.text}</div>
                              <div className="text-muted small">
                                By {annotation.username} at {formatTime(annotation.timestamp)}
                              </div>
                            </div>
                            {annotation.userId === mockCollaborationService.userId && (
                              <Button 
                                variant="outline-danger" 
                                size="sm"
                                onClick={() => deleteAnnotation(annotation.id)}
                              >
                                Delete
                              </Button>
                            )}
                          </div>
                        </ListGroup.Item>
                      ))}
                    </ListGroup>
                  )}
                </div>
                
                <Form onSubmit={addAnnotation}>
                  <Form.Group className="mb-2">
                    <Form.Control
                      type="text"
                      placeholder="Annotation Title"
                      value={newAnnotation.title}
                      onChange={(e) => setNewAnnotation({...newAnnotation, title: e.target.value})}
                    />
                  </Form.Group>
                  <Form.Group className="mb-2">
                    <Form.Control
                      as="textarea"
                      rows={2}
                      placeholder="Annotation Text"
                      value={newAnnotation.text}
                      onChange={(e) => setNewAnnotation({...newAnnotation, text: e.target.value})}
                    />
                  </Form.Group>
                  <Button type="submit" variant="primary" size="sm">
                    Add Annotation
                  </Button>
                </Form>
              </Tab>
            </Tabs>
          </>
        )}
      </Card.Body>
      
      {isConnected && (
        <Card.Footer>
          <Button variant="outline-danger" onClick={disconnect}>
            Disconnect
          </Button>
        </Card.Footer>
      )}
    </Card>
  );
};

CollaborationPanel.propTypes = {
  patientId: PropTypes.string,
  onFilterChange: PropTypes.func,
  onSelectionChange: PropTypes.func,
  onAnnotationAdd: PropTypes.func,
  onAnnotationUpdate: PropTypes.func,
  onAnnotationDelete: PropTypes.func
};

export default CollaborationPanel;
