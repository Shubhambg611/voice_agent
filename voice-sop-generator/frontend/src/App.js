import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  // Session and conversation state
  const [sessionId, setSessionId] = useState(null);
  const [conversationStarted, setConversationStarted] = useState(false);
  const [messages, setMessages] = useState([]);
  const [currentPhase, setCurrentPhase] = useState('welcome');
  
  // Voice and UI state
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [microphonePermission, setMicrophonePermission] = useState('unknown'); // granted, denied, unknown
  
  // Notes state
  const [structuredNotes, setStructuredNotes] = useState({
    challenges: '',
    lessons: '',
    growth: '',
    college: ''
  });
  
  // Error state
  const [error, setError] = useState(null);
  
  // Refs
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const messageEndRef = useRef(null);
  
  // Initialize session
  useEffect(() => {
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
  }, []);
  
  // Auto-scroll to bottom of messages
  useEffect(() => {
    if (messageEndRef.current) {
      messageEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);
  
  // Check microphone permission on load
  useEffect(() => {
    checkMicrophonePermissionStatus();
  }, []);
  
  // Check microphone permission status
  const checkMicrophonePermissionStatus = async () => {
    try {
      if (navigator.permissions) {
        const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
        setMicrophonePermission(permissionStatus.state);
        
        // Listen for permission changes
        permissionStatus.onchange = () => {
          setMicrophonePermission(permissionStatus.state);
        };
      }
    } catch (err) {
      console.log('Permission API not supported');
    }
  };
  
  // Test microphone access
  const testMicrophoneAccess = async () => {
    try {
      setError(null);
      
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Your browser does not support audio recording. Please use Chrome, Firefox, or Safari.');
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop()); // Stop immediately after testing
      
      setMicrophonePermission('granted');
      alert('✅ Microphone access granted! You can now start the voice interview.');
      
    } catch (err) {
      console.error('Microphone test failed:', err);
      setMicrophonePermission('denied');
      
      let errorMessage = '❌ Microphone access failed. ';
      
      if (err.name === 'NotAllowedError') {
        errorMessage += 'Please click "Allow" when prompted for microphone access.';
      } else if (err.name === 'NotFoundError') {
        errorMessage += 'No microphone found. Please connect a microphone and try again.';
      } else if (err.name === 'NotSupportedError') {
        errorMessage += 'Your browser does not support audio recording.';
      } else {
        errorMessage += 'Please check your browser settings and allow microphone access.';
      }
      
      setError(errorMessage);
    }
  };
  
  // Speech synthesis function
  const speakText = useCallback((text) => {
    return new Promise((resolve) => {
      if (!text || isSpeaking) {
        resolve();
        return;
      }
      
      setIsSpeaking(true);
      window.speechSynthesis.cancel();
      
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      utterance.volume = 0.8;
      
      // Try to use a good voice
      const voices = window.speechSynthesis.getVoices();
      const preferredVoice = voices.find(voice => 
        voice.name.includes('Google') && voice.lang.startsWith('en') ||
        voice.name.includes('Microsoft') && voice.lang.startsWith('en') ||
        voice.lang.startsWith('en')
      );
      
      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }
      
      utterance.onend = () => {
        setIsSpeaking(false);
        resolve();
      };
      
      utterance.onerror = () => {
        setIsSpeaking(false);
        resolve();
      };
      
      window.speechSynthesis.speak(utterance);
    });
  }, [isSpeaking]);
  
  // Start recording audio
  const startRecording = async () => {
    try {
      setError(null);
      
      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Your browser does not support audio recording. Please use Chrome, Firefox, or Safari.');
      }
      
      // First check existing permissions
      if (navigator.permissions) {
        const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
        console.log('Microphone permission status:', permissionStatus.state);
        
        if (permissionStatus.state === 'denied') {
          throw new Error('Microphone access was denied. Please allow microphone access in your browser settings and refresh the page.');
        }
      }
      
      // Request microphone access with explicit constraints
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        }
      });
      
      streamRef.current = stream;
      setMicrophonePermission('granted');
      
      // Create MediaRecorder
      const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 
                      MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' : 
                      'audio/wav';
      
      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        processAudio();
      };
      
      mediaRecorder.start(1000); // Collect data every second
      setIsListening(true);
      
      console.log('Started recording with mimeType:', mimeType);
      
      // Auto-stop after 15 seconds
      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          stopRecording();
        }
      }, 15000);
      
    } catch (err) {
      console.error('Error accessing microphone:', err);
      
      let errorMessage = 'Failed to access microphone. ';
      
      if (err.name === 'NotAllowedError') {
        errorMessage += 'Please allow microphone access and try again.';
        setMicrophonePermission('denied');
      } else if (err.name === 'NotFoundError') {
        errorMessage += 'No microphone found. Please connect a microphone and try again.';
      } else if (err.name === 'NotSupportedError') {
        errorMessage += 'Your browser does not support audio recording.';
      } else {
        errorMessage += err.message || 'Please check your microphone settings.';
      }
      
      setError(errorMessage);
    }
  };
  
  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    setIsListening(false);
  };
  
  // Process recorded audio
  const processAudio = async () => {
    if (chunksRef.current.length === 0) {
      setError('No audio recorded. Please try speaking louder or closer to the microphone.');
      return;
    }
    
    setIsProcessing(true);
    
    try {
      const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
      
      console.log('Processing audio blob:', audioBlob.size, 'bytes');
      
      // Check if audio is substantial
      if (audioBlob.size < 1000) {
        setError('Audio too short. Please speak for a longer duration.');
        setIsProcessing(false);
        return;
      }
      
      const formData = new FormData();
      formData.append('audio', audioBlob, 'audio.webm');
      formData.append('session_id', sessionId);
      
      // Send to backend for transcription
      const response = await fetch(`${API_BASE_URL}/api/transcribe`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.success && data.transcription) {
        console.log('Transcription received:', data.transcription);
        handleUserMessage(data.transcription);
      } else {
        setError(data.error || 'Could not transcribe audio. Please try speaking more clearly.');
      }
      
    } catch (err) {
      console.error('Error processing audio:', err);
      
      if (err.message.includes('Failed to fetch')) {
        setError('Failed to connect to server. Please check if the backend is running on port 5000.');
      } else {
        setError('Failed to process audio. Please check your internet connection and try again.');
      }
    } finally {
      setIsProcessing(false);
      chunksRef.current = [];
    }
  };
  
  // Handle user message
  const handleUserMessage = async (transcript) => {
    const userMessage = {
      id: Date.now(),
      text: transcript,
      isUser: true,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    
    try {
      // Send to backend for analysis
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          transcript,
          session_id: sessionId
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setStructuredNotes(data.notes);
        
        const aiMessage = {
          id: Date.now() + 1,
          text: data.analysis.follow_up_question,
          isUser: false,
          timestamp: new Date().toISOString()
        };
        
        setMessages(prev => [...prev, aiMessage]);
        
        // Speak the AI response
        await speakText(data.analysis.follow_up_question);
        
        if (data.analysis.completion_ready) {
          setCurrentPhase('ready-for-sop');
        }
      } else {
        setError(data.error || 'Failed to analyze response. Please try again.');
      }
    } catch (err) {
      console.error('Error analyzing response:', err);
      
      if (err.message.includes('Failed to fetch')) {
        setError('Failed to connect to server. Please check if the backend is running.');
      } else {
        setError('Failed to analyze response. Please try again.');
      }
    }
  };
  
  // Start conversation
  const startConversation = async () => {
    if (microphonePermission !== 'granted') {
      setError('Please test and allow microphone access first.');
      return;
    }
    
    setConversationStarted(true);
    setCurrentPhase('conversation');
    
    const welcomeMessage = "Hello! I'm your AI assistant for creating a Statement of Purpose. I'll help you structure your thoughts and experiences into a compelling personal statement. Let's start by talking about a significant challenge or turning point in your life that taught you something important. Click the microphone button when you're ready to share your story.";
    
    const aiMessage = {
      id: Date.now(),
      text: welcomeMessage,
      isUser: false,
      timestamp: new Date().toISOString()
    };
    
    setMessages([aiMessage]);
    await speakText(welcomeMessage);
  };
  
  // Generate SOP
  const generateSOP = async () => {
    setCurrentPhase('generating');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-sop`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          session_id: sessionId
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setCurrentPhase('sop');
        // For now, show in alert - you can create a proper SOP display component
        alert(`SOP Generated Successfully!\n\nWord Count: ${data.word_count}\n\nCheck the browser console for the full text.`);
        console.log('Generated SOP:', data.sop);
      } else {
        setError(data.error || 'Failed to generate SOP. Please try again.');
        setCurrentPhase('conversation');
      }
    } catch (err) {
      console.error('Error generating SOP:', err);
      setError('Failed to generate SOP. Please check your connection.');
      setCurrentPhase('conversation');
    }
  };
  
  // Reset error and go back to welcome
  const resetError = () => {
    setError(null);
    setCurrentPhase('welcome');
    setMessages([]);
    setStructuredNotes({
      challenges: '',
      lessons: '',
      growth: '',
      college: ''
    });
  };
  
  // Render error state
  if (error) {
    return (
      <div className="app">
        <div className="error-container">
          <div className="error-content">
            <div className="error-icon">⚠️</div>
            <h2>Oops! Something went wrong</h2>
            <p>{error}</p>
            <div className="error-actions">
              <button className="retry-btn" onClick={resetError}>
                Try Again
              </button>
              {microphonePermission !== 'granted' && (
                <button className="permission-btn" onClick={testMicrophoneAccess}>
                  Test Microphone
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  // Render welcome screen
  if (currentPhase === 'welcome') {
    return (
      <div className="app">
        <div className="welcome-container">
          <div className="welcome-content">
            <div className="header-section">
              <h1>🎤 Voice SOP Generator</h1>
              <p className="subtitle">Create your Statement of Purpose through natural conversation with AI</p>
            </div>
            
            <div className="features-section">
              <div className="feature-card">
                <div className="feature-icon">🗣️</div>
                <h3>Voice-First</h3>
                <p>Speak naturally and let AI guide your conversation</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">🧠</div>
                <h3>AI-Powered</h3>
                <p>Advanced analysis extracts key insights from your responses</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">📝</div>
                <h3>Professional</h3>
                <p>Generate polished, compelling statements ready for submission</p>
              </div>
            </div>
            
            {/* Microphone Permission Section */}
            <div className="permission-section">
              <div className={`permission-status ${microphonePermission}`}>
                {microphonePermission === 'granted' && (
                  <>
                    <span className="status-icon">✅</span>
                    <span>Microphone access granted</span>
                  </>
                )}
                {microphonePermission === 'denied' && (
                  <>
                    <span className="status-icon">❌</span>
                    <span>Microphone access denied</span>
                  </>
                )}
                {microphonePermission === 'unknown' && (
                  <>
                    <span className="status-icon">🎤</span>
                    <span>Microphone access required</span>
                  </>
                )}
              </div>
              
              {microphonePermission !== 'granted' && (
                <button className="permission-button" onClick={testMicrophoneAccess}>
                  🎤 Test Microphone Access
                </button>
              )}
            </div>
            
            <button 
              className="start-btn" 
              onClick={startConversation}
              disabled={microphonePermission !== 'granted'}
            >
              Start Voice Interview
              <span className="arrow">→</span>
            </button>
            
            <div className="instructions">
              <p>✓ Works on all browsers • ✓ Mobile friendly • ✓ Secure & private</p>
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  // Render generating screen
  if (currentPhase === 'generating') {
    return (
      <div className="app">
        <div className="generating-container">
          <div className="generating-content">
            <div className="spinner-large"></div>
            <h2>Generating Your Statement of Purpose</h2>
            <p>Analyzing your responses and crafting a personalized SOP...</p>
          </div>
        </div>
      </div>
    );
  }
  
  // Render conversation interface
  return (
    <div className="app">
      <div className="conversation-container">
        {/* Header */}
        <div className="header">
          <div className="header-left">
            <h2>Voice SOP Generator</h2>
            <div className="status-indicator">
              {isListening && <span className="status listening">🎤 Listening...</span>}
              {isSpeaking && <span className="status speaking">🗣️ Speaking...</span>}
              {isProcessing && <span className="status processing">⚙️ Processing...</span>}
              {!isListening && !isSpeaking && !isProcessing && <span className="status ready">Ready</span>}
            </div>
          </div>
          <div className="progress-info">
            <span>{messages.filter(m => m.isUser).length} responses</span>
          </div>
        </div>
        
        {/* Main Content */}
        <div className="main-content">
          {/* Conversation Panel */}
          <div className="conversation-panel">
            <div className="messages-container">
              {messages.length === 0 ? (
                <div className="empty-state">
                  <div className="avatar ai-avatar">🤖</div>
                  <div className="empty-message">
                    <p>Ready to start your conversation!</p>
                    <p>Click the microphone button below to begin.</p>
                  </div>
                </div>
              ) : (
                messages.map(message => (
                  <div key={message.id} className={`message ${message.isUser ? 'user-message' : 'ai-message'}`}>
                    <div className="message-header">
                      <div className={`avatar ${message.isUser ? 'user-avatar' : 'ai-avatar'}`}>
                        {message.isUser ? '👤' : '🤖'}
                      </div>
                      <div className="message-info">
                        <span className="sender">{message.isUser ? 'You' : 'AI Assistant'}</span>
                        <span className="timestamp">
                          {new Date(message.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                        </span>
                      </div>
                    </div>
                    <div className="message-content">
                      {message.text}
                    </div>
                  </div>
                ))
              )}
              <div ref={messageEndRef} />
            </div>
            
            {/* Controls */}
            <div className="controls-container">
              <div className="voice-controls">
                <button 
                  className={`voice-btn ${isListening ? 'listening' : ''} ${isProcessing ? 'processing' : ''}`}
                  onClick={isListening ? stopRecording : startRecording}
                  disabled={isProcessing || isSpeaking}
                >
                  {isListening ? (
                    <>🛑 Stop Recording</>
                  ) : isProcessing ? (
                    <>⚙️ Processing...</>
                  ) : (
                    <>🎤 Start Recording</>
                  )}
                </button>
                
                {currentPhase === 'ready-for-sop' && (
                  <button className="generate-btn" onClick={generateSOP}>
                    📝 Generate SOP
                  </button>
                )}
              </div>
              
              <div className="instructions-text">
                {isListening ? (
                  <p>🎤 Listening... Speak clearly about your experiences.</p>
                ) : isProcessing ? (
                  <p>⚙️ Processing your audio...</p>
                ) : (
                  <p>Click the microphone to record your response. Speak for 5-15 seconds.</p>
                )}
              </div>
            </div>
          </div>
          
          {/* Notes Panel */}
          <div className="notes-panel">
            <div className="notes-header">
              <h3>📝 Your Notes</h3>
              <p>Key insights extracted from your conversation</p>
            </div>
            
            <div className="notes-content">
              <div className="note-section challenges">
                <div className="note-header">
                  <span className="note-icon">🎯</span>
                  <h4>Challenges & Turning Points</h4>
                </div>
                <div className="note-text">
                  {structuredNotes.challenges || "Share your challenges and how they shaped you..."}
                </div>
              </div>
              
              <div className="note-section lessons">
                <div className="note-header">
                  <span className="note-icon">💡</span>
                  <h4>Lessons Learned</h4>
                </div>
                <div className="note-text">
                  {structuredNotes.lessons || "What important lessons did you learn?"}
                </div>
              </div>
              
              <div className="note-section growth">
                <div className="note-header">
                  <span className="note-icon">🌱</span>
                  <h4>Personal Growth</h4>
                </div>
                <div className="note-text">
                  {structuredNotes.growth || "How have you grown and changed?"}
                </div>
              </div>
              
              <div className="note-section college">
                <div className="note-header">
                  <span className="note-icon">🎓</span>
                  <h4>College Connection</h4>
                </div>
                <div className="note-text">
                  {structuredNotes.college || "How does this connect to your college goals?"}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;