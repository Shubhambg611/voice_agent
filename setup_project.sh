#!/bin/bash

# Voice SOP Generator - Project Structure Setup Script
# This script creates the complete production-ready project structure

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_NAME="voice-sop-generator"
PROJECT_DESCRIPTION="AI-powered Voice-operated Statement of Purpose Generator"

echo -e "${BLUE}🚀 Setting up ${PROJECT_NAME}...${NC}"
echo -e "${CYAN}Description: ${PROJECT_DESCRIPTION}${NC}"
echo ""

# Function to create directory and log it
create_dir() {
    mkdir -p "$1"
    echo -e "${GREEN}✓${NC} Created directory: $1"
}

# Function to create file with content and log it
create_file() {
    local file_path="$1"
    local content="$2"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$file_path")"
    
    # Create file with content
    echo "$content" > "$file_path"
    echo -e "${GREEN}✓${NC} Created file: $file_path"
}

echo -e "${YELLOW}📁 Creating project structure...${NC}"

# Create main project directory
create_dir "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Backend structure
echo -e "${PURPLE}🐍 Setting up backend structure...${NC}"
create_dir "backend"
create_dir "backend/config"
create_dir "backend/utils"
create_dir "backend/api"
create_dir "backend/models"
create_dir "backend/tests"
create_dir "backend/logs"
create_dir "backend/temp"

# Frontend structure
echo -e "${CYAN}⚛️ Setting up frontend structure...${NC}"
create_dir "frontend"
create_dir "frontend/public"
create_dir "frontend/public/icons"
create_dir "frontend/src"
create_dir "frontend/src/components"
create_dir "frontend/src/hooks"
create_dir "frontend/src/services"
create_dir "frontend/src/utils"
create_dir "frontend/src/contexts"
create_dir "frontend/src/styles"
create_dir "frontend/build"

# Deployment structure
echo -e "${BLUE}🚢 Setting up deployment structure...${NC}"
create_dir "deployment"
create_dir "deployment/nginx"
create_dir "deployment/nginx/ssl"
create_dir "deployment/docker"
create_dir "deployment/kubernetes"
create_dir "deployment/terraform"
create_dir "deployment/scripts"

# Monitoring structure
echo -e "${YELLOW}📊 Setting up monitoring structure...${NC}"
create_dir "monitoring"
create_dir "monitoring/prometheus"
create_dir "monitoring/grafana"
create_dir "monitoring/grafana/dashboards"
create_dir "monitoring/alerts"

# Documentation structure
echo -e "${GREEN}📚 Setting up documentation structure...${NC}"
create_dir "docs"
create_dir "scripts"
create_dir ".github"
create_dir ".github/workflows"

echo -e "${YELLOW}📝 Creating configuration files...${NC}"

# Backend files
create_file "backend/app.py" "# app.py - Production-ready Flask backend with Whisper + Gemini integration
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import redis
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import openai
import google.generativeai as genai
from werkzeug.utils import secure_filename
import tempfile
import uuid
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# AI Provider configuration
AI_PROVIDER = os.getenv('AI_PROVIDER', 'gemini').lower()

# Initialize AI clients based on provider
if AI_PROVIDER == 'openai':
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError('OPENAI_API_KEY is required when using OpenAI provider')
elif AI_PROVIDER == 'gemini':
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        raise ValueError('GEMINI_API_KEY is required when using Gemini provider')
    genai.configure(api_key=gemini_api_key)
else:
    raise ValueError(f'Unsupported AI provider: {AI_PROVIDER}')

# For Whisper transcription, we'll still use OpenAI (most reliable for speech-to-text)
whisper_api_key = os.getenv('OPENAI_API_KEY')
if whisper_api_key:
    openai.api_key = whisper_api_key
else:
    logger.warning('OPENAI_API_KEY not set - Whisper transcription will not be available')

# Redis for session management
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

class AIClient:
    \"\"\"Unified AI client that works with both OpenAI and Gemini\"\"\"
    
    def __init__(self, provider: str = 'gemini'):
        self.provider = provider.lower()
        
        if self.provider == 'gemini':
            self.model = genai.GenerativeModel('gemini-pro')
        elif self.provider == 'openai':
            # OpenAI client is already initialized globally
            pass
        else:
            raise ValueError(f'Unsupported provider: {provider}')
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        \"\"\"Generate text using the configured AI provider\"\"\"
        try:
            if self.provider == 'gemini':
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    )
                )
                return response.text
            
            elif self.provider == 'openai':
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f'Error generating text with {self.provider}: {e}')
            raise

class SessionManager:
    \"\"\"Thread-safe session management with Redis backend\"\"\"
    
    def __init__(self):
        self.local_cache = {}
        self.cache_lock = Lock()
        self.session_timeout = 3600  # 1 hour
    
    def create_session(self, session_id: str) -> Dict:
        \"\"\"Create new session with default data\"\"\"
        session_data = {
            'notes': {
                'challenges': '',
                'lessons': '',
                'growth': '',
                'college': ''
            },
            'conversation_history': [],
            'current_topic': 'challenges',
            'questions_asked': 0,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat()
        }
        
        # Store in Redis with expiration
        redis_client.setex(
            f'session:{session_id}', 
            self.session_timeout, 
            json.dumps(session_data)
        )
        
        # Cache locally for quick access
        with self.cache_lock:
            self.local_cache[session_id] = session_data
        
        return session_data
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        \"\"\"Get session data with fallback to Redis\"\"\"
        with self.cache_lock:
            if session_id in self.local_cache:
                return self.local_cache[session_id]
        
        # Fallback to Redis
        session_json = redis_client.get(f'session:{session_id}')
        if session_json:
            session_data = json.loads(session_json)
            with self.cache_lock:
                self.local_cache[session_id] = session_data
            return session_data
        
        return None
    
    def update_session(self, session_id: str, data: Dict):
        \"\"\"Update session data in both cache and Redis\"\"\"
        data['last_activity'] = datetime.now().isoformat()
        
        # Update Redis
        redis_client.setex(
            f'session:{session_id}', 
            self.session_timeout, 
            json.dumps(data)
        )
        
        # Update local cache
        with self.cache_lock:
            self.local_cache[session_id] = data

# Initialize components
session_manager = SessionManager()
ai_client = AIClient(AI_PROVIDER)

def create_app():
    \"\"\"Application factory for production deployment\"\"\"
    app = Flask(__name__)
    
    # Production configuration
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY', 'production-secret-key-change-me'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
        UPLOAD_FOLDER=tempfile.gettempdir(),
        WTF_CSRF_ENABLED=False,  # Disabled for API
        JSON_SORT_KEYS=False
    )
    
    # CORS configuration for production
    CORS(app, 
         origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
         methods=['GET', 'POST'],
         allow_headers=['Content-Type', 'Authorization'])
    
    # Rate limiting
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        storage_uri=f'redis://{os.getenv(\"REDIS_HOST\", \"localhost\")}:{os.getenv(\"REDIS_PORT\", 6379)}',
        default_limits=['1000 per hour', '100 per minute']
    )
    
    @app.before_request
    def before_request():
        \"\"\"Security and logging middleware\"\"\"
        logger.info(f'{request.method} {request.path} from {request.remote_addr}')
        
        if request.endpoint and request.endpoint.startswith('api.'):
            if not request.is_json and request.method == 'POST' and 'multipart/form-data' not in request.content_type:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    @app.after_request
    def after_request(response):
        \"\"\"Add security headers\"\"\"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response
    
    @app.route('/health')
    def health_check():
        \"\"\"Health check endpoint for load balancers\"\"\"
        try:
            # Test Redis connection
            redis_client.ping()
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'ai_provider': AI_PROVIDER,
                'services': {
                    'redis': 'connected',
                    'ai': 'available'
                }
            })
        except Exception as e:
            logger.error(f'Health check failed: {e}')
            return jsonify({
                'status': 'unhealthy',
                'error': str(e)
            }), 503
    
    @app.route('/api/transcribe', methods=['POST'])
    @limiter.limit('60 per minute')
    def transcribe_audio():
        \"\"\"Transcribe audio using OpenAI Whisper\"\"\"
        try:
            if not whisper_api_key:
                return jsonify({
                    'success': False,
                    'error': 'Whisper API not available'
                }), 503
            
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400
            
            audio_file = request.files['audio']
            session_id = request.form.get('session_id')
            
            if not session_id:
                return jsonify({'error': 'Session ID required'}), 400
            
            if audio_file.filename == '':
                return jsonify({'error': 'No audio file selected'}), 400
            
            # Save audio file temporarily
            filename = secure_filename(f'{uuid.uuid4()}.webm')
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(temp_path)
            
            try:
                # Transcribe with Whisper
                with open(temp_path, 'rb') as audio:
                    transcript = openai.Audio.transcribe(
                        'whisper-1', 
                        audio,
                        language='en'
                    )
                
                transcription = transcript.text.strip()
                
                if transcription:
                    logger.info(f'Transcribed for session {session_id}: {transcription[:50]}...')
                    return jsonify({
                        'success': True,
                        'transcription': transcription,
                        'session_id': session_id
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No speech detected'
                    }), 400
                    
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            logger.error(f'Transcription error: {e}')
            return jsonify({
                'success': False,
                'error': 'Transcription failed'
            }), 500
    
    @app.route('/api/analyze', methods=['POST'])
    @limiter.limit('30 per minute')
    def analyze_response():
        \"\"\"Analyze user response and generate follow-up\"\"\"
        try:
            data = request.get_json()
            transcript = data.get('transcript', '').strip()
            session_id = data.get('session_id')
            
            if not transcript or not session_id:
                return jsonify({'error': 'Missing transcript or session_id'}), 400
            
            # Get or create session
            session = session_manager.get_session(session_id)
            if not session:
                session = session_manager.create_session(session_id)
            
            # Add to conversation history
            session['conversation_history'].append({
                'role': 'user',
                'content': transcript,
                'timestamp': datetime.now().isoformat()
            })
            
            # Analyze with AI
            analysis = analyze_with_ai(transcript, session)
            
            # Update session notes
            if analysis.get('extracted_info'):
                for category, content in analysis['extracted_info'].items():
                    if content.strip():
                        current = session['notes'][category]
                        if current:
                            session['notes'][category] = f'{current}\\n• {content}'
                        else:
                            session['notes'][category] = f'• {content}'
            
            # Update session
            session['questions_asked'] += 1
            if analysis.get('next_topic'):
                session['current_topic'] = analysis['next_topic']
            
            session_manager.update_session(session_id, session)
            
            return jsonify({
                'success': True,
                'analysis': analysis,
                'notes': session['notes'],
                'session_id': session_id
            })
            
        except Exception as e:
            logger.error(f'Analysis error: {e}')
            return jsonify({
                'success': False,
                'error': 'Analysis failed'
            }), 500
    
    @app.route('/api/generate-sop', methods=['POST'])
    @limiter.limit('10 per hour')
    def generate_sop():
        \"\"\"Generate Statement of Purpose\"\"\"
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            
            if not session_id:
                return jsonify({'error': 'Session ID required'}), 400
            
            session = session_manager.get_session(session_id)
            if not session:
                return jsonify({'error': 'Session not found'}), 404
            
            notes = session['notes']
            
            # Validate that we have enough content
            total_content = sum(len(note.strip()) for note in notes.values())
            if total_content < 100:
                return jsonify({
                    'error': 'Insufficient content for SOP generation'
                }), 400
            
            # Generate SOP with AI
            sop = generate_sop_with_ai(notes)
            
            # Store generated SOP in session
            session['generated_sop'] = sop
            session['sop_generated_at'] = datetime.now().isoformat()
            session_manager.update_session(session_id, session)
            
            return jsonify({
                'success': True,
                'sop': sop,
                'word_count': len(sop.split()),
                'session_id': session_id
            })
            
        except Exception as e:
            logger.error(f'SOP generation error: {e}')
            return jsonify({
                'success': False,
                'error': 'SOP generation failed'
            }), 500
    
    return app

def analyze_with_ai(transcript: str, session: Dict) -> Dict:
    \"\"\"Analyze user response using configured AI provider\"\"\"
    try:
        # Get conversation context
        recent_history = session['conversation_history'][-5:]
        history_text = '\\n'.join([
            f'{msg[\"role\"].title()}: {msg[\"content\"]}' 
            for msg in recent_history
        ])
        
        current_topic = session['current_topic']
        questions_asked = session['questions_asked']
        
        prompt = f'''
You are helping an international student create a Statement of Purpose. 
Analyze their response and guide the conversation.

Current topic: {current_topic}
Questions asked: {questions_asked}
Recent conversation:
{history_text}

Latest response: \"{transcript}\"

Analyze and return JSON:
{{
    \"extracted_info\": {{
        \"challenges\": \"relevant challenges/turning points\",
        \"lessons\": \"lessons learned\", 
        \"growth\": \"personal/emotional growth\",
        \"college\": \"college-specific interests\"
    }},
    \"follow_up_question\": \"next question to ask\",
    \"should_change_topic\": boolean,
    \"next_topic\": \"challenges|lessons|growth|college\",
    \"response_quality\": \"high|medium|low\",
    \"completion_ready\": boolean
}}

Guidelines:
- Extract specific, detailed information
- Ask follow-up questions for deeper insights
- Guide toward college connection
- After 10+ exchanges, suggest completion
'''

        response_text = ai_client.generate_text(prompt, max_tokens=800, temperature=0.7)
        
        # Try to parse JSON from response
        try:
            # Clean the response to extract JSON
            import re
            json_match = re.search(r'\\{.*\\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
            else:
                # If no JSON found, try parsing the whole response
                result = json.loads(response_text)
                return result
        except json.JSONDecodeError:
            logger.warning(f'Could not parse AI response as JSON: {response_text}')
            # Return fallback response
            return create_fallback_analysis(transcript, session)
        
    except Exception as e:
        logger.error(f'AI analysis error: {e}')
        return create_fallback_analysis(transcript, session)

def create_fallback_analysis(transcript: str, session: Dict) -> Dict:
    \"\"\"Create fallback analysis when AI fails\"\"\"
    questions_asked = session['questions_asked']
    current_topic = session['current_topic']
    
    # Simple keyword-based analysis
    extracted_info = {
        'challenges': transcript if any(word in transcript.lower() for word in ['challeng', 'difficult', 'hard', 'struggle']) else '',
        'lessons': transcript if any(word in transcript.lower() for word in ['learn', 'lesson', 'realize', 'understand']) else '',
        'growth': transcript if any(word in transcript.lower() for word in ['grow', 'feel', 'chang', 'develop']) else '',
        'college': transcript if any(word in transcript.lower() for word in ['college', 'university', 'school', 'education']) else ''
    }
    
    # Basic follow-up questions
    follow_up_questions = {
        'challenges': 'Could you tell me more about the specific challenges you faced?',
        'lessons': 'What important lessons did you learn from this experience?',
        'growth': 'How did this experience change you personally or emotionally?',
        'college': 'How does this connect to your interest in the college you are applying to?'
    }
    
    return {
        'extracted_info': extracted_info,
        'follow_up_question': follow_up_questions.get(current_topic, 'Could you tell me more about that?'),
        'should_change_topic': questions_asked > 3,
        'next_topic': current_topic,
        'response_quality': 'medium',
        'completion_ready': questions_asked > 8
    }

def generate_sop_with_ai(notes: Dict) -> str:
    \"\"\"Generate Statement of Purpose using configured AI provider\"\"\"
    try:
        prompt = f'''
Generate a compelling Statement of Purpose (700-900 words) based on these notes:

CHALLENGES/TURNING POINTS:
{notes['challenges']}

LESSONS LEARNED:
{notes['lessons']}

PERSONAL GROWTH:
{notes['growth']}

COLLEGE CONNECTION:
{notes['college']}

Create a well-structured, authentic SOP that:
1. Opens with an engaging hook
2. Connects experiences to academic/career goals
3. Shows personal growth and maturity
4. Demonstrates fit with chosen institution
5. Maintains formal yet personal tone
6. Includes specific examples and details

Structure: Introduction → Challenges/Growth → Lessons → College Fit → Conclusion
'''

        response = ai_client.generate_text(prompt, max_tokens=1500, temperature=0.8)
        return response.strip()
        
    except Exception as e:
        logger.error(f'SOP generation error: {e}')
        raise

def create_socketio(app):
    \"\"\"Create SocketIO instance for real-time communication\"\"\"
    socketio = SocketIO(
        app,
        cors_allowed_origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
        logger=True,
        engineio_logger=False,
        ping_timeout=60,
        ping_interval=25
    )
    
    @socketio.on('connect')
    def handle_connect():
        logger.info(f'Client connected: {request.sid}')
        
        # Create session
        session_data = session_manager.create_session(request.sid)
        
        emit('connected', {
            'session_id': request.sid,
            'status': 'ready',
            'ai_provider': AI_PROVIDER
        })
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f'Client disconnected: {request.sid}')
    
    @socketio.on('ping')
    def handle_ping():
        emit('pong')
    
    return socketio

if __name__ == '__main__':
    # Production setup
    app = create_app()
    socketio = create_socketio(app)
    
    # Run with production settings
    socketio.run(
        app,
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=False,
        use_reloader=False
    )"

create_file "backend/requirements.txt" "# Production Flask Backend Dependencies
Flask==2.3.3
Flask-SocketIO==5.3.6
Flask-CORS==4.0.0
Flask-Limiter==3.5.0

# AI Integration
openai==1.3.7
google-generativeai==0.3.2

# Redis for session management and caching
redis==5.0.1

# Production utilities
python-dotenv==1.0.0
gunicorn==21.2.0
gevent==23.9.1

# Security and monitoring
Werkzeug==2.3.7

# Audio processing (if needed for local processing)
pydub==0.25.1

# For deployment and scaling
psutil==5.9.6"

create_file "backend/.env.example" "# AI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
AI_PROVIDER=gemini

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Flask Configuration
FLASK_ENV=production
SECRET_KEY=your-super-secret-key-change-in-production

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# Rate Limiting
RATE_LIMIT_ENABLED=true
DEFAULT_RATE_LIMIT=1000 per hour

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Server Configuration
HOST=0.0.0.0
PORT=5000
WORKERS=4"

create_file "backend/config/__init__.py" ""

create_file "backend/config/development.py" "import os

class DevelopmentConfig:
    DEBUG = True
    TESTING = False
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    
    # AI Configuration
    AI_PROVIDER = os.getenv('AI_PROVIDER', 'gemini')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    ALLOWED_ORIGINS = ['http://localhost:3000']
    LOG_LEVEL = 'DEBUG'"

create_file "backend/config/production.py" "import os

class ProductionConfig:
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.getenv('SECRET_KEY')
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    
    # AI Configuration
    AI_PROVIDER = os.getenv('AI_PROVIDER', 'gemini')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '').split(',')
    LOG_LEVEL = 'INFO'
    
    # Production optimizations
    JSON_SORT_KEYS = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB"

create_file "backend/gunicorn.conf.py" "import os

# Server socket
bind = f\"0.0.0.0:{os.getenv('PORT', 5000)}\"
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', 4))
worker_class = 'gevent'
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests, to help control memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s \"%(r)s\" %(s)s %(b)s \"%(f)s\" \"%(a)s\" %(L)s'

# Process naming
proc_name = 'voice-sop-generator'

# Preload app for better performance
preload_app = True"

create_file "backend/utils/__init__.py" ""

create_file "backend/api/__init__.py" ""

create_file "backend/models/__init__.py" ""

create_file "backend/tests/__init__.py" ""

# Frontend files
create_file "frontend/package.json" "{
  \"name\": \"voice-sop-generator-frontend\",
  \"version\": \"1.0.0\",
  \"description\": \"Frontend for Voice-operated SOP Generator\",
  \"private\": true,
  \"dependencies\": {
    \"react\": \"^18.2.0\",
    \"react-dom\": \"^18.2.0\",
    \"react-scripts\": \"5.0.1\",
    \"socket.io-client\": \"^4.7.2\",
    \"axios\": \"^1.5.0\",
    \"web-vitals\": \"^3.4.0\"
  },
  \"scripts\": {
    \"start\": \"react-scripts start\",
    \"build\": \"react-scripts build\",
    \"test\": \"react-scripts test\",
    \"eject\": \"react-scripts eject\",
    \"lint\": \"eslint src/\",
    \"lint:fix\": \"eslint src/ --fix\"
  },
  \"eslintConfig\": {
    \"extends\": [
      \"react-app\",
      \"react-app/jest\"
    ]
  },
  \"browserslist\": {
    \"production\": [
      \">0.2%\",
      \"not dead\",
      \"not op_mini all\"
    ],
    \"development\": [
      \"last 1 chrome version\",
      \"last 1 firefox version\",
      \"last 1 safari version\"
    ]
  },
  \"devDependencies\": {
    \"@tailwindcss/forms\": \"^0.5.6\",
    \"autoprefixer\": \"^10.4.15\",
    \"eslint\": \"^8.48.0\",
    \"postcss\": \"^8.4.29\",
    \"tailwindcss\": \"^3.3.3\"
  }
}"

create_file "frontend/.env.example" "# API Configuration
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=ws://localhost:5000

# Environment
REACT_APP_ENV=development

# Features
REACT_APP_ENABLE_ANALYTICS=false
REACT_APP_ENABLE_ERROR_REPORTING=false

# AI Configuration (Choose one)
REACT_APP_AI_PROVIDER=gemini
REACT_APP_OPENAI_API_KEY=
REACT_APP_GEMINI_API_KEY=

# Debug
REACT_APP_DEBUG=true"

create_file "frontend/tailwind.config.js" "/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
    './public/index.html'
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          900: '#1e3a8a',
        },
        voice: {
          speaking: '#ef4444',
          listening: '#10b981',
          idle: '#6b7280',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-gentle': 'bounce 2s infinite',
        'speaking': 'speaking 0.6s ease-in-out infinite alternate',
        'listening': 'listening 2s ease-in-out infinite',
      },
      keyframes: {
        speaking: {
          '0%': { transform: 'scale(1)', opacity: '0.8' },
          '100%': { transform: 'scale(1.2)', opacity: '1' },
        },
        listening: {
          '0%, 100%': { transform: 'scale(1)', opacity: '0.7' },
          '50%': { transform: 'scale(1.05)', opacity: '1' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}"

create_file "frontend/postcss.config.js" "module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}"

create_file "frontend/public/manifest.json" "{
  \"short_name\": \"SOP Generator\",
  \"name\": \"Voice-operated SOP Generator\",
  \"description\": \"AI-powered Statement of Purpose Generator using voice interaction\",
  \"icons\": [
    {
      \"src\": \"icons/icon-192x192.png\",
      \"sizes\": \"192x192\",
      \"type\": \"image/png\"
    },
    {
      \"src\": \"icons/icon-512x512.png\",
      \"sizes\": \"512x512\",
      \"type\": \"image/png\"
    }
  ],
  \"start_url\": \".\",
  \"display\": \"standalone\",
  \"theme_color\": \"#3b82f6\",
  \"background_color\": \"#ffffff\",
  \"categories\": [\"education\", \"productivity\", \"utilities\"]
}"

# Docker files
create_file "deployment/docker/Dockerfile.backend" "FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD [\"gunicorn\", \"--config\", \"gunicorn.conf.py\", \"app:app\"]"

create_file "deployment/docker/Dockerfile.frontend" "# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built application
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY deployment/nginx/nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost/ || exit 1

CMD [\"nginx\", \"-g\", \"daemon off;\"]"

create_file "deployment/docker/docker-compose.prod.yml" "version: '3.8'

services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data
    networks:
      - backend
    healthcheck:
      test: [\"CMD\", \"redis-cli\", \"ping\"]
      interval: 30s
      timeout: 10s
      retries: 3

  backend:
    build:
      context: ../../backend
      dockerfile: ../deployment/docker/Dockerfile.backend
    restart: unless-stopped
    environment:
      - REDIS_HOST=redis
      - FLASK_ENV=production
    depends_on:
      - redis
    networks:
      - backend
      - frontend
    volumes:
      - ../../backend/logs:/app/logs
      - ../../backend/temp:/app/temp

  frontend:
    build:
      context: ../../frontend
      dockerfile: ../deployment/docker/Dockerfile.frontend
    restart: unless-stopped
    ports:
      - \"80:80\"
      - \"443:443\"
    depends_on:
      - backend
    networks:
      - frontend
    volumes:
      - ../../deployment/nginx/ssl:/etc/nginx/ssl:ro

networks:
  backend:
    driver: bridge
  frontend:
    driver: bridge

volumes:
  redis_data:"

# Nginx configuration
create_file "deployment/nginx/nginx.conf" "events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # Logging
    log_format main '\$remote_addr - \$remote_user [\$time_local] \"\$request\" '
                   '\$status \$body_bytes_sent \"\$http_referer\" '
                   '\"\$http_user_agent\" \"\$http_x_forwarded_for\"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone \$binary_remote_addr zone=transcribe:10m rate=5r/s;

    upstream backend {
        server backend:5000;
    }

    server {
        listen 80;
        server_name _;

        # Security headers
        add_header X-Frame-Options \"SAMEORIGIN\" always;
        add_header X-XSS-Protection \"1; mode=block\" always;
        add_header X-Content-Type-Options \"nosniff\" always;
        add_header Referrer-Policy \"no-referrer-when-downgrade\" always;
        add_header Content-Security-Policy \"default-src 'self' http: https: data: blob: 'unsafe-inline'\" always;

        # Frontend
        location / {
            root /usr/share/nginx/html;
            index index.html index.htm;
            try_files \$uri \$uri/ /index.html;
        }

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        # Transcription endpoint (stricter rate limiting)
        location /api/transcribe {
            limit_req zone=transcribe burst=10 nodelay;
            proxy_pass http://backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            
            # Increase timeouts for audio processing
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # WebSocket support
        location /socket.io/ {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection \"upgrade\";
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        # Health check
        location /health {
            proxy_pass http://backend;
            access_log off;
        }
    }
}"

# GitHub Actions
create_file ".github/workflows/ci.yml" "name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd \"redis-cli ping\"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        cd backend
        python -m pytest tests/
      env:
        REDIS_HOST: localhost
        OPENAI_API_KEY: \${{ secrets.OPENAI_API_KEY }}

  test-frontend:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: frontend/package-lock.json
    
    - name: Install dependencies
      run: |
        cd frontend
        npm ci
    
    - name: Run tests
      run: |
        cd frontend
        npm test -- --coverage --watchAll=false
    
    - name: Build
      run: |
        cd frontend
        npm run build"

# Main documentation
create_file "README.md" "# Voice-Operated SOP Generator

## 🎤 AI-Powered Statement of Purpose Generator

A production-ready web application that helps international students create compelling Statements of Purpose through natural voice conversation.

### ✨ Features

- **🎯 Voice-First Interface**: Speak naturally, AI understands and guides
- **🧠 AI-Powered Analysis**: Advanced conversation flow with GPT-4
- **🌍 Universal Compatibility**: Works on all browsers via Whisper API
- **📱 Mobile Responsive**: Optimized for all devices
- **⚡ Real-Time Processing**: Live transcription and feedback
- **🎨 Professional Output**: Polished, ready-to-submit SOPs

### 🏗️ Architecture

- **Frontend**: React.js with advanced voice processing
- **Backend**: Flask with OpenAI Whisper & GPT-4 integration
- **Database**: Redis for session management
- **Deployment**: Docker, Kubernetes, Nginx
- **Monitoring**: Prometheus, Grafana

### 🚀 Quick Start

\`\`\`bash
# Clone and setup
git clone <repository-url>
cd voice-sop-generator
chmod +x scripts/setup.sh
./scripts/setup.sh

# Start development
./scripts/start-dev.sh
\`\`\`

### 📊 Production Deployment

\`\`\`bash
# Deploy with Docker Compose
docker-compose -f deployment/docker/docker-compose.prod.yml up -d

# Or with Kubernetes
kubectl apply -f deployment/kubernetes/
\`\`\`

### 🔧 Configuration

Copy \`.env.example\` files and configure:

- OpenAI API key for Whisper & GPT-4
- Redis connection settings
- CORS origins for production

### 📚 Documentation

- [API Documentation](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Architecture Overview](docs/ARCHITECTURE.md)

### 🤝 Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

Built with ❤️ for international students worldwide"

# Setup script
create_file "scripts/setup.sh" "#!/bin/bash

# Voice SOP Generator - Development Setup Script

set -e

echo \"🚀 Setting up Voice SOP Generator for development...\"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo \"❌ Python 3 is required but not installed.\"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo \"❌ Node.js is required but not installed.\"
    exit 1
fi

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    echo \"⚠️ Redis is not installed. Please install Redis for session management.\"
    echo \"Ubuntu/Debian: sudo apt install redis-server\"
    echo \"macOS: brew install redis\"
    echo \"Windows: Download from https://redis.io/download\"
fi

# Setup backend
echo \"🐍 Setting up backend...\"
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo \"📝 Created backend/.env - please configure your API keys\"
fi

cd ..

# Setup frontend
echo \"⚛️ Setting up frontend...\"
cd frontend

# Install dependencies
npm install

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo \"📝 Created frontend/.env - please configure your settings\"
fi

cd ..

echo \"✅ Setup complete!\"
echo \"\"
echo \"🔧 Configuration needed:\"
echo \"1. Backend: Configure backend/.env with your API keys:\"
echo \"   - GEMINI_API_KEY=your_gemini_key_here\"
echo \"   - OPENAI_API_KEY=your_openai_key_here (for Whisper)\"
echo \"   - AI_PROVIDER=gemini\"
echo \"2. Frontend: Configure frontend/.env with your settings\"
echo \"\"
echo \"🚀 To start development:\"
echo \"1. Start Redis: redis-server\"
echo \"2. Start backend: cd backend && source venv/bin/activate && python app.py\"
echo \"3. Start frontend: cd frontend && npm start\"
echo \"\"
echo \"🎯 Access your app at: http://localhost:3000\"
echo \"Happy coding! 🎉\""

create_file "scripts/start-dev.sh" "#!/bin/bash

# Development startup script

set -e

echo \"🚀 Starting Voice SOP Generator in development mode...\"

# Check if Redis is running
if ! pgrep redis-server > /dev/null; then
    echo \"Starting Redis...\"
    redis-server --daemonize yes
fi

# Start backend in background
echo \"🐍 Starting backend...\"
cd backend
source venv/bin/activate
python app.py &
BACKEND_PID=\$!
cd ..

# Start frontend
echo \"⚛️ Starting frontend...\"
cd frontend
npm start &
FRONTEND_PID=\$!
cd ..

echo \"\"
echo \"✅ Development servers started!\"
echo \"Frontend: http://localhost:3000\"
echo \"Backend: http://localhost:5000\"
echo \"\"
echo \"Press Ctrl+C to stop all servers\"

# Wait for interrupt signal
trap \"echo 'Stopping servers...'; kill \$BACKEND_PID \$FRONTEND_PID; exit\" INT
wait"

# Git ignore
create_file ".gitignore" "# Dependencies
node_modules/
backend/venv/
backend/__pycache__/
backend/**/__pycache__/

# Environment variables
.env
.env.local
.env.production

# Logs
*.log
backend/logs/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Build outputs
frontend/build/
backend/dist/

# Temporary files
backend/temp/
*.tmp
*.temp

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Coverage
coverage/
*.coverage

# Distribution
*.tar.gz
*.zip

# SSL certificates (keep structure, ignore actual certs)
deployment/nginx/ssl/*.pem
deployment/nginx/ssl/*.key
deployment/nginx/ssl/*.crt

# Terraform
deployment/terraform/*.tfstate
deployment/terraform/*.tfstate.backup
deployment/terraform/.terraform/

# Monitoring data
monitoring/grafana/data/
monitoring/prometheus/data/"

# Docker ignore
create_file ".dockerignore" "node_modules
npm-debug.log
.git
.gitignore
README.md
.env
.env.example
coverage
.nyc_output
*.log
.vscode
.idea"

# Make scripts executable
chmod +x scripts/*.sh

echo ""
echo -e "${GREEN}🎉 Project structure created successfully!${NC}"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo -e "${CYAN}1.${NC} cd $PROJECT_NAME"
echo -e "${CYAN}2.${NC} Configure .env files:"
echo -e "${PURPLE}   Backend:${NC} GEMINI_API_KEY, OPENAI_API_KEY (for Whisper)"
echo -e "${PURPLE}   Frontend:${NC} API URLs and settings"
echo -e "${CYAN}3.${NC} Run: ./scripts/setup.sh"
echo -e "${CYAN}4.${NC} Run: ./scripts/start-dev.sh"
echo ""
echo -e "${GREEN}🎯 Features:${NC}"
echo -e "${CYAN}•${NC} Gemini for conversation & SOP generation"
echo -e "${CYAN}•${NC} OpenAI Whisper for speech transcription"
echo -e "${CYAN}•${NC} Works on all browsers (Chrome, Firefox, Safari, Edge)"
echo -e "${CYAN}•${NC} Real-time voice processing with visual feedback"
echo -e "${CYAN}•${NC} Production-ready with Docker & Kubernetes support"
echo ""
echo -e "${BLUE}📚 Documentation:${NC}"
echo -e "${CYAN}•${NC} Main README: README.md"
echo -e "${CYAN}•${NC} API docs: docs/API.md"
echo -e "${CYAN}•${NC} Deployment: docs/DEPLOYMENT.md"
echo ""
echo -e "${GREEN}✨ Ready to build something amazing!${NC}"