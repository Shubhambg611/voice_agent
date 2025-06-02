# app.py - Production-ready Flask backend with Whisper + GPT-4 integration
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import redis
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, request as socket_request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import openai
from werkzeug.utils import secure_filename
import tempfile
import uuid
from threading import Lock
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is required")

# Redis for session management
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

class SessionManager:
    """Thread-safe session management with Redis backend"""
    
    def __init__(self):
        self.local_cache = {}
        self.cache_lock = Lock()
        self.session_timeout = 3600  # 1 hour
    
    def create_session(self, session_id: str) -> Dict:
        """Create new session with default data"""
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
            'last_activity': datetime.now().isoformat(),
            'academicBackground': [],
            'researchExperience': [],
            'careerGoals': [],
            'programFit': []
        }
        
        # Store in Redis with expiration
        redis_client.setex(
            f"session:{session_id}", 
            self.session_timeout, 
            json.dumps(session_data)
        )
        
        # Cache locally for quick access
        with self.cache_lock:
            self.local_cache[session_id] = session_data
        
        return session_data
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data with fallback to Redis"""
        with self.cache_lock:
            if session_id in self.local_cache:
                return self.local_cache[session_id]
        
        # Fallback to Redis
        session_json = redis_client.get(f"session:{session_id}")
        if session_json:
            session_data = json.loads(session_json)
            with self.cache_lock:
                self.local_cache[session_id] = session_data
            return session_data
        
        return None
    
    def update_session(self, session_id: str, data: Dict):
        """Update session data in both cache and Redis"""
        data['last_activity'] = datetime.now().isoformat()
        
        # Update Redis
        redis_client.setex(
            f"session:{session_id}", 
            self.session_timeout, 
            json.dumps(data)
        )
        
        # Update local cache
        with self.cache_lock:
            self.local_cache[session_id] = data
    
    def cleanup_sessions(self):
        """Remove expired sessions from local cache"""
        expired_sessions = []
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        with self.cache_lock:
            for session_id, data in self.local_cache.items():
                last_activity = datetime.fromisoformat(data.get('last_activity', ''))
                if last_activity < cutoff_time:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.local_cache[session_id]
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Initialize session manager
session_manager = SessionManager()

def create_app():
    """Application factory for production deployment"""
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
        key_func=get_remote_address,
        default_limits=["1000 per hour", "100 per minute"]
    )
    limiter.init_app(app)
    
    @app.before_request
    def before_request():
        """Security and logging middleware"""
        logger.info(f"{request.method} {request.path} from {request.remote_addr}")
        
        if request.endpoint and request.endpoint.startswith('api.'):
            if not request.is_json and request.method == 'POST' and 'multipart/form-data' not in request.content_type:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    @app.after_request
    def after_request(response):
        """Add security headers"""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response
    
    @app.route('/health')
    def health_check():
        """Health check endpoint for load balancers"""
        try:
            # Test Redis connection
            redis_client.ping()
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'redis': 'connected',
                    'openai': 'available'
                }
            })
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e)
            }), 503
    
    @app.route('/api/transcribe', methods=['POST'])
    @limiter.limit("60 per minute")
    def transcribe_audio():
        """Transcribe audio using OpenAI Whisper"""
        try:
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400
            
            audio_file = request.files['audio']
            session_id = request.form.get('session_id')
            
            if not session_id:
                return jsonify({'error': 'Session ID required'}), 400
            
            if audio_file.filename == '':
                return jsonify({'error': 'No audio file selected'}), 400
            
            # Save audio file temporarily
            filename = secure_filename(f"{uuid.uuid4()}.webm")
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(temp_path)
            
            try:
                # Transcribe with Whisper
                with open(temp_path, 'rb') as audio:
                    transcript = openai.Audio.transcribe(
                        "whisper-1", 
                        audio,
                        language="en"
                    )
                
                transcription = transcript.text.strip()
                
                if transcription:
                    logger.info(f"Transcribed for session {session_id}: {transcription[:50]}...")
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
            logger.error(f"Transcription error: {e}")
            return jsonify({
                'success': False,
                'error': 'Transcription failed'
            }), 500
    
    @app.route('/api/analyze', methods=['POST'])
    @limiter.limit("30 per minute")
    def analyze_response():
        """Analyze user response and generate follow-up using GPT-4"""
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
            
            # Analyze with GPT-4
            analysis = analyze_with_gpt4(transcript, session)
            
            # Update session notes using the same logic as original files
            if analysis.get('extracted_info'):
                for category, content in analysis['extracted_info'].items():
                    if content.strip():
                        current = session['notes'][category]
                        if current:
                            session['notes'][category] = f"{current}\n• {content}"
                        else:
                            session['notes'][category] = f"• {content}"
            
            # Update session tracking
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
            logger.error(f"Analysis error: {e}")
            return jsonify({
                'success': False,
                'error': 'Analysis failed'
            }), 500
    
    @app.route('/api/analyze-response', methods=['POST'])
    def analyze_response_endpoint():
        """Analyze user response using GPT-4 API - from response_analyzer.py"""
        try:
            data = request.json
            transcript = data.get('transcript', '')
            context = data.get('context', [])
            current_topic = data.get('currentTopic', 'challenges')
            conversation_stage = data.get('conversationStage', 0)
            
            if not transcript:
                return jsonify({
                    'success': False,
                    'message': 'No transcript provided'
                }), 400
            
            # Format conversation context for the prompt
            context_formatted = ""
            if context:
                context_formatted = "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in context])
            
            # Create prompt for GPT-4
            prompt = f"""
            You are an AI assistant helping international students craft college application essays.
            Analyze the student's response and help extract valuable information for their Statement of Purpose.
            
            Current topic: {current_topic}
            Conversation stage: {conversation_stage}
            
            Recent conversation context:
            {context_formatted}
            
            Student's response:
            "{transcript}"
            
            Analyze this response and:
            1. Extract key information for different SOP categories
            2. Suggest a meaningful follow-up question to get more valuable information
            3. Determine if we should move to a different topic
            
            Return a JSON object with:
            {{
                "analysis": "Brief analysis of the student's response",
                "categoryAssignments": {{
                    "challenges": "Extract relevant content about challenges/turning points",
                    "lessons": "Extract relevant content about lessons learned",
                    "growth": "Extract relevant content about emotional/personal growth",
                    "college": "Extract relevant content about college fit/specific interest"
                }},
                "followUpQuestion": "A meaningful follow-up question based on what needs further exploration",
                "shouldChangeTopic": boolean,
                "suggestedNextTopic": "One of: challenges, lessons, growth, college (if shouldChangeTopic is true)"
            }}
            
            Focus on helping the student develop a narrative about how their experiences have shaped them and why they're a good fit for college.
            """
            
            # Generate response from GPT-4
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )
            
            # Parse and return the analysis
            try:
                analysis = json.loads(response.choices[0].message.content)
                return jsonify({
                    'success': True,
                    **analysis
                })
            except json.JSONDecodeError:
                # If not valid JSON, extract JSON portion from the text
                json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group(0))
                        return jsonify({
                            'success': True,
                            **analysis
                        })
                    except json.JSONDecodeError:
                        return jsonify({
                            'success': False,
                            'message': 'Failed to parse analysis response',
                            'raw_response': response.choices[0].message.content
                        }), 500
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Failed to parse analysis response',
                        'raw_response': response.choices[0].message.content
                    }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e)
            }), 500
    
    @app.route('/api/analyze-response-quality', methods=['POST'])
    def analyze_response_quality():
        """Analyze if the user's response is complete enough for the current question"""
        try:
            data = request.get_json()
            transcript = data.get('transcript', '')
            current_topic = data.get('currentTopic', '')
            
            if not transcript or not current_topic:
                return jsonify({
                    'isComplete': False,
                    'feedback': 'I need more information to continue.'
                })
            
            # Use GPT-4 to analyze response quality
            prompt = f"""
            Analyze if this response about {current_topic} is detailed enough for an SOP:
            "{transcript}"
            
            Rate the response on a scale of 1-10 for:
            1. Specificity (includes concrete details)
            2. Depth (goes beyond surface level)
            3. Relevance (connects to academic/career goals)
            
            If the average score is below 6, what follow-up question would get better information?
            
            Return a JSON object with:
            {{
                "isComplete": boolean,
                "scores": {{
                    "specificity": number,
                    "depth": number,
                    "relevance": number,
                    "average": number
                }},
                "followUpQuestion": string if needed
            }}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return jsonify(result)
            else:
                # Fallback if JSON extraction fails
                return jsonify({
                    'isComplete': True,
                    'followUpQuestion': 'Could you provide more specific details about that?'
                })
                
        except Exception as e:
            logger.error(f"Error analyzing response quality: {e}")
            return jsonify({
                'isComplete': True,
                'error': str(e)
            })
    
    @app.route('/api/generate-sop', methods=['POST'])
    @limiter.limit("10 per hour")
    def generate_sop():
        """Generate Statement of Purpose using GPT-4"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            notes = data.get('notes', {})
            
            if session_id:
                # Get from session
                session = session_manager.get_session(session_id)
                if not session:
                    return jsonify({'error': 'Session not found'}), 404
                notes = session['notes']
            elif not notes:
                return jsonify({'error': 'Either session_id or notes required'}), 400
            
            # Validate that we have enough content
            total_content = sum(len(str(note).strip()) for note in notes.values())
            if total_content < 100:
                return jsonify({
                    'error': 'Insufficient content for SOP generation. Please continue the conversation.'
                }), 400
            
            # Generate SOP with GPT-4
            sop = generate_sop_with_gpt4(notes)
            
            # Store generated SOP in session if session_id provided
            if session_id:
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
            logger.error(f"SOP generation error: {e}")
            return jsonify({
                'success': False,
                'error': 'SOP generation failed'
            }), 500
    
    @app.route('/api/generate-sop-draft', methods=['POST'])
    def generate_sop_draft():
        """Generate SOP draft based on collected notes (alternative endpoint)"""
        try:
            data = request.get_json()
            notes = data.get('notes', {})
            
            # Check if we have enough information
            if not any(str(note).strip() for note in notes.values()):
                return jsonify({
                    'success': False,
                    'message': 'Not enough information collected for all required sections. Please continue the conversation.'
                })
            
            # Generate SOP
            sop = generate_sop_with_gpt4(notes)
            
            return jsonify({
                'success': True,
                'sop': sop
            })
            
        except Exception as e:
            logger.error(f"Error generating SOP draft: {e}")
            return jsonify({
                'success': False,
                'message': f"Error generating SOP: {str(e)}"
            })
    
    return app

def analyze_with_gpt4(transcript: str, session: Dict) -> Dict:
    """Analyze user response using GPT-4 - same functionality as original files"""
    try:
        # Get conversation context
        recent_history = session['conversation_history'][-5:]
        history_text = "\n".join([
            f"{msg['role'].title()}: {msg['content']}" 
            for msg in recent_history
        ])
        
        current_topic = session['current_topic']
        questions_asked = session['questions_asked']
        
        prompt = f"""
You are an AI assistant helping international students craft compelling college application essays.
Analyze the student's response and help extract valuable information for their Statement of Purpose.

Current topic: {current_topic}
Questions asked: {questions_asked}
Recent conversation:
{history_text}

Student's response:
"{transcript}"

Analyze this response and:
1. Extract key information for different SOP categories
2. Suggest a meaningful follow-up question to get more valuable information
3. Determine if we should move to a different topic

Return a JSON object with:
{{
    "analysis": "Brief analysis of the student's response",
    "extracted_info": {{
        "challenges": "Extract relevant content about challenges/turning points",
        "lessons": "Extract relevant content about lessons learned",
        "growth": "Extract relevant content about emotional/personal growth",
        "college": "Extract relevant content about college fit/specific interest"
    }},
    "follow_up_question": "A meaningful follow-up question based on what needs further exploration",
    "should_change_topic": boolean,
    "next_topic": "One of: challenges, lessons, growth, college (if should_change_topic is true)",
    "response_quality": "high/medium/low based on specificity and relevance",
    "completion_ready": boolean
}}

Focus on helping the student develop a narrative about how their experiences have shaped them and why they're a good fit for college.
Guidelines:
- Extract specific, detailed information
- Ask follow-up questions for deeper insights
- Guide toward college connection
- After 8+ exchanges, suggest completion
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        
        # Try to parse JSON from response
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    return result
                except json.JSONDecodeError:
                    return create_fallback_analysis(transcript, session)
            else:
                return create_fallback_analysis(transcript, session)
        
    except Exception as e:
        logger.error(f"GPT-4 analysis error: {e}")
        return create_fallback_analysis(transcript, session)

def create_fallback_analysis(transcript: str, session: Dict) -> Dict:
    """Create fallback analysis when AI fails - same as original"""
    questions_asked = session['questions_asked']
    current_topic = session['current_topic']
    
    # Simple keyword-based analysis
    keywords = {
        'challenges': ['challenge', 'difficult', 'hard', 'struggle', 'obstacle', 'problem'],
        'lessons': ['learn', 'lesson', 'realize', 'understand', 'taught', 'knowledge'],
        'growth': ['feel', 'emotion', 'grow', 'change', 'develop', 'improve'],
        'college': ['college', 'university', 'school', 'education', 'future', 'career']
    }
    
    # Check for keywords in transcript
    extracted_info = {category: '' for category in keywords}
    for category, words in keywords.items():
        if any(word in transcript.lower() for word in words):
            extracted_info[category] = transcript
    
    # Generate basic follow-up questions
    follow_up_questions = {
        'challenges': "Could you tell me more about the specific challenges you faced?",
        'lessons': "What important lessons did you learn from this experience?",
        'growth': "How did this experience change you personally or emotionally?",
        'college': "How does this connect to your interest in the college you're applying to?"
    }
    
    # Determine next topic
    topics = list(keywords.keys())
    current_index = topics.index(current_topic) if current_topic in topics else 0
    next_index = (current_index + 1) % len(topics)
    next_topic = topics[next_index]
    
    return {
        "analysis": "Basic analysis of response",
        "extracted_info": extracted_info,
        "follow_up_question": follow_up_questions.get(current_topic, "Could you tell me more about that?"),
        "should_change_topic": len(transcript) > 100,
        "next_topic": next_topic,
        "response_quality": "medium",
        "completion_ready": questions_asked > 8
    }

def generate_sop_with_gpt4(notes: Dict) -> str:
    """Generate Statement of Purpose using GPT-4 - same functionality as original"""
    try:
        # Handle both formats of notes
        challenges = str(notes.get('challenges', ''))
        lessons = str(notes.get('lessons', ''))
        growth = str(notes.get('growth', ''))
        college = str(notes.get('college', ''))
        
        prompt = f"""
Generate a compelling Statement of Purpose (SOP) for a college application
based on the following information:

CHALLENGES/TURNING POINTS:
{challenges}

LESSONS LEARNED:
{lessons}

EMOTIONAL GROWTH:
{growth}

COLLEGE FIT:
{college}

The SOP should be approximately 650-800 words, well-structured with proper paragraphs, and written in a formal
but engaging tone. Create a narrative arc that shows how the challenges led to growth and lessons learned,
and connect these experiences to the student's interest in their chosen college.

The essay should be reflective, authentic, and emotionally resonant while maintaining a formal academic tone.

Structure the SOP with these sections:
1. Introduction (hook, brief background, purpose statement)
2. Challenges/turning points (what happened and why it matters)
3. Lessons learned (how experiences shaped your perspective)
4. Personal growth (emotional/character development)
5. College fit (why this specific college aligns with your journey)
6. Brief conclusion (restate purpose and look forward)

Make the essay personal and specific, avoiding generic statements. Use a reflective, thoughtful tone
that demonstrates maturity and self-awareness.
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=1500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"SOP generation error: {e}")
        raise

def process_transcript(transcript: str, session: Dict):
    """Process transcript and categorize information - same as original files"""
    # Lowercase for easier matching
    text_lower = transcript.lower()
    
    # Keywords from original files
    academic_keywords = ['degree', 'university', 'college', 'major', 'gpa', 'course', 'study',
                       'academic', 'school', 'education', 'bachelor', 'master', 'grade', 'thesis']
    
    research_keywords = ['research', 'project', 'lab', 'publication', 'experiment', 'study', 'analysis',
                        'investigate', 'finding', 'discovered', 'methodology', 'data', 'collaborate']
    
    career_keywords = ['goal', 'career', 'future', 'aspire', 'aim', 'plan', 'objective', 'interest',
                      'ambition', 'profession', 'industry', 'job', 'work', 'position', 'role']
    
    program_keywords = ['program', 'department', 'faculty', 'professor', 'research group', 'specialization',
                       'interest in', 'align', 'contribute', 'why i chose', 'why this program', 'fit']
    
    # Map to our note categories
    category_mapping = {
        'challenges': career_keywords + ['challenge', 'difficult', 'hard', 'struggle'],
        'lessons': academic_keywords + ['learn', 'lesson', 'realize', 'understand'],
        'growth': ['feel', 'emotion', 'grow', 'change', 'develop', 'improve'],
        'college': program_keywords + research_keywords
    }
    
    # Determine the most likely category based on keyword count
    category_scores = {}
    for category, keywords in category_mapping.items():
        category_scores[category] = sum(1 for word in keywords if word in text_lower)
    
    # Find category with highest score
    highest_category = max(category_scores, key=category_scores.get)
    highest_score = category_scores[highest_category]
    
    # Only categorize if we have a clear signal (at least 2 keyword matches)
    if highest_score >= 2:
        # Store transcript in appropriate category if not already present
        current_content = session['notes'][highest_category]
        if transcript not in current_content:
            if current_content:
                session['notes'][highest_category] = f"{current_content}\n• {transcript}"
            else:
                session['notes'][highest_category] = f"• {transcript}"
    
    # Also update the alternative format for compatibility
    if 'academicBackground' not in session:
        session['academicBackground'] = []
        session['researchExperience'] = []
        session['careerGoals'] = []
        session['programFit'] = []
    
    # Map to alternative categories
    alt_mapping = {
        'challenges': 'careerGoals',
        'lessons': 'academicBackground',
        'growth': 'careerGoals',
        'college': 'programFit'
    }
    
    if highest_score >= 2:
        alt_category = alt_mapping.get(highest_category, 'academicBackground')
        if transcript not in session[alt_category]:
            session[alt_category].append(transcript)

def create_socketio(app):
    """Create SocketIO instance for real-time communication"""
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
        logger.info(f'Client connected: {socket_request.sid}')
        
        # Create session
        session_data = session_manager.create_session(socket_request.sid)
        
        emit('connection_status', {'status': 'connected'})
        emit('connected', {
            'session_id': socket_request.sid,
            'status': 'ready'
        })
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f'Client disconnected: {socket_request.sid}')
    
    @socketio.on('voice_transcript')
    def handle_voice_transcript(data):
        """Process real-time voice transcript and categorize information"""
        transcript = data.get('transcript', '')
        
        if not transcript or len(transcript) < 5:
            return
        
        # Get session
        session = session_manager.get_session(socket_request.sid)
        if not session:
            session = session_manager.create_session(socket_request.sid)
        
        # Process the transcript using the same logic as original
        process_transcript(transcript, session)
        
        # Update session
        session_manager.update_session(socket_request.sid, session)
        
        # Send updated data back to client
        emit('notes_update', session['notes'])
    
    @socketio.on('analyze_response_quality')
    def analyze_response_quality(data):
        """Analyze if the user's response is complete enough for the current question"""
        transcript = data.get('transcript', '')
        current_topic = data.get('currentTopic', '')
        
        if not transcript or not current_topic:
            emit('quality_feedback', {
                'isComplete': False,
                'feedback': 'I need more information to continue.'
            })
            return
        
        # Use GPT-4 to analyze response quality
        prompt = f"""
        Analyze if this response about {current_topic} is detailed enough for an SOP:
        "{transcript}"
        
        Rate the response on a scale of 1-10 for:
        1. Specificity (includes concrete details)
        2. Depth (goes beyond surface level)
        3. Relevance (connects to academic/career goals)
        
        If the average score is below 6, what follow-up question would get better information?
        
        Return a JSON object with:
        {{
            "isComplete": boolean,
            "scores": {{
                "specificity": number,
                "depth": number,
                "relevance": number,
                "average": number
            }},
            "followUpQuestion": string if needed
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                emit('quality_feedback', result)
            else:
                # Fallback if JSON extraction fails
                emit('quality_feedback', {
                    'isComplete': True,
                    'followUpQuestion': 'Could you provide more specific details about that?'
                })
        except Exception as e:
            logger.error(f"Error analyzing response quality: {e}")
            emit('quality_feedback', {
                'isComplete': True,
                'error': str(e)
            })
    
    @socketio.on('generate_sop_draft')
    def generate_sop_draft():
        """Generate SOP draft based on collected notes"""
        # Get session
        session = session_manager.get_session(socket_request.sid)
        if not session:
            emit('generation_error', {
                'message': 'Session not found. Please refresh and try again.'
            })
            return
        
        # Check if we have enough information
        notes = session['notes']
        if not any(str(note).strip() for note in notes.values()):
            emit('generation_error', {
                'message': 'Not enough information collected for all required sections. Please continue the conversation.'
            })
            return
        
        try:
            # Generate content with GPT-4
            sop = generate_sop_with_gpt4(notes)
            
            # Store in session
            session['generated_sop'] = sop
            session['sop_generated_at'] = datetime.now().isoformat()
            session_manager.update_session(socket_request.sid, session)
            
            emit('sop_draft', {
                'success': True,
                'sop': sop
            })
        except Exception as e:
            logger.error(f"Error generating SOP: {e}")
            emit('generation_error', {
                'message': f"Error generating SOP: {str(e)}"
            })
    
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
        )