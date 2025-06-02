import os

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
    LOG_LEVEL = 'DEBUG'
