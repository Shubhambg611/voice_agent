import os

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
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
