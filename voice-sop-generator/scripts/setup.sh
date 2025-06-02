#!/bin/bash

# Voice SOP Generator - Development Setup Script

set -e

echo "🚀 Setting up Voice SOP Generator for development..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    echo "⚠️ Redis is not installed. Please install Redis for session management."
    echo "Ubuntu/Debian: sudo apt install redis-server"
    echo "macOS: brew install redis"
    echo "Windows: Download from https://redis.io/download"
fi

# Setup backend
echo "🐍 Setting up backend..."
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created backend/.env - please configure your API keys"
fi

cd ..

# Setup frontend
echo "⚛️ Setting up frontend..."
cd frontend

# Install dependencies
npm install

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created frontend/.env - please configure your settings"
fi

cd ..

echo "✅ Setup complete!"
echo ""
echo "🔧 Configuration needed:"
echo "1. Backend: Configure backend/.env with your API keys:"
echo "   - GEMINI_API_KEY=your_gemini_key_here"
echo "   - OPENAI_API_KEY=your_openai_key_here (for Whisper)"
echo "   - AI_PROVIDER=gemini"
echo "2. Frontend: Configure frontend/.env with your settings"
echo ""
echo "🚀 To start development:"
echo "1. Start Redis: redis-server"
echo "2. Start backend: cd backend && source venv/bin/activate && python app.py"
echo "3. Start frontend: cd frontend && npm start"
echo ""
echo "🎯 Access your app at: http://localhost:3000"
echo "Happy coding! 🎉"
