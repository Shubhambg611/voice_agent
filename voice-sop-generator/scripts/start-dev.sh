#!/bin/bash

echo "🚀 Starting Voice SOP Generator in development mode..."

# Check if Redis is running, start if needed
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Starting Redis..."
    redis-server --daemonize yes
    sleep 2
fi

# Kill any existing processes
pkill -f "python app.py" 2>/dev/null || true
pkill -f "npm start" 2>/dev/null || true

# Start backend
echo "🐍 Starting backend..."
cd backend
source venv/bin/activate
python app.py &
BACKEND_PID=$!
cd ..

# Start frontend
echo "⚛️ Starting frontend..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Development servers started!"
echo "Frontend: http://localhost:3000"
echo "Backend: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for interrupt signal
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait