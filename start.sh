#!/bin/bash

# Shin Gateway - Start Backend and Frontend
# Usage: ./start.sh [--backend-only] [--frontend-only]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default ports
BACKEND_PORT=8080
FRONTEND_PORT=3000

# Parse arguments
BACKEND_ONLY=false
FRONTEND_ONLY=false

for arg in "$@"; do
    case $arg in
        --backend-only)
            BACKEND_ONLY=true
            shift
            ;;
        --frontend-only)
            FRONTEND_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./start.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backend-only   Start only the backend server"
            echo "  --frontend-only  Start only the frontend server"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
    esac
done

# Function to kill process on port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Stopping process on port $port (PID: $pid)${NC}"
        kill -9 $pid 2>/dev/null || true
        sleep 1
    fi
}

# Function to start backend
start_backend() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Starting Backend (FastAPI) on port $BACKEND_PORT${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    kill_port $BACKEND_PORT

    cd /home

    # Check if uvicorn is available
    if ! command -v uvicorn &> /dev/null; then
        echo -e "${YELLOW}Installing uvicorn...${NC}"
        pip install uvicorn fastapi pydantic pydantic-settings pyyaml httpx
    fi

    # Start backend in background
    nohup uvicorn main:app --host 0.0.0.0 --port $BACKEND_PORT --reload > /tmp/backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > /tmp/backend.pid

    echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
    echo -e "${GREEN}  Logs: /tmp/backend.log${NC}"
    echo -e "${GREEN}  URL: http://localhost:$BACKEND_PORT${NC}"
}

# Function to start frontend
start_frontend() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Starting Frontend (Next.js) on port $FRONTEND_PORT${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    kill_port $FRONTEND_PORT

    cd /home/admin-ui

    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}Installing npm dependencies...${NC}"
        npm install
    fi

    # Start frontend in background
    nohup npm run dev > /tmp/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > /tmp/frontend.pid

    echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"
    echo -e "${GREEN}  Logs: /tmp/frontend.log${NC}"
    echo -e "${GREEN}  URL: http://localhost:$FRONTEND_PORT${NC}"
}

# Main execution
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Shin Gateway - Server Startup                   ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$FRONTEND_ONLY" = true ]; then
    start_frontend
elif [ "$BACKEND_ONLY" = true ]; then
    start_backend
else
    start_backend
    echo ""
    start_frontend
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}All services started successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Commands:"
echo "  View backend logs:  tail -f /tmp/backend.log"
echo "  View frontend logs: tail -f /tmp/frontend.log"
echo "  Stop all:           ./stop.sh"
echo ""
