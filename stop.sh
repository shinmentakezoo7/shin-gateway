#!/bin/bash

# Shin Gateway - Stop Backend and Frontend
# Usage: ./stop.sh [--backend-only] [--frontend-only]

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
            echo "Usage: ./stop.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backend-only   Stop only the backend server"
            echo "  --frontend-only  Stop only the frontend server"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
    esac
done

# Function to kill process on port
kill_port() {
    local port=$1
    local name=$2
    local pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Stopping $name on port $port (PID: $pid)${NC}"
        kill -9 $pid 2>/dev/null || true
        echo -e "${GREEN}✓ $name stopped${NC}"
    else
        echo -e "${BLUE}$name is not running on port $port${NC}"
    fi
}

# Function to stop using PID file
stop_by_pid() {
    local pidfile=$1
    local name=$2
    if [ -f "$pidfile" ]; then
        local pid=$(cat $pidfile)
        if kill -0 $pid 2>/dev/null; then
            echo -e "${YELLOW}Stopping $name (PID: $pid)${NC}"
            kill -9 $pid 2>/dev/null || true
            echo -e "${GREEN}✓ $name stopped${NC}"
        fi
        rm -f $pidfile
    fi
}

# Main execution
echo ""
echo -e "${RED}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║           Shin Gateway - Server Shutdown                  ║${NC}"
echo -e "${RED}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$FRONTEND_ONLY" = true ]; then
    stop_by_pid /tmp/frontend.pid "Frontend"
    kill_port $FRONTEND_PORT "Frontend"
elif [ "$BACKEND_ONLY" = true ]; then
    stop_by_pid /tmp/backend.pid "Backend"
    kill_port $BACKEND_PORT "Backend"
else
    stop_by_pid /tmp/backend.pid "Backend"
    stop_by_pid /tmp/frontend.pid "Frontend"
    kill_port $BACKEND_PORT "Backend"
    kill_port $FRONTEND_PORT "Frontend"
fi

echo ""
echo -e "${GREEN}All services stopped.${NC}"
echo ""
