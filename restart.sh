#!/bin/bash

# Shin Gateway - Restart Backend and Frontend
# Usage: ./restart.sh [--backend-only] [--frontend-only]

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Shin Gateway - Server Restart                   ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Pass all arguments to stop and start scripts
./stop.sh "$@"
sleep 2
./start.sh "$@"
