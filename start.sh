#!/bin/bash
set -e

# Use PORT from Railway environment, fallback to 8501 for local testing
export PORT=${PORT:-8501}

# Create writable directory for ChromaDB (important for Railway's ephemeral storage)
export CHROMA_DB_PATH=/tmp/chroma_db
mkdir -p $CHROMA_DB_PATH

# Display startup information
echo "🚀 Starting Kolrose Policy Assistant"
echo "   Port: $PORT"
echo "   ChromaDB Path: $CHROMA_DB_PATH"
echo ""

# Run Streamlit with Railway-compatible settings
streamlit run BACKEND/app/app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false \
    --logger.level=info