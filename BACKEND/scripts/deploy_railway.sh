#!/bin/bash
# ============================================================================
# Deploy Kolrose RAG to Railway
# ============================================================================

set -e

echo "🚀 Deploying Kolrose Policy Assistant to Railway..."

# Check for Railway CLI
if ! command -v railway &> /dev/null; then
    echo "Installing Railway CLI..."
    npm install -g @railway/cli
fi

# Login
railway login

# Link project
railway link

# Set environment variables
read -p "OpenRouter API Key: " OPENROUTER_API_KEY
railway variables set OPENROUTER_API_KEY="$OPENROUTER_API_KEY"

# Deploy
railway up

echo "✅ Deployment triggered!"
echo "   Check status: https://railway.app/dashboard"