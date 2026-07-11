#!/bin/bash
# ============================================================================
# Deploy Kolrose RAG to Render
# ============================================================================

set -e

echo "🚀 Deploying Kolrose Policy Assistant to Render..."

# Check for Render CLI
if ! command -v render &> /dev/null; then
    echo "Installing Render CLI..."
    curl -fsSL https://raw.githubusercontent.com/render-oss/cli/main/install.sh | sh
fi

# Set environment variables
read -p "OpenRouter API Key: " OPENROUTER_API_KEY
render env set OPENROUTER_API_KEY="$OPENROUTER_API_KEY" --service kolrose-policy-rag

# Deploy using blueprint
render blueprint apply

echo "✅ Deployment triggered!"
echo "   Check status: https://dashboard.render.com"
echo "   App will be available at: https://kolrose-policy-rag.onrender.com"