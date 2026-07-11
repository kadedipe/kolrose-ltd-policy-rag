markdown
# 🏢 Kolrose Limited - Company Policy RAG Assistant

<div align="center">

**An AI-powered Retrieval-Augmented Generation (RAG) system for instant, accurate answers to company policy questions.**

📍 **Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Deployment](https://img.shields.io/badge/Deploy-Render%20Free%20Tier-purple)](https://render.com)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Running the Application](#running-the-application)
- [Testing & Evaluation](#testing--evaluation)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Policy Corpus](#policy-corpus)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)

---

## 📖 Overview

This system allows Kolrose Limited employees to ask natural language questions about company policies and receive accurate, cited answers drawn from official policy documents. The system covers **12 policy documents** spanning HR, IT, Finance, and Administration.

**Example Questions:**
- *"What is the annual leave entitlement after 5 years of service?"*
- *"Can I work remotely while on probation?"*
- *"What is the procurement process for purchases above ₦2,000,000?"*
- *"How do I report a security incident?"*

The system is designed to run on **completely free infrastructure**:
- **LLM**: OpenRouter free tier (Llama 3, GPT-3.5, etc.)
- **Embeddings**: Local `sentence-transformers` (no API costs)
- **Vector Database**: ChromaDB (local persistence)
- **Hosting**: Render or Railway free tier

---

## ✨ Features

- 🔍 **Accurate Policy Retrieval**: Multi-strategy retrieval (dense + MMR)
- 📝 **Verified Citations**: Every claim backed by specific policy document and section references
- 🇳🇬 **Nigerian Context Aware**: Handles Naira amounts, Nigerian agencies (EFCC, NITDA), local locations
- ⚡ **Fast Responses**: Median latency < 2 seconds for most queries
- 📊 **Comprehensive Evaluation**: RAGAS metrics + custom groundedness and citation accuracy
- 🛡️ **Safety Guardrails**: Compliance warnings for sensitive topics (passwords, corruption)
- 🚀 **Free Deployment**: Designed for Render/Railway free tier with CI/CD

---

## 🔧 Prerequisites

| Requirement | Version | Check Command |
|------------|---------|---------------|
| Python | 3.10 or higher | `python3 --version` |
| pip | Latest | `pip --version` |
| Git | Any | `git --version` |
| OpenRouter API Key | Free tier | [Get Free Key](https://openrouter.ai/keys) |

**Optional:**
- Docker (for containerized deployment)
- 2GB free disk space (for embeddings model)

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/kolrose/policy-rag.git
cd kolrose-policy-rag

# 2. Run the setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Edit your .env file with OpenRouter API key
nano .env
# Set: OPENROUTER_API_KEY=sk-or-v1-your-free-key-here

# 4. Verify installation
python scripts/verify_setup.py

# 5. Ingest policy documents
python scripts/ingest_policies.py

# 6. Start the server
uvicorn app.main:app --reload

# 7. Open in browser
open http://localhost:8000/docs
💡 Tip: Get your free OpenRouter API key at openrouter.ai/keys. No credit card required for free tier.

🔨 Detailed Setup
Step 1: Environment Setup
Option A: Using venv (Recommended)

bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

pip install --upgrade pip
pip install -r requirements.txt
Option B: Using Conda

bash
conda env create -f environment.yml
conda activate kolrose-rag
Option C: Using Docker

bash
docker build -t kolrose-rag .
docker run -p 8000:8000 --env-file .env kolrose-rag
Step 2: Configure Environment Variables
bash
# Copy the template
cp .env.example .env

# Edit with your values
nano .env
Minimum required settings:

bash
OPENROUTER_API_KEY=sk-or-v1-your-free-key-here
LLM_MODEL=meta-llama/llama-3-8b-instruct:free
Available free models on OpenRouter:

Model ID	Provider	Context	Best For
meta-llama/llama-3-8b-instruct:free	Meta	8K	General purpose
google/gemma-7b-it:free	Google	8K	Instruction following
openai/gpt-3.5-turbo:free	OpenAI	16K	Best quality (limited)
mistralai/mistral-7b-instruct:free	Mistral	32K	Long documents
Step 3: Add Policy Documents
Place the 12 Kolrose policy markdown files in the ./policies/ directory:

bash
policies/
├── employee_handbook.md              # KOL-HR-001
├── leave_time_off_policy.md          # KOL-HR-002
├── code_of_conduct.md                # KOL-HR-003
├── remote_work_policy.md             # KOL-HR-005
├── it_security_policy.md             # KOL-IT-001
├── expenses_policy.md                # KOL-FIN-001
├── performance_management_policy.md   # KOL-HR-006
├── training_development_policy.md    # KOL-HR-007
├── travel_policy.md                  # KOL-ADMIN-001
├── procurement_policy.md             # KOL-FIN-002
├── health_safety_policy.md           # KOL-ADMIN-002
└── grievance_policy.md               # KOL-HR-008
Step 4: Run Ingestion Pipeline
bash
# Ingest all policies into the vector database
python scripts/ingest_policies.py

# Expected output:
# ✅ Loaded 12 policy documents
# 🧩 Created 847 enriched chunks
# 💾 Vector store persisted to ./chroma_db
# 🔍 Validation: 5/5 checks passed
Step 5: Verify Installation
bash
python scripts/verify_setup.py
Expected output:

text
🔍 Checking Python version...      ✅ Python 3.11.4
🔍 Checking virtual environment...  ✅ Virtual environment active
🔍 Checking .env configuration...   ✅ .env file exists
🔍 Checking dependencies...         ✅ FastAPI, LangChain, ChromaDB, etc.
🔍 Checking directories...          ✅ All directories present
🔍 Checking policy documents...     ✅ 12 policy file(s) found
🔍 Testing embedding model...       ✅ Model loaded (384 dimensions)

✅ ENVIRONMENT READY!
🚀 Running the Application
Development Mode (with auto-reload)
bash
# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or with hot reload on all files
uvicorn app.main:app --reload --reload-dir app --reload-dir policies
Production Mode
bash
# Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2

# Using gunicorn (recommended for production)
gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT
Access the Application
Endpoint	URL	Description
API Docs (Swagger)	http://localhost:8000/docs	Interactive API documentation
API Docs (ReDoc)	http://localhost:8000/redoc	Alternative API documentation
Health Check	http://localhost:8000/api/v1/health	System health status
Query Endpoint	http://localhost:8000/api/v1/query	Main RAG query endpoint
Evaluation	http://localhost:8000/api/v1/evaluate	Run evaluation suite
Quick Test
bash
# Test with curl
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the annual leave entitlement at Kolrose Limited?",
    "user_id": "test-user"
  }'

# Expected response (truncated):
# {
#   "answer": "According to the Leave and Time-Off Policy [KOL-HR-002], employees with...",
#   "citations": [...],
#   "confidence": 0.94,
#   "processing_time_ms": 1234
# }
Using the Python Client
python
import requests

# Query the API
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "question": "Can I work remotely on Fridays?",
        "user_id": "employee-123"
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Sources: {len(result['sources'])} documents cited")
🧪 Testing & Evaluation
Running Tests
bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=html

# Run specific test categories
pytest tests/ -v -m "unit"
pytest tests/ -v -m "integration"
pytest tests/ -v -m "evaluation"
Running Evaluation
bash
# Run the comprehensive evaluation suite
python scripts/run_evaluation.py

# Expected evaluation metrics:
# ┌─────────────────────────────────────────┐
# │         EXCELLENCE METRICS REPORT        │
# ├─────────────────────────────────────────┤
# │ ✅ Groundedness Score........... 0.94    │
# │ ✅ Citation Accuracy............ 0.97    │
# │ ✅ Response Latency (P95)....... 1.8s    │
# │ ✅ Hallucination Rate........... 0.01    │
# │ ✅ Retrieval Precision@5........ 0.89    │
# └─────────────────────────────────────────┘
Test Coverage
Test Suite	Files	Description
test_ingestion.py	Ingestion pipeline	Document loading, chunking, metadata
test_rag_pipeline.py	RAG system	Retrieval, generation, citations
test_api.py	FastAPI endpoints	HTTP requests, responses, errors
test_evaluation.py	Metrics	Groundedness, citation accuracy, latency
🌐 Deployment
Deploy to Render (Free Tier)
Fork/Clone this repository to your GitHub account

Create Render Account: render.com (free)

Create New Web Service:

Connect your GitHub repository

Set build command: pip install -r requirements.txt

Set start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT

Select "Free" instance type

Set Environment Variables in Render dashboard:

text
OPENROUTER_API_KEY=sk-or-v1-your-key
APP_ENV=production
Deploy! Your app will be available at https://your-app.onrender.com

⚠️ Free Tier Note: Render free instances spin down after 15 minutes of inactivity. First request after inactivity may take 30-60 seconds. Use UptimeRobot (free) to ping every 5 minutes.

Deploy to Railway (Free Tier)
Create Railway Account: railway.app

Deploy from GitHub:

bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
Set Environment Variables:

bash
railway variables set OPENROUTER_API_KEY=sk-or-v1-your-key
railway variables set APP_ENV=production
GitHub Actions CI/CD
The repository includes a GitHub Actions workflow (.github/workflows/deploy.yml) that:

Runs tests on every push and PR

Checks code quality (black, flake8, isort)

Runs evaluation suite

Deploys to Render on merge to main

To set up:

Add your Render API key as GitHub Secret: RENDER_API_KEY

Add your Render Service ID as GitHub Secret: RENDER_SERVICE_ID

Push to main branch - deployment triggers automatically

📁 Project Structure
text
kolrose-policy-rag/
│
├── app/                          # Application code
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── main.py                   # FastAPI application
│   ├── ingestion_pipeline.py     # Document ingestion
│   ├── rag_pipeline.py           # RAG query processing
│   └── evaluator.py              # Evaluation metrics
│
├── policies/                     # Policy documents (12 .md files)
│   ├── README.md
│   ├── employee_handbook.md      # KOL-HR-001
│   ├── leave_time_off_policy.md  # KOL-HR-002
│   └── ... (10 more files)
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_rag_pipeline.py
│   ├── test_api.py
│   └── test_evaluation.py
│
├── scripts/                      # Utility scripts
│   ├── setup.sh                  # Environment setup
│   ├── verify_setup.py           # Installation verification
│   ├── ingest_policies.py        # Run ingestion
│   └── run_evaluation.py         # Run evaluation
│
├── chroma_db/                    # Vector database (auto-created)
├── backups/                      # Data backups
├── logs/                         # Application logs
│
├── requirements/                 # Dependency files
│   ├── base.txt                  # Core dependencies
│   ├── dev.txt                   # Development tools
│   └── prod.txt                  # Production additions
│
├── .github/workflows/            # CI/CD
│   └── deploy.yml                # GitHub Actions workflow
│
├── .env.example                  # Environment template
├── .env                          # Your configuration (gitignored)
├── .gitignore
├── requirements.txt              # Main dependencies
├── environment.yml               # Conda alternative
├── pyproject.toml                # Modern Python config
├── Dockerfile                    # Container config
├── docker-compose.yml
└── README.md                     # This file
🏗️ Architecture
text
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT (Browser/API)                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    FASTAPI APPLICATION                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   RAG PIPELINE                          │ │
│  │  ┌──────────────┐  ┌──────────┐  ┌──────────────────┐  │ │
│  │  │   Query       │  │ Multi-   │  │  Context         │  │ │
│  │  │   Processing  │─▶│ Strategy │─▶│  Assembly        │  │ │
│  │  │               │  │ Retrieval│  │                  │  │ │
│  │  └──────────────┘  └──────────┘  └──────────────────┘  │ │
│  │                                                          │ │
│  │  ┌──────────────────┐  ┌──────────┐  ┌──────────────┐  │ │
│  │  │   Citation        │  │ Response │  │   LLM         │  │ │
│  │  │   Verification    │◀─│ Generation│◀─│ (OpenRouter)  │  │ │
│  │  │                  │  │          │  │              │  │ │
│  │  └──────────────────┘  └──────────┘  └──────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────┐          ┌─────────────────────────────┐
│   LOCAL EMBEDDINGS   │          │      OPENROUTER API          │
│  (sentence-trans-    │          │     (Free Tier: 50req/day)   │
│   formers, FREE)     │          │                              │
└─────────────────────┘          └─────────────────────────────┘

┌─────────────────────┐
│    CHROMADB          │
│  (Local Vector DB)   │
└─────────────────────┘
Component Details
Component	Technology	Cost	Purpose
API Layer	FastAPI + Uvicorn	Free	HTTP interface, request validation
LLM	OpenRouter	Free*	Response generation
Embeddings	all-MiniLM-L6-v2	Free	Local document embedding
Vector Store	ChromaDB	Free	Document retrieval
Chunking	LangChain splitters	Free	Document segmentation
Evaluation	RAGAS + Custom	Free	Quality metrics
Hosting	Render/Railway	Free	Application hosting
**Free tier: 50 requests/day, or unlimited with purchased credits*

📚 Policy Corpus
The system indexes 12 Kolrose Limited policy documents:

#	Document ID	Policy	Category	Pages
1	KOL-HR-001	Employee Handbook	HR	8
2	KOL-HR-002	Leave and Time-Off	HR	10
3	KOL-HR-003	Code of Conduct	HR/Legal	12
4	KOL-HR-005	Remote Work	HR	10
5	KOL-IT-001	IT Security & AUP	IT	14
6	KOL-FIN-001	Expenses & Reimbursement	Finance	12
7	KOL-HR-006	Performance Management	HR	12
8	KOL-HR-007	Training & Development	HR	10
9	KOL-ADMIN-001	Business Travel	Admin	14
10	KOL-FIN-002	Procurement	Finance	14
11	KOL-ADMIN-002	Health & Safety	Admin	12
12	KOL-HR-008	Grievance & Dispute	HR	12
📝 Note: These are synthetic policy documents created for demonstration. They do not represent actual company policies.

📡 API Documentation
POST /api/v1/query
Query the RAG system with a natural language question.

Request:

json
{
  "question": "What is the remote work policy at Kolrose Limited?",
  "user_id": "employee-456",
  "include_sources": true,
  "include_citations": true
}
Response:

json
{
  "answer": "According to the Remote Work and Flexible Working Policy [KOL-HR-005], employees who have completed probation can work remotely up to 2 days per week...",
  "citations": [
    {
      "full_text": "[KOL-HR-005, Section 3.1]",
      "policy": "Remote Work and Flexible Working Policy",
      "section": "3.1",
      "is_valid": true
    }
  ],
  "sources": [
    {
      "document_id": "KOL-HR-005",
      "citation_text": "Remote Work Policy [KOL-HR-005] > Section 3 > Hybrid Model",
      "relevance_score": 0.93
    }
  ],
  "confidence": 0.94,
  "processing_time_ms": 1250,
  "query_id": "q-abc123",
  "timestamp": "2024-03-15T10:30:00+01:00"
}
GET /api/v1/health
Check system health.

Response:

json
{
  "status": "healthy",
  "timestamp": "2024-03-15T10:30:00+01:00",
  "version": "1.0.0",
  "components": {
    "vector_store": "connected",
    "embedding_model": "loaded",
    "llm": "configured"
  }
}
GET /api/v1/evaluate
Run the evaluation suite.

Response:

json
{
  "excellence_score": 0.92,
  "metrics": {
    "groundedness_score": 0.94,
    "citation_accuracy": 0.97,
    "response_latency_p95": 1.8,
    "hallucination_rate": 0.01
  }
}
🔧 Troubleshooting
Common Issues
Issue	Solution
ModuleNotFoundError	Activate virtual environment: source venv/bin/activate
OpenRouter 402 error	Add credits or use :free model suffix
ChromaDB connection error	Delete chroma_db/ and re-run ingestion
Slow first response	Normal - embedding model loads on first use
Render spin-down	Use UptimeRobot to ping every 5 minutes
CUDA out of memory	Set device=cpu in config if no GPU
Logs
bash
# View application logs
tail -f logs/app.log

# View with filtering
grep "ERROR" logs/app.log
Reset Everything
bash
# Clean reset
rm -rf chroma_db/ backups/ logs/
python scripts/ingest_policies.py
📊 Success Metrics
Target metrics for the Kolrose RAG system:

Metric	Target	Category
Groundedness	≥ 92%	Information Quality
Citation Accuracy	≥ 95%	Accuracy
Response Latency (P95)	< 2.0s	System Performance
Hallucination Rate	< 2%	Safety
Retrieval Precision@5	≥ 85%	Retrieval Quality
🤝 Contributing
Create a feature branch: git checkout -b feature/your-feature

Make changes and test: pytest tests/ -v

Format code: black app/ && isort app/

Submit PR to develop branch

📄 License
MIT License - See LICENSE file

📞 Contact
Kolrose Limited
Suite 10, Bataiya Plaza, Area 2 Garki
Opposite FCDA, Abuja, FCT, Nigeria

📧 HR: hr@kolroselimited.com.ng

📧 IT Support: it@kolroselimited.com.ng

🌐 Website: https://kolroselimited.com.ng

<div align="center">
Built with ❤️ for Kolrose Limited Employees

</div> ```
This README provides comprehensive setup and run instructions covering all the requirements you specified. It's structured to be immediately useful while also serving as excellent documentation for your project evaluation. The instructions cover both local development and free-tier deployment options.