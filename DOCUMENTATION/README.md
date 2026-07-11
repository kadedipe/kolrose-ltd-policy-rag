markdown
# 🏢 Kolrose Limited - AI Policy Assistant

<div align="center">

**A Retrieval-Augmented Generation (RAG) system for instant, accurate answers to company policy questions.**

📍 **Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Cost](https://img.shields.io/badge/Cost-FREE-success)]()

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Policy Corpus](#policy-corpus)
- [Guardrails](#guardrails)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 📖 Overview

The **Kolrose Limited AI Policy Assistant** is a Retrieval-Augmented Generation (RAG) system that enables employees to ask natural language questions about company policies and receive accurate, cited answers drawn directly from official policy documents.

**Example Questions:**
- *"What is the annual leave entitlement after 5 years of service?"*
- *"Can I work remotely while on probation?"*
- *"What is the procurement process for purchases above ₦2,000,000?"*
- *"How do I report a security incident?"*

The system runs on **completely free infrastructure**:
- **LLM**: OpenRouter free tier (Google Gemini Flash, Llama 3, etc.)
- **Embeddings**: Local `sentence-transformers` model (no API costs)
- **Vector Database**: ChromaDB (local persistence)
- **Hosting**: Streamlit Cloud (free tier)

---

## ✨ Features

### Core Capabilities
- 🔍 **Accurate Policy Retrieval** - Multi-strategy retrieval (MMR + cross-encoder re-ranking)
- 📝 **Verified Citations** - Every claim backed by `[Document ID, Section X.Y]` references
- 🇳🇬 **Nigerian Context Aware** - Handles Naira (₦) amounts, Nigerian agencies (EFCC, NITDA, NDPR)
- ⚡ **Fast Responses** - Median latency < 2 seconds for most queries
- 💰 **Zero Cost** - All components run on free tiers

### Safety & Quality
- 🛡️ **Corpus Boundaries** - Refuses to answer off-topic or out-of-scope questions
- 📏 **Output Limiting** - Prevents excessively long responses
- 📚 **Mandatory Citations** - Requires `[KOL-XX-NNN]` references in every answer
- ⚠️ **Sensitive Topic Detection** - Handles password sharing, corruption, harassment queries appropriately

### Developer Experience
- 🐍 **Clean Python Package** - Modular `app/` structure with type hints
- 📊 **Evaluation Suite** - RAGAS metrics + custom groundedness scoring
- 🚀 **One-Click Deploy** - Streamlit Cloud deployment ready
- 📝 **Comprehensive Logging** - Structured logs for debugging

---

## System Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Streamlit Web UI<br/>Port 8501]
        B[FastAPI Endpoints<br/>Port 8000]
    end
    
    subgraph "Backend Layer"
        C[Guardrails<br/>Topic + Length + Citations]
        D[RAG Pipeline<br/>Retrieval + Generation]
        E[Ingestion Pipeline<br/>Load → Chunk → Embed → Index]
    end
    
    subgraph "Data Layer"
        F[Policy Documents<br/>12 .md files]
        G[ChromaDB<br/>250 vectors]
        H[Embeddings<br/>all-MiniLM-L6-v2]
    end
    
    subgraph "External Services"
        I[OpenRouter API<br/>Gemini Flash - FREE]
    end
    
    A --> C
    B --> C
    C --> D
    D --> G
    D --> H
    D --> I
    E --> F
    E --> H
    E --> G
    
    style A fill:#1a5276,color:#fff
    style B fill:#1a5276,color:#fff
    style I fill:#2e86c1,color:#fff

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/kolrose/policy-rag.git
cd kolrose-policy-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your OpenRouter API key (get free key at https://openrouter.ai/keys)

# 5. Add policy documents to ./policies/ directory

# 6. Run ingestion (first time only)
python -m app.ingestion

# 7. Launch the app
streamlit run app/main.py

# 8. Open http://localhost:8501
🔧 Installation
Prerequisites
Requirement	Version	Check Command
Python	3.10+	python --version
pip	Latest	pip --version
Git	Any	git --version
OpenRouter API Key	Free	Get Key
Disk Space	~2 GB	For embedding model and vector store
Step-by-Step
1. Clone the Repository
bash
git clone https://github.com/kolrose/policy-rag.git
cd kolrose-policy-rag
2. Set Up Virtual Environment
bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Using conda
conda create -n kolrose-rag python=3.11
conda activate kolrose-rag
3. Install Dependencies
bash
pip install -r requirements.txt
4. Configure Environment
bash
cp .env.example .env
nano .env  # Edit with your API key
Required settings in .env:

bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
LLM_MODEL=google/gemini-2.0-flash-001
5. Add Policy Documents
Place your 12 Kolrose policy markdown files in the ./policies/ directory:

text
policies/
├── Code-of-Conduct.md
├── Employee-Handbook-Overview.md
├── Expenses-and-Reimbursement-Policy.md
├── Grievance-and-Dispute-Resolution-Policy.md
├── Health-and-Safety-Policy.md
├── IT-Security-and-Acceptable-Use-Policy.md
├── Leave-and-Time-Off-Policy.md
├── Performance-Management-Policy.md
├── Procurement-Policy.md
├── Remote-Work-Policy.md
├── Training-and-Development-Policy.md
└── Travel-Policy.md
6. Run Document Ingestion
bash
# Ingest policies into vector database (first time only)
python -m app.ingestion
7. Verify Installation
bash
# Check everything is working
python -c "from app import KolroseRAG; print('✅ System ready!')"
🚀 Usage
Web Application
bash
# Launch Streamlit app
streamlit run app/main.py

# Or use the entry point
streamlit run streamlit_app.py
Open http://localhost:8501 in your browser.

Python API
python
from app import KolroseRAG, load_vectorstore, load_embeddings
from app.config import COMPANY_INFO

# Load existing vector store
vectorstore = load_vectorstore()

# Initialize RAG system
rag = KolroseRAG(vectorstore)

# Single query
result = rag.query("What is the annual leave policy?")
print(result.answer)
print(result.citations)
print(result.metrics)

# Batch queries
questions = [
    "How do I request remote work?",
    "What are the password requirements?",
    "How are travel expenses reimbursed?",
]
results = rag.batch_query(questions)
for r in results:
    print(f"Q: {r.question}")
    print(f"A: {r.answer[:100]}...\n")
Command Line
bash
# Run ingestion
python -m app.ingestion

# Check vector store status
python -c "from app.ingestion import get_ingestion_stats; print(get_ingestion_stats())"
📁 Project Structure
text
kolrose-policy-rag/
│
├── app/                              # Application package
│   ├── __init__.py                   # Package initialization, exports
│   ├── config.py                     # Configuration from env vars
│   ├── main.py                       # Streamlit web application
│   ├── rag_system.py                 # Core RAG classes & functions
│   ├── guardrails.py                 # Safety & quality controls
│   └── ingestion.py                  # Document ingestion pipeline
│
├── policies/                         # Policy markdown files (12 docs)
│   ├── Code-of-Conduct.md
│   ├── Employee-Handbook-Overview.md
│   └── ... (10 more)
│
├── chroma_db/                        # Vector database (auto-created)
│
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment template (safe to commit)
├── .env                              # Your API keys (gitignored)
├── .gitignore
├── streamlit_app.py                  # Streamlit Cloud entry point
├── README.md                         # This file
└── LICENSE                           # MIT License
File Details
File	Purpose
app/__init__.py	Package exports, version info
app/config.py	Environment variable loading, constants
app/main.py	Streamlit UI, page layout, widgets
app/rag_system.py	RAG pipeline: retrieval, generation, citations
app/guardrails.py	Topic classification, length limits, citation enforcement
app/ingestion.py	Document loading, cleaning, chunking, indexing
streamlit_app.py	Entry point for Streamlit Cloud deployment
🏗️ Architecture
text
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                             │
│                   (Streamlit Web App)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    GUARDRAILS LAYER                           │
│  ┌──────────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │   Topic       │  │  Output  │  │    Citation          │  │
│  │ Classifier    │  │  Limiter │  │    Enforcer          │  │
│  └──────────────┘  └──────────┘  └──────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      RAG PIPELINE                             │
│  ┌──────────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │   Query       │  │  Multi-  │  │  Cross-Encoder      │  │
│  │   Processing  │─▶│ Strategy │─▶│  Re-ranking         │  │
│  │               │  │ Retrieval│  │                     │  │
│  └──────────────┘  └──────────┘  └──────────────────────┘  │
│                                                              │
│  ┌──────────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │   Context     │  │  LLM      │  │  Response           │  │
│  │   Formatting  │─▶│ Generation│─▶│  Processing         │  │
│  │               │  │(OpenRouter)│  │                     │  │
│  └──────────────┘  └──────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────┐    ┌─────────────────────┐
│  LOCAL EMBEDDINGS    │    │    OPENROUTER API    │
│ (sentence-transformers)│    │   (Free Tier LLM)    │
│     FREE, CPU/GPU    │    │  50-1000 req/day     │
└─────────────────────┘    └─────────────────────┘

┌─────────────────────┐
│     CHROMADB         │
│  (Local Vector DB)   │
│   Persisted to Disk  │
└─────────────────────┘
Component Details
Component	Technology	Cost	Purpose
Web UI	Streamlit	Free	Interactive query interface
LLM	OpenRouter (Gemini Flash)	Free*	Response generation
Embeddings	all-MiniLM-L6-v2	Free	Local document embedding
Vector Store	ChromaDB	Free	Document retrieval
Chunking	LangChain Splitters	Free	Document segmentation
Re-ranking	ms-marco-MiniLM-L-6-v2	Free	Cross-encoder scoring
Guardrails	Custom Python	Free	Safety & quality controls
Hosting	Streamlit Cloud	Free	Application hosting
**Free tier: 50 requests/day, or 1000/day with $10 one-time purchase*

📚 Policy Corpus
The system indexes 12 Kolrose Limited policy documents (~140 pages total):

#	Document ID	Policy	Category	Est. Pages
1	KOL-HR-001	Employee Handbook	HR	8
2	KOL-HR-002	Leave and Time-Off Policy	HR	10
3	KOL-HR-003	Code of Conduct and Ethics	HR/Legal	12
4	KOL-HR-005	Remote Work Policy	HR	10
5	KOL-IT-001	IT Security & Acceptable Use	IT	14
6	KOL-FIN-001	Expenses & Reimbursement	Finance	12
7	KOL-HR-006	Performance Management	HR	12
8	KOL-HR-007	Training & Development	HR	10
9	KOL-ADMIN-001	Business Travel Policy	Admin	14
10	KOL-FIN-002	Procurement Policy	Finance	14
11	KOL-ADMIN-002	Health & Safety Policy	Admin	12
12	KOL-HR-008	Grievance & Dispute	HR	12
📝 These are synthetic policy documents created for demonstration. They do not represent actual company policies and are safe for public repository inclusion.

🛡️ Guardrails
The system implements five layers of safety controls:

Guardrail	Implementation	Trigger	Response
Corpus Boundaries	TopicClassifier	Off-topic keywords detected	Polite refusal with example questions
Output Limiting	OutputLengthGuardrail	Response > 2000 characters	Truncate at sentence boundary
Mandatory Citations	CitationGuardrail	No [KOL-XX-NNN] found	Auto-append sources section
Sensitive Topics	TopicGuardrail	Password sharing, corruption, etc.	Redirect to compliance channels
Response Validation	ResponseValidator	Hallucination indicators	Warning logged, groundedness checked
Example Guardrail Responses
Off-topic query: "What's the best restaurant near the office?"

🚫 I'm a policy assistant for Kolrose Limited and can only answer questions about our company policies. Try asking about leave, remote work, security, expenses, or training.

Sensitive query: "Can I share my password with a colleague?"

⚠️ Passwords must NEVER be shared with anyone, including IT staff. Reference: IT Security Policy [KOL-IT-001]. Report compromises to security@kolroselimited.com.ng.

📊 Evaluation
Metrics Tracked
Metric	Target	Method
Groundedness	≥ 90%	Claim-to-source overlap analysis
Citation Accuracy	≥ 95%	Document ID + section verification
Response Latency (P95)	< 3.0s	End-to-end timing
Retrieval Precision@5	≥ 80%	Human relevance judgment
Hallucination Rate	< 5%	Cross-reference with full corpus
Running Evaluation
python
from app import KolroseRAG, load_vectorstore

rag = KolroseRAG(load_vectorstore())

test_questions = [
    "What is the annual leave policy?",
    "How do I request remote work?",
    "What are the password requirements?",
]

for q in test_questions:
    result = rag.query(q)
    print(f"Groundedness: {result.metrics.get('groundedness', 'N/A')}")
    print(f"Citations: {result.citations}")
🚀 Deployment
Streamlit Cloud (Recommended)
Push code to GitHub

Go to streamlit.io/cloud

Click "New app"

Select your repository

Set main file path: app/main.py

Add secrets in Streamlit dashboard:

text
OPENROUTER_API_KEY = "sk-or-v1-your-key"
Deploy!

Render (Free Tier)
Push code to GitHub

Create Web Service on Render

Build command: pip install -r requirements.txt

Start command: streamlit run app/main.py --server.port $PORT

Add environment variables

Local Production
bash
# Using gunicorn (Linux/Mac)
gunicorn -w 2 -k uvicorn.workers.UvicornWorker app.main:app

# Or run Streamlit directly
streamlit run app/main.py --server.port 8501
📡 API Reference
Python API
python
from app import KolroseRAG

# Initialize
rag = KolroseRAG(vectorstore)

# Query
result = rag.query(
    question="What is the leave policy?",
    k_retrieve=20,      # Initial candidates
    k_final=5,          # Final results after re-ranking
    enable_rerank=True, # Use cross-encoder
    enable_guardrails=True,
)

# Access results
print(result.answer)       # Generated response
print(result.citations)    # [KOL-HR-002, KOL-HR-002 Section 1.1]
print(result.sources)      # List of source documents
print(result.metrics)      # Timing and quality metrics
print(result.refused)      # Whether query was rejected
QueryResult Fields
Field	Type	Description
question	str	Original question
answer	str	Generated response with citations
citations	List[str]	Extracted document citations
sources	List[Dict]	Source document metadata
refused	bool	Whether query was blocked
category	str	Topic classification
metrics	Dict	Performance metrics
🔧 Troubleshooting
Common Issues
Issue	Solution
ModuleNotFoundError: app	Run from project root: cd kolrose-policy-rag
OpenRouter 401 error	Check API key in .env file
OpenRouter 404 error	Try different model: LLM_MODEL=google/gemini-2.0-flash-001
ChromaDB empty	Run ingestion: python -m app.ingestion
No policy files found	Add .md files to ./policies/ directory
Embedding model download	First run downloads ~80MB model
Streamlit port in use	Change port: streamlit run app/main.py --server.port 8502
CUDA out of memory	Set EMBEDDING_DEVICE=cpu in .env
Logs
bash
# Set log level in .env
LOG_LEVEL=DEBUG

# View application output directly in terminal
streamlit run app/main.py
Reset Everything
bash
# Remove vector store and re-ingest
rm -rf chroma_db/
python -m app.ingestion
🤝 Contributing
Fork the repository

Create a feature branch: git checkout -b feature/your-feature

Make changes

Run tests: python -m app.ingestion (verify ingestion still works)

Format code: black app/

Submit pull request

📄 License
MIT License - See LICENSE file for details.

📞 Contact
Kolrose Limited
Suite 10, Bataiya Plaza, Area 2 Garki
Opposite FCDA, Abuja, FCT, Nigeria

📧 HR: hr@kolroselimited.com.ng

📧 IT Support: it@kolroselimited.com.ng

🌐 Website: https://kolroselimited.com.ng

<div align="center">
Built with ❤️ for Kolrose Limited Employees

⬆ Back to Top

</div> ```
This README provides comprehensive documentation covering setup, usage, architecture, guardrails, evaluation, deployment, and troubleshooting for your Kolrose Limited RAG system.