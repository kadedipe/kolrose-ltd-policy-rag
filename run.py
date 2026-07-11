# run.py - Simple launcher for Kolrose RAG System
import subprocess
import sys
import os

# Set paths
os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), 'BACKEND')
os.environ['CHROMA_PATH'] = os.path.join(os.getcwd(), 'chroma_db')
os.environ['POLICIES_PATH'] = os.path.join(os.getcwd(), 'DATA', 'policies')

print("=" * 60)
print("🏢 Kolrose Limited - Policy Assistant")
print("📍 Suite 10, Bataiya Plaza, Abuja, FCT")
print("=" * 60)
print()
print("📥 Checking embedding model...")

# Pre-download model
from sentence_transformers import SentenceTransformer
print("   Downloading model (one-time, ~80MB)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("   ✅ Model ready!")

print("💾 Loading vector store...")
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
)

chroma_path = os.environ.get('CHROMA_PATH', './chroma_db')
vectorstore = Chroma(
    persist_directory=chroma_path,
    embedding_function=embeddings,
    collection_name="kolrose_policies_v2",
)
count = vectorstore._collection.count()
print(f"   ✅ Loaded {count} vectors")

print()
print("=" * 60)
print("🚀 Starting web interface...")
print("   Open http://localhost:8501 in your browser")
print("=" * 60)

# Launch Streamlit
subprocess.run([
    sys.executable, '-m', 'streamlit', 'run', 
    'FRONTEND/app/main.py',
    '--server.headless', 'true',
])