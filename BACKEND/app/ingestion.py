# app/ingestion.py
"""
Document Ingestion Pipeline for Kolrose Limited Policy Assistant.

Handles:
- Loading markdown policy documents
- Text cleaning and normalization
- Metadata extraction (document ID, version, department)
- Hierarchical chunking (header-aware + semantic)
- Embedding generation (local, free model)
- ChromaDB vector store creation and persistence
"""

import os
import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .config import (
    POLICIES_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COMPANY_INFO,
)

logger = logging.getLogger(__name__)

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================================
# DOCUMENT CLEANER
# ============================================================================

class DocumentCleaner:
    """
    Cleans and normalizes policy document text.
    Handles common artifacts from markdown, HTML, and PDF conversion.
    """
    
    @staticmethod
    def clean(text: str) -> str:
        """
        Apply all cleaning steps to document text.
        """
        text = DocumentCleaner._normalize_unicode(text)
        text = DocumentCleaner._normalize_whitespace(text)
        text = DocumentCleaner._fix_markdown_formatting(text)
        text = DocumentCleaner._remove_empty_sections(text)
        return text
    
    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize Unicode characters to ASCII equivalents"""
        replacements = {
            '\u2018': "'",   # Left single quote
            '\u2019': "'",   # Right single quote
            '\u201c': '"',   # Left double quote
            '\u201d': '"',   # Right double quote
            '\u2013': '-',   # En dash
            '\u2014': '--',  # Em dash
            '\u2026': '...', # Ellipsis
            '\u00a0': ' ',   # Non-breaking space
            '\u00a3': '₦',   # Pound → Naira
            '\u20a6': '₦',   # Naira sign
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace without breaking markdown structure"""
        # Replace tabs with spaces
        text = text.replace('\t', '    ')
        
        # Remove trailing whitespace per line
        lines = text.split('\n')
        cleaned = []
        prev_blank = False
        
        for line in lines:
            stripped = line.rstrip()
            is_blank = not stripped
            
            # Collapse multiple blank lines
            if is_blank and prev_blank:
                continue
            
            cleaned.append(stripped)
            prev_blank = is_blank
        
        return '\n'.join(cleaned)
    
    @staticmethod
    def _fix_markdown_formatting(text: str) -> str:
        """Fix common markdown formatting issues"""
        # Ensure space after headers
        text = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', text, flags=re.MULTILINE)
        
        # Normalize bold markers
        text = re.sub(r'__([^_]+)__', r'**\1**', text)
        
        # Fix list markers
        text = re.sub(r'^\s*[-–—]\s', '- ', text, flags=re.MULTILINE)
        
        return text
    
    @staticmethod
    def _remove_empty_sections(text: str) -> str:
        """Remove sections with headers but no content"""
        sections = text.split('\n## ')
        if len(sections) <= 1:
            return text
        
        kept = [sections[0]]
        for section in sections[1:]:
            lines = section.split('\n')
            content_lines = [
                l for l in lines[1:] 
                if l.strip() and not l.strip().startswith('#')
            ]
            if content_lines:
                kept.append('## ' + section)
        
        return '\n'.join(kept)


# ============================================================================
# METADATA EXTRACTOR
# ============================================================================

class MetadataExtractor:
    """
    Extracts structured metadata from Kolrose policy documents.
    Parses the specific header format used in Kolrose policies.
    """
    
    # Regex patterns for Kolrose document metadata
    PATTERNS = {
        'document_id': r'\*\*Document ID:\*\*\s*(KOL-\w+-\d+)',
        'version': r'\*\*Version:\*\*\s*([\d.]+)',
        'effective_date': r'\*\*Effective Date:\*\*\s*(.+?)$',
        'last_updated': r'\*\*Last Updated:\*\*\s*(.+?)$',
        'department': r'\*\*Department:\*\*\s*(.+?)$',
        'approved_by': r'\*\*Approved By:\*\*\s*(.+?)$',
    }
    
    # Policy name from main header
    POLICY_NAME_PATTERN = r'^#\s+Kolrose Limited\s*[-–]\s*(.+?)$'
    
    # Document categories by ID prefix
    CATEGORY_MAP = {
        'KOL-HR': 'Human Resources',
        'KOL-IT': 'Information Technology',
        'KOL-FIN': 'Finance',
        'KOL-ADMIN': 'Administration',
    }
    
    @classmethod
    def extract(cls, content: str, source_file: str) -> Dict[str, str]:
        """
        Extract all metadata from a policy document.
        """
        metadata = {
            'source_file': source_file,
            'document_id': 'UNKNOWN',
            'version': 'UNKNOWN',
            'department': 'UNKNOWN',
            'policy_name': source_file.replace('.md', '').replace('-', ' '),
            'approved_by': 'UNKNOWN',
            'effective_date': 'UNKNOWN',
        }
        
        # Extract from header block (first 1000 chars)
        header = content[:1000]
        
        for field, pattern in cls.PATTERNS.items():
            match = re.search(pattern, header, re.MULTILINE)
            if match:
                metadata[field] = match.group(1).strip()
        
        # Extract policy name from main header
        match = re.search(cls.POLICY_NAME_PATTERN, header, re.MULTILINE)
        if match:
            metadata['policy_name'] = match.group(1).strip()
        
        # Add category
        metadata['category'] = cls._get_category(metadata.get('document_id', ''))
        
        return metadata
    
    @classmethod
    def _get_category(cls, doc_id: str) -> str:
        """Determine document category from ID prefix"""
        for prefix, category in cls.CATEGORY_MAP.items():
            if doc_id.startswith(prefix):
                return category
        return 'General'
    
    @classmethod
    def extract_cross_references(cls, content: str) -> List[str]:
        """Extract references to other Kolrose policies"""
        pattern = r'\(?(KOL-\w+-\d+)\)?'
        refs = set(re.findall(pattern, content))
        return sorted(refs)
    
    @classmethod
    def extract_defined_terms(cls, content: str) -> List[str]:
        """Extract defined terms (bold text followed by definition)"""
        pattern = r'\*\*(.+?)\*\*\s+(?:is|are|means|refers to|shall mean)'
        terms = set(re.findall(pattern, content, re.IGNORECASE))
        return sorted(terms)


# ============================================================================
# DOCUMENT CHUNKER
# ============================================================================

class DocumentChunker:
    """
    Splits documents into semantic chunks while preserving
    header hierarchy for accurate citation tracking.
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Header-aware splitter
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "policy_title"),
                ("##", "section_header"),
                ("###", "subsection_header"),
                ("####", "sub_subsection_header"),
            ],
            return_each_line=False,
            strip_headers=False,
        )
        
        # Semantic splitter for fallback
        self.semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ", "\n### ", "\n#### ",
                "\n\n", "\n", ". ", "! ", "? ",
                ", ", " ", "",
            ],
            length_function=len,
        )
    
    def chunk_document(
        self,
        content: str,
        metadata: Dict[str, str],
    ) -> List[Document]:
        """
        Chunk a single document with hierarchy preservation.
        
        Returns:
            List of LangChain Document objects with enriched metadata
        """
        chunks = []
        
        # Try header-aware splitting first
        if self._has_sufficient_headers(content):
            try:
                md_chunks = self.header_splitter.split_text(content)
                
                for i, chunk in enumerate(md_chunks):
                    # Build hierarchical section path
                    section_path = self._build_section_path(chunk.metadata)
                    
                    # Merge original metadata with chunk metadata
                    chunk_metadata = {
                        **metadata,
                        'chunk_index': i,
                        'chunk_type': 'header_aware',
                        'section_path': section_path,
                        # Include header info
                        'policy_title': chunk.metadata.get('policy_title', ''),
                        'section_header': chunk.metadata.get('section_header', ''),
                        'subsection_header': chunk.metadata.get('subsection_header', ''),
                    }
                    
                    chunks.append(Document(
                        page_content=chunk.page_content,
                        metadata=chunk_metadata,
                    ))
                
                return chunks
                
            except Exception as e:
                logger.warning(f"Header splitting failed: {e}, falling back to semantic")
        
        # Fallback: Semantic splitting
        semantic_chunks = self.semantic_splitter.split_text(content)
        
        for i, chunk_text in enumerate(semantic_chunks):
            # Try to find nearest header for context
            section_context = self._find_nearest_header(content, chunk_text)
            
            chunk_metadata = {
                **metadata,
                'chunk_index': i,
                'chunk_type': 'semantic_fallback',
                'section_path': section_context or metadata.get('policy_name', 'Unknown'),
            }
            
            chunks.append(Document(
                page_content=chunk_text,
                metadata=chunk_metadata,
            ))
        
        return chunks
    
    def _has_sufficient_headers(self, content: str) -> bool:
        """Check if document has enough headers for hierarchical splitting"""
        headers = re.findall(r'^#{2,4}\s', content, re.MULTILINE)
        return len(headers) >= 3
    
    def _build_section_path(self, metadata: Dict) -> str:
        """Build hierarchical path from headers"""
        parts = []
        for key in ['policy_title', 'section_header', 'subsection_header']:
            value = metadata.get(key, '')
            if value:
                # Clean header text
                cleaned = re.sub(r'^#+\s*', '', value)
                parts.append(cleaned)
        
        return ' > '.join(parts) if parts else 'Main Section'
    
    def _find_nearest_header(self, full_content: str, chunk_text: str) -> Optional[str]:
        """Find the nearest header above a text chunk"""
        chunk_start = full_content.find(chunk_text[:100])
        if chunk_start == -1:
            return None
        
        preceding = full_content[:chunk_start]
        headers = list(re.finditer(
            r'^(#{2,4})\s+(.+?)$',
            preceding,
            re.MULTILINE,
        ))
        
        if headers:
            return headers[-1].group(2).strip()
        
        return None
    
    def chunk_all(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Document]:
        """
        Chunk all loaded documents.
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            if not content:
                continue
            
            chunks = self.chunk_document(content, metadata)
            
            # Add deterministic IDs and hashes
            for chunk in chunks:
                chunk.metadata['content_hash'] = hashlib.md5(
                    chunk.page_content.encode('utf-8')
                ).hexdigest()[:16]
            
            all_chunks.extend(chunks)
        
        # Add global chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk.metadata['chunk_id'] = f"chunk_{i:05d}"
        
        return all_chunks


# ============================================================================
# DOCUMENT LOADER
# ============================================================================

class PolicyDocumentLoader:
    """
    Loads policy documents from the filesystem.
    Supports markdown files with UTF-8 encoding.
    """
    
    def __init__(self, policies_path: str = POLICIES_PATH):
        self.policies_path = Path(policies_path)
        self.cleaner = DocumentCleaner()
        self.metadata_extractor = MetadataExtractor()
    
    def find_policy_files(self) -> List[Path]:
        """Find all policy markdown files (recursively)"""
        files = []
        
        if not self.policies_path.exists():
            logger.error(f"Policies path does not exist: {self.policies_path}")
            return files
        
        for root, dirs, filenames in os.walk(self.policies_path):
            for filename in sorted(filenames):
                if filename.endswith('.md') and filename != 'README.md':
                    files.append(Path(root) / filename)
        
        return files
    
    def load_document(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Load and process a single policy document"""
        try:
            # Read file
            loader = TextLoader(str(filepath), encoding='utf-8')
            docs = loader.load()
            
            if not docs:
                logger.warning(f"Empty document: {filepath}")
                return None
            
            raw_content = docs[0].page_content
            
            # Clean text
            cleaned_content = self.cleaner.clean(raw_content)
            
            # Extract metadata
            metadata = self.metadata_extractor.extract(
                cleaned_content,
                filepath.name,
            )
            
            return {
                'content': cleaned_content,
                'metadata': metadata,
                'filepath': str(filepath),
                'filename': filepath.name,
                'raw_length': len(raw_content),
                'cleaned_length': len(cleaned_content),
            }
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None
    
    def load_all(self) -> List[Dict[str, Any]]:
        """Load all policy documents"""
        files = self.find_policy_files()
        
        if not files:
            logger.warning(f"No policy files found in {self.policies_path}")
            return []
        
        documents = []
        for filepath in files:
            doc = self.load_document(filepath)
            if doc:
                documents.append(doc)
                logger.info(
                    f"Loaded: {doc['metadata'].get('document_id', 'N/A')} - "
                    f"{doc['metadata'].get('policy_name', filepath.name)}"
                )
        
        logger.info(f"Loaded {len(documents)} policy documents")
        return documents


# ============================================================================
# EMBEDDING LOADER
# ============================================================================

# Global cache for embeddings model
_embeddings_model = None


def load_embeddings(
    model_name: str = EMBEDDING_MODEL,
    device: str = EMBEDDING_DEVICE,
) -> HuggingFaceEmbeddings:
    """
    Load the embedding model (cached globally).
    
    Args:
        model_name: HuggingFace model identifier
        device: 'cpu' or 'cuda'
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    global _embeddings_model
    
    if _embeddings_model is None:
        logger.info(f"Loading embedding model: {model_name}")
        _embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 16,
            },
        )
        logger.info(f"Embedding model loaded on {device}")
    
    return _embeddings_model


# ============================================================================
# VECTOR STORE MANAGER
# ============================================================================

class VectorStoreManager:
    """
    Manages ChromaDB vector store operations.
    """
    
    def __init__(
        self,
        persist_directory: str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
    
    def exists(self) -> bool:
        """Check if vector store already exists"""
        path = Path(self.persist_directory)
        return path.exists() and any(path.iterdir())
    
    def create(
        self,
        chunks: List[Document],
        embeddings: HuggingFaceEmbeddings,
    ) -> Chroma:
        """
        Create a new vector store from document chunks.
        """
        logger.info(f"Creating vector store: {self.collection_name}")
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata={
                "organization": COMPANY_INFO['name'],
                "location": COMPANY_INFO['address'],
                "created_at": datetime.now().isoformat(),
                "chunk_count": str(len(chunks)),
            },
        )
        
        # Persist to disk
        vectorstore.persist()
        
        logger.info(f"Vector store created with {len(chunks)} vectors")
        logger.info(f"Persisted to: {self.persist_directory}")
        
        return vectorstore
    
    def load(self, embeddings: HuggingFaceEmbeddings) -> Chroma:
        """
        Load an existing vector store.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_directory}. "
                "Run ingestion first."
            )
        
        logger.info(f"Loading vector store: {self.collection_name}")
        
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
            collection_name=self.collection_name,
        )
        
        count = vectorstore._collection.count()
        logger.info(f"Loaded {count} vectors from {self.persist_directory}")
        
        return vectorstore
    
    def get_or_create(
        self,
        chunks: List[Document],
        embeddings: HuggingFaceEmbeddings,
    ) -> Chroma:
        """
        Load existing vector store or create new one.
        """
        if self.exists():
            return self.load(embeddings)
        return self.create(chunks, embeddings)


# ============================================================================
# MAIN INGESTION FUNCTION
# ============================================================================

def ingest_policies(
    policies_path: str = POLICIES_PATH,
    chroma_path: str = CHROMA_PATH,
    force_recreate: bool = False,
    verbose: bool = True,
) -> Tuple[Optional[Chroma], Dict[str, Any]]:
    """
    Complete ingestion pipeline:
    Load → Clean → Extract Metadata → Chunk → Embed → Index
    
    Args:
        policies_path: Path to policy markdown files
        chroma_path: Path for ChromaDB persistence
        force_recreate: If True, recreate vector store even if exists
        verbose: Print progress information
        
    Returns:
        Tuple of (vectorstore, ingestion_stats)
    """
    start_time = datetime.now()
    
    stats = {
        'started_at': start_time.isoformat(),
        'policies_path': policies_path,
        'chroma_path': chroma_path,
        'documents_found': 0,
        'documents_loaded': 0,
        'chunks_created': 0,
        'vectors_indexed': 0,
        'errors': [],
    }
    
    if verbose:
        print("=" * 60)
        print(f"📥 {COMPANY_INFO['name']} - Document Ingestion")
        print(f"📍 {COMPANY_INFO['address']}")
        print("=" * 60)
    
    # Step 1: Load documents
    if verbose:
        print("\n[1/4] Loading policy documents...")
    
    loader = PolicyDocumentLoader(policies_path)
    documents = loader.load_all()
    
    stats['documents_found'] = len(loader.find_policy_files())
    stats['documents_loaded'] = len(documents)
    
    if not documents:
        error_msg = f"No documents found in {policies_path}"
        stats['errors'].append(error_msg)
        if verbose:
            print(f"   ❌ {error_msg}")
        return None, stats
    
    if verbose:
        print(f"   ✅ Loaded {len(documents)} documents")
        for doc in documents:
            meta = doc['metadata']
            print(f"      📄 [{meta.get('document_id', 'N/A')}] {meta.get('policy_name', '?')}")
    
    # Step 2: Chunk documents
    if verbose:
        print(f"\n[2/4] Chunking documents...")
    
    chunker = DocumentChunker()
    chunks = chunker.chunk_all(documents)
    
    stats['chunks_created'] = len(chunks)
    
    if not chunks:
        error_msg = "No chunks created from documents"
        stats['errors'].append(error_msg)
        if verbose:
            print(f"   ❌ {error_msg}")
        return None, stats
    
    if verbose:
        print(f"   ✅ Created {len(chunks)} chunks")
        # Show chunk types
        types = {}
        for c in chunks:
            t = c.metadata.get('chunk_type', 'unknown')
            types[t] = types.get(t, 0) + 1
        for t, count in types.items():
            print(f"      {t}: {count} chunks")
    
    # Step 3: Load embeddings
    if verbose:
        print(f"\n[3/4] Loading embedding model...")
    
    embeddings = load_embeddings()
    
    if verbose:
        print(f"   ✅ Model: {EMBEDDING_MODEL}")
        print(f"   ✅ Device: {EMBEDDING_DEVICE}")
    
    # Step 4: Create/update vector store
    if verbose:
        print(f"\n[4/4] Indexing into vector store...")
    
    store_manager = VectorStoreManager(chroma_path)
    
    if force_recreate and store_manager.exists():
        if verbose:
            print(f"   🗑️ Removing existing vector store...")
        import shutil
        shutil.rmtree(chroma_path)
    
    try:
        vectorstore = store_manager.get_or_create(chunks, embeddings)
        stats['vectors_indexed'] = vectorstore._collection.count()
        
        if verbose:
            print(f"   ✅ Indexed {stats['vectors_indexed']} vectors")
            print(f"   💾 Persisted to: {chroma_path}")
            
    except Exception as e:
        error_msg = f"Vector store creation failed: {str(e)}"
        stats['errors'].append(error_msg)
        if verbose:
            print(f"   ❌ {error_msg}")
        return None, stats
    
    # Final stats
    elapsed = (datetime.now() - start_time).total_seconds()
    stats['completed_at'] = datetime.now().isoformat()
    stats['elapsed_seconds'] = elapsed
    stats['success'] = True
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✅ INGESTION COMPLETE!")
        print(f"   📄 {stats['documents_loaded']} documents")
        print(f"   🧩 {stats['chunks_created']} chunks")
        print(f"   📊 {stats['vectors_indexed']} vectors")
        print(f"   ⏱️ {elapsed:.1f} seconds")
        print(f"   💾 {chroma_path}")
        print(f"{'='*60}")
    
    return vectorstore, stats


def load_vectorstore(
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> Optional[Chroma]:
    """
    Load an existing vector store.
    Use this in the web app to avoid re-ingestion.
    
    Returns:
        Chroma vector store or None if not found
    """
    store_manager = VectorStoreManager(chroma_path, collection_name)
    
    if not store_manager.exists():
        logger.warning(f"Vector store not found at {chroma_path}")
        return None
    
    embeddings = load_embeddings()
    
    try:
        return store_manager.load(embeddings)
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_ingestion_stats(chroma_path: str = CHROMA_PATH) -> Dict[str, Any]:
    """Get statistics about the ingested vector store"""
    store_manager = VectorStoreManager(chroma_path)
    
    if not store_manager.exists():
        return {
            'exists': False,
            'message': f"No vector store at {chroma_path}",
        }
    
    try:
        embeddings = load_embeddings()
        vectorstore = store_manager.load(embeddings)
        count = vectorstore._collection.count()
        
        return {
            'exists': True,
            'vector_count': count,
            'path': chroma_path,
            'collection_name': COLLECTION_NAME,
            'embedding_model': EMBEDDING_MODEL,
        }
    except Exception as e:
        return {
            'exists': True,
            'error': str(e),
            'path': chroma_path,
        }


def check_policies_exist(policies_path: str = POLICIES_PATH) -> Tuple[bool, int, List[str]]:
    """Check if policy files exist"""
    path = Path(policies_path)
    
    if not path.exists():
        return False, 0, []
    
    files = []
    for root, dirs, filenames in os.walk(path):
        for f in sorted(filenames):
            if f.endswith('.md') and f != 'README.md':
                files.append(f)
    
    return len(files) > 0, len(files), files


if __name__ == "__main__":
    # Run ingestion when executed directly
    import sys
    
    print(f"\n{'='*60}")
    print(f"  {COMPANY_INFO['name']} - Policy Ingestion Tool")
    print(f"  {COMPANY_INFO['address']}")
    print(f"{'='*60}\n")
    
    # Check for policies
    exists, count, files = check_policies_exist()
    
    if not exists:
        print(f"❌ No policy files found in {POLICIES_PATH}")
        print("   Add markdown policy files and try again.")
        sys.exit(1)
    
    print(f"📁 Found {count} policy files")
    
    # Safe automated terminal check for non-interactive environments (Cloud/Railway)
    store_manager = VectorStoreManager()
    force_recreate_flag = False

    if store_manager.exists():
        # Check if environment variables or un-attached shell indicates production cloud build
        if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("PORT") or not sys.stdin.isatty():
            print(f"\n⚠️ Vector store already exists at {CHROMA_PATH}")
            print("   Non-interactive or Cloud environment detected. Bypassing prompt and skipping recreation.")
            force_recreate_flag = False
        else:
            # Fallback to interactive mode only if run locally in an active terminal
            try:
                response = input(
                    f"\n⚠️ Vector store already exists at {CHROMA_PATH}\n"
                    "   [R]ecreate | [S]kip | [Q]uit\n"
                    "   Choice: "
                ).strip().lower()
                
                if response == 'q':
                    print("Exiting.")
                    sys.exit(0)
                elif response == 's':
                    print("Skipping ingestion. Existing store preserved.")
                    sys.exit(0)
                elif response == 'r':
                    print("Recreating vector store...")
                    force_recreate_flag = True
            except EOFError:
                print("\n⚠️ Standard input missing. Falling back to default: Skipping store recreation.")
                force_recreate_flag = False
    else:
        # DB does not exist, force baseline creation pipeline
        force_recreate_flag = True
    
    # Execute the ingestion lifecycle safely
    if force_recreate_flag or not store_manager.exists():
        vectorstore, stats = ingest_policies(force_recreate=force_recreate_flag)
        if stats.get('success'):
            print(f"\n✅ Done! {stats['vectors_indexed']} vectors indexed.")
        else:
            print(f"\n❌ Ingestion failed: {stats.get('errors')}")
            sys.exit(1)
    else:
        print("\n✅ Ingestion step completed. Vector store verified and ready.")