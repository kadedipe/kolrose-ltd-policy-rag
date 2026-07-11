# app/rag_system.py
"""
Core RAG System for Kolrose Limited Policy Assistant.

Handles:
- Document retrieval (Top-k, MMR)
- Cross-encoder re-ranking
- Context formatting with citations
- LLM generation via OpenRouter
- Response processing
"""

import re
import time
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import requests

from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    RETRIEVAL_K,
    FINAL_K,
    MAX_OUTPUT_TOKENS,
    MAX_RESPONSE_CHARS,
    COMPANY_INFO,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class TopicCategory(Enum):
    """Categories for classifying user queries"""
    IN_CORPUS = "in_corpus"
    OUT_OF_CORPUS = "out_of_corpus"
    SENSITIVE = "sensitive"
    OFF_TOPIC = "off_topic"


class RetrievalMethod(Enum):
    """Available retrieval methods"""
    SIMILARITY = "similarity"
    MMR = "mmr"
    HYBRID = "hybrid"


@dataclass
class RetrievedDocument:
    """A retrieved document chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    score: float
    ce_score: Optional[float] = None
    rank: int = 0
    
    @property
    def document_id(self) -> str:
        return self.metadata.get('document_id', 'Unknown')
    
    @property
    def source_file(self) -> str:
        return self.metadata.get('source_file', 'Unknown')
    
    @property
    def section(self) -> str:
        return (self.metadata.get('section') or 
                self.metadata.get('h2') or 
                self.metadata.get('subsection', 'N/A'))


@dataclass
class QueryResult:
    """Complete result from a RAG query"""
    question: str
    answer: str
    sources: List[Dict[str, str]]
    citations: List[str]
    refused: bool = False
    category: str = "in_corpus"
    refusal_reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


# ============================================================================
# TOPIC CLASSIFIER
# ============================================================================

class TopicClassifier:
    """
    Classifies user queries to determine if they fall within
    Kolrose policy corpus boundaries.
    """
    
    # Topics covered by Kolrose policies
    CORPUS_TOPICS: Dict[str, List[str]] = {
        'leave': [
            'leave', 'vacation', 'pto', 'sick day', 'sick leave',
            'maternity', 'paternity', 'time off', 'annual leave',
            'carryover', 'accrual', 'holiday',
        ],
        'remote_work': [
            'remote', 'work from home', 'wfh', 'hybrid',
            'telecommuting', 'flexible work', 'home office',
        ],
        'security': [
            'password policy', 'password requirement', 'password must',
            'vpn', 'mfa', 'multi-factor', 'authentication',
            'access control', 'data protection',
        ],
        'conduct': [
            'code of conduct', 'ethics', 'confidentiality',
            'discipline', 'warning', 'termination', 'dress code',
        ],
        'expenses': [
            'expense', 'reimbursement', 'travel cost', 'per diem',
            'allowance', 'mileage', 'receipt',
        ],
        'performance': [
            'performance', 'review', 'appraisal', 'pip',
            'improvement plan', 'promotion', 'rating', 'bonus',
        ],
        'training': [
            'training', 'certification', 'development', 'learning',
            'education', 'mentorship', 'internship', 'graduate trainee',
        ],
        'travel': [
            'travel', 'flight', 'hotel', 'accommodation', 'transport',
            'airport', 'booking', 'travel advance',
        ],
        'procurement': [
            'procurement', 'purchase', 'vendor', 'supplier',
            'tender', 'contract', 'purchase order',
        ],
        'health_safety': [
            'health', 'safety', 'emergency', 'fire', 'evacuation',
            'first aid', 'medical', 'wellness',
        ],
        'grievance': [
            'grievance', 'complaint', 'dispute', 'appeal',
            'whistleblowing', 'report',
        ],
        'company_info': [
            'kolrose', 'abuja', 'bataiya plaza', 'headquarters',
            'office location', 'company address', 'working hours',
        ],
    }
    
    # Indicators that a query is off-topic
    OFF_TOPIC_INDICATORS: List[str] = [
        'restaurant', 'weather', 'sports', 'entertainment',
        'movie', 'music', 'recipe', 'cooking', 'gossip',
        'celebrity', 'stock market', 'crypto', 'bitcoin',
        'election', 'politics', 'religion', 'personal advice',
        'medical advice', 'legal advice', 'tax advice',
        'real estate', 'car dealer', 'fashion',
    ]
    
    # Sensitive topics requiring special handling
    SENSITIVE_TOPICS: Dict[str, List[str]] = {
        'password_sharing': [
            'share password', 'give password', 'tell password',
            'share my password', 'give my password', 'tell someone',
            'give out', 'hand over password', 'disclose password',
            'reveal password',
        ],
        'corruption': [
            'bribe', 'kickback', 'corruption', 'fraudulent',
            'illegal payment', 'under the table',
        ],
        'harassment': [
            'harass', 'bully', 'discriminate', 'hostile work',
            'sexual harassment',
        ],
        'unauthorized_access': [
            'someone else password', 'use their account',
            'log in as', 'access without permission',
            'hack', 'bypass security',
        ],
    }
    
    @classmethod
    def classify(cls, query: str) -> Tuple[TopicCategory, float, str]:
        """
        Classify a query into a topic category.
        
        Args:
            query: The user's question
            
        Returns:
            Tuple of (category, confidence, reasoning)
        """
        query_lower = query.lower().strip()
        
        # Step 1: Check sensitive topics (highest priority)
        for topic, keywords in cls.SENSITIVE_TOPICS.items():
            matches = [kw for kw in keywords if kw in query_lower]
            if matches:
                return (
                    TopicCategory.SENSITIVE,
                    0.95,
                    f"Sensitive topic detected: {topic} (matched: {matches[:2]})"
                )
        
        # Step 2: Check off-topic indicators
        off_topic_matches = [
            t for t in cls.OFF_TOPIC_INDICATORS 
            if t in query_lower
        ]
        if off_topic_matches:
            return (
                TopicCategory.OFF_TOPIC,
                0.95,
                f"Off-topic indicators: {off_topic_matches[:3]}"
            )
        
        # Step 3: Check in-corpus topics
        topic_scores = {}
        for topic, keywords in cls.CORPUS_TOPICS.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches > 0:
                topic_scores[topic] = matches / len(keywords)
        
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            confidence = min(topic_scores[best_topic] * 2, 0.95)
            return (
                TopicCategory.IN_CORPUS,
                confidence,
                f"Matched policy topic: {best_topic}"
            )
        
        # Step 4: Default to out of corpus
        return (
            TopicCategory.OUT_OF_CORPUS,
            0.70,
            "No matching policy topics found"
        )


# ============================================================================
# RETRIEVAL ENGINE
# ============================================================================

class RetrievalEngine:
    """
    Handles document retrieval with multiple strategies.
    """
    
    def __init__(self, vectorstore, cross_encoder=None):
        self.vectorstore = vectorstore
        self.collection = vectorstore._collection
        self.cross_encoder = cross_encoder
    
    def retrieve_similarity(
        self, 
        query_embedding: List[float], 
        k: int = 20
    ) -> List[RetrievedDocument]:
        """Basic similarity-based retrieval"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        docs = []
        for i in range(len(results['documents'][0])):
            docs.append(RetrievedDocument(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                score=1 - results['distances'][0][i],
                rank=i + 1,
            ))
        
        return docs
    
    def retrieve_mmr(
        self,
        query_embedding: List[float],
        k: int = 20,
        fetch_k: int = 30,
        lambda_mult: float = 0.7,
    ) -> List[RetrievedDocument]:
        """
        Maximal Marginal Relevance retrieval.
        Balances relevance with diversity.
        """
        # Fetch more candidates
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=['documents', 'metadatas', 'embeddings', 'distances']
        )
        
        # Extract embeddings
        candidate_embeddings = np.array(results['embeddings'][0])
        query_emb = np.array(query_embedding)
        
        # Normalize
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        candidate_embeddings = candidate_embeddings / (norms + 1e-10)
        
        # MMR selection
        selected_indices = []
        remaining = list(range(len(candidate_embeddings)))
        
        for _ in range(min(k, len(remaining))):
            if not selected_indices:
                similarities = np.dot(candidate_embeddings[remaining], query_emb)
                best_idx = remaining[int(np.argmax(similarities))]
            else:
                selected_embs = candidate_embeddings[selected_indices]
                mmr_scores = []
                
                for idx in remaining:
                    rel = float(np.dot(candidate_embeddings[idx], query_emb))
                    div = float(np.max(np.dot(
                        candidate_embeddings[idx:idx+1], 
                        selected_embs.T
                    )))
                    mmr = lambda_mult * rel - (1 - lambda_mult) * div
                    mmr_scores.append(mmr)
                
                best_idx = remaining[int(np.argmax(mmr_scores))]
            
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
        
        # Format results
        docs = []
        for rank, idx in enumerate(selected_indices):
            docs.append(RetrievedDocument(
                content=results['documents'][0][idx],
                metadata=results['metadatas'][0][idx],
                score=1 - results['distances'][0][idx],
                rank=rank + 1,
            ))
        
        return docs
    
    def rerank(
        self,
        query: str,
        candidates: List[RetrievedDocument],
        top_n: int = 5,
    ) -> List[RetrievedDocument]:
        """
        Re-rank candidates using cross-encoder model.
        """
        if not candidates or self.cross_encoder is None:
            return candidates[:top_n]
        
        # Prepare pairs
        pairs = [[query, doc.content[:500]] for doc in candidates]
        
        # Score
        scores = self.cross_encoder.predict(
            pairs, 
            batch_size=16,
            show_progress_bar=False,
        )
        
        # Assign scores
        for doc, score in zip(candidates, scores):
            doc.ce_score = float(score)
        
        # Sort by cross-encoder score
        reranked = sorted(candidates, key=lambda x: x.ce_score or 0, reverse=True)
        
        # Update ranks
        for i, doc in enumerate(reranked[:top_n]):
            doc.rank = i + 1
        
        return reranked[:top_n]


# ============================================================================
# RESPONSE FORMATTER
# ============================================================================

class ResponseFormatter:
    """Formats retrieved documents into LLM context with citations"""
    
    @staticmethod
    def format_context(docs: List[RetrievedDocument]) -> str:
        """Format documents for LLM context window"""
        parts = []
        for doc in docs:
            # Build citation header
            doc_id = doc.document_id
            source = doc.source_file
            section = doc.section
            
            header = f"[{doc_id}] {source}"
            if section and section != 'N/A':
                header += f" — Section: {section}"
            
            # Add score info
            score_info = f" (relevance: {doc.score:.2f}"
            if doc.ce_score:
                score_info += f", ce: {doc.ce_score:.2f}"
            score_info += ")"
            
            parts.append(
                f"{'─' * 60}\n"
                f"DOCUMENT {doc.rank}: {header}{score_info}\n"
                f"{'─' * 60}\n"
                f"{doc.content}\n"
            )
        
        return "\n".join(parts)
    
    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """Extract document citations from response text"""
        patterns = [
            r'\[KOL-\w+-\d+[,\s]*[§]?[\d.]*\]',
            r'\[KOL-\w+-\d+\]',
            r'KOL-\w+-\d+',
        ]
        
        citations = set()
        for pattern in patterns:
            citations.update(re.findall(pattern, text))
        
        return sorted(citations)
    
    @staticmethod
    def build_prompt(context: str, question: str) -> str:
        """Build the complete prompt for the LLM"""
        return f"""🏢 **{COMPANY_INFO['name']}** — HR Policy Assistant
📍 {COMPANY_INFO['address']}

You are an authoritative policy assistant. Follow these rules strictly:

**1. SOURCE GROUNDING**
   - Answer ONLY from the provided policy documents
   - Never invent policies or assume information

**2. CITATION FORMAT**
   - Every factual claim: [Document ID, Section X.Y]
   - Example: "Employees receive 15 days annual leave [KOL-HR-002, Section 1.1]"

**3. HANDLING GAPS**
   - If information is missing: "This topic is not covered in our current policy documents."
   - Never guess or provide unofficial information

**4. RESPONSE STRUCTURE**
   - Direct answer first, then supporting details
   - List multiple policy references when applicable
   - Be concise but complete

**POLICY DOCUMENTS:**
{context}

**QUESTION:** {question}

**ANSWER:**"""


# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMClient:
    """Handles API calls to OpenRouter for LLM generation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL
        self.model = DEFAULT_MODEL
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response from the LLM.
        
        Returns:
            Tuple of (response_text, metadata)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": COMPANY_INFO['website'],
            "X-Title": "Kolrose Policy RAG",
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data['choices'][0]['message']['content']
                metadata = {
                    'model': data.get('model', self.model),
                    'usage': data.get('usage', {}),
                    'status_code': 200,
                }
                return answer, metadata
            else:
                error_msg = f"API Error ({response.status_code}): {response.text[:200]}"
                logger.error(error_msg)
                return error_msg, {'status_code': response.status_code, 'error': response.text}
                
        except requests.exceptions.Timeout:
            return "Request timed out. Please try again.", {'status_code': 408}
        except Exception as e:
            logger.exception("LLM generation failed")
            return f"Error generating response: {str(e)}", {'status_code': 500}


# ============================================================================
# MAIN RAG SYSTEM
# ============================================================================

class KolroseRAG:
    """
    Complete RAG system for Kolrose Limited policy queries.
    
    Usage:
        rag = KolroseRAG(vectorstore, llm_api_key="sk-or-v1-...")
        result = rag.query("What is the annual leave policy?")
        print(result.answer)
        print(result.citations)
    """
    
    def __init__(
        self,
        vectorstore,
        llm_api_key: Optional[str] = None,
        cross_encoder=None,
    ):
        """
        Initialize the RAG system.
        
        Args:
            vectorstore: ChromaDB vector store instance
            llm_api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            cross_encoder: Optional cross-encoder model for re-ranking
        """
        self.vectorstore = vectorstore
        self.llm_client = LLMClient(api_key=llm_api_key)
        self.retrieval_engine = RetrievalEngine(vectorstore, cross_encoder)
        self.formatter = ResponseFormatter()
        self.classifier = TopicClassifier()
    
    def query(
        self,
        question: str,
        retrieval_method: RetrievalMethod = RetrievalMethod.MMR,
        k_retrieve: int = RETRIEVAL_K,
        k_final: int = FINAL_K,
        enable_rerank: bool = True,
        enable_guardrails: bool = True,
    ) -> QueryResult:
        """
        Execute a complete RAG query.
        
        Args:
            question: User's policy question
            retrieval_method: Retrieval strategy to use
            k_retrieve: Number of initial candidates to retrieve
            k_final: Number of final results after re-ranking
            enable_rerank: Whether to use cross-encoder re-ranking
            enable_guardrails: Whether to enforce topic boundaries
            
        Returns:
            QueryResult with answer, sources, citations, and metrics
        """
        start_time = time.time()
        
        # =====================================================================
        # Step 1: Topic Classification (Guardrail)
        # =====================================================================
        if enable_guardrails:
            category, confidence, reason = self.classifier.classify(question)
            
            # Handle out-of-corpus queries
            if category == TopicCategory.OFF_TOPIC:
                return QueryResult(
                    question=question,
                    answer=(
                        "🚫 I can only answer questions about **Kolrose Limited** "
                        "company policies.\n\n"
                        "**Try asking about:**\n"
                        "- Leave and time-off policies\n"
                        "- Remote work arrangements\n"
                        "- Security and IT policies\n"
                        "- Travel and expense reimbursement\n"
                        "- Training and development\n"
                        "- Procurement procedures"
                    ),
                    sources=[],
                    citations=[],
                    refused=True,
                    category=category.value,
                    refusal_reason=reason,
                    metrics={'total_ms': round((time.time() - start_time) * 1000)},
                )
            
            # Handle out-of-corpus (no matching topics)
            if category == TopicCategory.OUT_OF_CORPUS:
                return QueryResult(
                    question=question,
                    answer=(
                        "📋 I'm unable to answer this as it's outside our "
                        "policy documents. Please contact HR at "
                        f"{COMPANY_INFO['email_hr']} for assistance."
                    ),
                    sources=[],
                    citations=[],
                    refused=True,
                    category=category.value,
                    refusal_reason=reason,
                    metrics={'total_ms': round((time.time() - start_time) * 1000)},
                )
            
            # Handle sensitive topics
            if category == TopicCategory.SENSITIVE:
                return QueryResult(
                    question=question,
                    answer=(
                        f"⚠️ This relates to a sensitive topic. For matters involving "
                        f"security, ethics, or legal concerns, please use the appropriate "
                        f"reporting channels:\n\n"
                        f"📧 Compliance Officer: {COMPANY_INFO['email_compliance']}\n"
                        f"📞 Whistleblower Hotline: {COMPANY_INFO['hotline_whistleblower']}\n\n"
                        f"Reference: Code of Conduct [KOL-HR-003]"
                    ),
                    sources=[{
                        'document_id': 'KOL-HR-003',
                        'policy_name': 'Code of Conduct and Ethics',
                    }],
                    citations=['KOL-HR-003'],
                    refused=True,
                    category=category.value,
                    refusal_reason=reason,
                    metrics={'total_ms': round((time.time() - start_time) * 1000)},
                )
        
        # =====================================================================
        # Step 2: Retrieve Documents
        # =====================================================================
        retrieval_start = time.time()
        
        # Get query embedding
        from .ingestion import load_embeddings
        embeddings = load_embeddings()
        query_embedding = embeddings.embed_query(question)
        
        # Retrieve based on method
        if retrieval_method == RetrievalMethod.MMR:
            candidates = self.retrieval_engine.retrieve_mmr(
                query_embedding, k=k_retrieve
            )
        else:
            candidates = self.retrieval_engine.retrieve_similarity(
                query_embedding, k=k_retrieve
            )
        
        retrieval_time = time.time() - retrieval_start
        
        # =====================================================================
        # Step 3: Re-rank (Optional)
        # =====================================================================
        if enable_rerank:
            docs = self.retrieval_engine.rerank(question, candidates, top_n=k_final)
        else:
            docs = candidates[:k_final]
        
        # =====================================================================
        # Step 4: Build Prompt & Generate
        # =====================================================================
        context = self.formatter.format_context(docs)
        prompt = self.formatter.build_prompt(context, question)
        
        generation_start = time.time()
        answer, llm_metadata = self.llm_client.generate(prompt)
        generation_time = time.time() - generation_start
        
        # =====================================================================
        # Step 5: Process Response
        # =====================================================================
        
        # Enforce output length limits
        if len(answer) > MAX_RESPONSE_CHARS:
            # Truncate at last complete sentence
            truncated = answer[:MAX_RESPONSE_CHARS]
            last_break = max(
                truncated.rfind('. '),
                truncated.rfind('\n\n'),
                MAX_RESPONSE_CHARS - 100,
            )
            answer = answer[:last_break] + "\n\n[Response truncated for length]"
        
        # Extract citations
        citations = self.formatter.extract_citations(answer)
        
        # Build source list
        sources = []
        seen_ids = set()
        for doc in docs:
            if doc.document_id not in seen_ids and doc.document_id != 'Unknown':
                seen_ids.add(doc.document_id)
                sources.append({
                    'document_id': doc.document_id,
                    'policy_name': doc.metadata.get('policy_name', 'Unknown'),
                    'source_file': doc.source_file,
                    'section': doc.section,
                })
        
        # Calculate total time
        total_time = time.time() - start_time
        
        return QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            citations=citations,
            refused=False,
            category=TopicCategory.IN_CORPUS.value,
            metrics={
                'retrieval_ms': round(retrieval_time * 1000),
                'generation_ms': round(generation_time * 1000),
                'total_ms': round(total_time * 1000),
                'num_candidates': len(candidates),
                'num_final_docs': len(docs),
                'num_sources': len(sources),
                'num_citations': len(citations),
                'llm_metadata': llm_metadata,
            },
        )
    
    def batch_query(self, questions: List[str], **kwargs) -> List[QueryResult]:
        """Execute multiple queries"""
        return [self.query(q, **kwargs) for q in questions]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_policy_mentions(text: str) -> Dict[str, List[str]]:
    """
    Extract all policy document mentions from text.
    Useful for analytics and verification.
    """
    policy_pattern = r'KOL-\w+-\d+'
    section_pattern = r'Section\s+[\d.]+'
    
    policies = re.findall(policy_pattern, text)
    sections = re.findall(section_pattern, text)
    
    return {
        'policies': list(set(policies)),
        'sections': list(set(sections)),
    }


def compute_groundedness(answer: str, source_docs: List[RetrievedDocument]) -> float:
    """
    Estimate groundedness by checking if answer claims appear in source documents.
    Simple heuristic version.
    """
    if not source_docs:
        return 0.0
    
    # Split answer into sentences
    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return 1.0
    
    supported_count = 0
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Check if sentence has overlap with any source document
        for doc in source_docs:
            # Extract key terms (words > 4 chars)
            key_terms = [w for w in sentence_lower.split() if len(w) > 4]
            matches = sum(1 for term in key_terms if term in doc.content.lower())
            
            if len(key_terms) > 0 and matches / len(key_terms) > 0.4:
                supported_count += 1
                break
    
    return supported_count / len(sentences)


if __name__ == "__main__":
    # Quick test
    print(f"Kolrose RAG System v1.0")
    print(f"Classifier test: {TopicClassifier.classify('What is the leave policy?')}")
    print("Module loaded successfully!")