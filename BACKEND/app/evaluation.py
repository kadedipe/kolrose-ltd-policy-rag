# app/evaluation.py
"""
Complete Evaluation Framework for Kolrose Limited RAG System.
=============================================================
Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria

Comprehensive evaluation covering:
1. Groundedness - % of answers fully supported by retrieved evidence
2. Citation Accuracy - % of citations correctly referencing documents
3. Gold Answer Matching - Exact/Partial/Semantic match against gold answers
4. System Latency - P50/P95/P99 response time metrics
5. Ablation Studies - Compare retrieval k, re-ranking, retrieval methods
"""

import os
import sys
import json
import time
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import COMPANY_INFO, CHROMA_PATH
from app.ingestion import load_vectorstore, load_embeddings


# ============================================================================
# EVALUATION TEST SET (25 Questions covering all policy categories)
# ============================================================================

EVALUATION_QUESTIONS = [
    # ── LEAVE & TIME-OFF (6 questions) ──────────────────────────────────
    {
        "id": "EVAL-001",
        "question": "What is the annual leave entitlement for a new employee at Kolrose Limited?",
        "category": "leave",
        "difficulty": "easy",
        "expected_documents": ["KOL-HR-002"],
        "expected_keywords": ["15", "working days", "annual leave", "0-2 years"],
        "ground_truth_snippet": "15 working days for 0-2 years of service",
    },
    {
        "id": "EVAL-002",
        "question": "How many days of sick leave do employees receive per year?",
        "category": "leave",
        "difficulty": "easy",
        "expected_documents": ["KOL-HR-002"],
        "expected_keywords": ["12", "working days", "sick leave", "paid"],
        "ground_truth_snippet": "12 working days of paid sick leave per calendar year",
    },
    {
        "id": "EVAL-003",
        "question": "What is the maternity leave policy at Kolrose?",
        "category": "leave",
        "difficulty": "medium",
        "expected_documents": ["KOL-HR-002"],
        "expected_keywords": ["16 weeks", "112 calendar days", "full pay", "maternity"],
        "ground_truth_snippet": "16 weeks (112 calendar days) of maternity leave with full pay",
    },
    {
        "id": "EVAL-004",
        "question": "Can I carry over unused vacation days to the next year?",
        "category": "leave",
        "difficulty": "medium",
        "expected_documents": ["KOL-HR-002"],
        "expected_keywords": ["5", "carryover", "maximum", "March 31"],
        "ground_truth_snippet": "maximum of 5 unused leave days may be carried over",
    },
    {
        "id": "EVAL-005",
        "question": "How much notice do I need to give for annual leave?",
        "category": "leave",
        "difficulty": "easy",
        "expected_documents": ["KOL-HR-002"],
        "expected_keywords": ["14", "calendar days", "advance", "request"],
        "ground_truth_snippet": "requested at least 14 calendar days in advance",
    },
    {
        "id": "EVAL-006",
        "question": "What happens to my leave if I'm serving notice period?",
        "category": "leave",
        "difficulty": "hard",
        "expected_documents": ["KOL-HR-002"],
        "expected_keywords": ["notice period", "exhaust", "remaining leave", "discretion"],
        "ground_truth_snippet": "may be required to exhaust remaining leave during the notice period",
    },
    # ── REMOTE WORK (3 questions) ───────────────────────────────────────
    {
        "id": "EVAL-007",
        "question": "How many days per week can I work remotely at Kolrose?",
        "category": "remote_work",
        "difficulty": "easy",
        "expected_documents": ["KOL-HR-005"],
        "expected_keywords": ["2 days", "remote", "hybrid", "3 days on-site"],
        "ground_truth_snippet": "2 days remote per week, 3 days on-site",
    },
    {
        "id": "EVAL-008",
        "question": "Can I work remotely while on probation?",
        "category": "remote_work",
        "difficulty": "medium",
        "expected_documents": ["KOL-HR-005"],
        "expected_keywords": ["probation", "completed", "ineligible", "not approved"],
        "ground_truth_snippet": "must have completed probationary period",
    },
    {
        "id": "EVAL-009",
        "question": "What equipment does Kolrose provide for remote workers?",
        "category": "remote_work",
        "difficulty": "easy",
        "expected_documents": ["KOL-HR-005", "KOL-FIN-001"],
        "expected_keywords": ["laptop", "headset", "VPN", "security token"],
        "ground_truth_snippet": "laptop, headset, security token for MFA",
    },
    # ── SECURITY & IT (4 questions) ─────────────────────────────────────
    {
        "id": "EVAL-010",
        "question": "What are the password requirements at Kolrose?",
        "category": "security",
        "difficulty": "easy",
        "expected_documents": ["KOL-IT-001"],
        "expected_keywords": ["12 characters", "uppercase", "lowercase", "numbers", "special", "90 days"],
        "ground_truth_snippet": "minimum 12 characters, mixed case, numbers, special characters, 90-day expiry",
    },
    {
        "id": "EVAL-011",
        "question": "How do I report a security incident?",
        "category": "security",
        "difficulty": "medium",
        "expected_documents": ["KOL-IT-001"],
        "expected_keywords": ["immediately", "IT Security Hotline", "0800-KOL-ITSEC", "incident report"],
        "ground_truth_snippet": "call IT Security Hotline immediately, then submit Incident Report Form within 24 hours",
    },
    {
        "id": "EVAL-012",
        "question": "Is multi-factor authentication mandatory for all systems?",
        "category": "security",
        "difficulty": "easy",
        "expected_documents": ["KOL-IT-001"],
        "expected_keywords": ["MFA", "mandatory", "email", "VPN", "financial"],
        "ground_truth_snippet": "MFA is mandatory for email, VPN, financial systems, HR systems",
    },
    {
        "id": "EVAL-013",
        "question": "What is the data classification for client financial information?",
        "category": "security",
        "difficulty": "hard",
        "expected_documents": ["KOL-IT-001"],
        "expected_keywords": ["confidential", "restricted", "encrypted", "classification"],
        "ground_truth_snippet": "confidential data requires encrypted storage and transmission",
    },
    # ── EXPENSES & FINANCE (4 questions) ────────────────────────────────
    {
        "id": "EVAL-014",
        "question": "What is the maximum hotel rate for business travel in Abuja?",
        "category": "expenses",
        "difficulty": "medium",
        "expected_documents": ["KOL-FIN-001", "KOL-ADMIN-001"],
        "expected_keywords": ["₦35,000", "hotel", "Abuja", "per night"],
        "ground_truth_snippet": "maximum hotel rate ₦35,000 per night in Abuja",
    },
    {
        "id": "EVAL-015",
        "question": "What expenses are NOT reimbursable at Kolrose?",
        "category": "expenses",
        "difficulty": "medium",
        "expected_documents": ["KOL-FIN-001"],
        "expected_keywords": ["fines", "penalties", "personal entertainment", "alcohol"],
        "ground_truth_snippet": "fines, penalties, personal entertainment, lost items are not reimbursable",
    },
    {
        "id": "EVAL-016",
        "question": "How much can I claim for a home office chair?",
        "category": "expenses",
        "difficulty": "medium",
        "expected_documents": ["KOL-FIN-001"],
        "expected_keywords": ["₦100,000", "ergonomic", "chair", "once every 3 years"],
        "ground_truth_snippet": "ergonomic chair up to ₦100,000 once every 3 years",
    },
    {
        "id": "EVAL-017",
        "question": "What is the procurement threshold for formal tender?",
        "category": "expenses",
        "difficulty": "hard",
        "expected_documents": ["KOL-FIN-002"],
        "expected_keywords": ["₦2,000,001", "₦10,000,000", "formal tender", "3-5 bidders"],
        "ground_truth_snippet": "₦2,000,001 to ₦10,000,000 requires formal tender with 3-5 bidders",
    },
    # ── CODE OF CONDUCT (3 questions) ───────────────────────────────────
    {
        "id": "EVAL-018",
        "question": "What is the Kolrose policy on accepting gifts from vendors?",
        "category": "conduct",
        "difficulty": "medium",
        "expected_documents": ["KOL-HR-003"],
        "expected_keywords": ["₦10,000", "nominal", "declare", "cash", "prohibited"],
        "ground_truth_snippet": "gifts below ₦10,000 acceptable, cash gifts strictly prohibited",
    },
    {
        "id": "EVAL-019",
        "question": "How does Kolrose handle conflicts of interest?",
        "category": "conduct",
        "difficulty": "medium",
        "expected_documents": ["KOL-HR-003"],
        "expected_keywords": ["outside employment", "compete", "declare", "compliance"],
        "ground_truth_snippet": "must not engage in outside employment that competes with Kolrose",
    },
    {
        "id": "EVAL-020",
        "question": "What anti-corruption laws apply to Kolrose operations?",
        "category": "conduct",
        "difficulty": "hard",
        "expected_documents": ["KOL-HR-003"],
        "expected_keywords": ["EFCC", "Corrupt Practices Act", "zero tolerance", "compliance"],
        "ground_truth_snippet": "zero-tolerance policy complying with Corrupt Practices Act 2000 and EFCC regulations",
    },
    # ── PERFORMANCE & TRAINING (3 questions) ────────────────────────────
    {
        "id": "EVAL-021",
        "question": "How often are performance reviews conducted at Kolrose?",
        "category": "performance",
        "difficulty": "easy",
        "expected_documents": ["KOL-HR-006"],
        "expected_keywords": ["quarterly", "mid-year", "annual", "January to December"],
        "ground_truth_snippet": "quarterly check-ins, mid-year formal review, year-end appraisal",
    },
    {
        "id": "EVAL-022",
        "question": "What happens if I'm placed on a Performance Improvement Plan?",
        "category": "performance",
        "difficulty": "hard",
        "expected_documents": ["KOL-HR-006"],
        "expected_keywords": ["60-90 days", "support plan", "weekly check-ins", "termination"],
        "ground_truth_snippet": "60-90 day PIP with weekly reviews, support plan, possible termination if unsuccessful",
    },
    {
        "id": "EVAL-023",
        "question": "What training budget is available for technical certifications?",
        "category": "training",
        "difficulty": "medium",
        "expected_documents": ["KOL-HR-007"],
        "expected_keywords": ["₦300,000", "certification", "technical", "annually"],
        "ground_truth_snippet": "₦300,000 per technical employee annually for certifications",
    },
    # ── COMPANY INFO & NIGERIAN CONTEXT (2 questions) ───────────────────
    {
        "id": "EVAL-024",
        "question": "Where is Kolrose Limited headquarters located?",
        "category": "company_info",
        "difficulty": "easy",
        "expected_documents": ["KOL-HR-001"],
        "expected_keywords": ["Suite 10", "Bataiya Plaza", "Area 2 Garki", "Abuja", "FCT"],
        "ground_truth_snippet": "Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT",
    },
    {
        "id": "EVAL-025",
        "question": "What Nigerian data protection regulations does Kolrose comply with?",
        "category": "company_info",
        "difficulty": "hard",
        "expected_documents": ["KOL-HR-003", "KOL-IT-001"],
        "expected_keywords": ["NDPR", "Nigeria Data Protection Regulation", "NITDA", "compliance"],
        "ground_truth_snippet": "complies with Nigeria Data Protection Regulation (NDPR) 2019 and NITDA guidelines",
    },
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _extract_key_terms(text: str) -> List[str]:
    """Extract meaningful key terms from text."""
    text = re.sub(r'[^\w\s₦]', ' ', text.lower())
    words = text.split()
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'this', 'that', 'these', 'those',
        'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you', 'your',
        'he', 'she', 'his', 'her', 'which', 'who', 'whom', 'what', 'when',
        'where', 'how', 'and', 'but', 'or', 'not', 'no', 'if', 'then', 'than',
        'also', 'very', 'just', 'about', 'into', 'over', 'after', 'such',
        'only', 'other', 'new', 'most', 'between',
    }
    return [w for w in words if len(w) > 3 and w not in stop_words]


# ============================================================================
# 1. GROUNDEDNESS EVALUATOR
# ============================================================================

class GroundednessEvaluator:
    """
    Evaluates whether RAG answers are factually grounded in retrieved documents.
    Groundedness = % of answers whose content is fully supported by retrieved evidence.
    """
    
    def __init__(self, vectorstore, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
    
    def decompose_into_claims(self, text: str) -> List[str]:
        """Decompose answer into individual factual claims."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        skip_phrases = {
            'i hope this helps', 'let me know', 'is there anything else',
            'please let me know', 'feel free to ask',
        }
        claims = []
        for s in sentences:
            s = s.strip()
            if not s or s.lower() in skip_phrases:
                continue
            if re.match(r'^\[?KOL-\w+-\d+.*\]?\s*$', s):
                continue
            if len(s.split()) < 4:
                continue
            claims.append(s)
        return claims
    
    def calculate_claim_support(self, claim: str, source_docs: List[Dict]) -> Tuple[bool, float, Optional[str]]:
        """Check if a single claim is supported by source documents."""
        claim_terms = set(_extract_key_terms(claim))
        if not claim_terms:
            return False, 0.0, None
        
        best_score, best_doc = 0.0, None
        for doc in source_docs:
            content = doc.get('content', doc.get('page_content', ''))
            content_terms = set(_extract_key_terms(content))
            if not content_terms:
                continue
            
            overlap = len(claim_terms & content_terms) / len(claim_terms)
            
            claim_nums = set(re.findall(r'\d+', claim))
            content_nums = set(re.findall(r'\d+', content))
            num_bonus = min(len(claim_nums & content_nums) * 0.1, 0.3) if claim_nums else 0
            
            claim_naira = set(re.findall(r'₦[\d,]+', claim))
            content_naira = set(re.findall(r'₦[\d,]+', content))
            naira_bonus = min(len(claim_naira & content_naira) * 0.15, 0.3) if claim_naira else 0
            
            score = overlap + num_bonus + naira_bonus
            if score > best_score:
                best_score = score
                best_doc = doc.get('metadata', {}).get('document_id', 'Unknown')
        
        return best_score >= 0.3, min(best_score, 1.0), best_doc
    
    def evaluate_single_answer(self, question: str, answer: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate groundedness for one answer."""
        claims = self.decompose_into_claims(answer)
        if not claims:
            return {'groundedness_score': 0.0, 'total_claims': 0, 'supported_claims': 0,
                    'unsupported_claims': 0, 'claim_details': [], 'fully_grounded': False}
        
        claim_results, supported = [], 0
        for claim in claims:
            is_sup, score, src = self.calculate_claim_support(claim, retrieved_docs)
            claim_results.append({'claim': claim, 'supported': is_sup, 'support_score': round(score, 3), 'source_document': src})
            if is_sup:
                supported += 1
        
        groundedness = supported / len(claims)
        return {'groundedness_score': round(groundedness, 3), 'total_claims': len(claims),
                'supported_claims': supported, 'unsupported_claims': len(claims) - supported,
                'claim_details': claim_results, 'fully_grounded': groundedness == 1.0}
    
    def evaluate_all(self, rag_system, questions: List[Dict], verbose: bool = True) -> Dict:
        """Evaluate groundedness across all test questions."""
        results, start_time = [], time.time()
        if verbose:
            print(f"\n{'='*70}\n📊 GROUNDEDNESS EVALUATION - {COMPANY_INFO['name']}\n{'='*70}\n")
        
        for i, tc in enumerate(questions):
            if verbose:
                print(f"[{i+1}/{len(questions)}] {tc['id']}: {tc['question'][:80]}...")
            
            q_start = time.time()
            rag_result = rag_system.query(tc['question'], enable_guardrails=False)
            q_time = time.time() - q_start
            
            retrieved_docs = [{'content': s.get('snippet', ''), 'metadata': s} for s in rag_result.sources]
            eval_result = self.evaluate_single_answer(tc['question'], rag_result.answer, retrieved_docs)
            
            kw_found = [kw for kw in tc.get('expected_keywords', []) if kw.lower() in rag_result.answer.lower()]
            kw_missing = [kw for kw in tc.get('expected_keywords', []) if kw not in kw_found]
            kw_score = len(kw_found) / len(tc['expected_keywords']) if tc.get('expected_keywords') else 1.0
            
            results.append({
                'id': tc['id'], 'question': tc['question'], 'category': tc['category'],
                'difficulty': tc['difficulty'], 'answer': rag_result.answer[:500],
                'groundedness': eval_result['groundedness_score'], 'fully_grounded': eval_result['fully_grounded'],
                'total_claims': eval_result['total_claims'], 'supported_claims': eval_result['supported_claims'],
                'unsupported_claims': eval_result['unsupported_claims'], 'keyword_score': round(kw_score, 3),
                'keywords_found': kw_found, 'keywords_missing': kw_missing,
                'citations': rag_result.citations, 'num_citations': len(rag_result.citations),
                'latency_ms': round(q_time * 1000), 'refused': rag_result.refused,
            })
            
            if verbose:
                status = "✅" if eval_result['fully_grounded'] else "⚠️"
                print(f"     {status} Groundedness: {eval_result['groundedness_score']:.0%} "
                      f"({eval_result['supported_claims']}/{eval_result['total_claims']} claims) | "
                      f"Keywords: {kw_score:.0%} | Latency: {q_time*1000:.0f}ms")
        
        scores = [r['groundedness'] for r in results]
        kw_scores = [r['keyword_score'] for r in results]
        lats = [r['latency_ms'] for r in results]
        fully_g = sum(1 for r in results if r['fully_grounded'])
        
        cat_stats = defaultdict(lambda: {'scores': [], 'count': 0})
        for r in results:
            cat_stats[r['category']]['scores'].append(r['groundedness'])
            cat_stats[r['category']]['count'] += 1
        
        aggregate = {
            'total_questions': len(results),
            'avg_groundedness': round(np.mean(scores), 3) if scores else 0,
            'median_groundedness': round(np.median(scores), 3) if scores else 0,
            'fully_grounded_count': fully_g,
            'fully_grounded_pct': round(fully_g / len(results) * 100, 1) if results else 0,
            'avg_keyword_score': round(np.mean(kw_scores), 3) if kw_scores else 0,
            'avg_latency_ms': round(np.mean(lats), 0) if lats else 0,
            'p95_latency_ms': round(np.percentile(lats, 95), 0) if lats else 0,
            'category_breakdown': {
                cat: {'count': s['count'], 'avg_groundedness': round(np.mean(s['scores']), 3)}
                for cat, s in cat_stats.items()
            },
        }
        
        if verbose:
            print(f"\n{'='*70}\n📊 GROUNDEDNESS SUMMARY\n{'='*70}")
            print(f"   Avg: {aggregate['avg_groundedness']:.1%} | Fully Grounded: {fully_g}/{len(results)} ({aggregate['fully_grounded_pct']}%)")
            print(f"   Avg Latency: {aggregate['avg_latency_ms']:.0f}ms | P95: {aggregate['p95_latency_ms']:.0f}ms")
            for cat, s in sorted(aggregate['category_breakdown'].items()):
                bar = "█" * int(s['avg_groundedness'] * 20)
                print(f"   {cat:<20} {s['avg_groundedness']:.1%} {bar} ({s['count']}q)")
            print(f"{'='*70}")
        
        return {'results': results, 'aggregate': aggregate, 'evaluation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()}


# ============================================================================
# 2. CITATION ACCURACY EVALUATOR
# ============================================================================

class CitationAccuracyEvaluator:
    """
    Evaluates whether citations correctly point to passages that support the stated information.
    A citation is ACCURATE if: doc exists, section exists, and content supports the claim.
    """
    
    VALID_DOC_IDS = {
        'KOL-HR-001', 'KOL-HR-002', 'KOL-HR-003', 'KOL-HR-005', 'KOL-HR-006',
        'KOL-HR-007', 'KOL-HR-008', 'KOL-IT-001', 'KOL-FIN-001', 'KOL-FIN-002',
        'KOL-ADMIN-001', 'KOL-ADMIN-002',
    }
    
    def __init__(self, vectorstore, embeddings, corpus_documents: Dict[str, str] = None):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.corpus_documents = corpus_documents or {}
    
    def load_corpus_from_vectorstore(self):
        """Load full document texts from vector store."""
        if self.corpus_documents:
            return
        try:
            results = self.vectorstore._collection.get(include=['documents', 'metadatas'])
            for i, meta in enumerate(results['metadatas']):
                doc_id = meta.get('document_id', '')
                if doc_id and doc_id != 'UNKNOWN':
                    self.corpus_documents.setdefault(doc_id, []).append(results['documents'][i])
            for doc_id in self.corpus_documents:
                self.corpus_documents[doc_id] = '\n'.join(self.corpus_documents[doc_id])
        except Exception as e:
            print(f"⚠️ Could not load corpus: {e}")
    
    def extract_citations_from_answer(self, answer: str) -> List[Dict]:
        """Extract all citations with surrounding context."""
        citations = []
        for match in re.finditer(r'\[(KOL-\w+-\d+)[,\s]*[§]?(?:Section\s*)?([\d.]+)?\]', answer):
            doc_id, section = match.group(1), match.group(2) if match.group(2) else None
            before = answer[max(0, match.start()-200):match.start()]
            claim = before.split('.')[-1].strip() if before else ''
            citations.append({
                'doc_id': doc_id, 'section': section, 'full_citation': match.group(0),
                'context': answer[max(0, match.start()-300):match.end()+50].strip(),
                'claim': claim, 'position': match.start(), 'has_section': section is not None, 'pattern': 'full',
            })
        
        covered = {c['position'] for c in citations}
        for match in re.finditer(r'\[(KOL-\w+-\d+)\]', answer):
            if match.start() in covered:
                continue
            citations.append({
                'doc_id': match.group(1), 'section': None, 'full_citation': match.group(0),
                'context': answer[max(0, match.start()-200):match.end()+50].strip(),
                'claim': '', 'position': match.start(), 'has_section': False, 'pattern': 'doc_only',
            })
        return citations
    
    def verify_document_exists(self, doc_id: str) -> bool:
        return doc_id in self.VALID_DOC_IDS
    
    def verify_section_exists(self, doc_id: str, section: str) -> bool:
        if doc_id not in self.corpus_documents:
            return False
        content = self.corpus_documents[doc_id]
        for pat in [rf'#+\s+{re.escape(section)}\s', rf'#+\s+Section\s+{re.escape(section)}',
                     rf'\*\*Section\s+{re.escape(section)}\*\*']:
            if re.search(pat, content, re.IGNORECASE):
                return True
        return False
    
    def verify_content_matches(self, doc_id: str, claim: str, section: str = None) -> Tuple[bool, float, str]:
        if doc_id not in self.corpus_documents:
            return False, 0.0, ""
        content = self.corpus_documents[doc_id]
        search_content = content
        if section:
            sec_match = re.search(rf'(#+\s+(?:Section\s+)?{re.escape(section)}.*?)(?=#+\s+(?:Section\s+)?\d|$)',
                                  content, re.IGNORECASE | re.DOTALL)
            if sec_match:
                search_content = sec_match.group(0)
        
        claim_terms = _extract_key_terms(claim)
        if not claim_terms:
            return False, 0.0, ""
        
        search_lower = search_content.lower()
        matching = [t for t in claim_terms if t.lower() in search_lower]
        if not matching:
            return False, 0.0, ""
        
        claim_nums = set(re.findall(r'\d+', claim))
        content_nums = set(re.findall(r'\d+', search_content))
        num_ratio = len(claim_nums & content_nums) / len(claim_nums) if claim_nums else 0
        
        confidence = (len(matching) / len(claim_terms) * 0.6) + (num_ratio * 0.4)
        evidence = ""
        for term in matching[:3]:
            pos = search_lower.find(term.lower())
            if pos >= 0:
                evidence = search_content[max(0, pos-50):min(len(search_content), pos+100)].strip()
                break
        
        return confidence >= 0.4, confidence, evidence
    
    def evaluate_single_citation(self, citation: Dict, full_answer: str = "") -> Dict:
        doc_id, section, claim = citation['doc_id'], citation.get('section'), citation.get('claim', '')
        doc_exists = self.verify_document_exists(doc_id)
        section_exists = self.verify_section_exists(doc_id, section) if section else None
        content_matches, confidence, evidence = (False, 0.0, "")
        if doc_exists:
            content_matches, confidence, evidence = self.verify_content_matches(doc_id, claim, section)
        
        if not doc_exists:
            accuracy, is_accurate, issue = "invalid_document", False, f"Document {doc_id} does not exist"
        elif section and not section_exists:
            accuracy, is_accurate, issue = "invalid_section", False, f"Section {section} not found in {doc_id}"
        elif not content_matches:
            accuracy, is_accurate, issue = "unsupported_claim", False, "Content does not support claim"
        else:
            accuracy, is_accurate, issue = "accurate", True, None
        
        return {'citation': citation['full_citation'], 'doc_id': doc_id, 'section': section,
                'claim': claim[:150], 'is_accurate': is_accurate, 'accuracy': accuracy, 'issue': issue,
                'confidence': round(confidence, 3), 'evidence_snippet': evidence[:200],
                'checks': {'document_exists': doc_exists, 'section_exists': section_exists, 'content_matches': content_matches}}
    
    def evaluate_single_answer(self, answer: str, source_docs: List[Dict] = None) -> Dict:
        citations = self.extract_citations_from_answer(answer)
        if not citations:
            return {'citation_accuracy': 0.0, 'total_citations': 0, 'accurate_citations': 0,
                    'inaccurate_citations': 0, 'citation_details': [], 'all_accurate': False, 'has_citations': False}
        
        results, accurate = [], 0
        for c in citations:
            r = self.evaluate_single_citation(c, answer)
            results.append(r)
            if r['is_accurate']:
                accurate += 1
        
        accuracy = accurate / len(citations)
        return {'citation_accuracy': round(accuracy, 3), 'total_citations': len(citations),
                'accurate_citations': accurate, 'inaccurate_citations': len(citations) - accurate,
                'citation_details': results, 'all_accurate': accuracy == 1.0, 'has_citations': True}
    
    def evaluate_all(self, rag_system, questions: List[Dict], verbose: bool = True) -> Dict:
        self.load_corpus_from_vectorstore()
        results = []
        if verbose:
            print(f"\n{'='*70}\n📝 CITATION ACCURACY EVALUATION - {COMPANY_INFO['name']}\n{'='*70}\n")
        
        for i, tc in enumerate(questions):
            if verbose:
                print(f"[{i+1}/{len(questions)}] {tc['id']}: {tc['question'][:80]}...")
            
            rag_result = rag_system.query(tc['question'], enable_guardrails=False)
            if rag_result.refused:
                results.append({'id': tc['id'], 'citation_accuracy': None, 'total_citations': 0, 'refused': True})
                if verbose:
                    print("     ⏭️ Skipped (refused)")
                continue
            
            eval_result = self.evaluate_single_answer(rag_result.answer)
            results.append({
                'id': tc['id'], 'question': tc['question'], 'category': tc['category'],
                'answer': rag_result.answer[:500], 'citation_accuracy': eval_result['citation_accuracy'],
                'total_citations': eval_result['total_citations'], 'accurate_citations': eval_result['accurate_citations'],
                'inaccurate_citations': eval_result['inaccurate_citations'],
                'all_accurate': eval_result['all_accurate'], 'has_citations': eval_result['has_citations'],
                'citation_details': eval_result['citation_details'], 'refused': False,
            })
            
            if verbose and eval_result['has_citations']:
                status = "✅" if eval_result['all_accurate'] else "⚠️"
                print(f"     {status} Citations: {eval_result['accurate_citations']}/{eval_result['total_citations']} accurate")
                for d in eval_result['citation_details']:
                    if not d['is_accurate']:
                        print(f"        ❌ {d['citation']}: {d['issue']}")
        
        valid = [r for r in results if not r.get('refused') and r['has_citations']]
        if valid:
            scores = [r['citation_accuracy'] for r in valid]
            fully_acc = sum(1 for r in valid if r['all_accurate'])
            aggregate = {
                'total_questions': len(results), 'questions_with_citations': len(valid),
                'avg_citation_accuracy': round(np.mean(scores), 3),
                'fully_accurate_count': fully_acc,
                'fully_accurate_pct': round(fully_acc / len(valid) * 100, 1),
                'total_citations_found': sum(r['total_citations'] for r in valid),
                'total_citations_accurate': sum(r['accurate_citations'] for r in valid),
            }
        else:
            aggregate = {'total_questions': len(results), 'questions_with_citations': 0, 'avg_citation_accuracy': 0.0}
        
        if verbose:
            print(f"\n{'='*70}\n📊 CITATION ACCURACY SUMMARY\n{'='*70}")
            print(f"   Questions with citations: {aggregate.get('questions_with_citations', 0)}/{aggregate['total_questions']}")
            print(f"   Avg Accuracy: {aggregate.get('avg_citation_accuracy', 0):.1%}")
            print(f"   Fully Accurate: {aggregate.get('fully_accurate_count', 0)} questions")
            print(f"{'='*70}")
        
        return {'results': results, 'aggregate': aggregate, 'timestamp': datetime.now().isoformat()}


# ============================================================================
# 3. GOLD ANSWER MATCHER
# ============================================================================

class GoldAnswerMatcher:
    """Evaluates exact, partial, and semantic match against gold answers."""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def exact_match(self, answer: str, gold: str) -> bool:
        def normalize(t): return re.sub(r'[^\w\s]', '', re.sub(r'\s+', ' ', t.lower().strip()))
        return normalize(answer) == normalize(gold)
    
    def partial_match_score(self, answer: str, gold: str) -> float:
        gold_terms = _extract_key_terms(gold)
        if not gold_terms:
            return 0.0
        answer_lower = answer.lower()
        return sum(1 for t in gold_terms if t.lower() in answer_lower) / len(gold_terms)
    
    def semantic_similarity(self, answer: str, gold: str) -> float:
        try:
            a_emb = self.embeddings.embed_query(answer)
            g_emb = self.embeddings.embed_query(gold)
            return float(np.dot(a_emb, g_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(g_emb)))
        except Exception:
            return 0.0
    
    def evaluate_all(self, rag_system, questions: List[Dict], verbose: bool = True) -> Dict:
        results = []
        if verbose:
            print(f"\n{'='*70}\n🎯 GOLD ANSWER MATCHING - {COMPANY_INFO['name']}\n{'='*70}\n")
        
        for i, tc in enumerate(questions):
            gold = tc.get('ground_truth_snippet', '')
            if not gold:
                continue
            if verbose:
                print(f"[{i+1}/{len(questions)}] {tc['id']}: {tc['question'][:60]}...")
            
            rag_result = rag_system.query(tc['question'], enable_guardrails=False)
            if rag_result.refused:
                results.append({'id': tc['id'], 'exact_match': False, 'partial_match_score': 0.0,
                               'semantic_similarity': 0.0, 'refused': True})
                continue
            
            exact = self.exact_match(rag_result.answer, gold)
            partial = self.partial_match_score(rag_result.answer, gold)
            semantic = self.semantic_similarity(rag_result.answer, gold)
            
            results.append({
                'id': tc['id'], 'question': tc['question'], 'category': tc['category'],
                'answer': rag_result.answer[:300], 'gold_answer': gold,
                'exact_match': exact, 'partial_match_score': round(partial, 3),
                'semantic_similarity': round(semantic, 3), 'contains_essential': partial >= 0.7,
                'refused': False,
            })
            
            if verbose:
                status = "✅" if partial >= 0.7 else "⚠️" if partial >= 0.4 else "❌"
                print(f"     {status} Exact: {'Yes' if exact else 'No'} | Partial: {partial:.1%} | Semantic: {semantic:.2f}")
        
        valid = [r for r in results if not r.get('refused')]
        if valid:
            aggregate = {
                'total_questions': len(results),
                'exact_match_pct': round(sum(1 for r in valid if r['exact_match']) / len(valid) * 100, 1),
                'partial_match_pct': round(sum(1 for r in valid if r['contains_essential']) / len(valid) * 100, 1),
                'avg_partial_score': round(np.mean([r['partial_match_score'] for r in valid]), 3),
                'avg_semantic_similarity': round(np.mean([r['semantic_similarity'] for r in valid]), 3),
            }
        else:
            aggregate = {'total_questions': len(results), 'error': 'No valid results'}
        
        if verbose:
            print(f"\n{'='*70}\n📊 MATCHING SUMMARY\n{'='*70}")
            print(f"   Exact Match: {aggregate.get('exact_match_pct', 0)}%")
            print(f"   Partial Match (≥70%): {aggregate.get('partial_match_pct', 0)}%")
            print(f"   Avg Semantic Similarity: {aggregate.get('avg_semantic_similarity', 0):.2f}")
            print(f"{'='*70}")
        
        return {'results': results, 'aggregate': aggregate}


# ============================================================================
# 4. LATENCY BENCHMARK
# ============================================================================

class LatencyBenchmark:
    """Measures P50/P95/P99 latency from request to answer."""
    
    def __init__(self):
        self.warm_up_complete = False
    
    def run_single_query(self, rag_system, question: str) -> Dict:
        start = time.time()
        try:
            result = rag_system.query(question, enable_guardrails=False)
            return {'total_ms': round((time.time() - start) * 1000), 'success': True}
        except Exception as e:
            return {'total_ms': round((time.time() - start) * 1000), 'error': str(e), 'success': False}
    
    def warm_up(self, rag_system):
        for q in ["What is the leave policy?", "Where is the office?", "What are working hours?"]:
            self.run_single_query(rag_system, q)
        self.warm_up_complete = True
    
    def benchmark(self, rag_system, questions: List[str], warm_up_first: bool = True,
                  iterations: int = 1, verbose: bool = True) -> Dict:
        if warm_up_first and not self.warm_up_complete:
            print("🔥 Warming up...")
            self.warm_up(rag_system)
            print("   ✅ Ready\n")
        
        all_lats, query_results = [], []
        if verbose:
            print(f"\n{'='*70}\n⏱️ LATENCY BENCHMARK - {len(questions)} questions\n{'='*70}\n")
        
        for i, q in enumerate(questions):
            q_lats = []
            for _ in range(iterations):
                r = self.run_single_query(rag_system, q)
                if r['success']:
                    q_lats.append(r['total_ms'])
                    all_lats.append(r['total_ms'])
                time.sleep(0.1)
            
            if q_lats:
                query_results.append({
                    'question': q[:100], 'avg_ms': round(np.mean(q_lats), 0),
                    'p50_ms': round(np.percentile(q_lats, 50), 0),
                    'p95_ms': round(np.percentile(q_lats, 95), 0),
                })
                if verbose:
                    print(f"[{i+1}/{len(questions)}] {q[:60]}...")
                    print(f"     P50: {np.percentile(q_lats, 50):.0f}ms | P95: {np.percentile(q_lats, 95):.0f}ms")
        
        if all_lats:
            aggregate = {
                'total_measurements': len(all_lats),
                'p50_ms': round(np.percentile(all_lats, 50), 0),
                'p95_ms': round(np.percentile(all_lats, 95), 0),
                'p99_ms': round(np.percentile(all_lats, 99), 0),
                'mean_ms': round(np.mean(all_lats), 0),
                'min_ms': round(np.min(all_lats), 0), 'max_ms': round(np.max(all_lats), 0),
            }
        else:
            aggregate = {'error': 'No successful measurements'}
        
        if verbose:
            print(f"\n{'='*70}\n📊 LATENCY SUMMARY\n{'='*70}")
            print(f"   P50: {aggregate.get('p50_ms', 0)}ms | P95: {aggregate.get('p95_ms', 0)}ms | P99: {aggregate.get('p99_ms', 0)}ms")
            print(f"{'='*70}")
        
        return {'results': query_results, 'aggregate': aggregate, 'timestamp': datetime.now().isoformat()}


# ============================================================================
# 5. ABLATION STUDIES
# ============================================================================

class AblationStudy:
    """Compares retrieval k, re-ranking, and retrieval methods."""
    
    def __init__(self, vectorstore, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
    
    def compare_retrieval_k(self, rag_system, questions: List[str], k_values: List[int] = [3, 5, 10, 20],
                            verbose: bool = True) -> Dict:
        if verbose:
            print(f"\n🔬 ABLATION: Retrieval K = {k_values}")
        results = {}
        for k in k_values:
            lats = []
            for q in questions[:5]:
                r = rag_system.query(q, k_retrieve=k*2, k_final=k, enable_guardrails=False)
                lats.append(r.metrics.get('total_ms', 0))
            results[f'k={k}'] = {'avg_latency_ms': round(np.mean(lats), 0) if lats else 0}
            if verbose:
                print(f"   k={k}: {results[f'k={k}']['avg_latency_ms']}ms avg")
        return results
    
    def compare_rerank(self, rag_system, questions: List[str], verbose: bool = True) -> Dict:
        if verbose:
            print(f"\n🔬 ABLATION: Re-ranking On vs Off")
        results = {}
        for enabled in [True, False]:
            label = "with_rerank" if enabled else "without_rerank"
            lats = []
            for q in questions[:5]:
                r = rag_system.query(q, enable_rerank=enabled, enable_guardrails=False)
                lats.append(r.metrics.get('total_ms', 0))
            results[label] = {'avg_latency_ms': round(np.mean(lats), 0) if lats else 0}
            if verbose:
                print(f"   {label}: {results[label]['avg_latency_ms']}ms avg")
        return results
    
    def compare_retrieval_method(self, rag_system, questions: List[str], verbose: bool = True) -> Dict:
        from app.rag_system import RetrievalMethod
        if verbose:
            print(f"\n🔬 ABLATION: Retrieval Method")
        results = {}
        for method in [RetrievalMethod.SIMILARITY, RetrievalMethod.MMR]:
            label = method.value
            lats, srcs = [], []
            for q in questions[:5]:
                r = rag_system.query(q, retrieval_method=method, enable_guardrails=False)
                lats.append(r.metrics.get('total_ms', 0))
                srcs.append(r.metrics.get('num_sources', 0))
            results[label] = {'avg_latency_ms': round(np.mean(lats), 0) if lats else 0,
                              'avg_sources': round(np.mean(srcs), 1) if srcs else 0}
            if verbose:
                print(f"   {label}: {results[label]['avg_latency_ms']}ms, {results[label]['avg_sources']} sources")
        return results
    
    def run_all_ablations(self, rag_system, questions: List[str] = None, verbose: bool = True) -> Dict:
        if questions is None:
            questions = ["What is the leave policy?", "How do I request remote work?",
                        "What are the password requirements?", "How are expenses reimbursed?",
                        "Where is the office located?"]
        return {
            'retrieval_k': self.compare_retrieval_k(rag_system, questions, verbose=verbose),
            'rerank': self.compare_rerank(rag_system, questions, verbose=verbose),
            'retrieval_method': self.compare_retrieval_method(rag_system, questions, verbose=verbose),
            'timestamp': datetime.now().isoformat(),
        }


# ============================================================================
# COMPLETE EVALUATION RUNNER
# ============================================================================

class CompleteEvaluationRunner:
    """Runs all evaluations and generates comprehensive report."""
    
    def __init__(self, vectorstore, embeddings):
        self.groundedness_eval = GroundednessEvaluator(vectorstore, embeddings)
        self.citation_eval = CitationAccuracyEvaluator(vectorstore, embeddings)
        self.gold_matcher = GoldAnswerMatcher(embeddings)
        self.latency_bench = LatencyBenchmark()
        self.ablation_study = AblationStudy(vectorstore, embeddings)
    
    def run_all(self, rag_system, eval_questions: List[Dict] = None,
                latency_questions: List[str] = None, verbose: bool = True,
                output_dir: str = "./evaluation_results") -> Dict:
        if eval_questions is None:
            eval_questions = EVALUATION_QUESTIONS
        if latency_questions is None:
            latency_questions = [q['question'] for q in eval_questions[:15]]
        
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*70}\n📊 COMPLETE EVALUATION SUITE - {COMPANY_INFO['name']}\n{'='*70}")
        
        full_report = {}
        
        for phase, name, evaluator, args in [
            (1, "GROUNDEDNESS", self.groundedness_eval, (rag_system, eval_questions)),
            (2, "CITATION ACCURACY", self.citation_eval, (rag_system, eval_questions)),
            (3, "GOLD ANSWER MATCHING", self.gold_matcher, (rag_system, eval_questions)),
        ]:
            if verbose:
                print(f"\n{'─'*70}\n📋 PHASE {phase}/5: {name}\n{'─'*70}")
            full_report[name.lower().replace(' ', '_')] = evaluator.evaluate_all(*args, verbose=verbose)
        
        if verbose:
            print(f"\n{'─'*70}\n📋 PHASE 4/5: LATENCY BENCHMARK\n{'─'*70}")
        full_report['latency'] = self.latency_bench.benchmark(rag_system, latency_questions, verbose=verbose)
        
        if verbose:
            print(f"\n{'─'*70}\n📋 PHASE 5/5: ABLATION STUDIES\n{'─'*70}")
        full_report['ablations'] = self.ablation_study.run_all_ablations(rag_system, latency_questions[:5], verbose=verbose)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_time_s': round(time.time() - start_time, 1),
            'groundedness': {'avg_score': full_report['groundedness']['aggregate']['avg_groundedness']},
            'citation_accuracy': {'avg_score': full_report['citation_accuracy']['aggregate'].get('avg_citation_accuracy', 0)},
            'gold_matching': {'partial_match_pct': full_report['gold_answer_matching']['aggregate'].get('partial_match_pct', 0)},
            'latency': {'p50_ms': full_report['latency']['aggregate'].get('p50_ms', 0),
                       'p95_ms': full_report['latency']['aggregate'].get('p95_ms', 0)},
        }
        full_report['summary'] = summary
        
        for name, report in full_report.items():
            if name != 'summary':
                with open(os.path.join(output_dir, f"{name}.json"), 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        with open(os.path.join(output_dir, "summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        
        if verbose:
            print(f"\n{'='*70}\n✅ EVALUATION COMPLETE ({summary['total_time_s']:.0f}s)\n{'='*70}")
            print(f"   Groundedness: {summary['groundedness']['avg_score']:.1%}")
            print(f"   Citation Accuracy: {summary['citation_accuracy']['avg_score']:.1%}")
            print(f"   P50 Latency: {summary['latency']['p50_ms']}ms | P95: {summary['latency']['p95_ms']}ms")
            print(f"   Reports: {output_dir}/\n{'='*70}")
        
        return full_report


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_evaluation(rag_system=None, vectorstore=None, questions: List[Dict] = None,
                   output_path: str = None, verbose: bool = True) -> Dict:
    """Run groundedness evaluation."""
    if questions is None:
        questions = EVALUATION_QUESTIONS
    if vectorstore is None:
        vectorstore = load_vectorstore()
    if vectorstore is None:
        raise ValueError("Vector store not found. Run ingestion first.")
    if rag_system is None:
        from app.rag_system import KolroseRAG
        rag_system = KolroseRAG(vectorstore)
    
    evaluator = GroundednessEvaluator(vectorstore, load_embeddings())
    report = evaluator.evaluate_all(rag_system, questions, verbose=verbose)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return report


def quick_evaluation() -> bool:
    """Minimal evaluation for CI/CD. Returns True if passing threshold."""
    try:
        report = run_evaluation(questions=EVALUATION_QUESTIONS[:10], verbose=False)
        score = report['aggregate']['avg_groundedness']
        passed = score >= 0.60
        print(f"Quick Eval: {score:.1%} groundedness → {'✅ PASSED' if passed else '❌ FAILED'}")
        return passed
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return False


def generate_evaluation_report(report: Dict) -> str:
    """Generate markdown evaluation report."""
    agg = report['aggregate']
    lines = [
        f"# 📊 Kolrose Limited - RAG Evaluation Report",
        f"**Generated:** {report['timestamp']} | **Company:** {COMPANY_INFO['name']}",
        f"---",
        f"## 📈 Aggregate Metrics",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Questions | {agg['total_questions']} |",
        f"| **Avg Groundedness** | **{agg['avg_groundedness']:.1%}** |",
        f"| Fully Grounded | {agg['fully_grounded_count']}/{agg['total_questions']} ({agg['fully_grounded_pct']}%) |",
        f"| Avg Latency | {agg['avg_latency_ms']:.0f}ms | P95: {agg['p95_latency_ms']:.0f}ms |",
        f"---",
        f"## 📂 Category Breakdown",
        f"| Category | Questions | Avg Groundedness |",
        f"|----------|-----------|------------------|",
    ]
    for cat, stats in sorted(agg['category_breakdown'].items()):
        lines.append(f"| {cat} | {stats['count']} | {stats['avg_groundedness']:.1%} |")
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kolrose RAG Evaluation Suite")
    parser.add_argument("--mode", choices=["groundedness", "citations", "matching", "latency", "ablations", "full"],
                       default="full", help="Evaluation mode")
    parser.add_argument("--quick", action="store_true", help="Use 10 questions only")
    parser.add_argument("--output-dir", default="./evaluation_results", help="Output directory for reports")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()
    
    vectorstore = load_vectorstore()
    embeddings = load_embeddings()
    from app.rag_system import KolroseRAG
    rag_system = KolroseRAG(vectorstore)
    
    if args.mode == "full":
        runner = CompleteEvaluationRunner(vectorstore, embeddings)
        questions = EVALUATION_QUESTIONS[:10] if args.quick else EVALUATION_QUESTIONS
        report = runner.run_all(rag_system, eval_questions=questions, output_dir=args.output_dir, verbose=not args.quiet)
    elif args.mode == "latency":
        bench = LatencyBenchmark()
        questions = [q['question'] for q in EVALUATION_QUESTIONS[:15]]
        report = bench.benchmark(rag_system, questions, verbose=not args.quiet)
    elif args.mode == "ablations":
        study = AblationStudy(vectorstore, embeddings)
        questions = [q['question'] for q in EVALUATION_QUESTIONS[:5]]
        report = study.run_all_ablations(rag_system, questions, verbose=not args.quiet)
    elif args.mode == "matching":
        matcher = GoldAnswerMatcher(embeddings)
        questions = EVALUATION_QUESTIONS[:10] if args.quick else EVALUATION_QUESTIONS
        report = matcher.evaluate_all(rag_system, questions, verbose=not args.quiet)
    elif args.mode == "groundedness":
        evaluator = GroundednessEvaluator(vectorstore, embeddings)
        questions = EVALUATION_QUESTIONS[:10] if args.quick else EVALUATION_QUESTIONS
        report = evaluator.evaluate_all(rag_system, questions, verbose=not args.quiet)
    elif args.mode == "citations":
        evaluator = CitationAccuracyEvaluator(vectorstore, embeddings)
        questions = EVALUATION_QUESTIONS[:10] if args.quick else EVALUATION_QUESTIONS
        report = evaluator.evaluate_all(rag_system, questions, verbose=not args.quiet)