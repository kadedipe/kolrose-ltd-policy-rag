# success_metrics.py
"""
Comprehensive Success Metrics for Kolrose Limited Policy RAG System

This module defines, implements, and tracks all success metrics for the RAG system.
Metrics are specifically designed for the Nigerian company policy context with
measurable thresholds aligned to the project rubric requirements.
"""

import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from collections import defaultdict

# ============================================================================
# METRIC CATEGORIES AND DEFINITIONS
# ============================================================================

class MetricCategory(Enum):
    """Categories of success metrics"""
    INFORMATION_QUALITY = "information_quality"
    SYSTEM_PERFORMANCE = "system_performance"
    CITATION_ACCURACY = "citation_accuracy"
    USER_EXPERIENCE = "user_experience"
    RETRIEVAL_QUALITY = "retrieval_quality"

class MetricImportance(Enum):
    """Importance level for rubric compliance"""
    CRITICAL = "critical"      # Must exceed threshold for passing grade
    HIGH = "high"             # Strongly impacts quality score
    MEDIUM = "medium"         # Good to have, enhances system
    INFORMATIONAL = "informational"  # Tracks for improvement

@dataclass
class MetricDefinition:
    """Complete definition of a single success metric"""
    name: str
    display_name: str
    category: MetricCategory
    importance: MetricImportance
    description: str
    measurement_method: str
    target_threshold: float
    minimum_acceptable: float
    unit: str
    higher_is_better: bool
    kolrose_specific_rationale: str  # Why this matters for Kolrose context
    
    # For tracking
    current_value: Optional[float] = None
    last_measured: Optional[datetime] = None
    measurement_count: int = 0
    historical_values: List[float] = field(default_factory=list)

# ============================================================================
# METRIC DEFINITIONS - KOLROSE SPECIFIC
# ============================================================================

KOLROSE_SUCCESS_METRICS = {
    # ========================================================================
    # INFORMATION QUALITY METRICS (Rubric: "groundedness or citation accuracy")
    # ========================================================================
    
    "groundedness_score": MetricDefinition(
        name="groundedness_score",
        display_name="Response Groundedness",
        category=MetricCategory.INFORMATION_QUALITY,
        importance=MetricImportance.CRITICAL,
        description=(
            "Measures the proportion of factual claims in RAG responses that are "
            "directly supported by at least one retrieved document chunk from the "
            "Kolrose policy corpus. Each sentence is evaluated for source support."
        ),
        measurement_method=(
            "1. For each query, decompose response into individual factual claims\n"
            "2. For each claim, check if it appears in or is logically entailed by retrieved chunks\n"
            "3. Calculate: (supported claims) / (total claims)\n"
            "4. Use LLM-as-judge with Kolrose policy documents as ground truth\n"
            "5. Human verification on 20% sample for calibration"
        ),
        target_threshold=0.92,      # 92% - Excellence target
        minimum_acceptable=0.85,    # 85% - Minimum for passing
        unit="proportion (0-1)",
        higher_is_better=True,
        kolrose_specific_rationale=(
            "Critical for Kolrose because policy responses MUST be factually accurate. "
            "Incorrect policy information could lead to employee grievances, legal issues "
            "with Nigerian labor law compliance, or financial losses from misapplied policies. "
            "Example: Wrong PTO accrual information could cause payroll errors."
        ),
    ),
    
    "citation_accuracy": MetricDefinition(
        name="citation_accuracy",
        display_name="Citation Accuracy Rate",
        category=MetricCategory.CITATION_ACCURACY,
        importance=MetricImportance.CRITICAL,
        description=(
            "Measures the percentage of citations that correctly reference the exact "
            "Kolrose policy document ID, section number, and content. A citation is "
            "'accurate' if the referenced document exists, the section exists, and the "
            "cited content actually appears in that section."
        ),
        measurement_method=(
            "1. Extract all citations from response (format: [KOL-XX-NNN, Section X.X])\n"
            "2. Verify document ID exists in KOLROSE_POLICY_REGISTRY\n"
            "3. Verify section number exists in that document\n"
            "4. Verify cited content appears in that section (semantic similarity > 0.85)\n"
            "5. Calculate: (verified citations) / (total citations)\n"
            "6. Flag citations to non-existent documents/sections as hallucinations"
        ),
        target_threshold=0.95,      # 95% - Excellence target
        minimum_acceptable=0.90,    # 90% - Minimum for passing
        unit="proportion (0-1)",
        higher_is_better=True,
        kolrose_specific_rationale=(
            "Kolrose's 12 policy documents have specific document IDs (KOL-HR-001, etc.) "
            "and hierarchical section numbering. Citations must be precise because employees "
            "and managers need to verify information in original documents. Incorrect citations "
            "erode trust in the system. Nigerian regulatory audits may require exact policy references."
        ),
    ),
    
    "citation_recall": MetricDefinition(
        name="citation_recall",
        display_name="Citation Recall (Completeness)",
        category=MetricCategory.CITATION_ACCURACY,
        importance=MetricImportance.HIGH,
        description=(
            "Measures whether all factual claims that SHOULD have citations actually do. "
            "Detects 'orphan claims' - factual statements made without supporting citations."
        ),
        measurement_method=(
            "1. Identify all factual claims in response that are not common knowledge\n"
            "2. Check which claims have accompanying citations\n"
            "3. Calculate: (cited factual claims) / (total factual claims)\n"
            "4. Common knowledge exemption: 'Kolrose is located in Abuja' may not need citation"
        ),
        target_threshold=0.90,
        minimum_acceptable=0.80,
        unit="proportion (0-1)",
        higher_is_better=True,
        kolrose_specific_rationale=(
            "Multiple Kolrose policies interact (e.g., PTO + Remote Work + Travel). "
            "Responses about cross-policy topics must cite ALL relevant policies. "
            "Missing citations could cause employees to miss important related policies."
        ),
    ),
    
    "cross_policy_accuracy": MetricDefinition(
        name="cross_policy_accuracy",
        display_name="Cross-Policy Reference Accuracy",
        category=MetricCategory.INFORMATION_QUALITY,
        importance=MetricImportance.HIGH,
        description=(
            "Specifically measures accuracy when questions involve multiple Kolrose policies. "
            "Tests whether the system correctly identifies and reconciles policy interactions "
            "(e.g., PTO during remote work, expenses during travel)."
        ),
        measurement_method=(
            "1. Maintain test set of 20+ cross-policy questions\n"
            "2. Each question requires referencing 2+ policy documents\n"
            "3. Evaluate: Are all relevant policies identified?\n"
            "4. Evaluate: Are policy interactions correctly explained?\n"
            "5. Evaluate: Are any contradictory statements made?\n"
            "6. Score each cross-policy question 0-5, then normalize"
        ),
        target_threshold=0.88,
        minimum_acceptable=0.80,
        unit="proportion (0-1)",
        higher_is_better=True,
        kolrose_specific_rationale=(
            "Kolrose policies frequently interact: Remote Work (KOL-HR-005) affects Expense "
            "Reimbursement (KOL-FIN-001); Travel Policy (KOL-ADMIN-001) interacts with Leave "
            "(KOL-HR-002). Nigerian labor law requires consistent policy application across documents."
        ),
    ),
    
    "nigerian_context_accuracy": MetricDefinition(
        name="nigerian_context_accuracy",
        display_name="Nigerian Context Accuracy",
        category=MetricCategory.INFORMATION_QUALITY,
        importance=MetricImportance.MEDIUM,
        description=(
            "Measures accuracy on queries involving Nigeria-specific contexts: "
            "Nigerian labor law references, Naira (₦) amounts, local agencies (EFCC, NITDA), "
            "Nigerian holidays, Abuja-specific policies."
        ),
        measurement_method=(
            "1. Filter test queries for Nigerian-specific content\n"
            "2. Check accuracy of Naira amounts (no conversion errors)\n"
            "3. Verify Nigerian agency names and jurisdictions correct\n"
            "4. Confirm Nigerian public holiday dates accurate\n"
            "5. Validate Abuja office references (Suite 10, Bataiya Plaza, Area 2 Garki)"
        ),
        target_threshold=0.95,
        minimum_acceptable=0.90,
        unit="proportion (0-1)",
        higher_is_better=True,
        kolrose_specific_rationale=(
            "Kolrose operates in Nigerian context with Naira budgets, Nigerian labor law "
            "compliance, and Abuja-based headquarters. Errors in Nigerian-specific content "
            "could have legal and financial implications under Nigerian law."
        ),
    ),
    
    # ========================================================================
    # SYSTEM METRICS (Rubric: "latency and other system metrics")
    # ========================================================================
    
    "response_latency_p95": MetricDefinition(
        name="response_latency_p95",
        display_name="Response Latency (P95)",
        category=MetricCategory.SYSTEM_PERFORMANCE,
        importance=MetricImportance.CRITICAL,
        description=(
            "95th percentile of end-to-end response time from query submission "
            "to complete response delivery, including retrieval, generation, and "
            "citation verification."
        ),
        measurement_method=(
            "Server-side middleware captures timestamps:\n"
            "  t1 = request received\n"
            "  t2 = retrieval complete\n"
            "  t3 = generation complete\n"
            "  t4 = citation verification complete\n"
            "  t5 = response sent\n"
            "P95 calculated over rolling 1000-query window\n"
            "Segmented by: query complexity (single vs multi-policy)"
        ),
        target_threshold=2.0,       # 2 seconds - Excellence target
        minimum_acceptable=3.5,     # 3.5 seconds - Maximum acceptable
        unit="seconds",
        higher_is_better=False,
        kolrose_specific_rationale=(
            "Nigerian office environments may have variable internet connectivity. "
            "Employees at Kolrose's Abuja office or remote workers across Nigeria "
            "need responsive answers. Long latency could discourage use of the system."
        ),
    ),
    
    "retrieval_latency_p50": MetricDefinition(
        name="retrieval_latency_p50",
        display_name="Retrieval Latency (P50)",
        category=MetricCategory.SYSTEM_PERFORMANCE,
        importance=MetricImportance.HIGH,
        description=(
            "Median time for the retrieval phase only (embedding + similarity search). "
            "Excludes LLM generation time."
        ),
        measurement_method=(
            "Timing wrapper around vectorstore.similarity_search() calls.\n"
            "Excludes embedding model API call time if using remote embeddings."
        ),
        target_threshold=0.3,       # 300ms
        minimum_acceptable=0.5,     # 500ms
        unit="seconds",
        higher_is_better=False,
        kolrose_specific_rationale=(
            "With 12 policy documents (~140 pages), the retrieval index should be fast. "
            "Slow retrieval suggests indexing problems or need for optimization."
        ),
    ),
    
    "tokens_per_response": MetricDefinition(
        name="tokens_per_response",
        display_name="Response Token Efficiency",
        category=MetricCategory.SYSTEM_PERFORMANCE,
        importance=MetricImportance.MEDIUM,
        description=(
            "Average number of tokens in generated responses. Tracks efficiency "
            "and cost-effectiveness of responses."
        ),
        measurement_method=(
            "Count input tokens (retrieved context) and output tokens (generated response)\n"
            "Track ratio of output tokens to input tokens\n"
            "Monitor for unnecessarily verbose responses"
        ),
        target_threshold=400,       # ~400 output tokens optimal
        minimum_acceptable=600,     # Up to 600 acceptable
        unit="tokens",
        higher_is_better=False,     # Lower is more efficient (within reason)
        kolrose_specific_rationale=(
            "OpenAI API costs in Naira terms. Efficient token usage reduces operational "
            "costs for Kolrose Limited. Nigerian companies are cost-sensitive to SaaS pricing."
        ),
    ),
    
    "uptime_percentage": MetricDefinition(
        name="uptime_percentage",
        display_name="System Uptime",
        category=MetricCategory.SYSTEM_PERFORMANCE,
        importance=MetricImportance.HIGH,
        description=(
            "Percentage of time the RAG system is available and responding to queries. "
            "Measured over 30-day rolling window."
        ),
        measurement_method=(
            "Health check endpoint pinged every 60 seconds\n"
            "Uptime = (successful responses) / (total pings)\n"
            "Planned maintenance windows excluded"
        ),
        target_threshold=0.995,     # 99.5% (3.6 hours downtime/month)
        minimum_acceptable=0.99,    # 99% (7.2 hours downtime/month)
        unit="proportion (0-1)",
        higher_is_better=True,
        kolrose_specific_rationale=(
            "Kolrose operates during Nigerian business hours (8AM-5PM WAT, Mon-Fri). "
            "System must be available during core hours. Nigerian power infrastructure "
            "considerations mean hosted solution (Render) uptime is critical."
        ),
    ),
    
    # ========================================================================
    # RETRIEVAL QUALITY METRICS
    # ========================================================================
    
    "retrieval_precision_at_5": MetricDefinition(
        name="retrieval_precision_at_5",
        display_name="Retrieval Precision@5",
        category=MetricCategory.RETRIEVAL_QUALITY,
        importance=MetricImportance.HIGH,
        description=(
            "Among the top 5 retrieved chunks, what proportion are actually "
            "relevant to answering the query? Measures retrieval quality."
        ),
        measurement_method=(
            "1. For 100 test queries, retrieve top 5 chunks\n"
            "2. Human annotators label each chunk as: Highly Relevant / Somewhat Relevant / Not Relevant\n"
            "3. Precision@5 = (Highly + Somewhat Relevant) / 5\n"
            "4. Also calculate Precision@3 for stricter measure\n"
            "5. Annotators must be familiar with Kolrose policies"
        ),
        target_threshold=0.85,      # 85% precision
        minimum_acceptable=0.75,    # 75% minimum
        unit="proportion (0-1)",
        higher_is_better=True,
        kolrose_specific_rationale=(
            "With 12 Kolrose policies covering different domains (HR, IT, Finance, Admin), "
            "the retriever must correctly identify which policy document(s) are relevant. "
            "Poor precision could retrieve IT policy for an HR question."
        ),
    ),
    
    "retrieval_recall_at_10": MetricDefinition(
        name="retrieval_recall_at_10",
        display_name="Retrieval Recall@10",
        category=MetricCategory.RETRIEVAL_QUALITY,
        importance=MetricImportance.HIGH,
        description=(
            "Of all relevant chunks that exist in the corpus for a query, "
            "what proportion are captured in the top 10 retrieved chunks?"
        ),
        measurement_method=(
            "1. For 50 test queries, manually identify ALL relevant chunks in corpus\n"
            "2. Retrieve top 10 chunks\n"
            "3. Recall@10 = (relevant retrieved chunks) / (total relevant chunks)\n"
            "4. Target: Don't miss critical policy sections"
        ),
        target_threshold=0.90,
        minimum_acceptable=0.80,
        unit="proportion (0-1)",
        higher_is_better=True,
        kolrose_specific_rationale=(
            "Some Kolrose policy topics are covered in multiple documents. "
            "For example, 'leave' appears in Leave Policy (KOL-HR-002), Remote Work (KOL-HR-005), "
            "and Performance Management (KOL-HR-006). High recall ensures all relevant "
            "policy sections are considered."
        ),
    ),
    
    "document_coverage_in_retrieval": MetricDefinition(
        name="document_coverage_in_retrieval",
        display_name="Policy Document Retrieval Coverage",
        category=MetricCategory.RETRIEVAL_QUALITY,
        importance=MetricImportance.MEDIUM,
        description=(
            "Ensures all 12 Kolrose policy documents appear in retrieval results "
            "across diverse queries. Detects if any policy is systematically 'hidden' "
            "from retrieval."
        ),
        measurement_method=(
            "1. Run 200 diverse test queries\n"
            "2. Track which documents appear in top-10 results\n"
            "3. Calculate: (documents appearing at least once) / 12\n"
            "4. Target: All 12 documents retrievable\n"
            "5. Investigate any document appearing < 5 times"
        ),
        target_threshold=1.0,       # All 12 documents
        minimum_acceptable=0.92,    # At least 11 of 12
        unit="proportion (0-1)",
        higher_is_better=True,
        kolrose_specific_rationale=(
            "Each of Kolrose's 12 policies serves a distinct purpose. If the Health & Safety "
            "policy (KOL-ADMIN-002) never appears in results, employees cannot get safety "
            "information. Systematic non-retrieval indicates indexing problems."
        ),
    ),
    
    # ========================================================================
    # USER EXPERIENCE & BUSINESS METRICS
    # ========================================================================
    
    "response_helpfulness_rating": MetricDefinition(
        name="response_helpfulness_rating",
        display_name="User-Perceived Helpfulness",
        category=MetricCategory.USER_EXPERIENCE,
        importance=MetricImportance.HIGH,
        description=(
            "Average user rating of how helpful the response was in answering "
            "their policy question. Captured via in-app feedback mechanism."
        ),
        measurement_method=(
            "Post-response thumbs up/down with optional 1-5 star rating\n"
            "Track per policy category (HR, IT, Finance, Admin)\n"
            "Correlate with groundedness and citation accuracy scores"
        ),
        target_threshold=4.2,       # 4.2 out of 5
        minimum_acceptable=3.8,     # 3.8 out of 5
        unit="rating (1-5)",
        higher_is_better=True,
        kolrose_specific_rationale=(
            "Kolrose employees (150 staff) need quick, accurate policy answers. "
            "Low helpfulness ratings indicate the system is not meeting real user needs. "
            "Nigerian workplace culture values direct, clear communication."
        ),
    ),
    
    "policy_category_balance": MetricDefinition(
        name="policy_category_balance",
        display_name="Response Quality Across Policy Categories",
        category=MetricCategory.USER_EXPERIENCE,
        importance=MetricImportance.MEDIUM,
        description=(
            "Ensures consistent quality across Kolrose's four policy categories: "
            "HR, IT, Finance, and Administration. Detects category-specific weaknesses."
        ),
        measurement_method=(
            "Segment all metrics by policy category:\n"
            "  - Category 1: HR (KOL-HR-001 through KOL-HR-008)\n"
            "  - Category 2: IT (KOL-IT-001)\n"
            "  - Category 3: Finance (KOL-FIN-001, KOL-FIN-002)\n"
            "  - Category 4: Admin (KOL-ADMIN-001, KOL-ADMIN-002)\n"
            "Max allowable variance between categories: 15%"
        ),
        target_threshold=0.85,      # Minimum score in any category
        minimum_acceptable=0.75,
        unit="proportion (0-1)",
        higher_is_better=True,
        kolrose_specific_rationale=(
            "Kolrose employees from different departments will query different policies. "
            "IT staff query KOL-IT-001 heavily; HR queries KOL-HR policies. Weak performance "
            "in one category disadvantages that department."
        ),
    ),
    
    # ========================================================================
    # SAFETY & COMPLIANCE METRICS
    # ========================================================================
    
    "compliance_flag_accuracy": MetricDefinition(
        name="compliance_flag_accuracy",
        display_name="Compliance Warning Accuracy",
        category=MetricCategory.INFORMATION_QUALITY,
        importance=MetricImportance.HIGH,
        description=(
            "Accuracy of compliance warnings triggered by the system. Measures whether "
            "warnings are correctly raised (true positives) and not incorrectly raised "
            "(false positives) for sensitive topics."
        ),
        measurement_method=(
            "Sensitive topics for Kolrose:\n"
            "  - Password/credential sharing → Should warn\n"
            "  - Bribery/corruption questions → Should warn\n"
            "  - Unauthorized access → Should warn\n"
            "  - Normal PTO questions → Should NOT warn\n"
            "Calculate: Precision and Recall of warnings"
        ),
        target_threshold=0.95,
        minimum_acceptable=0.90,
        unit="proportion (0-1)",
        higher_is_better=True,
        kolrose_specific_rationale=(
            "Nigerian companies face strict anti-corruption regulations (EFCC). Kolrose's "
            "Code of Conduct (KOL-HR-003) mandates reporting of certain topics. The system "
            "must correctly identify when to escalate. False warnings erode trust; missed "
            "warnings create compliance risk."
        ),
    ),
    
    "hallucination_rate": MetricDefinition(
        name="hallucination_rate",
        display_name="Hallucination (Fabrication) Rate",
        category=MetricCategory.INFORMATION_QUALITY,
        importance=MetricImportance.CRITICAL,
        description=(
            "Rate at which the system generates completely fabricated information "
            "not present in any Kolrose policy document. Includes invented policy names, "
            "fake section numbers, or made-up policy provisions."
        ),
        measurement_method=(
            "1. For each response, identify all factual claims\n"
            "2. Search entire Kolrose corpus (not just retrieved chunks) for each claim\n"
            "3. Claims with zero corpus support = hallucinations\n"
            "4. Categorize: Invented policy, wrong amount, fake procedure, etc.\n"
            "5. Track severity: Minor (wrong number) vs Major (invented policy)"
        ),
        target_threshold=0.02,      # 2% hallucination rate
        minimum_acceptable=0.05,    # 5% maximum
        unit="proportion (0-1)",
        higher_is_better=False,
        kolrose_specific_rationale=(
            "Hallucinations in policy context are dangerous. Inventing a PTO policy could "
            "cause payroll errors. Creating fake security procedures could create actual "
            "security vulnerabilities. Nigerian labor law could hold company liable for "
            "system-generated misinformation."
        ),
    ),
}

# ============================================================================
# METRIC CALCULATION ENGINE
# ============================================================================

class KolroseMetricsCalculator:
    """
    Calculates all success metrics for the Kolrose RAG system.
    """
    
    def __init__(self):
        self.metrics_definitions = KOLROSE_SUCCESS_METRICS
        self.query_log = []
        
    def calculate_groundedness(
        self,
        response: str,
        retrieved_chunks: List[Dict],
        use_llm_judge: bool = True,
    ) -> Tuple[float, Dict]:
        """
        Calculate groundedness score for a single response.
        
        Uses sentence-level decomposition and source verification.
        """
        import re
        
        # Step 1: Decompose response into factual claims
        sentences = re.split(r'(?<=[.!?])\s+', response)
        factual_claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Skip citation-only fragments
            if re.match(r'^\[KOL-\w+-\d+.*\]$', sentence):
                continue
            
            # Skip purely conversational elements
            if sentence.lower() in [
                'i hope this helps', 'let me know if you have questions',
                'is there anything else', ''
            ]:
                continue
            
            factual_claims.append(sentence)
        
        if not factual_claims:
            return 1.0, {"claims": [], "verifications": []}
        
        # Step 2: Verify each claim against retrieved chunks
        verifications = []
        supported_count = 0
        
        for claim in factual_claims:
            is_supported = self._verify_claim_against_chunks(
                claim,
                retrieved_chunks,
            )
            
            verifications.append({
                "claim": claim,
                "supported": is_supported,
                "supporting_chunks": self._find_supporting_chunks(
                    claim, retrieved_chunks
                ) if is_supported else [],
            })
            
            if is_supported:
                supported_count += 1
        
        # Step 3: Calculate score
        groundedness = supported_count / len(factual_claims)
        
        return groundedness, {
            "total_claims": len(factual_claims),
            "supported_claims": supported_count,
            "unsupported_claims": len(factual_claims) - supported_count,
            "verifications": verifications,
        }
    
    def _verify_claim_against_chunks(
        self,
        claim: str,
        chunks: List[Dict],
        similarity_threshold: float = 0.7,
    ) -> bool:
        """
        Verify if a claim is supported by any retrieved chunk.
        Uses keyword overlap and semantic similarity heuristics.
        """
        claim_lower = claim.lower()
        
        for chunk in chunks:
            content = chunk.get("content", chunk.get("page_content", ""))
            content_lower = content.lower()
            
            # Method 1: Direct substring match (exact policy quote)
            if len(claim) > 30 and claim_lower[:50] in content_lower:
                return True
            
            # Method 2: Key number matching (policy specifics)
            claim_numbers = set(re.findall(r'\d+', claim))
            content_numbers = set(re.findall(r'\d+', content))
            common_numbers = claim_numbers & content_numbers
            
            # Extract key terms from claim
            key_terms = [word for word in claim_lower.split() 
                        if len(word) > 4 and word not in {
                            'which', 'would', 'could', 'should', 'their', 'there',
                            'about', 'these', 'those', 'being', 'have', 'been'
                        }]
            
            matching_terms = sum(1 for term in key_terms if term in content_lower)
            
            # Scoring
            if len(key_terms) > 0:
                term_overlap = matching_terms / len(key_terms)
                
                # Number matching bonus (strong signal for policy content)
                number_bonus = min(len(common_numbers) * 0.15, 0.3)
                
                combined_score = term_overlap + number_bonus
                
                if combined_score >= similarity_threshold:
                    return True
        
        return False
    
    def _find_supporting_chunks(
        self,
        claim: str,
        chunks: List[Dict],
    ) -> List[str]:
        """Find which chunks support a claim"""
        supporting = []
        
        for chunk in chunks:
            content = chunk.get("content", chunk.get("page_content", ""))
            metadata = chunk.get("metadata", {})
            citation = metadata.get("citation_text", "Unknown")
            
            if self._verify_claim_against_chunks(claim, [chunk], similarity_threshold=0.6):
                supporting.append(citation)
        
        return supporting
    
    def calculate_citation_accuracy(
        self,
        response: str,
        citations: List[Dict],
        corpus_documents: Dict[str, str],
    ) -> Tuple[float, Dict]:
        """
        Calculate citation accuracy by verifying each citation against source documents.
        """
        if not citations:
            return 0.0, {"error": "No citations to evaluate"}
        
        verification_results = []
        verified_count = 0
        
        for citation in citations:
            result = self._verify_single_citation(citation, corpus_documents)
            verification_results.append(result)
            
            if result["is_accurate"]:
                verified_count += 1
        
        accuracy = verified_count / len(citations)
        
        return accuracy, {
            "total_citations": len(citations),
            "verified_citations": verified_count,
            "failed_citations": len(citations) - verified_count,
            "details": verification_results,
        }
    
    def _verify_single_citation(
        self,
        citation: Dict,
        corpus_documents: Dict[str, str],
    ) -> Dict:
        """
        Verify a single citation against the actual policy documents.
        """
        doc_id = citation.get("document_id", "")
        section = citation.get("section", "")
        cited_content = citation.get("cited_content", "")
        
        result = {
            "citation": citation,
            "is_accurate": False,
            "checks": {},
        }
        
        # Check 1: Document exists
        doc_exists = doc_id in KOLROSE_POLICY_REGISTRY
        result["checks"]["document_exists"] = doc_exists
        
        if not doc_exists:
            result["failure_reason"] = f"Document {doc_id} not in registry"
            return result
        
        # Check 2: Document content available
        if doc_id not in corpus_documents:
            result["failure_reason"] = f"Document {doc_id} content not loaded"
            return result
        
        doc_content = corpus_documents[doc_id]
        
        # Check 3: Section exists in document
        section_pattern = f"## {section}" if section else ""
        section_exists = section_pattern in doc_content if section else True
        result["checks"]["section_exists"] = section_exists
        
        # Check 4: Cited content in document (if provided)
        if cited_content:
            content_found = cited_content.lower()[:100] in doc_content.lower()
            result["checks"]["content_matches"] = content_found
        else:
            content_found = True  # Can't verify without content
        
        result["is_accurate"] = doc_exists and section_exists and content_found
        
        return result
    
    def calculate_latency_metrics(
        self,
        latency_samples: List[float],
    ) -> Dict[str, float]:
        """
        Calculate latency percentiles from samples.
        """
        if not latency_samples:
            return {"error": "No latency samples"}
        
        samples_array = np.array(latency_samples)
        
        return {
            "p50": float(np.percentile(samples_array, 50)),
            "p95": float(np.percentile(samples_array, 95)),
            "p99": float(np.percentile(samples_array, 99)),
            "mean": float(np.mean(samples_array)),
            "min": float(np.min(samples_array)),
            "max": float(np.max(samples_array)),
            "std_dev": float(np.std(samples_array)),
            "sample_count": len(latency_samples),
        }
    
    def calculate_hallucination_rate(
        self,
        response: str,
        full_corpus: Dict[str, str],
    ) -> Tuple[float, List[Dict]]:
        """
        Detect hallucinations by checking all claims against entire corpus.
        """
        import re
        
        # Decompose into claims
        sentences = re.split(r'(?<=[.!?])\s+', response)
        claims = [s.strip() for s in sentences 
                 if s.strip() and len(s.split()) > 4]
        
        hallucinations = []
        
        for claim in claims:
            # Search entire corpus for this claim
            found_in_corpus = False
            supporting_docs = []
            
            for doc_id, doc_content in full_corpus.items():
                # Check for substantial overlap
                claim_words = set(claim.lower().split())
                # Sliding window search
                words = doc_content.lower().split()
                
                for i in range(len(words) - len(claim_words) + 1):
                    window = set(words[i:i + len(claim_words)])
                    overlap = len(claim_words & window) / len(claim_words)
                    
                    if overlap > 0.7:
                        found_in_corpus = True
                        supporting_docs.append(doc_id)
                        break
                
                if found_in_corpus:
                    break
            
            if not found_in_corpus:
                hallucinations.append({
                    "claim": claim,
                    "claim_length": len(claim),
                    "severity": "major" if len(claim) > 50 else "minor",
                })
        
        hallucination_rate = len(hallucinations) / len(claims) if claims else 0
        return hallucination_rate, hallucinations


# ============================================================================
# TEST QUERY SUITE FOR KOLROSE METRICS
# ============================================================================

KOLROSE_TEST_QUERIES = {
    # Single policy queries
    "single_policy": [
        {
            "id": "SP-001",
            "query": "What is the annual leave entitlement at Kolrose Limited?",
            "expected_documents": ["KOL-HR-002"],
            "expected_sections": ["1.1"],
            "difficulty": "easy",
            "ground_truth_keywords": ["15 working days", "0-2 years", "20 working days"],
        },
        {
            "id": "SP-002",
            "query": "What are the password requirements for Kolrose systems?",
            "expected_documents": ["KOL-IT-001"],
            "expected_sections": ["2.2"],
            "difficulty": "easy",
            "ground_truth_keywords": ["12 characters", "uppercase", "lowercase", "90 days"],
        },
        {
            "id": "SP-003",
            "query": "How do I request maternity leave?",
            "expected_documents": ["KOL-HR-002"],
            "expected_sections": ["3"],
            "difficulty": "easy",
            "ground_truth_keywords": ["16 weeks", "112 calendar days", "medical certificate"],
        },
        {
            "id": "SP-004",
            "query": "What is the maximum hotel rate for business travel in Abuja?",
            "expected_documents": ["KOL-ADMIN-001"],
            "expected_sections": ["4.1"],
            "difficulty": "medium",
            "ground_truth_keywords": ["₦35,000", "per night", "3-Star"],
        },
        {
            "id": "SP-005",
            "query": "What procurement method is required for purchases above ₦2,000,000?",
            "expected_documents": ["KOL-FIN-002"],
            "expected_sections": ["2.4"],
            "difficulty": "medium",
            "ground_truth_keywords": ["Formal Tender", "3-5 bidders"],
        },
    ],
    
    # Cross-policy queries
    "cross_policy": [
        {
            "id": "CP-001",
            "query": "Can I work remotely while on annual leave?",
            "expected_documents": ["KOL-HR-002", "KOL-HR-005"],
            "expected_sections": ["1", "3"],
            "difficulty": "medium",
            "ground_truth_keywords": ["leave", "remote", "not expected to work"],
        },
        {
            "id": "CP-002",
            "query": "What expenses can I claim when traveling for a client meeting in Lagos?",
            "expected_documents": ["KOL-FIN-001", "KOL-ADMIN-001"],
            "expected_sections": ["2.1", "5"],
            "difficulty": "hard",
            "ground_truth_keywords": ["₦35,000", "per diem", "receipts"],
        },
        {
            "id": "CP-003",
            "query": "If I'm on a Performance Improvement Plan, can I still work remotely?",
            "expected_documents": ["KOL-HR-006", "KOL-HR-005"],
            "expected_sections": ["5", "2.3"],
            "difficulty": "hard",
            "ground_truth_keywords": ["not approved", "ineligible", "PIP"],
        },
        {
            "id": "CP-004",
            "query": "What training budget is available for IT security certifications?",
            "expected_documents": ["KOL-HR-007", "KOL-IT-001"],
            "expected_sections": ["2.2", "2.4"],
            "difficulty": "medium",
            "ground_truth_keywords": ["₦300,000", "certification", "security"],
        },
        {
            "id": "CP-005",
            "query": "Can I expense a new office chair for my remote work setup?",
            "expected_documents": ["KOL-FIN-001", "KOL-HR-005"],
            "expected_sections": ["2.5", "4.1"],
            "difficulty": "medium",
            "ground_truth_keywords": ["₦100,000", "ergonomic", "once every 3 years"],
        },
    ],
    
    # Nigerian context queries
    "nigerian_context": [
        {
            "id": "NG-001",
            "query": "What Nigerian public holidays does Kolrose observe?",
            "expected_documents": ["KOL-HR-002"],
            "expected_sections": ["5"],
            "difficulty": "easy",
            "ground_truth_keywords": ["Federal Government", "gazetted", "200%"],
        },
        {
            "id": "NG-002",
            "query": "How does Kolrose comply with NDPR data protection requirements?",
            "expected_documents": ["KOL-HR-003", "KOL-IT-001"],
            "expected_sections": ["2.2", "4"],
            "difficulty": "hard",
            "ground_truth_keywords": ["Nigeria Data Protection Regulation", "DPO"],
        },
        {
            "id": "NG-003",
            "query": "What anti-corruption laws apply to Kolrose operations?",
            "expected_documents": ["KOL-HR-003"],
            "expected_sections": ["4"],
            "difficulty": "medium",
            "ground_truth_keywords": ["EFCC", "Corrupt Practices Act 2000", "zero tolerance"],
        },
    ],
    
    # Edge cases and out-of-scope
    "edge_cases": [
        {
            "id": "EC-001",
            "query": "What is the best restaurant near Bataiya Plaza?",
            "expected_documents": [],
            "expected_sections": [],
            "difficulty": "easy",
            "ground_truth_keywords": ["not in policy", "cannot answer"],
        },
        {
            "id": "EC-002",
            "query": "Can I share my password with the IT support team?",
            "expected_documents": ["KOL-IT-001", "KOL-HR-003"],
            "expected_sections": ["2.3", "5"],
            "difficulty": "easy",
            "ground_truth_keywords": ["must not be shared", "even IT staff"],
        },
        {
            "id": "EC-003",
            "query": "My manager denied my leave. What can I do?",
            "expected_documents": ["KOL-HR-008", "KOL-HR-002"],
            "expected_sections": ["1.2", "2"],
            "difficulty": "hard",
            "ground_truth_keywords": ["grievance", "appeal", "HR"],
        },
    ],
}

# ============================================================================
# METRICS REPORTING DASHBOARD
# ============================================================================

class MetricsDashboard:
    """
    Generates metrics reports and tracks historical performance.
    """
    
    def __init__(self, metrics_calculator: KolroseMetricsCalculator):
        self.calculator = metrics_calculator
        self.historical_metrics = defaultdict(list)
        
    def generate_excellence_report(
        self,
        current_metrics: Dict[str, float],
        target_metrics: Dict[str, MetricDefinition] = None,
    ) -> str:
        """
        Generate a formatted excellence report showing metrics vs targets.
        """
        if target_metrics is None:
            target_metrics = KOLROSE_SUCCESS_METRICS
        
        report = []
        report.append("=" * 80)
        report.append("KOLROSE LIMITED - RAG SYSTEM EXCELLENCE METRICS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S WAT')}")
        report.append(f"Location: Suite 10, Bataiya Plaza, Area 2 Garki, Abuja, FCT")
        report.append("=" * 80)
        
        # Group metrics by category
        categories = defaultdict(list)
        for metric_name, metric_def in target_metrics.items():
            categories[metric_def.category.value].append((metric_name, metric_def))
        
        for category, metrics in categories.items():
            report.append(f"\n{'─' * 80}")
            report.append(f"📊 {category.upper().replace('_', ' ')}")
            report.append(f"{'─' * 80}")
            
            for metric_name, metric_def in metrics:
                current_value = current_metrics.get(metric_name)
                
                if current_value is not None:
                    if metric_def.higher_is_better:
                        status = "✅" if current_value >= metric_def.target_threshold else \
                                "⚠️" if current_value >= metric_def.minimum_acceptable else "❌"
                    else:
                        status = "✅" if current_value <= metric_def.target_threshold else \
                                "⚠️" if current_value <= metric_def.minimum_acceptable else "❌"
                    
                    report.append(f"\n{status} {metric_def.display_name}")
                    report.append(f"   Current: {current_value:.3f} {metric_def.unit}")
                    report.append(f"   Target:  {metric_def.target_threshold} {metric_def.unit}")
                    report.append(f"   Minimum: {metric_def.minimum_acceptable} {metric_def.unit}")
                else:
                    report.append(f"\n⚪ {metric_def.display_name}")
                    report.append(f"   Not yet measured")
        
        # Overall assessment
        report.append(f"\n{'═' * 80}")
        report.append("OVERALL ASSESSMENT")
        report.append(f"{'═' * 80}")
        
        critical_metrics = {
            name: defn for name, defn in target_metrics.items()
            if defn.importance == MetricImportance.CRITICAL
        }
        
        critical_passing = sum(
            1 for name, defn in critical_metrics.items()
            if current_metrics.get(name) is not None and (
                (defn.higher_is_better and current_metrics[name] >= defn.minimum_acceptable) or
                (not defn.higher_is_better and current_metrics[name] <= defn.minimum_acceptable)
            )
        )
        
        report.append(f"Critical Metrics Passing: {critical_passing}/{len(critical_metrics)}")
        
        if critical_passing == len(critical_metrics):
            report.append("🏆 ALL CRITICAL METRICS PASSING - EXCELLENCE ACHIEVED")
        
        return "\n".join(report)


# ============================================================================
# MAIN METRICS EXECUTION
# ============================================================================

def run_metrics_evaluation(
    rag_system,
    test_queries: Dict = None,
) -> Dict[str, Any]:
    """
    Execute comprehensive metrics evaluation for the Kolrose RAG system.
    """
    if test_queries is None:
        test_queries = KOLROSE_TEST_QUERIES
    
    calculator = KolroseMetricsCalculator()
    dashboard = MetricsDashboard(calculator)
    
    all_results = {
        "groundedness_scores": [],
        "citation_accuracy_scores": [],
        "latency_samples": [],
        "hallucination_rates": [],
        "query_details": [],
    }
    
    # Run all test queries
    all_queries = []
    for category, queries in test_queries.items():
        all_queries.extend(queries)
    
    print(f"Running metrics evaluation on {len(all_queries)} test queries...")
    
    for i, test_case in enumerate(all_queries):
        query_start = time.time()
        
        # Execute query
        result = rag_system.query(test_case["query"])
        latency = time.time() - query_start
        
        # Calculate metrics
        groundedness, ground_details = calculator.calculate_groundedness(
            result["answer"],
            result.get("retrieved_chunks", []),
        )
        
        hallucination_rate, hallucinations = calculator.calculate_hallucination_rate(
            result["answer"],
            result.get("full_corpus", {}),
        )
        
        # Store results
        all_results["groundedness_scores"].append(groundedness)
        all_results["latency_samples"].append(latency)
        all_results["hallucination_rates"].append(hallucination_rate)
        
        all_results["query_details"].append({
            "query_id": test_case["id"],
            "query": test_case["query"],
            "category": test_case.get("difficulty", "unknown"),
            "groundedness": groundedness,
            "latency": latency,
            "hallucination_rate": hallucination_rate,
            "hallucinations_found": len(hallucinations),
        })
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(all_queries)} queries...")
    
    # Aggregate results
    aggregate_metrics = {
        "groundedness_score": np.mean(all_results["groundedness_scores"]) if all_results["groundedness_scores"] else 0,
        "response_latency_p95": float(np.percentile(all_results["latency_samples"], 95)) if all_results["latency_samples"] else 0,
        "hallucination_rate": np.mean(all_results["hallucination_rates"]) if all_results["hallucination_rates"] else 0,
        "total_queries_evaluated": len(all_queries),
    }
    
    # Generate report
    report = dashboard.generate_excellence_report(aggregate_metrics)
    print(report)
    
    return {
        "aggregate_metrics": aggregate_metrics,
        "detailed_results": all_results,
        "report": report,
    }