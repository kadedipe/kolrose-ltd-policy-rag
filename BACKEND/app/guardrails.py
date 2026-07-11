# app/guardrails.py
"""
Guardrails System for Kolrose Limited Policy Assistant.

Implements safety and quality controls:
1. Corpus boundary detection - Refuse out-of-scope queries
2. Output length limiting - Prevent excessively long responses
3. Mandatory citation enforcement - Require document references
4. Sensitive topic detection - Handle security/legal queries appropriately
5. Response validation - Check generated answers against policies
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .config import (
    COMPANY_INFO,
    MAX_RESPONSE_CHARS,
    MAX_OUTPUT_TOKENS,
    CITATION_REQUIRED,
    SENSITIVE_TOPICS_ENABLED,
    ENABLE_GUARDRAILS,
)

logger = logging.getLogger(__name__)


# ============================================================================
# GUARDRAIL TYPES AND CONFIGURATIONS
# ============================================================================

class GuardrailSeverity(Enum):
    """Severity levels for guardrail violations"""
    BLOCK = "block"         # Completely block the response
    WARN = "warn"          # Allow but add warning
    MODIFY = "modify"      # Modify the response
    LOG_ONLY = "log_only"  # Only log, no user-facing action


class GuardrailType(Enum):
    """Types of guardrails"""
    TOPIC_CLASSIFICATION = "topic_classification"
    OUTPUT_LENGTH = "output_length"
    CITATION_ENFORCEMENT = "citation_enforcement"
    SENSITIVE_CONTENT = "sensitive_content"
    RESPONSE_VALIDATION = "response_validation"
    HALLUCINATION_CHECK = "hallucination_check"


@dataclass
class GuardrailResult:
    """Result from a guardrail check"""
    passed: bool
    guardrail_type: GuardrailType
    severity: GuardrailSeverity
    message: str = ""
    action_taken: str = ""
    modified_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# GUARDRAIL 1: TOPIC CLASSIFICATION
# ============================================================================

class TopicGuardrail:
    """
    Ensures queries fall within Kolrose policy corpus boundaries.
    Refuses to answer off-topic or out-of-scope questions.
    """
    
    # Policy-related topics (must match at least one)
    POLICY_TOPICS = {
        'leave': [
            'leave', 'vacation', 'pto', 'sick day', 'sick leave',
            'maternity', 'paternity', 'time off', 'annual leave',
            'carryover', 'accrual', 'public holiday',
        ],
        'remote_work': [
            'remote', 'work from home', 'wfh', 'hybrid',
            'telecommuting', 'flexible work', 'home office',
        ],
        'security_it': [
            'password', 'vpn', 'mfa', 'multi-factor',
            'authentication', 'access control', 'data protection',
            'cybersecurity', 'acceptable use',
        ],
        'conduct_ethics': [
            'code of conduct', 'ethics', 'confidentiality',
            'discipline', 'warning', 'termination', 'dress code',
            'conflict of interest', 'anti-corruption',
        ],
        'expenses_finance': [
            'expense', 'reimbursement', 'travel cost', 'per diem',
            'allowance', 'mileage', 'receipt', 'procurement',
            'purchase', 'tender', 'contract',
        ],
        'performance': [
            'performance', 'review', 'appraisal', 'pip',
            'improvement plan', 'promotion', 'rating', 'bonus',
            'salary review',
        ],
        'training': [
            'training', 'certification', 'development', 'learning',
            'education', 'mentorship', 'internship', 'graduate',
        ],
        'travel': [
            'travel', 'flight', 'hotel', 'accommodation',
            'transport', 'airport', 'booking', 'travel advance',
        ],
        'health_safety': [
            'health', 'safety', 'emergency', 'fire', 'evacuation',
            'first aid', 'medical', 'wellness', 'ergonomic',
        ],
        'grievance_hr': [
            'grievance', 'complaint', 'dispute', 'appeal',
            'whistleblowing', 'harassment', 'discrimination',
        ],
        'company_info': [
            'kolrose', 'abuja', 'bataiya plaza', 'headquarters',
            'office location', 'company address', 'working hours',
        ],
    }
    
    # Clearly off-topic indicators
    OFF_TOPIC_PATTERNS = [
        r'\b(restaurant|food|menu|cuisine)\b',
        r'\b(weather|temperature|rain|sunny)\b',
        r'\b(sports?|football|basketball|soccer)\b',
        r'\b(movie|film|cinema|netflix|show)\b',
        r'\b(music|song|concert|album)\b',
        r'\b(recipe|cooking|baking|ingredient)\b',
        r'\b(celebrity|gossip|rumor)\b',
        r'\b(crypto|bitcoin|stock|trading|investment)\b',
        r'\b(election|politics|government|president)\b',
        r'\b(religion|church|mosque|prayer)\b',
        r'\b(dating|relationship advice|love)\b',
        r'\b(medical advice|diagnosis|symptom|treatment)\b',
        r'\b(legal advice|sue|attorney|lawyer)\b',
        r'\b(tax advice|tax return|irs|firs)\b',
    ]
    
    # Sensitive topics requiring special handling
    SENSITIVE_PATTERNS = {
        'credential_sharing': [
            r'share\s+(my\s+)?password',
            r'give\s+(out\s+)?(my\s+)?password',
            r'tell\s+(someone|anyone)\s+(my\s+)?password',
            r'disclose\s+(my\s+)?password',
            r'reveal\s+(my\s+)?password',
        ],
        'corruption': [
            r'\b(bribe|bribery|kickback)\b',
            r'\b(corruption|fraudulent)\b',
            r'\b(illegal\s+payment)\b',
            r'\b(under\s+the\s+table)\b',
        ],
        'harassment_reporting': [
            r'\b(harass|harassment|bully|bullying)\b',
            r'\b(discriminat|hostile\s+work)\b',
            r'\b(sexual\s+harassment)\b',
        ],
        'security_breach': [
            r'\b(data\s+breach|hacked|stolen\s+data)\b',
            r'\b(security\s+incident|ransomware)\b',
            r'\b(unauthorized\s+access)\b',
        ],
    }
    
    # Example questions for helpful redirection
    EXAMPLE_QUESTIONS = [
        "What is the annual leave entitlement?",
        "How do I request remote work?",
        "What are the password requirements?",
        "How are travel expenses reimbursed?",
        "What training budget is available?",
        "Where is Kolrose Limited located?",
    ]
    
    @classmethod
    def check(cls, query: str) -> GuardrailResult:
        """
        Check if a query falls within acceptable topic boundaries.
        """
        query_lower = query.lower().strip()
        
        # Step 1: Check sensitive topics first
        if SENSITIVE_TOPICS_ENABLED:
            for topic, patterns in cls.SENSITIVE_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        return GuardrailResult(
                            passed=False,
                            guardrail_type=GuardrailType.SENSITIVE_CONTENT,
                            severity=GuardrailSeverity.BLOCK,
                            message=f"Sensitive topic detected: {topic}",
                            action_taken="Blocked sensitive query, redirected to compliance channels",
                            modified_response=cls._get_sensitive_response(topic),
                            metadata={'topic': topic, 'matched_pattern': pattern},
                        )
        
        # Step 2: Check off-topic patterns
        for pattern in cls.OFF_TOPIC_PATTERNS:
            if re.search(pattern, query_lower):
                return GuardrailResult(
                    passed=False,
                    guardrail_type=GuardrailType.TOPIC_CLASSIFICATION,
                    severity=GuardrailSeverity.BLOCK,
                    message="Query is off-topic for policy assistant",
                    action_taken="Blocked off-topic query",
                    modified_response=cls._get_off_topic_response(),
                    metadata={'matched_pattern': pattern},
                )
        
        # Step 3: Check in-corpus topics
        matched_topics = []
        for topic, keywords in cls.POLICY_TOPICS.items():
            if any(kw in query_lower for kw in keywords):
                matched_topics.append(topic)
        
        if not matched_topics:
            return GuardrailResult(
                passed=False,
                guardrail_type=GuardrailType.TOPIC_CLASSIFICATION,
                severity=GuardrailSeverity.BLOCK,
                message="No matching policy topics found",
                action_taken="Blocked out-of-corpus query",
                modified_response=cls._get_out_of_corpus_response(),
            )
        
        # Query is acceptable
        return GuardrailResult(
            passed=True,
            guardrail_type=GuardrailType.TOPIC_CLASSIFICATION,
            severity=GuardrailSeverity.LOG_ONLY,
            message=f"Query matches topics: {matched_topics}",
            metadata={'matched_topics': matched_topics},
        )
    
    @classmethod
    def _get_off_topic_response(cls) -> str:
        """Generate response for off-topic queries"""
        examples = "\n".join(f"- {q}" for q in cls.EXAMPLE_QUESTIONS[:4])
        return (
            f"🚫 I'm a **policy assistant** for {COMPANY_INFO['name']} and can only "
            f"answer questions about our company policies.\n\n"
            f"**Try asking questions like:**\n{examples}\n\n"
            f"For other inquiries, please contact HR at {COMPANY_INFO['email_hr']}."
        )
    
    @classmethod
    def _get_out_of_corpus_response(cls) -> str:
        """Generate response for out-of-corpus queries"""
        return (
            f"📋 Your question doesn't appear to relate to our documented policies. "
            f"I can help with questions about leave, remote work, security, "
            f"expenses, training, and other Kolrose Limited procedures.\n\n"
            f"For assistance with other matters, contact HR at {COMPANY_INFO['email_hr']}."
        )
    
    @classmethod
    def _get_sensitive_response(cls, topic: str) -> str:
        """Generate response for sensitive topic queries"""
        responses = {
            'credential_sharing': (
                f"⚠️ **Password Security Notice**\n\n"
                f"Passwords must NEVER be shared with anyone, including IT staff. "
                f"This is a violation of our IT Security Policy [KOL-IT-001].\n\n"
                f"If you need IT assistance, contact {COMPANY_INFO['email_security']} "
                f"or call {COMPANY_INFO['hotline_it_security']}.\n\n"
                f"If you suspect your password has been compromised, report it immediately."
            ),
            'corruption': (
                f"⚠️ **Ethics & Compliance Notice**\n\n"
                f"{COMPANY_INFO['name']} maintains a zero-tolerance policy on corruption "
                f"and bribery. Such matters must be reported through proper channels:\n\n"
                f"📧 Compliance Officer: {COMPANY_INFO['email_compliance']}\n"
                f"📞 Whistleblower Hotline: {COMPANY_INFO['hotline_whistleblower']}\n\n"
                f"Reference: Code of Conduct [KOL-HR-003, Section 4]"
            ),
            'harassment_reporting': (
                f"⚠️ **Workplace Conduct Notice**\n\n"
                f"Harassment and discrimination are serious violations of our "
                f"Code of Conduct [KOL-HR-003]. If you're experiencing or witnessing "
                f"harassment, please report it:\n\n"
                f"📧 HR: {COMPANY_INFO['email_hr']}\n"
                f"📞 Compliance Hotline: {COMPANY_INFO['hotline_whistleblower']}\n\n"
                f"All reports are handled confidentially and retaliation is prohibited."
            ),
            'security_breach': (
                f"⚠️ **Security Incident Reporting**\n\n"
                f"If you suspect or have identified a security incident, "
                f"report it IMMEDIATELY:\n\n"
                f"📞 IT Security Hotline: {COMPANY_INFO['hotline_it_security']}\n"
                f"📧 Email: {COMPANY_INFO['email_security']}\n\n"
                f"Reference: IT Security Policy [KOL-IT-001, Section 5]"
            ),
        }
        return responses.get(
            topic,
            f"⚠️ This topic requires special handling. Please contact "
            f"{COMPANY_INFO['email_compliance']} for guidance."
        )


# ============================================================================
# GUARDRAIL 2: OUTPUT LENGTH LIMITER
# ============================================================================

class OutputLengthGuardrail:
    """
    Controls response length to prevent excessively long outputs.
    """
    
    # Limits
    HARD_CHAR_LIMIT = MAX_RESPONSE_CHARS  # 2000
    SOFT_CHAR_LIMIT = 1500
    HARD_SENTENCE_LIMIT = 12
    SOFT_SENTENCE_LIMIT = 8
    
    @classmethod
    def check(cls, response: str) -> GuardrailResult:
        """
        Check and enforce output length limits.
        """
        original_length = len(response)
        sentence_count = cls._count_sentences(response)
        
        # No issue
        if (original_length <= cls.SOFT_CHAR_LIMIT and 
            sentence_count <= cls.SOFT_SENTENCE_LIMIT):
            return GuardrailResult(
                passed=True,
                guardrail_type=GuardrailType.OUTPUT_LENGTH,
                severity=GuardrailSeverity.LOG_ONLY,
                metadata={
                    'original_length': original_length,
                    'sentence_count': sentence_count,
                },
            )
        
        # Soft limit exceeded - truncate gracefully
        if original_length > cls.HARD_CHAR_LIMIT:
            truncated = cls._truncate_at_sentence_boundary(
                response, cls.HARD_CHAR_LIMIT
            )
            return GuardrailResult(
                passed=False,
                guardrail_type=GuardrailType.OUTPUT_LENGTH,
                severity=GuardrailSeverity.MODIFY,
                message=f"Response truncated from {original_length} to {len(truncated)} chars",
                action_taken="Truncated response at sentence boundary",
                modified_response=truncated + "\n\n_[Response shortened for clarity]_",
                metadata={
                    'original_length': original_length,
                    'truncated_length': len(truncated),
                    'original_sentences': sentence_count,
                },
            )
        
        # Within hard limit but over soft limit - add note
        if sentence_count > cls.SOFT_SENTENCE_LIMIT:
            return GuardrailResult(
                passed=True,
                guardrail_type=GuardrailType.OUTPUT_LENGTH,
                severity=GuardrailSeverity.WARN,
                message=f"Response is {sentence_count} sentences (soft limit: {cls.SOFT_SENTENCE_LIMIT})",
                metadata={
                    'original_length': original_length,
                    'sentence_count': sentence_count,
                },
            )
        
        return GuardrailResult(
            passed=True,
            guardrail_type=GuardrailType.OUTPUT_LENGTH,
            severity=GuardrailSeverity.LOG_ONLY,
            metadata={'original_length': original_length},
        )
    
    @classmethod
    def _count_sentences(cls, text: str) -> int:
        """Count sentences in text"""
        return len(re.findall(r'[.!?]+', text))
    
    @classmethod
    def _truncate_at_sentence_boundary(cls, text: str, max_chars: int) -> str:
        """Truncate text at the nearest sentence boundary"""
        if len(text) <= max_chars:
            return text
        
        truncated = text[:max_chars]
        
        # Find last sentence break
        last_period = truncated.rfind('. ')
        last_exclaim = truncated.rfind('! ')
        last_question = truncated.rfind('? ')
        last_newline = truncated.rfind('\n\n')
        
        break_point = max(last_period, last_exclaim, last_question, last_newline)
        
        if break_point > max_chars * 0.5:
            return truncated[:break_point + 1]
        
        return truncated[:max_chars - 50] + "..."
    
    @classmethod
    def _count_paragraphs(cls, text: str) -> int:
        """Count paragraphs"""
        return len([p for p in text.split('\n\n') if p.strip()])


# ============================================================================
# GUARDRAIL 3: CITATION ENFORCEMENT
# ============================================================================

class CitationGuardrail:
    """
    Ensures responses include proper citations to policy documents.
    """
    
    # Valid citation patterns
    CITATION_PATTERNS = [
        r'\[KOL-\w+-\d+[,\s]*[§]?[\d.]*\]',  # [KOL-HR-002, Section 1.1]
        r'\[KOL-\w+-\d+\]',                     # [KOL-HR-002]
        r'KOL-\w+-\d+',                         # KOL-HR-002 (bare)
    ]
    
    # Minimum citations required
    MIN_CITATIONS = 1
    
    @classmethod
    def check(cls, response: str, source_docs: List[Dict] = None) -> GuardrailResult:
        """
        Check if response contains sufficient citations.
        """
        citations = cls._extract_citations(response)
        
        if len(citations) >= cls.MIN_CITATIONS:
            return GuardrailResult(
                passed=True,
                guardrail_type=GuardrailType.CITATION_ENFORCEMENT,
                severity=GuardrailSeverity.LOG_ONLY,
                message=f"Found {len(citations)} citations",
                metadata={
                    'citation_count': len(citations),
                    'citations': citations,
                },
            )
        
        # Missing citations - add sources section
        if source_docs:
            modified = cls._append_sources_section(response, source_docs)
            return GuardrailResult(
                passed=False,
                guardrail_type=GuardrailType.CITATION_ENFORCEMENT,
                severity=GuardrailSeverity.MODIFY,
                message="No citations found in response",
                action_taken="Appended sources section to response",
                modified_response=modified,
                metadata={'citation_count': 0},
            )
        
        return GuardrailResult(
            passed=False,
            guardrail_type=GuardrailType.CITATION_ENFORCEMENT,
            severity=GuardrailSeverity.WARN,
            message="No citations found, no sources available to append",
            metadata={'citation_count': 0},
        )
    
    @classmethod
    def _extract_citations(cls, text: str) -> List[str]:
        """Extract all citations from text"""
        citations = set()
        for pattern in cls.CITATION_PATTERNS:
            citations.update(re.findall(pattern, text))
        return sorted(citations)
    
    @classmethod
    def _append_sources_section(cls, response: str, source_docs: List[Dict]) -> str:
        """Append a sources section to the response"""
        seen = set()
        source_lines = []
        
        for doc in source_docs:
            doc_id = doc.get('document_id', 'Unknown')
            if doc_id != 'Unknown' and doc_id not in seen:
                seen.add(doc_id)
                policy_name = doc.get('policy_name', 'Policy Document')
                section = doc.get('section', '')
                
                line = f"- [{doc_id}] {policy_name}"
                if section and section != 'N/A':
                    line += f" — {section}"
                source_lines.append(line)
        
        if source_lines:
            response += "\n\n---\n📚 **Policy Sources Referenced:**\n" + "\n".join(source_lines)
        
        return response


# ============================================================================
# GUARDRAIL 4: RESPONSE VALIDATOR
# ============================================================================

class ResponseValidator:
    """
    Validates generated responses against source documents.
    Checks for potential hallucinations and inconsistencies.
    """
    
    # Patterns that may indicate hallucinations
    HALLUCINATION_INDICATORS = [
        r"I (think|believe|assume|guess)",
        r"(probably|maybe|perhaps|possibly)",
        r"(in my opinion|I would say)",
        r"based on (my|general) knowledge",
        r"typically companies",
        r"generally speaking",
    ]
    
    # Nigerian-specific validation
    NIGERIAN_CONTEXT_CHECKS = {
        'currency': {
            'pattern': r'₦\s*\d[\d,]*(?!\s*(million|billion|thousand))',
            'message': 'Naira amounts should match policy documents exactly',
        },
        'agencies': {
            'pattern': r'\b(EFCC|NITDA|NDPR|NESREA|PENCOM|ITF|NSITF)\b',
            'message': 'Nigerian agency references should be verified',
        },
    }
    
    @classmethod
    def check(
        cls, 
        response: str, 
        source_docs: List[Dict] = None
    ) -> GuardrailResult:
        """
        Validate response quality and flag potential issues.
        """
        issues = []
        
        # Check for hallucination indicators
        for pattern in cls.HALLUCINATION_INDICATORS:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                issues.append(f"Potential hallucination indicator: '{matches[0]}'")
        
        # Check for very short responses
        if len(response) < 30:
            issues.append("Response is unusually short")
        
        # Check for error messages in response
        if 'error' in response.lower() and len(response) < 100:
            issues.append("Response may contain an error message")
        
        if issues:
            return GuardrailResult(
                passed=False,
                guardrail_type=GuardrailType.RESPONSE_VALIDATION,
                severity=GuardrailSeverity.WARN,
                message=f"Found {len(issues)} potential issue(s)",
                metadata={'issues': issues},
            )
        
        return GuardrailResult(
            passed=True,
            guardrail_type=GuardrailType.RESPONSE_VALIDATION,
            severity=GuardrailSeverity.LOG_ONLY,
            message="Response validation passed",
        )
    
    @classmethod
    def compute_groundedness(
        cls, 
        answer: str, 
        source_contents: List[str]
    ) -> float:
        """
        Compute a groundedness score by checking overlap between
        answer claims and source documents.
        """
        # Split answer into claims
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        if not sentences:
            return 1.0
        
        supported = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Extract key terms
            key_terms = [
                w for w in sentence_lower.split() 
                if len(w) > 4 and w not in {
                    'which', 'would', 'could', 'should', 'their', 'there',
                    'about', 'these', 'those', 'being', 'have', 'been',
                }
            ]
            
            if not key_terms:
                supported += 1
                continue
            
            # Check against source documents
            for content in source_contents:
                content_lower = content.lower()
                matches = sum(1 for term in key_terms if term in content_lower)
                
                if matches / len(key_terms) > 0.4:
                    supported += 1
                    break
        
        return supported / len(sentences)


# ============================================================================
# INTEGRATED GUARDRAIL SYSTEM
# ============================================================================

class GuardrailSystem:
    """
    Orchestrates all guardrails and applies them to queries and responses.
    """
    
    def __init__(self):
        self.topic_guard = TopicGuardrail()
        self.length_guard = OutputLengthGuardrail()
        self.citation_guard = CitationGuardrail()
        self.validator = ResponseValidator()
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'blocked_queries': 0,
            'modified_responses': 0,
            'warnings_issued': 0,
        }
    
    def check_query(self, query: str) -> GuardrailResult:
        """
        Apply input guardrails to a query.
        """
        if not ENABLE_GUARDRAILS:
            return GuardrailResult(
                passed=True,
                guardrail_type=GuardrailType.TOPIC_CLASSIFICATION,
                severity=GuardrailSeverity.LOG_ONLY,
                message="Guardrails disabled",
            )
        
        self.stats['total_queries'] += 1
        
        result = self.topic_guard.check(query)
        
        if not result.passed and result.severity == GuardrailSeverity.BLOCK:
            self.stats['blocked_queries'] += 1
        
        return result
    
    def check_response(
        self, 
        response: str, 
        source_docs: List[Dict] = None,
        source_contents: List[str] = None,
    ) -> Tuple[str, List[GuardrailResult]]:
        """
        Apply output guardrails to a response.
        
        Returns:
            Tuple of (potentially_modified_response, list_of_guardrail_results)
        """
        results = []
        modified_response = response
        
        if not ENABLE_GUARDRAILS:
            return modified_response, results
        
        # Guardrail 2: Length check (always apply)
        length_result = self.length_guard.check(modified_response)
        results.append(length_result)
        if length_result.modified_response:
            modified_response = length_result.modified_response
            self.stats['modified_responses'] += 1
        
        # Guardrail 3: Citation enforcement (if required)
        if CITATION_REQUIRED:
            citation_result = self.citation_guard.check(modified_response, source_docs)
            results.append(citation_result)
            if citation_result.modified_response:
                modified_response = citation_result.modified_response
        
        # Guardrail 4: Response validation
        validation_result = self.validator.check(modified_response, source_docs)
        results.append(validation_result)
        if not validation_result.passed:
            self.stats['warnings_issued'] += 1
        
        # Optional: Groundedness check
        if source_contents:
            groundedness = self.validator.compute_groundedness(
                modified_response, source_contents
            )
            results[-1].metadata['groundedness_score'] = groundedness
        
        return modified_response, results
    
    def get_stats(self) -> Dict[str, int]:
        """Get guardrail statistics"""
        return self.stats.copy()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def apply_all_guardrails(
    query: str,
    response: str,
    source_docs: List[Dict] = None,
) -> Tuple[bool, str, List[Dict]]:
    """
    Apply all guardrails to a query-response pair.
    
    Args:
        query: The user's question
        response: The generated response
        source_docs: Source documents used for generation
        
    Returns:
        Tuple of (is_blocked, final_response, guardrail_log)
    """
    system = GuardrailSystem()
    
    # Check query
    query_result = system.check_query(query)
    
    if query_result.modified_response:
        # Query was blocked - return the modified response
        return True, query_result.modified_response, [{
            'guardrail': query_result.guardrail_type.value,
            'passed': query_result.passed,
            'message': query_result.message,
            'action': query_result.action_taken,
        }]
    
    # Check response
    source_contents = [d.get('content', '') for d in (source_docs or [])]
    final_response, response_results = system.check_response(
        response, source_docs, source_contents
    )
    
    # Build log
    log = []
    for result in response_results:
        log.append({
            'guardrail': result.guardrail_type.value,
            'passed': result.passed,
            'message': result.message,
            'severity': result.severity.value,
        })
    
    return False, final_response, log


if __name__ == "__main__":
    # Quick test
    print("Guardrails Module Test")
    print("-" * 40)
    
    system = GuardrailSystem()
    
    # Test topic classification
    test_queries = [
        "What is the leave policy?",
        "What's the best restaurant?",
        "Can I share my password?",
    ]
    
    for q in test_queries:
        result = system.check_query(q)
        print(f"\nQ: {q}")
        print(f"  Passed: {result.passed}")
        print(f"  Message: {result.message}")
    
    print(f"\nStats: {system.get_stats()}")