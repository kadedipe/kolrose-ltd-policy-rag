# tests/test_basic.py
"""
Minimal test suite for Kolrose Limited RAG System.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_configuration():
    """Test that configuration loads correctly"""
    from app.config import COMPANY_INFO, CHUNK_SIZE, RETRIEVAL_K
    
    assert COMPANY_INFO['name'] == 'Kolrose Limited'
    assert 'Abuja' in COMPANY_INFO['address']
    assert CHUNK_SIZE > 0
    assert RETRIEVAL_K > 0


def test_topic_classifier_in_corpus():
    """Test classifier recognizes in-corpus queries"""
    from app.rag_system import TopicClassifier
    
    queries = [
        "What is the leave policy?",
        "How do I request remote work?",
        "What are the password requirements?",
        "How are expenses reimbursed?",
    ]
    
    for q in queries:
        category, _, _ = TopicClassifier.classify(q)
        assert category.value == 'in_corpus', f"'{q}' should be in_corpus, got {category.value}"


def test_topic_classifier_off_topic():
    """Test classifier blocks off-topic queries"""
    from app.rag_system import TopicClassifier
    
    queries = [
        "What is the best restaurant?",
        "Who will win the elections?",
        "What's the weather like?",
    ]
    
    for q in queries:
        category, _, _ = TopicClassifier.classify(q)
        assert category.value == 'off_topic', f"'{q}' should be off_topic, got {category.value}"


def test_topic_classifier_sensitive():
    """Test classifier detects sensitive queries"""
    from app.rag_system import TopicClassifier
    
    queries = [
        "Can I share my password?",
        "How do I bribe someone?",
    ]
    
    for q in queries:
        category, _, _ = TopicClassifier.classify(q)
        assert category.value == 'sensitive', f"'{q}' should be sensitive, got {category.value}"


def test_guardrail_topic_check():
    """Test guardrail topic checking"""
    from app.guardrails import TopicGuardrail
    
    # Should pass
    result = TopicGuardrail.check("What is the leave policy?")
    assert result.passed
    
    # Should block
    result = TopicGuardrail.check("What is the best restaurant?")
    assert not result.passed
    assert result.modified_response is not None


def test_output_length_guardrail():
    """Test output length limiting"""
    from app.guardrails import OutputLengthGuardrail
    
    # Short response
    short = "This is a short answer."
    result = OutputLengthGuardrail.check(short)
    assert result.passed
    
    # Long response
    long_response = "This is a sentence. " * 100
    result = OutputLengthGuardrail.check(long_response)
    assert not result.passed or len(result.modified_response or '') <= OutputLengthGuardrail.HARD_CHAR_LIMIT + 50


def test_app_imports():
    """Test that all app modules can be imported"""
    from app import __version__
    from app.config import API_CONFIG, DB_CONFIG, RAG_CONFIG, GUARDRAIL_CONFIG, COMPANY_CONFIG, APP_CONFIG
    from app.rag_system import KolroseRAG, TopicClassifier, RetrievalEngine
    from app.guardrails import GuardrailSystem, TopicGuardrail, OutputLengthGuardrail, CitationGuardrail
    from app.ingestion import ingest_policies, load_vectorstore, load_embeddings
    
    assert __version__ is not None