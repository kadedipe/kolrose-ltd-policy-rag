# tests/conftest.py
"""Pytest configuration for Kolrose RAG tests."""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set test environment
os.environ['APP_ENV'] = 'testing'
os.environ['OPENROUTER_API_KEY'] = 'sk-or-v1-test-key'
os.environ['LOG_LEVEL'] = 'ERROR'  # Reduce noise during tests
os.environ['ENABLE_GUARDRAILS'] = 'true'

@pytest.fixture
def sample_policy_text():
    """Sample policy text for testing"""
    return """# Kolrose Limited - Test Policy

**Document ID:** KOL-TEST-001
**Version:** 1.0
**Effective Date:** January 1, 2024
**Department:** Test Department

## Section 1: Overview

This is a test policy for unit testing.

## Section 2: Leave Policy

Employees receive 15 days of annual leave per year.

## Section 3: Remote Work

Remote work is allowed up to 2 days per week.
"""

@pytest.fixture
def company_info():
    """Company information fixture"""
    from app.config import COMPANY_INFO
    return COMPANY_INFO