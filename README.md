# Production RAG Platform

A production-oriented Retrieval-Augmented Generation (RAG) platform focused on reliable retrieval, reranking, grounding, evaluation, security, observability, and deployment of LLM-powered applications.

The system combines hybrid BM25 + dense retrieval, cross-encoder reranking, relevance grading, local LLM generation, grounding verification, LangGraph orchestration, FastAPI APIs, SSE streaming, a React frontend, Docker deployment, and automated evaluation.

---

## Overview

This project implements an end-to-end RAG pipeline designed around measurable retrieval and generation quality rather than relying solely on LLM output.

The implemented pipeline consists of:

1. Query transformation
2. Hybrid retrieval
   - BM25 lexical retrieval
   - Dense vector retrieval
3. Reciprocal Rank Fusion (RRF)
4. Cross-encoder reranking
5. Relevance grading
6. Context selection
7. Local LLM generation
8. Grounding verification
9. Retry handling
10. Evaluation and observability

The application is exposed through a FastAPI backend and a React + TypeScript frontend.

---

## Architecture

```text
User Query
    |
    v
Query Transformation
    |
    v
Hybrid Retrieval
(BM25 + Dense)
    |
    v
RRF Fusion
    |
    v
Cross-Encoder Reranking
    |
    v
Relevance Grading
    |
    v
Context Selection
    |
    v
Qwen LLM Generation
    |
    v
Grounding Check
    |
    v
Final Answer