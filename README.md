# DocuMind AI — Secure Multimodal RAG Platform

> A production-oriented multimodal Retrieval-Augmented Generation (RAG) platform for querying enterprise documents containing text, tables, charts, and images.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL%20%7C%20Auth%20%7C%20Storage-green.svg)](https://supabase.com/)
[![pgvector](https://img.shields.io/badge/Vector%20Search-pgvector-purple.svg)](https://github.com/pgvector/pgvector)
[![Gemini](https://img.shields.io/badge/LLM-Gemini-orange.svg)](https://ai.google.dev/)
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-Sentence%20Transformers-yellow.svg)](https://www.sbert.net/)

---

## 🚀 Live Application

**Deployed Application:**  
https://enterprise-grade-rag.streamlit.app/

> The application requires user authentication. Each user's documents are isolated using Supabase Authentication and PostgreSQL Row Level Security (RLS).

---

# 📌 Overview

DocuMind AI is a secure, multimodal RAG platform designed to answer questions from uploaded enterprise documents.

Unlike a basic PDF chatbot that only processes text, DocuMind extracts and retrieves:

- 📄 Text
- 📊 Tables
- 📈 Charts
- 🖼️ Images
- 📝 Image captions

The system combines semantic vector retrieval with multimodal generation to provide grounded answers based only on retrieved document context.

The platform also supports:

- Multi-user authentication
- User-level document isolation
- Persistent document storage
- Persistent vector storage
- Multimodal retrieval
- Source/page attribution
- Cloud deployment
- Storage persistence across redeployments

---

# 🎯 Problem Statement

Traditional document QA systems often have several limitations:

1. They only process text.
2. Important information inside charts and images can be ignored.
3. Local vector databases may not survive cloud redeployments.
4. Local image files can disappear after deployment.
5. Application-level filtering alone is not sufficient for strong multi-user isolation.
6. Answers may be difficult to trace back to their original source.

DocuMind addresses these problems through a cloud-persistent, multimodal RAG architecture.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │        USER          │
                         │ Login / Upload / Ask │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      STREAMLIT       │
                         │       app.py         │
                         └───────┬───────┬──────┘
                                 │       │
                       Upload PDF│       │Question
                                 │       │
                ┌────────────────▼─┐     │
                │    INGESTION     │     │
                │                  │     │
                │ PDF Parsing      │     │
                │ Text/Table       │     │
                │ Image Extraction │     │
                │ Image Captioning │     │
                └────────┬─────────┘     │
                         │               │
              ┌──────────┼──────────┐    │
              │          │          │    │
              ▼          ▼          ▼    │
           Text/      Captions    Images │
           Tables                  │      │
              │          │        │      │
              └─────┬────┘        │      │
                    ▼             ▼      │
              BGE Embeddings   Supabase  │
                    │           Storage   │
                    ▼                    │
           ┌─────────────────┐           │
           │ PostgreSQL +    │           │
           │ pgvector        │           │
           │                 │           │
           │ chunks          │           │
           │ embeddings      │           │
           │ metadata        │           │
           └────────┬────────┘           │
                    │                    │
                    │          ┌─────────▼─────────┐
                    │          │  QUERY EMBEDDING  │
                    │          │    BGE-small      │
                    │          └─────────┬─────────┘
                    │                    │
                    │          ┌─────────▼─────────┐
                    │          │  match_chunks RPC │
                    │          │    pgvector       │
                    │          └──────┬──────┬─────┘
                    │                 │      │
                    │          Text/Table   Images
                    │                 │      │
                    │                 │   Storage
                    │                 │      │
                    │          ┌──────▼──────▼─────┐
                    │          │    RETRIEVED      │
                    │          │ TEXT + RAW IMAGE  │
                    │          └─────────┬─────────┘
                    │                    │
                    │          ┌─────────▼─────────┐
                    │          │      GEMINI       │
                    │          │ Multimodal LLM    │
                    │          └─────────┬─────────┘
                    │                    │
                    └────────────────────▼
                             GROUNDED ANSWER
                              + SOURCES
