# DocuMind AI — Secure Multimodal RAG Platform

A secure, cloud-deployed **multimodal Retrieval-Augmented Generation (RAG)** platform for querying enterprise PDF documents containing **text, tables, charts, and images**.

DocuMind AI combines semantic vector search, multimodal LLM generation, persistent cloud storage, authentication, and database-level row security to provide grounded document question answering for multiple users.

---

## 🚀 Live Demo

**Live Application:**  
https://enterprise-grade-rag.streamlit.app/

**GitHub Repository:**  
https://github.com/SaiKrishna680/enterprise-grade-rag

---

## 📌 Overview

Traditional PDF question-answering systems often focus only on extracted text. This can cause important information contained in tables, charts, and images to be ignored.

DocuMind AI addresses this by building a multimodal RAG pipeline that:

- Extracts text and tables from PDFs
- Extracts images from documents
- Generates captions for extracted images
- Creates semantic embeddings using BGE-small
- Stores embeddings in PostgreSQL using pgvector
- Stores original PDFs and images in Supabase Storage
- Retrieves text, tables, and images separately
- Sends retrieved text and actual image pixels to Gemini
- Generates answers grounded only in retrieved document context
- Provides document and page-level source attribution
- Supports multiple users with isolated document access
- Persists data across application redeployments

---

# 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │        USER         │
                         │ Login / Upload / Ask│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      STREAMLIT      │
                         │       app.py        │
                         └─────────┬───────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
              Upload                              Question
                 │                                   │
                 ▼                                   ▼
        ┌─────────────────┐                  ┌─────────────────┐
        │  PDF INGESTION  │                  │ Query Embedding │
        └────────┬────────┘                  │   BGE-small     │
                 │                           └────────┬────────┘
                 ▼                                    │
        ┌─────────────────┐                           ▼
        │   PDF PARSING   │                  ┌─────────────────┐
        │                 │                  │   pgvector RPC  │
        │ Text / Tables   │                  │  match_chunks() │
        │ Images          │                  └───────┬─────────┘
        └────────┬────────┘                          │
                 │                         ┌────────┴────────┐
                 │                         │                 │
                 ▼                         ▼                 ▼
        ┌─────────────────┐           Text/Table          Images
        │ Image Captioning│             Top 5              Top 2
        └────────┬────────┘               │                 │
                 │                        │                 ▼
                 │                        │          Supabase Storage
                 ▼                        │                 │
          BGE Embeddings                  │            Raw Images
                 │                        │                 │
                 └──────────┬─────────────┴─────────────────┘
                            ▼
                    ┌─────────────────┐
                    │ Retrieved Context│
                    │ Text + Tables +  │
                    │ Actual Images    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      Gemini     │
                    │ Multimodal LLM  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Grounded Answer │
                    │  + Sources      │
                    └─────────────────┘
