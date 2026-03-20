---
title: "Why PostgreSQL FTS is ideal for full-text search in Agent projects"
topic: technical
data_type: comparison-architecture
complexity: complex
point_count: 7
source_language: zh
user_language: en
---

## Main Topic

This article makes the case that for most AI Agent projects, PostgreSQL full-text search (FTS) with modern extensions like `pg_tokenizer.rs` and `VectorChord-BM25` is a better architectural choice than Elasticsearch. It presents a three-layer evolutionary path from built-in FTS to BM25 sparse retrieval, all within the PostgreSQL ecosystem.

## Learning Objectives
After viewing this infographic, the viewer should understand:
1. Why full-text search requirements differ between traditional web search and Agent projects
2. The key advantages of keeping full-text search in PostgreSQL compared to using Elasticsearch
3. The three-layer evolutionary upgrade path for PostgreSQL FTS
4. How the end-to-end pipeline from tokenization to BM25 indexing works
5. When to use which layer based on project requirements

## Target Audience
- **Knowledge Level**: Intermediate to Expert
  - Developers building AI Agent systems or RAG applications
  - Architects choosing data infrastructure for AI projects
- **Context**: Evaluating whether to use Elasticsearch or PostgreSQL for full-text search in an Agent project
- **Expectations**: Clear comparison, practical upgrade path, and architectural tradeoffs

## Content Type Analysis
- **Data Structure**: Multiple interconnected modules including: core thesis, comparison table, three-layer architecture, end-to-end pipeline, recommendation by use case
- **Key Relationships**: PostgreSQL vs Elasticsearch comparison across multiple architectural dimensions; evolutionary progression from simpler to more capable FTS layers
- **Visual Opportunities**:
  - Side-by-side comparison table between PostgreSQL FTS route and Elasticsearch route
  - Flow diagram showing the three-layer upgrade path
  - End-to-end write/read pipeline diagram for VectorChord-BM25
  - Decision matrix for which layer to choose based on requirements
  - Architecture diagram showing how everything stays within PostgreSQL

## Key Data Points (Verbatim)
- "Most Agent projects, I still prioritize PostgreSQL over Elasticsearch for full-text search"
- "Agent full-text search is almost always combined with filtering conditions"
- "In Agent projects, consistency, simplified architecture, and lower operations surface are more important than theoretical peak performance of a dedicated search system"
- "Four problems with Elasticsearch for most Agent projects: dual-write synchronization, consistency complexity, filtering logic split, higher operations cost"
- "Three-layer evolution path: PostgreSQL built-in FTS → pg_tokenizer.rs → VectorChord-BM25"
- "pg_tokenizer.rs: configurable text processing pipeline with character filters, pre-tokenizer, token filters, custom models"
- "VectorChord-BM25: native BM25 inverted index in PostgreSQL with Block-WeakAND top-k pruning"
- "WeakAND idea: If the theoretical maximum score of a posting block can't even pass the current top-k threshold, skip the entire block"

## Layout × Style Signals
- Content type: Technical comparison and architecture guide → dense-modules (prefconfigured)
- Tone: Technical, precision-focused, practical engineering → pop-laboratory (prefconfigured)
- Audience: Engineers building AI systems → technical precision aesthetic works well
- Complexity: Complex with 7+ distinct points → dense-modules layout is ideal for high-density information

## Design Instructions (from user input)
- All text on the infographic must be in English (user explicitly requested "注意图上全英文" = all text English)

## Recommended Combinations
1. **dense-modules + pop-laboratory** (Recommended): Already configured in project preferences. Dense modules fits this technical guide with multiple comparison points, and pop-laboratory's coordinate blueprint style matches the technical precision perfectly.
2. **dense-modules + technical-schematic**: Alternative engineering blueprint style, more traditional technical diagram feel, darker blueprint background.
3. **comparison-matrix + corporate-memphis**: Clean flat design focused on the side-by-side comparison, better if emphasizing the PostgreSQL vs Elasticsearch decision.
