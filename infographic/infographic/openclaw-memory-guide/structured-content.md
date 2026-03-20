# OpenClaw 记忆方案选型指南
## OpenClaw Memory Solutions Selection Guide

## Overview

本文是 OpenClaw Agent 生态中五种记忆方案的完整对比指南。AI 记忆分为三个不同维度（单会话内、跨会话、外部知识库），不同方案解决不同问题。本文帮你快速找到适合自己需求的组合方案。

This infographic compares five memory solutions in the OpenClaw Agent ecosystem. AI memory has three dimensions (intra-session, cross-session, external knowledge base), and each solution solves different problems. Choose the right combination for your needs.

## Learning Objectives
The viewer will understand:
1. The three different dimensions of AI memory and what problems each solves
2. Key features of each of the five main OpenClaw memory solutions
3. The difference between basic `memory-lancedb` and enhanced `memory-lancedb-pro`
4. How to select the right combination based on your use case

---

## Section 1: 记忆的三个维度 / Three Dimensions of AI Memory

**Key Concept**: AI memory addresses three fundamentally different problems.

**Content**:
| 维度 / Dimension | 问题 / Problem | 解决思路 / Approach |
|------------------|----------------|---------------------|
| **单会话内记忆 / Intra-session Memory** | 对话太长超出上下文窗口，截断会丢信息 | 压缩旧消息，保持上下文窗口内不超限 |
| **跨会话记忆 / Cross-session Memory** | 重启/换设备后，AI 忘记之前的决定 | 持久化存储，需要时自动召回相关内容 |
| **外部知识库 / External Knowledge Base** | 个人笔记/文档需要被 AI 搜索引用 | 建立索引，支持混合搜索召回 |

**Visual Element**:
- Type: Three colored rectangular modules arranged horizontally
- Subject: Three problem-solution pairs
- Treatment: Each dimension in its own colored box with clear header and content table

**Text Labels** (bilingual):
- Headline: "记忆的三个维度 / Three Dimensions of Memory"
- Labels: "单会话内 / Intra-session", "跨会话 / Cross-session", "外部知识库 / External Knowledge"

---

## Section 2: 方案一 / Solution 1: lossless-claw

**Key Concept**: Single-session lossless compression that preserves all original information.

**Content**:
- **定位**: 替换 OpenClaw 原生滑动窗口截断
- **核心算法**: LCM (Lossless Context Management) - DAG 分层压缩
- **关键特性**: 原始信息永远保留在 SQLite，需要时可 `lcm_grep` / `lcm_expand` 搜索找回
- **开箱即用**: 安装后替换 `contextEngine` 插槽自动工作
- **适合**: 你经常有长对话，希望超出上下文窗口也不丢原始信息，纯本地无需额外服务

**Visual Element**:
- Type: Dedicated module with rounded corners
- Subject: Solution card with feature checkmarks
- Treatment: Icon representing layered compression with DAG diagram

**Text Labels** (bilingual):
- Headline: "lossless-claw"
- Subhead: "单会话无损压缩 / Single-session Lossless Compression"
- Features: "DAG 分层压缩", "原始信息全保留", "纯本地 开箱即用"

---

## Section 3: 方案二 / Solution 2: qmd

**Key Concept**: OpenClaw native hybrid search memory backend.

**Content**:
- **定位**: OpenClaw 原生集成的本地混合搜索后端
- **核心特性**: BM25 关键词 + 向量语义 + LLM 重排序 三重混合搜索
- **架构**: 三个 GGUF 模型（嵌入+重排序+查询扩展）全栈本地运行，不需要云服务
- **自动降级**: qmd 故障时自动降级到内置 SQLite 索引器
- **适合**: 需要 OpenClaw 原生集成，有大量 Markdown 笔记需要搜索，偏爱纯本地方案

**Visual Element**:
- Type: Dedicated module with rounded corners
- Subject: Solution card with three-layer search pipeline visualization
- Treatment: Three connected boxes showing the hybrid search flow

**Text Labels** (bilingual):
- Headline: "qmd"
- Subhead: "原生混合搜索后端 / Native Hybrid Search Backend"
- Features: "BM25+向量+重排序", "全栈本地运行", "OpenClaw 原生集成"

---

## Section 4: 方案三 / Solution 3: Nowledge Mem

**Key Concept**: Local-first structured cross-session memory.

**Content**:
- **定位**: 本地优先的结构化知识记忆
- **核心特性**: 每条记忆标记类型（事实/决策/偏好/计划），知识演化保留版本历史
- **全自动**: `autoRecall` 会话开始自动插入，`autoCapture` 会话结束自动保存
- **架构**: 默认本地存储，不需要云账户，隐私可控；Cursor/Claude/OpenClaw 多工具共享
- **适合**: "AI 下次会话就忘了我做过什么决定" 是你的痛点，希望 AI 记住偏好和决策，偏爱本地优先

**Visual Element**:
- Type: Dedicated module with rounded corners
- Subject: Solution card showing structured memory organization
- Treatment: Different colored boxes for different memory types (facts/decisions/preferences)

**Text Labels** (bilingual):
- Headline: "Nowledge Mem"
- Subhead: "本地结构化跨会话记忆 / Local Structured Cross-session Memory"
- Features: "结构化记忆分类", "自动召回 自动保存", "本地优先 多工具共享"

---

## Section 5: 方案四 / Solution 4: mem9.ai

**Key Concept**: Cloud-hosted persistent memory infrastructure.

**Content**:
- **定位**: 云端托管的持久化记忆基础设施
- **核心特性**: 零运维，开箱即用，不需要自己搭建服务
- **渐进式混合搜索**: 从纯关键词开始，自动升级混合搜索，不需要重建索引
- **开源**: Apache 2.0，支持自托管
- **适合**: 多设备使用 OpenClaw 需要记忆同步，不想折腾基础设施，接受云存储

**Visual Element**:
- Type: Dedicated module with rounded corners
- Subject: Solution card with cloud icon representing cloud hosting
- Treatment: Cloud graphic with connected devices showing sync

**Text Labels** (bilingual):
- Headline: "mem9.ai"
- Subhead: "云端持久化基础设施 / Cloud Hosted Persistent Infrastructure"
- Features: "零运维 开箱即用", "跨设备跨 Agent 共享", "支持自托管"

---

## Section 6: 方案五 / Solution 5: memory-lancedb-pro

**Key Concept**: Enhanced LanceDB hybrid retrieval memory layer with full pipeline.

**Content**:
- **定位**: OpenClaw 生态功能最完整的 LanceDB 记忆方案
- **架构**: 嵌入式零服务，完全本地运行，不需要独立数据库进程
- **完整检索管线**: 向量 → BM25 → RRF 融合 → 重排序 → 时效加权 → 噪声过滤 → MMR 多样性去重
- **原生多模态**: 支持存储检索文本、图片、音频、视频
- **适合**: 需要完整端到端记忆检索体验，偏爱纯本地嵌入式架构，需要高级检索特性

**Visual Element**:
- Type: Dedicated module with rounded corners (highlighted to emphasize newest solution)
- Subject: Solution card with feature list showing the complete retrieval pipeline
- Treatment: Highlighted border to draw attention

**Text Labels** (bilingual):
- Headline: "memory-lancedb-pro"
- Subhead: "增强型 LanceDB 混合检索 / Enhanced LanceDB Hybrid Retrieval"
- Features: "完整检索管线", "嵌入式零服务", "原生多模态支持"

---

## Section 7: 功能对比 / Feature Comparison: LanceDB

**Key Concept**: Comparison between built-in basic `memory-lancedb` and enhanced `memory-lancedb-pro`.

**Content**:
| Feature | Built-in `memory-lancedb` | `memory-lancedb-pro` |
|---------|---------------------------|----------------------|
| Vector search | ✅ | ✅ |
| BM25 full-text retrieval | ❌ | ✅ |
| Hybrid fusion (Vector + BM25) | ❌ | ✅ |
| Cross-encoder reranking | ❌ | ✅ |
| Recency boost / time decay | ❌ | ✅ |
| MMR diversity deduplication | ❌ | ✅ |
| Multi-scope access isolation | ❌ | ✅ |
| Result noise filtering | ❌ | ✅ |
| Adaptive retrieval | ❌ | ✅ |
| Complete management CLI | ❌ | ✅ |
| Any OpenAI-compatible Embedding | Limited | ✅ |

**Visual Element**:
- Type: Comparison table with checkmarks and crosses
- Subject: Feature matrix comparing two versions
- Treatment: Green checkmarks for supported features, red X for unsupported

**Text Labels** (bilingual):
- Headline: "功能对比 / Feature Comparison: Built-in vs Pro"
- Column labels: "Feature", "Built-in", "Pro"

---

## Section 8: 一分钟选型对照表 / One-Minute Selection Guide

**Key Concept**: Quick reference matching your needs to the recommended solution.

**Content**:
| Your Need | Recommended Solution |
|-----------|-----------------------|
| 长对话单会话内不丢信息 / Long conversation, don't lose info | **lossless-claw** (Required) |
| 跨会话记住决定偏好 / Remember decisions across sessions | **Nowledge Mem** (recommended for local) |
| 多设备同步 + 零运维 / Multi-device sync + zero ops | **mem9.ai** |
| 搜索个人 Markdown 笔记文档 / Search personal Markdown notes | **qmd** |
| 嵌入式本地向量搜索 / Embedded local vector search | **memory-lancedb-pro** |
| 全都想要 / Want everything | **Install all** (they work well together) |

**Visual Element**:
- Type: Quick reference table
- Subject: Decision matrix for quick selection
- Treatment: Bold text for recommended solutions, clean readable table

**Text Labels** (bilingual):
- Headline: "一分钟选型 / One-Minute Selection"
- Column labels: "你的需求 / Your Need", "推荐方案 / Recommended"

---

## Section 9: 推荐组合方案 / Recommended Combinations

**Key Concept**: Most users combine multiple solutions for complete coverage.

**Content**:
1. **个人开发者/重度笔记用户（完全本地）**:
   ```
   lossless-claw + qmd/LanceDB
   ```
   - lossless-claw: 单会话内无损压缩
   - qmd/LanceDB: OpenClaw 原生混合搜索

2. **需要跨会话记忆**:
   ```
   lossless-claw + Nowledge Mem + qmd/LanceDB
   ```
   - Adds structured cross-session memory to the above

3. **多设备流动用户**:
   ```
   lossless-claw + mem9.ai + qmd/LanceDB
   ```
   - Cloud sync across all your devices

**Visual Element**:
- Type: Three combination boxes with ASCII-style tree diagram
- Subject: Three common configuration patterns
- Treatment: Indented hierarchy showing the combination structure

**Text Labels** (bilingual):
- Headline: "推荐组合 / Recommended Combinations"
- Labels: "完全本地 / Full Local", "跨会话记忆 / Cross-session", "多设备同步 / Multi-device"

---

## Data Points (Verbatim)

### Key Quotes
- "OpenClaw 为什么总失忆？一套从单会话到永久记忆的 Memory 方案" — Original article title

### Key Terms
- **LCM (Lossless Context Management)**: DAG layered compression algorithm that preserves all original messages
- **Hybrid Search**: Combination of BM25 keyword search + vector semantic search + LLM reranking
- **DAG**: Directed Acyclic Graph - structure for layered incremental compression

---

## Design Instructions

### Style Preferences
- Style: pop-laboratory (technical blueprint with coordinate grid)
- Bilingual: Display both Chinese and English text
- Technical precision: Clear grid lines, coordinate markers, clean engineering aesthetic

### Layout Preferences
- Layout: dense-modules (high-density modules)
- Structure: 3x3 grid of modules:
  - Top row: Section 1 (three dimensions) spans all three columns
  - Middle row: Five solutions in two rows (3 + 2 modules)
  - Bottom row: Comparison table + selection guide + recommended combinations

### Other Requirements
- Target aspect ratio: landscape (16:9) for blog embedding
- Keep all technical terms exact as in source
- Use color coding to differentiate solution categories
