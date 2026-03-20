# VectorChord: Vector Search Inside PostgreSQL For Agent Projects

## Overview
For AI Agent RAG applications, VectorChord brings scalable, low-cost vector search natively inside PostgreSQL, eliminating the need for a separate dedicated vector database.

---

## Section 1: MAIN THESIS (Top-Left)

**Key Concept**: Agent projects don't need a separate vector database.

**Content**:
Keep vector search **inside PostgreSQL** together with your business data.

VectorChord enables billion-scale vector search at 1/5th the cost of dedicated vector databases.

**Visual Element**:
- Type: Large hero block
- Subject: Bold headline on chalkboard
- Treatment: Hand-drawn chalk text

**Text Labels**:
- Headline: "VectorChord"
- Subhead: "Vector Search Inside PostgreSQL"

---

## Section 2: Agent Vector Storage Requirements (Top-Right)

**Key Concept**: Agent projects have different requirements than traditional vector search.

**Content**:
• **Frequent updates**: Knowledge base grows continuously
• **Scalable growth**: From thousands to billions of vectors
• **Cost sensitive**: Early-stage projects can't afford high overhead
• **Data consistency**: Vectors live with business data already in PostgreSQL

**Visual Element**:
- Type: Four bullet points with simple doodle icons
- Treatment: Yellow chalk highlights for keywords

**Text Labels**:
- Headline: "What Agents Need"

---

## Section 3: Comparison (Bottom-Left)

**Key Concept**: VectorChord vs dedicated HNSW vector database for 1B 768-dim vectors.

**Content**:
| Metric | VectorChord<br>IVF + RaBitQ | Dedicated DB<br>HNSW |
|--------|---------------------------|-------------------|
| Storage | ~100 GB | ~1000 GB |
| Build Time | ~20 minutes | >10 hours |
| Memory | <1 GB | >100 GB |
| Monthly Cost | ~$247 | ~$1000+ |

**Build speed is 30x faster, cost is 1/5**

**Visual Element**:
- Type: Simple comparison table
- Treatment: Green chalk checkmarks for VectorChord advantages

**Text Labels**:
- Headline: "VectorChord vs Dedicated DB"

---

## Section 4: Core Technology (Bottom-Right)

**Key Concept**: IVF + RaBitQ quantization enables efficient scaling.

**Content**:
- **IVF inverted index**: Clustered storage for sequential IO
- **RaBitQ quantization**: 4x-32x compression with guaranteed error bounds
- **Columnar layout**: Better cache utilization = faster queries
- **Streamed build**: Build index on disk without loading all data into memory

**Visual Element**:
- Type: Four-point tech summary with bullet points
- Treatment: Blue chalk for technical points

**Text Labels**:
- Headline: "Why It's Faster & Cheaper"

---

## Design Instructions

- **ALL TEXT IN ENGLISH** (no Chinese anywhere on the image)
- **All words must be correctly spelled** — no garbled or merged characters
- **chalkboard style**: blackboard background with hand-drawn chalk text/diagrams
- 4 clear quadrant modules with plenty of space between them
- Authentic chalkboard texture with hand-drawn imperfect lines
- Use yellow/pink/blue/green colored chalk for highlights
