# Why PostgreSQL FTS For Agent Projects

## Overview
A simple, clear argument: For AI Agent projects, full-text search belongs inside PostgreSQL.

---

## Section 1: MAIN THESIS (A-01)

**Key Concept**: Stop splitting your data across multiple systems.

**Content**:
Most Agent projects don't need a dedicated ES cluster.

Keep full-text search **inside PostgreSQL** together with all your other business data.

**Visual Element**:
- Type: Large hero block on chalkboard
- Subject: Bold headline with clear statement
- Treatment: Hand-drawn chalk text, large size

**Text Labels**:
- Headline: "PostgreSQL FTS"
- Subhead: "Full-Text Search Inside PostgreSQL For AI Agents"

---

## Section 2: Agent Search Requirements (A-02)

**Key Concept**: Agent search has different requirements compared to traditional search.

**Content**:
• **Always filtered**: Permission checks, tenant isolation, time ranges
• **Mixed data**: Documents + chat history + memory + structured fields
• **Consistency**: No stale results after updates/deletes
• **Simple architecture**: One system instead of two

**Visual Element**:
- Type: Four bullet points with simple hand-drawn icons
- Subject: Key requirements overview
- Treatment: Colored chalk for highlights

**Text Labels**:
- Headline: "Agent Search Requirements"

---

## Section 3: PostgreSQL vs ES (B-01)

**Key Concept**: Architectural comparison.

**Content**:
| Dimension | PostgreSQL | ES |
|-----------|------------|---------------|
| Consistency | ✓ Same transaction | Needs async sync |
| Filtering | ✓ Native SQL | Extra processing |
| Operations | ✓ One database | Extra cluster |
| Hybrid Search | ✓ Vector + BM25 in one DB | Multiple systems |

**Visual Element**:
- Type: Simple hand-drawn comparison table
- Subject: Side-by-side comparison
- Treatment: Green checkmarks for PostgreSQL advantages

**Text Labels**:
- Headline: "PostgreSQL vs ES"

---

## Section 4: Incremental Upgrade Path (B-02)

**Key Concept**: Start simple and incrementally upgrade as your needs grow.

**Content**:
**Layer 1**
Built-in FTS
`tsvector + GIN`

**Layer 2**
pg_tokenizer.rs
Better tokenization

**Layer 3**
VectorChord-BM25
Native BM25 index

**Visual Element**:
- Type: Three connected blocks left-to-right
- Subject: Progressive upgrade path
- Treatment: Different chalk colors for each layer

**Text Labels**:
- Headline: "Incremental Upgrade Path"

---

## Design Instructions

- **ALL TEXT IN ENGLISH** (no Chinese anywhere on the image)
- **All words must be spelled correctly** — no garbled or merged characters
- **chalkboard style**: blackboard background with hand-drawn chalk text and diagrams
- 4 clear sections/modules with plenty of space between them
- Hand-drawn chalk aesthetic, authentic chalkboard texture
- Use yellow/pink/blue/ green chalk for highlights
