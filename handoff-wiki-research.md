# Handoff: Deep Research for Wiki Knowledge Base Design

Paste everything below this line into Claude (claude.ai) for deep research.

---

## Context

I'm building a Hebrew-first AI assistant platform for my auto-parts importing business in Israel. The system uses local GPU inference (Gemma E4B) with a **wiki-first retrieval** approach — agents navigate a topic map, read full wiki pages, and reason over complete documents. No chunk-based RAG as the primary path.

### How the system works
- There's a topic map (~300 tokens) listing all wiki pages by category
- When a user asks a question, the agent first reads the topic map, picks the relevant page(s), reads them in full, then answers
- Each wiki page should be 800-1500 tokens, self-contained, cross-linked with `[[page-name]]`
- The agent has 8K context per slot (Hebrew has ~3x token overhead vs English)
- Vector search exists as a fallback but wiki-first is the primary retrieval path

### My business
- Auto-parts importer in Israel
- Customers: garages (mechanics), parts stores/retailers, private buyers
- Products: brake systems, filters, engine parts, electrical, suspension, etc.
- Suppliers: international (mostly Europe/Asia) and local distributors
- ERP system: Madaf (Israeli DOS-based ERP), data synced to Supabase
- Team: small (~18 employees), sales agents, warehouse staff

### Current data in Supabase (clean, filtered)
- 1,139 active customers (purchased in last 2 years)
- 6,225 active products (sold in last 18 months or in stock)
- 300,271 sale lines across 4 years (2023-2026)
- 35,529 sales documents
- 344 suppliers
- 17,826 stock entries by branch

### Current wiki pages (17 pages, all have PLACEHOLDER content that needs replacing)

**Catalog & Products:**
- brake-systems — types, brands, compatibility
- filters-catalog — oil, air, fuel, cabin filters
- parts-taxonomy — categorization system
- brand-tiers — OEM, premium, standard, budget

**Business Rules:**
- pricing-rules — margin tiers, discounts, minimums
- customer-segments — customer types and terms
- order-fulfillment — order processing flow
- payment-collections — payment terms, credit
- return-warranty — returns and warranty policy

**Market & Sales:**
- common-vehicles-il — popular vehicles in Israel
- sales-patterns — seasonal trends, basket analysis
- competitive-landscape — market position

**Operations:**
- supplier-profiles — active suppliers and terms
- inventory-management — reorder rules, stock levels
- delivery-zones — delivery areas and timelines

**Reference:**
- madaf-workflow — ERP system operations
- hebrew-auto-parts-glossary — Hebrew-English terminology

## What I need from you

I want to design these wiki pages properly — not just dump data in. The agents will use these pages to make real business decisions (pricing, customer advice, inventory recommendations), so accuracy and structure matter enormously.

### Please research and advise on:

1. **Knowledge architecture for domain-specific AI agents**
   - What categories of knowledge should a wholesale/import business assistant have?
   - How should declarative knowledge (facts, rules) vs procedural knowledge (workflows, decision trees) be structured differently?
   - What's the right granularity — when should one page become two?
   - How do you handle knowledge that changes frequently (prices, stock) vs rarely (policies, processes)?

2. **Auto-parts domain expertise**
   - What does an experienced auto-parts salesperson need to know that ISN'T in an ERP?
   - What tribal knowledge typically exists in auto-parts businesses?
   - What are the common failure modes when auto-parts businesses try to codify knowledge?
   - What makes Israeli auto-parts market specifically different (vehicle mix, import regulations, seasonal patterns)?

3. **Wiki page design patterns**
   - Best structure for each page type (pricing rules vs product catalog vs workflow)
   - What should be static in the wiki vs dynamically queried from the database?
   - How to write pages that help an LLM reason correctly (avoid ambiguity, provide decision criteria)
   - Cross-linking strategy — when should an agent need 1 page vs 2-3 pages?

4. **Validation and maintenance**
   - How to validate wiki content before deployment (what questions to test with)
   - How to detect when a wiki page is stale or wrong
   - Who in the organization should own each page type
   - What's the review cadence for different knowledge types

5. **Specific page recommendations**
   - For each of my 17 pages, what sections/structure would you recommend?
   - Which pages should be data-driven (generated from Supabase) vs human-written?
   - Which pages are highest priority (most agent value per effort)?
   - Are there pages I'm missing that a real auto-parts assistant would need?

### Important constraints
- Hebrew is the primary language (agents respond in Hebrew, customers speak Hebrew)
- 800-1500 tokens per page (roughly 300-500 Hebrew words)
- The agent has limited context (8K per slot) — pages must be concise and information-dense
- Wrong information is worse than no information — if the agent gives bad pricing advice or wrong compatibility info, it damages trust
- The wiki is git-tracked and version-controlled

### What I DON'T want
- Generic "AI knowledge base best practices" that apply to any chatbot
- Suggestions to use chunk-based RAG instead (we already have that as fallback)
- Advice about the tech stack (that's decided)
- Templates or boilerplate — I want thinking about what makes THIS domain special

Give me a thorough, specific analysis. Think about what an expert auto-parts business consultant would say if they were designing a knowledge system for AI agents in this exact business.
