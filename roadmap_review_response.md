# ROADMAP Review Response — Corrections & Directives

> **For:** Claude Code roadmap generation
> **From:** Shachar (reviewed with Claude)
> **Date:** 2026-03-16
> **Action:** Apply ALL corrections below before generating `ROADMAP_COMPANY_WEBSITE.md`

---

## Overall Assessment

The draft is solid. The business understanding, tech stack, gap analysis, and dependency mapping are largely correct. There are **5 corrections** that must be applied. None change the fundamental architecture — they improve phasing, reduce risk, and fix incorrect dependencies.

---

## Correction 1: Fix Module 2.8 Dependency

**Problem:** Module 2.8 (Agent Customer View) is listed as depending on 2.6 (Order Creation). This is wrong.

**Why it's wrong:** "Recent orders" in the customer view come from **Madaf sync data** — historical invoices synced from Madaf's export. They are NOT website-created orders. The customer view shows what the customer has bought in the past (from Madaf invoice history), not orders placed through the portal.

**Fix:**
```
2.8 Agent Customer View
  DEPENDS ON: 0.2 (DB Schema), 0.4 (Madaf Sync Parsers)
  NOT: 2.6
```

---

## Correction 2: Split Phase 2 into 2A (Madaf-only) and 2B (TecDoc-dependent)

**Problem:** Phase 2 is treated as a single block, with the entire phase implicitly blocked by the TecDoc API purchase. This hides the fact that the most valuable agent features need ZERO TecDoc dependency.

**Why this matters:** The killer feature — instant price history lookup replacing manual Madaf scrolling — depends only on Madaf sync data. Stock browsing, customer management, and special price requests also need only Madaf data. These can ship immediately after Phase 1, while TecDoc is still being purchased/integrated. This significantly accelerates time-to-value for the agent portal.

**Fix:** Split Phase 2 into two sub-phases:

### Phase 2A — Agent Portal: Madaf-Powered Tools (NO TecDoc dependency)

| # | Module | What | Depends On | Status |
|---|--------|------|------------|--------|
| 2A.1 | Price Lookup & History | Agent UI: search customer+SKU → full price history (date, price, invoice#). Shows last price, tier price, cost price. Agent-scoped to own customers. | 0.2, 0.4 | READY |
| 2A.2 | Stock Browser | Full catalog search (OEM#, SKU, name, brand). Current stock quantities. Category filtering. | 0.2, 0.4 | READY |
| 2A.3 | Special Price Request | Agent requests price for customer+SKU → manager approval notification → saved as new last price if approved → audit trail. | 0.2, 0.4, 2A.1 | READY |
| 2A.4 | Agent Customer View | Agent's assigned customers, contact info, recent orders (from Madaf invoice history), quick-start new order link. | 0.2, 0.4 | READY |

### Phase 2B — Agent Portal: TecDoc Integration & Orders (requires TecDoc + Madaf inbound)

| # | Module | What | Depends On | Status |
|---|--------|------|------------|--------|
| 2B.1 | TecDoc API Client | TecDoc integration with Redis caching (TTL: 24-48h articles, 7d vehicles). Article search by vehicle linkage. License-compliant. | Redis setup | BLOCKED (TecDoc purchase) |
| 2B.2 | OEM Bridge | Mapping engine: TecDoc OEM numbers ↔ Madaf SKUs. Normalized lookup table. Runs on catalog sync. Match rate tracking. Manual override table. | 0.4, 2B.1 | BLOCKED (TecDoc purchase) |
| 2B.3 | Parts Finder Page | Agent UI: license plate → vehicle → TecDoc parts + oil results, filtered to company catalog. Stock shown. "Add to order" links. | 2B.1, 2B.2, 1.2 | BLOCKED (TecDoc purchase) |
| 2B.4 | Order Creation | Build order: select customer, add items, auto-suggest prices (last→tier→cost+margin). Low-stock warnings. Below-threshold flagging → manager approval. Submit → Madaf push. | 2A.1, 2A.2, 0.8 | BLOCKED (Madaf inbound PoC) |

**Key insight:** Phase 2A is entirely READY after Phase 0 completes. Phase 2B is BLOCKED on external dependencies. This means agents get real value (price history, stock, customer view) potentially weeks or months before TecDoc arrives.

---

## Correction 3: Add Module 0.8 — Madaf Inbound Proof-of-Concept

**Problem:** Pushing orders INTO Madaf via UI automation is flagged as "new, unproven work" and a risk, but has no scheduled implementation. Module 2B.4 (Order Creation) is BLOCKED on it, and so is the entire Phase 3 client order flow. If Madaf inbound automation turns out to be infeasible or too fragile, the order flow for both agent and client portals needs redesign.

**Why it must be in Phase 0:** This is the highest-risk technical unknown in the project. Discovering it's broken in Phase 2B is too late — by then the entire portal architecture assumes orders can be pushed to Madaf. Test it early, when there's time to design alternatives.

**Fix:** Add to Phase 0:

| # | Module | What | Review Tier | Status |
|---|--------|------|-------------|--------|
| 0.8 | Madaf Inbound PoC | Prototype: create a single test order in Madaf via UI automation. Validate: which fields must be filled, how to handle errors/timeouts, how to confirm success. Document findings. If infeasible, design alternative order handoff flow (e.g., formatted email/WhatsApp to office staff, CSV batch import). | Tier 3 | READY |

**Fallback plan if Madaf inbound fails:** Orders are submitted via the website and formatted into a structured summary (PDF or message) that office staff manually enter into Madaf. Less elegant but functional. The website still handles the approval workflow — only the final Madaf entry step becomes manual.

---

## Correction 4: Split Module 1.1 into Schema and Data Curation

**Problem:** Module 1.1 (Oil Supplement Database) reads like a single task, but it combines two very different types of work:
- **Schema + tooling** — code task, estimable, can be done in a Claude Code session
- **Data population for 50 engines** — manual research task requiring cross-referencing manufacturer recommendation tools, verifying specs against OEM manuals, matching against Madaf catalog. This is weeks of work, not a coding session.

**Fix:** Split into:

| # | Module | What | Type | Status |
|---|--------|------|------|--------|
| 1.1a | Oil DB Schema & Tooling | Prisma schema for vehicle_oil_specs, oil_products, materialized view. Admin UI or script for adding/editing entries. Verification status tracking (source, date, verified flag). Import script for Madaf oil catalog. | Code task | READY |
| 1.1b | Oil DB Data Population | Research and verify oil specs for top 50 Israeli engine codes. Sources: manufacturer recommendation tools (Motul, Castrol, Shell, Fuchs, TotalEnergies, Liqui Moly), OEM service manuals. Cross-reference against Madaf oil catalog. Every entry must have verification_source and verified_date. | Research task | READY (needs Madaf oil catalog export + data.gov.il fleet analysis for top engines) |

**Note for /implement blocks:** Module 1.1a gets a normal implementation spec. Module 1.1b gets a research protocol spec — it defines the verification workflow, data entry format, and quality gates, but the actual data work is manual.

---

## Correction 5: Add RTL/i18n to Module 0.1 Scaffolding

**Problem:** The scaffolding module doesn't mention RTL or internationalization. Hebrew RTL support is non-trivial, especially in a parts catalog context where:
- Page layout is RTL (Hebrew text flows right-to-left)
- Part numbers, OEM numbers, and brand names are LTR
- Mixed-direction content appears in the same line (e.g., "פילטר שמן 15400-RTA-003 Honda")
- Tables with Hebrew headers and English/numeric data need proper alignment
- shadcn/ui components need RTL-aware configuration

Retrofitting RTL later is painful and creates inconsistencies. Set it up from day one.

**Fix:** Add to Module 0.1 scope:

```
0.1 Project Scaffolding (updated scope):
  - Next.js App Router monorepo
  - Tailwind CSS + shadcn/ui
  - Environment variable management
  - Prisma initial setup
  - i18n framework setup (next-intl or similar)
    - Hebrew (primary), English, Arabic
    - RTL layout as default, LTR for specific content zones
    - Bidirectional text handling utilities (for mixed Hebrew + part numbers)
    - RTL-aware shadcn/ui component configuration
  - Git repo with README and decision log
```

---

## Answers to Your 7 Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | Module breakdown correct? | Split Phase 2 into 2A/2B as described in Correction 2. Split 1.1 into 1.1a/1.1b as in Correction 4. Otherwise granularity is right. |
| 2 | Dependencies right? | Fix 2.8: depends on 0.2 + 0.4, NOT 2.6. See Correction 1. |
| 3 | Phase 4 handling? | Skeleton specs in the roadmap is correct. Keep them clearly marked as future/blocked on real data. Do not exclude. |
| 4 | Inbound Madaf sync? | Add as Phase 0 module (0.8). See Correction 3. Do NOT fold into 2B.4 — it's a prerequisite, not part of order creation. |
| 5 | Missing modules? | Add 0.8 (Madaf Inbound PoC). Add RTL/i18n to 0.1 scope. Notification system stays folded into approval modules (2A.3, 3.3) for now — extract later only if complexity warrants it. |
| 6 | Tech stack locked? | Yes. Next.js App Router + Supabase + tRPC + Prisma + Redis + AWS Israel. Confirmed. |
| 7 | Business logic correct? | Yes. Pricing cascade, order approval flow, and agent scoping are all accurately described. |

---

## Updated Module Count

After corrections: **30 modules across 6 sub-phases**

| Phase | Modules | Status |
|-------|---------|--------|
| Phase 0 — Foundation | 0.1–0.8 (8 modules) | Mostly READY |
| Phase 1 — Oil Finder MVP | 1.1a, 1.1b, 1.2, 1.3, 1.4 (5 modules) | READY (pending Madaf oil export) |
| Phase 2A — Agent Portal: Madaf Tools | 2A.1–2A.4 (4 modules) | READY after Phase 0 |
| Phase 2B — Agent Portal: TecDoc & Orders | 2B.1–2B.4 (4 modules) | BLOCKED (TecDoc + Madaf inbound) |
| Phase 3 — Client Portal | 3.1–3.4 (4 modules) | Depends on Phase 2B |
| Phase 4 — Management Dashboard | 4.1–4.5 (5 modules) | BLOCKED (needs real data) |

---

## Proceed

Apply all 5 corrections, then generate `ROADMAP_COMPANY_WEBSITE.md` with self-contained `/implement` blocks per module.
