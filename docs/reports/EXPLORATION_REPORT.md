# Madaf Data Exploration Report

_Generated: 2026-03-31 | Source: Live Madaf DBF export (2026-03-30) | 63 tables, 163K records, ~140MB_

---

## Executive Summary

MHS has a **complete, well-structured ERP database** with rich data across customers, products, sales, purchasing, payments, and inventory. The data quality is high — Hebrew encoding works (cp1255), relationships are clean, and the core business entities are fully populated.

**Key numbers:**
- 3,473 customers (284 actively buying in 2026)
- 19,286 products (2,665 in stock, 1,399 sold this year)
- ~2,555 sales documents in Q1 2026 = ~6.2M ILS revenue
- Projected annual revenue: ~15.5M ILS
- 31.1M ILS in open customer balances (outstanding debt)
- 344 suppliers, 18 employees, 6,488 part cross-references
- 14 years of historical data directories (2010-2025)

**Bottom line:** This data is more than enough to power the entire Cloudy Claude platform — agent portal, client portal, management dashboard, and NL assistant.

---

## What We Have

### 1. Customer Data (NsfCli) — 3,473 records

| Metric | Value |
|--------|-------|
| Total customers | 3,473 |
| With phone numbers | 2,604 (75%) |
| With email | 610 (18%) |
| With obligo (credit limit) | 530 |
| Blocked (חסום group) | 708 |
| Active buyers (2026) | 284 |
| Payment terms כ (current month) | 2,449 (71%) |
| Payment terms ל (30 days) | 1,021 (29%) |

**Agent assignment is complete:** Every customer has a SALE_EMP. The top agents by customer count: #1 (564 customers), #19 (528), #8 (369), #4 (293), #7 (290).

**Geographic spread:** Customers across Israel — Lod (109), Tel Aviv (84), Holon (84), Ramla (83), Beer Sheva (79), Netanya (74), Haifa (70), Jerusalem (68).

### 2. Product Catalog (NsfAVP) — 19,286 records

| Metric | Value |
|--------|-------|
| Total products | 19,286 |
| In stock (QTY > 0) | 2,665 (14%) |
| Sold in 2026 | 1,399 |
| Sold in 2025 | 1,355 |
| With list price (PRICE) | 18,841 (98%) |
| With cost price (LAST_ALUT) | 16,907 (88%) |
| With 2+ price tiers | 4,329 (22%) |

**Product categories (PROD_TYPE):** BRAKE (2,793), FAIR1/air filters (1,911), FOIL1/oil filters (1,009), AC/מיזוג (817), OIL (517), FUEL1 (493), FAIR2 (490), PLUG (410), FUEL2 (405), BELT (236)... 8,482 products have no category.

**Top vendors (VND_NO):** 110 (5,588 products), TRS (3,116), ZVL (1,464), 400 (917), 252 (899), 149 (884), 295 (846), 600 (760).

**Part cross-references:** 6,488 supersession mappings (old part → new part). Critical for "this part replaces that part" lookups.

### 3. Sales Data (NsfHead + Nsftrns) — 2,555 docs, 17,681 lines (2026)

| Month | Documents | Revenue (ILS) |
|-------|-----------|---------------|
| Jan 2026 | 904 | 2,395,835 |
| Feb 2026 | 902 | 2,345,659 |
| Mar 2026 (partial) | 748 | 1,457,588 |

**Average daily sales:** ~51,659 ILS | **Projected annual:** ~15.5M ILS

**Revenue concentration:**
- Top customer: 482K ILS (8% of total)
- Top 10 customers: ~1.6M ILS (26% of total)
- 284 active customers, but long tail — median customer buys only 14 distinct products

**Best-selling products (by revenue):**
1. MN7701-5A (Mannol 5W30 oil 5L): 365K ILS
2. 26300-35505 (Hyundai/Kia OEM oil filter): 287K ILS
3. MN7701-4A (Mannol 5W30 oil 4L): 195K ILS
4. MN7707-5A (Mannol Energy 5W30 5L): 166K ILS
5. MN7715-5A (Mannol VW 5W30 5L): 159K ILS

**Oil and filters dominate sales.** The top 15 revenue products are almost entirely oil (vendor 600 = Mannol) and OEM oil filters (vendor 402).

### 4. Pricing & Margins

| Metric | Value |
|--------|-------|
| Average gross margin (list price vs cost) | 81.5% |
| Median gross margin | 85.4% |
| Average actual discount (BRUTO→NETO in transactions) | 73.3% |

**How pricing works:** BRUTO (catalog/list price) is an inflated reference. Customers pay NETO, which is typically 70-85% below BRUTO. The "discount" field reflects this gap — it's not a promotional discount but the structural pricing model. 74% of all line items have a "discount" applied.

**Margin by category (highest):** Bulbs (92%), AC/מיזוג (90%), Air filters (88%), Wipers (88%), Fuel filters (86%), Oil filters (84%).
**Margin by category (lowest):** Clutch (61%), Water pumps (61%), Engine parts (61%), Timing (66%), Front suspension (64%).

**Multi-tier pricing:** 4,329 products have PRC2 set (a second price tier), 3,974 have PRC3. PRC4 is unused. This maps to customer pricing tiers (likely: retail, garage, wholesale).

### 5. Financial Data

| Metric | Value |
|--------|-------|
| Open customer debt (all) | 31.1M ILS |
| Largest single debtor | 4.77M ILS |
| Top 5 debtors combined | 14.8M ILS |
| Payment records (2026) | 951 |
| Total payments received | 10.1M ILS |
| Payment methods | Check (543), Transfer (168), Cash (139), Credit card (59) |

**Payment aging data exists** — the 07_invoice_aging.xlsx report already buckets unpaid invoices by 0-30/31-60/61-90/91-120/120+ days.

### 6. Purchasing & Supply Chain

| Metric | Value |
|--------|-------|
| Goods-in documents (2026) | 352 |
| Total purchase value | 4.19M ILS |
| Purchase orders | 586 lines (354 received, 232 pending) |
| Top supplier: SCT-Germany | 1.50M ILS (36% of spend) |
| #2: Guangzhou Hechang | 841K ILS |
| #3: El Saber (Jenin) | 530K ILS |

### 7. Quote/Order Pipeline (Archive)

| Metric | Value |
|--------|-------|
| Quotes (הצעה) | 1,106 (958 open, 148 closed) |
| Orders (הזמנה) | 2,270 (596 open, 1,674 closed) |
| Date range | 2012 to 2026 |

### 8. Reference Data

- **18 employees/agents** with names, phones, branches
- **344 suppliers** with contact info
- **248 product groups/categories**
- **150 accounting codes** (chart of accounts)
- **2 branches** (almost all stock is in branch 1)
- **14 contacts** (customer contact persons)
- **172 address book** entries

---

## What We Can Build

### Tier 1: Ready Today (data is complete and validated)

#### 1.1 Agent Dashboard — "Who do I call and what do I sell them?"
- **Customer 360 view:** Balance, payment history, purchase history, last order date, products bought, agent assignment
- **Open balance alerts:** 411 customers with outstanding debt, top debtors highlighted
- **Payment aging:** Overdue invoices bucketed by days — already computed
- **Customer activity scoring:** Last purchase date, order frequency, product breadth (avg 32.6 products per active customer, median 14)

#### 1.2 Sales Analytics
- **Sales by agent:** Revenue, doc count, customer count per agent — 10 active agents with clear performance differences (Agent #11: 1.48M vs Agent #13: 7.9K)
- **Sales by product:** Revenue, quantity, margin per product
- **Sales by customer:** Revenue ranking, product mix, order frequency
- **Monthly/daily trends:** Jan/Feb/Mar 2026 data shows stable ~2.4M/month
- **Sales by city/region:** Customer geographic data enables territory analysis

#### 1.3 Stock & Inventory
- **Current stock levels:** 2,665 products in stock, by warehouse location
- **Stock value:** Can compute (QTY * LAST_ALUT) for inventory valuation
- **Low stock alerts:** MIN_QTY exists per product for reorder point detection
- **Stock by branch:** NsfPSnif has per-branch data (though almost all is branch 1)

#### 1.4 Product Intelligence
- **Part cross-reference lookup:** 6,488 supersession mappings — "this old part is now this new part"
- **Product profitability:** Margin analysis per product, per category
- **Catalog search:** Hebrew product names, OEM numbers, vendor codes
- **Price tier management:** 3 price tiers per product (PRICE, PRC2, PRC3)

### Tier 2: Needs Computation (data exists, requires analytics pipeline)

#### 2.1 Customer Health / Churn Detection
- **Buying pattern analysis:** Compare CUR_SALE vs PREV_SALE per product per customer
- **Basket gap analysis:** Customer X buys filters but not oil — suggest oil
- **Churn signals:** Customers who bought last year but not this year (compare 2025 vs 2026 data)
- **Purchase frequency decline:** Track order-to-order intervals

#### 2.2 Demand Forecasting
- **Product demand trends:** Monthly sales velocity per product
- **Seasonal patterns:** Need multi-year data (available in year directories 2010-2025)
- **Reorder point optimization:** Current MIN_QTY values vs actual consumption rates

#### 2.3 Supplier Analytics
- **Supplier spend analysis:** SCT-Germany = 36% of all purchasing — concentration risk
- **Cost trend tracking:** LAST_ALUT and AVER_ALUT per product over time
- **Purchasing patterns:** Purchase order status (232 pending PO lines)
- **Supplier lead time:** (needs PO date vs goods-in date correlation)

#### 2.4 Quote-to-Order Funnel
- **Conversion rate:** 1,106 quotes → 2,270 orders in archive — can measure conversion
- **Quote aging:** 958 open quotes — which are stale?
- **Win/loss analysis:** By agent, by customer, by product type

### Tier 3: Needs Multi-Year Data (load from year directories)

#### 3.1 Historical Trend Analysis
- **Year-over-year revenue comparison:** 14 years of data (2010-2025) in separate directories
- **Customer lifetime value:** Full purchase history across years
- **Product lifecycle:** When products were introduced, peak sales, decline
- **Seasonal patterns:** Multi-year same-month comparisons

#### 3.2 Deep ML Models
- **Customer clustering:** RFM (Recency, Frequency, Monetary) segmentation
- **Market basket analysis:** Association rules (what's bought together)
- **Demand forecasting:** Prophet/ARIMA with multi-year seasonality
- **Price elasticity:** How discount changes affect volume

### Tier 4: NL Assistant Capabilities

With the data above loaded into PostgreSQL, the Claude-powered NL assistant can answer:

**Immediate (single-table queries):**
- "מה היתרה של לקוח 300855?" → Customer balance lookup
- "כמה מכרנו החודש?" → Monthly sales total
- "מה המלאי של מסנן שמן טויוטה?" → Stock lookup by product name
- "מי הלקוחות של אסף?" → Agent customer list

**Cross-table (joins):**
- "מי הלקוח הכי גדול של כל סוכן?" → Agent top customers
- "איזה לקוחות קנו שמן אבל לא מסנני שמן?" → Basket gap query
- "מה הרווח הגולמי על מסננים?" → Product category margin
- "איזה לקוחות לא קנו מינואר?" → Churn detection

**Analytical (aggregation + logic):**
- "השווה מכירות ינואר מול פברואר לפי סוכן" → Period comparison
- "מי הלקוחות עם החוב הגבוה ביותר שהזמנו אצלם?" → Balance vs. recent activity
- "איזה מוצרים המלאי שלהם נמוך מהמינימום?" → Reorder alerts
- "מהו שיעור ההמרה מהצעה להזמנה?" → Funnel analysis

---

## Data Gaps & Quality Issues

### Missing / Sparse Data

| Gap | Impact | Mitigation |
|-----|--------|------------|
| **CLI_TYPE (customer type) is 99.9% empty** | Can't distinguish garages / shops / importers from data alone | Need manual classification or inference from purchase patterns |
| **MACH_TYPE (vehicle make) 93% empty** | Product-to-vehicle mapping is sparse | TecDoc API will fill this gap for parts lookup |
| **SPK_NO (supplier link) is 0% populated** | Can't link products to suppliers via this field | Use goods-in documents (כניסה) to infer product-supplier links |
| **PRC4 (4th price tier) unused** | Only 3 price tiers active | Not a real gap — 3 tiers is enough |
| **No customer email for 82%** | Can't do email marketing | WhatsApp via phone numbers (75% have phones) |
| **No vehicle info per customer** | Can't pre-filter parts by customer's vehicles | Phase 2 feature: build vehicle-customer associations over time |

### Data Quality Notes

| Issue | Details |
|-------|---------|
| **DISCOUNT field is misleading** | Shows 83.7% average — this is BRUTO→NETO gap, not a promotional discount. The real customer pricing is in NETO. |
| **העברה (transfers) dominate NsfHead** | 58,428 of 62,033 docs are year-opening balance carry-forwards. Must filter these out for any sales analysis. |
| **Single-year working database** | Current data is 2026 only. Historical years exist as separate directories but aren't loaded yet. |
| **Archive (Q tables) contain quotes/orders, not sales** | NsfQHead/NsfQTrns are the quote/order pipeline, not historical sales. Historical sales are in year directories. |
| **8,482 products without PROD_TYPE** | 44% of products have no category. These may be older or less-used items. |
| **Negative margins on some products** | A few products show cost > price — likely data entry errors or loss leaders. Filter these for reporting. |

### What's Needed for Full Platform

1. **Multi-year data load:** Copy year directories (2010-2025) to enable historical analysis, YoY comparison, and proper ML training
2. **Customer type classification:** Manual or rule-based assignment of customer types (garage, shop, importer, individual)
3. **Product categorization cleanup:** 8.5K products without PROD_TYPE need categories
4. **Contact enrichment:** Build customer contact data (phone, email, WhatsApp) from transaction interactions

---

## Existing Work (Already Done)

The previous session produced significant deliverables:

| Asset | Location | Description |
|-------|----------|-------------|
| Table map (markdown) | `docs/madaf_table_map.md` | Complete mapping of all 63 tables with fields, relationships, doc types |
| Table map (JSON) | `docs/madaf_table_map.json` | Machine-readable field schemas for all tables |
| Converted data (XLSX) | `DATA/converted/` | All 65 DBFs exported to Excel for manual inspection |
| CSV exports | `DATA/csv_export/` | 21 key tables exported to CSV |
| Tiered data | `DATA/tiered/hot\|warm\|cold/` | Data categorized by update frequency |
| Business reports | `DATA/reports/` | 10 pre-built Excel reports (balances, sales, stock, aging, etc.) |
| Client profiles | `DATA/reports/client_profiles.csv` | Customer activity scoring |
| Inspector tool | `tools/madaf/madaf_inspect.py` | Deep inspection tool for individual DBF tables |

---

## Recommended Next Steps

### Immediate (This Week)
1. **Load multi-year data** — Copy 2024 and 2025 year directories to enable YoY comparison
2. **Customer type classification** — Rule-based: if customer buys >X products/month → garage, if bulk oil → shop, etc.
3. **Set up PostgreSQL target** — Create the normalized schema from the plan's Phase 2

### Short-term (This Month)
4. **Build the ETL pipeline** — DBF → PostgreSQL using the tiered sync config
5. **Agent dashboard MVP** — Customer 360, open balances, recent orders (uses Tier 1 data)
6. **NL assistant prototype** — Schema docs + Claude API for natural language queries

### Medium-term (Next Quarter)
7. **Basket gap analysis** — Which products should each customer be buying but isn't?
8. **Churn detection** — Flag customers whose purchasing is declining
9. **Demand forecasting** — Using multi-year historical data
