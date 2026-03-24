# Parts Finder — Data Verification Plan v2

## Context

1,839 rows of vehicle parts data compiled from Claude's training data. During audit, 8/10 brake pad cross-reference rows checked were completely hallucinated. This plan catalogs every free resource available to verify the data, organized by category and by database.

---

## Master Resource Directory

### Multi-Category Databases (verify EVERYTHING in one place)

| Resource | URL | Free? | What It Covers | Why It Matters |
|----------|-----|-------|----------------|----------------|
| **TecDoc China Catalog** | https://www.tecalliance.cn | FREE | All parts: filters, brakes, bulbs, belts — 1000+ brands, full cross-ref | **SINGLE BEST RESOURCE** — industry standard, vehicle to part lookup with all cross-refs |
| **TecDoc Web Catalog** | https://web.tecalliance.net | Free trial 14 days | Same as above, EU/global version | Register for trial, verify everything in 2 weeks |
| **TecDoc Mobile App** | Google Play / App Store | FREE (basic) | Vehicle to parts lookup, cross-references | Free tier has limits; premium unlocks everything |
| **Autodoc Catalog** | https://www.autodoc.co.uk | FREE | Filters, brakes, bulbs — shows OEM + aftermarket cross-refs per vehicle | Great for spot-checking, organized by vehicle |
| **Spareto** | https://spareto.com | FREE | Any part number to full cross-reference table | Enter ANY number, get all brand equivalents |
| **Plenty.Parts** | https://plenty.parts | FREE | Part number to OEM refs, cross-refs, vehicle fitment | Structured data, good for systematic verification |
| **Parts-Crossreference.com** | https://parts-crossreference.com | FREE | Enter any part number to all brand equivalents | Covers MANN, Filtron, Wix, Fram, Baldwin, etc. |
| **RockAuto** | https://www.rockauto.com | FREE | Vehicle to all parts with brand options | US-focused but covers all makes; shows OEM refs |

### TecDoc API (for programmatic/bulk verification)

| Resource | URL | Free? | Notes |
|----------|-----|-------|-------|
| **TecDoc API (RapidAPI)** | https://rapidapi.com/ronhartman/api/tecdoc-catalog | Freemium | API access to TecDoc data; could script bulk verification |
| **Apify TecDoc Scraper** | https://apify.com/making-data-meaningful/tecdoc | Freemium | Pre-built scraper for TecDoc catalog data |
| **TecAlliance Dev Portal** | https://developer.tecalliance.cn | Free docs | API documentation; China catalog is free to query |

---

### Brake Pads — Dedicated Resources

| Resource | URL | Free? | Coverage | Special Feature |
|----------|-----|-------|----------|-----------------|
| **Textar Brakebook** | https://brakebook.com (select Textar) | FREE | All vehicles, weekly updates | **Expert search by dimensions** — perfect when you only have the old pad |
| **Pagid Brakebook** | https://brakebook.com (select Pagid) | FREE | Same data, Pagid brand | Pagid = Textar = TMD Friction (same company) |
| **Mintex Brakebook** | https://brakebook.com (select Mintex) | FREE | Same data, Mintex brand | Mintex = Textar = TMD Friction |
| **Ferodo Catalog** | https://www.ferodo.com/catalogue | FREE | Vehicle to Ferodo + OEM + TRW/Textar/Brembo/Bosch cross-refs | **One lookup = 6+ brand cross-refs** |
| **Brembo Parts Catalog** | https://www.bremboparts.com/europe/en | FREE | Vehicle to Brembo pads + OEM refs + competitor equivalents | Can search by vehicle OR by plate (UK/EU) |
| **TRW/ZF Aftermarket** | https://www.trwaftermarket.com/en/catalog | FREE | Vehicle to TRW pads, discs, drums, calipers | Shows OEM cross-refs |
| **Bosch Parts Finder** | https://www.boschaftermarket.com/gb/en/diagnostics/parts-finder/ | FREE | Vehicle to Bosch brake parts | Also covers filters and bulbs |
| **ATE/ZF Aftermarket** | https://aftermarket.zf.com/go/en/ | FREE | Vehicle to ATE brake parts | ZF umbrella (includes TRW) |

### Filters — Dedicated Resources

| Resource | URL | Free? | Coverage | Special Feature |
|----------|-----|-------|----------|-----------------|
| **MANN Online Catalog** | https://catalog.mann-filter.com/EU/eng/vehicle | FREE | Vehicle to oil/air/cabin/fuel + OEM cross-refs | **One lookup = oil + air + cabin for same vehicle** |
| **Filtron Catalog** | https://www.filtron.eu/en/catalogue | FREE | Vehicle to Filtron parts + OEM refs | Popular in Israel |
| **Filtron Downloadable XLS** | https://www.filtron.eu (downloads section) | FREE | **Complete filter catalog with vehicle applications as Excel** | **KEY SHORTCUT** — only freely downloadable structured dataset across all filter manufacturers. Use as foundation for bulk cross-referencing. |
| **MAHLE/Knecht Catalog** | https://www.mahle-aftermarket.com/eu/en/e-cat/ | FREE | Vehicle to MAHLE filters + OEM cross-refs | OE supplier to many manufacturers |
| **Hengst Filter Catalog** | https://www.hengst.com/en/catalog | FREE | Vehicle to Hengst filters + OEM refs | OE supplier to VW Group |
| **oilfilter-crossreference.com** | https://www.oilfilter-crossreference.com | FREE | Any oil filter number to 200,000+ cross-refs | **Best for number-to-number** |
| **airfilter-crossreference.com** | https://www.airfilter-crossreference.com | FREE | Same for air filters | Same database family |
| **fuelfilter-crossreference.com** | https://www.fuelfilter-crossreference.com | FREE | Same for fuel filters | Same database family |
| **FilterXRef.com** | https://www.filterxref.com | FREE | 1,000,000+ filter cross-references | Baldwin, Wix, Mann, Donaldson, Fleetguard |
| **MANN Filters R Us** | https://www.mannfiltersrus.com/oem-part-number-lookup | FREE | OEM number to MANN equivalent | Direct OEM to MANN |

### Vehicle Bulbs — Dedicated Resources

| Resource | URL | Free? | Coverage | Special Feature |
|----------|-----|-------|----------|-----------------|
| **OSRAM Lamp Guide** | https://www.osram.com/apps/bulb-finder/ | FREE | Vehicle to every bulb position with ECE type | **70+ brands, all Israeli fleet** |
| **Philips Bulb Finder** | https://www.philips.com/c-m-au/automotive-bulb-finder | FREE | Vehicle to all bulb positions | 60+ manufacturers, from 1958 onward |
| **Ring Automotive** | https://www.ringautomotive.com/bulb-finder | FREE | Vehicle to bulb types | Strong European focus |
| **PowerBulbs** | https://www.powerbulbs.com | FREE | Vehicle to bulb types by position | Shows halogen vs LED variants |
| **BulbCharts.com** | https://bulbcharts.com | FREE | Make/model to all bulb positions | Simple, organized by year |

### Oil Specs — Dedicated Resources

| Resource | URL | Free? | Coverage | Special Feature |
|----------|-----|-------|----------|-----------------|
| **OilSpecifications.org** | https://www.oilspecifications.org | FREE | OEM approval codes, viscosity specs | Best reference for approval code definitions |
| **Castrol Oil Advisor** | https://www.castrol.com/.../which-oil.html | FREE | Vehicle to recommended oil spec + viscosity | Shows OEM approval code |
| **Mobil Oil Advisor** | https://www.mobil.com/.../product-advisor | FREE | Vehicle to recommended oil + spec | Shows viscosity + approval |
| **Total/Elf LubAdvisor** | https://lubadvisor.totalenergies.com | FREE | Vehicle to oil spec + capacity | **Shows oil capacity** |
| **Shell LubeMatch** | https://www.shell.com/.../lubematch.html | FREE | Vehicle to oil recommendation | Spec + capacity |
| **Liqui Moly Oil Guide** | https://www.liqui-moly.com/en/oil-guide.html | FREE | Vehicle to oil spec + capacity | Very detailed |

### OEM Parts Portals (verify OEM part numbers)

| Make | URL | Free? |
|------|-----|-------|
| Toyota | https://autoparts.toyota.com | Yes |
| Hyundai | https://www.hyundaipartsdeal.com | Yes |
| Kia | https://www.kiapartsdeal.com | Yes |
| VW Group | https://parts.vw.com | Yes |
| BMW | https://www.realoem.com | Yes |
| Mercedes | https://mbparts.mbusa.com | Yes |
| Stellantis (PSA/Fiat) | https://public.servicebox.peugeot.com | Yes |
| Renault | https://partsouq.com | Yes |
| Ford | https://parts.ford.com | Yes |
| Mazda | https://www.mazdapartsdeal.com | Yes |
| Subaru | https://parts.subaru.com | Yes |

---

## Priority Order

| Priority | Category | Rows | Risk | Why |
|----------|----------|------|------|-----|
| **P0** | Brake pads | 98 | **SAFETY-CRITICAL** | Proven 80% hallucination rate on cross-refs. Wrong pads = liability. |
| **P1** | Oil filter OEM numbers | 378 | **HIGH** | Wrong oil filter size = oil leak = engine damage |
| **P2** | Oil filter MANN/Filtron cross-refs | 378 | **MEDIUM** | Wrong cross-ref = customer gets wrong filter |
| **P3** | Air + cabin filter cross-refs | 378 | **MEDIUM** | Wrong part, customer frustration |
| **P4** | Oil specs | 512 | **LOW-MED** | Viscosities are standardized; capacities may be off |
| **P5** | Vehicle bulbs | 125 | **LOW** | ECE types are standardized; trim-level variation is main risk |
| **P6** | Model name mappings | 726 | **LOW** | Wrong name does not equal wrong part; test during development |

---

## Verification Strategy Options

### Option A: Manual — Row-by-Row (12h, no coding)

Use Ferodo for brakes, MANN for filters, OSRAM for bulbs, Castrol/Total for oil. 50 unique part shapes per category. Reliable but slow.

### Option B: Semi-Automated — TecDoc China + Script (~4h)

The TecDoc China catalog (tecalliance.cn) is free. Write a scraper that:
1. For each make+model+year in our CSV, queries TecDoc for brake pads / filters / bulbs
2. Extracts the returned part numbers
3. Compares against our CSV and flags mismatches

This could verify hundreds of rows in minutes. The Apify TecDoc actor or RapidAPI TecDoc endpoint can accelerate this further.

### Option C: TecDoc API Trial — Bulk Verify (fastest, 14-day window)

1. Register at shop.tecalliance.net for TecDoc Catalogue Classic (14 days free with credit card)
2. Use the REST API to bulk-query all 1,839 rows
3. Compare TecDoc's authoritative data against our CSVs
4. Export verified dataset in one batch

**Option C is by far the most efficient** — turns 12 hours of manual work into a few hours of scripting.

### Option D: Hybrid — Use Multiple Free Tools in Parallel

Different tools are best for different categories:
- **Brake pads**: Textar Brakebook (brakebook.com) — fastest, vehicle to Textar + Mintex + Pagid + OEM + WVA number
- **Filters**: MANN catalog (catalog.mann-filter.com) — one lookup per vehicle = oil + air + cabin
- **Bulbs**: OSRAM bulb finder — one lookup per vehicle = all positions
- **Oil specs**: Total LubAdvisor — one lookup per vehicle = viscosity + capacity + approval
- **Cross-refs**: oilfilter-crossreference.com / Spareto — enter our OEM number, check if MANN/Filtron match

### Option E: Filtron XLS Shortcut (filters only)

Filtron (owned by MANN+HUMMEL) publishes a downloadable Excel file with their complete filter catalog including vehicle applications and OEM cross-references. This is the **only freely downloadable structured dataset** across all filter manufacturers.

1. Download the Filtron XLS from filtron.eu
2. Script a comparison: for each vehicle in our DB, look up the Filtron match
3. Cross-check OEM numbers and Filtron/MANN part numbers against our CSVs
4. Bulk-flag mismatches

This can verify all 378 filter rows programmatically without any web scraping.

---

## The 5 Tools That Cover Everything

| # | Tool | URL | Covers |
|---|------|-----|--------|
| 1 | **TecDoc** | tecalliance.cn | All categories — filters, brakes, bulbs, everything |
| 2 | **Ferodo Catalog** | ferodo.com/catalogue | Brake pads + full cross-ref chain (TRW/Textar/Brembo/Bosch/Mintex) |
| 3 | **MANN Catalog** | catalog.mann-filter.com | Oil + air + cabin filters with OEM cross-refs |
| 4 | **OSRAM Lamp Guide** | osram.com/apps/bulb-finder | All bulbs per vehicle, 70+ brands |
| 5 | **oilfilter-crossreference.com** | oilfilter-crossreference.com | Instant part-number-to-part-number for 200,000+ filters |

Total free databases identified: **40+**
Total requiring registration: 2 (TecDoc trial, TecDoc API)
Total completely free, no registration: **38+**

---

## Red Flags by Category

### Brake Pads
- Brembo numbers that don't appear on bremboparts.com = likely hallucinated
- TRW GDB numbers: GDB3000+ is newer generation, GDB1000-2000 is older
- Textar numbers always start with 2XXXXXX
- Ferodo road pads always start with FDB (not FDS/FCP which are racing)
- Bosch brake pads always start with 0 986 49X XXX

### Filters
- Toyota oil filters always start with 04152 (cartridge) or 90915 (spin-on)
- Hyundai/Kia oil filters always start with 26300
- VW Group: petrol = 04E/06L prefix, diesel = 03N/03L prefix
- BMW: always 11 42X prefix for oil filters

### Bulbs
- Main risk is trim-level variation (base = halogen, premium = LED)
- An H7 is an H7 regardless of brand — bulb type verification is simpler than part number verification
- "LED" entries mean non-replaceable integrated module (no bulb to sell)

---

## Edge Case Strategy: AI Fallback

For vehicles not covered by the static database:
1. System queries Claude Haiku API (~$0.001/query) for specs
2. AI result is flagged as **unverified** and displayed with a disclaimer
3. Miss is logged for manual verification and DB addition within 24-48h
4. Over time the DB self-improves: every miss becomes a permanent addition

Expected miss rate after initial build: ~10-15%, declining to <5% within 2-3 months of real customer queries.

---

## After Verification: Connect to Tevel Inventory

The verification plan covers data correctness. The next step is mapping verified specs to Tevel's actual product catalog:

- **Oil**: Spec (e.g., '0W-20 API SP') filters the oil product catalog to show matching products
- **Filters**: OEM part number to cross-reference table to MANN/Filtron part numbers to Tevel SKU
- **Brake pads**: OEM part number to cross-reference table to Brembo/TRW/Textar to Tevel SKU
- **Bulbs**: ECE type (e.g., 'H7') to any H7 in Tevel's bulb catalog (universal fit)

This is where the competitive advantage lives — anyone could build the plate-to-spec pipeline, but only Tevel can map specs directly to in-stock products with prices.

---

## Verification Tracking

Update each row's confidence column:
- `VERIFIED` — confirmed via catalog lookup
- `HIGH` — matches training data patterns, not independently verified
- `MEDIUM` — uncertain, needs verification
- `LOW` — conflicting data found, investigate further
- `WRONG` — confirmed incorrect, needs replacement

### Verification log template

For each verification session, record:
```
Date: YYYY-MM-DD
Category: [brake pads / oil filters / air filters / etc.]
Tool used: [Ferodo catalog / MANN catalog / OSRAM bulb finder / etc.]
Rows checked: X
Rows confirmed correct: Y
Rows corrected: Z
Rows unable to verify: W
Notes: [any patterns, issues, or observations]
```

---

## Timeline

| Week | Task | Time | Cumulative |
|------|------|------|------------|
| **Week 1** | P0: Brake pads via Ferodo/Textar Brakebook | 3h | 3h |
| **Week 1** | P1+P2+P3: All filters via MANN catalog (or Filtron XLS shortcut) | 5h | 8h |
| **Week 2** | P4: Oil specs spot-check (top 30 engines) | 2h | 10h |
| **Week 2** | P5: Bulbs via OSRAM lamp guide | 2h | 12h |
| **Ongoing** | P6: Model mappings during development | — | — |

Total dedicated verification time: **~12 hours** spread across 2 weeks.

If using TecDoc API trial (Option C), total time drops to **~4 hours** of scripting + review.
