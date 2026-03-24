# Oil Specs DB Expansion — Web Claude Handoff

## Goal
Compile oil specifications for the **top 50 most common Israeli vehicles** with **real engine codes** (not descriptions like "1.6 Petrol").

## Why This Matters
Our DB currently stores engine descriptions ("1.6 Petrol") as engine codes. But the Israeli government API returns real codes like "G4FD", "2ZR-FE", "N20B20". This mismatch means oil lookups fail at the exact-match tier and fall back to imprecise brand defaults.

## Output Format
Fill a CSV file with these exact columns (see template at `data/seed/oil_specs_expanded_template.csv`):

```
make,model,year_from,year_to,engine_code,fuel_type,oil_viscosity,oil_capacity_l,oil_spec,oil_oem_approval,oil_change_interval_km
```

### Column Rules
| Column | Format | Example | Notes |
|--------|--------|---------|-------|
| make | Title Case English | `Toyota` | Must match DB: Toyota, Hyundai, Kia, Mazda, Skoda, Volkswagen, Suzuki, BMW, Mercedes-Benz, Nissan, Honda, Subaru, Peugeot, Citroen, Renault, Fiat, Mitsubishi |
| model | Title Case English | `Corolla` | Standard English model name |
| year_from | Integer | `2014` | Start of generation |
| year_to | Integer | `2018` | End of generation |
| engine_code | Manufacturer code | `2ZR-FE` | **CRITICAL: Use the real engine code, NOT descriptions.** This is the `degem_manoa` value from the gov API |
| fuel_type | Lowercase | `petrol` | One of: `petrol`, `diesel`, `hybrid`, `phev`, `electric` |
| oil_viscosity | SAE format | `5W-30` | With dash |
| oil_capacity_l | Float | `4.2` | Liters with filter. Use `0.0` if unknown |
| oil_spec | API/ACEA/ILSAC | `API SP, ACEA A5/B5` | Comma-separated |
| oil_oem_approval | OEM spec | `VW 504.00/507.00` | Leave empty if none |
| oil_change_interval_km | Integer | `15000` | Manufacturer recommended km |

## Priority Vehicles (Israeli Market)
Research these first — they represent ~70% of the Israeli fleet:

### Tier 1 — Highest Priority (60% of market)
1. **Toyota**: Corolla (2008-2024), Yaris (2010-2024), RAV4 (2013-2024), Camry (2012-2024), C-HR (2017-2024), Land Cruiser (2010-2024)
2. **Hyundai**: i20 (2012-2024), i30 (2012-2024), Tucson (2010-2024), Kona (2018-2024), Ioniq (2017-2024), Santa Fe (2013-2024)
3. **Kia**: Sportage (2010-2024), Picanto (2012-2024), Ceed (2013-2024), Niro (2017-2024), Sorento (2013-2024), Stonic (2018-2024)

### Tier 2 — Important (20% of market)
4. **Mazda**: 3 (2014-2024), CX-5 (2013-2024), 2 (2015-2024), CX-30 (2020-2024)
5. **Skoda**: Octavia (2013-2024), Fabia (2015-2024), Karoq (2018-2024), Kodiaq (2017-2024)
6. **Volkswagen**: Golf (2013-2024), Polo (2014-2024), Tiguan (2016-2024), T-Roc (2018-2024)
7. **Suzuki**: Swift (2012-2024), Vitara (2015-2024), SX4 S-Cross (2014-2024), Baleno (2016-2024)

### Tier 3 — Cover if time allows (10% of market)
8. **Nissan**: Qashqai (2014-2024), Juke (2015-2024), X-Trail (2014-2024)
9. **Honda**: Civic (2012-2024), Jazz/Fit (2014-2024), HR-V (2015-2024)
10. **Mitsubishi**: Outlander (2013-2024), ASX (2013-2024), L200 (2015-2024)

## Research Sources (in priority order)
1. **Castrol Product Finder** — https://www.castrol.com/en_gb/united-kingdom/home/car-engine-oil-702702702702702/car-engine-oil-702702702702702702702/oil-selector-702702702702702702702702702702.html
   - Select make → model → year → engine → get oil spec
2. **Shell LubeMatch** — https://www.shell.com/motorist/oils-lubricants/lubematch.html
   - Very comprehensive, includes capacity
3. **LIQUI MOLY Oil Guide** — https://www.liqui-moly.com/en/service/oil-guide.html
   - Strong for European + Israeli market
4. **Mobil Advisor** — https://www.mobil.com/en/lubricants/for-personal-vehicles/our-products/auto-oil-advisor
   - Uniquely includes oil capacity + filter

## How to Find Real Engine Codes
The engine code must match what the Israeli government API returns in the `degem_manoa` field. Common patterns:
- **Toyota**: `1ZR-FE`, `2ZR-FE`, `2ZR-FXE` (hybrid), `M20A-FKS`, `A25A-FKS`
- **Hyundai/Kia**: `G4FA`, `G4FD`, `G4FJ`, `G4FP`, `D4FD` (diesel), `G4LD`
- **VW/Skoda**: `CZCA`, `CZDA`, `DFGA`, `DKRF`, `CZEA`
- **Mazda**: `PE-VPS`, `PY-VPS`, `SH-VPTS`
- **BMW**: `N20B20`, `B48B20`, `N47D20`
- **Mercedes**: `M274`, `M264`, `OM654`
- **Suzuki**: `K12C`, `K14C`, `M16A`
- **Nissan**: `HR12DE`, `MR20DD`, `HR13DDT`

If you can't find the engine code from the oil advisor, try:
- Wikipedia article for the model (usually lists engine codes per generation)
- The manufacturer's specs page

## Generation Splits
Each row covers a **model generation** (year_from to year_to). Use the standard generation boundaries:
- Don't use a single 2008-2024 range — split by generation (facelift/redesign years)
- Example: Corolla E170 (2014-2018), Corolla E210 (2019-2024)

## Verification
After completing, the data can be imported with:
```
python scripts/import_data.py --db parts_finder.db --specs data/seed/oil_specs_expanded.csv
```
Then verified with:
```
python scripts/diagnose_lookup.py --plate 4552333
```
(Should show Tier 1 or Tier 2 match instead of brand_default)

## Save As
`parts-finder/data/seed/oil_specs_expanded.csv`
