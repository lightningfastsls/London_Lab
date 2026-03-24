# Parts Finder — Cross-Reference Verification Handoff

## What This Is

You are being asked to verify aftermarket part number cross-references for a vehicle spare parts lookup system (Parts Finder). The data was compiled from Claude's training data and contains **known hallucinations** — during audit, 8 out of 10 brake pad cross-reference rows were completely fabricated.

## What Needs To Be Done

We have CSV files with vehicle parts data. Each row has an OEM part number plus aftermarket equivalents from brands like Brembo, TRW, Textar, Ferodo, Bosch, Mintex (brake pads) and MANN, Filtron (filters). **The aftermarket cross-references need to be verified against real manufacturer catalogs.**

## Files To Verify

All files are in `parts-finder/data/seed/`:

| File | Rows | Category | Priority |
|------|------|----------|----------|
| `brake_pads_specs.csv` | 98 | Brake pad cross-refs (6 brands) | **P0 — SAFETY-CRITICAL** |
| `filter_specs.csv` | 190 | Oil/air/cabin filter cross-refs (MANN, Filtron) | P1 |
| `stage1_filter_specs.csv` | 52 | Same | P1 |
| `expansion_filter_specs.csv` | 96 | Same | P1 |
| `stage2_filter_specs.csv` | 40 | Same | P1 |

## CSV Structure

### Brake Pads (`brake_pads_specs.csv`)
```
make,model,year_from,year_to,platform,axle,disc_diameter_mm,oem_part_number,brembo,trw,textar,ferodo,bosch,mintex,confidence,notes
Toyota,Corolla,2013,2019,E180,front,275,04465-02410,P83155,GDB3591,2536101,FDB4612,0986495348,MDB3416,HIGH,...
```

Columns to verify: `brembo`, `trw`, `textar`, `ferodo`, `bosch`, `mintex` — these are the aftermarket part numbers that may be hallucinated.

The `oem_part_number` column is the anchor — use it to look up what the correct aftermarket equivalents should be.

### Filters (`filter_specs.csv` etc.)
```
make,model,year_from,year_to,engine_code,fuel_type,oil_filter_oem,oil_filter_mann,oil_filter_filtron,air_filter_oem,air_filter_mann,air_filter_filtron,cabin_filter_oem,cabin_filter_mann,cabin_filter_filtron
```

Columns to verify: `*_mann` and `*_filtron` for each filter type.

## Free Verification Resources

### Best Single Tools Per Category

| Category | Tool | URL | Method |
|----------|------|-----|--------|
| Brake pads | **Ferodo Catalog** | https://www.ferodo.com/catalogue | Vehicle lookup → returns Ferodo + OEM + TRW/Textar/Brembo/Bosch cross-refs. **One lookup = 6 brands** |
| Brake pads | **Textar Brakebook** | https://brakebook.com | Vehicle or dimension search → Textar + Mintex + Pagid + OEM + WVA |
| Brake pads | **Brembo Catalog** | https://www.bremboparts.com/europe/en | Vehicle search → Brembo pads + OEM refs |
| Filters | **MANN Catalog** | https://catalog.mann-filter.com/EU/eng/vehicle | Vehicle search → oil + air + cabin filters + OEM cross-refs |
| Filters | **Filtron Catalog** | https://www.filtron.eu/en/catalogue | Vehicle search → Filtron parts + OEM refs |
| Any part | **Spareto** | https://spareto.com | Search by ANY part number → full cross-reference table |
| Any part | **Plenty.Parts** | https://plenty.parts | Part number → OEM refs + cross-refs + vehicle fitment |

### Cross-Reference Sites (enter any part number)
- https://www.oilfilter-crossreference.com — 200K+ oil filter cross-refs
- https://www.airfilter-crossreference.com — air filter cross-refs
- https://parts-crossreference.com — all part types
- https://spareto.com/products?keywords=PART_NUMBER — universal search

### OEM Parts Portals
- Toyota: https://autoparts.toyota.com
- Hyundai: https://www.hyundaipartsdeal.com
- VW Group: https://parts.vw.com
- BMW: https://www.realoem.com
- Mercedes: https://mbparts.mbusa.com

## Known Red Flags

### Brake Pad Number Formats
- **Brembo**: Always `P` + 5 digits (e.g., P83155)
- **TRW**: Always `GDB` + 4 digits (e.g., GDB3591)
- **Textar**: Always 7 digits starting with `2` (e.g., 2536101)
- **Ferodo**: Road pads always `FDB` + 4 digits (e.g., FDB4612). FDS/FCP = racing pads (wrong)
- **Bosch**: Always `0 986 49X XXX` format (e.g., 0986495348)
- **Mintex**: Always `MDB` + 4 digits (e.g., MDB3416)

### Filter Number Formats
- **Toyota oil filters**: Always start with `04152` (cartridge) or `90915` (spin-on)
- **Hyundai/Kia oil filters**: Always start with `26300`
- **VW Group**: Petrol = `04E`/`06L` prefix, diesel = `03N`/`03L` prefix
- **BMW**: Always `11 42X` prefix for oil filters

## Automated Verification Script

A script exists at `parts-finder/scripts/verify_crossrefs.py` that queries Spareto programmatically and compares results against our CSV. Run it with:

```bash
python scripts/verify_crossrefs.py data/seed/brake_pads_specs.csv
python scripts/verify_crossrefs.py data/seed/filter_specs.csv --category oil_filter
```

It outputs a JSON report of matches, mismatches, and not-found items. Use this as a starting point, then manually verify any mismatches using the catalogs above.

## Verification Workflow

For each unique OEM part number:
1. Look it up in the appropriate catalog (Ferodo for brakes, MANN for filters)
2. Record what the catalog says the cross-references are
3. Compare against our CSV values
4. If mismatch → update CSV with the catalog's numbers
5. Update the `confidence` column to `VERIFIED`

## Output

Save verified data as:
- `brake_pads_specs_verified.csv` (or update in place with confidence = VERIFIED)
- For filters: update confidence column in existing files

For each verification session, note:
```
Date: YYYY-MM-DD
Category: [brake pads / oil filters / etc.]
Tool used: [Ferodo / MANN / Spareto / etc.]
Rows checked: X
Correct: Y
Corrected: Z
Unable to verify: W
```

## Priority Order

1. Toyota Corolla E180/E210, Yaris, C-HR, RAV4, Camry (TNGA family)
2. Hyundai i30/Tucson + Kia Sportage (shared platforms)
3. VW Golf/Tiguan + Skoda Octavia (MQB platform) — **8 rows already VERIFIED**
4. BMW 3 Series F30/G20 + X1
5. Mercedes C-Class W205/W206 + GLC + A-Class
6. Peugeot 208/2008 + Citroen C3 (CMP platform)
7. Everything else

## What NOT To Do

- Do NOT use Claude's training data to verify — that's the same data source that generated the potentially wrong numbers (circular verification)
- Do NOT mark rows as VERIFIED without checking an external catalog
- Do NOT change OEM part numbers without confirming against the manufacturer's parts portal
