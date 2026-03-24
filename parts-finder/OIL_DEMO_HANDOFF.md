# Oil Demo Handoff

## Goal
Build a self-contained oil finder demo at `parts-finder/oil_demo.html` that:
1. Takes a license plate (same UX as the real Parts Finder)
2. Calls the plate lookup API to get vehicle + oil spec
3. Shows matching products from the real inventory (embedded in the HTML)
4. No prices shown (intentionally omitted for demo)

## Current State — What's Done

### 1. `parts-finder/oil_demo.html` — DONE
- Full HTML page with plate input, API call, results rendering
- Matches the Parts Finder dark glassmorphism style (DM Sans, Space Mono, gold accents)
- Embeds ~50 engine oil products from the real inventory (`oil_inventory_data.json`)
- Matching logic: primary = viscosity match, secondary = OEM keyword scoring (VW 504/507, Hyundai, Toyota, BMW, Mercedes)
- Stock indicators (green/yellow/red), OEM match badges
- Calls `POST /api/plate-lookup` on the same origin

### 2. Database seeded — DONE
- Ran `python scripts/seed_database.py --db parts_finder.db` from `parts-finder/`
- 193 vehicle oil specs imported from `oil-finder-free.jsx` (3 EVs skipped)
- DB file: `parts-finder/parts_finder.db`
- Intermediate CSV: `parts-finder/data/seed/oil_db_seed.csv`

### 3. Static file serving added to FastAPI — DONE
- `parts-finder/src/parts_finder/api/app.py` now has a `GET /` route serving `oil_demo.html`
- Also fixed `_DEFAULT_MAPPING_PATH` from `parents[2]` to `parents[3]` (was pointing to `src/` instead of repo root)

### 4. Server starts successfully
```powershell
cd parts-finder
$env:PYTHONPATH = "C:\Users\shach\PycharmProjects\mickey_london_lab\parts-finder\src"
.\..\venv\Scripts\python.exe -m uvicorn "parts_finder.api.app:create_app" --factory --host 127.0.0.1 --port 8000
```
Or from bash:
```bash
cd "C:/Users/shach/PycharmProjects/mickey_london_lab/parts-finder"
PYTHONPATH="C:/Users/shach/PycharmProjects/mickey_london_lab/parts-finder/src" \
  "C:/Users/shach/PycharmProjects/mickey_london_lab/.venv/Scripts/python.exe" \
  -m uvicorn "parts_finder.api.app:create_app" --factory --host 127.0.0.1 --port 8000
```
- `http://127.0.0.1:8000/` serves the demo HTML (confirmed 200)
- `POST http://127.0.0.1:8000/api/plate-lookup` is the API endpoint

## What Needs Testing / May Need Fixing

### Plate lookup end-to-end flow
- The server starts and serves the HTML, but the **full plate → oil spec flow hasn't been tested with a real plate yet**
- The flow is: plate input → `POST /api/plate-lookup` → gov API (data.gov.il) resolves plate to vehicle → `LookupEngine` matches vehicle to DB specs → response includes `categories.oil` with viscosity/spec/oem_approval/capacity
- **Possible issues:**
  - Gov API may require internet access / may have rate limits
  - Vehicle name comes back in Hebrew → `NameMapper` translates using `data/hebrew_names.json` → if a name isn't mapped, the DB lookup may miss
  - The DB only has oil specs (no filters/brakes/bulbs) since we seeded from the oil-only JSX file

### HTML rendering of API response
- The `renderResults()` function in the HTML expects the API response shape:
  ```json
  {
    "vehicle": { "plate": "...", "make": "...", "model": "...", "year": 2022, "engine_code": "...", "fuel_type": "..." },
    "categories": { "oil": { "viscosity": "5W-30", "spec": "ACEA C3", "oem_approval": "VW 504.00/507.00", "capacity_l": 4.0, "change_interval_km": 15000, "confidence": "high" } },
    "data_source": "database",
    "coverage": "1/7 categories"
  }
  ```
- If the shape differs, the HTML may not render correctly

### Inventory matching
- Works client-side in the browser (no API needed for this part)
- Can test independently: the HTML has a `matchProducts(viscosity, oemApproval, makeName)` function

## Key Files

| File | Purpose |
|------|---------|
| `parts-finder/oil_demo.html` | The demo page (HTML + embedded JS + inventory data) |
| `parts-finder/oil_inventory_data.json` | Full 85-product inventory extracted from Excel (reference) |
| `parts-finder/parts_finder.db` | Seeded SQLite DB (193 vehicle oil specs) |
| `parts-finder/src/parts_finder/api/app.py` | FastAPI app factory (modified: static serving + path fix) |
| `parts-finder/src/parts_finder/api/routes.py` | API route: `POST /api/plate-lookup` |
| `parts-finder/src/parts_finder/lookup_engine.py` | Plate → vehicle → specs orchestrator |
| `parts-finder/src/parts_finder/api/response_builder.py` | Builds the JSON response shape |
| `parts-finder/src/parts_finder/api/schemas.py` | Pydantic models for request/response |
| `parts-finder/data/hebrew_names.json` | Hebrew→English vehicle name mapping |
| `oil-finder-free.jsx` | Source vehicle oil spec database (repo root) |

## Changes Made This Session

1. **`parts-finder/oil_demo.html`** — Complete rewrite (was a simple inventory catalog browser, now plate-lookup + inventory matching)
2. **`parts-finder/src/parts_finder/api/app.py`** — Added `FileResponse` import, `StaticFiles` import, `GET /` route for demo HTML, fixed `_DEFAULT_MAPPING_PATH` from `parents[2]` to `parents[3]`
3. **`parts-finder/parts_finder.db`** — New file, seeded via `scripts/seed_database.py`

## Quick Test Checklist
- [ ] Start server (command above)
- [ ] Open `http://127.0.0.1:8000/`
- [ ] Enter a real Israeli plate number (7-8 digits)
- [ ] Verify vehicle card appears with correct make/model
- [ ] Verify oil spec card shows viscosity, OEM approval, capacity
- [ ] Verify matching products panel shows inventory items sorted by relevance
- [ ] Try a VW/Skoda plate → should show 504/507 OEM-matched items first
- [ ] Try a Toyota plate → should show 0W-20 items with Toyota OEM items first
