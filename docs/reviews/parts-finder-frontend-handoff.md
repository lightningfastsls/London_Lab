# Implementation Handoff: React Parts Finder UI (Phase 7.1)

**Tier:** 2 (new module, multiple files, UI logic)
**Date:** 2026-02-25
**Status:** Ready for review

## What Was Built

Converted the static oil-finder prototype (`oil-finder-free.jsx`) into a dynamic, API-driven React frontend covering all 7 product categories. Four portable React modules that call the real FastAPI backend via `POST /api/plate-lookup` and render vehicle info + category cards with the prototype's dark-theme design language.

## Files Created

| File | Action | Lines | Purpose |
|------|--------|-------|---------|
| `parts-finder/frontend/src/api/partsApi.js` | CREATE | 80 | API client — `lookupPlate()` with typed error classes |
| `parts-finder/frontend/src/components/VehicleInfo.jsx` | CREATE | 71 | Vehicle identity card (make, model, year, plate badge) |
| `parts-finder/frontend/src/components/CategoryCard.jsx` | CREATE | 432 | Smart card — renders any of 7 category types with per-type icons, colors, layouts |
| `parts-finder/frontend/src/components/PartsFinder.jsx` | CREATE | 352 | Main orchestrator — plate input, API state, error handling, coverage summary, grid |

**Total:** 935 lines across 4 files.

## Architecture Decisions

1. **Typed error classes over generic errors** — `PlateFormatError`, `PlateNotFoundError`, `ServiceError` enable `instanceof` checks in the UI for specific error messaging per HTTP status (400/404/503). Mirrors the backend's `ErrorResponse` model.

2. **Strategy pattern for category rendering** — `CategoryCard.jsx` uses a `CONTENT_RENDERERS` map dispatching on `type` string rather than a switch/if-else chain. Adding an 8th category means adding one renderer function + one map entry.

3. **`hexToRgba` utility for per-category theming** — Each category has a single hex accent color (e.g., oil=#eab308, brakes=#ef4444). The utility generates rgba variants at different opacities for border, background, and icon tint from that one value.

4. **Hebrew BiDi via heuristic** — `VehicleInfo.jsx` wraps text in `<span dir="rtl">` only when Hebrew Unicode range is detected. The overall layout stays LTR since plate numbers and category labels are all Latin/numeric.

5. **No shared component extraction** — `Tag` is duplicated in both `CategoryCard.jsx` and `PartsFinder.jsx`. Deliberate trade-off: each file is self-contained and portable without a shared module. Acceptable at this scale (2 files).

6. **No build toolchain** — These are raw React modules, not a standalone app. User imports `<PartsFinder />` into any React project (Vite, Next.js, CRA). No `package.json`, `index.html`, or build config included per plan scope.

## Public API

```jsx
// Entry point — drop into any React app
import PartsFinder from "./components/PartsFinder";

// Standalone API client (if needed separately)
import { lookupPlate, PlateFormatError, PlateNotFoundError, ServiceError } from "./api/partsApi";

// Individual components (if composing a custom layout)
import VehicleInfo from "./components/VehicleInfo";     // props: { vehicle }
import CategoryCard from "./components/CategoryCard";   // props: { type, data, isAiFallback }
```

## What I'm Unsure About

1. **EV detection heuristic** — Currently checks if all non-coolant categories are null. This works today but will break if the backend starts returning brake/bulb data for EVs. A more robust approach would check `vehicle.fuel_type` for "electric". Left as-is since the backend doesn't populate those categories for EVs yet.

2. **CORS during development** — `API_BASE_URL` defaults to `window.location.origin`, which works in production (same-origin) but will fail during local dev (React on :5173, FastAPI on :8000). Needs either a Vite proxy config, a `.env` override, or CORS middleware on the backend. Not a code bug — an infrastructure gap.

## Review Focus Areas

- [ ] **API contract alignment** — Do all 35 field accesses in JSX match the Pydantic schemas? (PR reviewer verified ✅, but worth a second look)
- [ ] **AI fallback badge logic** — `isAiFallback={isAiSource}` now badges all cards when `data_source` is `hybrid` or `ai_fallback`. Is this the right UX, or should only AI-sourced categories be badged? (Backend doesn't yet flag per-category source)
- [ ] **Null guards** — `change_interval_km` has a null guard. Are there other numeric fields that could arrive null from AI fallback?
- [ ] **Responsive layout** — CSS Grid uses `repeat(auto-fit, minmax(320px, 1fr))`. Does this behave well on very narrow screens (<360px)?

## Test Coverage

No React unit tests — out of scope per plan ("Unit tests (React Testing Library) — not specified in ROADMAP test plan"). Verification is visual:

| Check | Method |
|-------|--------|
| API field alignment | PR reviewer verified all 35 field accesses against Pydantic schemas |
| Component rendering | Manual visual check in a React app with mock data |
| Error states | Test with invalid plate (400), unknown plate (404), server down (503) |
| Dead code removal | Removed unused `Zap` icon and `Card` component post-review |

## Verification Results

```
Files created:   4/4 ✓
PR review:       APPROVED with minor fixes (all fixes applied)
  - Fixed isAiFallback logic bug (hybrid case was silently dropped)
  - Removed dead code (Zap icon, unused Card component)
  - Added NaN guard on change_interval_km
  - Updated header comment
```

## Dependencies

- `react` (peer dependency — user's project provides this)
- No other dependencies. Uses vanilla `fetch`, inline SVGs, inline styles.

## What's Next

1. **Phase 7.2** — Tevel SKU integration (blocked on Tevel data feed)
2. **CORS/proxy setup** — Configure dev environment for cross-origin API calls
3. **Visual QA** — Import into a Vite project and test all 7 category renders with real backend data
4. **Per-category AI badge** — When backend adds per-category `source` field, update `isAiFallback` to be granular instead of global
