# Brake Pads Cross-Reference Verification Guide
## Parts Finder — Tevel Group

### Why This Matters
The brake_pads_specs.csv contains ~98 rows with 6 aftermarket cross-references each.
During audit, the entire VW MQB group (8 rows) was found to have WRONG Brembo/TRW/Textar/Ferodo/Bosch/Mintex numbers.
Brake pads are **safety-critical** — every cross-reference must be verified before production.

---

## Step-by-Step Verification Method

### Tool 1: Ferodo Catalog (BEST — gives full cross-reference chain)

1. Go to **https://www.ferodo.com/catalogue**
2. Select vehicle: Make → Model → Engine variant → Year
3. The catalog returns:
   - Ferodo part number (FDB_____)
   - OEM reference numbers
   - Cross-references to: TRW, Textar, Brembo, Bosch, ATE, Mintex
4. **Record all numbers** — this single lookup verifies 6+ brands at once

### Tool 2: Brembo Parts Catalog

1. Go to **https://www.bremboparts.com/europe/en**
2. Search by vehicle (Make → Model → Engine → Year)
3. Returns Brembo pad code (P_____) + OEM reference
4. Use this to confirm the Brembo column specifically

### Tool 3: Spareto Cross-Reference (backup/spot-check)

1. Go to **https://spareto.com**
2. Search by ANY known part number (OEM, Brembo, Ferodo, TRW, etc.)
3. Returns full cross-reference table with all brands
4. Good for verifying a single number you're unsure about

### Tool 4: Textar Brakebook (most data-rich)

1. Go to **https://www.textar.com/en/brakebook**
2. Search by vehicle or by part number
3. Returns Textar number + WVA number + dimensions + OEM ref
4. The WVA number is the universal EU brake pad identifier — use it to cross-check

---

## Verification Workflow

### Phase 1: OEM Numbers (fastest, ~30 min)
For each row, verify the OEM part number against the manufacturer's parts site:
- Toyota: https://autoparts.toyota.com
- Hyundai: https://www.hyundaipartsdeal.com
- VW Group: https://parts.vw.com or ETKA
- BMW: https://www.realoem.com
- Mercedes: https://mbparts.mbusa.com
- PSA: https://public.servicebox.peugeot.com

**If the OEM number is wrong, the entire row is wrong.**

### Phase 2: Aftermarket Cross-References (~2-3 hours)
For each unique pad shape (there are ~50 unique OEM numbers):

1. Look up the OEM number in Ferodo catalog → get Ferodo FDB number + cross-refs
2. Spot-check Brembo number via bremboparts.com vehicle lookup
3. If any number doesn't match, use Spareto to find the correct one
4. Update the CSV

### Phase 3: Spot-Check with Physical Catalogs
If Tevel has physical Brembo/Ferodo/TRW catalogs (common in Israeli parts shops):
- Cross-check 10-15 high-volume vehicles against the printed catalog
- These are the most authoritative source

---

## Priority Order (verify these first)

### Tier 1 — Highest Volume (verify first)
1. Toyota Corolla E180 + E210 (front + rear)
2. VW Golf 7/8 MQB (front + rear) ← ALREADY FIXED
3. Hyundai i30 PD + Tucson TL (front + rear)
4. Kia Sportage QL (front + rear)
5. Skoda Octavia 3/4 MQB ← ALREADY FIXED
6. Toyota Yaris + C-HR

### Tier 2 — Important
7. BMW 3 Series F30 + G20
8. Mercedes C-Class W205
9. Peugeot 208/2008 + Citroen C3 (CMP platform)
10. Toyota RAV4 + Camry (TNGA-K)

### Tier 3 — Can wait
11. Mazda 3/CX-5
12. Renault/Nissan
13. Ford Focus
14. Subaru
15. Everything else

---

## What to Look For (Red Flags)

- **Brembo numbers that don't appear on bremboparts.com** → likely hallucinated
- **TRW GDB numbers where the numeric part seems too high** (GDB3000+ is newer; GDB1000-2000 is older generation)
- **Textar numbers that don't start with 2** → Textar pad sets are always 2XXXXXX format
- **Ferodo numbers that don't start with FDB** → road pads are always FDB; FDS/FCP are racing
- **Bosch numbers that don't start with 0986** → always 0 986 49X XXX format
- **Multiple vehicles sharing the SAME aftermarket number but DIFFERENT OEM numbers** → verify this is correct platform sharing, not a copy-paste error

---

## After Verification

Update each row's confidence column:
- `VERIFIED` — confirmed via Ferodo/Brembo/Spareto catalog lookup
- `HIGH` — matches training data, not independently verified
- `MEDIUM` — uncertain, needs verification
- `LOW` — conflicting data found, investigate further
- `WRONG` — confirmed incorrect, needs replacement

Save the verified CSV as `brake_pads_specs_verified.csv` alongside the original.
