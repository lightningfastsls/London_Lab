"""Validate oil_specs_expanded.csv before import."""
import csv
import io
import sys
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

csv_path = Path("data/seed/oil_specs_expanded.csv")
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows: {len(rows)}")

# Check makes
makes = Counter(r["make"] for r in rows)
print(f"\nMakes ({len(makes)}):")
for make, count in makes.most_common():
    print(f"  {make:20s} {count} rows")

# Check for known DB makes that might not match
db_makes = {"Toyota", "Hyundai", "Kia", "Mazda", "Skoda", "Volkswagen",
            "Suzuki", "BMW", "Mercedes-Benz", "Nissan", "Honda", "Subaru",
            "Peugeot", "Citroen", "Renault", "Fiat", "Chevrolet", "Ford",
            "Seat", "Daihatsu", "Opel", "Volvo", "Audi", "Mitsubishi"}
new_makes = set(makes.keys()) - db_makes
if new_makes:
    print(f"\nNEW makes (not in hebrew_names.json): {sorted(new_makes)}")

# Check fuel types
fuels = Counter(r["fuel_type"] for r in rows)
print(f"\nFuel types: {dict(fuels)}")

# Check for empty required fields
issues = []
for i, r in enumerate(rows, 2):
    for field in ["make", "model", "year_from", "year_to", "engine_code", "fuel_type", "oil_viscosity"]:
        if not r.get(field, "").strip():
            issues.append(f"Row {i}: empty {field}")
    # Check year range validity
    try:
        yf, yt = int(r["year_from"]), int(r["year_to"])
        if yf > yt:
            issues.append(f"Row {i}: year_from ({yf}) > year_to ({yt})")
        if yf < 2000 or yt > 2030:
            issues.append(f"Row {i}: suspicious year range {yf}-{yt}")
    except ValueError:
        issues.append(f"Row {i}: non-numeric year")

if issues:
    print(f"\nIssues ({len(issues)}):")
    for issue in issues[:20]:
        print(f"  {issue}")
else:
    print("\nNo issues found!")

# Check for duplicates (make+model+year_from+year_to+engine_code)
keys = Counter(
    (r["make"], r["model"], r["year_from"], r["year_to"], r["engine_code"])
    for r in rows
)
dupes = {k: v for k, v in keys.items() if v > 1}
if dupes:
    print(f"\nDuplicate keys ({len(dupes)}):")
    for k, v in dupes.items():
        print(f"  {k}: {v} times")
else:
    print("No duplicate keys.")
