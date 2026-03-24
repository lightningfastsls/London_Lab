"""Verify brake pad cross-references using Spareto OEM lookup pages.

The OEM lookup endpoint (spareto.com/oe/<number>) returns the official
cross-reference table and is far more reliable than keyword search.

Usage::

    python scripts/verify_via_oe_lookup.py data/seed/brake_pads_specs.csv
    python scripts/verify_via_oe_lookup.py data/seed/brake_pads_specs.csv --delay 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[1]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# Brands we want to extract
TARGET_BRANDS = {
    "brembo": re.compile(r"\bP\s?(\d{5})\b"),
    "trw": re.compile(r"\bGDB\s?(\d{4})\b"),
    "textar": re.compile(r"\b(2\d{6})\b"),
    "ferodo": re.compile(r"\bFDB\s?(\d{4})\b"),
    "bosch": re.compile(r"\b(0\s?986\s?\d{6})\b"),
    "mintex": re.compile(r"\bMDB\s?(\d{4})\b"),
}


def normalize_oem(oem: str) -> str:
    """Strip spaces and dashes from OEM number for URL."""
    return re.sub(r"[\s\-]", "", oem).lower()


def query_oe_page(oem: str, delay: float = 2.0) -> dict[str, list[str]]:
    """Query spareto.com/oe/<oem> and extract brand cross-references."""
    oem_clean = normalize_oem(oem)
    url = f"https://spareto.com/oe/{oem_clean}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return {"_error": [f"HTTP {resp.status_code}"], "_url": [url]}
    except requests.RequestException as exc:
        return {"_error": [str(exc)], "_url": [url]}

    soup = BeautifulSoup(resp.text, "html.parser")
    page_text = soup.get_text(" ", strip=True)

    found: dict[str, list[str]] = {"_url": [url]}

    for brand, pattern in TARGET_BRANDS.items():
        matches = set()
        for m in pattern.finditer(page_text):
            part = m.group(0).strip()
            # Normalize
            if brand == "brembo":
                part = f"P{m.group(1)}"
            elif brand == "trw":
                part = f"GDB{m.group(1)}"
            elif brand == "ferodo":
                part = f"FDB{m.group(1)}"
            elif brand == "mintex":
                part = f"MDB{m.group(1)}"
            elif brand == "bosch":
                part = re.sub(r"\s+", "", part)
            matches.add(part)
        if matches:
            found[brand] = sorted(matches)

    return found


def verify_csv(csv_path: Path, delay: float) -> list[dict]:
    """Verify all unique OEM numbers in a brake pads CSV."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Collect unique OEM numbers
    oem_set: dict[str, list[int]] = {}
    for i, row in enumerate(rows, start=2):
        oem = row.get("oem_part_number", "").strip()
        if oem:
            oem_set.setdefault(oem, []).append(i)

    print(f"CSV has {len(rows)} rows, {len(oem_set)} unique OEM numbers")
    print()

    # Query each unique OEM
    oe_cache: dict[str, dict[str, list[str]]] = {}
    for idx, oem in enumerate(oem_set, 1):
        print(f"  [{idx}/{len(oem_set)}] OEM: {oem}", end="", flush=True)
        result = query_oe_page(oem, delay)
        oe_cache[oem] = result
        brands_found = len([k for k in result if not k.startswith("_")])
        if "_error" in result:
            print(f" ERROR: {result['_error'][0]}")
        else:
            print(f" -> {brands_found} target brands found")
        if idx < len(oem_set):
            time.sleep(delay)

    # Compare each CSV row
    brand_cols = ["brembo", "trw", "textar", "ferodo", "bosch", "mintex"]
    results = []

    total_checks = 0
    matches = 0
    mismatches = 0
    not_found = 0

    for i, row in enumerate(rows, start=2):
        oem = row.get("oem_part_number", "").strip()
        online = oe_cache.get(oem, {})
        row_result = {
            "row": i,
            "make": row.get("make", ""),
            "model": row.get("model", ""),
            "axle": row.get("axle", ""),
            "oem": oem,
            "checks": {},
        }

        for brand in brand_cols:
            csv_val = row.get(brand, "").strip()
            if not csv_val:
                continue
            total_checks += 1
            online_vals = online.get(brand, [])

            if not online_vals:
                status = "NOT_FOUND_ONLINE"
                not_found += 1
            else:
                csv_clean = re.sub(r"[\s\-/]", "", csv_val).upper()
                online_clean = [re.sub(r"[\s\-/]", "", v).upper() for v in online_vals]
                if csv_clean in online_clean:
                    status = "MATCH"
                    matches += 1
                else:
                    status = "MISMATCH"
                    mismatches += 1

            row_result["checks"][brand] = {
                "csv": csv_val,
                "online": online_vals,
                "status": status,
            }

        results.append(row_result)

    # Print summary
    print(f"\n{'='*70}")
    print(f"VERIFICATION REPORT (OE Lookup): {csv_path.name}")
    print(f"{'='*70}")
    print(f"\nRows checked:       {len(rows)}")
    print(f"Cross-refs checked: {total_checks}")
    print(f"  MATCH:            {matches} ({100*matches//max(total_checks,1)}%)")
    print(f"  MISMATCH:         {mismatches} ({100*mismatches//max(total_checks,1)}%)")
    print(f"  NOT FOUND ONLINE: {not_found} ({100*not_found//max(total_checks,1)}%)")

    # Print corrections needed
    if mismatches > 0:
        print(f"\n{'~'*70}")
        print("CORRECTIONS NEEDED:")
        print(f"{'~'*70}")
        seen = set()
        for r in results:
            for brand, check in r["checks"].items():
                if check["status"] == "MISMATCH":
                    key = (r["oem"], brand)
                    if key in seen:
                        continue
                    seen.add(key)
                    print(f"\n  OEM {r['oem']} ({r['make']} {r['model']} {r['axle']})")
                    print(f"    {brand.upper():8s}  CSV: {check['csv']:15s}  ->  Online: {check['online']}")

    # Save JSON report
    report_path = csv_path.parent / f"{csv_path.stem}_oe_verification.json"
    report_data = {
        "summary": {
            "total_rows": len(rows),
            "unique_oem": len(oem_set),
            "total_checks": total_checks,
            "matches": matches,
            "mismatches": mismatches,
            "not_found": not_found,
        },
        "oe_cache": {oem: data for oem, data in oe_cache.items()},
        "row_results": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to: {report_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify brake pad cross-refs via Spareto OE lookup"
    )
    parser.add_argument("csv_file", type=Path, help="CSV file to verify")
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Delay between HTTP requests (default: 2.0s)"
    )
    args = parser.parse_args()

    csv_path = args.csv_file
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path

    verify_csv(csv_path, args.delay)


if __name__ == "__main__":
    main()
