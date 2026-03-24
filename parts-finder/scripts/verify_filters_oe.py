"""Verify filter cross-references using Spareto OEM lookup pages.

Checks MANN and Filtron cross-refs for oil, air, and cabin filters.

Usage::

    python scripts/verify_filters_oe.py data/seed/filter_specs.csv
    python scripts/verify_filters_oe.py data/seed/stage1_filter_specs.csv --delay 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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

# Filter brands and patterns
# MANN filter numbers: letter(s) + space + digits + optional space + letter/digits
# e.g., HU 6006 z, C 25 040, CU 1919, W 712/83
MANN_PATTERN = re.compile(
    r"\b((?:HU|W|H|C|CU|CUK|FP)\s?\d[\d\s/]*\d?(?:\s?[a-z])?)\b",
    re.IGNORECASE,
)

# Filtron: letter(s) + space + digits + optional /digits
# e.g., OE 685/2, AP 142/9, K 1210, OP 616/3
FILTRON_PATTERN = re.compile(
    r"\b((?:OE|OP|AP|AK|K|AE)\s?\d[\d\s/]*\d?(?:\s?[A-Z])?)\b",
    re.IGNORECASE,
)


def normalize_oem(oem: str) -> str:
    """Strip spaces and dashes from OEM number for URL."""
    return re.sub(r"[\s\-]", "", oem).lower()


def normalize_part(part: str) -> str:
    """Normalize for comparison: strip spaces, lowercase."""
    return re.sub(r"\s+", "", part).upper()


def query_oe_page(oem: str) -> dict[str, list[str]]:
    """Query spareto.com/oe/<oem> and extract MANN + Filtron refs."""
    oem_clean = normalize_oem(oem)
    url = f"https://spareto.com/oe/{oem_clean}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return {"_error": f"HTTP {resp.status_code}", "_url": url}
    except requests.RequestException as exc:
        return {"_error": str(exc), "_url": url}

    soup = BeautifulSoup(resp.text, "html.parser")
    page_text = soup.get_text(" ", strip=True)

    found: dict[str, list[str]] = {"_url": url}

    # Extract MANN numbers
    mann_matches = set()
    for m in MANN_PATTERN.finditer(page_text):
        part = m.group(1).strip()
        mann_matches.add(part)
    if mann_matches:
        found["mann"] = sorted(mann_matches)

    # Extract Filtron numbers
    filtron_matches = set()
    for m in FILTRON_PATTERN.finditer(page_text):
        part = m.group(1).strip()
        filtron_matches.add(part)
    if filtron_matches:
        found["filtron"] = sorted(filtron_matches)

    return found


def verify_csv(csv_path: Path, delay: float) -> dict:
    """Verify all filter cross-refs in a CSV."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    # Detect which filter types exist
    filter_types = []
    for ft in ["oil_filter", "air_filter", "cabin_filter"]:
        if f"{ft}_oem" in headers:
            filter_types.append(ft)

    print(f"CSV: {len(rows)} rows, filter types: {filter_types}")
    print()

    # Collect unique OEM numbers across all filter types
    all_oems: set[str] = set()
    for row in rows:
        for ft in filter_types:
            oem = row.get(f"{ft}_oem", "").strip()
            if oem:
                all_oems.add(oem)

    print(f"Unique OEM numbers to look up: {len(all_oems)}")
    print()

    # Query each OEM
    oe_cache: dict[str, dict] = {}
    for idx, oem in enumerate(sorted(all_oems), 1):
        print(f"  [{idx}/{len(all_oems)}] OEM: {oem}", end="", flush=True)
        result = query_oe_page(oem)
        oe_cache[oem] = result
        mann_count = len(result.get("mann", []))
        filtron_count = len(result.get("filtron", []))
        if "_error" in result:
            print(f" ERROR: {result['_error']}")
        else:
            print(f" -> MANN:{mann_count} Filtron:{filtron_count}")
        if idx < len(all_oems):
            time.sleep(delay)

    # Compare
    total = 0
    matches = 0
    mismatches = 0
    not_found = 0
    mismatch_details = []

    for i, row in enumerate(rows, 2):
        make = row.get("make", "")
        model = row.get("model", "")
        engine = row.get("engine_code", "")

        for ft in filter_types:
            oem = row.get(f"{ft}_oem", "").strip()
            if not oem:
                continue

            online = oe_cache.get(oem, {})

            for brand in ["mann", "filtron"]:
                csv_val = row.get(f"{ft}_{brand}", "").strip()
                if not csv_val:
                    continue
                total += 1

                online_vals = online.get(brand, [])
                if not online_vals:
                    status = "NOT_FOUND"
                    not_found += 1
                else:
                    csv_norm = normalize_part(csv_val)
                    online_norm = [normalize_part(v) for v in online_vals]
                    if csv_norm in online_norm:
                        status = "MATCH"
                        matches += 1
                    else:
                        status = "MISMATCH"
                        mismatches += 1
                        mismatch_details.append({
                            "row": i,
                            "vehicle": f"{make} {model} ({engine})",
                            "filter_type": ft,
                            "brand": brand,
                            "oem": oem,
                            "csv": csv_val,
                            "online": online_vals,
                        })

    # Report
    print(f"\n{'='*70}")
    print(f"FILTER VERIFICATION REPORT: {csv_path.name}")
    print(f"{'='*70}")
    print(f"\nCross-refs checked: {total}")
    print(f"  MATCH:            {matches} ({100*matches//max(total,1)}%)")
    print(f"  MISMATCH:         {mismatches} ({100*mismatches//max(total,1)}%)")
    print(f"  NOT FOUND ONLINE: {not_found} ({100*not_found//max(total,1)}%)")

    if mismatch_details:
        print(f"\n{'~'*70}")
        print("MISMATCHES:")
        print(f"{'~'*70}")
        seen = set()
        for d in mismatch_details:
            key = (d["oem"], d["brand"], d["filter_type"])
            if key in seen:
                continue
            seen.add(key)
            print(f"\n  {d['vehicle']} - {d['filter_type']}")
            print(f"    OEM: {d['oem']}")
            print(f"    {d['brand'].upper():8s}  CSV: {d['csv']:20s}  Online: {d['online']}")

    # Save JSON
    report_path = csv_path.parent / f"{csv_path.stem}_filter_verification.json"
    report = {
        "summary": {
            "total": total, "matches": matches,
            "mismatches": mismatches, "not_found": not_found,
        },
        "mismatches": mismatch_details,
        "oe_cache": oe_cache,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Verify filter cross-refs via Spareto OE lookup"
    )
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    csv_path = args.csv_file
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path

    verify_csv(csv_path, args.delay)


if __name__ == "__main__":
    main()
