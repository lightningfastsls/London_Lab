"""Check whether OEM part numbers exist on Spareto.

For each unique OEM number across all filter CSVs, checks if
spareto.com/oe/<number> returns a page with products.
Flags OEM numbers that return 404 or empty pages as potentially fabricated.

Usage::

    python scripts/verify_oem_existence.py
    python scripts/verify_oem_existence.py --delay 1.5
    python scripts/verify_oem_existence.py --include-brakes
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
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

FILTER_FILES = [
    "data/seed/filter_specs.csv",
    "data/seed/stage1_filter_specs.csv",
    "data/seed/expansion_filter_specs.csv",
    "data/seed/stage2_filter_specs.csv",
]

BRAKE_FILE = "data/seed/brake_pads_specs.csv"


def normalize_oem(oem: str) -> str:
    """Strip spaces and dashes for URL."""
    return re.sub(r"[\s\-]", "", oem).lower()


def check_oe_page(oem: str) -> dict:
    """Check if Spareto has an OE page for this number."""
    oem_clean = normalize_oem(oem)
    url = f"https://spareto.com/oe/{oem_clean}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
    except requests.RequestException as exc:
        return {"exists": None, "error": str(exc), "url": url, "products": 0}

    if resp.status_code == 404:
        return {"exists": False, "url": url, "products": 0}
    if resp.status_code != 200:
        return {"exists": None, "error": f"HTTP {resp.status_code}", "url": url, "products": 0}

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(" ", strip=True)

    # Check for "Nothing Matches" or similar empty indicators
    if "Nothing Matches" in text or "no results" in text.lower():
        return {"exists": False, "url": url, "products": 0}

    # Count product listings
    products = soup.find_all("div", class_=re.compile(r"product"))
    if not products:
        products = soup.find_all("a", href=re.compile(r"/product/"))

    return {"exists": True, "url": url, "products": len(products)}


def collect_oem_numbers(include_brakes: bool) -> dict[str, list[dict]]:
    """Collect all unique OEM numbers with their vehicle associations."""
    oem_map: dict[str, list[dict]] = defaultdict(list)

    # Filter CSVs
    for fp_str in FILTER_FILES:
        fp = REPO_ROOT / fp_str
        if not fp.exists():
            continue
        with open(fp, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for ft in ["oil_filter", "air_filter", "cabin_filter"]:
                    oem = row.get(f"{ft}_oem", "").strip()
                    if oem:
                        oem_map[oem].append({
                            "make": row.get("make", ""),
                            "model": row.get("model", ""),
                            "engine": row.get("engine_code", ""),
                            "type": ft,
                            "source": fp.name,
                        })

    # Brake pads
    if include_brakes:
        fp = REPO_ROOT / BRAKE_FILE
        if fp.exists():
            with open(fp, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    oem = row.get("oem_part_number", "").strip()
                    if oem:
                        oem_map[oem].append({
                            "make": row.get("make", ""),
                            "model": row.get("model", ""),
                            "engine": row.get("axle", ""),
                            "type": "brake_pad",
                            "source": fp.name,
                        })

    return dict(oem_map)


def main():
    parser = argparse.ArgumentParser(description="Check OEM numbers exist on Spareto")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--include-brakes", action="store_true")
    args = parser.parse_args()

    oem_map = collect_oem_numbers(args.include_brakes)
    total = len(oem_map)
    print(f"Checking {total} unique OEM numbers on Spareto...")
    print()

    results: dict[str, dict] = {}
    exists_count = 0
    not_found_count = 0
    error_count = 0

    for idx, oem in enumerate(sorted(oem_map.keys()), 1):
        vehicles = oem_map[oem]
        first = vehicles[0]
        label = f"{first['make']} {first['model']} {first['type']}"

        print(f"  [{idx}/{total}] {oem:25s} ({label})", end="", flush=True)
        result = check_oe_page(oem)
        results[oem] = result

        if result["exists"] is True:
            exists_count += 1
            print(f"  OK ({result['products']} products)")
        elif result["exists"] is False:
            not_found_count += 1
            print(f"  NOT FOUND")
        else:
            error_count += 1
            print(f"  ERROR: {result.get('error', '?')}")

        if idx < total:
            time.sleep(args.delay)

    # Summary
    print(f"\n{'='*60}")
    print(f"OEM EXISTENCE CHECK SUMMARY")
    print(f"{'='*60}")
    print(f"Total checked:  {total}")
    print(f"  EXISTS:       {exists_count} ({100*exists_count//total}%)")
    print(f"  NOT FOUND:    {not_found_count} ({100*not_found_count//total}%)")
    print(f"  ERROR:        {error_count} ({100*error_count//total}%)")

    if not_found_count > 0:
        print(f"\n{'~'*60}")
        print("OEM NUMBERS NOT FOUND ON SPARETO:")
        print(f"{'~'*60}")
        for oem, result in sorted(results.items()):
            if result["exists"] is False:
                vehicles = oem_map[oem]
                makes = set(f"{v['make']} {v['model']}" for v in vehicles)
                types = set(v["type"] for v in vehicles)
                print(f"  {oem:25s}  {', '.join(types):20s}  {', '.join(makes)}")

    # Save report
    report_path = REPO_ROOT / "data" / "seed" / "oem_existence_report.json"
    report = {
        "summary": {
            "total": total,
            "exists": exists_count,
            "not_found": not_found_count,
            "errors": error_count,
        },
        "not_found": [
            {"oem": oem, "vehicles": oem_map[oem]}
            for oem, r in results.items() if r["exists"] is False
        ],
        "all_results": {oem: r for oem, r in results.items()},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
