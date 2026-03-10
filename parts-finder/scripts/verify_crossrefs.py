"""Verify aftermarket cross-references against online catalogs.

Queries Spareto (spareto.com) with OEM part numbers from our CSVs and
compares the returned aftermarket cross-references against our data.
Outputs a report of matches, mismatches, and missing brands.

Supports brake pads, oil filters, air filters, and cabin filters.

Usage::

    python scripts/verify_crossrefs.py data/seed/brake_pads_specs.csv
    python scripts/verify_crossrefs.py data/seed/filter_specs.csv --category oil_filter
    python scripts/verify_crossrefs.py data/seed/brake_pads_specs.csv --delay 3
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

# Fix Windows console encoding (only when running as main script)
def _fix_encoding() -> None:
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace"
            )
        except (AttributeError, ValueError):
            pass

import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Configuration -----------------------------------------------------------

SPARETO_URL = "https://spareto.com/products?keywords={query}"
SPARETO_PAGES = 3  # Max pages to check per OEM number

# Brands we care about and how to normalize their names from Spareto results
BRAND_ALIASES = {
    "brembo": "brembo",
    "trw": "trw",
    "textar": "textar",
    "ferodo": "ferodo",
    "bosch": "bosch",
    "mintex": "mintex",
    "mann": "mann",
    "mann-filter": "mann",
    "mann filter": "mann",
    "filtron": "filtron",
    "mahle": "mahle",
    "knecht": "mahle",
    "hengst": "hengst",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class CrossRefResult:
    """Result of querying a cross-reference site for one OEM part number."""
    oem_part_number: str
    found_brands: dict[str, list[str]] = field(default_factory=dict)
    # brand_name (normalized) -> list of part numbers found
    total_products: int = 0
    error: str = ""


@dataclass
class VerificationRow:
    """One row from our CSV compared against online data."""
    csv_row_num: int
    make: str
    model: str
    oem_pn: str
    category: str  # "front_brake_pad", "rear_brake_pad", "oil_filter", etc.
    checks: dict[str, dict] = field(default_factory=dict)
    # brand -> {"csv": "P83155", "online": ["P83155"], "status": "MATCH|MISMATCH|NOT_FOUND"}


# --- Spareto Scraper ---------------------------------------------------------

def query_spareto(oem_pn: str, delay: float = 2.0) -> CrossRefResult:
    """Query Spareto for cross-references to an OEM part number."""
    result = CrossRefResult(oem_part_number=oem_pn)

    for page in range(1, SPARETO_PAGES + 1):
        url = SPARETO_URL.format(query=quote(oem_pn))
        if page > 1:
            url += f"&page={page}"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                if page == 1:
                    result.error = f"HTTP {resp.status_code}"
                break

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract product count from first page
            if page == 1:
                count_el = soup.find(string=re.compile(r"\d+ Products?"))
                if count_el:
                    m = re.search(r"(\d+)\s+Products?", count_el)
                    if m:
                        result.total_products = int(m.group(1))

            # Find product cards/listings
            # Spareto shows brand name and part number in product listings
            products = soup.find_all("div", class_=re.compile(r"product"))
            if not products:
                # Try alternate selectors
                products = soup.find_all("a", href=re.compile(r"/product/"))

            for prod in products:
                text = prod.get_text(" ", strip=True)
                # Extract brand and part number from product text
                _extract_brand_part(text, result)

            # Also scan all text for known brand patterns
            page_text = soup.get_text(" ", strip=True)
            _scan_text_for_brands(page_text, result)

            # Check if there are more pages
            if result.total_products <= page * 12:
                break

        except requests.RequestException as exc:
            if page == 1:
                result.error = str(exc)
            break

        if page < SPARETO_PAGES:
            time.sleep(delay)

    return result


def _extract_brand_part(text: str, result: CrossRefResult) -> None:
    """Extract brand name and part number from a product listing text."""
    text_lower = text.lower()
    for alias, normalized in BRAND_ALIASES.items():
        # Use word boundary to avoid "mann" matching inside "zimmermann"
        if not re.search(rf"\b{re.escape(alias)}\b", text_lower):
            continue
        if True:  # Keep indent level
            # Try to extract part number near the brand name
            # Common patterns: "BRAND ® PARTNUMBER" or "BRAND PARTNUMBER"
            patterns = [
                # "TEXTAR ® 2536101"
                re.compile(
                    rf"{re.escape(alias)}\s*[®]*\s*([A-Z0-9][\w\s/\-]{{2,20}})",
                    re.IGNORECASE,
                ),
                # "2536101 - TEXTAR"
                re.compile(
                    rf"([A-Z0-9][\w\s/\-]{{2,20}})\s*-?\s*{re.escape(alias)}",
                    re.IGNORECASE,
                ),
            ]
            for pat in patterns:
                m = pat.search(text)
                if m:
                    part = m.group(1).strip().rstrip("-– ")
                    # Clean up common noise
                    part = re.sub(r"\s*(Brake Pad|disc brake|Set|Oil Filter|next|business|dispatch).*", "", part, flags=re.IGNORECASE)
                    part = part.strip()
                    # Skip noise: delivery text, HTML artifacts, generic words
                    noise = {"on", "in", "the", "for", "and", "set", "new", "old"}
                    if (len(part) >= 3
                            and not part.lower().startswith("brake")
                            and part.lower() not in noise
                            and not re.match(r"^(on|in|dispatch|next)\b", part, re.IGNORECASE)):
                        result.found_brands.setdefault(normalized, [])
                        if part not in result.found_brands[normalized]:
                            result.found_brands[normalized].append(part)


def _scan_text_for_brands(text: str, result: CrossRefResult) -> None:
    """Scan page text for specific brand part number patterns."""
    # Known part number formats
    _PATTERNS = {
        "brembo": re.compile(r"\bP\s?(\d{5})\b"),         # P83155 or P 83155
        "trw": re.compile(r"\bGDB\s?(\d{4})\b"),          # GDB3591
        "textar": re.compile(r"\b(2[3-6]\d{5})\b"),        # 2536101 (brake pads: 23xxxxx-26xxxxx)
        "ferodo": re.compile(r"\bFDB\s?(\d{4})\b"),        # FDB4612
        "bosch": re.compile(r"\b(0\s?986\s?49\d{4})\b"),   # 0986495348 (brake: 098649xxxx)
        "mintex": re.compile(r"\bMDB\s?(\d{4})\b"),        # MDB3416
        "mann": re.compile(r"\b(W\s?\d{2,4}\s?/\s?\d{1,4})\b"),  # W 712/83 (require the slash)
        "filtron": re.compile(r"\b(OP\s?\d{3,5}\s?/\s?\d{1,3})\b"),  # OP 616/3
    }

    for brand, pattern in _PATTERNS.items():
        for m in pattern.finditer(text):
            part = m.group(0).strip()
            # Reconstruct canonical form
            if brand == "brembo":
                part = f"P{m.group(1)}"
            elif brand == "trw":
                part = f"GDB{m.group(1)}"
            elif brand == "ferodo":
                part = f"FDB{m.group(1)}"
            elif brand == "mintex":
                part = f"MDB{m.group(1)}"
            elif brand == "bosch":
                part = re.sub(r"\s+", "", part)  # Remove spaces: "0 986 495348" -> "0986495348"

            result.found_brands.setdefault(brand, [])
            if part not in result.found_brands[brand]:
                result.found_brands[brand].append(part)


# --- CSV Comparison ----------------------------------------------------------

def _detect_category(csv_path: Path, headers: list[str]) -> str:
    """Detect the CSV category from filename and headers."""
    name = csv_path.stem.lower()
    if "brake" in name or "axle" in [h.lower() for h in headers]:
        return "brake_pad"
    if "filter" in name:
        return "filter"
    if "bulb" in name:
        return "bulb"
    return "unknown"


def _get_brake_pad_brands(row: dict) -> dict[str, str]:
    """Extract brand columns from brake pad CSV row."""
    return {
        "brembo": row.get("brembo", "").strip(),
        "trw": row.get("trw", "").strip(),
        "textar": row.get("textar", "").strip(),
        "ferodo": row.get("ferodo", "").strip(),
        "bosch": row.get("bosch", "").strip(),
        "mintex": row.get("mintex", "").strip(),
    }


def _get_filter_brands(row: dict, filter_type: str) -> dict[str, str]:
    """Extract brand columns from filter CSV row."""
    prefix = filter_type  # "oil_filter", "air_filter", "cabin_filter"
    return {
        "mann": row.get(f"{prefix}_mann", "").strip(),
        "filtron": row.get(f"{prefix}_filtron", "").strip(),
    }


def compare_row(
    row: dict,
    row_num: int,
    online: CrossRefResult,
    category: str,
    filter_type: str = "",
) -> VerificationRow:
    """Compare one CSV row against online cross-reference data."""
    make = row.get("make", "")
    model = row.get("model", "")
    oem_pn = row.get("oem_part_number", "") or row.get(f"{filter_type}_oem", "")

    if category == "brake_pad":
        csv_brands = _get_brake_pad_brands(row)
        cat_label = f"{row.get('axle', 'unknown')}_brake_pad"
    else:
        csv_brands = _get_filter_brands(row, filter_type)
        cat_label = filter_type

    vrow = VerificationRow(
        csv_row_num=row_num,
        make=make,
        model=model,
        oem_pn=oem_pn,
        category=cat_label,
    )

    for brand, csv_val in csv_brands.items():
        if not csv_val:
            continue
        online_vals = online.found_brands.get(brand, [])
        if not online_vals:
            status = "NOT_FOUND_ONLINE"
        elif csv_val in online_vals:
            status = "MATCH"
        else:
            # Check without spaces/dashes for fuzzy match
            csv_clean = re.sub(r"[\s\-/]", "", csv_val).upper()
            online_clean = [re.sub(r"[\s\-/]", "", v).upper() for v in online_vals]
            if csv_clean in online_clean:
                status = "MATCH"
            else:
                status = "MISMATCH"

        vrow.checks[brand] = {
            "csv": csv_val,
            "online": online_vals,
            "status": status,
        }

    return vrow


# --- Main --------------------------------------------------------------------

def verify_csv(csv_path: Path, delay: float, filter_type: str = "") -> list[VerificationRow]:
    """Verify all rows in a CSV file against online cross-references."""
    results: list[VerificationRow] = []
    seen_oem: dict[str, CrossRefResult] = {}  # Cache: same OEM number → same result

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        category = _detect_category(csv_path, headers)

        rows = list(reader)
        total = len(rows)

        for i, row in enumerate(rows, start=2):
            # Get OEM part number
            if category == "brake_pad":
                oem_pn = row.get("oem_part_number", "").strip()
            else:
                oem_pn = row.get(f"{filter_type}_oem", "").strip()
                if not oem_pn:
                    continue

            make = row.get("make", "")
            model = row.get("model", "")
            axle = row.get("axle", "")
            print(
                f"  [{i-1}/{total}] {make} {model}"
                f"{' ' + axle if axle else ''}"
                f" - OEM: {oem_pn}",
                end="",
                flush=True,
            )

            # Query (or use cache)
            if oem_pn in seen_oem:
                online = seen_oem[oem_pn]
                print(" (cached)")
            else:
                online = query_spareto(oem_pn, delay=delay)
                seen_oem[oem_pn] = online
                brands_found = len(online.found_brands)
                if online.error:
                    print(f" ERROR: {online.error}")
                else:
                    print(f" -> {online.total_products} products, {brands_found} brands")
                time.sleep(delay)

            vrow = compare_row(row, i, online, category, filter_type)
            results.append(vrow)

    return results


def print_report(results: list[VerificationRow], csv_path: Path) -> None:
    """Print a summary report of verification results."""
    print(f"\n{'='*70}")
    print(f"VERIFICATION REPORT: {csv_path.name}")
    print(f"{'='*70}")

    total = len(results)
    all_checks = 0
    matches = 0
    mismatches = 0
    not_found = 0
    mismatch_details = []

    for r in results:
        for brand, check in r.checks.items():
            all_checks += 1
            if check["status"] == "MATCH":
                matches += 1
            elif check["status"] == "MISMATCH":
                mismatches += 1
                mismatch_details.append(r)
            elif check["status"] == "NOT_FOUND_ONLINE":
                not_found += 1

    print(f"\nRows checked:       {total}")
    print(f"Cross-refs checked: {all_checks}")
    print(f"  MATCH:            {matches}")
    print(f"  MISMATCH:         {mismatches}")
    print(f"  NOT FOUND ONLINE: {not_found}")

    if mismatches > 0:
        print(f"\n{'─'*70}")
        print("MISMATCHES (require manual verification):")
        print(f"{'─'*70}")
        seen = set()
        for r in results:
            for brand, check in r.checks.items():
                if check["status"] == "MISMATCH":
                    key = (r.oem_pn, brand)
                    if key in seen:
                        continue
                    seen.add(key)
                    print(f"\n  Row {r.csv_row_num}: {r.make} {r.model} [{r.category}]")
                    print(f"    OEM:     {r.oem_pn}")
                    print(f"    Brand:   {brand.upper()}")
                    print(f"    CSV:     {check['csv']}")
                    print(f"    Online:  {check['online']}")

    # Save detailed results as JSON
    report_path = csv_path.parent / f"{csv_path.stem}_verification.json"
    report_data = []
    for r in results:
        report_data.append({
            "row": r.csv_row_num,
            "make": r.make,
            "model": r.model,
            "oem_pn": r.oem_pn,
            "category": r.category,
            "checks": r.checks,
        })
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify aftermarket cross-references against online catalogs"
    )
    parser.add_argument("csv_file", type=Path, help="CSV file to verify")
    parser.add_argument(
        "--category", default="",
        help="Filter type for filter CSVs: oil_filter, air_filter, cabin_filter"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Delay between HTTP requests in seconds (default: 2.0)"
    )
    args = parser.parse_args()

    csv_path = args.csv_file
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path

    print(f"Verifying: {csv_path.name}")
    print(f"Delay: {args.delay}s between requests")
    print()

    results = verify_csv(csv_path, args.delay, args.category)
    print_report(results, csv_path)


if __name__ == "__main__":
    _fix_encoding()
    main()
