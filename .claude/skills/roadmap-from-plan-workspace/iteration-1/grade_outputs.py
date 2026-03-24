"""
Programmatic grading of roadmap-from-plan skill outputs.
Checks structural assertions against generated ROADMAP files.
Writes grading.json to each run directory.
"""
import json
import re
from pathlib import Path

WORKSPACE = Path(__file__).parent

def count_implement_blocks(text: str) -> int:
    """Count /implement occurrences (the command, not references to it)."""
    # Match lines starting with /implement (the actual command blocks)
    return len(re.findall(r'^/implement\b', text, re.MULTILINE))

def has_review_tier(text: str, tier: int = None) -> bool:
    """Check if Review Tier is present, optionally for a specific tier number."""
    if tier is not None:
        return bool(re.search(rf'\*\*Review Tier:\*\*\s*{tier}', text))
    return bool(re.search(r'\*\*Review Tier:\*\*\s*[123]', text))

def count_review_tiers(text: str) -> int:
    """Count how many modules have Review Tier assignments."""
    return len(re.findall(r'\*\*Review Tier:\*\*\s*[123]', text))

def has_adr_reference(text: str) -> bool:
    """Check for ADR-001 or sr=300000 references."""
    return bool(re.search(r'ADR-001|300[,.]?000|sr\s*=\s*300000', text))

def has_test_plan(text: str) -> int:
    """Count Test plan sections."""
    return len(re.findall(r'\*\*Test plan:\*\*', text))

def has_phase_gate(text: str) -> bool:
    """Check for Phase Gate section."""
    return bool(re.search(r'Phase\s+\d+\s+Gate|## Phase.*Gate', text, re.IGNORECASE))

def has_kg_mention(text: str) -> bool:
    """Check for KG extraction / theoretical content mention."""
    return bool(re.search(r'knowledge graph|/reduce|theoretical|KG extraction', text, re.IGNORECASE))

def has_dataclass(text: str) -> bool:
    """Check for Python dataclass code snippet."""
    return bool(re.search(r'@dataclass|class\s+\w+Config', text))

def has_dependency_reference(text: str, from_step: str = None) -> bool:
    """Check if dependencies are noted."""
    return bool(re.search(r'\*\*Depends on:\*\*\s*(?!None)', text))

def has_backwards_compat(text: str) -> bool:
    """Check for backwards compatibility mention."""
    return bool(re.search(r'backward|backwards|compat|thin wrapper|wrapper', text, re.IGNORECASE))

def has_gap_detection(text: str) -> bool:
    """Check for [ASSUMED] markers or gap flagging."""
    return bool(re.search(r'\[ASSUMED\]|gap|under-specified|underspecified|vague|clarif|assumption', text, re.IGNORECASE))

def count_test_items_per_plan(text: str) -> list:
    """For each Test plan section, count the numbered items."""
    sections = re.split(r'\*\*Test plan:\*\*', text)
    counts = []
    for section in sections[1:]:  # skip text before first test plan
        # Count numbered items (1., 2., etc.) up to the next ** section
        next_section = re.search(r'\*\*Exit criteria|\*\*\w+:', section)
        if next_section:
            section = section[:next_section.start()]
        items = re.findall(r'^\s*\d+\.', section, re.MULTILINE)
        counts.append(len(items))
    return counts


def grade_ml_feature_extraction(text: str) -> list:
    """Grade assertions for the ML feature extraction test case."""
    n_implement = count_implement_blocks(text)
    n_tiers = count_review_tiers(text)
    n_test_plans = has_test_plan(text)

    return [
        {
            "text": "A1: Output contains exactly 4 /implement blocks",
            "passed": n_implement == 4,
            "evidence": f"Found {n_implement} /implement blocks"
        },
        {
            "text": "A2: Every module has a Review Tier assignment",
            "passed": n_tiers >= 4,
            "evidence": f"Found {n_tiers} Review Tier assignments (expected 4)"
        },
        {
            "text": "A3: At least one module is Tier 3 (DSP/ML content)",
            "passed": has_review_tier(text, 3),
            "evidence": "Tier 3 found" if has_review_tier(text, 3) else "No Tier 3 found"
        },
        {
            "text": "A4: References ADR-001 or sr=300000",
            "passed": has_adr_reference(text),
            "evidence": "ADR/sample rate reference found" if has_adr_reference(text) else "No DSP parameter reference"
        },
        {
            "text": "A5: Every module has a Test plan section",
            "passed": n_test_plans >= 4,
            "evidence": f"Found {n_test_plans} Test plan sections (expected 4)"
        },
        {
            "text": "A6: Contains a Phase Gate section",
            "passed": has_phase_gate(text),
            "evidence": "Phase Gate found" if has_phase_gate(text) else "No Phase Gate section"
        },
        {
            "text": "A7: Mentions theoretical content / KG extraction",
            "passed": has_kg_mention(text),
            "evidence": "KG/theoretical mention found" if has_kg_mention(text) else "No KG extraction mention"
        },
        {
            "text": "A8: Contains Python dataclass/config code snippet",
            "passed": has_dataclass(text),
            "evidence": "Dataclass definition found" if has_dataclass(text) else "No dataclass code snippet"
        }
    ]


def grade_cli_consolidation(text: str) -> list:
    """Grade assertions for the CLI consolidation test case."""
    n_implement = count_implement_blocks(text)

    return [
        {
            "text": "B1: Output contains exactly 3 /implement blocks",
            "passed": n_implement == 3,
            "evidence": f"Found {n_implement} /implement blocks"
        },
        {
            "text": "B2: Step 2 explicitly depends on Step 1",
            "passed": has_dependency_reference(text),
            "evidence": "Dependency reference found" if has_dependency_reference(text) else "No dependency chain found"
        },
        {
            "text": "B3: No modules assigned Tier 3",
            "passed": not has_review_tier(text, 3),
            "evidence": "No Tier 3 (correct)" if not has_review_tier(text, 3) else "Tier 3 found (incorrect for infra work)"
        },
        {
            "text": "B4: Mentions backwards compatibility",
            "passed": has_backwards_compat(text),
            "evidence": "Backwards compat mentioned" if has_backwards_compat(text) else "No backwards compat mention"
        },
        {
            "text": "B5: Contains a Phase Gate section",
            "passed": has_phase_gate(text),
            "evidence": "Phase Gate found" if has_phase_gate(text) else "No Phase Gate section"
        }
    ]


def grade_streaming_viewer(text: str) -> list:
    """Grade assertions for the streaming viewer (vague plan) test case."""
    n_implement = count_implement_blocks(text)
    test_counts = count_test_items_per_plan(text)
    min_tests = min(test_counts) if test_counts else 0

    return [
        {
            "text": "C1: Output contains 4 /implement blocks",
            "passed": n_implement == 4,
            "evidence": f"Found {n_implement} /implement blocks"
        },
        {
            "text": "C2: Contains [ASSUMED] markers OR explicit gap flagging",
            "passed": has_gap_detection(text),
            "evidence": "Gap detection found" if has_gap_detection(text) else "No gap flagging (critical miss for vague plan)"
        },
        {
            "text": "C3: References ADR-001 or sr=300000",
            "passed": has_adr_reference(text),
            "evidence": "ADR/sample rate reference found" if has_adr_reference(text) else "No DSP parameter reference"
        },
        {
            "text": "C4: At least one module is Tier 3",
            "passed": has_review_tier(text, 3),
            "evidence": "Tier 3 found" if has_review_tier(text, 3) else "No Tier 3 (real-time DSP should be Tier 3)"
        },
        {
            "text": "C5: Test plans have >= 3 specific test cases per module",
            "passed": min_tests >= 3 if test_counts else False,
            "evidence": f"Test counts per module: {test_counts}" if test_counts else "No test plans found"
        }
    ]


def grade_run(eval_name: str, run_type: str):
    """Grade a single run and write grading.json."""
    run_dir = WORKSPACE / eval_name / run_type / "outputs"

    # Find the ROADMAP file
    roadmap_files = list(run_dir.glob("ROADMAP_*.md"))
    if not roadmap_files:
        print(f"  SKIP: No ROADMAP file found in {run_dir}")
        return None

    text = roadmap_files[0].read_text(encoding="utf-8")

    # Select grading function
    if eval_name == "ml-feature-extraction":
        expectations = grade_ml_feature_extraction(text)
    elif eval_name == "cli-consolidation":
        expectations = grade_cli_consolidation(text)
    elif eval_name == "streaming-viewer":
        expectations = grade_streaming_viewer(text)
    else:
        print(f"  SKIP: Unknown eval {eval_name}")
        return None

    passed = sum(1 for e in expectations if e["passed"])
    total = len(expectations)

    result = {
        "eval_name": eval_name,
        "run_type": run_type,
        "source_file": roadmap_files[0].name,
        "expectations": expectations,
        "summary": {
            "passed": passed,
            "failed": total - passed,
            "total": total,
            "pass_rate": round(passed / total, 2) if total > 0 else 0
        }
    }

    # Write grading.json
    grading_path = WORKSPACE / eval_name / run_type / "grading.json"
    grading_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  {eval_name}/{run_type}: {passed}/{total} passed ({result['summary']['pass_rate']:.0%})")

    return result


def main():
    print("=== Grading roadmap-from-plan outputs ===\n")

    all_results = []
    for eval_name in ["ml-feature-extraction", "cli-consolidation", "streaming-viewer"]:
        for run_type in ["with_skill", "old_skill"]:
            print(f"Grading {eval_name} / {run_type}...")
            result = grade_run(eval_name, run_type)
            if result:
                all_results.append(result)

    # Summary
    print("\n=== Summary ===")
    for result in all_results:
        status = "NEW" if result["run_type"] == "with_skill" else "OLD"
        s = result['summary']
        print(f"  [{status}] {result['eval_name']}: {s['passed']}/{s['total']} ({s['pass_rate']:.0%})")

    # Write aggregate
    aggregate_path = WORKSPACE / "grading_summary.json"
    aggregate_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nAggregate written to {aggregate_path}")


if __name__ == "__main__":
    main()
