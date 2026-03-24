import argparse
import re
import shutil
import sys
from datetime import date
from pathlib import Path


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "task"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a new task folder from templates.")
    parser.add_argument("title", nargs="+", help="Task title for the folder slug.")
    args = parser.parse_args()

    title = " ".join(args.title)
    slug = slugify(title)

    repo_root = Path(__file__).resolve().parent.parent
    template_dir = repo_root / "tasks" / "_template"
    if not template_dir.exists():
        print(f"Missing template directory: {template_dir}")
        return 1

    date_str = date.today().isoformat()
    task_dir = repo_root / "tasks" / f"{date_str}_{slug}"
    if task_dir.exists():
        print(f"Task folder already exists: {task_dir}")
        return 1

    task_dir.mkdir(parents=True)

    created = []
    for template in sorted(template_dir.glob("*.md")):
        dest = task_dir / template.name
        shutil.copyfile(template, dest)
        created.append(dest)

    print(f"Created task folder: {task_dir}")
    for path in created:
        try:
            rel = path.relative_to(repo_root)
            print(f" - {rel}")
        except ValueError:
            print(f" - {path}")

    print("Next steps:")
    print("  1) Spec Refiner fills 00_task_brief.md")
    print("  2) Implementer writes 10_impl_notes.md")
    print("  3) Verifier writes 20_verification.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
