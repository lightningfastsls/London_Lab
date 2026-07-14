#!/usr/bin/env python
"""Render thesis_draft.md -> thesis_draft.html (clean academic style) and optional PDF.

Usage:
    .venv/bin/python docs/thesis/render.py            # -> thesis_draft.html
    .venv/bin/python docs/thesis/render.py --pdf      # also -> thesis_draft.pdf (WeasyPrint)

Re-run this after every edit to thesis_draft.md to refresh the readable output.
Figures are referenced relative to this folder as figures/figNN_*.png.
"""
import re
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "thesis_draft.md"
OUT_HTML = HERE / "thesis_draft.html"
OUT_PDF = HERE / "thesis_draft.pdf"

CSS = """
:root{--ink:#1a1a1a;--muted:#5b6470;--rule:#d8dce2;--accent:#7a1f2b;}
*{box-sizing:border-box;}
body{margin:0;background:#f4f5f7;color:var(--ink);
  font:11pt/1.5 "Georgia","Times New Roman",serif;}
.page{max-width:820px;margin:0 auto;background:#fff;padding:44px 70px 52px;
  box-shadow:0 1px 4px rgba(0,0,0,.08);}
h1{font-size:1.9rem;line-height:1.25;margin:0 0 .2em;}
h2{font-size:1.3rem;margin:1.5em 0 .5em;padding-bottom:.25em;border-bottom:1px solid var(--rule);}
h3{font-size:1.06rem;margin:1.2em 0 .35em;color:#222;}
p{text-align:justify;margin:0 0 .7em;}
em{color:#333;}
a{color:var(--accent);text-decoration:none;}
code{font-family:"SF Mono",ui-monospace,Consolas,monospace;font-size:.85em;
  background:#f0f1f4;padding:1px 5px;border-radius:4px;}
pre{background:#f6f7f9;border:1px solid var(--rule);border-radius:8px;padding:12px 14px;
  overflow-x:auto;font-size:.82em;}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92em;}
th,td{border:1px solid var(--rule);padding:6px 10px;text-align:left;}
th{background:#f0f1f4;}
figure{margin:0.9em 0;text-align:center;page-break-inside:avoid;}
figure img{max-width:66%;max-height:210px;height:auto;width:auto;
  border:1px solid var(--rule);border-radius:6px;}
figure.tall img{max-width:62%;max-height:330px;}
figure.wide img{max-width:84%;max-height:235px;}
figcaption{font-size:.78rem;color:var(--muted);margin-top:.35em;text-align:left;
  line-height:1.4;padding:0 4px;}
blockquote{border-left:3px solid var(--accent);margin:1em 0;padding:.2em 1em;color:#444;
  background:#faf7f7;}
hr{border:none;border-top:1px solid var(--rule);margin:2em 0;}
.titleblock{margin-bottom:1.2em;}
.titleblock p{text-align:center;color:var(--muted);margin:.2em 0;}
@media print{
  body{background:#fff;font-size:11pt;line-height:1.5;}
  .page{box-shadow:none;max-width:none;padding:0;margin:0;}
  h2{page-break-after:avoid;} figure{page-break-inside:avoid;}
}
"""

FIG_RE = re.compile(r'<p>\s*(<img[^>]*alt="([^"]*)"[^>]*>)\s*</p>')
WIDE = {"fig01", "fig03", "fig04", "fig06", "fig07"}
TALL = {"fig05"}


def figurize(html: str) -> str:
    """Wrap standalone images in <figure> with the alt text as <figcaption>."""
    def repl(m):
        img, alt = m.group(1), m.group(2)
        src = re.search(r'src="figures/(fig\d+)', img)
        key = src.group(1) if src else ""
        cls = " class='wide'" if key in WIDE else (" class='tall'" if key in TALL else "")
        return f"<figure{cls}>{img}<figcaption>{alt}</figcaption></figure>"
    return FIG_RE.sub(repl, html)


def main():
    if not SRC.exists():
        sys.exit(f"Source not found: {SRC}")
    text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["extra", "tables", "sane_lists", "toc"],
    )
    body = figurize(body)
    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>USV Thesis: Draft</title>"
        f"<style>{CSS}</style></head><body><div class='page'>{body}</div></body></html>"
    )
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_HTML}")

    if "--pdf" in sys.argv:
        from weasyprint import HTML
        HTML(string=html, base_url=str(HERE)).write_pdf(str(OUT_PDF))
        print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
