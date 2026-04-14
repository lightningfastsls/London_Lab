#!/usr/bin/env python3
"""Generate PDF progress report from markdown with embedded figures."""

import base64
import re
from pathlib import Path

import markdown
import weasyprint

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_MD = PROJECT_ROOT / "docs" / "human" / "progress-report-2026-04-05.md"
OUTPUT_PDF = PROJECT_ROOT / "docs" / "human" / "progress-report-2026-04-05.pdf"

# Figures to embed after specific sections
FIGURE_INSERTIONS = {
    "### Figures produced": [
        ("results/sequential_structure/transition_matrix.png", "Transition Matrix P(B|A)"),
        ("results/sequential_structure/transition_matrix_within_bout.png", "Within-Bout Transition Matrix"),
        ("results/sequential_structure/entropy_convergence.png", "Entropy Rate Convergence"),
        ("results/sequential_structure/mutual_information_lag.png", "Mutual Information at Lag"),
        ("results/sequential_structure/zipf_distribution.png", "Zipf Rank-Frequency Distribution"),
    ],
    "## 3. Temporal Dynamics": [
        ("results/temporal_dynamics/call_raster.png", "Call Raster — Full Timeline"),
        ("results/temporal_dynamics/call_rate_hourly.png", "Hourly Call Rate"),
        ("results/temporal_dynamics/type_composition_hourly.png", "Type Composition Over Time"),
        ("results/temporal_dynamics/ici_distribution.png", "Inter-Call Interval Distribution"),
        ("results/temporal_dynamics/bout_structure.png", "Bout Structure"),
    ],
    "### C. UMAP + HDBSCAN": [
        ("results/recluster_umap_hdbscan/umap_hdbscan_scatter.png", "UMAP + HDBSCAN Clustering"),
    ],
}


def embed_image(path: Path, caption: str) -> str:
    """Convert image to base64-embedded HTML img tag."""
    if not path.exists():
        return f'<p><em>[Missing figure: {path.name}]</em></p>'
    data = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<figure style="text-align:center; margin: 1.5em 0;">'
        f'<img src="data:image/png;base64,{data}" '
        f'style="max-width:100%; height:auto;" />'
        f'<figcaption style="font-size:0.85em; color:#555; margin-top:0.3em;">'
        f'{caption}</figcaption></figure>'
    )


def main():
    md_text = REPORT_MD.read_text()

    # Cut after section 5 (Next Steps table) — stop before section 6
    cut_marker = "## 6. Tools & Infrastructure Built"
    idx = md_text.find(cut_marker)
    if idx != -1:
        md_text = md_text[:idx].rstrip()

    # Insert figures at appropriate locations
    for marker, figures in FIGURE_INSERTIONS.items():
        if marker in md_text:
            figure_html = "\n".join(
                embed_image(PROJECT_ROOT / rel_path, caption)
                for rel_path, caption in figures
            )
            # Insert after the marker's paragraph/section
            # Find end of the line containing the marker
            marker_pos = md_text.find(marker)
            # Find the next blank line or section after the marker block
            next_section = md_text.find("\n## ", marker_pos + 1)
            if next_section == -1:
                next_section = len(md_text)
            # For "Figures produced", replace the bullet list with actual figures
            if "Figures produced" in marker:
                # Find the start of the bullet list
                bullet_start = md_text.find("\n- `results/sequential", marker_pos)
                if bullet_start != -1:
                    md_text = (
                        md_text[:bullet_start]
                        + "\n\n"
                        + figure_html
                        + "\n\n"
                        + md_text[next_section:]
                    )
            else:
                # Insert figures right after the section heading's content block
                # Find the next ## heading after the marker
                insert_pos = md_text.find("\n## ", marker_pos + len(marker))
                if insert_pos == -1:
                    insert_pos = len(md_text)
                md_text = (
                    md_text[:insert_pos]
                    + "\n\n"
                    + figure_html
                    + "\n"
                    + md_text[insert_pos:]
                )

    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code"],
    )

    # Wrap in full HTML with styling
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{
        size: A4;
        margin: 2cm 2.5cm;
        @bottom-center {{
            content: counter(page);
            font-size: 9pt;
            color: #888;
        }}
    }}
    body {{
        font-family: 'Segoe UI', Calibri, Arial, sans-serif;
        font-size: 10.5pt;
        line-height: 1.5;
        color: #222;
    }}
    h1 {{
        font-size: 18pt;
        border-bottom: 2px solid #333;
        padding-bottom: 6pt;
        margin-bottom: 4pt;
    }}
    h2 {{
        font-size: 14pt;
        color: #1a3a5c;
        border-bottom: 1px solid #ccc;
        padding-bottom: 3pt;
        margin-top: 20pt;
        page-break-after: avoid;
    }}
    h3 {{
        font-size: 11.5pt;
        color: #2a5a2a;
        margin-top: 14pt;
        page-break-after: avoid;
    }}
    strong {{
        color: #111;
    }}
    ul {{
        margin-left: 0;
        padding-left: 1.5em;
    }}
    li {{
        margin-bottom: 4pt;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 10pt 0;
        font-size: 9.5pt;
    }}
    th {{
        background-color: #1a3a5c;
        color: white;
        padding: 6pt 8pt;
        text-align: left;
    }}
    td {{
        padding: 5pt 8pt;
        border-bottom: 1px solid #ddd;
    }}
    tr:nth-child(even) td {{
        background-color: #f5f5f5;
    }}
    figure {{
        page-break-inside: avoid;
    }}
    figcaption {{
        font-style: italic;
    }}
    hr {{
        border: none;
        border-top: 1px solid #ccc;
        margin: 12pt 0;
    }}
    code {{
        background-color: #f0f0f0;
        padding: 1pt 3pt;
        border-radius: 2pt;
        font-size: 9.5pt;
    }}
    ol {{
        padding-left: 1.5em;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    # Generate PDF
    doc = weasyprint.HTML(string=html)
    doc.write_pdf(str(OUTPUT_PDF))
    print(f"PDF saved to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
