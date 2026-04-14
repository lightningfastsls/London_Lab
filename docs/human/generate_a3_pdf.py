#!/usr/bin/env python3
"""Generate PDF for A3 acoustic feature deep-dive report with embedded figures."""

import base64
from pathlib import Path

import markdown
import weasyprint

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_MD = PROJECT_ROOT / "docs" / "human" / "a3-acoustic-feature-deep-dive.md"
OUTPUT_PDF = PROJECT_ROOT / "docs" / "human" / "a3-acoustic-feature-deep-dive.pdf"
RESULTS_DIR = PROJECT_ROOT / "results" / "acoustic_feature_analysis"

# Figures to embed after specific marker lines in the markdown
# Key = marker text that appears on its own line, Value = list of (filename, caption)
FIGURE_INSERTIONS = {
    "### Figure: Correlation Matrix": [
        ("correlation_matrix.png", "Acoustic Feature Correlation Matrix (hierarchically clustered)"),
    ],
    "### Figure: PCA Scree Plot": [
        ("pca_scree.png", "PCA Scree Plot — variance explained per component"),
    ],
    "### Figure: PCA Biplot": [
        ("pca_biplot.png", "PCA Biplot — PC1 vs PC2, colored by syllable type, with feature loading arrows"),
    ],
    "### Figure: UMAP by Syllable Type": [
        ("umap_by_type.png", "UMAP of 10 acoustic features — colored by syllable type"),
    ],
    "### Figure: UMAP by Features": [
        ("umap_by_feature.png", "UMAP colored by individual features (duration, slope, sinuosity, bandwidth, mean power, tonality)"),
    ],
    "### Figure: Within-Type Violin Plots": [
        ("within_type_violins.png", "Within-type feature distributions — red dashed lines mark classification thresholds"),
    ],
    "### Figure: Boundary Cases": [
        ("boundary_cases.png", "Boundary case analysis — low-confidence calls on UMAP (left) and feature distributions (right)"),
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

    # Replace figure marker lines with embedded images
    for marker, figures in FIGURE_INSERTIONS.items():
        if marker in md_text:
            figure_html = "\n".join(
                embed_image(RESULTS_DIR / filename, caption)
                for filename, caption in figures
            )
            md_text = md_text.replace(marker, figure_html)

    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code"],
    )

    # Wrap in full HTML with styling (same style as progress report)
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
