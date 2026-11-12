"""report.md -> report.pdf (CommonMark + GFM tables via markdown-it-py, rendered by WeasyPrint).

Usage (from the repo root, in any venv with `pip install weasyprint markdown-it-py`):
    python report/build_pdf.py report/report.md report/report.pdf
"""
import re, sys
from pathlib import Path
from markdown_it import MarkdownIt
from weasyprint import HTML

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False}).enable("table")
body = md.render(src.read_text())

# Links to repository files mean nothing inside a PDF: keep the text, drop the link.
body = re.sub(r'<a href="\.\./[^"]*">(.*?)</a>', r"\1", body, flags=re.S)
# Keep a figure (image + its caption paragraph) on one page.
body = re.sub(r"<h3>(Figure [^<]*)</h3>\s*<p>(<img [^>]*>)</p>\s*<p>(.*?)</p>",
              r'<figure><h3>\1</h3>\2<figcaption>\3</figcaption></figure>', body, flags=re.S)

css = """
@page { size: A4; margin: 20mm 17mm 20mm 17mm;
        @bottom-center { content: counter(page) " / " counter(pages); font: 8pt 'DejaVu Sans'; color: #666; } }
body { font: 9.6pt/1.42 'DejaVu Serif', serif; color: #111; }
h1 { font: bold 19pt 'DejaVu Sans'; margin: 0 0 4pt; }
h2 { font: bold 13pt 'DejaVu Sans'; margin: 16pt 0 5pt; border-bottom: 0.6pt solid #999; padding-bottom: 2pt;
     break-after: avoid; }
h3 { font: bold 10.5pt 'DejaVu Sans'; margin: 11pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 5pt; }
ul, ol { margin: 0 0 6pt; padding-left: 16pt; }
li { margin-bottom: 2pt; }
hr { border: 0; margin: 4pt 0; }
code { font: 8.4pt 'DejaVu Sans Mono', monospace; background: #f2f2f2; padding: 0 1.5pt; border-radius: 2pt; }
pre { background: #f6f6f6; border: 0.5pt solid #ddd; border-radius: 3pt; padding: 5pt 7pt; margin: 4pt 0 7pt;
      white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 7.6pt; line-height: 1.3; }
table { border-collapse: collapse; margin: 4pt 0 9pt; font: 8pt/1.3 'DejaVu Sans'; width: auto; max-width: 100%;
        break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 0.5pt solid #bbb; padding: 2pt 4.5pt; vertical-align: top; }
th { background: #ececec; text-align: left; }
td code, th code { font-size: 7.4pt; }
figure { break-inside: avoid; margin: 10pt 0 12pt; }
figure img { width: 100%; }
figcaption { font: 8.8pt/1.4 'DejaVu Serif'; color: #333; margin-top: 3pt; }
strong { font-weight: bold; }
"""
html = f'<!doctype html><html><head><meta charset="utf-8"><title>Batch Scaling of Qwen3-4B Inference</title>' \
       f'<style>{css}</style></head><body>{body}</body></html>'
HTML(string=html, base_url=str(src.parent)).write_pdf(out)
print("wrote", out)
