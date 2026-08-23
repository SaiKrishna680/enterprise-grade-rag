"""
Generates a synthetic 3-page 'financial report' PDF to sanity-check the
ingestion pipeline:
  Page 1: narrative text + a data table
  Page 2: narrative text + an EMBEDDED RASTER image (a matplotlib chart
          saved as PNG, then placed into the PDF as an image object)
  Page 3: narrative text + a PURE VECTOR chart drawn with ReportLab's own
          drawing primitives (rects/lines) -> this will NOT show up via
          page.get_images(), only via full-page rasterization.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from pathlib import Path

OUT_DIR = Path(__file__).parent
CHART_PNG = OUT_DIR / "_tmp_chart.png"
OUT_PDF = Path("/home/claude/ragproject/data/raw_pdfs/test_report.pdf")

# --- build a small matplotlib chart to embed as a raster image on page 2 ---
fig, ax = plt.subplots(figsize=(4, 2.5))
ax.bar(["FY22", "FY23", "FY24"], [82.5, 91.3, 104.7])
ax.set_title("Revenue by Fiscal Year ($B)")
fig.tight_layout()
fig.savefig(CHART_PNG, dpi=150)
plt.close(fig)

styles = getSampleStyleSheet()
story = []

# ---------------- Page 1: text + table ----------------
story.append(Paragraph("Acme Corp — Annual Report FY2024", styles["Title"]))
story.append(Spacer(1, 12))
story.append(Paragraph(
    "This report summarizes Acme Corp's financial performance for fiscal "
    "year 2024. Revenue grew year over year driven by strong demand in "
    "the core segment. The following table breaks down revenue by segment.",
    styles["Normal"]))
story.append(Spacer(1, 12))

table_data = [
    ["Segment", "FY23 ($M)", "FY24 ($M)", "Growth"],
    ["Hardware", "412", "468", "+13.6%"],
    ["Software", "289", "355", "+22.8%"],
    ["Services", "156", "171", "+9.6%"],
]
t = Table(table_data)
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
]))
story.append(t)
story.append(PageBreak())

# ---------------- Page 2: text + embedded raster image ----------------
story.append(Paragraph("Revenue Trend", styles["Heading1"]))
story.append(Paragraph(
    "The chart below shows total company revenue over the past three "
    "fiscal years, illustrating consistent double-digit growth.",
    styles["Normal"]))
story.append(Spacer(1, 12))
story.append(Image(str(CHART_PNG), width=300, height=187))
story.append(PageBreak())

# ---------------- Page 3: text (vector chart added after via canvas) ----------------
story.append(Paragraph("Segment Margin Overview", styles["Heading1"]))
story.append(Paragraph(
    "Margins improved across all segments in FY24, with Software showing "
    "the largest expansion due to operating leverage.",
    styles["Normal"]))

doc = SimpleDocTemplate(str(OUT_PDF), pagesize=letter)
doc.build(story)

# --- now reopen with a raw canvas overlay to add a PURE VECTOR bar chart
# onto page 3 (this simulates charts exported from Excel/PowerPoint as
# vector graphics, which page.get_images() cannot see) ---
import fitz  # PyMuPDF, used only to append a vector-drawn page cleanly

vec_pdf_path = OUT_DIR / "_tmp_vector_page.pdf"
c = canvas.Canvas(str(vec_pdf_path), pagesize=letter)
c.setFont("Helvetica-Bold", 12)
c.drawString(72, 700, "Segment Margin (%) — vector-drawn chart, no embedded image")
bars = [("Hardware", 18), ("Software", 34), ("Services", 22)]
x = 100
for label, value in bars:
    c.setFillColor(colors.HexColor("#2563eb"))
    c.rect(x, 500, 60, value * 4, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.drawString(x, 490, label)
    x += 100
c.save()

# stitch: replace page 3 of OUT_PDF with the vector page, keep pages 1-2
base = fitz.open(str(OUT_PDF))
vec = fitz.open(str(vec_pdf_path))
final = fitz.open()
final.insert_pdf(base, from_page=0, to_page=0)   # page 1
final.insert_pdf(base, from_page=1, to_page=1)   # page 2
final.insert_pdf(vec, from_page=0, to_page=0)    # page 3 (vector chart only)
final.save(str(OUT_PDF), incremental=False)
final.close()
base.close()
vec.close()

CHART_PNG.unlink()
vec_pdf_path.unlink()

print(f"Test PDF written to {OUT_PDF} ({OUT_PDF.stat().st_size} bytes)")
