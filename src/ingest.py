"""
Phase 2 — Document Ingestion Pipeline for Multimodal RAG (financial reports)

Extracts three kinds of content from each PDF in data/raw_pdfs/:
  1. Text        -> per-page plain text
  2. Tables      -> serialized to markdown (kept in the TEXT retrieval path,
                    not treated as images -- tables are structured data and
                    embed/retrieve far better as text than as pictures)
  3. Images      -> two sources, both saved:
       - "embedded": raster images (photos, logos, some charts) pulled
                      directly out of the PDF's image objects
       - "page_render": a full-page screenshot of every page

Why both image sources? Charts built in Excel/PowerPoint/matplotlib and
placed into a PDF are very often *vector* graphics -- lines and rectangles
drawn directly on the page, not embedded image objects. Vector charts are
INVISIBLE to raster image extraction (page.get_images() will simply never
find them). Rendering the whole page as a bitmap is the only reliable way
to guarantee a chart gets captured regardless of how it was created. This
is exactly the kind of silent gap that makes RAG systems miss information
without ever raising an error -- so we don't rely on raster extraction alone.

Output:
  data/processed/text_chunks.jsonl   (chunk_type: "text" or "table")
  data/processed/image_assets.jsonl  (source: "embedded" or "page_render")
  data/processed/images/             (extracted embedded raster images)
  data/processed/page_renders/       (full-page PNG renders)
"""

# import json
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import List, Tuple

# import pymupdf
# import pdfplumber

# RAW_PDF_DIR = Path("data/raw_pdfs")
# OUTPUT_DIR = Path("data/processed")
# IMAGES_DIR = OUTPUT_DIR / "images"
# PAGE_RENDERS_DIR = OUTPUT_DIR / "page_renders"

# MIN_EMBEDDED_IMAGE_BYTES = 5000  # filters out tiny icons/decorative dots
# RENDER_DPI = 150


# @dataclass
# class TextChunk:
#     doc_id: str
#     page_number: int
#     chunk_type: str  # "text" | "table"
#     content: str
#     chunk_id: str


# @dataclass
# class ImageAsset:
#     doc_id: str
#     page_number: int
#     image_id: str
#     file_path: str
#     source: str  # "embedded" | "page_render"


# def extract_text_and_images(pdf_path: Path) -> Tuple[List[TextChunk], List[ImageAsset]]:
#     doc_id = pdf_path.stem
#     text_chunks: List[TextChunk] = []
#     image_assets: List[ImageAsset] = []

#     pdf_doc = pymupdf.open(pdf_path)

#     for page_index, page in enumerate(pdf_doc):
#         page_number = page_index + 1

#         # ---- text ----
#         text = page.get_text("text").strip()
#         if text:
#             text_chunks.append(TextChunk(
#                 doc_id=doc_id,
#                 page_number=page_number,
#                 chunk_type="text",
#                 content=text,
#                 chunk_id=f"{doc_id}_p{page_number}_text",
#             ))

#         # ---- embedded raster images ----
#         for img_index, img in enumerate(page.get_images(full=True)):
#             xref = img[0]
#             base_image = pdf_doc.extract_image(xref)
#             image_bytes = base_image["image"]
#             if len(image_bytes) < MIN_EMBEDDED_IMAGE_BYTES:
#                 continue  # likely an icon, bullet, or decorative element

#             IMAGES_DIR.mkdir(parents=True, exist_ok=True)
#             filename = f"{doc_id}_p{page_number}_img{img_index}.{base_image['ext']}"
#             out_path = IMAGES_DIR / filename
#             out_path.write_bytes(image_bytes)

#             image_assets.append(ImageAsset(
#                 doc_id=doc_id,
#                 page_number=page_number,
#                 image_id=filename,
#                 file_path=str(out_path),
#                 source="embedded",
#             ))

#         # ---- full-page render (catches vector-drawn charts too) ----
#         PAGE_RENDERS_DIR.mkdir(parents=True, exist_ok=True)
#         pix = page.get_pixmap(dpi=RENDER_DPI)
#         render_filename = f"{doc_id}_p{page_number}_render.png"
#         render_path = PAGE_RENDERS_DIR / render_filename
#         pix.save(str(render_path))

#         image_assets.append(ImageAsset(
#             doc_id=doc_id,
#             page_number=page_number,
#             image_id=render_filename,
#             file_path=str(render_path),
#             source="page_render",
#         ))

#     pdf_doc.close()
#     return text_chunks, image_assets


# def rows_to_markdown(table: List[List[str]]) -> str:
#     rows = [[cell if cell else "" for cell in row] for row in table]
#     header, body = rows[0], rows[1:]
#     lines = [
#         "| " + " | ".join(header) + " |",
#         "| " + " | ".join(["---"] * len(header)) + " |",
#     ]
#     lines += ["| " + " | ".join(row) + " |" for row in body]
#     return "\n".join(lines)


# def extract_tables(pdf_path: Path) -> List[TextChunk]:
#     """Tables are structured data, not pictures -- serialize to markdown so
#     they embed and retrieve through the same text pipeline as narrative
#     chunks, rather than needing a vision model just to read a number."""
#     doc_id = pdf_path.stem
#     table_chunks: List[TextChunk] = []

#     with pdfplumber.open(pdf_path) as pdf:
#         for page_index, page in enumerate(pdf.pages):
#             page_number = page_index + 1
#             for t_index, table in enumerate(page.extract_tables()):
#                 if not table or len(table) < 2:
#                     continue
#                 table_chunks.append(TextChunk(
#                     doc_id=doc_id,
#                     page_number=page_number,
#                     chunk_type="table",
#                     content=rows_to_markdown(table),
#                     chunk_id=f"{doc_id}_p{page_number}_table{t_index}",
#                 ))
#     return table_chunks


# def process_all_pdfs() -> None:
#     pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf"))
#     if not pdf_files:
#         print(f"No PDFs found in {RAW_PDF_DIR}/ -- drop your 10-K / annual report PDFs there and re-run.")
#         return

#     all_text_chunks: List[TextChunk] = []
#     all_image_assets: List[ImageAsset] = []

#     for pdf_path in pdf_files:
#         print(f"Processing {pdf_path.name} ...")
#         text_chunks, image_assets = extract_text_and_images(pdf_path)
#         table_chunks = extract_tables(pdf_path)

#         all_text_chunks.extend(text_chunks)
#         all_text_chunks.extend(table_chunks)
#         all_image_assets.extend(image_assets)

#         n_embedded = sum(1 for a in image_assets if a.source == "embedded")
#         n_renders = sum(1 for a in image_assets if a.source == "page_render")
#         print(f"  -> {len(text_chunks)} text chunks, {len(table_chunks)} tables, "
#               f"{n_embedded} embedded images, {n_renders} page renders")

#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#     with open(OUTPUT_DIR / "text_chunks.jsonl", "w") as f:
#         for chunk in all_text_chunks:
#             f.write(json.dumps(asdict(chunk)) + "\n")

#     with open(OUTPUT_DIR / "image_assets.jsonl", "w") as f:
#         for asset in all_image_assets:
#             f.write(json.dumps(asdict(asset)) + "\n")

#     print(f"\nDone. {len(all_text_chunks)} text/table chunks, "
#           f"{len(all_image_assets)} image assets -> {OUTPUT_DIR}/")


# if __name__ == "__main__":
#     process_all_pdfs()


"""
Phase 2 — Document Ingestion Pipeline for Multimodal RAG (financial reports)

Extracts three kinds of content from each PDF in data/raw_pdfs/:
  1. Text        -> per-page plain text
  2. Tables      -> serialized to markdown (kept in the TEXT retrieval path,
                    not treated as images -- tables are structured data and
                    embed/retrieve far better as text than as pictures)
  3. Images      -> two sources, both saved:
       - "embedded": raster images (photos, logos, some charts) pulled
                      directly out of the PDF's image objects
       - "page_render": a full-page screenshot of every page

Why both image sources? Charts built in Excel/PowerPoint/matplotlib and
placed into a PDF are very often *vector* graphics -- lines and rectangles
drawn directly on the page, not embedded image objects. Vector charts are
INVISIBLE to raster image extraction (page.get_images() will simply never
find them). Rendering the whole page as a bitmap is the only reliable way
to guarantee a chart gets captured regardless of how it was created. This
is exactly the kind of silent gap that makes RAG systems miss information
without ever raising an error -- so we don't rely on raster extraction alone.

Output:
  data/processed/text_chunks.jsonl   (chunk_type: "text" or "table")
  data/processed/image_assets.jsonl  (source: "embedded" or "page_render")
  data/processed/images/             (extracted embedded raster images)
  data/processed/page_renders/       (full-page PNG renders)
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import pymupdf
import pdfplumber

RAW_PDF_DIR = Path("data/raw_pdfs")
OUTPUT_DIR = Path("data/processed")
IMAGES_DIR = OUTPUT_DIR / "images"
PAGE_RENDERS_DIR = OUTPUT_DIR / "page_renders"
# KNOWN LIMITATION: these paths are shared across all users. Two different
# users uploading a same-named PDF will overwrite each other's extracted
# image files on disk (Chroma's user_id-tagged metadata still keeps the
# two users' TEXT isolated correctly, but an overwritten image file means
# one of them would see the other's picture). Low-probability for the
# current "just me / people I send it to" usage, but real. The clean fix
# is namespacing these paths by user_id once file storage moves to
# Supabase Storage in a later step -- not worth a local-disk-only patch
# for a problem that move solves properly anyway.

MIN_EMBEDDED_IMAGE_BYTES = 5000  # filters out tiny icons/decorative dots
RENDER_DPI = 150


@dataclass
class TextChunk:
    doc_id: str
    page_number: int
    chunk_type: str  # "text" | "table"
    content: str
    chunk_id: str


@dataclass
class ImageAsset:
    doc_id: str
    page_number: int
    image_id: str
    file_path: str
    source: str  # "embedded" | "page_render"


def extract_text_and_images(pdf_path: Path) -> Tuple[List[TextChunk], List[ImageAsset]]:
    doc_id = pdf_path.stem
    text_chunks: List[TextChunk] = []
    image_assets: List[ImageAsset] = []

    pdf_doc = pymupdf.open(pdf_path)

    for page_index, page in enumerate(pdf_doc):
        page_number = page_index + 1

        # ---- text ----
        text = page.get_text("text").strip()
        if text:
            text_chunks.append(TextChunk(
                doc_id=doc_id,
                page_number=page_number,
                chunk_type="text",
                content=text,
                chunk_id=f"{doc_id}_p{page_number}_text",
            ))

        # ---- embedded raster images ----
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]
            if len(image_bytes) < MIN_EMBEDDED_IMAGE_BYTES:
                continue  # likely an icon, bullet, or decorative element

            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"{doc_id}_p{page_number}_img{img_index}.{base_image['ext']}"
            out_path = IMAGES_DIR / filename
            out_path.write_bytes(image_bytes)

            image_assets.append(ImageAsset(
                doc_id=doc_id,
                page_number=page_number,
                image_id=filename,
                file_path=str(out_path),
                source="embedded",
            ))

        # ---- full-page render (catches vector-drawn charts too) ----
        PAGE_RENDERS_DIR.mkdir(parents=True, exist_ok=True)
        pix = page.get_pixmap(dpi=RENDER_DPI)
        render_filename = f"{doc_id}_p{page_number}_render.png"
        render_path = PAGE_RENDERS_DIR / render_filename
        pix.save(str(render_path))

        image_assets.append(ImageAsset(
            doc_id=doc_id,
            page_number=page_number,
            image_id=render_filename,
            file_path=str(render_path),
            source="page_render",
        ))

    pdf_doc.close()
    return text_chunks, image_assets


def rows_to_markdown(table: List[List[str]]) -> str:
    rows = [[cell if cell else "" for cell in row] for row in table]
    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def extract_tables(pdf_path: Path) -> List[TextChunk]:
    """Tables are structured data, not pictures -- serialize to markdown so
    they embed and retrieve through the same text pipeline as narrative
    chunks, rather than needing a vision model just to read a number."""
    doc_id = pdf_path.stem
    table_chunks: List[TextChunk] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1
            for t_index, table in enumerate(page.extract_tables()):
                if not table or len(table) < 2:
                    continue
                table_chunks.append(TextChunk(
                    doc_id=doc_id,
                    page_number=page_number,
                    chunk_type="table",
                    content=rows_to_markdown(table),
                    chunk_id=f"{doc_id}_p{page_number}_table{t_index}",
                ))
    return table_chunks


def process_all_pdfs() -> None:
    pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {RAW_PDF_DIR}/ -- drop your 10-K / annual report PDFs there and re-run.")
        return

    all_text_chunks: List[TextChunk] = []
    all_image_assets: List[ImageAsset] = []

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name} ...")
        text_chunks, image_assets = extract_text_and_images(pdf_path)
        table_chunks = extract_tables(pdf_path)

        all_text_chunks.extend(text_chunks)
        all_text_chunks.extend(table_chunks)
        all_image_assets.extend(image_assets)

        n_embedded = sum(1 for a in image_assets if a.source == "embedded")
        n_renders = sum(1 for a in image_assets if a.source == "page_render")
        print(f"  -> {len(text_chunks)} text chunks, {len(table_chunks)} tables, "
              f"{n_embedded} embedded images, {n_renders} page renders")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "text_chunks.jsonl", "w") as f:
        for chunk in all_text_chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")

    with open(OUTPUT_DIR / "image_assets.jsonl", "w") as f:
        for asset in all_image_assets:
            f.write(json.dumps(asdict(asset)) + "\n")

    print(f"\nDone. {len(all_text_chunks)} text/table chunks, "
          f"{len(all_image_assets)} image assets -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    process_all_pdfs()