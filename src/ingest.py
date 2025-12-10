import os
import re
import json
from PyPDF2 import PdfReader
from collections import defaultdict

# Clean text for chunks
def clean_text(text):
    """
    Clean extracted text for ingestion:
    - Replace multiple dots with a single dot
    - Remove page headers/footers like 'Page 1 of 10'
    - Collapse multiple whitespaces into a single space
    """
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'Page \d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# PDF reader
def read_pdf_file(file_path):
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return pages_text

# TXT reader
def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Chunk PDF pages
def chunk_pdf_file(file_path, chunk_size=350):
    pages = read_pdf_file(file_path)
    chunks = []

    base_name = os.path.basename(file_path)
    file_stem = os.path.splitext(base_name)[0]  # e.g. "lecture_1"

    for page_idx, page_text in enumerate(pages):
        text_clean = clean_text(page_text)
        words = text_clean.split()
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "id": f"{base_name}_p{page_idx + 1}_c{len(chunks)}",
                "text": chunk_text,
                "source": base_name,
                "page": page_idx + 1,      # 1-based page/slide number
                "title": file_stem         # default: file name as title
            })
    return chunks, pages  # pages → for TOC parsing

# Chunk TXT file
def chunk_txt_file(file_path, chunk_size=350):
    text = read_txt_file(file_path)
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i+chunk_size])
        chunks.append({
            "id": f"{os.path.basename(file_path)}_chunk_{len(chunks)}",
            "text": chunk_text,
            "source": os.path.basename(file_path),
            "page": 0,
            "title": "Unknown"
        })
    return chunks

# Roman numeral converter
def roman_to_int(s: str) -> int:
    rom_val = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
    s = s.lower()
    total = 0
    prev = 0
    for c in reversed(s):
        curr = rom_val.get(c, 0)
        if curr < prev:
            total -= curr
        else:
            total += curr
            prev = curr
    return total

# Parse table of contents (TOC) from first pages of the PDF
def parse_toc_from_pages(pages, toc_page_count=5):
    toc_entries = []
    toc_text = "\n".join(pages[:toc_page_count])
    toc_lines = re.findall(
        r'^(.*?)\s*\.{2,}\s*(\d+|[ivxlcdmIVXLCDM]+)$',
        toc_text,
        flags=re.MULTILINE
    )
    for title, page_raw in toc_lines:
        title = title.strip()
        start_page = int(page_raw) - 1 if page_raw.isdigit() else roman_to_int(page_raw) - 1
        toc_entries.append({
            "title": title,
            "start_page": start_page
        })

    # Set end_page for each TOC entry
    for i in range(len(toc_entries)):
        if i < len(toc_entries) - 1:
            toc_entries[i]["end_page"] = toc_entries[i + 1]["start_page"] - 1
        else:
            toc_entries[i]["end_page"] = 999  # large sentinel

    return toc_entries

# Assign TOC titles to PDF chunks based on page number
def assign_toc_to_chunks(chunks, toc_entries):
    for chunk in chunks:
        page = chunk["page"]
        for entry in toc_entries:
            if entry["start_page"] <= page <= entry["end_page"]:
                chunk["title"] = entry["title"]

    # Manual corrections (optional)
    fix_titles = {}
    for chunk in chunks:
        if chunk["title"] in fix_titles:
            chunk["title"] = fix_titles[chunk["title"]]

    return chunks

# Ingestion pipeline (supports PDF and/or TXT)
def ingest_all(files, file_type="pdf", chunk_size=350, debug=False):
    all_chunks = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()

        if file_type in ["pdf", "both"] and ext == ".pdf":
            chunks, pages = chunk_pdf_file(f, chunk_size)
            toc_entries = parse_toc_from_pages(pages)

            # Only override titles if we actually found TOC entries
            if toc_entries:
                chunks = assign_toc_to_chunks(chunks, toc_entries)

            all_chunks.extend(chunks)


        elif file_type in ["txt", "both"] and ext == ".txt":
            chunks = chunk_txt_file(f, chunk_size)
            all_chunks.extend(chunks)

    if debug:
        chunks_by_file = defaultdict(list)
        for c in all_chunks:
            chunks_by_file[c["source"]].append(c)
        for filename, file_chunks in chunks_by_file.items():
            print(f"\n--- First 5 chunks for {filename} ---")
            for chunk in file_chunks[:5]:
                print(chunk)

    return all_chunks

# Main - scan data/ for PDF/TXT files and save chunks.json
if __name__ == "__main__":
    DATA_DIR = "data"

    files = []
    for root, dirs, filenames in os.walk(DATA_DIR):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in [".pdf", ".txt"]:
                files.append(os.path.join(root, name))

    if not files:
        print("No PDF or TXT files found in data/ directory.")
        exit(0)

    print("Files to ingest:")
    for f in files:
        print("  -", f)

    # Support both PDF and TXT files
    all_chunks = ingest_all(files, file_type="both", chunk_size=350, debug=False)

    os.makedirs("data", exist_ok=True)
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_chunks)} chunks to data/chunks.json")