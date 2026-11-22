# AKTU ExamMate — Multi-Subject PYQ Repetition Analyzer & Study Planner

A retro terminal web application for AKTU engineering students. It extracts questions from examination papers, detects exact and semantically similar repetitions, and builds evidence-based study plans. Each subject now has its own isolated source data, question corpus, repetition results and embedding cache.

## Supported subjects

| Subject | Code | Initial state |
|---|---|---|
| Data Structures | `BCS301` | Active — existing corpus preserved (58 papers, 1,389 questions, 174 repeated groups) |
| Operating Systems | `BCS401` | Ready for source PDFs |
| Computer Organization & Architecture | `BCS302` | Ready for source PDFs |
| Discrete Mathematics | `BAS301` | Ready for source PDFs |
| Theory of Automata & Formal Languages | `BCS501` | Ready for source PDFs |

The other subjects become active after their page text has been prepared, questions extracted and repetition analysis completed.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Add a new subject corpus

Example: add Operating Systems (`BCS401`).

### 1. Put the PDFs in the subject's own folder

```text
data/subjects/BCS401/pdfs/raw/
    os-2024.pdf
    os-2023.pdf
    os-2022.pdf
```

Use filenames containing a four-digit examination year where possible. Do not place BCS401 papers in the BCS301 folder or combine two subjects into one input CSV.

### 2. Extract page text

```bash
python prepare_pages.py --subject BCS401
```

This uses PyMuPDF to extract selectable/native PDF text and creates:

```text
data/subjects/BCS401/source/cleaned.csv
```

The page CSV uses these columns: `source_file`, `page_number`, `final_text`, `extraction_method`, `year`. The script reports pages with little or no selectable text. Those pages may be scans and need OCR before question extraction; this utility does not perform OCR. Review the CSV and correct any wrong/missing years before continuing.

### 3. Extract individual questions

```bash
python extract.py --subject BCS401
```

This validates the paper against BCS401's code, subject aliases and distinctive keywords, then extracts questions into that subject's own files. If you add or replace papers after a previous run, use `--force`.

### 4. Detect repeated questions

```bash
python analyze.py --subject BCS401
```

The semantic embedding cache and repetition outputs are stored only inside `data/subjects/BCS401/`. No BCS301 cache or result file is reused.

### 5. Validate the corpus

```bash
python check.py --subject BCS401
```

Search a subject's extracted questions with:

```bash
python check_question.py "deadlock" 2024 --subject BCS401
```

### 6. Open the application

```bash
python app.py
```

Open `http://127.0.0.1:5050`. A subject appears as mounted in the selector when both its question corpus and repetition analysis files exist. Selecting a mounted subject loads that subject's questions, repeated groups, review list and study planner data.

## Subject configuration

Subject names, aliases and validation keywords are configured in `config.py` under `SUBJECT_REGISTRY`. If papers use a different course code or an alternate official subject name, add it to that subject's `legacy_codes` or `aliases`. Keyword-based validation requires at least three configured subject-specific keyword matches. Inspect skipped-paper reasons before trusting a corpus.

## Data layout

```text
data/
└── subjects/
    ├── BCS301/
    │   ├── pdfs/raw/
    │   ├── source/cleaned.csv
    │   ├── cache/
    │   ├── questions.csv
    │   ├── questions.json
    │   ├── repeated_questions.csv
    │   ├── repeated_questions.json
    │   └── review_required.csv
    └── BCS401/
        └── (same structure, created as you add its papers)
```

The original legacy `data/` files are retained as a backup. The active multi-subject application reads and writes under `data/subjects/<SUBJECT_CODE>/`. Existing BCS301 source text, question outputs, repetition results and caches were copied there without deleting the original files.

## CLI tools

```bash
# Prepare page text from PDFs
python prepare_pages.py --subject BCS401

# Run extraction and repetition analysis for one subject
python extract.py --subject BCS401
python analyze.py --subject BCS401

# Force rebuilds after changing source material
python extract.py --subject BCS401 --force
python analyze.py --subject BCS401 --force

# Validate and search
python check.py --subject BCS401
python check_question.py "deadlock" 2024 --subject BCS401
```

## Technical specifications

- **Backend:** Python 3, Flask
- **Frontend:** Vanilla HTML5/CSS3/JavaScript with retro CRT terminal styling
- **PDF text extraction:** PyMuPDF (native/selectable text only)
- **Multilingual embeddings:** `sentence-transformers` — `paraphrase-multilingual-MiniLM-L12-v2`
- **Exact matching:** Normalized text hashing
- **Semantic matching:** Cosine similarity over normalized embeddings
- **Review queue:** Similarity range `0.72 <= similarity < 0.82`
- **Repeated-question frequency:** Counts distinct examination years, not intra-paper duplicate appearances
- **External API requirement:** None for the extraction and repetition analysis pipeline

## Important limitations

1. The question-boundary parser uses common AKTU numbering patterns. Review extracted results, especially unusual paper layouts.
2. Scanned PDFs need OCR before running `extract.py`; a page with no selectable text is not a successfully extracted page.
3. Subject validation depends on course-code/name matches or the configured keyword list. Review skipped papers and false positives.
4. Repetition counts indicate historical recurrence, not a guarantee that a question will appear again.
5. The subject syllabus map in `syllabus.py` is separate from paper extraction and is not yet used to auto-tag every extracted question.
