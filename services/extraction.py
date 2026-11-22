"""
AKTU ExamMate — Question Extraction Service

Extracts individual questions from AKTU Data Structures exam papers.
Handles bilingual (Hindi + English) text, OCR noise, and varied paper formats.
"""
import re
import os
import json
import hashlib
import pandas as pd
from typing import List, Dict, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_SUBJECT_CODE, SUBJECT_REGISTRY, get_subject_paths, normalize_subject_code
)


def load_csv(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load one subject's page-level extracted text CSV."""
    if not csv_path:
        csv_path = str(get_subject_paths(DEFAULT_SUBJECT_CODE)["cleaned_csv"])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Page-text CSV not found: {csv_path}. Prepare the PDF page text first."
        )
    return pd.read_csv(csv_path, encoding='utf-8-sig')


def validate_subject_paper(full_text: str, source_file: str, subject_code: str) -> Tuple[bool, str]:
    """Validate a paper against the selected subject's code, aliases and keywords."""
    code = normalize_subject_code(subject_code)
    spec = SUBJECT_REGISTRY[code]
    text_upper = (full_text or "").upper()

    codes = [code, *spec.get("legacy_codes", [])]
    for candidate in codes:
        escaped = re.escape(candidate.upper()).replace(r"\-", r"[\s-]?")
        if re.search(rf"(?<![A-Z0-9]){escaped}(?![A-Z0-9])", text_upper):
            return True, f"Subject code '{candidate}' found"

    for alias in spec.get("aliases", []):
        if alias.upper() in text_upper:
            return True, f"Subject name '{alias}' found"

    keywords = spec.get("keywords", [])
    found = [keyword for keyword in keywords if keyword.upper() in text_upper]
    if len(found) >= 3:
        return True, f"Multiple {code} keywords found ({len(found)}): {', '.join(found[:5])}"

    return False, (
        f"No {code} code/name found and only {len(found)} distinctive subject keywords matched"
    )


def validate_ds_paper(full_text: str, source_file: str) -> Tuple[bool, str]:
    """Backward-compatible Data Structures validator."""
    return validate_subject_paper(full_text, source_file, "BCS301")

def extract_year_from_text(full_text: str) -> Optional[int]:
    """
    Extract the examination year from the paper text itself.
    AKTU papers typically show "EXAMINATION 2024-25" meaning the 2025 exam session.
    """
    # Pattern: "EXAMINATION 20XX-YY" — the second year is the actual exam year
    patterns = [
        r'EXAMINATION\s+(\d{4})\s*[-–]\s*(\d{2,4})',
        r'THEORY\s+EXAMINATION\s+(\d{4})\s*[-–]\s*(\d{2,4})',
        r'परीक्षा\s+(\d{4})\s*[-–]\s*(\d{2,4})',
    ]
    
    text_upper = full_text.upper()
    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            start_year = int(match.group(1))
            end_part = match.group(2)
            if len(end_part) == 2:
                end_year = (start_year // 100) * 100 + int(end_part)
            else:
                end_year = int(end_part)
            # The end year is the actual examination year
            return end_year
    
    # Pattern: standalone "20XX" near "EXAMINATION"
    match = re.search(r'EXAMINATION[^\d]*(\d{4})', text_upper)
    if match:
        return int(match.group(1))
    
    return None


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from filename like '04-2025.pdf' or '15-2024.pdf'."""
    match = re.search(r'(\d{4})', filename)
    if match:
        return int(match.group(1))
    return None


def clean_text(text: str) -> str:
    """Clean OCR artifacts and normalize text while preserving content."""
    if not text:
        return ""
    # Remove common OCR noise patterns
    text = re.sub(r'[\|।]+\s*$', '', text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def is_header_or_footer(line: str) -> bool:
    """Check if a line is a header, footer, instruction, or metadata."""
    line_stripped = line.strip()
    if not line_stripped:
        return True
    
    line_upper = line_stripped.upper()
    
    skip_patterns = [
        r'^PRINTED\s+PAGE',
        r'^QP\d+',
        r'^\d+\s*\|\s*P\s*A\s*G\s*E',
        r'^ROLL\s*NO',
        r'^PAPER\s*I[Dd]',
        r'^SUB(JECT)?\s*CODE',
        r'^B\.?\s*TECH',
        r'^\(SEM',
        r'^SEM\s',
        r'^THEORY\s+EXAMINATION',
        r'^DATA\s+STRUCTURE',
        r'^TIME\s*:',
        r'^M\.?\s*MARKS',
        r'^TOTAL\s+MARKS',
        r'^NOTE\s*:',
        r'^SECTION\s+[A-E]',
        r'^\d+\s*[Xx×]\s*\d+\s*=',
        r'^ATTEMPT\s+ALL',
        r'^ATTEMPT\s+ANY',
        r'^IF\s+REQUIRE',
        r'^IN\s+CASE\s+OF',
        r'^MISSING\s+DATA',
        r'^CHOOSE\s+SUITABLY',
        r'^\d+\s*\|\s*\d+[-–]',
        r'^WWW\.AKTU',
        r'^CO\d+',
        r'^K\d+$',
        r'^LEVEL$',
        r'^\d+\s*(HRS|HOURS)',
        r'^[—–\-=_]{3,}',
        r'^\d+\.\d+\.\d+\.\d+',  # IP addresses
        r'^ANY\s+SPECIAL\s+PAPER',
        r'^REGISTRATION\s*NO',
        r'^COURSE\s*:',
        r'^BRANCH',
        r'^Q\s*\.?\s*CODE',
        r'^TOTAL\s+NUMBER\s+OF\s+PAGES',
        r'^MAX\s+MARKS',
        r'^THE\s+FIGURES\s+IN\s+THE',
        r'^ANSWER\s+QUESTION',
    ]
    
    for pat in skip_patterns:
        if re.search(pat, line_upper):
            return True
    
    # Lines that are only numbers, symbols, or very short noise
    if re.match(r'^[\d\s\|।\-–_=\*\+\.,:;]+$', line_stripped) and len(line_stripped) < 20:
        return True
    
    return False


def normalize_for_matching(text: str) -> str:
    """Normalize text for exact matching comparison."""
    text = text.lower().strip()
    # Remove marks references like (2), (7), etc.
    text = re.sub(r'\(\s*\d+\s*\)', '', text)
    # Remove trailing marks numbers
    text = re.sub(r'\s+\d+\s*$', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common punctuation differences
    text = re.sub(r'[.,:;!?]+$', '', text)
    return text.strip()


def separate_hindi_english(text: str) -> Tuple[str, str]:
    """
    Separate Hindi and English parts from bilingual text.
    Returns (english_text, original_full_text).
    """
    # Devanagari Unicode range: \u0900-\u097F
    # Split by detecting language transitions
    
    # If mostly English (no Devanagari), return as-is
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    total_alpha = len(re.findall(r'[a-zA-Z\u0900-\u097F]', text))
    
    if total_alpha == 0:
        return text, text
    
    if hindi_chars / max(total_alpha, 1) < 0.05:
        # Mostly English
        return text, text
    
    # Extract English parts — keep sentences/phrases that are primarily English
    # Split into chunks and classify each
    english_parts = []
    
    # Split by common delimiters between Hindi and English versions
    # Often the Hindi translation follows the English with a newline or specific pattern
    lines = re.split(r'\n+', text)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_hindi = len(re.findall(r'[\u0900-\u097F]', line))
        line_alpha = len(re.findall(r'[a-zA-Z]', line))
        
        if line_alpha > 0 and (line_hindi / max(line_alpha + line_hindi, 1) < 0.3):
            english_parts.append(line)
        elif line_alpha > line_hindi:
            # Mixed line, extract English words
            eng_words = re.findall(r'[a-zA-Z][a-zA-Z\s\d\(\)\[\]\{\}\+\-\*/=<>,.:;\'\"]+', line)
            if eng_words:
                english_parts.append(' '.join(eng_words))
    
    english_text = ' '.join(english_parts).strip()
    if not english_text:
        english_text = text  # Fallback: keep original
    
    return english_text, text


def extract_questions_from_paper(full_text: str, source_file: str, year: int, 
                                  page_texts: Dict[int, str]) -> List[Dict]:
    """
    Extract individual questions from a paper's full text.
    Returns list of question dicts.
    """
    questions = []
    
    # Question patterns for AKTU papers
    # Pattern 1: Lettered sub-questions "a.", "b.", "a)", "(a)" etc.
    # Pattern 2: Numbered questions "1.", "2." 
    # Pattern 3: Roman numerals "(i)", "(ii)"
    # Pattern 4: Table format with "Q no." column
    
    # Determine which page each piece of text comes from
    page_positions = {}
    current_pos = 0
    for page_num in sorted(page_texts.keys()):
        page_text = page_texts[page_num]
        page_positions[page_num] = (current_pos, current_pos + len(page_text))
        current_pos += len(page_text) + 1  # +1 for newline join
    
    def find_page_for_position(pos: int) -> int:
        for pg, (start, end) in page_positions.items():
            if start <= pos <= end:
                return pg
        return min(page_texts.keys())
    
    # Clean the text, removing headers/footers line by line
    lines = full_text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not is_header_or_footer(line):
            cleaned_lines.append(line)
    
    working_text = '\n'.join(cleaned_lines)
    
    # Strategy: Find question boundaries using numbering patterns
    # Main question pattern: digit followed by period at start of line
    # Sub-question patterns: letter or roman numeral
    
    # Match patterns like:
    # "a. question text" or "(a) question text" or "a) question text"
    # "1. question text"
    # "(i) question text" or "i. question text"
    
    question_pattern = re.compile(
        r'''
        (?:^|\n)\s*
        (?:
            (?:Q\s*(?:no\.?)?\s*)?                        # Optional "Q no."
            (?:
                (\d+)\s*\.\s*                              # "1." main question
                |
                \(?\s*([a-gA-G])\s*[).\s]\s*               # "(a)" or "a." or "a)"
                |
                \(?\s*(i{1,3}|iv|v|vi{0,3})\s*[).\s]\s*    # "(i)" roman numerals
            )
        )
        (.+?)                                              # Question text (non-greedy)
        (?=                                                # Lookahead for next question
            \n\s*(?:Q\s*(?:no\.?)?\s*)?(?:\d+\s*\.|(?:\(?\s*[a-gA-G]\s*[).\s])|(?:\(?\s*(?:i{1,3}|iv|v|vi{0,3})\s*[).\s]))
            |
            \n\s*SECTION\s+[A-E]
            |
            $
        )
        ''',
        re.VERBOSE | re.DOTALL | re.IGNORECASE
    )
    
    matches = list(question_pattern.finditer(working_text))
    
    if not matches or len(matches) < 3:
        # Fallback: simpler line-by-line extraction
        questions.extend(_extract_questions_simple(working_text, source_file, year, 
                                                   full_text, page_positions))
    else:
        for match in matches:
            main_num = match.group(1)
            sub_letter = match.group(2)
            roman = match.group(3)
            q_text = match.group(4).strip()
            
            if main_num:
                q_number = main_num
            elif sub_letter:
                q_number = sub_letter.lower()
            elif roman:
                q_number = roman.lower()
            else:
                q_number = "?"
            
            # Find the page
            pos = match.start()
            page_num = find_page_for_position(pos)
            
            # Clean the question text
            q_text = clean_text(q_text)
            
            if len(q_text) < 10:
                continue  # Too short to be a real question
            
            # Separate Hindi/English
            english_text, original_text = separate_hindi_english(q_text)
            english_text = clean_text(english_text)
            
            if len(english_text) < 10:
                english_text = q_text  # Fallback
            
            # Determine confidence
            confidence = _assess_confidence(english_text, q_text, year, source_file)
            
            questions.append({
                "question_number": q_number,
                "original_question": q_text,
                "english_question": english_text,
                "year": year,
                "source_file": source_file,
                "page_number": page_num,
                "extraction_confidence": confidence
            })
    
    # Deduplicate within the same paper (same question appearing due to regex overlap)
    questions = _deduplicate_within_paper(questions)
    
    return questions


def _extract_questions_simple(text: str, source_file: str, year: int,
                               full_text: str, page_positions: Dict) -> List[Dict]:
    """
    Simpler extraction for papers where the regex pattern doesn't work well.
    Uses line-by-line analysis.
    """
    questions = []
    lines = text.split('\n')
    
    current_question = None
    current_number = None
    current_lines = []
    
    def find_page_for_line(line_idx: int) -> int:
        # Approximate page from character position
        char_pos = sum(len(lines[i]) + 1 for i in range(min(line_idx, len(lines))))
        for pg, (start, end) in page_positions.items():
            if start <= char_pos <= end:
                return pg
        return min(page_positions.keys()) if page_positions else 1
    
    # Patterns for question starts
    q_start = re.compile(
        r'^\s*(?:Q\s*(?:no\.?)?\s*)?'
        r'(?:'
        r'(\d+)\s*\.'
        r'|'
        r'\(?\s*([a-gA-G])\s*[).]\s*'
        r'|'
        r'\(?\s*(i{1,3}|iv|v|vi{0,3})\s*[).]\s*'
        r')'
        r'\s*(.+)',
        re.IGNORECASE
    )
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped or is_header_or_footer(line_stripped):
            continue
        
        match = q_start.match(line_stripped)
        if match:
            # Save previous question
            if current_number and current_lines:
                q_text = ' '.join(current_lines)
                q_text = clean_text(q_text)
                if len(q_text) >= 10:
                    english, original = separate_hindi_english(q_text)
                    english = clean_text(english)
                    if len(english) < 10:
                        english = q_text
                    page = find_page_for_line(current_start_line)
                    conf = _assess_confidence(english, q_text, year, source_file)
                    questions.append({
                        "question_number": current_number,
                        "original_question": q_text,
                        "english_question": english,
                        "year": year,
                        "source_file": source_file,
                        "page_number": page,
                        "extraction_confidence": conf
                    })
            
            # Start new question
            main_num = match.group(1)
            sub_letter = match.group(2)
            roman = match.group(3)
            rest = match.group(4).strip() if match.group(4) else ""
            
            if main_num:
                current_number = main_num
            elif sub_letter:
                current_number = sub_letter.lower()
            elif roman:
                current_number = roman.lower()
            else:
                current_number = "?"
            
            current_lines = [rest] if rest else []
            current_start_line = i
        else:
            # Continuation of current question
            if current_number is not None:
                current_lines.append(line_stripped)
    
    # Save last question
    if current_number and current_lines:
        q_text = ' '.join(current_lines)
        q_text = clean_text(q_text)
        if len(q_text) >= 10:
            english, original = separate_hindi_english(q_text)
            english = clean_text(english)
            if len(english) < 10:
                english = q_text
            page = find_page_for_line(current_start_line)
            conf = _assess_confidence(english, q_text, year, source_file)
            questions.append({
                "question_number": current_number,
                "original_question": q_text,
                "english_question": english,
                "year": year,
                "source_file": source_file,
                "page_number": page,
                "extraction_confidence": conf
            })
    
    return questions


def _assess_confidence(english_text: str, original_text: str, year: int, 
                        source_file: str) -> str:
    """
    Assess extraction confidence for a question.
    Returns 'high', 'medium', or 'low'.
    """
    score = 1.0
    
    # Check text quality
    if len(english_text) < 15:
        score -= 0.3
    
    # Check for excessive OCR noise indicators
    noise_chars = len(re.findall(r'[^\w\s\(\)\[\]\{\}\+\-\*/=<>,.?!:;\'"@#$%&]', english_text))
    if noise_chars / max(len(english_text), 1) > 0.1:
        score -= 0.3
    
    # Check if it looks like a real question (has question words or imperative verbs)
    q_indicators = ['what', 'how', 'why', 'when', 'which', 'define', 'explain', 'write',
                     'draw', 'construct', 'find', 'calculate', 'convert', 'derive',
                     'discuss', 'differentiate', 'compare', 'illustrate', 'describe',
                     'implement', 'demonstrate', 'trace', 'evaluate', 'prove', 'show',
                     'consider', 'assume', 'given', 'suppose', 'state', 'enumerate',
                     'list', 'mention', 'give', 'create', 'insert', 'delete', 'sort',
                     'search', 'traverse']
    text_lower = english_text.lower()
    has_indicator = any(ind in text_lower for ind in q_indicators)
    if not has_indicator:
        score -= 0.2
    
    # Year validation: check that file year matches extracted context
    file_year = extract_year_from_filename(source_file)
    if file_year and abs(file_year - year) > 1:
        score -= 0.2
    
    if score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    else:
        return "low"


def _is_instruction_not_question(text: str) -> bool:
    """Filter out instructions/headers that were mistakenly captured as questions."""
    text_lower = text.lower().strip()
    instruction_patterns = [
        r'^attempt\s+(all|any)',
        r'^any\s+special\s+paper',
        r'^note\s*:',
        r'^section\s+[a-e]',
        r'^answer\s+question',
        r'^the\s+figures\s+in',
        r'^marks?\s+are\s+',
        r'^write\s+answer\s+of\s+each\s+part\s+in\s+short',
        r'^all\s+parts\s+carry\s+equal',
        r'^if\s+require\s+any\s+missing',
        r'^in\s+case\s+of\s+any\s+missing',
        r'^choose\s+suitably',
    ]
    for pat in instruction_patterns:
        if re.search(pat, text_lower):
            return True
    return False


def _deduplicate_within_paper(questions: List[Dict]) -> List[Dict]:
    """Remove duplicate questions and instructions within the same paper."""
    seen = set()
    unique = []
    for q in questions:
        # Filter out instructions
        if _is_instruction_not_question(q['english_question']):
            continue
        
        # Key on normalized text
        key = normalize_for_matching(q['english_question'])
        if len(key) < 5:
            continue
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


def _get_cache_key(df: pd.DataFrame) -> str:
    """Generate a cache key from the dataframe content."""
    content = df.to_csv(index=False)
    return hashlib.md5(content.encode()).hexdigest()


def run_extraction(
    force: bool = False, subject_code: str = DEFAULT_SUBJECT_CODE
) -> Tuple[pd.DataFrame, Dict]:
    """Extract questions for exactly one subject corpus."""
    code = normalize_subject_code(subject_code)
    paths = get_subject_paths(code)

    print("=" * 60)
    print(f"AKTU ExamMate — Question Extraction [{code}: {SUBJECT_REGISTRY[code]['name']}]")
    print("=" * 60)

    df = load_csv(str(paths["cleaned_csv"]))
    required = {"source_file", "page_number", "final_text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Page-text CSV for {code} is missing columns: {sorted(missing)}")
    if "year" not in df.columns:
        df["year"] = None
    if "extraction_method" not in df.columns:
        df["extraction_method"] = "unknown"

    cache_key = _get_cache_key(df)
    cache_path = paths["extraction_cache"]
    if not force and cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("cache_key") == cache_key and cache.get("subject_code", code) == code:
                print("✓ Using cached extraction results")
                return pd.DataFrame(cache["questions"]), cache["summary"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    print(f"Processing {df['source_file'].nunique()} papers, {len(df)} total pages...")
    all_questions = []
    validation = {
        "subject_code": code,
        "subject_name": SUBJECT_REGISTRY[code]["name"],
        "total_pdfs": int(df["source_file"].nunique()),
        "eligible_papers": 0,
        "skipped_papers": [],
        "total_pages_processed": 0,
        "total_questions_extracted": 0,
        "papers_processed": [],
        "uncertain_questions": 0,
    }

    for source_file, group in df.groupby("source_file"):
        group = group.sort_values("page_number")
        page_texts = {}
        for _, row in group.iterrows():
            try:
                page_number = int(row["page_number"])
            except (TypeError, ValueError):
                continue
            text = str(row["final_text"]) if pd.notna(row["final_text"]) else ""
            page_texts[page_number] = text

        if not page_texts:
            validation["skipped_papers"].append({"file": source_file, "reason": "No readable page text"})
            continue

        full_text = "\n".join(page_texts[page] for page in sorted(page_texts))
        valid, reason = validate_subject_paper(full_text, source_file, code)
        if not valid:
            validation["skipped_papers"].append({"file": source_file, "reason": reason})
            continue

        year_value = group.iloc[0].get("year")
        try:
            csv_year = int(year_value) if pd.notna(year_value) else None
        except (TypeError, ValueError):
            csv_year = None
        text_year = extract_year_from_text(full_text)
        if text_year and csv_year and abs(text_year - csv_year) <= 1:
            year = text_year
        else:
            year = text_year or csv_year or extract_year_from_filename(str(source_file))

        if not year:
            validation["skipped_papers"].append({"file": source_file, "reason": "Could not determine year"})
            continue

        questions = extract_questions_from_paper(full_text, str(source_file), int(year), page_texts)
        validation["eligible_papers"] += 1
        validation["total_pages_processed"] += len(page_texts)
        validation["papers_processed"].append({
            "file": source_file,
            "year": int(year),
            "pages": len(page_texts),
            "questions_extracted": len(questions),
            "extraction_method": str(group.iloc[0].get("extraction_method", "unknown")),
        })
        validation["uncertain_questions"] += sum(
            1 for question in questions if question.get("extraction_confidence") == "low"
        )
        all_questions.extend(questions)
        print(f"  ✓ {source_file} (year={year}): {len(questions)} questions extracted")

    validation["total_questions_extracted"] = len(all_questions)
    if all_questions:
        questions_df = pd.DataFrame(all_questions)
    else:
        questions_df = pd.DataFrame(columns=[
            "question_number", "original_question", "english_question",
            "year", "source_file", "page_number", "extraction_confidence"
        ])

    questions_df.to_csv(paths["questions_csv"], index=False, encoding="utf-8-sig")
    questions_json = questions_df.to_dict("records")
    with paths["questions_json"].open("w", encoding="utf-8") as f:
        json.dump(questions_json, f, ensure_ascii=False, indent=2)

    cache_data = {
        "subject_code": code,
        "cache_key": cache_key,
        "questions": questions_json,
        "summary": validation,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Extraction Summary:")
    print(f"  Subject: {code} — {SUBJECT_REGISTRY[code]['name']}")
    print(f"  Total PDFs: {validation['total_pdfs']}")
    print(f"  Eligible papers: {validation['eligible_papers']}")
    print(f"  Skipped papers: {len(validation['skipped_papers'])}")
    print(f"  Pages processed: {validation['total_pages_processed']}")
    print(f"  Questions extracted: {validation['total_questions_extracted']}")
    print(f"  Uncertain questions: {validation['uncertain_questions']}")
    print("=" * 60)
    return questions_df, validation


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract questions for one AKTU subject")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT_CODE, help="Subject code, e.g. BCS301 or BCS401")
    parser.add_argument("--force", action="store_true", help="Ignore extraction cache and rebuild")
    args = parser.parse_args()
    questions_df, summary = run_extraction(force=args.force, subject_code=args.subject)
    print(f"\nSaved {len(questions_df)} questions for {args.subject.upper()}.")
