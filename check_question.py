"""Inspect or search extracted questions for a selected subject."""
import argparse
import pandas as pd
from config import DEFAULT_SUBJECT_CODE, SUBJECT_REGISTRY, get_subject_paths


def inspect_questions(keyword="", year=None, limit=10, subject_code=DEFAULT_SUBJECT_CODE):
    paths = get_subject_paths(subject_code, create=False)
    questions_path = paths["questions_csv"]
    if not questions_path.exists():
        print(f"Questions file not found for {subject_code}. Run extraction first.")
        return

    df = pd.read_csv(questions_path, encoding="utf-8-sig")
    if year is not None:
        df = df[df["year"] == year]
    if keyword:
        df = df[df["english_question"].astype(str).str.contains(keyword, case=False, na=False)]

    print(f"{subject_code}: found {len(df)} matching questions (showing up to {limit})\n")
    for _, row in df.head(limit).iterrows():
        print(f"[{row['year']}] Q{row['question_number']} ({row['source_file']} p.{row['page_number']}) [Confidence: {row['extraction_confidence']}]:")
        print(f"  {row['english_question']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search extracted PYQ questions")
    parser.add_argument("keyword", nargs="?", default="")
    parser.add_argument("year", nargs="?", type=int)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT_CODE, choices=SUBJECT_REGISTRY.keys())
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    inspect_questions(args.keyword, args.year, args.limit, args.subject)
