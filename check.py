"""Validation runner for a selected subject corpus."""
import argparse
import json
import pandas as pd

from config import DEFAULT_SUBJECT_CODE, SUBJECT_REGISTRY, get_subject_paths, normalize_subject_code
from services.study_planner import generate_study_plan


def run_validation(subject_code=DEFAULT_SUBJECT_CODE):
    code = normalize_subject_code(subject_code)
    paths = get_subject_paths(code, create=False)
    print("=" * 70)
    print(f"AKTU ExamMate — Validation [{code}: {SUBJECT_REGISTRY[code]['name']}]")
    print("=" * 70)

    required_files = [
        paths["cleaned_csv"], paths["questions_csv"], paths["questions_json"],
        paths["repeated_questions_csv"], paths["repeated_questions_json"], paths["review_required_csv"],
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  - {path}")
        return False

    raw_df = pd.read_csv(paths["cleaned_csv"], encoding="utf-8-sig")
    questions_df = pd.read_csv(paths["questions_csv"], encoding="utf-8-sig")
    with paths["questions_json"].open("r", encoding="utf-8") as handle:
        questions_json = json.load(handle)
    with paths["repeated_questions_json"].open("r", encoding="utf-8") as handle:
        repeated = json.load(handle)
    repeated_df = pd.read_csv(paths["repeated_questions_csv"], encoding="utf-8-sig")
    review_df = pd.read_csv(paths["review_required_csv"], encoding="utf-8-sig")

    print("\n1. Source data")
    print(f"   Pages in source CSV: {len(raw_df)}")
    print(f"   Papers in source CSV: {raw_df['source_file'].nunique()}")
    print("\n2. Extracted questions")
    print(f"   CSV questions: {len(questions_df)}")
    print(f"   JSON questions: {len(questions_json)}")
    print(f"   Papers represented in questions: {questions_df['source_file'].nunique() if not questions_df.empty else 0}")
    required_columns = [
        "question_number", "original_question", "english_question",
        "year", "source_file", "page_number", "extraction_confidence",
    ]
    absent = [column for column in required_columns if column not in questions_df.columns]
    if absent:
        print(f"   ERROR: missing question columns: {absent}")
        return False
    if len(questions_df) != len(questions_json):
        print("   ERROR: CSV and JSON question counts differ")
        return False
    if questions_df.empty:
        print("   ERROR: no questions were extracted")
        return False

    years = sorted(int(year) for year in questions_df["year"].dropna().unique())
    print(f"   Examination years: {years[0]}–{years[-1]} ({len(years)} distinct years)" if years else "   Examination years: unavailable")
    print(f"   Confidence: {questions_df['extraction_confidence'].value_counts().to_dict()}")

    print("\n3. Repetition analysis")
    print(f"   Repeated groups: {len(repeated)}")
    print(f"   Repeated instances: {len(repeated_df)}")
    print(f"   Borderline review rows: {len(review_df)}")
    valid_frequency = all(
        group.get("distinct_year_count") == len(set(group.get("years_appeared", [])))
        and group.get("distinct_year_count", 0) >= 2
        for group in repeated
    )
    print(f"   Distinct-year counting: {'PASS' if valid_frequency else 'FAIL'}")
    if not valid_frequency:
        return False

    print("\n4. Study planner")
    for days, hours in [(3, 4.0), (7, 4.0), (14, 3.5), (30, 4.0)]:
        plan = generate_study_plan(days, hours, repeated, questions_json)
        if "daily_plan" not in plan or len(plan["daily_plan"]) != days:
            print(f"   FAIL: {days}-day plan")
            return False
        print(f"   PASS: {days} days, {hours} hours/day")

    print("\nVALIDATION PASSED")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate one subject's extracted corpus and analysis")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT_CODE, choices=SUBJECT_REGISTRY.keys())
    args = parser.parse_args()
    raise SystemExit(0 if run_validation(args.subject) else 1)
