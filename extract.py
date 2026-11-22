"""CLI question extraction entry point."""
import argparse
from config import DEFAULT_SUBJECT_CODE, SUBJECT_REGISTRY
from services.extraction import run_extraction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract questions from one subject corpus")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT_CODE, choices=SUBJECT_REGISTRY.keys())
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached results exist")
    args = parser.parse_args()
    questions_df, summary = run_extraction(force=args.force, subject_code=args.subject)
    print(f"\nExtraction complete for {args.subject}: {len(questions_df)} questions.")
