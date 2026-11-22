"""CLI repetition-analysis entry point."""
import argparse
from config import DEFAULT_SUBJECT_CODE, SUBJECT_REGISTRY
from services.repetition import run_repetition_detection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse repeated questions for one subject")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT_CODE, choices=SUBJECT_REGISTRY.keys())
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached results exist")
    args = parser.parse_args()
    repeated, review, summary = run_repetition_detection(force=args.force, subject_code=args.subject)
    print(f"\nAnalysis complete for {args.subject}: {len(repeated)} repeated groups.")
