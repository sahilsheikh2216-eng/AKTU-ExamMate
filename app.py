"""AKTU ExamMate — multi-subject Flask application."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_SUBJECT_CODE, SUBJECT_REGISTRY, get_subject_paths,
    normalize_subject_code, subject_is_mounted,
)
from services.study_planner import generate_study_plan  # noqa: E402

app = Flask(__name__, template_folder="templates")

# Data is cached separately per subject. Switching subjects never reuses another
# course's in-memory data.
_CACHE = {}


def _read_json(path):
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def get_cached_data(subject_code=DEFAULT_SUBJECT_CODE, refresh=False):
    code = normalize_subject_code(subject_code)
    if code in _CACHE and not refresh:
        return _CACHE[code]

    paths = get_subject_paths(code, create=False)
    questions = _read_json(paths["questions_json"])
    repeated = _read_json(paths["repeated_questions_json"])
    review = []
    if paths["review_required_csv"].exists():
        try:
            review_df = pd.read_csv(paths["review_required_csv"], encoding="utf-8-sig")
            review = review_df.to_dict("records") if len(review_df) else []
        except Exception:
            review = []

    _CACHE[code] = (questions, repeated, review)
    return _CACHE[code]


def _subject_stats(code):
    questions, repeated, review = get_cached_data(code)
    papers = {q.get("source_file") for q in questions if q.get("source_file")}
    years = []
    for question in questions:
        try:
            year = int(question.get("year"))
            years.append(year)
        except (TypeError, ValueError):
            continue
    return {
        "papers_count": len(papers),
        "questions_count": len(questions),
        "repeated_groups_count": len(repeated),
        "review_count": len(review),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/subjects")
def api_subjects():
    results = []
    for code, spec in SUBJECT_REGISTRY.items():
        stats = _subject_stats(code)
        mounted = subject_is_mounted(code)
        results.append({
            "code": code,
            "name": spec["name"],
            "description": spec["description"],
            "status": "active" if mounted else "blank",
            "mounted": mounted,
            "badge": f"{stats['papers_count']} PAPERS LOADED" if mounted else "BLANK // UNMOUNTED",
            "stats": stats,
        })
    return jsonify(results)


@app.route("/api/data")
def api_data():
    code = request.args.get("subject", DEFAULT_SUBJECT_CODE)
    try:
        code = normalize_subject_code(code)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    questions, repeated, review = get_cached_data(code)
    stats = _subject_stats(code)
    return jsonify({
        "subject_code": code,
        "subject_name": SUBJECT_REGISTRY[code]["name"],
        "questions": questions,
        "repeated": repeated,
        "review": review,
        "stats": stats,
    })


@app.route("/api/study-plan", methods=["POST"])
def api_study_plan():
    data = request.get_json(silent=True) or {}
    try:
        code = normalize_subject_code(data.get("subject_code", DEFAULT_SUBJECT_CODE))
        days = int(data.get("days", 14))
        hours = float(data.get("hours", 4.0))
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    questions, repeated, _ = get_cached_data(code)
    if not questions:
        return jsonify({"error": f"No question corpus is mounted for {code} yet."}), 404
    return jsonify(generate_study_plan(days, hours, repeated, questions))


@app.route("/api/pipeline/run", methods=["POST"])
def api_run_pipeline():
    data = request.get_json(silent=True) or {}
    try:
        code = normalize_subject_code(data.get("subject_code", DEFAULT_SUBJECT_CODE))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    from services.extraction import run_extraction
    from services.repetition import run_repetition_detection

    try:
        questions_df, extraction_summary = run_extraction(force=True, subject_code=code)
        repeated, review, repetition_summary = run_repetition_detection(force=True, subject_code=code)
    except Exception as exc:
        return jsonify({"error": str(exc), "subject_code": code}), 400

    _CACHE.pop(code, None)
    return jsonify({
        "status": "success",
        "subject_code": code,
        "questions_extracted": len(questions_df),
        "repeated_groups": len(repeated),
        "review_groups": len(review),
        "extraction": extraction_summary,
        "repetition": repetition_summary,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print("\n=======================================================")
    print("AKTU ExamMate — Multi-Subject Flask Interface")
    print(f"http://127.0.0.1:{port}")
    print("=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
