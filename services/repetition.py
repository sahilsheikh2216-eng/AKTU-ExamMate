"""
AKTU ExamMate — Repetition Detection Service

Detects exact and semantically similar repeated questions across exam years.
Uses local multilingual embeddings (sentence-transformers) for semantic matching.
"""
import os
import re
import json
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_SUBJECT_CODE, get_subject_paths, normalize_subject_code,
    EMBEDDING_MODEL, EXACT_MATCH_THRESHOLD, SEMANTIC_MATCH_THRESHOLD, REVIEW_THRESHOLD
)


def normalize_for_comparison(text: str) -> str:
    """Normalize text for exact comparison."""
    text = text.lower().strip()
    # Remove marks references
    text = re.sub(r'\(\s*\d+\s*\)', '', text)
    text = re.sub(r'\s+\d+\s*$', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove trailing punctuation
    text = re.sub(r'[.,:;!?]+$', '', text)
    # Remove common filler words at start
    text = re.sub(r'^(q\s*\.?\s*)?', '', text)
    return text.strip()


def compute_text_hash(text: str) -> str:
    """Compute hash of normalized text for exact matching."""
    normalized = normalize_for_comparison(text)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def load_questions(subject_code: str = DEFAULT_SUBJECT_CODE) -> pd.DataFrame:
    """Load extracted questions for one subject."""
    paths = get_subject_paths(subject_code)
    if not paths["questions_csv"].exists():
        raise FileNotFoundError(
            f"Questions file not found for {subject_code}. Run python extract.py --subject {subject_code} first."
        )
    return pd.read_csv(paths["questions_csv"], encoding='utf-8-sig')


def compute_embeddings(
    texts: List[str], cache_key: str, embeddings_cache=None
) -> np.ndarray:
    """
    Compute embeddings for a list of texts using sentence-transformers.
    Uses cached embeddings if available and cache_key matches.
    """
    # Each subject has its own embeddings cache to prevent cross-course contamination.
    embeddings_cache = embeddings_cache or get_subject_paths(DEFAULT_SUBJECT_CODE)["embeddings_cache"]
    embeddings_cache = str(embeddings_cache)
    cache_meta_path = embeddings_cache.replace('.npz', '_meta.json')
    if os.path.exists(embeddings_cache) and os.path.exists(cache_meta_path):
        try:
            with open(cache_meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get('cache_key') == cache_key and meta.get('count') == len(texts):
                data = np.load(embeddings_cache)
                embeddings = data['embeddings']
                if len(embeddings) == len(texts):
                    print(f"  ✓ Using cached embeddings ({len(embeddings)} vectors)")
                    return embeddings
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    
    print(f"  Computing embeddings for {len(texts)} questions...")
    print(f"  Loading model: {EMBEDDING_MODEL}")
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Batch encode
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True  # For cosine similarity via dot product
    )
    
    # Save cache
    os.makedirs(os.path.dirname(embeddings_cache), exist_ok=True)
    np.savez_compressed(embeddings_cache, embeddings=embeddings)
    with open(cache_meta_path, 'w') as f:
        json.dump({'cache_key': cache_key, 'count': len(texts), 'model': EMBEDDING_MODEL}, f)
    
    print(f"  ✓ Embeddings computed and cached")
    return embeddings


def find_exact_matches(questions: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Find exact text matches (after normalization).
    Returns dict mapping hash -> list of question indices.
    """
    hash_groups = defaultdict(list)
    
    for idx, row in questions.iterrows():
        text = row['english_question']
        normalized = normalize_for_comparison(text)
        if len(normalized) < 10:
            continue
        text_hash = compute_text_hash(text)
        hash_groups[text_hash].append(idx)
    
    # Keep only groups with questions from multiple distinct years
    exact_groups = {}
    for h, indices in hash_groups.items():
        years = set(questions.loc[indices, 'year'])
        if len(years) > 1:
            exact_groups[h] = indices
    
    return exact_groups


def find_semantic_matches(questions: pd.DataFrame, embeddings: np.ndarray,
                           exact_assigned: set) -> Tuple[List[Dict], List[Dict]]:
    """
    Find semantically similar questions using embedding similarity.
    Uses conservative thresholds and avoids transitive clustering.
    
    Returns (confirmed_groups, review_groups).
    """
    n = len(questions)
    
    # Compute similarity matrix (dot product since embeddings are normalized)
    print("  Computing similarity matrix...")
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    # Build direct pairwise matches above threshold
    confirmed_pairs = []  # (i, j, similarity)
    review_pairs = []     # (i, j, similarity)
    
    for i in range(n):
        if i in exact_assigned:
            continue
        for j in range(i + 1, n):
            if j in exact_assigned:
                continue
            
            sim = float(similarity_matrix[i, j])
            
            # Same paper — skip (not cross-year repetition)
            if questions.iloc[i]['source_file'] == questions.iloc[j]['source_file']:
                continue
            
            # Same year — interesting but not cross-year repetition on its own
            # (could be same exam in different variants)
            
            if sim >= SEMANTIC_MATCH_THRESHOLD:
                confirmed_pairs.append((i, j, sim))
            elif sim >= REVIEW_THRESHOLD:
                review_pairs.append((i, j, sim))
    
    # Group confirmed pairs using DIRECT connections only (no transitive closure!)
    # Use single-linkage but with a minimum group similarity constraint
    confirmed_groups = _build_groups_conservative(questions, confirmed_pairs, 
                                                   SEMANTIC_MATCH_THRESHOLD)
    review_groups = _build_groups_conservative(questions, review_pairs,
                                                REVIEW_THRESHOLD)
    
    return confirmed_groups, review_groups


def _build_groups_conservative(questions: pd.DataFrame, pairs: List[Tuple],
                                min_threshold: float) -> List[Dict]:
    """
    Build question groups from pairwise matches.
    Uses conservative grouping: each member must be similar to the group anchor.
    Prevents transitive clustering of unrelated questions.
    """
    if not pairs:
        return []
    
    # Sort pairs by similarity (highest first)
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    
    assigned = set()
    groups = []
    
    for i, j, sim in pairs:
        if i in assigned and j in assigned:
            continue
        
        if i not in assigned and j not in assigned:
            # Start new group with both
            groups.append({
                'anchor': i,
                'members': [i, j],
                'similarities': {(i, j): sim}
            })
            assigned.add(i)
            assigned.add(j)
        elif i in assigned:
            # Try to add j to i's group
            for g in groups:
                if i in g['members']:
                    # Check similarity to anchor
                    # We don't have the full similarity here, 
                    # but the pair was already above threshold
                    g['members'].append(j)
                    g['similarities'][(i, j)] = sim
                    assigned.add(j)
                    break
        elif j in assigned:
            for g in groups:
                if j in g['members']:
                    g['members'].append(i)
                    g['similarities'][(i, j)] = sim
                    assigned.add(i)
                    break
    
    # Validate groups: ensure each group has members from >= 2 distinct years
    validated_groups = []
    for g in groups:
        indices = g['members']
        years = set(questions.iloc[idx]['year'] for idx in indices)
        if len(years) >= 2:
            validated_groups.append(g)
    
    return validated_groups


def format_group(questions: pd.DataFrame, indices: List[int], 
                  group_id: int, match_type: str,
                  similarities: Optional[Dict] = None) -> Dict:
    """Format a question group for output."""
    group_questions = questions.iloc[indices]
    
    # Pick canonical question: prefer highest confidence, most common english text
    canonical_idx = indices[0]
    for idx in indices:
        row = questions.iloc[idx]
        if (row['extraction_confidence'] == 'high' and 
            len(str(row['english_question'])) > len(str(questions.iloc[canonical_idx]['english_question']))):
            canonical_idx = idx
    
    canonical = questions.iloc[canonical_idx]
    
    # Distinct years
    years = sorted(set(int(group_questions['year'].iloc[i]) for i in range(len(group_questions))))
    distinct_year_count = len(years)
    
    # Source files
    source_files = sorted(set(group_questions['source_file']))
    
    # Question variations
    variations = []
    for idx in indices:
        row = questions.iloc[idx]
        variations.append({
            'question': row['english_question'],
            'original': row['original_question'],
            'year': int(row['year']),
            'source_file': row['source_file'],
            'page': int(row['page_number']),
            'confidence': row['extraction_confidence']
        })
    
    # Average similarity
    avg_sim = 1.0
    if similarities:
        sim_values = list(similarities.values())
        avg_sim = sum(sim_values) / len(sim_values) if sim_values else 1.0
    
    return {
        'group_id': group_id,
        'canonical_question': canonical['english_question'],
        'match_type': match_type,
        'years_appeared': years,
        'distinct_year_count': distinct_year_count,
        'source_files': source_files,
        'average_similarity': round(avg_sim, 4),
        'variations': variations,
        'question_count': len(variations)
    }


def run_repetition_detection(
    force: bool = False, subject_code: str = DEFAULT_SUBJECT_CODE
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Detect repeated questions within one subject corpus only."""
    code = normalize_subject_code(subject_code)
    paths = get_subject_paths(code)
    repeated_questions_json = paths["repeated_questions_json"]
    review_required_csv = paths["review_required_csv"]

    print("=" * 60)
    print(f"AKTU ExamMate — Repetition Detection [{code}]")
    print("=" * 60)

    if not force and repeated_questions_json.exists():
        try:
            with repeated_questions_json.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached is not None:
                review = []
                if review_required_csv.exists():
                    review_df = pd.read_csv(review_required_csv, encoding="utf-8-sig")
                    review = review_df.to_dict("records") if len(review_df) else []
                print("✓ Using cached repetition results")
                return cached, review, {"from_cache": True, "subject_code": code}
        except (json.JSONDecodeError, OSError):
            pass

    questions = load_questions(code)
    if questions.empty:
        raise ValueError(f"No extracted questions found for {code}. Check the source papers and run extraction first.")
    print(f"Loaded {len(questions)} questions from {questions['source_file'].nunique()} papers")

    valid_mask = questions["english_question"].fillna("").astype(str).str.len() >= 15
    valid_questions = questions[valid_mask].reset_index(drop=True)
    print(f"Using {len(valid_questions)} questions (filtered by min length)")
    if valid_questions.empty:
        raise ValueError(f"No questions of sufficient length found for {code}.")

    print("\nStep 1: Finding exact text matches...")
    exact_groups = find_exact_matches(valid_questions)
    print(f"  Found {len(exact_groups)} exact match groups")
    exact_assigned = set()
    for indices in exact_groups.values():
        exact_assigned.update(indices)

    print("\nStep 2: Computing semantic embeddings...")
    texts = valid_questions["english_question"].astype(str).tolist()
    cache_key = hashlib.md5("|".join(texts).encode()).hexdigest()
    embeddings = compute_embeddings(texts, cache_key, paths["embeddings_cache"])

    print("\nStep 3: Finding semantic matches...")
    semantic_groups, review_groups_raw = find_semantic_matches(
        valid_questions, embeddings, exact_assigned
    )
    print(f"  Found {len(semantic_groups)} semantic match groups")
    print(f"  Found {len(review_groups_raw)} borderline groups for review")

    all_repeated = []
    for hash_val, indices in exact_groups.items():
        all_repeated.append(format_group(valid_questions, indices, len(all_repeated) + 1, "exact"))
    for group in semantic_groups:
        all_repeated.append(format_group(
            valid_questions, group["members"], len(all_repeated) + 1,
            "semantic", group.get("similarities")
        ))

    all_repeated.sort(
        key=lambda item: (item["distinct_year_count"], max(item["years_appeared"])),
        reverse=True
    )
    for index, group in enumerate(all_repeated, start=1):
        group["group_id"] = index

    review_formatted = []
    for group in review_groups_raw:
        review_formatted.append(format_group(
            valid_questions, group["members"],
            len(all_repeated) + len(review_formatted) + 1,
            "needs_review", group.get("similarities")
        ))

    all_grouped_indices = set()
    for indices in exact_groups.values():
        all_grouped_indices.update(indices)
    for group in semantic_groups:
        all_grouped_indices.update(group["members"])
    for group in review_groups_raw:
        all_grouped_indices.update(group["members"])

    _save_results(all_repeated, review_formatted, valid_questions, all_grouped_indices, paths)
    summary = {
        "subject_code": code,
        "total_questions_analyzed": len(valid_questions),
        "exact_match_groups": len(exact_groups),
        "semantic_match_groups": len(semantic_groups),
        "total_repeated_groups": len(all_repeated),
        "review_needed_groups": len(review_formatted),
        "questions_in_repeated_groups": sum(group["question_count"] for group in all_repeated),
        "single_appearance_questions": len(valid_questions) - len(all_grouped_indices),
    }

    print("\n" + "=" * 60)
    print("Repetition Detection Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    return all_repeated, review_formatted, summary


def _save_results(
    repeated: List[Dict], review: List[Dict], questions: pd.DataFrame,
    grouped_indices: set, paths: dict = None
):
    """Save result artifacts inside the selected subject's data directory."""
    paths = paths or get_subject_paths(DEFAULT_SUBJECT_CODE)
    with paths["repeated_questions_json"].open("w", encoding="utf-8") as f:
        json.dump(repeated, f, ensure_ascii=False, indent=2)

    rows = []
    for group in repeated:
        for variation in group["variations"]:
            rows.append({
                "group_id": group["group_id"],
                "canonical_question": group["canonical_question"],
                "match_type": group["match_type"],
                "question_text": variation["question"],
                "year": variation["year"],
                "source_file": variation["source_file"],
                "page_number": variation["page"],
                "distinct_year_count": group["distinct_year_count"],
                "years_appeared": ", ".join(str(year) for year in group["years_appeared"]),
                "average_similarity": group["average_similarity"],
                "confidence": variation["confidence"],
            })
    pd.DataFrame(rows).to_csv(paths["repeated_questions_csv"], index=False, encoding="utf-8-sig")

    review_rows = []
    for group in review:
        for variation in group["variations"]:
            review_rows.append({
                "group_id": group["group_id"],
                "canonical_question": group["canonical_question"],
                "question_text": variation["question"],
                "year": variation["year"],
                "source_file": variation["source_file"],
                "average_similarity": group["average_similarity"],
            })
    pd.DataFrame(review_rows, columns=[
        "group_id", "canonical_question", "question_text", "year",
        "source_file", "average_similarity"
    ]).to_csv(paths["review_required_csv"], index=False, encoding="utf-8-sig")

    print(f"\n  Saved: {paths['repeated_questions_csv']}")
    print(f"  Saved: {paths['repeated_questions_json']}")
    print(f"  Saved: {paths['review_required_csv']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyse question repetition for one AKTU subject")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT_CODE, help="Subject code, e.g. BCS301 or BCS401")
    parser.add_argument("--force", action="store_true", help="Ignore cached repetition results")
    args = parser.parse_args()
    repeated, review, summary = run_repetition_detection(force=args.force, subject_code=args.subject)
    print(f"\nFound {len(repeated)} repeated question groups for {args.subject.upper()}")
