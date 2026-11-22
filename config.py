"""AKTU ExamMate configuration with isolated per-subject corpora."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SUBJECTS_DIR = DATA_DIR / "subjects"
DEFAULT_SUBJECT_CODE = "BCS301"

# Add subject-specific aliases and discriminating keywords here when onboarding a course.
SUBJECT_REGISTRY = {
    "BCS301": {
        "name": "Data Structures",
        "description": "B.Tech 3rd Semester Data Structures & Algorithms",
        "aliases": ["DATA STRUCTURE", "DATA STRUCTURES", "DATA STRUCTURES USING C", "DATA STRUCTURE USING C"],
        "keywords": [
            "LINKED LIST", "STACK", "QUEUE", "BINARY TREE", "SORTING", "GRAPH",
            "HASHING", "ARRAY", "RECURSION", "AVL", "TRAVERSAL", "INFIX",
            "POSTFIX", "PREFIX", "BINARY SEARCH TREE", "BFS", "DFS", "HEAP"
        ],
        "legacy_codes": ["KCS301", "RCS305", "NCS301", "NCS-301", "KCS-301", "BCS-301", "RCS-305"],
    },
    "BCS401": {
        "name": "Operating Systems",
        "description": "B.Tech 4th Semester Operating Systems",
        "aliases": ["OPERATING SYSTEM", "OPERATING SYSTEMS"],
        "keywords": [
            "PROCESS SCHEDULING", "CPU SCHEDULING", "DEADLOCK", "SEMAPHORE",
            "PAGE REPLACEMENT", "VIRTUAL MEMORY", "MEMORY MANAGEMENT",
            "PROCESS SYNCHRONIZATION", "DISK SCHEDULING", "FILE SYSTEM",
            "BANKER ALGORITHM", "BANKER'S ALGORITHM", "THREAD MANAGEMENT"
        ],
        "legacy_codes": ["KCS401", "BCS-401", "KCS-401"],
    },
    "BCS302": {
        "name": "Computer Organization & Architecture",
        "description": "B.Tech 3rd Semester Computer Organization and Architecture",
        "aliases": ["COMPUTER ORGANIZATION", "COMPUTER ORGANISATION", "COMPUTER ARCHITECTURE"],
        "keywords": [
            "INSTRUCTION CYCLE", "ADDRESSING MODE", "CACHE MEMORY", "CONTROL UNIT",
            "ARITHMETIC LOGIC UNIT", "PIPELINING", "REGISTER ORGANIZATION",
            "INPUT OUTPUT ORGANIZATION", "MEMORY HIERARCHY", "MICROPROGRAMMED CONTROL"
        ],
        "legacy_codes": ["KCS302", "BCS-302", "KCS-302"],
    },
   "BCC302": {
    "name": "Python Programming",
    "description": "AKTU B.Tech Python Programming",
    "aliases": [
        "PYTHON PROGRAMMING",
        "PROGRAMMING IN PYTHON"
    ],
    "keywords": [
        "PYTHON VARIABLES",
        "PYTHON DATA TYPES",
        "CONTROL FLOW",
        "LIST COMPREHENSION",
        "TUPLES",
        "DICTIONARIES",
        "FUNCTIONS IN PYTHON",
        "RECURSION",
        "FILE HANDLING",
        "EXCEPTION HANDLING",
        "PYTHON MODULES",
        "PYTHON PACKAGES",
        "NUMPY",
        "MATPLOTLIB",
        "TKINTER",
        "GUI PROGRAMMING"
    ],
    "legacy_codes": ['BCC402']
    },
    "BCC401": {
    "name": "Cyber Security",
    "description": "AKTU B.Tech Cyber Security",
    "aliases": [
        "CYBER SECURITY",
        "INFORMATION SECURITY",
        "COMPUTER SECURITY"
    ],
    "keywords": [
        "CYBER SECURITY",
        "INFORMATION SECURITY",
        "NETWORK SECURITY",
        "CRYPTOGRAPHY",
        "AUTHENTICATION",
        "ACCESS CONTROL",
        "MALWARE",
        "PHISHING",
        "FIREWALL",
        "INTRUSION DETECTION",
        "SECURITY PROTOCOLS",
        "DIGITAL SIGNATURE",
        "ENCRYPTION",
        "CYBER ATTACKS",
        "VULNERABILITY"
    ],
    "legacy_codes": []
    },
}


def normalize_subject_code(subject_code: str) -> str:
    code = (subject_code or DEFAULT_SUBJECT_CODE).strip().upper()
    if code not in SUBJECT_REGISTRY:
        raise ValueError(
            f"Unknown subject code: {code}. Available subjects: {', '.join(SUBJECT_REGISTRY)}"
        )
    return code


def get_subject_paths(subject_code: str = DEFAULT_SUBJECT_CODE, create: bool = True) -> dict:
    """Return all input/output paths for one isolated subject corpus."""
    code = normalize_subject_code(subject_code)
    base = SUBJECTS_DIR / code
    paths = {
        "base_dir": base,
        "pdf_dir": base / "pdfs" / "raw",
        "source_dir": base / "source",
        "cleaned_csv": base / "source" / "cleaned.csv",
        "cache_dir": base / "cache",
        "questions_csv": base / "questions.csv",
        "questions_json": base / "questions.json",
        "repeated_questions_csv": base / "repeated_questions.csv",
        "repeated_questions_json": base / "repeated_questions.json",
        "review_required_csv": base / "review_required.csv",
        "extraction_cache": base / "cache" / "extraction_cache.json",
        "embeddings_cache": base / "cache" / "embeddings_cache.npz",
    }
    if create:
        for key in ("pdf_dir", "source_dir", "cache_dir"):
            paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def subject_is_mounted(subject_code: str) -> bool:
    paths = get_subject_paths(subject_code, create=False)
    return paths["questions_json"].exists() and paths["repeated_questions_json"].exists()


# Backwards-compatible default-subject constants for older utility code.
_DEFAULT_PATHS = get_subject_paths(DEFAULT_SUBJECT_CODE)
CSV_PATH = str(_DEFAULT_PATHS["cleaned_csv"])
QUESTIONS_CSV = str(_DEFAULT_PATHS["questions_csv"])
QUESTIONS_JSON = str(_DEFAULT_PATHS["questions_json"])
REPEATED_CSV = str(_DEFAULT_PATHS["repeated_questions_csv"])
REPEATED_JSON = str(_DEFAULT_PATHS["repeated_questions_json"])
REVIEW_CSV = str(_DEFAULT_PATHS["review_required_csv"])
CACHE_DIR = str(_DEFAULT_PATHS["cache_dir"])
EXTRACTION_CACHE = str(_DEFAULT_PATHS["extraction_cache"])
EMBEDDINGS_CACHE = str(_DEFAULT_PATHS["embeddings_cache"])

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EXACT_MATCH_THRESHOLD = 0.98
SEMANTIC_MATCH_THRESHOLD = 0.82
REVIEW_THRESHOLD = 0.72

# Legacy Data Structures aliases retained for compatibility.
DS_SUBJECT_CODES = {"BCS301", "KCS301", "RCS305", "NCS301", "NCS-301", "KCS-301", "BCS-301", "RCS-305"}
DS_SUBJECT_NAMES = set(SUBJECT_REGISTRY["BCS301"]["aliases"])
