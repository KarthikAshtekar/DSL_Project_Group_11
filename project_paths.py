"""Shared project paths for runnable scripts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARCHIVES_DIR = PROJECT_ROOT / "archives"
BERT_OUTPUTS_DIR = OUTPUTS_DIR / "bert"
BERT_RESULTS_DIR = BERT_OUTPUTS_DIR / "results"
BERT_LOGS_DIR = BERT_OUTPUTS_DIR / "logs"


def ensure_standard_directories() -> None:
    """Create the standard project folders if they do not already exist."""

    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUTS_DIR,
        PLOTS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        ARCHIVES_DIR,
        BERT_OUTPUTS_DIR,
        BERT_RESULTS_DIR,
        BERT_LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def first_existing_path(*candidates: Path) -> Path:
    """Return the first existing path from the provided candidates."""

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "None of the candidate paths exist: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


ensure_standard_directories()
