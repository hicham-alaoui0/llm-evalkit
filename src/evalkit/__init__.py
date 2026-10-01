"""evalkit: deterministic evaluation for RAG / LLM question answering."""

from .compare import compare
from .report import summarize
from .scoring import score
from .stats import mcnemar_exact, wilson

__version__ = "0.1.0"
__all__ = ["score", "summarize", "compare", "wilson", "mcnemar_exact"]
