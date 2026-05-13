"""Evaluation helpers for Agent/RAG regression runs."""

from multimodal_rag_agent.eval.badcase import extract_badcase_drafts, write_badcase_drafts
from multimodal_rag_agent.eval.compare import build_baseline_comparison, load_summary
from multimodal_rag_agent.eval.dataset import load_eval_cases, select_cases
from multimodal_rag_agent.eval.metrics import evaluate_case, evaluate_retrieval_case, summarize_results
from multimodal_rag_agent.eval.models import EvalActual, EvalCase, EvalResult

__all__ = [
    "EvalActual",
    "EvalCase",
    "EvalResult",
    "build_baseline_comparison",
    "evaluate_case",
    "evaluate_retrieval_case",
    "extract_badcase_drafts",
    "load_eval_cases",
    "load_summary",
    "select_cases",
    "summarize_results",
    "write_badcase_drafts",
]
