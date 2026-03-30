"""
Granite Vision Extraction Evaluation Framework.

Evaluates Granite 4.0 3B Vision extraction quality on de-identified
medical documents before production rollout.

Metrics:
  1. Field exact-match accuracy (per document type)
  2. Table extraction accuracy (TEDS-like structural score)
  3. Abstention rate (how often the model returns null for fields)
  4. Hallucination rate (extracted values not present in source)
  5. Latency (p50, p95, p99)
  6. Fallback rate (how often Granite falls back to primary OCR)

Usage:
  python evaluation/granite_eval.py --data-dir ./evaluation/test_data --report-dir ./evaluation/reports
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OCR_PACKAGE_ROOT = _REPO_ROOT / "documnet ocr"
if _OCR_PACKAGE_ROOT.exists():
    ocr_root = str(_OCR_PACKAGE_ROOT)
    if ocr_root not in sys.path:
        sys.path.insert(0, ocr_root)


@dataclass
class FieldEvalResult:
    """Evaluation result for a single extracted field."""
    field_name: str
    expected: Any
    predicted: Any
    exact_match: bool
    partial_match: bool  # substring or normalized match
    is_hallucination: bool  # predicted value not in source text
    is_abstention: bool  # predicted is null when expected is not


@dataclass
class DocumentEvalResult:
    """Evaluation result for a single document."""
    doc_id: str
    doc_type: str
    filename: str
    total_fields: int
    exact_matches: int
    partial_matches: int
    hallucinations: int
    abstentions: int
    field_results: List[FieldEvalResult] = field(default_factory=list)
    table_accuracy: float = 0.0
    latency_ms: float = 0.0
    extraction_errors: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregate evaluation report."""
    total_documents: int = 0
    total_fields: int = 0
    exact_match_rate: float = 0.0
    partial_match_rate: float = 0.0
    hallucination_rate: float = 0.0
    abstention_rate: float = 0.0
    table_accuracy_mean: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    fallback_rate: float = 0.0
    per_doc_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    document_results: List[DocumentEvalResult] = field(default_factory=list)


def _unwrap_extracted_payload(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Support backend payloads that wrap extracted data with provenance."""
    extracted = fields.get("extracted_data")
    if isinstance(extracted, dict):
        return extracted
    return fields


def _flatten_fields(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict/list structures into comparable field paths."""
    if isinstance(value, dict):
        flattened: Dict[str, Any] = {}
        if not value and prefix:
            flattened[prefix] = {}
            return flattened
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_fields(item, next_prefix))
        return flattened

    if isinstance(value, list):
        flattened = {}
        if not value and prefix:
            flattened[prefix] = []
            return flattened
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            flattened.update(_flatten_fields(item, next_prefix))
        return flattened

    return {prefix or "value": value}


def _normalize(value: Any) -> str:
    """Normalize a value for comparison."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value).strip().lower().replace("-", "").replace("/", "").replace(" ", "")


def _is_in_source(value: str, source_text: str) -> bool:
    """Check if the extracted value appears in the source text."""
    if not value or not source_text:
        return False
    norm_val = _normalize(value)
    norm_src = _normalize(source_text)
    if not norm_val:
        return True  # empty/null is not a hallucination
    return norm_val in norm_src


def evaluate_field(
    field_name: str,
    expected: Any,
    predicted: Any,
    source_text: str = "",
) -> FieldEvalResult:
    """Evaluate a single extracted field against ground truth."""
    norm_expected = _normalize(expected)
    norm_predicted = _normalize(predicted)

    exact_match = norm_expected == norm_predicted
    partial_match = (
        exact_match
        or (norm_expected in norm_predicted if norm_expected else False)
        or (norm_predicted in norm_expected if norm_predicted else False)
    )

    is_abstention = bool(norm_expected and not norm_predicted)
    is_hallucination = bool(
        norm_predicted
        and not exact_match
        and source_text
        and not _is_in_source(str(predicted), source_text)
    )

    return FieldEvalResult(
        field_name=field_name,
        expected=expected,
        predicted=predicted,
        exact_match=exact_match,
        partial_match=partial_match,
        is_hallucination=is_hallucination,
        is_abstention=is_abstention,
    )


def evaluate_table_structure(expected_html: str, predicted_html: str) -> float:
    """Compute a simplified TEDS-like table accuracy score.

    Compares row/column counts and cell content overlap.
    Returns a score between 0.0 and 1.0.
    """
    import re

    def extract_cells(html: str) -> List[str]:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", html, re.DOTALL | re.IGNORECASE)
        return [_normalize(c) for c in cells]

    def count_rows(html: str) -> int:
        return len(re.findall(r"<tr", html, re.IGNORECASE))

    expected_cells = extract_cells(expected_html)
    predicted_cells = extract_cells(predicted_html)

    if not expected_cells:
        return 1.0 if not predicted_cells else 0.0

    # Structure score: row count similarity
    expected_rows = count_rows(expected_html)
    predicted_rows = count_rows(predicted_html)
    row_score = 1.0 - abs(expected_rows - predicted_rows) / max(expected_rows, 1)
    row_score = max(0.0, row_score)

    # Content score: cell overlap
    expected_set = set(expected_cells)
    predicted_set = set(predicted_cells)
    if expected_set:
        content_score = len(expected_set & predicted_set) / len(expected_set)
    else:
        content_score = 1.0

    # Weighted combination
    return round(0.4 * row_score + 0.6 * content_score, 4)


def evaluate_document(
    doc_id: str,
    doc_type: str,
    filename: str,
    expected_fields: Dict[str, Any],
    predicted_fields: Dict[str, Any],
    source_text: str = "",
    expected_table_html: str = "",
    predicted_table_html: str = "",
    latency_ms: float = 0.0,
) -> DocumentEvalResult:
    """Evaluate extraction results for a single document."""
    field_results = []
    expected_flat = _flatten_fields(expected_fields)
    predicted_flat = _flatten_fields(_unwrap_extracted_payload(predicted_fields))
    all_field_names = set(expected_flat.keys()) | set(predicted_flat.keys())

    for field_name in sorted(all_field_names):
        expected_val = expected_flat.get(field_name)
        predicted_val = predicted_flat.get(field_name)
        result = evaluate_field(field_name, expected_val, predicted_val, source_text)
        field_results.append(result)

    table_accuracy = 0.0
    if expected_table_html or predicted_table_html:
        table_accuracy = evaluate_table_structure(expected_table_html, predicted_table_html)

    total = len(field_results)
    return DocumentEvalResult(
        doc_id=doc_id,
        doc_type=doc_type,
        filename=filename,
        total_fields=total,
        exact_matches=sum(1 for r in field_results if r.exact_match),
        partial_matches=sum(1 for r in field_results if r.partial_match),
        hallucinations=sum(1 for r in field_results if r.is_hallucination),
        abstentions=sum(1 for r in field_results if r.is_abstention),
        field_results=field_results,
        table_accuracy=table_accuracy,
        latency_ms=latency_ms,
    )


def aggregate_results(doc_results: List[DocumentEvalResult]) -> EvalReport:
    """Aggregate individual document results into a report."""
    if not doc_results:
        return EvalReport()

    total_fields = sum(d.total_fields for d in doc_results)
    total_exact = sum(d.exact_matches for d in doc_results)
    total_partial = sum(d.partial_matches for d in doc_results)
    total_hallucinations = sum(d.hallucinations for d in doc_results)
    total_abstentions = sum(d.abstentions for d in doc_results)

    latencies = [d.latency_ms for d in doc_results if d.latency_ms > 0]
    table_scores = [d.table_accuracy for d in doc_results if d.table_accuracy > 0]

    # Per doc-type breakdown
    per_type: Dict[str, Dict[str, float]] = {}
    for d in doc_results:
        if d.doc_type not in per_type:
            per_type[d.doc_type] = {"total_fields": 0, "exact_matches": 0, "hallucinations": 0, "docs": 0}
        per_type[d.doc_type]["total_fields"] += d.total_fields
        per_type[d.doc_type]["exact_matches"] += d.exact_matches
        per_type[d.doc_type]["hallucinations"] += d.hallucinations
        per_type[d.doc_type]["docs"] += 1

    for dt, stats in per_type.items():
        stats["exact_match_rate"] = round(stats["exact_matches"] / max(stats["total_fields"], 1), 4)
        stats["hallucination_rate"] = round(stats["hallucinations"] / max(stats["total_fields"], 1), 4)

    sorted_latencies = sorted(latencies) if latencies else [0.0]

    def percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        index = max(0, math.ceil((len(values) * pct)) - 1)
        index = min(index, len(values) - 1)
        return round(values[index], 2)

    return EvalReport(
        total_documents=len(doc_results),
        total_fields=total_fields,
        exact_match_rate=round(total_exact / max(total_fields, 1), 4),
        partial_match_rate=round(total_partial / max(total_fields, 1), 4),
        hallucination_rate=round(total_hallucinations / max(total_fields, 1), 4),
        abstention_rate=round(total_abstentions / max(total_fields, 1), 4),
        table_accuracy_mean=round(statistics.mean(table_scores), 4) if table_scores else 0.0,
        latency_p50_ms=percentile(sorted_latencies, 0.50),
        latency_p95_ms=percentile(sorted_latencies, 0.95),
        latency_p99_ms=percentile(sorted_latencies, 0.99),
        per_doc_type=per_type,
        document_results=doc_results,
    )


def load_test_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load test data from a directory of JSON files.

    Each JSON file should contain:
    {
        "doc_id": "...",
        "doc_type": "lab_report",
        "filename": "sample_lab.png",
        "source_text": "...",
        "expected_fields": { ... },
        "expected_table_html": "...",
        "image_path": "..."
    }
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning("Test data directory not found: %s", data_dir)
        return []

    test_cases = []
    for fpath in sorted(data_path.glob("*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                test_cases.append(json.load(f))
        except Exception as e:
            logger.warning("Failed to load test case %s: %s", fpath, e)

    logger.info("Loaded %d test cases from %s", len(test_cases), data_dir)
    return test_cases


def run_evaluation(
    test_cases: List[Dict[str, Any]],
    granite_url: str = "http://localhost:8005/v1",
) -> EvalReport:
    """Run the full evaluation against a live Granite endpoint.

    For offline evaluation (no GPU), pass pre-computed predictions
    in each test case under 'predicted_fields'.
    """
    doc_results = []

    for tc in test_cases:
        predicted = tc.get("predicted_fields", {})
        latency = tc.get("latency_ms", 0.0)

        # If no pre-computed predictions and Granite is available, call it
        if not predicted and tc.get("image_path"):
            try:
                import httpx
                import base64

                image_path = tc["image_path"]
                with open(image_path, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode()

                from medicscan_ocr.backends.granite_vision import (
                    DOCUMENT_SCHEMAS, _build_kvp_prompt,
                )
                schema = DOCUMENT_SCHEMAS.get(tc["doc_type"], {})
                if schema:
                    prompt = _build_kvp_prompt(schema)
                    start = time.monotonic()
                    resp = httpx.post(
                        f"{granite_url}/chat/completions",
                        json={
                            "model": "ibm-granite/granite-4.0-3b-vision",
                            "messages": [{"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                                {"type": "text", "text": prompt},
                            ]}],
                            "max_tokens": 4096,
                            "temperature": 0,
                        },
                        timeout=120,
                    )
                    latency = (time.monotonic() - start) * 1000
                    resp.raise_for_status()
                    raw = resp.json()["choices"][0]["message"]["content"]
                    predicted = json.loads(raw)
            except Exception as e:
                logger.warning("Live Granite call failed for %s: %s", tc.get("doc_id"), e)
                predicted = {}

        result = evaluate_document(
            doc_id=tc.get("doc_id", "unknown"),
            doc_type=tc.get("doc_type", "unknown"),
            filename=tc.get("filename", "unknown"),
            expected_fields=tc.get("expected_fields", {}),
            predicted_fields=predicted,
            source_text=tc.get("source_text", ""),
            expected_table_html=tc.get("expected_table_html", ""),
            predicted_table_html=tc.get("predicted_table_html", ""),
            latency_ms=latency,
        )
        doc_results.append(result)

    return aggregate_results(doc_results)


def save_report(report: EvalReport, report_dir: str) -> Path:
    """Save evaluation report to JSON."""
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = report_path / f"granite_eval_{timestamp}.json"

    # Convert to serializable dict (exclude full document_results for summary)
    summary = {
        "total_documents": report.total_documents,
        "total_fields": report.total_fields,
        "exact_match_rate": report.exact_match_rate,
        "partial_match_rate": report.partial_match_rate,
        "hallucination_rate": report.hallucination_rate,
        "abstention_rate": report.abstention_rate,
        "table_accuracy_mean": report.table_accuracy_mean,
        "latency_p50_ms": report.latency_p50_ms,
        "latency_p95_ms": report.latency_p95_ms,
        "latency_p99_ms": report.latency_p99_ms,
        "fallback_rate": report.fallback_rate,
        "per_doc_type": report.per_doc_type,
        "timestamp": timestamp,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Also save detailed results
    detail_path = report_path / f"granite_eval_{timestamp}_detailed.json"
    detailed = []
    for dr in report.document_results:
        detailed.append({
            "doc_id": dr.doc_id,
            "doc_type": dr.doc_type,
            "filename": dr.filename,
            "total_fields": dr.total_fields,
            "exact_matches": dr.exact_matches,
            "partial_matches": dr.partial_matches,
            "hallucinations": dr.hallucinations,
            "abstentions": dr.abstentions,
            "table_accuracy": dr.table_accuracy,
            "latency_ms": dr.latency_ms,
            "extraction_errors": dr.extraction_errors,
            "field_results": [asdict(fr) for fr in dr.field_results],
        })
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)

    logger.info("Report saved to %s", filepath)
    return filepath


def print_report(report: EvalReport) -> None:
    """Print a human-readable evaluation summary."""
    print("\n" + "=" * 70)
    print("  GRANITE VISION 4.0 3B — EXTRACTION EVALUATION REPORT")
    print("=" * 70)
    print(f"  Documents evaluated:    {report.total_documents}")
    print(f"  Total fields:           {report.total_fields}")
    print(f"  Exact match rate:       {report.exact_match_rate:.1%}")
    print(f"  Partial match rate:     {report.partial_match_rate:.1%}")
    print(f"  Hallucination rate:     {report.hallucination_rate:.1%}")
    print(f"  Abstention rate:        {report.abstention_rate:.1%}")
    print(f"  Table accuracy (mean):  {report.table_accuracy_mean:.1%}")
    print(f"  Latency p50:            {report.latency_p50_ms:.0f}ms")
    print(f"  Latency p95:            {report.latency_p95_ms:.0f}ms")
    print(f"  Latency p99:            {report.latency_p99_ms:.0f}ms")
    print()
    print("  Per Document Type:")
    print("  " + "-" * 66)
    for doc_type, stats in sorted(report.per_doc_type.items()):
        print(f"    {doc_type:25s}  exact={stats['exact_match_rate']:.1%}  "
              f"halluc={stats['hallucination_rate']:.1%}  docs={int(stats['docs'])}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Granite Vision Extraction Evaluation")
    parser.add_argument("--data-dir", default="./evaluation/test_data",
                        help="Directory containing test case JSON files")
    parser.add_argument("--report-dir", default="./evaluation/reports",
                        help="Directory to save evaluation reports")
    parser.add_argument("--granite-url", default="http://localhost:8005/v1",
                        help="Granite Vision vLLM endpoint URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    test_cases = load_test_data(args.data_dir)
    if not test_cases:
        print("No test cases found. Create JSON files in", args.data_dir)
        print("See docstring in granite_eval.py for the expected format.")
        return

    report = run_evaluation(test_cases, granite_url=args.granite_url)
    save_report(report, args.report_dir)
    print_report(report)


if __name__ == "__main__":
    main()
