from __future__ import annotations

from evaluation.granite_eval import aggregate_results, evaluate_document


def test_evaluate_document_flattens_nested_fields():
    result = evaluate_document(
        doc_id="doc-1",
        doc_type="lab_report",
        filename="lab.png",
        expected_fields={
            "patient_name": "John Doe",
            "tests": [
                {"test_name": "Hemoglobin", "result_value": "13.2"},
                {"test_name": "WBC", "result_value": "7.1"},
            ],
        },
        predicted_fields={
            "patient_name": "John Doe",
            "tests": [
                {"test_name": "Hemoglobin", "result_value": "13.2"},
                {"test_name": "WBC", "result_value": "7.1"},
            ],
        },
        source_text="John Doe Hemoglobin 13.2 WBC 7.1",
    )

    assert result.total_fields == 5
    assert result.exact_matches == 5
    assert result.hallucinations == 0
    assert result.abstentions == 0


def test_evaluate_document_counts_hallucinated_extra_field():
    result = evaluate_document(
        doc_id="doc-2",
        doc_type="prescription",
        filename="rx.png",
        expected_fields={"patient_name": "Jane Doe"},
        predicted_fields={
            "patient_name": "Jane Doe",
            "refill_authorization_code": "ZX-999",
        },
        source_text="Patient name Jane Doe Prescription written today",
    )

    hallucinated = {
        field.field_name: field for field in result.field_results if field.is_hallucination
    }

    assert result.total_fields == 2
    assert result.exact_matches == 1
    assert result.hallucinations == 1
    assert "refill_authorization_code" in hallucinated


def test_evaluate_document_unwraps_backend_payload():
    result = evaluate_document(
        doc_id="doc-3",
        doc_type="discharge_summary",
        filename="discharge.png",
        expected_fields={"patient_name": "Alice", "discharge_date": "2026-03-01"},
        predicted_fields={
            "extracted_data": {
                "patient_name": "Alice",
                "discharge_date": "2026-03-01",
            },
            "provenance": {"page_number": 1},
        },
        source_text="Alice was discharged on 2026-03-01",
    )

    report = aggregate_results([result])

    assert result.exact_matches == 2
    assert report.exact_match_rate == 1.0
    assert report.hallucination_rate == 0.0
