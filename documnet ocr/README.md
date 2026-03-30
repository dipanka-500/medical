# MediScan OCR-X v1.0

Adaptive OCR system for document images, PDFs, ZIP bundles of page images, and optionally Office docs with conversion support:

- preprocessing for difficult scans
- printed vs handwritten detection using your `binary_handwriting_classifier.pth`
- routing logic that now follows your main policy exactly:
  `printed+english -> FireRed`
  `printed+indic -> Surya`
  `handwritten+english -> Chandra`
  `handwritten+indic -> Sarvam`
- a real Sarvam Document Intelligence adapter for sending PDFs and images through the API
- page bundling and collation so multi-page inputs can be split into page images and merged back into one result
- structured JSON output with confidence, route trace, and fallback results

## Why this build

You asked for an OCR stack that combines specialist models instead of trusting a single model everywhere. This project turns that into a working Python service:

- Sarvam Vision is the implemented remote API path for handwritten Indic and API-driven document OCR.
- FireRed, Surya, and Chandra are now the actual primary routes chosen by policy, not just described in comments.
- LayoutLMv3 and Table Transformer are treated as layout/table enrichers.
- Donut and Pixtral are treated as structure-aware fallback or review models for difficult pages.

## Architecture

1. Input document or image
2. Adaptive preprocessing
3. Document intelligence analysis
4. Dynamic OCR routing
5. Backend execution
6. Result fusion
7. Structured JSON output

## Implemented pieces

### Working now

- image preprocessing with OpenCV
- handwriting classification with your checkpoint
- handwriting English-vs-Indic script detection without relying on a remote API
- document bundling for images, ZIPs, and PDFs
- routing engine
- Sarvam Vision Document Intelligence adapter
- FireRed backend hook with runtime checks for local model dependencies
- CLI
- FastAPI wrapper that accepts local file paths
- tests for routing and end-to-end dry-run behavior

### Still optional

- docTR
- Donut
- Pixtral
- LayoutLMv3
- Table Transformer

These are still optional because the current machine does not have `transformers` or the heavier OCR runtimes installed.

## Install

```bash
pip install -e .
```

For Sarvam API support:

```bash
pip install -e ".[sarvam]"
```

For local PDF page splitting:

```bash
pip install -e ".[docsplit]"
```

## Environment

```powershell
$env:SARVAM_API_KEY="your_api_key"
```

Optional overrides:

- `MEDISCAN_DEFAULT_LANGUAGE_CODE` defaults to `en-IN`
- `MEDISCAN_DEFAULT_INDIC_LANGUAGE_CODE` defaults to `hi-IN`
- `MEDISCAN_SARVAM_OUTPUT_FORMAT` defaults to `md`
- `MEDISCAN_HANDWRITING_INDEX` defaults to `1`

## Usage

Run the CLI:

```bash
python -m medicscan_ocr.cli sample.pdf --language hi-IN --pretty
```

Force the Sarvam backend:

```bash
python -m medicscan_ocr.cli sample.jpg --backend sarvam_vision --language ta-IN
```

Dry-run the routing logic without calling any remote API:

```bash
python -m medicscan_ocr.cli sample.jpg --dry-run --pretty
```

Run the local HTTP API:

```bash
python -m uvicorn medicscan_ocr.app:app --reload
```

Then call:

```http
POST /ocr
{
  "path": "C:\\path\\to\\document.pdf",
  "language_hint": "hi-IN",
  "backend": "auto",
  "dry_run": false
}
```

## Router policy

- `FireRed-OCR`: main model for printed English
- `Surya`: main model for printed Indic
- `Chandra OCR 2`: main model for handwritten English
- `Sarvam Vision`: main model for handwritten Indic
- `docTR`: fallback OCR backend
- `LayoutLMv3`: layout-aware enrichment and section reasoning
- `Table Transformer`: table detection and structure recognition
- `Donut`: end-to-end document understanding fallback for structure-heavy pages
- `Pixtral`: multimodal reviewer for highly complex layouts or contradiction checks

If a multi-page file can be split locally into page images, the service collates those pages back into one ordered result. If local PDF or Office conversion support is missing, the pipeline falls back to native-file submission where the backend supports it.

## Official documentation used

- FireRed-OCR repo: https://github.com/FireRedTeam/FireRed-OCR
- Surya repo: https://github.com/VikParuchuri/surya
- docTR repo: https://github.com/mindee/doctr
- Chandra repo: https://github.com/datalab-to/chandra
- Sarvam Vision docs: https://docs.sarvam.ai/api-reference-docs/getting-started/models/sarvam-vision
- Sarvam Document Intelligence overview: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/document-intelligence/overview
- LayoutLMv3 docs: https://huggingface.co/docs/transformers/main/model_doc/layoutlmv3
- Table Transformer docs: https://huggingface.co/docs/transformers/main/model_doc/table-transformer
- Donut docs: https://huggingface.co/docs/transformers/main/model_doc/donut
- Pixtral docs: https://huggingface.co/docs/transformers/main/model_doc/pixtral

## Notes on Sarvam Vision

The current Sarvam docs describe a job-based Document Intelligence flow:

1. `client.document_intelligence.create_job(language=..., output_format=...)`
2. `job.upload_file("document.pdf")`
3. `job.start()`
4. `job.wait_until_complete()`
5. `job.download_output("./output.zip")`

This repository follows that flow in the `SarvamVisionBackend`.

## Review fixes already applied

- blocked ZIP path traversal during extraction
- added timeouts for local command backends
- improved multi-page collation for Chandra and Surya outputs
- removed the earlier router bug that over-preferred Sarvam just because an API key existed
