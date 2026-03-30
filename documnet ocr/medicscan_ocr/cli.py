from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from medicscan_ocr.config import load_settings
from medicscan_ocr.service import MediScanOCRService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive OCR pipeline with Sarvam Vision integration."
    )
    parser.add_argument("input_path", help="Path to an image or PDF.")
    parser.add_argument(
        "--language",
        dest="language_hint",
        default=None,
        help="Language hint like en-IN, hi-IN, ta-IN.",
    )
    parser.add_argument(
        "--document-type",
        dest="document_type_hint",
        default=None,
        help="Optional override: printed, handwritten, or mixed.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="Backend override. Examples: auto, sarvam_vision, surya_command, chandra_command.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and route without executing OCR backends.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON to stdout.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    settings = load_settings()
    service = MediScanOCRService(settings)
    result = service.process(
        input_path=args.input_path,
        language_hint=args.language_hint,
        backend=args.backend,
        document_type_hint=args.document_type_hint,
        dry_run=args.dry_run,
    )

    output_payload = result.to_dict()
    output_text = json.dumps(
        output_payload,
        indent=2 if args.pretty else None,
        ensure_ascii=False,
    )
    print(output_text)

    if args.output:
        output_path = service.write_result(result, args.output)
        print("Saved result to {0}".format(output_path), file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
