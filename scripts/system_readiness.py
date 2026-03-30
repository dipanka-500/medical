#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT / "deployment" / "jarvislabs" / "jarvislabs.env"

SAMPLE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2qW9kAAAAASUVORK5CYII="
)

MEDICAL_MODEL_PREFERENCES = (
    "biomistral_7b",
    "medix_r1_2b",
    "mellama_13b",
    "chatdoctor",
    "deepseek_r1",
)

MEDISCAN_MODEL_PREFERENCES = (
    "medgemma_4b",
    "medix_r1_2b",
    "chexagent_3b",
    "biomedclip",
    "hulu_med_7b",
)


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {"pass", "warn", "skip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jarvis full-stack readiness harness for MedAI.",
    )
    parser.add_argument(
        "--mode",
        choices=("static", "health", "live", "full"),
        default="full",
        help="Which readiness stages to run.",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to deployment env file.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Default HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--auth-email",
        default="",
        help="Patient email for platform live probes. Defaults to a generated address.",
    )
    parser.add_argument(
        "--auth-password",
        default="MedAIReadiness!234",
        help="Password for generated or existing readiness user.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip lightweight pytest suites during static/full mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    return parser.parse_args()


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip().strip("'").strip('"')
    return parsed


def env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: float = 20.0,
) -> tuple[int, bytes, dict[str, str]]:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.getcode(), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers.items())


def request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> tuple[int, dict[str, Any] | list[Any] | None]:
    body = None
    merged_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        merged_headers.setdefault("Content-Type", "application/json")
    status_code, raw_body, _ = request(
        method,
        url,
        headers=merged_headers,
        data=body,
        timeout=timeout,
    )
    return status_code, safe_json(raw_body)


def request_form(
    url: str,
    fields: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> tuple[int, dict[str, Any] | list[Any] | None]:
    encoded = urllib.parse.urlencode(
        {key: str(value) for key, value in fields.items() if value is not None},
    ).encode("utf-8")
    merged_headers = dict(headers or {})
    merged_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    status_code, raw_body, _ = request(
        "POST",
        url,
        headers=merged_headers,
        data=encoded,
        timeout=timeout,
    )
    return status_code, safe_json(raw_body)


def request_multipart(
    url: str,
    *,
    fields: dict[str, Any] | None = None,
    files: list[tuple[str, str, str, bytes]] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> tuple[int, dict[str, Any] | list[Any] | None]:
    boundary = f"----MedAIReadiness{uuid.uuid4().hex}"
    body = bytearray()

    for key, value in (fields or {}).items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )

    for field_name, filename, content_type, payload in files or []:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        body.extend(payload)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    merged_headers = dict(headers or {})
    merged_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    status_code, raw_body, _ = request(
        "POST",
        url,
        headers=merged_headers,
        data=bytes(body),
        timeout=timeout,
    )
    return status_code, safe_json(raw_body)


def safe_json(body: bytes) -> dict[str, Any] | list[Any] | None:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def choose_model(model_keys: list[str], preferences: tuple[str, ...]) -> str | None:
    if not model_keys:
        return None
    for preferred in preferences:
        if preferred in model_keys:
            return preferred
    return sorted(model_keys)[0]


class ReadinessHarness:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.env_file = Path(args.env_file)
        self.file_env = parse_env_file(self.env_file)
        self.env = {**self.file_env, **os.environ}
        self.results: list[CheckResult] = []

        self.public_port = self.env.get("PUBLIC_PORT", "6006")
        self.platform_port = self.env.get("PLATFORM_PORT", "8000")
        self.general_llm_port = self.env.get("GENERAL_LLM_PORT", "8004")
        self.medical_llm_port = self.env.get("MEDICAL_LLM_PORT", "8002")
        self.mediscan_port = self.env.get("MEDISCAN_PORT", "8001")
        self.ocr_port = self.env.get("OCR_PORT", "8003")

        self.platform_url = f"http://127.0.0.1:{self.platform_port}"
        self.general_llm_url = f"http://127.0.0.1:{self.general_llm_port}"
        self.medical_llm_url = f"http://127.0.0.1:{self.medical_llm_port}"
        self.mediscan_url = f"http://127.0.0.1:{self.mediscan_port}"
        self.ocr_url = f"http://127.0.0.1:{self.ocr_port}"
        self.openrag_url = self.env.get("OPENRAG_URL", "http://127.0.0.1:8006")
        self.context_graph_url = self.env.get("CONTEXT_GRAPH_URL", "http://127.0.0.1:8007")
        self.context1_url = self.env.get("CONTEXT1_URL", "http://127.0.0.1:8008")
        self.medical_llm_api_key = self.env.get("MEDICAL_LLM_API_KEY", "")

    def add(self, name: str, status: str, message: str, **details: Any) -> None:
        self.results.append(CheckResult(name=name, status=status, message=message, details=details))

    def run(self) -> int:
        if self.args.mode in {"static", "full"}:
            self.run_static_checks()
        if self.args.mode in {"health", "full"}:
            self.run_health_checks()
        if self.args.mode in {"live", "full"}:
            self.run_live_probes()

        if self.args.json:
            payload = [asdict(result) for result in self.results]
            print(json.dumps(payload, indent=2))
        else:
            self.print_human()

        return 0 if all(result.ok for result in self.results) else 1

    def print_human(self) -> None:
        symbols = {"pass": "PASS", "warn": "WARN", "skip": "SKIP", "fail": "FAIL"}
        for result in self.results:
            print(f"[{symbols.get(result.status, result.status.upper())}] {result.name}: {result.message}")
            for key, value in sorted(result.details.items()):
                print(f"  - {key}: {value}")

    def run_static_checks(self) -> None:
        required_paths = [
            ROOT / "deployment" / "jarvislabs" / "launch_stack.sh",
            ROOT / "deployment" / "jarvislabs" / "launch_text_chat_stack.sh",
            ROOT / "deployment" / "jarvislabs" / "bootstrap_envs.sh",
            ROOT / "scripts" / "download_models.py",
            ROOT / "scripts" / "system_readiness.py",
            ROOT / "platform",
            ROOT / "general_llm",
            ROOT / "medical_llm",
            ROOT / "mediscan_v70_sota_production",
            ROOT / "documnet ocr",
        ]

        missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.exists()]
        if missing:
            self.add(
                "static.paths",
                "fail",
                "Required repository paths are missing.",
                missing=", ".join(missing),
            )
        else:
            self.add("static.paths", "pass", "Required repository paths are present.")

        venv_targets = [
            ("platform", ROOT / "platform" / ".venv" / "bin" / "python"),
            ("general_llm", ROOT / "general_llm" / ".venv" / "bin" / "python"),
            ("medical_llm", ROOT / "medical_llm" / ".venv" / "bin" / "python"),
            ("mediscan", ROOT / "mediscan_v70_sota_production" / "mediscan_v70" / ".venv" / "bin" / "python"),
            ("ocr", ROOT / "documnet ocr" / ".venv" / "bin" / "python"),
        ]

        optional_targets = []
        if env_bool(self.env.get("JARVIS_INSTALL_OPTIONAL_STACK")) or env_bool(self.env.get("ENABLE_OPENRAG")):
            optional_targets.append(("openrag", ROOT / "openrag_service" / ".venv" / "bin" / "python"))
        if env_bool(self.env.get("JARVIS_INSTALL_OPTIONAL_STACK")) or env_bool(self.env.get("ENABLE_CONTEXT_GRAPH")):
            optional_targets.append(("context_graph", ROOT / "context_graph_service" / ".venv" / "bin" / "python"))
        if env_bool(self.env.get("JARVIS_INSTALL_OPTIONAL_STACK")) or env_bool(self.env.get("ENABLE_CONTEXT1_AGENT")):
            optional_targets.append(("context1_agent", ROOT / "context1_agent" / ".venv" / "bin" / "python"))

        missing_venvs = [name for name, path in (venv_targets + optional_targets) if not path.exists()]
        if missing_venvs:
            self.add(
                "static.venvs",
                "warn",
                "Some service virtualenvs are missing. Run deployment/jarvislabs/bootstrap_envs.sh.",
                missing=", ".join(missing_venvs),
            )
        else:
            self.add("static.venvs", "pass", "Service virtualenvs are present.")

        if not self.env_file.exists():
            self.add(
                "static.env",
                "warn",
                "Env file is missing. Copy deployment/jarvislabs/jarvislabs.env.example first.",
                env_file=str(self.env_file),
            )
        else:
            self.add("static.env", "pass", "Env file loaded.", env_file=str(self.env_file))

        if not self.args.skip_tests:
            for result in self.run_test_suites():
                self.results.append(result)

    def run_test_suites(self) -> list[CheckResult]:
        suite_defs = [
            (
                "tests.platform",
                ROOT / "platform",
                [
                    ROOT / "platform" / "tests" / "test_master_router_health.py",
                    ROOT / "platform" / "tests" / "test_voice_support.py",
                ],
            ),
            (
                "tests.medical_llm",
                ROOT / "medical_llm",
                [
                    ROOT / "medical_llm" / "tests" / "test_api.py",
                    ROOT / "medical_llm" / "tests" / "test_routing.py",
                ],
            ),
            (
                "tests.ocr",
                ROOT / "documnet ocr",
                [
                    ROOT / "documnet ocr" / "tests" / "test_service_dry_run.py",
                    ROOT / "documnet ocr" / "tests" / "test_document_prepare.py",
                    ROOT / "documnet ocr" / "tests" / "test_routing.py",
                ],
            ),
            (
                "tests.evaluation",
                ROOT,
                [ROOT / "evaluation" / "test_granite_eval.py"],
            ),
        ]

        def run_suite(name: str, workdir: Path, tests: list[Path]) -> CheckResult:
            python_bin = workdir / ".venv" / "bin" / "python"
            if not python_bin.exists():
                return CheckResult(name=name, status="skip", message="Skipped because virtualenv is missing.")

            for test_path in tests:
                if not test_path.exists():
                    return CheckResult(
                        name=name,
                        status="skip",
                        message="Skipped because one or more test files are missing.",
                        details={"missing": str(test_path.relative_to(ROOT))},
                    )

            cmd = [str(python_bin), "-m", "pytest", "-q", *[str(test) for test in tests]]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return CheckResult(name=name, status="fail", message="Timed out while running pytest.")

            output = (proc.stdout + "\n" + proc.stderr).strip()
            if proc.returncode == 0:
                return CheckResult(name=name, status="pass", message="Pytest suite passed.")
            return CheckResult(
                name=name,
                status="fail",
                message=f"Pytest suite failed with exit code {proc.returncode}.",
                details={"output_tail": "\n".join(output.splitlines()[-10:])},
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(run_suite, name, workdir, tests)
                for name, workdir, tests in suite_defs
            ]
            return [future.result() for future in futures]

    def run_health_checks(self) -> None:
        services = {
            "health.platform": f"{self.platform_url}/api/v1/health/ready",
            "health.general_llm": f"{self.general_llm_url}/v1/models",
            "health.medical_llm": f"{self.medical_llm_url}/ready",
            "health.mediscan": f"{self.mediscan_url}/health",
            "health.ocr": f"{self.ocr_url}/health",
        }

        if env_bool(self.env.get("ENABLE_OPENRAG")):
            services["health.openrag"] = f"{self.openrag_url.rstrip('/')}/health"
        if env_bool(self.env.get("ENABLE_CONTEXT_GRAPH")):
            services["health.context_graph"] = f"{self.context_graph_url.rstrip('/')}/health"
        if env_bool(self.env.get("ENABLE_CONTEXT1_AGENT")):
            services["health.context1"] = f"{self.context1_url.rstrip('/')}/health"

        def probe(name: str, url: str) -> CheckResult:
            try:
                status_code, payload = request_json("GET", url, timeout=self.args.timeout)
            except Exception as exc:
                return CheckResult(name=name, status="fail", message=f"Request failed: {exc}")

            if 200 <= status_code < 300:
                detail_status = payload.get("status") if isinstance(payload, dict) else None
                detail_ready = payload.get("ready") if isinstance(payload, dict) else None
                return CheckResult(
                    name=name,
                    status="pass",
                    message=f"HTTP {status_code}",
                    details={"payload_status": detail_status, "ready": detail_ready},
                )

            return CheckResult(
                name=name,
                status="fail",
                message=f"HTTP {status_code}",
                details={"payload": payload},
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(services)) as pool:
            futures = [pool.submit(probe, name, url) for name, url in services.items()]
            for future in futures:
                self.results.append(future.result())

    def run_live_probes(self) -> None:
        self.results.append(self.probe_general_llm())
        self.results.append(self.probe_medical_llm())
        self.results.append(self.probe_mediscan())
        self.results.append(self.probe_ocr())

        if env_bool(self.env.get("ENABLE_OPENRAG")):
            self.results.append(self.probe_openrag())
        else:
            self.add("live.openrag", "skip", "Skipped because ENABLE_OPENRAG is false.")

        if env_bool(self.env.get("ENABLE_CONTEXT_GRAPH")):
            self.results.append(self.probe_context_graph())
        else:
            self.add("live.context_graph", "skip", "Skipped because ENABLE_CONTEXT_GRAPH is false.")

        if env_bool(self.env.get("ENABLE_CONTEXT1_AGENT")):
            self.results.append(self.probe_context1())
        else:
            self.add("live.context1", "skip", "Skipped because ENABLE_CONTEXT1_AGENT is false.")

        self.results.append(self.probe_platform())

    def probe_general_llm(self) -> CheckResult:
        payload = {
            "model": self.env.get("GENERAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            "messages": [{"role": "user", "content": "Reply with the single word ready."}],
            "max_tokens": 16,
            "temperature": 0,
        }
        try:
            status_code, body = request_json(
                "POST",
                f"{self.general_llm_url}/v1/chat/completions",
                payload=payload,
                timeout=max(self.args.timeout, 60.0),
            )
        except Exception as exc:
            return CheckResult("live.general_llm", "fail", f"Request failed: {exc}")

        if status_code != 200 or not isinstance(body, dict):
            return CheckResult(
                "live.general_llm",
                "fail",
                f"Unexpected response: HTTP {status_code}",
                details={"payload": body},
            )

        try:
            content = body["choices"][0]["message"]["content"]
        except Exception:
            content = ""
        if not content:
            return CheckResult(
                "live.general_llm",
                "fail",
                "Chat completion returned no content.",
                details={"payload": body},
            )

        return CheckResult(
            "live.general_llm",
            "pass",
            "General LLM chat completion succeeded.",
            details={"reply": content[:120]},
        )

    def medical_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.medical_llm_api_key:
            headers["X-API-Key"] = self.medical_llm_api_key
        return headers

    def probe_medical_llm(self) -> CheckResult:
        headers = self.medical_headers()
        try:
            status_code, body = request_json(
                "GET",
                f"{self.medical_llm_url}/models",
                headers=headers,
                timeout=max(self.args.timeout, 60.0),
            )
        except Exception as exc:
            return CheckResult("live.medical_llm", "fail", f"Model list failed: {exc}")

        if status_code != 200 or not isinstance(body, dict):
            return CheckResult(
                "live.medical_llm",
                "fail",
                f"Unexpected response: HTTP {status_code}",
                details={"payload": body},
            )

        models = body.get("models", {}) if isinstance(body, dict) else {}
        model_key = choose_model(list(models), MEDICAL_MODEL_PREFERENCES)
        if not model_key:
            return CheckResult("live.medical_llm", "fail", "No medical LLM models were registered.")

        load_status, load_body = request_json(
            "POST",
            f"{self.medical_llm_url}/models/{model_key}/load",
            headers=headers,
            timeout=max(self.args.timeout, 120.0),
        )
        if load_status != 200:
            return CheckResult(
                "live.medical_llm",
                "fail",
                f"Explicit load failed for {model_key}.",
                details={"payload": load_body},
            )

        analyze_status, analyze_body = request_json(
            "POST",
            f"{self.medical_llm_url}/analyze",
            payload={
                "query": "Explain hypertension in one sentence.",
                "mode": "patient",
                "enable_rag": False,
                "force_models": [model_key],
                "use_cache": False,
                "session_id": f"readiness-{uuid.uuid4().hex[:8]}",
            },
            headers=headers,
            timeout=max(self.args.timeout, 180.0),
        )

        request_json(
            "POST",
            f"{self.medical_llm_url}/models/{model_key}/unload",
            headers=headers,
            timeout=max(self.args.timeout, 120.0),
        )

        if analyze_status != 200 or not isinstance(analyze_body, dict):
            return CheckResult(
                "live.medical_llm",
                "fail",
                f"Analyze probe failed for {model_key}.",
                details={"payload": analyze_body},
            )

        answer = analyze_body.get("report_text") or analyze_body.get("answer") or ""
        return CheckResult(
            "live.medical_llm",
            "pass",
            f"Medical LLM load/analyze/unload succeeded via {model_key}.",
            details={"reply": str(answer)[:160]},
        )

    def probe_mediscan(self) -> CheckResult:
        try:
            status_code, body = request_json(
                "GET",
                f"{self.mediscan_url}/models",
                timeout=max(self.args.timeout, 60.0),
            )
        except Exception as exc:
            return CheckResult("live.mediscan", "fail", f"Model list failed: {exc}")

        if status_code != 200 or not isinstance(body, dict):
            return CheckResult(
                "live.mediscan",
                "fail",
                f"Unexpected response: HTTP {status_code}",
                details={"payload": body},
            )

        models = body.get("models", {})
        model_key = choose_model(list(models), MEDISCAN_MODEL_PREFERENCES)
        if not model_key:
            return CheckResult("live.mediscan", "fail", "No MediScan models were registered.")

        load_status, load_body = request_json(
            "POST",
            f"{self.mediscan_url}/models/{model_key}/load",
            timeout=max(self.args.timeout, 180.0),
        )
        unload_status, unload_body = request_json(
            "POST",
            f"{self.mediscan_url}/models/{model_key}/unload",
            timeout=max(self.args.timeout, 120.0),
        )

        if load_status != 200:
            return CheckResult(
                "live.mediscan",
                "fail",
                f"Explicit load failed for {model_key}.",
                details={"payload": load_body},
            )
        if unload_status != 200:
            return CheckResult(
                "live.mediscan",
                "fail",
                f"Explicit unload failed for {model_key}.",
                details={"payload": unload_body},
            )

        return CheckResult(
            "live.mediscan",
            "pass",
            f"MediScan load/unload succeeded via {model_key}.",
        )

    def probe_ocr(self) -> CheckResult:
        try:
            status_code, body = request_multipart(
                f"{self.ocr_url}/ocr",
                fields={"backend": "auto", "dry_run": "true"},
                files=[("file", "readiness.png", "image/png", SAMPLE_PNG)],
                timeout=max(self.args.timeout, 60.0),
            )
        except Exception as exc:
            return CheckResult("live.ocr", "fail", f"OCR dry-run failed: {exc}")

        if status_code != 200 or not isinstance(body, dict):
            return CheckResult(
                "live.ocr",
                "fail",
                f"Unexpected response: HTTP {status_code}",
                details={"payload": body},
            )

        route = body.get("route", {}) if isinstance(body, dict) else {}
        return CheckResult(
            "live.ocr",
            "pass",
            "OCR dry-run succeeded.",
            details={"primary_backend": route.get("primary_backend")},
        )

    def probe_openrag(self) -> CheckResult:
        try:
            status_code, body = request_form(
                f"{self.openrag_url.rstrip('/')}/search",
                {"query": "hypertension treatment", "top_k": 3, "use_reranker": "false"},
                timeout=max(self.args.timeout, 90.0),
            )
        except Exception as exc:
            return CheckResult("live.openrag", "fail", f"Search failed: {exc}")

        if status_code != 200 or not isinstance(body, dict):
            return CheckResult(
                "live.openrag",
                "fail",
                f"Unexpected response: HTTP {status_code}",
                details={"payload": body},
            )

        return CheckResult(
            "live.openrag",
            "pass",
            "OpenRAG search succeeded.",
            details={"count": body.get("count", 0)},
        )

    def probe_context_graph(self) -> CheckResult:
        patient_id = f"readiness-{uuid.uuid4().hex[:8]}"
        session_id = f"session-{uuid.uuid4().hex[:8]}"

        try:
            store_status, store_body = request_json(
                "POST",
                f"{self.context_graph_url.rstrip('/')}/memory/short-term/message",
                payload={
                    "session_id": session_id,
                    "patient_id": patient_id,
                    "role": "user",
                    "content": "Readiness probe message",
                },
                timeout=max(self.args.timeout, 60.0),
            )
            history_status, history_body = request_json(
                "GET",
                f"{self.context_graph_url.rstrip('/')}/memory/short-term/{session_id}",
                timeout=max(self.args.timeout, 60.0),
            )
        except Exception as exc:
            return CheckResult("live.context_graph", "fail", f"Context graph probe failed: {exc}")

        if store_status != 200 or history_status != 200:
            return CheckResult(
                "live.context_graph",
                "fail",
                "Context graph probe returned non-200 responses.",
                details={"store": store_body, "history": history_body},
            )

        message_count = len(history_body.get("messages", [])) if isinstance(history_body, dict) else 0
        return CheckResult(
            "live.context_graph",
            "pass",
            "Context graph short-term memory probe succeeded.",
            details={"messages": message_count},
        )

    def probe_context1(self) -> CheckResult:
        try:
            status_code, body = request_json(
                "POST",
                f"{self.context1_url.rstrip('/')}/query",
                payload={"question": "Summarize the follow-up plan for hypertension."},
                timeout=max(self.args.timeout, 90.0),
            )
        except Exception as exc:
            return CheckResult("live.context1", "fail", f"Context1 probe failed: {exc}")

        if status_code != 200 or not isinstance(body, dict):
            return CheckResult(
                "live.context1",
                "fail",
                f"Unexpected response: HTTP {status_code}",
                details={"payload": body},
            )

        return CheckResult(
            "live.context1",
            "pass",
            "Context1 query succeeded.",
            details={"keys": ", ".join(sorted(body.keys())[:6])},
        )

    def probe_platform(self) -> CheckResult:
        email = self.args.auth_email or f"readiness+{int(time.time())}@example.com"
        password = self.args.auth_password

        try:
            register_status, register_body = request_json(
                "POST",
                f"{self.platform_url}/api/v1/auth/register",
                payload={
                    "email": email,
                    "password": password,
                    "full_name": "Readiness Probe",
                    "role": "patient",
                },
                timeout=max(self.args.timeout, 60.0),
            )
        except Exception as exc:
            return CheckResult(
                "live.platform",
                "fail",
                f"Platform probe failed before register: {exc}",
            )
        if register_status not in {201, 409}:
            return CheckResult(
                "live.platform",
                "fail",
                f"Register failed with HTTP {register_status}.",
                details={"payload": register_body},
            )

        try:
            login_status, login_body = request_json(
                "POST",
                f"{self.platform_url}/api/v1/auth/login",
                payload={"email": email, "password": password},
                timeout=max(self.args.timeout, 60.0),
            )
        except Exception as exc:
            return CheckResult(
                "live.platform",
                "fail",
                f"Platform probe failed before login: {exc}",
            )
        if login_status != 200 or not isinstance(login_body, dict):
            return CheckResult(
                "live.platform",
                "fail",
                f"Login failed with HTTP {login_status}.",
                details={"payload": login_body},
            )

        access_token = login_body.get("access_token")
        if not access_token:
            return CheckResult(
                "live.platform",
                "fail",
                "Login succeeded but no access token was returned.",
                details={"payload": login_body},
            )

        auth_headers = {"Authorization": f"Bearer {access_token}"}
        try:
            me_status, me_body = request_json(
                "GET",
                f"{self.platform_url}/api/v1/auth/me",
                headers=auth_headers,
                timeout=max(self.args.timeout, 60.0),
            )
        except Exception as exc:
            return CheckResult(
                "live.platform",
                "fail",
                f"Platform probe failed before /auth/me: {exc}",
            )
        if me_status != 200 or not isinstance(me_body, dict):
            return CheckResult(
                "live.platform",
                "fail",
                f"/auth/me failed with HTTP {me_status}.",
                details={"payload": me_body},
            )

        patient_id = str(me_body.get("id") or uuid.uuid4())
        try:
            ask_status, ask_body = request_json(
                "POST",
                f"{self.platform_url}/api/v1/chat/ask",
                payload={
                    "query": "Give me a short greeting and tell me what you can do.",
                    "mode": "patient",
                    "patient_id": patient_id,
                    "web_search": env_bool(self.env.get("ENABLE_OPENRAG")),
                    "deep_reasoning": env_bool(self.env.get("ENABLE_CONTEXT1_AGENT")),
                },
                headers=auth_headers,
                timeout=max(self.args.timeout, 180.0),
            )
        except Exception as exc:
            return CheckResult(
                "live.platform",
                "fail",
                f"Platform probe failed before /chat/ask: {exc}",
            )
        if ask_status != 200 or not isinstance(ask_body, dict):
            return CheckResult(
                "live.platform",
                "fail",
                f"/chat/ask failed with HTTP {ask_status}.",
                details={"payload": ask_body},
            )

        routing = ask_body.get("routing", {})
        supplementary = ask_body.get("supplementary", {})
        return CheckResult(
            "live.platform",
            "pass",
            "Platform auth and routed chat probe succeeded.",
            details={
                "primary_engine": routing.get("primary_engine"),
                "secondary_engines": ", ".join(routing.get("secondary_engines", [])),
                "supplementary_keys": ", ".join(sorted(supplementary.keys())),
            },
        )


def main() -> int:
    args = parse_args()
    harness = ReadinessHarness(args)
    return harness.run()


if __name__ == "__main__":
    raise SystemExit(main())
