"""
MediScan AI v7.0 — Parallel Executor & Circuit Breaker
Handles concurrent model inference with fault tolerance.
"""
from __future__ import annotations


import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing — reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for GPU/model fault tolerance."""

    failure_threshold: int = 3
    recovery_timeout: float = 60.0  # seconds
    _failure_count: int = field(default=0, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _last_failure_time: float = field(default=0.0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker → HALF_OPEN (attempting recovery)")
        return self._state

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self, error: Exception) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        logger.error(f"Circuit breaker failure #{self._failure_count}: {error}")

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker → OPEN (blocking calls)")

    def can_execute(self) -> bool:
        state = self.state
        return state != CircuitState.OPEN


class ParallelExecutor:
    """Runs multiple models in parallel with circuit breakers per model."""

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute_parallel(
        self,
        tasks: list[dict[str, Any]],
        timeout: float = 300.0,
    ) -> list[dict[str, Any]]:
        """Execute multiple model inference tasks in parallel.

        Args:
            tasks: List of dicts with:
                - model_key: str
                - callable: Callable that returns dict
                - args: tuple
                - kwargs: dict
            timeout: Max seconds to wait for all tasks

        Returns:
            List of result dicts with model_key, result/error, timing
        """
        results = []
        futures = {}

        for task in tasks:
            model_key = task["model_key"]

            # Check circuit breaker
            if model_key not in self.circuit_breakers:
                self.circuit_breakers[model_key] = CircuitBreaker()

            cb = self.circuit_breakers[model_key]
            if not cb.can_execute():
                results.append({
                    "model_key": model_key,
                    "status": "circuit_open",
                    "error": "Circuit breaker is OPEN — model temporarily disabled",
                    "duration": 0.0,
                })
                continue

            # Submit to thread pool
            future = self._executor.submit(
                self._run_task, model_key, task["callable"],
                task.get("args", ()), task.get("kwargs", {}),
            )
            futures[future] = model_key

        # Collect results. If one or more tasks time out, return partial
        # results instead of failing the entire analysis.
        completed_keys: set[str] = set()
        try:
            for future in as_completed(futures, timeout=timeout):
                model_key = futures[future]
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                    completed_keys.add(model_key)
                    if result["status"] == "success":
                        self.circuit_breakers[model_key].record_success()
                    else:
                        self.circuit_breakers[model_key].record_failure(
                            Exception(result.get("error", "Unknown error"))
                        )
                except Exception as e:
                    completed_keys.add(model_key)
                    self.circuit_breakers[model_key].record_failure(e)
                    results.append({
                        "model_key": model_key,
                        "status": "error",
                        "error": str(e),
                        "duration": 0.0,
                    })
        except FuturesTimeoutError:
            logger.warning(
                "Parallel execution timed out after %.1fs; returning partial results",
                timeout,
            )
            for future, model_key in futures.items():
                if model_key in completed_keys:
                    continue
                if future.done():
                    try:
                        result = future.result(timeout=0)
                        results.append(result)
                        if result["status"] == "success":
                            self.circuit_breakers[model_key].record_success()
                        else:
                            self.circuit_breakers[model_key].record_failure(
                                Exception(result.get("error", "Unknown error"))
                            )
                    except Exception as e:
                        self.circuit_breakers[model_key].record_failure(e)
                        results.append({
                            "model_key": model_key,
                            "status": "error",
                            "error": str(e),
                            "duration": 0.0,
                        })
                else:
                    future.cancel()
                    self.circuit_breakers[model_key].record_failure(
                        TimeoutError(f"Timed out after {timeout}s")
                    )
                    results.append({
                        "model_key": model_key,
                        "status": "timeout",
                        "error": f"Timed out after {timeout}s",
                        "duration": timeout,
                    })

        return results

    def execute_sequential(
        self,
        tasks: list[dict[str, Any]],
        timeout: float = 300.0,
    ) -> list[dict[str, Any]]:
        """Execute tasks one at a time using the same circuit-breaker logic."""
        results = []

        for task in tasks:
            model_key = task["model_key"]

            if model_key not in self.circuit_breakers:
                self.circuit_breakers[model_key] = CircuitBreaker()

            cb = self.circuit_breakers[model_key]
            if not cb.can_execute():
                results.append({
                    "model_key": model_key,
                    "status": "circuit_open",
                    "error": "Circuit breaker is OPEN — model temporarily disabled",
                    "duration": 0.0,
                })
                continue

            try:
                result = self._run_task(
                    model_key,
                    task["callable"],
                    task.get("args", ()),
                    task.get("kwargs", {}),
                )
                results.append(result)
                if result["status"] == "success":
                    cb.record_success()
                else:
                    cb.record_failure(Exception(result.get("error", "Unknown error")))
            except Exception as exc:
                cb.record_failure(exc)
                results.append({
                    "model_key": model_key,
                    "status": "error",
                    "error": str(exc),
                    "duration": 0.0,
                })

        return results

    async def execute_async(
        self,
        tasks: list[dict[str, Any]],
        timeout: float = 300.0,
    ) -> list[dict[str, Any]]:
        """Async version of parallel execution."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.execute_parallel, tasks, timeout
        )

    def _run_task(
        self,
        model_key: str,
        callable_fn: Callable,
        args: tuple,
        kwargs: dict,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Run a single model inference task with retry, backoff, and error handling.

        v7.0: Automatic retry with exponential backoff (1s, 2s) for transient
        failures. OOM errors skip retry (no point retrying without more memory).
        """
        last_error = None

        for attempt in range(1 + max_retries):
            start = time.time()
            try:
                result = callable_fn(*args, **kwargs)
                duration = time.time() - start

                if attempt > 0:
                    logger.info(f"✓ {model_key} succeeded on retry {attempt} in {duration:.2f}s")
                else:
                    logger.info(f"✓ {model_key} completed in {duration:.2f}s")

                return {
                    "model_key": model_key,
                    "status": "success",
                    "result": result,
                    "duration": duration,
                    "retries": attempt,
                }
            except torch.cuda.OutOfMemoryError as e:
                duration = time.time() - start
                logger.error(f"✗ {model_key} OOM after {duration:.2f}s (no retry): {e}")
                torch.cuda.empty_cache()
                return {
                    "model_key": model_key,
                    "status": "oom",
                    "error": f"GPU OOM: {e}",
                    "duration": duration,
                }
            except Exception as e:
                duration = time.time() - start
                last_error = e

                if attempt < max_retries:
                    backoff = 2 ** attempt  # 1s, 2s
                    logger.warning(
                        f"⟳ {model_key} attempt {attempt + 1} failed in {duration:.2f}s, "
                        f"retrying in {backoff}s: {e}"
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"✗ {model_key} failed after {1 + max_retries} attempts in {duration:.2f}s: {e}"
                    )

        return {
            "model_key": model_key,
            "status": "error",
            "error": str(last_error),
            "duration": time.time() - start,
            "retries": max_retries,
        }

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)


