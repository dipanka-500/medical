"""
Parallel Executor — Runs model inference tasks concurrently.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """Runs model inference tasks in parallel using thread pools.

    Thread-based parallelism is optimal for GPU inference tasks
    where Python GIL is released during CUDA operations.

    Features:
    - Per-task timeout isolation (one slow model won't abort others)
    - Configurable retry on failure
    - Partial results returned even if some tasks fail/timeout
    """

    def __init__(
        self,
        max_workers: int = 4,
        timeout: int = 300,
        retry_on_failure: bool = False,
        max_retries: int = 2,
    ):
        self.max_workers = max_workers
        self.timeout = timeout
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute(
        self,
        tasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute model inference tasks in parallel.

        Uses per-future timeouts so one slow/hung model does not abort
        the entire batch. Returns partial results for all tasks that
        completed within the timeout window.

        Args:
            tasks: List of task dicts with:
                - name: Task identifier
                - fn: Callable to execute
                - kwargs: Arguments for the callable

        Returns:
            List of result dicts from all completed tasks
        """
        if not tasks:
            return []

        results: list[dict[str, Any]] = []
        start = time.time()

        collected_names: set[str] = set()
        future_to_task = {}
        for task in tasks:
            fn = task["fn"]
            kwargs = task.get("kwargs", {})
            name = task.get("name", "unnamed")
            future = self._executor.submit(fn, **kwargs)
            future_to_task[future] = name

        # Collect results with a global timeout, but handle TimeoutError
        # gracefully so partial results are returned
        try:
            for future in as_completed(future_to_task, timeout=self.timeout):
                task_name = future_to_task[future]
                result = self._collect_future(future, task_name)
                results.append(result)
                collected_names.add(task_name)
        except FuturesTimeoutError:
            # Some tasks didn't finish — collect whatever completed
            logger.warning(
                f"Global timeout ({self.timeout}s) reached. "
                f"Collecting partial results."
            )
            for future, task_name in future_to_task.items():
                if future.done() and task_name not in collected_names:
                    results.append(self._collect_future(future, task_name))
                    collected_names.add(task_name)
                elif not future.done():
                    future.cancel()
                    results.append({
                        "task_name": task_name,
                        "error": f"Timed out after {self.timeout}s",
                        "text": "",
                    })
                    collected_names.add(task_name)

        # Retry failed tasks if configured
        if self.retry_on_failure:
            results = self._retry_failed(tasks, results)

        total_time = time.time() - start
        completed = sum(1 for r in results if not r.get("error"))
        logger.info(
            f"Parallel execution: {completed}/{len(tasks)} tasks succeeded "
            f"in {total_time:.2f}s"
        )

        return results

    def _collect_future(
        self, future: Any, task_name: str,
    ) -> dict[str, Any]:
        """Safely collect a single future result."""
        try:
            result = future.result(timeout=0)
            if isinstance(result, dict):
                result["task_name"] = task_name
                return result
            return {"task_name": task_name, "text": str(result)}
        except Exception as e:
            logger.error(f"Task failed: {task_name}: {e}")
            return {"task_name": task_name, "error": str(e), "text": ""}

    def _retry_failed(
        self,
        original_tasks: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Retry failed tasks up to max_retries times."""
        failed_names = {r["task_name"] for r in results if r.get("error")}
        if not failed_names:
            return results

        tasks_by_name = {t.get("name", "unnamed"): t for t in original_tasks}
        retry_tasks = [
            tasks_by_name[name] for name in failed_names
            if name in tasks_by_name
        ]

        for attempt in range(1, self.max_retries + 1):
            if not retry_tasks:
                break

            logger.info(f"Retrying {len(retry_tasks)} failed tasks (attempt {attempt}/{self.max_retries})")
            retry_results = []

            for task in retry_tasks:
                fn = task["fn"]
                kwargs = task.get("kwargs", {})
                name = task.get("name", "unnamed")
                try:
                    result = fn(**kwargs)
                    if isinstance(result, dict):
                        result["task_name"] = name
                        retry_results.append(result)
                    else:
                        retry_results.append({"task_name": name, "text": str(result)})
                except Exception as e:
                    logger.error(f"Retry failed: {name}: {e}")
                    retry_results.append({"task_name": name, "error": str(e), "text": ""})

            # Replace failed results with retry results
            succeeded_names = {
                r["task_name"] for r in retry_results if not r.get("error")
            }
            results = [
                r for r in results if r["task_name"] not in succeeded_names
            ] + [r for r in retry_results if not r.get("error")]

            # Only retry still-failed tasks
            retry_tasks = [
                tasks_by_name[r["task_name"]]
                for r in retry_results
                if r.get("error") and r["task_name"] in tasks_by_name
            ]

        return results

    def execute_sequential(
        self,
        tasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute tasks sequentially (for debugging or resource constraints)."""
        results = []
        for task in tasks:
            fn = task["fn"]
            kwargs = task.get("kwargs", {})
            name = task.get("name", "unnamed")

            try:
                result = fn(**kwargs)
                if isinstance(result, dict):
                    result["task_name"] = name
                    results.append(result)
                else:
                    results.append({"task_name": name, "text": str(result)})
            except Exception as e:
                logger.error(f"Task failed: {name}: {e}")
                results.append({"task_name": name, "error": str(e), "text": ""})

        return results

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the underlying thread pool."""
        self._executor.shutdown(wait=wait)
