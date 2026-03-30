"""Tests for parallel execution and timeout handling."""

from __future__ import annotations

import time

import pytest

from core.execution.parallel_executor import ParallelExecutor


def _slow_task(**kwargs):
    time.sleep(kwargs.get("delay", 5))
    return {"text": "slow result"}


def _fast_task(**kwargs):
    return {"text": f"fast result: {kwargs.get('value', '')}"}


def _failing_task(**kwargs):
    raise RuntimeError("intentional failure")


class TestParallelExecutor:
    """Tests for ParallelExecutor."""

    def test_empty_tasks(self):
        executor = ParallelExecutor()
        assert executor.execute([]) == []

    def test_single_task(self):
        executor = ParallelExecutor()
        results = executor.execute([
            {"name": "task1", "fn": _fast_task, "kwargs": {"value": "hello"}},
        ])
        assert len(results) == 1
        assert results[0]["text"] == "fast result: hello"
        assert results[0]["task_name"] == "task1"

    def test_multiple_tasks(self):
        executor = ParallelExecutor(max_workers=2)
        tasks = [
            {"name": f"task_{i}", "fn": _fast_task, "kwargs": {"value": str(i)}}
            for i in range(4)
        ]
        results = executor.execute(tasks)
        assert len(results) == 4

    def test_failed_task_returns_error(self):
        executor = ParallelExecutor()
        results = executor.execute([
            {"name": "good", "fn": _fast_task, "kwargs": {}},
            {"name": "bad", "fn": _failing_task, "kwargs": {}},
        ])
        assert len(results) == 2
        errors = [r for r in results if r.get("error")]
        assert len(errors) == 1
        assert errors[0]["task_name"] == "bad"

    def test_timeout_returns_partial_results(self):
        executor = ParallelExecutor(max_workers=2, timeout=1)
        results = executor.execute([
            {"name": "fast", "fn": _fast_task, "kwargs": {}},
            {"name": "slow", "fn": _slow_task, "kwargs": {"delay": 10}},
        ])
        # fast task should succeed, slow task should timeout
        assert any(r["task_name"] == "fast" and not r.get("error") for r in results)
        slow_results = [r for r in results if r["task_name"] == "slow"]
        if slow_results:
            assert slow_results[0].get("error")

    def test_retry_on_failure(self):
        call_count = {"n": 0}

        def _fail_then_succeed(**kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                raise RuntimeError("first call fails")
            return {"text": "succeeded on retry"}

        executor = ParallelExecutor(retry_on_failure=True, max_retries=2)
        results = executor.execute([
            {"name": "flaky", "fn": _fail_then_succeed, "kwargs": {}},
        ])
        succeeded = [r for r in results if not r.get("error")]
        assert len(succeeded) == 1

    def test_sequential_execution(self):
        executor = ParallelExecutor()
        tasks = [
            {"name": f"task_{i}", "fn": _fast_task, "kwargs": {"value": str(i)}}
            for i in range(3)
        ]
        results = executor.execute_sequential(tasks)
        assert len(results) == 3
