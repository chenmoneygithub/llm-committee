"""One bounded request pool over rolling questions and dependency-ready nodes."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .failures import LocalTaskFailure


def run_ready_pool(
    graphs, *, question_limit, request_limit, token_count, execute, stop, on_start, on_done, on_error, on_progress
):
    if not 1 <= question_limit <= 8 or not 1 <= request_limit <= 64:
        raise ValueError("Expected 1–8 active questions and 1–64 in-flight requests")
    pending = iter(graphs)
    active, futures = {}, {}
    peak_requests = peak_questions = 0

    def fill_questions():
        nonlocal peak_questions
        while not stop.is_set() and len(active) < question_limit:
            graph = next(pending, None)
            if graph is None:
                break
            active[graph.question.id] = graph
            peak_questions = max(peak_questions, len(active))
            on_start(graph)

    def fail(graph, exc):
        stop.set()
        on_error(graph, exc)

    with ThreadPoolExecutor(max_workers=request_limit, thread_name_prefix="committee-request") as pool:
        fill_questions()
        while active or futures:
            # Round-robin admission: no question gets a private multiplicative request pool.
            while not stop.is_set() and len(futures) < request_limit:
                admitted = False
                for graph in list(active.values()):
                    if stop.is_set() or len(futures) >= request_limit:
                        break
                    task = graph.ready()
                    if task is None:
                        continue
                    try:
                        request = task.build(graph.values, token_count)
                        graph.submitted.add(task.key)
                        future = pool.submit(execute, request, task.parse)
                        futures[future] = (graph, task, request)
                        peak_requests = max(peak_requests, len(futures))
                        admitted = True
                    except Exception as exc:
                        fail(graph, exc)
                        break
                if not admitted:
                    break
            if not futures:
                if active and not stop.is_set():
                    raise RuntimeError("Unfinished question graph has no runnable nodes")
                break
            done, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                graph, task, request = futures.pop(future)
                try:
                    try:
                        value = future.result()
                    except LocalTaskFailure as exc:
                        graph.reject(task, exc)
                        on_error(graph, exc)
                    else:
                        graph.accept(task, request, value)
                    if graph.complete:
                        on_done(graph)
                        del active[graph.question.id]
                except Exception as exc:
                    fail(graph, exc)
            # Fill freed QUESTION slots immediately, not after the original batch finishes.
            fill_questions()
            on_progress(
                {
                    "active_question_ids": list(active),
                    "in_flight_requests": len(futures),
                    "peak_in_flight_requests": peak_requests,
                    "peak_active_questions": peak_questions,
                }
            )
    return {"peak_in_flight_requests": peak_requests, "peak_active_questions": peak_questions}
