import json
import os
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = os.environ.get("RAG_BASE_URL", "http://localhost:8000")
ENDPOINT = f"{BASE_URL}/api/v1/rag/query"
ROUNDS = 2
TIMEOUT_SECONDS = 180

QUERIES = [
    "What is PostgreSQL and what is it used for?",
    "How do PostgreSQL indexes improve query performance?",
    "How do execution plans help optimize PostgreSQL queries?",
    "Why are statistics important to PostgreSQL query planning?",
    "How does PostgreSQL handle concurrency?",
    "What is Redis used for in caching?",
    "How does Redis key expiration work?",
    "What Redis data structures are available?",
]


def percentile(values, percentile_value):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)
    fraction = position - lower
    return round(
        ordered[lower] * (1 - fraction) + ordered[upper] * fraction,
        2,
    )


def summarize(values):
    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 2) if values else None,
        "median_ms": round(statistics.median(values), 2) if values else None,
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": round(min(values), 2) if values else None,
        "max_ms": round(max(values), 2) if values else None,
    }


def main():
    session = requests.Session()

    for endpoint in ("/health", "/ready"):
        response = session.get(BASE_URL + endpoint, timeout=10)
        response.raise_for_status()
        print(f"{endpoint}: {response.status_code}")

    records = []

    # One warm-up request is excluded from measurements.
    warmup = session.post(
        ENDPOINT,
        json={"query": QUERIES[0]},
        timeout=TIMEOUT_SECONDS,
    )
    warmup.raise_for_status()
    print("Warm-up completed.")

    for round_number in range(1, ROUNDS + 1):
        for index, query in enumerate(QUERIES, start=1):
            started = time.perf_counter()
            status_code = None
            error = None
            payload = None

            try:
                response = session.post(
                    ENDPOINT,
                    json={"query": query},
                    timeout=TIMEOUT_SECONDS,
                )
                status_code = response.status_code
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                error = str(exc)

            client_ms = (time.perf_counter() - started) * 1000
            diagnostics = payload.get("diagnostics", {}) if payload else {}
            stages = diagnostics.get("stages", []) if diagnostics else []

            record = {
                "round": round_number,
                "query_index": index,
                "query": query,
                "status_code": status_code,
                "success": error is None and payload is not None,
                "error": error,
                "client_latency_ms": round(client_ms, 2),
                "server_total_duration_ms": diagnostics.get(
                    "total_duration_ms"
                ),
                "grounded": payload.get("grounded") if payload else None,
                "retry_count": diagnostics.get("retry_count"),
                "candidate_count": diagnostics.get("candidate_count"),
                "relevant_context_count": diagnostics.get(
                    "relevant_context_count"
                ),
                "stage_timings": stages,
            }
            records.append(record)

            print(
                f"Round {round_number}/{ROUNDS} | "
                f"Query {index}/{len(QUERIES)} | "
                f"HTTP {status_code} | "
                f"{client_ms:.0f} ms | "
                f"grounded={record['grounded']} | "
                f"error={error or '-'}"
            )

    successful = [r for r in records if r["success"]]
    client_latencies = [r["client_latency_ms"] for r in successful]
    server_latencies = [
        r["server_total_duration_ms"]
        for r in successful
        if isinstance(r["server_total_duration_ms"], (int, float))
    ]

    total_elapsed_seconds = sum(client_latencies) / 1000
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "rounds": ROUNDS,
        "unique_queries": len(QUERIES),
        "warmup_requests": 1,
        "measured_requests": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "grounded_responses": sum(
            r["grounded"] is True for r in successful
        ),
        "ungrounded_responses": sum(
            r["grounded"] is False for r in successful
        ),
        "sequential_client_latency": summarize(client_latencies),
        "server_reported_latency": summarize(server_latencies),
        "approx_sequential_throughput_qps": (
            round(len(successful) / total_elapsed_seconds, 3)
            if total_elapsed_seconds > 0
            else None
        ),
        "records": records,
        "limitations": [
            "Small benchmark with eight unique queries.",
            "Sequential requests only; no concurrent load test.",
            "Client latency includes HTTP overhead and response transfer.",
            "Throughput is approximate and excludes warm-up time.",
        ],
    }

    output = Path(os.environ.get("RAG_OUTPUT", "benchmarks/results/api_latest.json"))
    output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== END-TO-END API BASELINE ===")
    print(json.dumps({
        key: value
        for key, value in summary.items()
        if key not in ("records", "limitations")
    }, indent=2))
    print(f"\nFull results saved to: {output}")


if __name__ == "__main__":
    main()
