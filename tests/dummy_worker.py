#!/usr/bin/env python3
"""Minimal worker used for integration testing."""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from typing import Optional

import redis


# Shared shutdown flag injected by the test harness at runtime. Using a mutable
# object allows the controller to signal cooperative termination across threads
# without relying on operating system signals.
SHUTDOWN_EVENT: Optional[threading.Event] = None


def _should_stop() -> bool:
    event = SHUTDOWN_EVENT
    return bool(event and event.is_set())


def main() -> int:
    parser = argparse.ArgumentParser(description="Dummy worker for tests")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    client = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)

    iterations = max(args.iterations, 0)
    delay = max(args.sleep_seconds, 0.0)

    for step in range(iterations):
        if _should_stop():
            break

        marker = f"worker:run:{uuid.uuid4().hex}"
        payload = json.dumps({"iteration": step, "marker": marker})
        client.set(marker, payload)
        client.set("worker:last_iteration", str(step))

        if delay:
            # Sleep in small increments to react promptly to shutdown events.
            remaining = delay
            while remaining > 0:
                if _should_stop():
                    break
                chunk = min(0.05, remaining)
                time.sleep(chunk)
                remaining -= chunk
            if _should_stop():
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
