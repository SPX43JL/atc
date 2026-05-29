#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--interval", type=float, default=5)
    args = p.parse_args()

    deadline = time.time() + args.timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(args.url, timeout=10) as resp:
                if 200 <= resp.status < 500:
                    print(f"ready {args.url} status={resp.status}")
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = repr(exc)
        time.sleep(args.interval)
    raise SystemExit(f"timeout waiting for {args.url}; last={last}")


if __name__ == "__main__":
    main()
