from __future__ import annotations

import json
import os
from urllib import request

BASE_URL = "http://127.0.0.1:8000/api/v1"
# Needed when the server runs with RAG_AUTH_ENABLED=true.
API_KEY = os.environ.get("RAG_API_KEY")


def call(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload else None
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    http_request = request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method="POST" if payload else "GET",
    )
    with request.urlopen(http_request) as response:
        return json.load(response)


def main() -> None:
    print("Health:")
    print(json.dumps(call("/health"), indent=2))

    document = call(
        "/documents/text",
        {
            "title": "Smoke Test Handbook",
            "source": "smoke-test",
            "text": (
                "Employees receive 24 paid leave days each year. "
                "Multi-factor authentication is mandatory for cloud accounts."
            ),
        },
    )
    print("\nDocument:")
    print(json.dumps(document, indent=2))

    answer = call(
        "/query",
        {"question": "How many paid leave days are available?", "top_k": 3},
    )
    print("\nAnswer:")
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
