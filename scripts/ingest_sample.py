from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import request

API_URL = "http://127.0.0.1:8000/api/v1/documents/text"
# Needed when the server runs with RAG_AUTH_ENABLED=true.
API_KEY = os.environ.get("RAG_API_KEY")


def main() -> None:
    sample = Path("sample_data/company_handbook.md")
    payload = json.dumps(
        {
            "title": "Acme Engineering Handbook",
            "source": str(sample),
            "text": sample.read_text(encoding="utf-8"),
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    http_request = request.Request(API_URL, data=payload, headers=headers, method="POST")
    with request.urlopen(http_request) as response:
        print(json.dumps(json.load(response), indent=2))


if __name__ == "__main__":
    main()
