#!/usr/bin/env python3
"""Test the RAG /ask endpoint."""
import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_URL = "http://localhost:8000/api/v1/ask"


def main(question: str, top_k: int, use_hybrid: bool, stream: bool) -> None:
    payload = {
        "question": question,
        "top_k": top_k,
        "use_hybrid": use_hybrid,
    }

    if stream:
        url = "http://localhost:8000/api/v1/stream"
        print(f"Streaming from {url}...")
        with httpx.stream("POST", url, json=payload, timeout=120.0) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "chunk" in data:
                        print(data["chunk"], end="", flush=True)
                    elif data.get("done"):
                        print("\n\n--- Done ---")
                        if data.get("answer"):
                            print(f"Final answer length: {len(data['answer'])} chars")
        return

    print(f"POST {DEFAULT_URL}")
    print(f"Question: {question}\n")

    response = httpx.post(DEFAULT_URL, json=payload, timeout=120.0)
    response.raise_for_status()
    result = response.json()

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result.get("answer", ""))
    print()
    print(f"Search mode: {result.get('search_mode')}")
    print(f"Chunks used: {result.get('chunks_used')}")
    print(f"Sources: {len(result.get('sources', []))}")
    for i, source in enumerate(result.get("sources", [])[:5], 1):
        print(f"  {i}. {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG ask endpoint")
    parser.add_argument("question", nargs="?", default="What is transformer attention?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    main(
        question=args.question,
        top_k=args.top_k,
        use_hybrid=not args.no_hybrid,
        stream=args.stream,
    )
