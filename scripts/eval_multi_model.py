"""Compare RAGAS scores across multiple judge models (Week 6).

This harness answers the question: "does my RAG quality score depend on which
judge LLM I use?" It runs the RAG pipeline ONCE to produce answers, then scores
that same set of answers with each judge model in turn and prints a side-by-side
table. Judges run through OpenRouter, so any model id works.

Reuses the building blocks from scripts/eval_ragas.py.

Usage:
    uv run python scripts/eval_multi_model.py
    uv run python scripts/eval_multi_model.py --judge-models openai/gpt-4o-mini,google/gemini-2.0-flash-001
    uv run python scripts/eval_multi_model.py --limit 2
"""

import argparse
import asyncio
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root (for src)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts dir (for eval_ragas)

warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas import EvaluationDataset, evaluate  # noqa: E402

from eval_ragas import (  # noqa: E402
    DATASET_PATH,
    METRICS,
    build_samples,
    make_judge,
    make_ragas_embeddings,
)

from src.config import get_settings  # noqa: E402

RESULTS_PATH = Path(__file__).parent / "ragas_multi_model_results.json"

DEFAULT_JUDGES = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-5",
]


def score_with_judge(samples, settings, model, embeddings):
    """Score the (already generated) samples with one judge model."""
    judge = make_judge(settings, model)
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=METRICS,
        llm=judge,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    return {m.name: round(float(df[m.name].mean()), 4) for m in METRICS if m.name in df.columns}


def print_table(results, metric_names):
    col = 20
    header = f"{'judge model':<34}" + "".join(f"{m:<{col}}" for m in metric_names)
    line = "=" * len(header)
    print("\n" + line)
    print(header)
    print(line)
    for model, scores in results.items():
        if scores is None:
            print(f"{model:<34}(failed / skipped)")
        else:
            print(f"{model:<34}" + "".join(f"{scores.get(m, float('nan')):<{col}.4f}" for m in metric_names))
    print(line)


def main():
    parser = argparse.ArgumentParser(description="Compare RAGAS scores across judge models")
    parser.add_argument(
        "--judge-models",
        default=",".join(DEFAULT_JUDGES),
        help="Comma-separated OpenRouter model ids",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N questions")
    args = parser.parse_args()

    models = [m.strip() for m in args.judge_models.split(",") if m.strip()]
    settings = get_settings()

    dataset = json.loads(DATASET_PATH.read_text())
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"Evaluating {len(dataset)} questions across {len(models)} judges: {models}\n")

    # Generate answers ONCE, then reuse for every judge (isolates the judge variable).
    samples = asyncio.run(build_samples(dataset, args.top_k, settings))
    embeddings = make_ragas_embeddings(settings)

    metric_names = [m.name for m in METRICS]
    results = {}
    for model in models:
        print(f"\n--- Scoring with judge: {model} ---")
        try:
            results[model] = score_with_judge(samples, settings, model, embeddings)
        except Exception as e:
            print(f"    [error] judge '{model}' failed: {e}")
            results[model] = None

    print_table(results, metric_names)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved comparison to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
