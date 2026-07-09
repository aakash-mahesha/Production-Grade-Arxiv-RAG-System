"""RAGAS evaluation for the arXiv RAG pipeline (Week 6).

For every question in ``scripts/eval_dataset.json`` this script:

  1. Runs the *live* RAG pipeline in-process (embed -> hybrid search -> LLM answer).
  2. Collects (question, answer, retrieved_contexts, ground_truth) into a
     RAGAS ``SingleTurnSample``.
  3. Scores the whole set with four RAGAS metrics:
        - faithfulness      : is the answer grounded in the retrieved chunks?
        - answer_relevancy  : does the answer address the question?
        - context_precision : are the relevant chunks ranked highly?
        - context_recall    : did retrieval fetch what the reference needs?

Judge LLM runs through OpenRouter (OpenAI-compatible), so you can compare
different judges by changing RAGAS_JUDGE_MODEL or passing --judge-model.
Embeddings reuse the Jina API key because OpenRouter has no embeddings endpoint.

Usage:
    uv run python scripts/eval_ragas.py
    uv run python scripts/eval_ragas.py --judge-model anthropic/claude-sonnet-5
    uv run python scripts/eval_ragas.py --top-k 3 --limit 2
"""

import argparse
import asyncio
import json
import sys
import warnings
from pathlib import Path

# Allow "python scripts/eval_ragas.py" by putting the project root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The classic metric instances still work in ragas 0.4 but warn; silence the noise.
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.embeddings import JinaEmbeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import Settings, get_settings  # noqa: E402
from src.services.embeddings.factory import make_embeddings_client  # noqa: E402
from src.services.llm.factory import make_llm_client  # noqa: E402
from src.services.opensearch.factory import make_opensearch_client  # noqa: E402

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_PATH = Path(__file__).parent / "ragas_results.json"

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# RAGAS/LangChain otherwise requests up to 65535 output tokens per judge call,
# which is wasteful and triggers 402s on limited-credit accounts. Metric outputs
# are small, so cap it.
JUDGE_MAX_TOKENS = 2048


async def run_pipeline(question, top_k, opensearch_client, embeddings_client, llm_client):
    """Run one question through embed -> hybrid search -> generate.

    Returns (answer_text, list_of_context_strings) which map onto RAGAS's
    ``response`` and ``retrieved_contexts`` fields.
    """
    query_embedding = None
    try:
        query_embedding = await embeddings_client.embed_query(question)
    except Exception as e:
        print(f"    [warn] embedding failed, falling back to BM25: {e}")

    results = opensearch_client.search_unified(
        query=question,
        query_embedding=query_embedding,
        size=top_k,
        from_=0,
        categories=None,
        use_hybrid=query_embedding is not None,
        min_score=0.0,
    )

    chunks = []
    for hit in results.get("hits", []):
        text = hit.get("chunk_text") or hit.get("abstract", "")
        if text:
            chunks.append(
                {
                    "arxiv_id": hit.get("arxiv_id", ""),
                    "title": hit.get("title", ""),
                    "chunk_text": text,
                }
            )

    if not chunks:
        return "No relevant information found in the indexed papers.", []

    rag = await llm_client.generate_rag_answer(query=question, chunks=chunks, model=None)
    contexts = [c["chunk_text"] for c in chunks]
    return rag.get("answer", ""), contexts


async def build_samples(dataset, top_k, settings: Settings):
    """Run every question through the pipeline and collect RAGAS samples."""
    opensearch_client = make_opensearch_client()
    embeddings_client = make_embeddings_client()
    llm_client = make_llm_client(settings)

    health = await llm_client.health_check()
    if health.get("status") != "healthy":
        raise SystemExit(f"LLM provider not healthy: {health.get('message')}")
    print(f"Generation LLM: {settings.llm_provider} ({getattr(llm_client, 'model', '?')})\n")

    samples = []
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i}/{len(dataset)}] {question}")
        answer, contexts = await run_pipeline(
            question, top_k, opensearch_client, embeddings_client, llm_client
        )
        print(f"    retrieved {len(contexts)} chunks, generated {len(answer)} chars")
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )
    return samples


def make_judge(settings: Settings, model_override: str | None):
    """Build the RAGAS judge LLM, routed through OpenRouter."""
    if not settings.openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (needed for the RAGAS judge LLM).")

    model = model_override or settings.ragas_judge_model
    print(f"Judge LLM: {model} (via OpenRouter)")
    chat = ChatOpenAI(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
        max_tokens=JUDGE_MAX_TOKENS,
        default_headers={
            "HTTP-Referer": settings.openrouter_app_url,
            "X-Title": settings.openrouter_app_name,
        },
    )
    return LangchainLLMWrapper(chat)


def make_ragas_embeddings(settings: Settings):
    """Build the embeddings model RAGAS uses for answer_relevancy (Jina)."""
    if not settings.jina_api_key:
        raise SystemExit("JINA_API_KEY is not set (needed for the answer_relevancy metric).")

    print(f"Judge embeddings: {settings.ragas_embedding_model} (via Jina)\n")
    emb = JinaEmbeddings(
        jina_api_key=settings.jina_api_key,
        model_name=settings.ragas_embedding_model,
    )
    return LangchainEmbeddingsWrapper(emb)


def print_summary(result):
    """Print aggregate + per-question scores and persist them to JSON."""
    df = result.to_pandas()
    metric_cols = [m.name for m in METRICS]

    print("\n" + "=" * 60)
    print("RAGAS RESULTS (mean across questions)")
    print("=" * 60)
    summary = {}
    for col in metric_cols:
        if col in df.columns:
            value = float(df[col].mean())
            summary[col] = round(value, 4)
            print(f"  {col:<20} {value:.4f}")
    print("=" * 60)

    RESULTS_PATH.write_text(
        json.dumps(
            {"summary": summary, "per_question": json.loads(df.to_json(orient="records"))},
            indent=2,
        )
    )
    print(f"\nSaved detailed results to {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG pipeline")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per question")
    parser.add_argument("--judge-model", default=None, help="Override judge model (OpenRouter id)")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N questions")
    args = parser.parse_args()

    settings = get_settings()

    dataset = json.loads(DATASET_PATH.read_text())
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"Loaded {len(dataset)} evaluation questions from {DATASET_PATH.name}\n")

    # Step 1 + 2: run the pipeline and collect samples (async).
    samples = asyncio.run(build_samples(dataset, args.top_k, settings))

    # Step 3: score with RAGAS (sync; manages its own event loop internally).
    judge = make_judge(settings, args.judge_model)
    embeddings = make_ragas_embeddings(settings)

    print("Scoring with RAGAS (this makes several judge-LLM calls per question)...\n")
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=METRICS,
        llm=judge,
        embeddings=embeddings,
    )

    print_summary(result)


if __name__ == "__main__":
    main()
