import json
import logging
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.dependencies import EmbeddingsServiceDep, LLMServiceDep, OpenSearchServiceDep
from src.schemas import AskRequest, AskResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])


async def _check_llm_ready(llm_client) -> None:
    """Verify the configured LLM provider is healthy before generation."""
    health = await llm_client.health_check()
    if health.get("status") != "healthy":
        raise HTTPException(
            status_code=503,
            detail=health.get("message", "LLM service is unavailable"),
        )


async def _retrieve_chunks(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
) -> tuple[List[Dict], List[str], str]:
    """Retrieve chunks and source URLs for RAG."""
    query_embedding = None
    if request.use_hybrid:
        try:
            query_embedding = await embeddings_service.embed_query(request.question)
            logger.info("Generated query embedding for hybrid search")
        except Exception as e:
            logger.warning("Failed to generate embeddings, falling back to BM25: %s", e)

    search_results = opensearch_client.search_unified(
        query=request.question,
        query_embedding=query_embedding,
        size=request.top_k,
        from_=0,
        categories=request.categories,
        use_hybrid=request.use_hybrid and query_embedding is not None,
        min_score=0.0,
    )

    chunks: List[Dict] = []
    sources_set: set[str] = set()

    for hit in search_results.get("hits", []):
        arxiv_id = hit.get("arxiv_id", "")
        chunks.append(
            {
                "arxiv_id": arxiv_id,
                "title": hit.get("title", ""),
                "chunk_text": hit.get("chunk_text") or hit.get("abstract", ""),
                "section_name": hit.get("section_name"),
            }
        )

        if arxiv_id:
            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

    search_mode = "hybrid" if (request.use_hybrid and query_embedding) else "bm25"
    return chunks, list(sources_set), search_mode


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    opensearch_client: OpenSearchServiceDep,
    embeddings_service: EmbeddingsServiceDep,
    llm_client: LLMServiceDep,
) -> AskResponse:
    """Answer a question using hybrid search + configured LLM."""
    try:
        if not opensearch_client.health_check():
            raise HTTPException(status_code=503, detail="Search service is currently unavailable")

        await _check_llm_ready(llm_client)

        chunks, sources, search_mode = await _retrieve_chunks(
            request, opensearch_client, embeddings_service
        )

        if not chunks:
            return AskResponse(
                query=request.question,
                answer="I couldn't find any relevant information in the papers to answer your question.",
                sources=[],
                chunks_used=0,
                search_mode=search_mode,
            )

        rag_response = await llm_client.generate_rag_answer(
            query=request.question,
            chunks=chunks,
            model=request.model,
        )

        return AskResponse(
            query=request.question,
            answer=rag_response.get("answer", "Unable to generate answer"),
            sources=sources or rag_response.get("sources", []),
            chunks_used=len(chunks),
            search_mode=search_mode,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing ask request: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/ask/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchServiceDep,
    embeddings_service: EmbeddingsServiceDep,
    llm_client: LLMServiceDep,
) -> StreamingResponse:
    """Stream an answer using hybrid search + configured LLM."""

    async def generate_stream():
        try:
            if not opensearch_client.health_check():
                yield f"data: {json.dumps({'error': 'Search service unavailable'})}\n\n"
                return

            await _check_llm_ready(llm_client)

            chunks, sources, search_mode = await _retrieve_chunks(
                request, opensearch_client, embeddings_service
            )

            if not chunks:
                yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'done': True})}\n\n"
                return

            yield f"data: {json.dumps({'sources': sources, 'chunks_used': len(chunks), 'search_mode': search_mode})}\n\n"

            full_response = ""
            async for text_chunk in llm_client.stream_rag_text(
                query=request.question,
                chunks=chunks,
                model=request.model,
            ):
                full_response += text_chunk
                yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

            yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"

        except HTTPException as e:
            yield f"data: {json.dumps({'error': e.detail})}\n\n"
        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@stream_router.post("/stream")
async def ask_question_stream_alias(
    request: AskRequest,
    opensearch_client: OpenSearchServiceDep,
    embeddings_service: EmbeddingsServiceDep,
    llm_client: LLMServiceDep,
) -> StreamingResponse:
    """Alias for /ask/stream (Gradio UI compatibility)."""
    return await ask_question_stream(
        request=request,
        opensearch_client=opensearch_client,
        embeddings_service=embeddings_service,
        llm_client=llm_client,
    )
