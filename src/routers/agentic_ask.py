import logging

from fastapi import APIRouter, HTTPException

from src.dependencies import AgenticServiceDep
from src.schemas import AgenticAskResponse, AskRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agentic"])


@router.post("/agentic-ask", response_model=AgenticAskResponse)
async def agentic_ask(
    request: AskRequest,
    agentic_service: AgenticServiceDep,
) -> AgenticAskResponse:
    """Answer a question using the agentic RAG workflow (guardrail -> retrieve ->
    grade -> rewrite/generate) built on LangGraph."""
    try:
        result = await agentic_service.run(
            question=request.question,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            categories=request.categories,
            model=request.model,
        )
        return AgenticAskResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agentic ask failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
