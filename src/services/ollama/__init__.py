import httpx
from src.config import Settings


class OllamaClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_host
        self.timeout = settings.ollama_timeout

    async def health_check(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/version")
                if response.status_code == 200:
                    return {"status": "healthy", "message": "Ollama is running"}
                return {"status": "unhealthy", "message": f"Status: {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}

