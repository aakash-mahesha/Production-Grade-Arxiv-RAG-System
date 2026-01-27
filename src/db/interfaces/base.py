from abc import ABC, abstractmethod
from typing import Any, ContextManager, Dict, List, Optional

from sqlalchemy.orm import Session


class BaseDatabase(ABC):
    @abstractmethod
    def startup(self) -> None:
        pass

    @abstractmethod
    def teardown(self) -> None:
        pass

    @abstractmethod
    def get_session(self) -> ContextManager[Session]:
        pass

class BaseRepository(ABC):
    def __init__(self, session: Session):
        self.session = session
    
    @abstractmethod
    def create(self, data: Dict[str, Any]) -> Any:
        pass
    @abstractmethod
    def get_by_id(self, record_id: Any) -> Optional[Any]:
        """Get a record by ID."""

    @abstractmethod
    def update(self, record_id: Any, data: Dict[str, Any]) -> Optional[Any]:
        """Update a record by ID."""

    @abstractmethod
    def delete(self, record_id: Any) -> bool:
        """Delete a record by ID."""

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """List records with pagination."""