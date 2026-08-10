"""DocumentStore abstract base class."""

from abc import ABC, abstractmethod
from typing import List, Optional

from mind_graph_db.core.types import Document


class DocumentStore(ABC):
    """Abstract interface for storing, retrieving, and managing raw text documents."""

    @abstractmethod
    def put(self, document: Document) -> str:
        """Store or update a document.

        Args:
            document: Document instance to store.

        Returns:
            The document ID.
        """
        pass

    @abstractmethod
    def get(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID.

        Args:
            document_id: Unique document identifier.

        Returns:
            The Document if found, otherwise None.
        """
        pass

    @abstractmethod
    def update(self, document: Document) -> bool:
        """Update an existing document.

        Args:
            document: Document instance with updated fields.

        Returns:
            True if updated, False if document ID does not exist.
        """
        pass


    @abstractmethod
    def delete(self, document_id: str) -> bool:
        """Delete a document by ID.

        Args:
            document_id: Unique document identifier.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """Retrieve a paginated list of documents.

        Args:
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.

        Returns:
            List of Document objects.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored documents."""
        pass
