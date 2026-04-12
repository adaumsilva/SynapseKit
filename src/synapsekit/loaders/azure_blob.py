"""AzureBlobLoader — load files from Azure Blob Storage."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any, cast

from .base import Document

logger = logging.getLogger(__name__)


class AzureBlobLoader:
    """Load files from Azure Blob Storage into Documents.

    Supports authentication via either connection string or account URL + credential.
    """

    _TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".yaml",
        ".yml",
        ".xml",
        ".log",
        ".rst",
        ".py",
    }

    _EXTRACTABLE_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".json", ".html", ".htm"}

    def __init__(
        self,
        container_name: str,
        connection_string: str | None = None,
        account_url: str | None = None,
        credential: str | None = None,
        prefix: str | None = None,
        max_files: int | None = None,
    ) -> None:
        if not container_name:
            raise ValueError("container_name must be provided")

        if not connection_string and not (account_url and credential):
            raise ValueError("Provide either connection_string or both account_url and credential")

        self.container_name = container_name
        self.connection_string = connection_string
        self.account_url = account_url
        self.credential = credential
        self.prefix = prefix
        self.max_files = max_files

    def load(self) -> list[Document]:
        """Synchronously fetch files from Azure Blob Storage."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.aload())
        finally:
            loop.close()

    async def aload(self) -> list[Document]:
        """Asynchronously fetch files from Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            raise ImportError(
                "Azure Blob Storage dependencies required: pip install synapsekit[azure]"
            ) from None

        loop = asyncio.get_running_loop()

        if self.connection_string:
            client = await loop.run_in_executor(
                None,
                lambda: BlobServiceClient.from_connection_string(self.connection_string),
            )
        else:
            client = await loop.run_in_executor(
                None,
                lambda: BlobServiceClient(account_url=self.account_url, credential=self.credential),
            )

        container_client = client.get_container_client(self.container_name)
        blobs = await loop.run_in_executor(
            None,
            lambda: list(container_client.list_blobs(name_starts_with=self.prefix)),
        )

        if self.max_files is not None:
            blobs = blobs[: self.max_files]

        docs: list[Document] = []

        for blob in blobs:
            if blob.name.endswith("/"):
                continue

            try:
                blob_client = container_client.get_blob_client(blob.name)
                content_bytes = await loop.run_in_executor(
                    None,
                    self._download_blob_bytes,
                    blob_client,
                )
                content_type = self._blob_content_type(blob)
                text = self._extract_text(blob.name, content_bytes, content_type)
                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "source": "azure_blob",
                            "container": self.container_name,
                            "blob_name": blob.name,
                            "size": getattr(blob, "size", None),
                            "content_type": content_type,
                            "last_modified": (
                                blob.last_modified.isoformat()
                                if getattr(blob, "last_modified", None)
                                else None
                            ),
                        },
                    )
                )
            except Exception as exc:
                logger.warning("AzureBlobLoader: skipping blob %r — %s", blob.name, exc)

        return docs

    @staticmethod
    def _download_blob_bytes(blob_client: Any) -> bytes:
        return cast(bytes, blob_client.download_blob().readall())

    def _blob_content_type(self, blob: Any) -> str | None:
        content_settings = getattr(blob, "content_settings", None)
        if content_settings is None:
            return None
        return getattr(content_settings, "content_type", None)

    def _extract_text(self, blob_name: str, content: bytes, content_type: str | None) -> str:
        ext = os.path.splitext(blob_name)[1].lower()

        if ext in self._TEXT_EXTENSIONS:
            return content.decode("utf-8", errors="replace")

        extracted = self._extract_with_supported_loader(blob_name, content, ext)
        if extracted is not None:
            return extracted

        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            descriptor = content_type or ext or "unknown"
            return f"[Binary file: {descriptor}]"

    def _extract_with_supported_loader(
        self, blob_name: str, content: bytes, ext: str
    ) -> str | None:
        if ext not in self._EXTRACTABLE_EXTENSIONS:
            return None

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(content)
                temp_path = tmp.name

            docs = self._run_loader_for_extension(ext, temp_path)
            return "\n\n".join(doc.text for doc in docs if doc.text)
        except Exception as exc:
            logger.warning("AzureBlobLoader: extraction fallback for %r — %s", blob_name, exc)
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _run_loader_for_extension(self, ext: str, path: str) -> list[Document]:
        if ext == ".pdf":
            from .pdf import PDFLoader

            return PDFLoader(path).load()
        if ext == ".docx":
            from .docx import DocxLoader

            return DocxLoader(path).load()
        if ext == ".xlsx":
            from .excel import ExcelLoader

            return ExcelLoader(path).load()
        if ext == ".pptx":
            from .pptx import PowerPointLoader

            return PowerPointLoader(path).load()
        if ext == ".csv":
            from .csv import CSVLoader

            return CSVLoader(path).load()
        if ext == ".json":
            from .json_loader import JSONLoader

            return JSONLoader(path).load()
        if ext in {".html", ".htm"}:
            from .html import HTMLLoader

            return HTMLLoader(path).load()

        return []
