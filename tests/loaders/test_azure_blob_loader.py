"""Tests for AzureBlobLoader."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders import AzureBlobLoader, Document


class TestAzureBlobLoader:
    def test_init_requires_container_name(self):
        with pytest.raises(ValueError, match="container_name must be provided"):
            AzureBlobLoader(container_name="", connection_string="UseDevelopmentStorage=true")

    def test_init_requires_auth(self):
        with pytest.raises(
            ValueError,
            match="Provide either connection_string or both account_url and credential",
        ):
            AzureBlobLoader(container_name="docs")

    def test_init_allows_connection_string(self):
        loader = AzureBlobLoader(container_name="docs", connection_string="conn")
        assert loader.container_name == "docs"
        assert loader.connection_string == "conn"
        assert loader.account_url is None
        assert loader.credential is None

    def test_init_allows_account_credentials(self):
        loader = AzureBlobLoader(
            container_name="docs",
            account_url="https://acct.blob.core.windows.net",
            credential="secret",
            prefix="reports/",
            max_files=5,
        )
        assert loader.container_name == "docs"
        assert loader.account_url == "https://acct.blob.core.windows.net"
        assert loader.credential == "secret"
        assert loader.prefix == "reports/"
        assert loader.max_files == 5

    @pytest.mark.asyncio
    async def test_aload_missing_dependencies(self):
        import sys

        loader = AzureBlobLoader(container_name="docs", connection_string="conn")

        with patch.dict(sys.modules, {"azure": None, "azure.storage": None}):
            with pytest.raises(ImportError, match="synapsekit\\[azure\\]"):
                await loader.aload()

    @patch.dict("sys.modules", {"azure.storage.blob": MagicMock()})
    def test_load_with_text_and_binary_blobs_connection_string(self):
        import sys

        blob_service_client_cls = sys.modules["azure.storage.blob"].BlobServiceClient

        mock_service = MagicMock()
        blob_service_client_cls.from_connection_string.return_value = mock_service

        container_client = MagicMock()
        mock_service.get_container_client.return_value = container_client

        txt_blob = MagicMock()
        txt_blob.name = "docs/readme.txt"
        txt_blob.size = 13
        txt_blob.content_settings = MagicMock(content_type="text/plain")
        txt_blob.last_modified = MagicMock()
        txt_blob.last_modified.isoformat.return_value = "2026-04-01T10:00:00+00:00"

        bin_blob = MagicMock()
        bin_blob.name = "images/logo.png"
        bin_blob.size = 2048
        bin_blob.content_settings = MagicMock(content_type="image/png")
        bin_blob.last_modified = MagicMock()
        bin_blob.last_modified.isoformat.return_value = "2026-04-01T11:00:00+00:00"

        container_client.list_blobs.return_value = [txt_blob, bin_blob]

        txt_blob_client = MagicMock()
        txt_blob_client.download_blob.return_value.readall.return_value = b"hello from azure"

        bin_blob_client = MagicMock()
        bin_blob_client.download_blob.return_value.readall.return_value = b"\x89PNG\x00\x01"

        def get_blob_client(name: str):
            if name == "docs/readme.txt":
                return txt_blob_client
            return bin_blob_client

        container_client.get_blob_client.side_effect = get_blob_client

        loader = AzureBlobLoader(container_name="docs", connection_string="conn")
        docs = loader.load()

        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

        assert docs[0].text == "hello from azure"
        assert docs[0].metadata["source"] == "azure_blob"
        assert docs[0].metadata["container"] == "docs"
        assert docs[0].metadata["blob_name"] == "docs/readme.txt"
        assert docs[0].metadata["content_type"] == "text/plain"

        assert docs[1].text == "[Binary file: image/png]"
        assert docs[1].metadata["blob_name"] == "images/logo.png"

        container_client.list_blobs.assert_called_once_with(name_starts_with=None)

    @patch.dict("sys.modules", {"azure.storage.blob": MagicMock()})
    def test_load_extracts_supported_file_via_loader(self):
        import sys

        blob_service_client_cls = sys.modules["azure.storage.blob"].BlobServiceClient

        mock_service = MagicMock()
        blob_service_client_cls.from_connection_string.return_value = mock_service

        container_client = MagicMock()
        mock_service.get_container_client.return_value = container_client

        pdf_blob = MagicMock()
        pdf_blob.name = "reports/monthly.pdf"
        pdf_blob.size = 1024
        pdf_blob.content_settings = MagicMock(content_type="application/pdf")
        pdf_blob.last_modified = None

        container_client.list_blobs.return_value = [pdf_blob]

        pdf_blob_client = MagicMock()
        pdf_blob_client.download_blob.return_value.readall.return_value = b"%PDF-1.7 fake bytes"
        container_client.get_blob_client.return_value = pdf_blob_client

        loader = AzureBlobLoader(container_name="docs", connection_string="conn")

        with patch.object(loader, "_run_loader_for_extension") as run_loader:
            run_loader.return_value = [
                Document(text="Extracted Page 1", metadata={}),
                Document(text="Extracted Page 2", metadata={}),
            ]
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Extracted Page 1\n\nExtracted Page 2"
        run_loader.assert_called_once()

    @patch.dict("sys.modules", {"azure.storage.blob": MagicMock()})
    def test_load_with_account_url_and_credential(self):
        import sys

        blob_service_client_cls = sys.modules["azure.storage.blob"].BlobServiceClient

        mock_service = MagicMock()
        blob_service_client_cls.return_value = mock_service

        container_client = MagicMock()
        mock_service.get_container_client.return_value = container_client

        blob = MagicMock()
        blob.name = "reports/weekly.md"
        blob.size = 42
        blob.content_settings = MagicMock(content_type="text/markdown")
        blob.last_modified = None

        container_client.list_blobs.return_value = [blob]

        blob_client = MagicMock()
        blob_client.download_blob.return_value.readall.return_value = b"# Weekly report"
        container_client.get_blob_client.return_value = blob_client

        loader = AzureBlobLoader(
            container_name="reports",
            account_url="https://acct.blob.core.windows.net",
            credential="secret",
            prefix="reports/",
            max_files=1,
        )

        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "# Weekly report"
        assert docs[0].metadata["container"] == "reports"

        blob_service_client_cls.assert_called_once_with(
            account_url="https://acct.blob.core.windows.net",
            credential="secret",
        )
        container_client.list_blobs.assert_called_once_with(name_starts_with="reports/")

    @patch.dict("sys.modules", {"azure.storage.blob": MagicMock()})
    def test_load_skips_virtual_directories(self):
        import sys

        blob_service_client_cls = sys.modules["azure.storage.blob"].BlobServiceClient

        mock_service = MagicMock()
        blob_service_client_cls.from_connection_string.return_value = mock_service

        container_client = MagicMock()
        mock_service.get_container_client.return_value = container_client

        folder_marker = MagicMock()
        folder_marker.name = "folder/"

        real_blob = MagicMock()
        real_blob.name = "folder/file.txt"
        real_blob.size = 5
        real_blob.content_settings = MagicMock(content_type="text/plain")
        real_blob.last_modified = None

        container_client.list_blobs.return_value = [folder_marker, real_blob]

        blob_client = MagicMock()
        blob_client.download_blob.return_value.readall.return_value = b"hello"
        container_client.get_blob_client.return_value = blob_client

        loader = AzureBlobLoader(container_name="docs", connection_string="conn")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["blob_name"] == "folder/file.txt"

    @patch.dict("sys.modules", {"azure.storage.blob": MagicMock()})
    def test_aload_runs(self):
        import sys

        blob_service_client_cls = sys.modules["azure.storage.blob"].BlobServiceClient

        mock_service = MagicMock()
        blob_service_client_cls.from_connection_string.return_value = mock_service

        container_client = MagicMock()
        mock_service.get_container_client.return_value = container_client

        blob = MagicMock()
        blob.name = "note.txt"
        blob.size = 4
        blob.content_settings = MagicMock(content_type="text/plain")
        blob.last_modified = None

        container_client.list_blobs.return_value = [blob]

        blob_client = MagicMock()
        blob_client.download_blob.return_value.readall.return_value = b"test"
        container_client.get_blob_client.return_value = blob_client

        loader = AzureBlobLoader(container_name="docs", connection_string="conn")
        docs = asyncio.run(loader.aload())

        assert len(docs) == 1
        assert docs[0].text == "test"

    def test_extract_with_supported_loader_returns_none_for_unsupported_ext(self):
        loader = AzureBlobLoader(container_name="docs", connection_string="conn")
        out = loader._extract_with_supported_loader("blob.bin", b"raw", ".bin")
        assert out is None
