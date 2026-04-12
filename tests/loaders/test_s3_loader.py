"""Tests for S3Loader."""

from __future__ import annotations

import asyncio
import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders import Document, S3Loader


class TestS3Loader:
    def test_init_requires_bucket_name(self):
        with pytest.raises(ValueError, match="bucket_name must be provided"):
            S3Loader(bucket_name="")

    def test_init_with_defaults(self):
        loader = S3Loader(bucket_name="docs")
        assert loader.bucket_name == "docs"
        assert loader.prefix is None
        assert loader.file_extensions is None
        assert loader.region_name == "us-east-1"

    def test_init_normalizes_extensions(self):
        loader = S3Loader(bucket_name="docs", file_extensions=["txt", ".PDF", "  .json  "])
        assert loader.file_extensions == {".txt", ".pdf", ".json"}

    @pytest.mark.asyncio
    async def test_aload_missing_dependencies(self):
        import sys

        loader = S3Loader(bucket_name="docs")

        with patch.dict(sys.modules, {"boto3": None}):
            with pytest.raises(ImportError, match="synapsekit\\[s3\\]"):
                await loader.aload()

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    def test_load_with_text_and_binary_objects(self):
        import sys

        mock_boto3 = sys.modules["boto3"]
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator

        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "docs/readme.txt",
                        "LastModified": dt.datetime(2026, 4, 1, 10, 0, tzinfo=dt.UTC),
                        "Size": 12,
                    },
                    {
                        "Key": "images/logo.png",
                        "LastModified": dt.datetime(2026, 4, 1, 11, 0, tzinfo=dt.UTC),
                        "Size": 2048,
                    },
                ]
            }
        ]

        txt_body = MagicMock()
        txt_body.read.return_value = b"hello s3 text"

        bin_body = MagicMock()
        bin_body.read.return_value = b"\x89PNG\x00\x01"

        def get_object(**kwargs):
            if kwargs["Key"] == "docs/readme.txt":
                return {"Body": txt_body, "ContentType": "text/plain"}
            return {"Body": bin_body, "ContentType": "image/png"}

        mock_client.get_object.side_effect = get_object

        loader = S3Loader(bucket_name="docs")
        docs = loader.load()

        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

        assert docs[0].text == "hello s3 text"
        assert docs[0].metadata["source"] == "s3"
        assert docs[0].metadata["bucket"] == "docs"
        assert docs[0].metadata["key"] == "docs/readme.txt"

        assert docs[1].text == "[Binary file: image/png]"
        assert docs[1].metadata["key"] == "images/logo.png"

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    def test_load_applies_prefix_and_extension_filters(self):
        import sys

        mock_boto3 = sys.modules["boto3"]
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator

        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "reports/a.txt", "LastModified": None, "Size": 1},
                    {"Key": "reports/b.pdf", "LastModified": None, "Size": 2},
                ]
            }
        ]

        body = MagicMock()
        body.read.return_value = b"ok"
        mock_client.get_object.return_value = {"Body": body, "ContentType": "text/plain"}

        loader = S3Loader(bucket_name="docs", prefix="reports/", file_extensions=[".txt"])
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["key"] == "reports/a.txt"
        paginator.paginate.assert_called_once_with(Bucket="docs", Prefix="reports/")

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    def test_load_extracts_supported_file_via_loader(self):
        import sys

        mock_boto3 = sys.modules["boto3"]
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "reports/monthly.pdf", "LastModified": None, "Size": 10}]}
        ]

        body = MagicMock()
        body.read.return_value = b"%PDF-1.7 fake bytes"
        mock_client.get_object.return_value = {"Body": body, "ContentType": "application/pdf"}

        loader = S3Loader(bucket_name="docs")

        with patch.object(loader, "_run_loader_for_extension") as run_loader:
            run_loader.return_value = [
                Document(text="Extracted Page 1", metadata={}),
                Document(text="Extracted Page 2", metadata={}),
            ]
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Extracted Page 1\n\nExtracted Page 2"
        run_loader.assert_called_once()

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    def test_load_respects_max_files(self):
        import sys

        mock_boto3 = sys.modules["boto3"]
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "a.txt", "LastModified": None, "Size": 1},
                    {"Key": "b.txt", "LastModified": None, "Size": 2},
                ]
            }
        ]

        body = MagicMock()
        body.read.return_value = b"x"
        mock_client.get_object.return_value = {"Body": body, "ContentType": "text/plain"}

        loader = S3Loader(bucket_name="docs", max_files=1)
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["key"] == "a.txt"

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    def test_aload_runs(self):
        import sys

        mock_boto3 = sys.modules["boto3"]
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "note.txt", "LastModified": None, "Size": 4}]}
        ]

        body = MagicMock()
        body.read.return_value = b"test"
        mock_client.get_object.return_value = {"Body": body, "ContentType": "text/plain"}

        loader = S3Loader(bucket_name="docs")
        docs = asyncio.run(loader.aload())

        assert len(docs) == 1
        assert docs[0].text == "test"

    def test_extract_with_supported_loader_returns_none_for_unsupported_ext(self):
        loader = S3Loader(bucket_name="docs")
        out = loader._extract_with_supported_loader("blob.bin", b"raw", ".bin")
        assert out is None
