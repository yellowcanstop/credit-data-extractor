from __future__ import annotations
from pydantic import Field
from shared.workflows.base_request import BaseRequest


class BlobStorageRequest(BaseRequest):
    """Defines the base class for requests to interact with Azure Blob Storage."""

    storage_account_name: str = Field(
        description="The name of the Azure Storage account.")
    container_name: str = Field(
        description="The name of the container within the storage account.")
    blob_name: str = Field(
        description="The name of the blob within the container.")