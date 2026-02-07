"""Write bytes to a blob in Azure Blob Storage.

This module provides the blueprint for an Azure Function activity that writes a byte array to a blob in Azure Blob Storage.
"""

from __future__ import annotations
from pydantic import Field
from shared.workflows.validation_result import ValidationResult
from storage.models.blob_storage_request import BlobStorageRequest
from storage.services.azure_storage_client_factory import AzureStorageClientFactory
import shared.identity as identity
import azure.durable_functions as df
import logging

name = "WriteBytesToBlob"
bp = df.Blueprint()
storage_factory = AzureStorageClientFactory(identity.default_credential)


@bp.function_name(name)
@bp.activity_trigger(input_name="input", activity=name)
def run(input: Request) -> bool:
    """Writes a byte array to a blob in Azure Blob Storage.

    :param input: The blob storage information including the buffer byte array, storage account, container, and blob name.
    :return: True if the byte array was successfully written to the blob; otherwise, False.
    """

    validation_result = input.validate()
    if not validation_result.is_valid:
        logging.error(f"Invalid input: {validation_result.to_str()}")
        return False

    blob_container_client = storage_factory.get_blob_service_client(
        input.storage_account_name).get_container_client(input.container_name)

    if not blob_container_client.exists():
        blob_container_client.create_container()

    blob_client = blob_container_client.get_blob_client(input.blob_name)

    blob_client.upload_blob(input.content, overwrite=input.overwrite)

    return True


class Request(BlobStorageRequest):
    """Defines the request payload for the `WriteBytesToBlob` activity."""

    content: bytes = Field(
        description="The byte array content to write to the blob.")
    overwrite: bool = Field(
        default=True,
        description="A flag indicating whether to overwrite an existing blob with the same name. Default is `True`."
    )

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if not self.storage_account_name:
            result.add_error("storage_account_name is required")

        if not self.container_name:
            result.add_error("container_name is required")

        if not self.blob_name:
            result.add_error("blob_name is required")

        if not self.content:
            result.add_error("content is required")

        return result

    @staticmethod
    def to_json(obj: Request) -> str:
        """Converts the object instance to a JSON string."""

        return obj.model_dump_json()

    @staticmethod
    def from_json(json_str: str) -> Request:
        """Converts a JSON string to the object instance."""

        return Request.model_validate_json(json_str)