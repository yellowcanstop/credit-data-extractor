"""Get the document folders from a blob container.

This module provides the blueprint for an Azure Function activity that retrieves the document folders from a container in Azure Blob Storage.
"""

from __future__ import annotations
from documents.models.document_batch_request import DocumentBatchRequest
from documents.models.document_folder import DocumentFolders, DocumentFolder
from storage.services.azure_storage_client_factory import AzureStorageClientFactory
import shared.identity as identity
from shared import app_settings
import azure.durable_functions as df
import logging

name = "GetDocumentFolders"
bp = df.Blueprint()
storage_factory = AzureStorageClientFactory(identity.default_credential)


@bp.function_name(name)
@bp.activity_trigger(input_name="input", activity=name)
def run(input: DocumentBatchRequest) -> DocumentFolders:
    """Retrieves the document folders from a container in Azure Blob Storage.

    :param input: The document batch request containing the container name.
    :return: A list of `DocumentFolder` objects representing the document folders in the container.
    """

    grouped_documents = storage_factory.get_blobs_by_folder_at_root(
        app_settings.azure_storage_account, input.container_name, ".*\\.(pdf)$")

    logging.info(
        f"Found {len(grouped_documents)} folders in {input.container_name}")

    result = DocumentFolders(folders=[])
    for folder_name, document_file_names in grouped_documents.items():
        result.folders.append(DocumentFolder(container_name=input.container_name,
                                             name=folder_name,
                                             document_file_names=document_file_names))

    return result