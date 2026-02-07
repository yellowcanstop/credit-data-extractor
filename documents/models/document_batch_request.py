from __future__ import annotations
from pydantic import Field
from shared.workflows.validation_result import ValidationResult
from shared.workflows.base_request import BaseRequest


class DocumentBatchRequest(BaseRequest):
    """Defines a request to process a batch of documents in a Storage container."""

    container_name: str = Field(
        description='The name of the Azure Blob Storage container containing the document folders.'
    )

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if not self.container_name:
            result.add_error("container_name is required")

        return result

    @staticmethod
    def to_json(obj: DocumentBatchRequest) -> str:
        """
        Convert the DocumentBatchRequest object to a JSON string.
        """
        return obj.model_dump_json()

    @staticmethod
    def from_json(json_str: str) -> DocumentBatchRequest:
        """
        Convert a JSON string to an DocumentBatchRequest object.
        """
        return DocumentBatchRequest.model_validate_json(json_str)