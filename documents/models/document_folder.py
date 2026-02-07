from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from shared.workflows.validation_result import ValidationResult


class DocumentFolder(BaseModel):
    """Defines a folder containing a set of documents."""

    container_name: Optional[str] = Field(
        description='The name of the Azure Blob Storage container containing the documents.'
    )
    name: Optional[str] = Field(
        description='The name of the folder containing the documents.'
    )
    document_file_names: Optional[list[str]] = Field(
        description='A list of the blob names of the document files in the container.'
    )

    @staticmethod
    def to_json(obj: DocumentFolder) -> str:
        """
        Convert the DocumentFolder object to a JSON string.
        """
        return obj.model_dump_json()

    @staticmethod
    def from_json(json_str: str) -> DocumentFolder:
        """
        Convert a JSON string to a DocumentFolder object.
        """
        return DocumentFolder.model_validate_json(json_str)

    def validate(self) -> ValidationResult:
        """
        Validate the DocumentFolder object.

        :return: A ValidationResult object containing any validation errors.
        """
        result = ValidationResult()

        if not self.container_name:
            result.add_error("container_name is required")

        if not self.name:
            result.add_error("name is required")

        if not self.document_file_names or len(self.document_file_names) == 0:
            result.add_error("document_file_names is required")

        return result


class DocumentFolders(BaseModel):
    """Defines a list of DocumentFolder objects."""

    folders: list[DocumentFolder] = Field(
        default_factory=list,
        description='A list of DocumentFolder objects.'
    )

    @staticmethod
    def to_json(obj: DocumentFolders) -> str:
        """
        Convert the DocumentFolders object to a JSON string.
        """
        return obj.model_dump_json()

    @staticmethod
    def from_json(json_str: str) -> DocumentFolders:
        """
        Convert a JSON string to a DocumentFolders object.
        """
        return DocumentFolders.model_validate_json(json_str)