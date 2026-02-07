from __future__ import annotations
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Defines the result of a validation operation, containing a list of messages and a flag indicating if the validation was successful."""

    is_valid: bool = Field(
        default=True,
        description='Indicates whether the validation was successful or not.'
    )
    messages: list[str] = Field(
        default_factory=list,
        description='A list of messages generated during the validation process.'
    )

    def add_message(self, message: str):
        """Adds a message to the list of messages without changing the `is_valid` flag.

        :param message: The message to add.
        """

        self.messages.append(message)

    def add_error(self, message: str):
        """Adds an error message to the list of messages and sets the `is_valid` flag to `False`.

        :param message: The error message to add.
        """

        self.is_valid = False
        self.messages.append(message)

    def merge(self, result: ValidationResult):
        """Merges the messages of another `ValidationResult` instance into the current instance and updates the `is_valid` flag accordingly.

        :param result: The `ValidationResult` instance to merge.
        """

        self.is_valid = self.is_valid and result.is_valid
        self.messages.extend(result.messages)

    def to_str(self):
        """Returns a string representation of the validation result messages as a comma-separated list."""

        return ", ".join(self.messages)

    @staticmethod
    def to_json(obj: ValidationResult) -> str:
        """Converts the object instance to a JSON string."""

        return obj.model_dump_json()

    @staticmethod
    def from_json(json_str: str) -> ValidationResult:
        """Converts a JSON string to the object instance."""

        return ValidationResult.model_validate_json(json_str)