from __future__ import annotations
from pydantic import Field
from shared.workflows.validation_result import ValidationResult
import logging


class WorkflowResult(ValidationResult):
    """Defines the result of a workflow operation (orchestration or activity), containing a list of activity results in addition to the validation messages."""

    name: str = Field(
        description='The name of the workflow operation.'
    )

    activity_results: list[WorkflowResult] = Field(
        default_factory=list,
        description='A list of activity results generated during the workflow operation.'
    )

    def add_message(self, action: str, message: str):
        """Adds a structured message to the list of messages without changing the `is_valid` flag.

        :param action: The action that generated the message, e.g. a function name.
        :param message: The message to add.
        """

        log = f"{self.name}::{action} - {message}"
        logging.info(log)
        super().add_message(log)

    def add_error(self, action: str, message: str):
        """Adds a structured error message to the list of messages and sets the `is_valid` flag to `False`.

        :param action: The action that generated the error message, e.g. a function name.
        :param message: The error message to add.
        """

        log = f"{self.name}::{action} - {message}"
        logging.error(log)
        super().add_error(log)

    def add_activity_result(self, action: str, message: str, result: WorkflowResult):
        """Adds an activity result to the list of activity results, and logs a message.

        :param action: The action that generated the result, e.g. a function name.
        :param message: The message to log.
        :param result: The `WorkflowResult` instance to add as an activity result.
        """

        self.activity_results.append(result)
        log = f"{self.name}::{action} - {message}"
        logging.info(log)

    @staticmethod
    def to_json(obj: WorkflowResult) -> str:
        """Converts the object instance to a JSON string. Required for serialization in Azure Functions when passing the result between functions.

        :param obj: The object instance to convert.
        :return: A JSON string representing the object instance.
        """

        return obj.model_dump_json()

    @staticmethod
    def from_json(json_str: str) -> WorkflowResult:
        """Converts a JSON string to the object instance. Required for deserialization in Azure Functions when receiving the result from another function.

        :param json_str: The JSON string to convert.
        :return: A object instance created from the JSON string.
        """

        return WorkflowResult.model_validate_json(json_str)