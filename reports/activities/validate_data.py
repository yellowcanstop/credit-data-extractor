"""Validates extracted data from an invoice for expected fields.

This module provides the blueprint for an Azure Function activity that validates extracted data from an invoice.
"""

from __future__ import annotations
from pydantic import Field
from enum import Flag, auto
from shared.workflows.workflow_result import WorkflowResult
from invoices.models.invoice import Invoice
from shared.workflows.base_request import BaseRequest
from shared.workflows.validation_result import ValidationResult
import azure.durable_functions as df

name = "ValidateInvoice"
bp = df.Blueprint()


@bp.function_name(name)
@bp.activity_trigger(input_name="input", activity=name)
def run(input: Request) -> Result:
    """Validates extracted data from an invoice for expected fields.

    :param input: The request containing the extracted invoice data.
    :return: The validation result.
    """

    result = Result(name=input.name or name, status=ResultStatus.Undetermined)

    validation_result = input.validate()
    if not validation_result.is_valid:
        result.merge(validation_result)
        return result

    data = input.data
    if not data.invoice_id:
        result.status |= ResultStatus.InvoiceIdMissing
        result.add_error(name, "invoice_id is required")

    __validate_items__(data, result)

    if result.is_valid:
        result.status = ResultStatus.Success
    else:
        result.status = ResultStatus.Fail

    return result


def __validate_items__(data: Invoice, result: Result):
    if not data.items:
        result.status |= ResultStatus.ItemsMissing
        result.add_error("items is required")
    else:
        for i, item in enumerate(data.items):
            if not item.product_code:
                result.status |= ResultStatus.ItemProductCodeMissing
                result.add_error(f"items[{i}].product_code is required")
            if not item.quantity:
                result.status |= ResultStatus.ItemQuantityMissing
                result.add_error(f"items[{i}].quantity is required")
            if not item.total:
                result.status |= ResultStatus.ItemTotalMissing
                result.add_error(f"items[{i}].total is required")


class Request(BaseRequest):
    """Defines the request payload for the `ValidateInvoice` activity."""

    name: str = Field(
        description="The name of the invoice blob."
    )
    data: Invoice = Field(
        description="The extracted invoice data."
    )

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if not self.name:
            result.add_error("name is required")

        if not self.data:
            result.add_error("data is required")

        return result

    @staticmethod
    def to_json(obj: Request) -> str:
        """Converts the object instance to a JSON string. Required for serialization in Azure Functions when passing the request between functions."""

        return obj.model_dump_json()

    @staticmethod
    def from_json(json_str: str) -> Request:
        """Converts a JSON string to the object instance. Required for deserialization in Azure Functions when receiving the request from another function."""

        return Request.model_validate_json(json_str)


class Result(WorkflowResult):
    """Defines the result payload for the `ValidateInvoice` activity."""

    status: ResultStatus = Field(
        description="The validation status of the invoice data."
    )

    @staticmethod
    def to_json(obj: Result) -> str:
        """Converts the object instance to a JSON string. Required for serialization in Azure Functions when passing the result between functions."""

        return obj.model_dump_json()

    @staticmethod
    def from_json(json_str: str) -> Result:
        """Converts a JSON string to the object instance. Required for deserialization in Azure Functions when receiving the result from another function."""

        return Result.model_validate_json(json_str)


class ResultStatus(Flag):
    """Defines the possible validation statuses for extracted invoice data."""

    Undetermined = 0
    Fail = auto()
    Success = auto()
    InvoiceIdMissing = auto()
    ItemsMissing = auto()
    ItemProductCodeMissing = auto()
    ItemQuantityMissing = auto()
    ItemTotalMissing = auto()