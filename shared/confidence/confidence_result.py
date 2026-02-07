from __future__ import annotations
from typing import Generic, TypeVar
from pydantic import BaseModel, Field
import json
import importlib

DataT = TypeVar("DataT")

OVERALL_CONFIDENCE_KEY = "_overall"


class ConfidenceResult(BaseModel, Generic[DataT]):
    """Defines a class for wrapping the confidence score of a model."""

    data: DataT = Field(description="The data to wrap.")
    confidence_scores: dict = Field(
        description="The confidence scores for the data.")
    overall_confidence: float = Field(
        description="The overall confidence score for the data.")

    @staticmethod
    def to_json(obj: ConfidenceResult) -> str:
        """
        Convert the ConfidenceResult object to a JSON string.
        """

        obj_dict = obj.model_dump()

        # As we are using a generic type and have limited control over how Azure Functions
        # serializes the object, we need to add the data model type to the JSON output.
        if isinstance(obj.data, BaseModel):
            fqcn = obj.data.__class__.__module__ + "." + obj.data.__class__.__qualname__
            obj_dict["_data_model"] = fqcn

        return json.dumps(obj_dict)

    @staticmethod
    def from_json(json_str: str) -> ConfidenceResult:
        """
        Convert a JSON string to a ConfidenceResult object.
        """

        obj = json.loads(json_str)
        data = obj.get("data")

        # As we are using a generic type and have limited control over how Azure Functions
        # deserializes the object, we process the data model type from the prior
        # serialization step to reconstruct the object.
        model_name = obj.get("_data_model")
        if model_name and isinstance(data, dict):
            module_name, class_name = model_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            model_cls = getattr(module, class_name)
            if issubclass(model_cls, BaseModel):
                obj["data"] = model_cls.model_validate(data)

        # Remove the extra key as it's not part of the model
        obj.pop("_data_model", None)
        return ConfidenceResult.model_validate(obj)