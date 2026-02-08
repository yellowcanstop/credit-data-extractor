"""Processes a document in a folder in a Storage container.

The workflows orchestrate the detection of the document type, and if it's an invoice, extract the invoice data from the folder and save the extracted data to a database.
"""

from __future__ import annotations
from reports.activities import extract_data
from shared.confidence.confidence_result import ConfidenceResult
from shared.workflows.workflow_result import WorkflowResult
from documents.models.document_folder import DocumentFolder
import azure.durable_functions as df

name = "ProcessDocumentWorkflow"
bp = df.Blueprint()

CONFIDENCE_THRESHOLD = 0.8


@bp.function_name(name)
@bp.orchestration_trigger(context_name="context", orchestration=name)
def run(context: df.DurableOrchestrationContext):
    # Step 1: Extract the input from the context
    input: DocumentFolder = context.get_input()
    result = WorkflowResult(name=input.name)

    # Step 2: Validate the input
    validation_result = input.validate()
    if not validation_result.is_valid:
        result.merge(validation_result)
        return result

    result.add_message("DocumentFolder.validate", "input is valid")

    # Step 3: Process each file
    for document in input.document_file_names:
        extracted_data = yield context.call_activity(
            extract_data.name,
            extract_data.Request(
                container_name=input.container_name,
                blob_name=document))

        if not extracted_data:
            result.add_error(
                extract_data.name,
                f"Failed to extract data for {document}.")
            continue

    return result.model_dump()