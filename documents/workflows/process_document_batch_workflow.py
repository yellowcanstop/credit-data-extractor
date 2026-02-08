"""Processes a batch of document folders in a Storage container.

For each of the documents in the Storage container, a sub-orchestration is started.
"""

from __future__ import annotations
from shared.workflows.workflow_result import WorkflowResult
from documents.workflows import process_document_workflow
from documents.models.document_folder import DocumentFolders
from documents.models.document_batch_request import DocumentBatchRequest
import azure.durable_functions as df
from azure.durable_functions.models.Task import TaskBase
import azure.functions as func
import logging
from documents.activities import get_document_folders

name = "ProcessDocumentBatchWorkflow"
http_trigger_name = "ProcessDocumentBatchHttp"
bp = df.Blueprint()


@bp.function_name(http_trigger_name)
@bp.route(route="process-documents", methods=["POST"])
@bp.durable_client_input(client_name="client")
async def process_document_batch_http(req: func.HttpRequest, client):
    """Starts a new instance of the ProcessDocumentBatchWorkflow orchestration in response to an HTTP request.

    :param req: The HTTP request trigger containing the document batch request in the body.
    :param client: The Durable Orchestration Client to start the workflow.
    :return: The 202 Accepted response with a dictionary of orchestrator management URLs.
    """
    request_body = req.get_json()
    document_batch_request = DocumentBatchRequest.model_validate(request_body)

    instance_id = await client.start_new(name, client_input=document_batch_request)

    logging.info(f"Started workflow with instance ID: {instance_id}")

    return client.create_check_status_response(req, instance_id)


@bp.function_name(name)
@bp.orchestration_trigger(context_name="context", orchestration=name)
def run(context: df.DurableOrchestrationContext):
    """Orchestrates the processing of a batch of document folders in a Storage container.

    :param context: The Durable Orchestration Context containing the input data for the workflow.
    :return: The `WorkflowResult` of the workflow operation containing the validation messages and activity results.
    """

    # Step 1: Extract the input from the context
    input: DocumentBatchRequest = context.get_input()
    result = WorkflowResult(name=name)

    # Step 2: Validate the input
    validation_result = input.validate()
    if not validation_result.is_valid:
        result.merge(validation_result)
        return result

    result.add_message("DocumentBatchRequest.validate", "input is valid")

    # Step 3: Get the document folders from the blob container
    document_folders: DocumentFolders = yield context.call_activity(get_document_folders.name, input)

    result.add_message(get_document_folders.name,
                       f"Retrieved {len(document_folders.folders)} document folders.")

    # Step 4: Process the documents in each folder.
    process_document_tasks: list[TaskBase] = []
    for folder in document_folders.folders:
        process_document_task = context.call_sub_orchestrator(
            process_document_workflow.name, folder)
        process_document_tasks.append(process_document_task)

    yield context.task_all(process_document_tasks)

    for task in process_document_tasks:
        task_result = WorkflowResult.model_validate(task.result)
        result.add_activity_result(process_document_workflow.name,
                                   "Processed document folder.",
                                   task_result)

    return result.model_dump()