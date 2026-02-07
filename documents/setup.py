import azure.durable_functions as df
from documents.activities import get_document_folders, classify_document
from documents.workflows import process_document_batch_workflow, process_document_workflow


def register_documents(app: df.DFApp):
    """Register the document-related activities and workflows with the Durable Functions app."""
    app.register_functions(get_document_folders.bp)
    app.register_functions(process_document_batch_workflow.bp)
    app.register_functions(process_document_workflow.bp)