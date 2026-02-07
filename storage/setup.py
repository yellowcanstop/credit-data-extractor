import azure.durable_functions as df
from storage.activities import write_bytes_to_blob


def register_storage(app: df.DFApp):
    """Register the storage-related activities and workflows with the Durable Functions app."""
    app.register_functions(write_bytes_to_blob.bp)