import azure.durable_functions as df
from reports.activities import validate_data, extract_data


def register_invoices(app: df.DFApp):
    """Register the related activities and workflows with the Durable Functions app."""
    app.register_functions(validate_data.bp)
    app.register_functions(extract_data.bp)