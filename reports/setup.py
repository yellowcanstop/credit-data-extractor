import azure.durable_functions as df
from reports.activities import extract_data


def register_reports(app: df.DFApp):
    """Register the related activities and workflows with the Durable Functions app."""
    app.register_functions(extract_data.bp)