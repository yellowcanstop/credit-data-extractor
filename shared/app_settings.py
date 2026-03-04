"""Defines the configuration settings for the Azure Functions application.

The variables are defined by environment variables configured in the from the `local.settings.json` file when running locally, and from the Azure Function App settings when running in Azure.
"""

import os
from configuration import Configuration
config = Configuration()

otel_exporter_otlp_endpoint = config.get_value(
    "OTEL_EXPORTER_OTLP_ENDPOINT", None)
azure_aiservices_endpoint = config.get_value(
    "AZURE_AISERVICES_ENDPOINT", None)
azure_openai_key = config.get_value("AZURE_OPENAI_API_KEY", None)
azure_aiservices_key = config.get_value("AZURE_AISERVICES_API_KEY", None)
azure_openai_endpoint = config.get_value("AZURE_OPENAI_ENDPOINT", None)
azure_openai_chat_deployment = config.get_value(
    "AZURE_OPENAI_CHAT_DEPLOYMENT", None)
azure_client_id = config.get_value("AZURE_CLIENT_ID", None)
azure_storage_account = config.get_value(
    "AZURE_STORAGE_ACCOUNT", None)
azure_storage_queues_connection_string = config.get_value(
    "AZURE_STORAGE_QUEUES_CONNECTION_STRING", None)
blob_account_url = config.get_value("BLOB_ACCOUNT_URL", None)
results_container_name = config.get_value("RESULTS_CONTAINER_NAME", "parser-outputs")