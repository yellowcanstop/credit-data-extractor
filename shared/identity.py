"""Defines the default Azure credential for the application to authenticate with Azure services via Python SDKs.

When running locally, the Azure CLI credentials are used. When running in Azure, the application's managed identity is used.
"""

from azure.identity import DefaultAzureCredential
from shared import app_settings

default_credential = DefaultAzureCredential(
    exclude_environment_credential=True,
    exclude_interactive_browser_credential=True,
    exclude_visual_studio_code_credential=True,
    exclude_shared_token_cache_credential=True,
    exclude_developer_cli_credential=True,
    exclude_powershell_credential=True,
    exclude_workload_identity_credential=True,
    process_timeout=10,
    managed_identity_client_id=app_settings.azure_client_id
)