import azure.functions as func
import azure.durable_functions as df
from documents.setup import register_documents
from reports.setup import register_reports
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
import logging
from shared import app_settings
import json

logger = logging.getLogger(__name__)

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="hello")
def test_function(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse("The backend is alive!", status_code=200)

register_documents(app)
register_reports(app)

@app.route(route="upload", methods=["POST"])
async def upload_leads(req: func.HttpRequest):
    logger.info("Processing bulk file upload...")

    container_name = req.form.get('container_name')
    if not container_name:
        return func.HttpResponse("Missing 'container_name' in request.", status_code=400)

    files = req.files.getlist('files')
    if not files:
        return func.HttpResponse("No files provided in the 'files' field.", status_code=400)

    blob_service = BlobServiceClient.from_connection_string(app_settings.blob_account_url)

    uploaded_blobs = []

    try:
        async with blob_service:
            container_client = blob_service.get_container_client(container_name)
            
            try:
                await container_client.create_container()
                logger.info(f"Created new container: {container_name}")
            except ResourceExistsError:
                logger.info(f"Container {container_name} already exists. Proceeding...")

            for file in files:
                # file.filename will be "customer_1/report1.pdf" (the virtual path)
                blob_name = file.filename 
                blob_client = container_client.get_blob_client(blob_name)
                
                file_body = file.read()
                await blob_client.upload_blob(file_body, overwrite=True)
                
                uploaded_blobs.append(blob_name)
                logger.info(f"Uploaded: {blob_name} to {container_name}")

        return func.HttpResponse(
            json.dumps({
                "message": "Bulk upload successful",
                "container": container_name,
                "uploaded_count": len(uploaded_blobs),
                "files": uploaded_blobs
            }),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logger.error(f"Bulk upload failed: {str(e)}")
        return func.HttpResponse(f"Internal Error: {str(e)}", status_code=500)


@app.route(route="status/{instance_id}", methods=["GET"])
@app.durable_client_input(client_name="client")
async def check_status(req: func.HttpRequest, client: df.DurableOrchestrationClient):
    instance_id = req.route_params["instance_id"]
    logger.info(f"Checking status for instance ID: {instance_id}")
    status = await client.get_status(instance_id)
    if not status:
        return func.HttpResponse(
            json.dumps({"error": "Instance ID not found or not yet initialized."}),
            mimetype="application/json",
            status_code=404
        )
    status_data = status.to_json()
    return func.HttpResponse(
        json.dumps(status_data), 
        mimetype="application/json",
        status_code=200
    )