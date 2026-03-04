import azure.functions as func
import azure.durable_functions as df
from documents.setup import register_documents
from reports.setup import register_reports
from azure.storage.blob.aio import BlobServiceClient
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
            
            if not await container_client.exists():
                await container_client.create_container()
                logger.info(f"Created new container: {container_name}")

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