import azure.functions as func
import azure.durable_functions as df
from documents.setup import register_documents
from reports.setup import register_reports

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="hello")
def test_function(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse("The backend is alive!", status_code=200)

register_documents(app)
register_reports(app)