import logging
import azure.functions as func
import json

# Import helper script
from .predict import predict_image_from_url


def main(req: func.HttpRequest) -> func.HttpResponse:
    """This function is the main entry point for the Azure Function.

    Parameters
    ----------
    req : func.HttpRequest
        The request object.

    Returns
    -------
    func.HttpResponse
        The response object.
    """
    image_url = req.params.get("img")
    logging.info("Image URL received: " + image_url)
    
    results = predict_image_from_url(image_url)
    
    headers = {"Content-type": "application/json", "Access-Control-Allow-Origin": "*"}
    
    return func.HttpResponse(json.dumps(results), headers=headers)
