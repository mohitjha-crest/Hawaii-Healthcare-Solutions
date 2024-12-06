import os

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv
load_dotenv()


def analyze_layout(file_path=None, file_url=None):
    """
    Analyze a document using Form Recognizer's layout model.

    Args:
        file_path (str): Path to the local file.
        file_url (str): URL to a file in cloud storage.

    Returns:
        None
    """
    document_analysis_client = DocumentAnalysisClient(
        endpoint=os.environ.get('OCR_ENDPOINT'),
        credential=AzureKeyCredential(os.environ.get('OCR_API_KEY'))
    )
    try:
        if file_path:
            with open(file_path, "rb") as f:
                poller = document_analysis_client.begin_analyze_document("prebuilt-layout", document=f)
        elif file_url:
            poller = document_analysis_client.begin_analyze_document_from_url("prebuilt-layout", document_url=file_url)
        else:
            print("Please provide either a file path or a file URL.")
            return

        result = poller.result()

        return result.content

    except Exception as e:
        print(f"An error occurred: {e}")
