from dotenv import load_dotenv
import os
import sys
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

def main():
    try:
        # Load configuration settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Check if the environment variables are set
        if not ai_endpoint or not ai_key:
            raise ValueError("AI_SERVICE_ENDPOINT and AI_SERVICE_KEY must be set in the environment.")

        # Get image file path
        image_file = 'images/bi.png'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Read image data
        with open(image_file, "rb") as f:
            image_data = f.read()

        # Authenticate Azure Form Recognizer client (which supports OCR)
        document_analysis_client = DocumentAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )

        # Analyze image for text extraction (OCR)
        extract_text_from_image(image_file, image_data, document_analysis_client)

    except Exception as ex:
        print(f"Error: {ex}")

def extract_text_from_image(image_filename, image_data, client):
    print(f'\nReading text from image: {image_filename}')

    try:
        # Perform the Read operation (using Form Recognizer's OCR functionality)
        poller = client.begin_analyze_document("prebuilt-read", image_data)

        # Wait for the operation to complete and get results
        result = poller.result()

        # Display text extraction results
        for page in result.pages:
            for line in page.lines:
                print(f"Text: {line.content}")
    
    except Exception as e:
        print(f"Error occurred while reading the image: {e}")

if __name__ == "__main__":
    main()
