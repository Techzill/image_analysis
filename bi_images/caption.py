from dotenv import load_dotenv
import os
from PIL import Image
import sys
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

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

        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )
        
        # Analyze image
        analyze_image(image_file, image_data, cv_client)
        
    except Exception as ex:
        print(f"Error: {ex}")

def analyze_image(image_filename, image_data, cv_client):
    print(f'\nAnalyzing image: {image_filename}')

    try:
        # Get analysis result with specified features
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS
            ]
        )

        # Display analysis results
        if result.caption:
            print("\nCaption:")
            print(f" Caption: '{result.caption.text}' (confidence: {result.caption.confidence * 100:.2f}%)")

        if result.dense_captions:
            print("\nDense Captions:")
            for caption in result.dense_captions.list:
                print(f" Caption: '{caption.text}' (confidence: {caption.confidence * 100:.2f}%)")

    except HttpResponseError as e:
        print(f"HTTP Response Error: Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")

if __name__ == "__main__":
    main()
