from dotenv import load_dotenv
import os
import sys
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

def main():
    try:
        # Load configuration settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        if not ai_endpoint or not ai_key:
            raise ValueError("AI_SERVICE_ENDPOINT and AI_SERVICE_KEY must be set in the environment.")

        # Get image file path
        image_file = 'images/bi.png'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Verify the file exists
        if not os.path.isfile(image_file):
            raise FileNotFoundError(f"The file '{image_file}' does not exist. Please provide a valid image path.")

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
        # Get result with specified features to be retrieved
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.TAGS]
        )

        # Debugging: Check the type and content of result.tags
        print(f"\nDebugging Info: result.tags type: {type(result.tags)}")
        print(f"Content: {result.tags}")

        # Display analysis results
        if isinstance(result.tags, list):  # Check if it's a list
            print("\nTags:")
            for tag in result.tags:
                print(f" Tag: '{tag.name}' (confidence: {tag.confidence * 100:.2f}%)")
        else:
            print("Unexpected format for tags:", result.tags)

    except HttpResponseError as e:
        print(f"HTTP Response Error: Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")

if __name__ == "__main__":
    main()
