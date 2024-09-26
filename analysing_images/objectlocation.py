from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def main():
    try:
        # Load environment variables from a .env file
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Ensure both the endpoint and key are available
        if not ai_endpoint or not ai_key:
            raise ValueError("AI_SERVICE_ENDPOINT and AI_SERVICE_KEY must be set in the environment.")

        # Set default image file path
        image_file = 'images/pic_a.jpg'
        # Override the default if a file path is provided as a command-line argument
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Check if the specified image file exists
        if not os.path.isfile(image_file):
            raise FileNotFoundError(f"The file '{image_file}' does not exist. Please provide a valid image path.")

        # Read the image data in binary format
        with open(image_file, "rb") as f:
            image_data = f.read()

        # Create a client for the Azure AI Vision service
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )
        
        # Analyze the image using the Vision service
        analyze_image(image_file, image_data, cv_client)

    except Exception as ex:
        print(f"Error: {ex}")

def analyze_image(image_filename, image_data, cv_client):
    print(f'\nAnalyzing image: {image_filename}')

    try:
        # Analyze the image for objects
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.OBJECTS]
        )

        # Debug: Print the raw analysis result to understand the structure
        print("Analysis result:", result)

        # Check if objects are detected in the image
        if result.objects:
            print("\nObjects in image:")

            # Open the image for drawing bounding boxes
            image = Image.open(image_filename)
            fig = plt.figure(figsize=(image.width / 100, image.height / 100))
            plt.axis('off')
            draw = ImageDraw.Draw(image)
            color = 'cyan'

            # Iterate over each detected object
            for detected_object in result.objects:
                # Debug: Print the detected object and its type
                print("Detected object:", detected_object)
                print("Type of detected object:", type(detected_object))

                # Ensure that detected_object is not a string
                if isinstance(detected_object, str):
                    print("Error: Detected object is a string, not an object.")
                    continue

                # Print object name and confidence level
                print(" {} (confidence: {:.2f}%)".format(detected_object.name, detected_object.confidence * 100))

                # Draw the bounding box around the detected object
                r = detected_object.rectangle
                bounding_box = ((r.x, r.y), (r.x + r.w, r.y + r.h))
                draw.rectangle(bounding_box, outline=color, width=3)
                plt.annotate(detected_object.name, (r.x, r.y), backgroundcolor=color)

            # Display and save the annotated image with bounding boxes
            plt.imshow(image)
            plt.tight_layout(pad=0)
            outputfile = 'objects.jpg'
            fig.savefig(outputfile)
            print('Results saved in', outputfile)
        else:
            print("No objects detected in the image.")

    except HttpResponseError as e:
        # Handle HTTP errors from the Vision API
        print(f"HTTP Response Error: Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")

if __name__ == "__main__":
    main()
