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
        # Load configuration settings from the .env file
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Ensure that the environment variables are set
        if not ai_endpoint or not ai_key:
            raise ValueError("AI_SERVICE_ENDPOINT and AI_SERVICE_KEY must be set in the environment.")

        # Set the default image file path
        image_file = 'images/pic_c.jpg'

        # If an image path is provided as a command-line argument, use it instead
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Read the image data from the file
        with open(image_file, "rb") as f:
            image_data = f.read()

        # Authenticate and create an Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )
        
        # Analyze the image using the specified client
        analyze_image(image_file, image_data, cv_client)

    except Exception as ex:
        # Catch and print any exceptions that occur
        print(f"Error: {ex}")

def analyze_image(image_filename, image_data, cv_client):
    print('\nAnalyzing image...')

    try:
        # Analyze the image to retrieve specified visual features
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.PEOPLE  # Focus on detecting people in the image
            ]
        )

        # Debug: Print the entire result object to understand its structure
        print("\nFull analysis result:")
        print(result)

        # Check if any people were detected in the image
        if result.people:
            print("\nPeople in image:")
            for detected_person in result.people:
                # Print details of each detected person
                print(f"Detected person details: {detected_person}")

                # Check if a bounding box is available for the detected person
                if hasattr(detected_person, 'bounding_box'):
                    print("Bounding box found")
                    
                    # Open the image for drawing
                    image = Image.open(image_filename)
                    fig = plt.figure(figsize=(image.width/100, image.height/100))
                    plt.axis('off')
                    draw = ImageDraw.Draw(image)
                    color = 'cyan'
                    
                    # Get the bounding box coordinates and draw it on the image
                    r = detected_person.bounding_box
                    bounding_box = [(r.x, r.y), (r.x + r.width, r.y + r.height)]
                    draw.rectangle(bounding_box, outline=color, width=3)

                    # Save the annotated image
                    plt.imshow(image)
                    plt.tight_layout(pad=0)
                    outputfile = 'people.jpg'
                    fig.savefig(outputfile)
                    print('Results saved in', outputfile)
                else:
                    print("Detected object does not have a bounding box")

    except HttpResponseError as e:
        # Handle any HTTP response errors from the Azure API
        print(f"Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")

if __name__ == "__main__":
    # Entry point of the script
    main()
