from dotenv import load_dotenv
import os
import sys
import requests

def ForegroundMatting(endpoint, key, image_file):
    # Define the URL for the foreground matting API
    url = f"{endpoint}computervision/imageanalysis:segment?api-version=2023-02-01-preview&mode=foregroundMatting"
    
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream"  # Sending binary data (image file)
    }
    
    # Read the image data from the file
    with open(image_file, "rb") as file:
        image_data = file.read()
    
    # Make a POST request to the Azure API
    response = requests.post(url, headers=headers, data=image_data)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Save the foreground-matted image
        with open("foreground_matted.png", "wb") as file:
            file.write(response.content)
        print("Foreground matted and saved in foreground_matted.png")
    else:
        print(f"Error: {response.status_code}, {response.text}")

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

        # Call the ForegroundMatting function to process the image
        ForegroundMatting(ai_endpoint, ai_key, image_file)

    except Exception as ex:
        # Catch and print any exceptions that occur
        print(f"Error: {ex}")

if __name__ == "__main__":
    main()
