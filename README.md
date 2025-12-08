# 🖼️ Azure AI Vision Projects
This repository contains multiple Python scripts that demonstrate the use of Azure AI Vision services for image analysis, object detection, background removal, and captioning. Each script is designed for a specific task, making it easy to process images programmatically and obtain structured or visual outputs.

## 🏷️ Projects & Scripts Overview
1️⃣ Background Removal (Background.py)

## Purpose: Remove backgrounds from images.
## Functionality:

Uses Azure AI Vision’s background removal API.

Accepts image files and produces a new image with the background removed.

Handles errors and validates environment configuration.

## Outcome:
Generates a clean image with the background removed, saved as background_removed.png.

# 2️⃣ Foreground Matting (Foreground.py)

## Purpose: Extract and isolate the foreground from images.
## Functionality:

Uses Azure AI Vision’s foreground matting API.

Processes images to retain only the subject while separating from the background.

Outputs the result as foreground_matted.png.

# 3️⃣ Image Captioning (Capption.py)

## Purpose: Generate captions and dense descriptions for images.
## Functionality:

Uses Azure AI Vision’s captioning features.

Produces both a main caption and multiple dense captions describing image regions.

Displays captions along with confidence scores.

# 4️⃣ Object Detection (ObjectLocation.py)

## Purpose: Detect objects in images and highlight them visually.
Functionality:

Uses Azure AI Vision object detection.

Annotates detected objects with bounding boxes and labels.

Saves annotated images with detected objects marked (objects.jpg).

# 5️⃣ People Detection (PeopleLocation.py)

## Purpose: Detect and locate people in images.
## Functionality:

Uses Azure AI Vision’s people detection feature.

Draws bounding boxes around detected people and saves annotated images (people.jpg).

Prints detailed information about each detected person.

# 6️⃣ Image Tagging (Tags.py)

Purpose: Assign descriptive tags to images.
Functionality:

Uses Azure AI Vision’s tagging feature.

Produces a list of tags with confidence scores for each image.

Helps categorize and index images for retrieval or analysis.

# 🧰 Key Features

End-to-end image analysis pipelines using Python.

Integration with Azure AI Vision for multiple computer vision tasks.

Automated background removal, foreground matting, and image captioning.

Detection of objects, people, and image tags.

Outputs saved as annotated images or printed analysis results.

Environment-based configuration using .env for secure API keys.

# ⚙️ Technologies Used

Python 3.13

Azure AI Vision Service

PIL (Python Imaging Library) for image processing

Matplotlib for annotated image visualization

Requests library for API calls

Dotenv for configuration management
