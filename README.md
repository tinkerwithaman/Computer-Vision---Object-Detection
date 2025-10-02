# Object Detection with a Pre-trained Model

## Description
This project demonstrates how to use a pre-trained deep learning model for object detection. It uses the **Faster R-CNN model with a ResNet-50 backbone**, provided by `torchvision`, which has been pre-trained on the COCO dataset.

The script loads an image from a specified URL, preprocesses it, runs it through the model, and then draws bounding boxes and labels for the detected objects. The final annotated image is saved to the `assets` folder. This showcases the power of transfer learning for complex computer vision tasks.

## Features
-   Uses a state-of-the-art pre-trained Faster R-CNN model.
-   Detects 91 common object classes from the COCO dataset.
-   Loads image directly from a URL for easy testing.
-   Visualizes results by drawing bounding boxes and labels on the image.

## Setup and Installation

1.  **Clone the repository and navigate to the directory.**
2.  **Create a virtual environment and activate it.**
3.  **Install the dependencies:** `pip install -r requirements.txt`
4.  **Run the script:** `python src/main.py`

## Example Output
```
Model loaded.
Detecting objects in image from URL...
Detected 3 objects.
Saving output image to: assets/object_detection_output.jpg
Done.
```
*(An image named `object_detection_output.jpg` will be saved in the `assets` folder with bounding boxes drawn around detected objects like people, cars, etc.)*
