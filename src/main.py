import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np

# COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def load_model():
    """Load a pre-trained Faster R-CNN model."""
    # Using a model with a ResNet-50 backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    print("Model loaded.")
    return model

def detect_objects(model, image_url, threshold=0.7):
    """
    Detects objects in an image from a URL.
    
    Args:
        model: The pre-trained object detection model.
        image_url (str): The URL of the image to process.
        threshold (float): The confidence threshold to filter detections.

    Returns:
        tuple: The original image and the list of predictions.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        return None, None

    # Convert image to tensor
    img_tensor = F.to_tensor(img)
    
    # Perform detection
    with torch.no_grad():
        predictions = model([img_tensor])

    # Filter predictions based on the score
    pred = predictions[0]
    pred_boxes = pred['boxes'][pred['scores'] > threshold].cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred['labels'][pred['scores'] > threshold].cpu().numpy())]
    
    return img, {'boxes': pred_boxes, 'labels': pred_class}

def draw_boxes(image, predictions):
    """
    Draws bounding boxes on the image.

    Args:
        image (PIL.Image): The original image.
        predictions (dict): A dictionary with 'boxes' and 'labels'.

    Returns:
        PIL.Image: The image with bounding boxes drawn.
    """
    draw = ImageDraw.Draw(image)
    try:
        # Load a font
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        # If arial.ttf is not found, use the default font
        font = ImageFont.load_default()

    for i in range(len(predictions['boxes'])):
        box = predictions['boxes'][i]
        label = predictions['labels'][i]
        
        # Draw bounding box
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
        
        # Draw label
        text_size = font.getbbox(label)
        text_position = (box[0], box[1] - text_size[3] - 2)
        draw.rectangle([text_position, (text_position[0] + text_size[2], text_position[1] + text_size[3])], fill="red")
        draw.text(text_position, label, fill="white", font=font)
        
    return image

if __name__ == "__main__":
    # URL of an image with objects to detect
    IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg" # A photo of cats and a remote
    
    model = load_model()
    
    print(f"Detecting objects in image from URL: {IMAGE_URL}")
    original_image, preds = detect_objects(model, IMAGE_URL)
    
    if original_image and preds:
        print(f"Detected {len(preds['boxes'])} objects.")
        
        # Draw the boxes on the image
        output_image = draw_boxes(original_image, preds)
        
        # Save the output image
        output_path = "assets/object_detection_output.jpg"
        output_image.save(output_path)
        print(f"Saving output image to: {output_path}")
        print("Done.")

        # Optionally, display the image
        # output_image.show()
