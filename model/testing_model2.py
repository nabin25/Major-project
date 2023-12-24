import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_model import HandSignClassifier

# Load the trained model
model = HandSignClassifier(num_classes=36)
model.load_state_dict(torch.load('hand_sign_classifier4.pth'))
model.eval()

# Define the transformation for inference
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Function to check if an image is blurry
def is_blurry(image):
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < 0.4

# Set up video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

# Variables for frame extraction and prediction
frame_interval = 30  
frame_count = 30

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture video.")
        break

    frame_count += 1

    # Extract frame at specified intervals
    if frame_count % frame_interval == 0:
        print(f"Captured frame {frame_count}")

        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess the image
        input_image = transform(pil_image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            model.eval()
            output = model(input_image)

        # Get the predicted class
        predicted_class = torch.argmax(output).item()

        # Map class index to label
        class_mapping = {0: 'ब', 1: 'भ', 2: 'च',3: 'छ', 4: 'स', 5: 'द', 6: 'ड', 7: 'ध', 8:'ढ',9: 'ग', 10: 'घ', 11: 'ज्ञ', 12: 'ह', 13: 'ज', 14: 'झ', 15: 'क', 16: 'ख', 17: 'क्ष', 18: 'ल', 19: 'ष', 20: 'म', 21: 'न', 22: 'ण', 23: 'ङ', 24: 'प', 25: 'फ', 26: 'र', 27: 'श', 28: 'त', 29: 'ट', 30: 'थ', 31: 'ठ', 32: 'त्र', 33: 'व', 34: 'ञ', 35: 'य'} 

        predicted_label = class_mapping[predicted_class]

        print(f"Prediction: {predicted_label}")
cap.release()
