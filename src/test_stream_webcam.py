import cv2
import numpy as np
from PIL import Image
from model import EmotionCNN
import torch
from torchvision import transforms
import os
import tkinter as tk
from tkinter import simpledialog

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Grayscale(),        # Convert to grayscale
    transforms.Resize((64, 64)),    # Resize to 64x64
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image
])

# Emotion target labels
emotion_tab = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# List checkpoint files
checkpoint_dir = 'checkpoints'
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

if len(checkpoint_files) == 0:
    raise FileNotFoundError("No checkpoint files found in the 'checkpoints' directory.")
elif len(checkpoint_files) == 1:
    checkpoint_file = checkpoint_files[0]
else:
    # Create a simple Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Create a list of checkpoint files for the user to choose from
    choice = simpledialog.askinteger("Checkpoint Selection", 
                                     "Select the checkpoint file to use:\n" + 
                                     "\n".join([f"{i}: {file}" for i, file in enumerate(checkpoint_files)]))

    if choice is None or choice < 0 or choice >= len(checkpoint_files):
        raise ValueError("Invalid selection. Please restart and select a valid checkpoint file.")

    checkpoint_file = checkpoint_files[choice]

# Load the model.
model = EmotionCNN().to(device)
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_file), map_location=device))
model.eval()

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Process the first detected face
    for (x, y, w, h) in faces:
        # Crop the frame to the detected face
        face = frame[y:y+h, x:x+w]

        # Convert the cropped face from BGR (OpenCV) to RGB (PIL)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face_rgb)

        # Transform the face
        transformed_face = transform(pil_face).unsqueeze(0)  # Add batch dimension

        # Predict the emotion
        with torch.no_grad():
            output = model(transformed_face)
            _, predicted = torch.max(output.data, 1)
            emotion = emotion_tab[predicted.item()]
        # Display the emotion label on the original frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show the frame with the detected face and emotion label
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()