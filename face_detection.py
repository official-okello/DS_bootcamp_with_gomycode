# Face Detection using Viola-Jones Algorithm
# This script uses OpenCV to detect faces from the webcam using the Viola-Jones algorithm.

import os
os.system("pip install opencv-python-headless")
import cv2
import streamlit as st

# Load the face cascade classifier from the XML file
face_cascade = cv2.CascadeClassifier('C:/Users/USER/OneDrive/Desktop/DS_GOMYCODE/ML/scripts/XML/haarcascade_frontalface_default .xml')

# Ensure the XML file path is correct and accessible
if not face_cascade.empty():
    st.success("Face cascade classifier loaded successfully.")
else:
    st.error("Failed to load face cascade classifier. Please check the XML file path.")

# Instructions for the user
st.markdown("""
### How to Use:
1. Click 'Detect Faces' to start your webcam.
2. Adjust detection parameters for better accuracy.
3. Choose rectangle color for detected faces.
4. Save the processed image if needed.
5. Press 'q' to exit face detection.
""")

# User adjustments
color_hex = st.color_picker("Choose rectangle color", "#00FF00")
color = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

scale_factor = st.slider("Scale Factor", min_value=1.1, max_value=1.5, step=0.05, value=1.3)
min_neighbors = st.slider("Min Neighbors", min_value=3, max_value=10, value=5)

# Function to capture frames from webcam and detect faces using the Viola-Jones algorithm
def detect_faces():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        st.error("Error: Could not open webcam. Please ensure your webcam is connected and accessible.")
        return
    st.write("Webcam is open. Press 'q' to stop the detection.")

    # Loop to continuously capture frames from the webcam
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()

        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces using the face cascade classifier with user-defined parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Draw rectangles around the detected faces using the chosen color
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Save the processed image if requested
    if st.button("Save Image"):
        cv2.imwrite("detected_faces.jpg", frame)
        st.success("Image saved successfully!")

# Streamlit application for face detection
def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")

    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        detect_faces()

if __name__ == "__main__":
    app()
