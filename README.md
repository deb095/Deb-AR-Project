# Deb AR Project – Emotion-Based 3D Face Visualisation

## Overview
This project is a computer vision and 3D visualisation system developed as part of an academic coursework project.  
It combines **face verification**, **emotion detection**, and **interactive 3D rendering** using Python-based tools.

The system verifies a user’s identity, detects facial emotions in real time using a webcam, captures happy expressions, and visualises them as a movable 3D face.

---

## Features
- Face verification using DeepFace (VGG-Face)
- Real-time emotion detection from webcam input
- Automatic capture of happy facial expressions
- Interactive 3D face visualisation with rotation and movement controls
- Keyboard and mouse-based interaction
- Modular design with separate face detection and emotion detection scripts

---

## Technologies Used
- **Python 3**
- **OpenCV**
- **DeepFace**
- **NumPy**
- **Threading**
- **Unity (for texture integration and 3D experimentation)**

---

## Project Structure
Deb-AR-Project/
│
├── faceDetection.py # Identity verification system
├── emotionDetection.py # Emotion detection + 3D viewer
├── reference.jpg # Reference image for face verification
├── happy_capture.jpg # Latest captured happy face
├── README.md # Project documentation
└── screenshots/ # Output images and visual results


---

## How It Works
1. The user’s identity is verified using a reference image.
2. Once verified, the emotion detection system starts automatically.
3. Facial emotions are detected in real time using the webcam.
4. When a happy expression is detected, the face is saved.
5. The saved face can be rendered as an interactive 3D object.
6. The 3D face can be rotated, moved, and zoomed using keyboard controls.

---

## Controls

### Emotion Detection Window
- **Q** – Quit application  
- **B** – Open 3D face viewer  
- **Mouse Click** – Click “Create 3D Face” button

### 3D Face Viewer
- **W / A / S / D** – Move face
- **J / L** – Rotate face
- **U / I** – Zoom out / in
- **R** – Reset view
- **Q** – Close viewer

---

## Author
**Debarghya Pal**  
Ravensbourne University London  
Computer Vision & 3D Coursework (2024/2025)

---

## License
This project is licensed under the MIT License.

⚠️ Note:
Reference images and generated face captures are excluded from the repository
for privacy reasons. Users must provide their own reference image locally.

