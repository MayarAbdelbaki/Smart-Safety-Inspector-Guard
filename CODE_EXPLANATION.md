# Code Explanation

## One-Sentence Summary

**The autocar continuously detects human faces using its camera, captures images when faces are found, sends those images to the PC server via HTTP, the PC processes them through Hugging Face AI to identify the person, and sends the result (like "Person: unknown person") back to the autocar which displays it in the terminal.**

---

## Detailed Explanation

### autocar_main.py (Runs on Autocar)

**What it does:**
- Uses the camera to continuously capture video frames
- Detects human faces in each frame using OpenCV's Haar Cascade classifier
- When a face is detected, saves the image locally and sends it to the PC via HTTP POST request
- Runs a background message receiver server that listens for messages from the PC
- Displays received messages (like person identification results) in the terminal

**Key Components:**
1. **FaceCapture class**: Handles camera initialization, face detection, image capture, and sending images to PC
2. **MessageHandler class**: HTTP server that receives messages from PC and displays them
3. **Message receiver thread**: Runs in background to receive messages without blocking face detection

---

### pc_server.py (Runs on PC)

**What it does:**
- Runs a Flask web server that receives images from the autocar via HTTP POST
- Saves received images to a local folder
- Sends images to Hugging Face Space API for person identification using Gradio client
- Parses the AI response to extract person information (e.g., "Person: unknown person")
- Sends the identification result back to the autocar via HTTP POST

**Key Components:**
1. **Flask web server**: Receives images on port 5000
2. **Hugging Face integration**: Uses Gradio client to send images and get AI results
3. **Message parser**: Extracts person name from AI response
4. **Autocar communication**: Sends results back to autocar on port 5001

---

## Complete Flow

1. **Autocar detects face** → Camera captures video, OpenCV detects faces
2. **Autocar captures image** → Saves to `captured_faces/` folder
3. **Autocar sends to PC** → HTTP POST with base64-encoded image to `http://<PC_IP>:5000/webhook`
4. **PC receives image** → Flask server receives, decodes, saves to `received_images/`
5. **PC sends to Hugging Face** → Gradio client sends image to your Space API
6. **Hugging Face returns** → AI returns person name, PPE status, and optionally fall detection
7. **PC sends result to autocar** → HTTP POST to `http://<AUTOCAR_IP>:5001/message`
8. **Autocar displays** → Message receiver shows "Person: unknown person" in terminal

---

## Technical Details

**Network Communication:**
- Autocar → PC: Port 5000 (images)
- PC → Autocar: Port 5001 (messages)
- Both use HTTP POST with JSON payloads

**Image Processing:**
- Images encoded as base64 for HTTP transmission
- OpenCV used for face detection (Haar Cascade)
- Hugging Face AI used for person identification

**Threading:**
- Autocar uses threading to run message receiver in background
- Allows face detection and message receiving simultaneously

**Error Handling:**
- Connection timeouts handled gracefully
- Clear error messages guide troubleshooting
- System continues even if one component fails
