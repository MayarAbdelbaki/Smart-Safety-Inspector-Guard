"""
SSIG — PC client (webcam as edge device).

Optional component: runs on a PC with a webcam when an autocar is not used.
- Captures video, detects faces (OpenCV Haar cascade), sends images to the main PC server.
- Receives identification/compliance results from the server and optionally uses TTS.

Set MAIN_PC_IP (and MAIN_PC_WEBHOOK_PORT if needed) to the IP of the machine running
pc_server.py. Run: python pc_client.py
"""

import os
import sys
# Load .env if python-dotenv is installed (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['GTK_WARNING'] = '0'

import cv2
import time
import requests
import base64
import threading
import http.server
import socketserver
import json
from datetime import datetime
import pyttsx3

# ==================== CONFIGURATION ====================
# Main PC Server Configuration (set in environment or edit for your network)
MAIN_PC_IP = os.getenv("MAIN_PC_IP", "YOUR_MAIN_PC_IP_HERE")  # e.g. 192.168.1.100
MAIN_PC_WEBHOOK_PORT = int(os.getenv("MAIN_PC_WEBHOOK_PORT", "5000"))
MAIN_PC_WEBHOOK_URL = f"http://{MAIN_PC_IP}:{MAIN_PC_WEBHOOK_PORT}/webhook"

# Message Receiver Configuration
MESSAGE_PORT = 5002  # Port for receiving messages from main PC (different from autocar)
MESSAGE_HOST = "0.0.0.0"  # Listen on all interfaces

# Face Detection Configuration
SAVE_FOLDER = "captured_faces"  # Images saved here
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAPTURE_INTERVAL = 5  # Wait 5 seconds between captures
CAMERA_INDEX = 0  # Webcam index (0 = default camera)

# ==================== TEXT TO SPEECH ====================
def init_tts_engine():
    """Initialize text-to-speech engine."""
    try:
        engine = pyttsx3.init()
        # Set speech rate (optional - adjust as needed)
        engine.setProperty('rate', 150)  # Speed of speech
        return engine
    except Exception as e:
        print(f"⚠ Warning: Could not initialize TTS engine: {str(e)}")
        print(f"  TTS will be disabled. Install pyttsx3: pip install pyttsx3")
        return None

# TTS availability flag (we'll create engine instances as needed)
TTS_AVAILABLE = True
try:
    # Test if TTS is available
    test_engine = pyttsx3.init()
    test_engine = None  # Clean up test
except:
    TTS_AVAILABLE = False

def speak_message(message):
    """Convert message to speech if it contains person information."""
    if not TTS_AVAILABLE:
        return
    
    try:
        # Check if message contains "Person:" information
        if "Person:" in message or "person" in message.lower():
            # Extract person name or use default message
            if "unknown person" in message.lower():
                speech_text = "There is unknown person"
            elif "Person:" in message:
                # Extract person name
                parts = message.split("Person:", 1)
                if len(parts) > 1:
                    person_name = parts[1].strip()
                    if person_name:
                        speech_text = f"There is {person_name}"
                    else:
                        speech_text = "There is unknown person"
                else:
                    speech_text = "There is unknown person"
            else:
                speech_text = "There is unknown person"
            
            # Speak the message in a separate thread to avoid blocking
            def speak():
                try:
                    # Create a new engine instance each time to avoid getting stuck
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say(speech_text)
                    engine.runAndWait()
                    # Engine will be cleaned up automatically
                except Exception as e:
                    print(f"⚠ Error in TTS: {str(e)}")
            
            # Run TTS in background thread
            tts_thread = threading.Thread(target=speak, daemon=True)
            tts_thread.start()
    except Exception as e:
        print(f"⚠ Error processing TTS: {str(e)}")


# ==================== MESSAGE RECEIVER ====================
class MessageHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for receiving messages from main PC."""
    
    def do_POST(self):
        """Handle POST requests with messages."""
        if self.path == '/message':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                message = data.get('message', '')
                timestamp = data.get('timestamp', '')
                source = data.get('source', 'PC')
                
                # Display message in terminal
                if timestamp:
                    try:
                        dt = datetime.fromtimestamp(timestamp)
                        time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        time_str = str(timestamp)
                else:
                    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"\n[{time_str}] Message from {source}:")
                print(f"  {message}\n")
                
                # Convert to speech if it's a person detection message
                if "Person:" in message:
                    speak_message(message)
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'status': 'success', 'message': 'Message received'}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                print(f"✗ Error processing message: {str(e)}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'status': 'error', 'message': str(e)}
                self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        """Handle GET requests (health check)."""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'healthy', 'service': 'PC Client Message Receiver'}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Override to reduce log noise."""
        if args[0].startswith('"GET') or args[0].startswith('"POST'):
            return
        super().log_message(format, *args)


def start_message_receiver():
    """Start the message receiver server in a separate thread."""
    try:
        with socketserver.TCPServer((MESSAGE_HOST, MESSAGE_PORT), MessageHandler) as httpd:
            print(f"Message receiver started on port {MESSAGE_PORT}")
            print(f"Waiting for messages from main PC...")
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"✗ Port {MESSAGE_PORT} is already in use")
            print(f"  Another instance might be running, or port is occupied")
        else:
            print(f"✗ Error starting message receiver: {str(e)}")
    except KeyboardInterrupt:
        print("\nMessage receiver stopped")


# ==================== FACE DETECTION ====================
class FaceCapture:
    def __init__(self):
        """Initialize the face capture system."""
        self.save_folder = SAVE_FOLDER
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT
        self.image_counter = 1
        self.last_capture_time = 0
        self.capture_interval = CAPTURE_INTERVAL
        self.webhook_url = MAIN_PC_WEBHOOK_URL
        
        # Create save folder if it doesn't exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            print(f"Created folder: {self.save_folder}")
        else:
            self._update_image_counter()
        
        # Initialize camera
        self.camera = None
        self._init_camera()
        
        # Initialize face detection
        self.face_cascade = None
        self._init_face_detector()
    
    def _init_camera(self):
        """Initialize camera using standard webcam."""
        # Try to open camera
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        
        if not self.camera.isOpened():
            raise Exception("Camera not found or could not be opened")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera initialized: {actual_width}x{actual_height}")
    
    def _init_face_detector(self):
        """Initialize Haar Cascade face detector."""
        # Try different possible paths for Haar Cascade
        possible_paths = [
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',  # OpenCV default
        ]
        
        for haar_path in possible_paths:
            if os.path.exists(haar_path):
                self.face_cascade = cv2.CascadeClassifier(haar_path)
                if not self.face_cascade.empty():
                    print(f"Face detector initialized from: {haar_path}")
                    return
        
        # If no file found, try using OpenCV's default
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            if not self.face_cascade.empty():
                print("Face detector initialized (using OpenCV default)")
                return
        except:
            pass
        
        raise Exception("Failed to load face cascade classifier. Please check OpenCV installation.")
    
    def _update_image_counter(self):
        """Update image counter based on existing files."""
        existing_files = [f for f in os.listdir(self.save_folder) if f.endswith('.jpg')]
        if existing_files:
            numbers = []
            for f in existing_files:
                try:
                    num = int(f.split('.')[0])
                    numbers.append(num)
                except ValueError:
                    continue
            if numbers:
                self.image_counter = max(numbers) + 1
    
    def detect_face(self, frame):
        """Detect faces in the given frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        return len(faces) > 0
    
    def _send_to_webhook(self, image_path):
        """Send image to main PC webhook server."""
        if not self.webhook_url:
            print("  ⚠ Webhook URL not configured, skipping send")
            return False
        
        try:
            # Read image file and encode to base64
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Prepare payload
            payload = {
                'image': img_base64,
                'filename': os.path.basename(image_path),
                'timestamp': time.time(),
                'source': 'PC_CLIENT'  # Identify this as coming from PC client
            }
            
            print(f"  Sending image to main PC: {self.webhook_url}")
            
            # Send POST request to webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"  ✓ Image sent to main PC successfully")
                return True
            else:
                print(f"  ✗ Webhook request failed with status {response.status_code}")
                print(f"     Response: {response.text[:200]}")
                return False
                
        except requests.exceptions.ConnectTimeout:
            print(f"✗ Connection timeout: Cannot reach main PC at {self.webhook_url}")
            print(f"  Please check:")
            print(f"  1. Main PC server is running (python pc_server.py)")
            print(f"  2. Main PC IP address is correct: {MAIN_PC_IP}")
            print(f"  3. Both PCs are on the same network")
            print(f"  4. Firewall allows connections on port {MAIN_PC_WEBHOOK_PORT}")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"✗ Connection error: Cannot connect to main PC at {self.webhook_url}")
            print(f"  Error: {str(e)}")
            print(f"  Please check:")
            print(f"  1. Main PC server is running")
            print(f"  2. Main PC IP address is correct")
            print(f"  3. Test connection: ping {MAIN_PC_IP} or curl http://{MAIN_PC_IP}:{MAIN_PC_WEBHOOK_PORT}/health")
            return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Error sending to webhook: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error sending to webhook: {str(e)}")
            return False
    
    def capture_image(self, frame):
        """Capture and save the image."""
        current_time = time.time()
        
        # Check if enough time has passed since last capture
        if current_time - self.last_capture_time < self.capture_interval:
            return None
        
        # Save the original frame
        filename = f"{self.image_counter}.jpg"
        filepath = os.path.join(self.save_folder, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"Face captured and saved: {filepath}")
        
        # Send to webhook if configured
        if self.webhook_url:
            self._send_to_webhook(filepath)
        
        self.image_counter += 1
        self.last_capture_time = current_time
        
        return filepath
    
    def run(self, show_preview=True):
        """Main loop to continuously detect faces and capture images."""
        print("Starting face detection and capture system...")
        print(f"Images will be saved to: {self.save_folder}")
        print(f"Capture interval: {self.capture_interval} seconds")
        print(f"Main PC webhook: {self.webhook_url}")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.camera.read()
                
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Detect face in the frame
                face_detected = self.detect_face(frame)
                
                # If face detected, capture the image
                if face_detected:
                    saved_path = self.capture_image(frame)
                    if saved_path:
                        print(f"✓ Face detected and captured!")
                
                # Show preview if enabled
                if show_preview:
                    display_frame = frame.copy()
                    status_text = "Face Detected!" if face_detected else "No Face"
                    cv2.putText(display_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if face_detected else (0, 0, 255), 2)
                    
                    time_until_next = max(0, self.capture_interval - (time.time() - self.last_capture_time))
                    if face_detected and time_until_next > 0:
                        countdown_text = f"Next capture in: {time_until_next:.1f}s"
                        cv2.putText(display_frame, countdown_text, (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    cv2.imshow("PC Client - Face Capture System", display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quit requested by user")
                        break
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources and close windows."""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        print(f"System stopped. Total images captured: {self.image_counter - 1}")


# ==================== MAIN ====================
def test_main_pc_connection():
    """Test if main PC server is reachable before starting."""
    print("Testing connection to main PC...")
    print(f"  Main PC IP: {MAIN_PC_IP}")
    print(f"  Main PC Port: {MAIN_PC_WEBHOOK_PORT}")
    
    # Test HTTP connection
    try:
        response = requests.get(f"http://{MAIN_PC_IP}:{MAIN_PC_WEBHOOK_PORT}/health", timeout=5)
        if response.status_code == 200:
            print("  ✓ Main PC server is reachable and responding")
            return True
        else:
            print(f"  ✗ Main PC server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectTimeout:
        print("  ✗ Connection timeout - Main PC server is not responding")
        print("     Make sure pc_server.py is running on main PC!")
        return False
    except requests.exceptions.ConnectionError:
        print("  ✗ Connection error - Cannot reach main PC")
        print("     Check:")
        print(f"     1. Main PC server is running: python pc_server.py")
        print(f"     2. Main PC IP is correct: {MAIN_PC_IP}")
        print(f"     3. Both PCs are on same network")
        print(f"     4. Firewall allows port {MAIN_PC_WEBHOOK_PORT}")
        return False
    except Exception as e:
        print(f"  ✗ Error testing connection: {str(e)}")
        return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("PC Client - Face Detection and Communication")
    print("=" * 60)
    print()
    
    # Test main PC connection first
    if not test_main_pc_connection():
        print()
        print("⚠ WARNING: Cannot connect to main PC server!")
        print("  The program will continue, but images won't be sent.")
        print("  Fix the connection and restart for full functionality.")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
        print()
    
    # Start message receiver in background
    print("Starting message receiver...")
    message_thread = threading.Thread(target=start_message_receiver, daemon=True)
    message_thread.start()
    time.sleep(0.5)  # Give it a moment to start
    print("✓ Message receiver started (running in background)")
    print(f"  Listening on port {MESSAGE_PORT} for messages from main PC")
    print(f"  Main PC will send messages to: http://<THIS_PC_IP>:{MESSAGE_PORT}/message")
    if TTS_AVAILABLE:
        print("  ✓ Text-to-speech enabled - will announce person detection")
    else:
        print("  ⚠ Text-to-speech disabled - install pyttsx3 to enable")
    print()
    
    # Initialize and run face capture
    print("Initializing face detection system...")
    try:
        face_capture = FaceCapture()
        print()
        face_capture.run(show_preview=True)
    except Exception as e:
        print(f"✗ Error initializing face capture: {str(e)}")
        print("  Check camera connection and permissions")


if __name__ == "__main__":
    main()
