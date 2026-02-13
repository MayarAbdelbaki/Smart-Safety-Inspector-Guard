"""
SSIG — Autocar edge client.

Runs on the autocar (or any device with camera on the same network as the PC server).
- Captures video, detects faces (OpenCV Haar cascade), and sends images to the PC server.
- Listens for identification/compliance results from the PC and displays them.
- Optional text-to-speech (pyttsx3) to announce person name or "unknown person".

Configure PC_IP and PC_WEBHOOK_PORT (environment or code) to point to the machine
running pc_server.py. Run: python3 autocar_main.py
"""

import os
import sys
import warnings

# Load .env if python-dotenv is installed (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Suppress GTK warnings BEFORE importing other modules
os.environ['GTK_WARNING'] = '0'
os.environ['G_MESSAGES_DEBUG'] = '0'
os.environ['NO_AT_BRIDGE'] = '1'  # Disable accessibility bridge warnings

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*GTK.*")
warnings.filterwarnings("ignore", message=".*Gtk.*")
warnings.filterwarnings("ignore", message=".*theme.*")
warnings.filterwarnings("ignore", message=".*murrine.*")

# Redirect stderr to filter GTK warnings
class GTKWarningFilter:
    """Filter to suppress GTK warnings from stderr."""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, message):
        # Filter out GTK warnings
        if 'Gtk-WARNING' in message or 'GTK-WARNING' in message:
            if 'murrine' in message or 'theme' in message:
                return  # Suppress these warnings
        self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()

# Apply filter to stderr
sys.stderr = GTKWarningFilter(sys.stderr)

import cv2
import time
import requests
import base64
import threading
import http.server
import socketserver
import json
from datetime import datetime
from pop import Util

# Try to import pyttsx3 - handle Python version incompatibility gracefully
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except (ImportError, SyntaxError) as e:
    # pyttsx3 requires Python 3.7+ (uses 'from __future__ import annotations')
    # If running on Python 3.6 or earlier, TTS will be disabled
    PYTTSX3_AVAILABLE = False
    pyttsx3 = None
    print(f"⚠ Warning: pyttsx3 not available (requires Python 3.7+): {str(e)}")
    print(f"  TTS functionality will be disabled. Upgrade to Python 3.7+ to enable TTS.")

# ==================== CONFIGURATION ====================
# PC Configuration (set PC_IP in environment or edit here for your network)
PC_IP = os.getenv("PC_IP", "YOUR_PC_IP_HERE")  # e.g. 192.168.1.100
PC_WEBHOOK_PORT = int(os.getenv("PC_WEBHOOK_PORT", "5000"))
PC_WEBHOOK_URL = f"http://{PC_IP}:{PC_WEBHOOK_PORT}/webhook"

# Message Receiver Configuration
MESSAGE_PORT = 5001  # Port for receiving messages from PC
MESSAGE_HOST = "0.0.0.0"  # Listen on all interfaces

# Face Detection Configuration
SAVE_FOLDER = "captured_faces"  # Images saved here
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAPTURE_INTERVAL = 5  # Wait 5 seconds between captures

# ==================== TEXT TO SPEECH ====================
def init_tts_engine():
    """Initialize text-to-speech engine."""
    if not PYTTSX3_AVAILABLE:
        return None
    try:
        engine = pyttsx3.init()
        # Set speech rate (optional - adjust as needed)
        engine.setProperty('rate', 150)  # Speed of speech
        return engine
    except Exception as e:
        print(f"⚠ Warning: Could not initialize TTS engine: {str(e)}")
        print(f"  TTS will be disabled. Install pyttsx3: pip3 install pyttsx3")
        return None

# TTS availability flag (we'll create engine instances as needed)
TTS_AVAILABLE = PYTTSX3_AVAILABLE
if PYTTSX3_AVAILABLE:
    try:
        # Test if TTS is available
        test_engine = pyttsx3.init()
        test_engine = None  # Clean up test
    except:
        TTS_AVAILABLE = False

def speak_message(message):
    """Convert message to speech if it contains person information."""
    if not TTS_AVAILABLE or not PYTTSX3_AVAILABLE:
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
            # Create a new engine instance each time to ensure it works every time
            def speak():
                try:
                    # Create new engine instance for each message to avoid getting stuck
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)  # Speed of speech
                    engine.say(speech_text)
                    engine.runAndWait()
                    # Engine will be cleaned up automatically after use
                except Exception as e:
                    print(f"⚠ Error in TTS: {str(e)}")
            
            # Run TTS in background thread
            tts_thread = threading.Thread(target=speak, daemon=True)
            tts_thread.start()
    except Exception as e:
        print(f"⚠ Error processing TTS: {str(e)}")


# ==================== MESSAGE RECEIVER ====================
class MessageHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for receiving messages from PC."""
    
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
            response = {'status': 'healthy', 'service': 'Message Receiver'}
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
            print(f"Waiting for messages from PC...")
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
        self.webhook_url = PC_WEBHOOK_URL
        
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
        """Initialize camera using hardware-specific settings."""
        Util.enable_imshow()
        cam = Util.gstrmer(width=self.width, height=self.height)
        self.camera = cv2.VideoCapture(cam, cv2.CAP_GSTREAMER)
        
        if not self.camera.isOpened():
            raise Exception("Camera not found or could not be opened")
        
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera initialized: {actual_width}x{actual_height}")
    
    def _init_face_detector(self):
        """Initialize Haar Cascade face detector."""
        haar_face = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_face)
        
        if self.face_cascade.empty():
            raise Exception("Failed to load face cascade classifier")
        
        print("Face detector initialized")
    
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
        """Send image to PC webhook server."""
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
                'timestamp': time.time()
            }
            
            print(f"  Sending image to PC: {self.webhook_url}")
            
            # Send POST request to webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"  ✓ Image sent to PC successfully")
                return True
            else:
                print(f"  ✗ Webhook request failed with status {response.status_code}")
                print(f"     Response: {response.text[:200]}")
                return False
                
        except requests.exceptions.ConnectTimeout:
            print(f"✗ Connection timeout: Cannot reach PC at {self.webhook_url}")
            print(f"  Please check:")
            print(f"  1. PC server is running (python pc_server.py)")
            print(f"  2. PC IP address is correct: {PC_IP}")
            print(f"  3. PC and autocar are on the same network")
            print(f"  4. Firewall allows connections on port {PC_WEBHOOK_PORT}")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"✗ Connection error: Cannot connect to PC at {self.webhook_url}")
            print(f"  Error: {str(e)}")
            print(f"  Please check:")
            print(f"  1. PC server is running")
            print(f"  2. PC IP address is correct")
            print(f"  3. Test connection: ping {PC_IP} or curl http://{PC_IP}:{PC_WEBHOOK_PORT}/health")
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
        print(f"PC webhook: {self.webhook_url}")
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
                    
                    cv2.imshow("Face Capture System", display_frame)
                    
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
def main():
    """Main entry point."""
    print("=" * 60)
    print("Autocar Main Program - All-in-One")
    print("=" * 60)
    if "YOUR_PC_IP" in PC_IP or not PC_IP or PC_IP.strip() == "":
        print("⚠️  Configure PC_IP (environment variable or in code) to your PC server IP.")
        print("    Example: set PC_IP=192.168.1.100  (Windows) or export PC_IP=192.168.1.100  (Linux)")
        print()
    # Start message receiver in background
    print("Starting message receiver...")
    message_thread = threading.Thread(target=start_message_receiver, daemon=True)
    message_thread.start()
    print("✓ Message receiver started (running in background)")
    print(f"  Listening on port {MESSAGE_PORT} for messages from PC")
    print(f"  PC will send messages to: http://<AUTOCAR_IP>:{MESSAGE_PORT}/message")
    if TTS_AVAILABLE:
        print("  ✓ Text-to-speech enabled - will announce person detection")
    else:
        print("  ⚠ Text-to-speech disabled - install pyttsx3 to enable")
    print()
    
    # Initialize and run face capture
    print("Initializing face detection system...")
    face_capture = FaceCapture()
    print()
    
    face_capture.run(show_preview=True)


if __name__ == "__main__":
    main()
