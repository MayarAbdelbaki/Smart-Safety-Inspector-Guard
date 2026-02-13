"""
SSIG — PC relay server.

Runs on a PC on the same network as the autocar (and optionally a PC client).
- Exposes /webhook: receives images from autocar or PC client (HTTP POST, base64 or file).
- Sends each image to the Hugging Face Space (Gradio API) for PPE + face + fall detection.
- Parses the API response and forwards the result (e.g. person name, compliance) to the
  autocar and/or PC client via HTTP POST.

Configure AUTOCAR_IP, HUGGINGFACE_SPACE_URL (and optionally PC_CLIENT_IP) via
environment variables or a .env file (use python-dotenv). Run: python pc_server.py
"""

from flask import Flask, request, jsonify
import base64
import os
import time
import requests
import tempfile
import re
import json
import threading
from datetime import datetime
from gradio_client import Client, file
from io import BytesIO

# Load .env if python-dotenv is installed (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import PIL for image optimization
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠ Warning: PIL/Pillow not available. Image optimization disabled.")
    print("  Install with: pip install Pillow")

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

app = Flask(__name__)

# ==================== CONFIGURATION ====================
# Network Configuration
PC_PORT = 5000
HOST = "0.0.0.0"  # Listen on all interfaces

# Autocar Configuration (set via environment variables for your network)
AUTOCAR_IP = os.getenv("AUTOCAR_IP", "YOUR_AUTOCAR_IP_HERE")  # e.g. 192.168.1.101
AUTOCAR_MESSAGE_PORT = int(os.getenv("AUTOCAR_MESSAGE_PORT", "5001"))

# PC Client Configuration (optional - for additional PC with webcam)
PC_CLIENT_IP = os.getenv("PC_CLIENT_IP", "YOUR_PC_CLIENT_IP_HERE")
PC_CLIENT_MESSAGE_PORT = int(os.getenv("PC_CLIENT_MESSAGE_PORT", "5002"))

# Hugging Face Configuration (your deployed Space URL or API endpoint)
HUGGINGFACE_SPACE_URL = os.getenv(
    "HUGGINGFACE_SPACE_URL",
    "https://YOUR-USERNAME-YOUR-SPACE.hf.space"
)
ENABLE_HUGGINGFACE = True  # Set to False to disable HF API calls
HF_TIMEOUT = 120  # Timeout in seconds for HF API calls (increase if space is slow to wake up)
HF_MAX_IMAGE_SIZE = 1920  # Maximum image dimension (resize if larger to speed up upload)

# Person ID Mapping (customize with your Members folder names and IDs)
# Keys must match filenames in huggingface/Members/ (without extension)
PERSON_ID_MAPPING = {
    "Amr": "1010",
    "Mariam": "1011",
    "Mayar": "1012",
    "Ashraf": "1013",
    "Ayman": "1014",
    "Badrawy": "1015"
}  # Add or edit entries to match your reference faces

# Folders
RECEIVED_IMAGES_FOLDER = "received_images"  # Images saved here

# ==================== INITIALIZATION ====================
HF_CLIENT = None  # Will be initialized on first use

# TTS availability flag (we'll create engine instances as needed)
TTS_AVAILABLE = PYTTSX3_AVAILABLE
if PYTTSX3_AVAILABLE:
    try:
        # Test if TTS is available
        test_engine = pyttsx3.init()
        test_engine = None  # Clean up test
    except:
        TTS_AVAILABLE = False

# Create folder for received images
if not os.path.exists(RECEIVED_IMAGES_FOLDER):
    os.makedirs(RECEIVED_IMAGES_FOLDER)
    print(f"Created folder: {RECEIVED_IMAGES_FOLDER}")


# ==================== HELPER FUNCTIONS ====================
def get_next_image_number():
    """Get the next image number based on existing files."""
    existing_files = [f for f in os.listdir(RECEIVED_IMAGES_FOLDER) if f.endswith('.jpg')]
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        try:
            num = int(f.split('.')[0])
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1


def optimize_image_size(image_data, max_size=HF_MAX_IMAGE_SIZE):
    """Resize image if it's too large to speed up upload."""
    if not PIL_AVAILABLE:
        return image_data
    
    try:
        # Open image
        img = Image.open(BytesIO(image_data))
        width, height = img.size
        
        # Check if resizing is needed
        if width <= max_size and height <= max_size:
            return image_data
        
        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        # Resize image
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert back to bytes
        output = BytesIO()
        img_resized.save(output, format='JPEG', quality=85, optimize=True)
        optimized_data = output.getvalue()
        
        original_size = len(image_data) / 1024
        optimized_size = len(optimized_data) / 1024
        reduction = ((original_size - optimized_size) / original_size) * 100
        
        print(f"  📐 Image optimized: {width}x{height} → {new_width}x{new_height}")
        print(f"  📦 Size: {original_size:.1f} KB → {optimized_size:.1f} KB ({reduction:.1f}% reduction)")
        
        return optimized_data
    except Exception as e:
        print(f"  ⚠ Could not optimize image: {str(e)}, using original")
        return image_data

def get_hf_client():
    """Initialize and return Hugging Face Gradio client."""
    global HF_CLIENT
    
    if HF_CLIENT is None:
        try:
            print(f"  Initializing Hugging Face client...")
            # Initialize with timeout configuration
            HF_CLIENT = Client(HUGGINGFACE_SPACE_URL)
            print(f"  ✓ Hugging Face client initialized")
            print(f"  ⚙️  Timeout: {HF_TIMEOUT}s, Max image size: {HF_MAX_IMAGE_SIZE}px")
        except Exception as e:
            print(f"  ✗ Failed to initialize Hugging Face client: {str(e)}")
            return None
    
    return HF_CLIENT

def extract_json_from_hf_response(hf_response):
    """Extract and parse JSON from Hugging Face response.
    
    The Gradio client returns a tuple/list: [annotated_image, json_string]
    We need to extract the JSON string (second element) and parse it.
    """
    if not hf_response:
        return None
    
    try:
        # Gradio client returns a tuple/list with [image, json_string]
        if isinstance(hf_response, (list, tuple)) and len(hf_response) >= 2:
            json_string = hf_response[1]  # Second element is the JSON string
            
            # If it's already a string, parse it
            if isinstance(json_string, str):
                try:
                    return json.loads(json_string)
                except json.JSONDecodeError:
                    # If parsing fails, return the string as-is
                    return json_string
            # If it's already a dict/list, return it directly
            elif isinstance(json_string, (dict, list)):
                return json_string
            else:
                # Try to convert to string and parse
                return json.loads(str(json_string))
        
        # If response is a dict directly, return it
        elif isinstance(hf_response, dict):
            return hf_response
        
        # If response is a JSON string, parse it
        elif isinstance(hf_response, str):
            try:
                return json.loads(hf_response)
            except json.JSONDecodeError:
                return hf_response
        
        # Fallback: return the response as-is
        return hf_response
        
    except Exception as e:
        print(f"  ⚠ Error extracting JSON from HF response: {str(e)}")
        return None

def send_to_huggingface(image_data, filename):
    """Send image to Hugging Face Space API for PPE detection.
    
    Returns: tuple (raw_response, parsed_json) or (None, None) on error
    """
    if not ENABLE_HUGGINGFACE:
        return None, None
    
    start_time = time.time()
    print(f"  ⏱ Starting Hugging Face API call...")
    
    try:
        # Time client initialization
        client_init_start = time.time()
        client = get_hf_client()
        if client is None:
            return None, None
        client_init_time = time.time() - client_init_start
        if client_init_time > 0.1:
            print(f"  ⏱ Client init time: {client_init_time:.2f}s")
        
        # Optimize image size if needed (speeds up upload)
        optimize_start = time.time()
        optimized_image_data = optimize_image_size(image_data, HF_MAX_IMAGE_SIZE)
        optimize_time = time.time() - optimize_start
        if optimize_time > 0.1:
            print(f"  ⏱ Image optimization time: {optimize_time:.2f}s")
        
        # Save image temporarily
        file_save_start = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(optimized_image_data)
            temp_file_path = temp_file.name
        file_save_time = time.time() - file_save_start
        if file_save_time > 0.1:
            print(f"  ⏱ File save time: {file_save_time:.2f}s")
        
        print(f"  📤 Sending image to Hugging Face (size: {len(image_data)/1024:.1f} KB)...")
        predict_start = time.time()
        
        try:
            # ✅ CORRECT: Use "predict" without leading slash, or no api_name
            # Option 1: With api_name (no leading slash)
            result = client.predict(
                file(temp_file_path),
                api_name="predict"  # ✅ Changed from "/predict" to "predict"
            )
            
            predict_time = time.time() - predict_start
            total_time = time.time() - start_time
            
            print(f"✓ Image sent to Hugging Face API successfully")
            print(f"  ⏱ Predict time: {predict_time:.2f}s")
            print(f"  ⏱ Total time: {total_time:.2f}s")
            print(f"  Response type: {type(result)}")
            print(f"  Response length: {len(result) if isinstance(result, (list, tuple)) else 'N/A'}")
            
            # Extract JSON from response
            json_extract_start = time.time()
            parsed_json = extract_json_from_hf_response(result)
            json_extract_time = time.time() - json_extract_start
            if json_extract_time > 0.1:
                print(f"  ⏱ JSON extraction time: {json_extract_time:.2f}s")
            
            if parsed_json:
                print(f"  ✓ JSON extracted successfully")
                if isinstance(parsed_json, list):
                    print(f"  Detections count: {len(parsed_json)}")
                elif isinstance(parsed_json, dict):
                    print(f"  JSON keys: {list(parsed_json.keys())}")
            
            return result, parsed_json
            
        except Exception as predict_error:
            # Try without api_name (uses default endpoint)
            try:
                print(f"  Trying without api_name parameter (using default endpoint)...")
                predict_start = time.time()
                result = client.predict(file(temp_file_path))  # ✅ No api_name needed
                predict_time = time.time() - predict_start
                total_time = time.time() - start_time
                
                print(f"✓ Image sent to Hugging Face API successfully")
                print(f"  ⏱ Predict time: {predict_time:.2f}s")
                print(f"  ⏱ Total time: {total_time:.2f}s")
                print(f"  Response type: {type(result)}")
                print(f"  Response length: {len(result) if isinstance(result, (list, tuple)) else 'N/A'}")
                
                # Extract JSON from response
                parsed_json = extract_json_from_hf_response(result)
                if parsed_json:
                    print(f"  ✓ JSON extracted successfully")
                    if isinstance(parsed_json, list):
                        print(f"  Detections count: {len(parsed_json)}")
                    elif isinstance(parsed_json, dict):
                        print(f"  JSON keys: {list(parsed_json.keys())}")
                
                return result, parsed_json
            except Exception as e2:
                print(f"✗ Error in predict call: {str(predict_error)}")
                print(f"✗ Fallback also failed: {str(e2)}")
                raise predict_error
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        total_time = time.time() - start_time
        print(f"✗ Error sending to Hugging Face API: {str(e)}")
        print(f"  ⏱ Failed after: {total_time:.2f}s")
        global HF_CLIENT
        HF_CLIENT = None
        return None, None

def speak_message(message):
    """Convert message to speech if it contains person information."""
    if not TTS_AVAILABLE or not PYTTSX3_AVAILABLE:
        return
    
    try:
        # Speak the message in a separate thread to avoid blocking
        # Create a new engine instance each time to ensure it works every time
        def speak():
            try:
                # Create new engine instance for each message to avoid getting stuck
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)  # Speed of speech
                engine.say(message)
                engine.runAndWait()
                # Engine will be cleaned up automatically after use
            except Exception as e:
                print(f"⚠ Error in TTS: {str(e)}")
        
        # Run TTS in background thread
        tts_thread = threading.Thread(target=speak, daemon=True)
        tts_thread.start()
    except Exception as e:
        print(f"⚠ Error processing TTS: {str(e)}")

def get_person_id(person_name):
    """Get person ID from name using PERSON_ID_MAPPING."""
    if not person_name:
        return None
    
    # Clean the person name (remove "unknown person", extra spaces, etc.)
    person_name_clean = person_name.strip()
    
    # Check if it's "unknown person" or similar
    if "unknown" in person_name_clean.lower():
        return None
    
    # Try exact match first
    if person_name_clean in PERSON_ID_MAPPING:
        return PERSON_ID_MAPPING[person_name_clean]
    
    # Try case-insensitive match
    for name, person_id in PERSON_ID_MAPPING.items():
        if name.lower() == person_name_clean.lower():
            return person_id
    
    return None

def enrich_detection_results_with_person_ids(json_result):
    """Enrich detection results with person IDs from PERSON_ID_MAPPING."""
    if not json_result:
        return json_result
    
    try:
        # If it's a list of detections
        if isinstance(json_result, list):
            enriched = []
            for detection in json_result:
                if isinstance(detection, dict):
                    enriched_detection = detection.copy()
                    
                    # Check for person name in various possible fields
                    person_name = None
                    for key in ['person', 'name', 'person_name', 'Person']:
                        if key in enriched_detection:
                            person_name = enriched_detection[key]
                            break
                    
                    # If person name found, add person_id
                    if person_name:
                        person_id = get_person_id(person_name)
                        if person_id:
                            enriched_detection['person_id'] = person_id
                            print(f"  ✓ Mapped '{person_name}' to ID: {person_id}")
                        else:
                            enriched_detection['person_id'] = None
                    
                    enriched.append(enriched_detection)
                else:
                    enriched.append(detection)
            return enriched
        
        # If it's a dict
        elif isinstance(json_result, dict):
            enriched = json_result.copy()
            
            # Check for person name in various possible fields
            person_name = None
            for key in ['person', 'name', 'person_name', 'Person']:
                if key in enriched:
                    person_name = enriched[key]
                    break
            
            # If person name found, add person_id
            if person_name:
                person_id = get_person_id(person_name)
                if person_id:
                    enriched['person_id'] = person_id
                    print(f"  ✓ Mapped '{person_name}' to ID: {person_id}")
                else:
                    enriched['person_id'] = None
            
            return enriched
        
        # Return as-is if not list or dict
        return json_result
        
    except Exception as e:
        print(f"  ⚠ Error enriching detection results with person IDs: {str(e)}")
        return json_result

def check_for_unknown_in_json(json_result):
    """Check if 'UNKNOWN' appears in the JSON detection results."""
    if not json_result:
        return False
    
    try:
        # Convert to string for searching
        json_str = json.dumps(json_result).upper()
        # Check for "UNKNOWN" (case-insensitive)
        if "UNKNOWN" in json_str:
            return True
        
        # Also check if it's a list of detections
        if isinstance(json_result, list):
            for detection in json_result:
                if isinstance(detection, dict):
                    # Check all values in the detection dict
                    for key, value in detection.items():
                        if isinstance(value, str) and "unknown" in value.lower():
                            return True
        
        # Check if it's a dict
        elif isinstance(json_result, dict):
            for key, value in json_result.items():
                if isinstance(value, str) and "unknown" in value.lower():
                    return True
                elif isinstance(value, (list, dict)):
                    # Recursively check nested structures
                    if check_for_unknown_in_json(value):
                        return True
        
        return False
    except Exception as e:
        print(f"  ⚠ Error checking for UNKNOWN in JSON: {str(e)}")
        return False

def parse_person_from_hf_response(hf_response):
    """Parse person information from Hugging Face response."""
    if not hf_response:
        return None
    
    try:
        # Convert response to string for parsing
        response_str = str(hf_response)
        response_lower = response_str.lower()
        
        # Handle markdown format: **Person:** unknown person
        # Pattern 1: **Person:** unknown person (with markdown bold)
        # Matches: **Person:** unknown person
        pattern1 = r'\*\*Person:\*\*\s*([^\n\*\#]+?)(?:\n|\*\*|$)'
        match1 = re.search(pattern1, response_str, re.IGNORECASE | re.MULTILINE)
        if match1:
            person_name = match1.group(1).strip()
            # Clean up any remaining markdown or special chars, but keep spaces
            person_name = re.sub(r'\*\*', '', person_name).strip()
            person_name = re.sub(r'[^\w\s-]', '', person_name).strip()
            if person_name:
                return f"Person: {person_name}"
        
        # Pattern 2: Person: unknown person (without markdown, but might have other formatting)
        # Matches: Person: unknown person (stops at newline or next **)
        pattern2 = r'Person:\s*([^\n\*\#]+?)(?:\n|\*\*|$)'
        match2 = re.search(pattern2, response_str, re.IGNORECASE | re.MULTILINE)
        if match2:
            person_name = match2.group(1).strip()
            # Remove markdown formatting if present
            person_name = re.sub(r'\*\*', '', person_name).strip()
            person_name = re.sub(r'[^\w\s-]', '', person_name).strip()
            if person_name:
                return f"Person: {person_name}"
        
        # Pattern 3: Handle list/tuple responses
        if isinstance(hf_response, (list, tuple)) and len(hf_response) > 0:
            first_item = str(hf_response[0])
            # Try patterns on first item
            match = re.search(r'Person:\s*([^\n\*\#]+?)(?:\n|\*\*|$)', first_item, re.IGNORECASE | re.MULTILINE)
            if match:
                person_name = match.group(1).strip()
                person_name = re.sub(r'\*\*', '', person_name).strip()
                person_name = re.sub(r'[^\w\s-]', '', person_name).strip()
                if person_name:
                    return f"Person: {person_name}"
        
        # Pattern 4: Handle dict responses
        if isinstance(hf_response, dict):
            for key, value in hf_response.items():
                value_str = str(value)
                match = re.search(r'Person:\s*([^\n\*\#]+?)(?:\n|\*\*|$)', value_str, re.IGNORECASE | re.MULTILINE)
                if match:
                    person_name = match.group(1).strip()
                    person_name = re.sub(r'\*\*', '', person_name).strip()
                    person_name = re.sub(r'[^\w\s-]', '', person_name).strip()
                    if person_name:
                        return f"Person: {person_name}"
        
        # Pattern 5: Simple string fallback - look for "unknown person"
        if isinstance(hf_response, str):
            # Try to extract "unknown person" or similar
            if "unknown person" in response_lower:
                return "Person: unknown person"
            # If response is short and simple, use it
            if len(response_str.strip()) > 0 and len(response_str.strip()) < 100:
                cleaned = re.sub(r'[^\w\s-]', '', response_str.strip())
                if cleaned:
                    return f"Person: {cleaned}"
        
        return None
        
    except Exception as e:
        print(f"  Error parsing HF response: {str(e)}")
        return None


def send_message_to_autocar(message):
    """Send a message to the autocar."""
    try:
        url = f"http://{AUTOCAR_IP}:{AUTOCAR_MESSAGE_PORT}/message"
        payload = {
            "message": message,
            "timestamp": time.time(),
            "source": "PC"
        }
        
        response = requests.post(
            url,
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✓ Message sent to autocar: {message}")
            return True
        else:
            print(f"✗ Failed to send message to autocar: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Error sending message to autocar: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error sending message to autocar: {str(e)}")
        return False


def send_message_to_pc_client(message):
    """Send a message to the PC client (if configured)."""
    if not PC_CLIENT_IP:
        return False
    
    try:
        url = f"http://{PC_CLIENT_IP}:{PC_CLIENT_MESSAGE_PORT}/message"
        payload = {
            "message": message,
            "timestamp": time.time(),
            "source": "PC"
        }
        
        response = requests.post(
            url,
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✓ Message sent to PC client: {message}")
            return True
        else:
            print(f"✗ Failed to send message to PC client: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Error sending message to PC client: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error sending message to PC client: {str(e)}")
        return False


# ==================== FLASK ROUTES ====================
@app.route('/webhook', methods=['POST'])
def receive_image():
    """Receive images from autocar."""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
        except Exception as e:
            return jsonify({'error': f'Invalid base64 image data: {str(e)}'}), 400
        
        # Generate filename
        image_counter = get_next_image_number()
        filename = f"{image_counter}.jpg"
        filepath = os.path.join(RECEIVED_IMAGES_FOLDER, filename)
        
        # Save image
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        # Get metadata
        original_filename = data.get('filename', 'unknown')
        timestamp = data.get('timestamp', time.time())
        source = data.get('source', 'AUTOCAR')  # Source: AUTOCAR or PC_CLIENT
        received_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"✓ Image received: {filename}")
        print(f"  Source: {source}")
        print(f"  Original filename: {original_filename}")
        print(f"  Timestamp: {timestamp}")
        print(f"  Received at: {received_time}")
        print(f"  Saved to: {filepath}")
        
        # Send to Hugging Face API
        hf_response = None
        hf_json_result = None
        person_info = None
        if ENABLE_HUGGINGFACE:
            print(f"  Sending to Hugging Face API...")
            hf_response, hf_json_result = send_to_huggingface(image_data, filename)
            
            # Parse person information from raw response (for backward compatibility)
            if hf_response:
                person_info = parse_person_from_hf_response(hf_response)
                if person_info:
                    print(f"  {person_info}")
                    # Send message to autocar
                    send_message_to_autocar(person_info)
                    # Send message to PC client (if configured)
                    send_message_to_pc_client(person_info)
        
        # Prepare response
        response_data = {
            'status': 'success',
            'message': 'Image received and saved',
            'filename': filename,
            'filepath': filepath,
            'received_at': received_time
        }
        
        # Add Hugging Face JSON result (parsed JSON object)
        if hf_json_result is not None:
            # Enrich detection results with person IDs
            enriched_results = enrich_detection_results_with_person_ids(hf_json_result)
            response_data['detection_results'] = enriched_results
            print(f"  ✓ JSON result included in response")
            # Print the JSON result to console for debugging
            print(f"  📋 Detection Results (JSON):")
            print(json.dumps(enriched_results, indent=2))
        
        # Add person info (for backward compatibility)
        if person_info:
            response_data['person_info'] = person_info
        
        # Optionally include raw response for debugging (can be large, so commented out by default)
        # if hf_response:
        #     response_data['huggingface_raw_response'] = str(hf_response)[:500]  # Truncate for size
        
        # Check for UNKNOWN in JSON and speak if found
        if hf_json_result is not None:
            if check_for_unknown_in_json(hf_json_result):
                print(f"  🔊 UNKNOWN person detected - speaking announcement...")
                speak_message("There is UNKNOWN person")
        
        # Print summary of response being sent
        print(f"  📤 Response being sent to client:")
        print(f"     Status: {response_data.get('status')}")
        print(f"     Filename: {response_data.get('filename')}")
        if 'detection_results' in response_data:
            if isinstance(response_data['detection_results'], list):
                print(f"     Detection Results: {len(response_data['detection_results'])} detection(s)")
            else:
                print(f"     Detection Results: Available")
        if 'person_info' in response_data:
            print(f"     Person Info: {response_data['person_info']}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"✗ Error processing webhook request: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'PC Server',
        'images_folder': RECEIVED_IMAGES_FOLDER,
        'total_images': len([f for f in os.listdir(RECEIVED_IMAGES_FOLDER) if f.endswith('.jpg')])
    }), 200


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with server information."""
    config = {
        'host': HOST,
        'port': PC_PORT,
        'images_folder': RECEIVED_IMAGES_FOLDER,
        'huggingface_enabled': ENABLE_HUGGINGFACE,
        'autocar_ip': AUTOCAR_IP,
        'autocar_message_port': AUTOCAR_MESSAGE_PORT,
        'pc_client_ip': PC_CLIENT_IP,
        'pc_client_message_port': PC_CLIENT_MESSAGE_PORT
    }
    
    if ENABLE_HUGGINGFACE:
        config['huggingface_space_url'] = HUGGINGFACE_SPACE_URL
    
    return jsonify({
        'service': 'PC Server - All-in-One',
        'endpoints': {
            'webhook': '/webhook (POST) - Receive images from autocar or PC client',
            'health': '/health (GET) - Health check',
            'index': '/ (GET) - This page'
        },
        'configuration': config
    }), 200


# ==================== MAIN ====================
if __name__ == '__main__':
    print("=" * 60)
    print("PC Server - All-in-One")
    print("=" * 60)
    print(f"Server starting on {HOST}:{PC_PORT}")
    print(f"Images will be saved to: {os.path.abspath(RECEIVED_IMAGES_FOLDER)}")
    print(f"Webhook endpoint: http://{HOST}:{PC_PORT}/webhook")
    print("=" * 60)
    if ENABLE_HUGGINGFACE:
        print(f"Hugging Face Space: {HUGGINGFACE_SPACE_URL}")
        print("  ✓ Images will be sent to Hugging Face for PPE detection")
        print("  Using gradio-client for communication")
    print(f"Autocar Communication: http://{AUTOCAR_IP}:{AUTOCAR_MESSAGE_PORT}/message")
    print("  ✓ Person detection results will be sent to autocar")
    if PC_CLIENT_IP:
        print(f"PC Client Communication: http://{PC_CLIENT_IP}:{PC_CLIENT_MESSAGE_PORT}/message")
        print("  ✓ Person detection results will be sent to PC client")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(host=HOST, port=PC_PORT, debug=False)
