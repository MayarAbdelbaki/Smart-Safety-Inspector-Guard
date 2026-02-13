"""
PPE Detection & Face Recognition — Gradio application for Hugging Face Spaces.

This module provides:
- Person detection (YOLOv8) and face recognition (DeepFace) against a reference gallery.
- PPE detection (helmet, safety vest, mask) via a custom YOLO model (best.pt).
- Optional fall detection via Roboflow API.
- Optional persistence: Supabase (detections_log, fall_alerts, Storage).

Configuration is via environment variables (Supabase, Roboflow). No secrets in code.
"""

import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import os
from pathlib import Path
import json
from datetime import datetime
from supabase import create_client, Client
import base64
from io import BytesIO
import requests

# ---------------------------
# Configuration
# ---------------------------
MEMBERS_FOLDER = "Members"  # Folder with reference photos
PPE_CLASSES = {
    "helmet": ["helmet", "hard-hat", "hardhat", "safety-helmet"],
    "vest": ["vest", "safety-vest", "reflective-vest", "high-visibility-vest"],
    "mask": ["mask", "face-mask", "surgical-mask", "n95"]
}

# Supabase Configuration (from environment variables)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip('/')  # Remove trailing slash if present
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")  # Anon key for database operations
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")  # Service role key for Storage

# Roboflow Fall Detection (set via environment variables; optional)
# Get API key: https://roboflow.com → Your workspace → API key
# Model ID format: "workspace/model_name/version"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "fall-detection-ca3o8/4")

# Initialize Supabase client if credentials are provided
# Use service role key for both database and storage operations (bypasses RLS)
# This is recommended for server-side applications
supabase: Client = None
supabase_storage: Client = None  # Same as supabase, but kept for clarity

# Priority: Use service role key if available (recommended), otherwise fallback to anon key
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        # Use service role key for all operations (bypasses RLS)
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        supabase_storage = supabase  # Same client for both operations
        print("✅ Supabase client initialized with SERVICE ROLE KEY (bypasses RLS)")
        print("   This is recommended for server-side applications")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize Supabase with service role key: {str(e)}")
        supabase = None
        supabase_storage = None
elif SUPABASE_URL and SUPABASE_KEY:
    try:
        # Fallback to anon key (may fail if RLS is enabled)
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase_storage = supabase
        print("⚠️ Supabase client initialized with ANON KEY")
        print("   ⚠️ Warning: Storage uploads will FAIL due to RLS policies!")
        print("   ⚠️ Warning: Database operations may fail if RLS policies are enabled")
        print("   💡 SOLUTION: Add SUPABASE_SERVICE_ROLE_KEY as a SECRET in Hugging Face Space settings")
        print("   💡 Go to: Settings → Variables and secrets → New secret")
        print("   💡 Name: SUPABASE_SERVICE_ROLE_KEY")
        print("   💡 Value: Your service_role key from Supabase Dashboard → Settings → API")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize Supabase: {str(e)}")
        print("   Detection will work but results won't be saved to database")
        supabase = None
        supabase_storage = None
else:
    print("⚠️ Warning: SUPABASE_URL and SUPABASE_KEY not set")
    print("   Set them as environment variables in Hugging Face Space settings")
    print("   Detection will work but results won't be saved to database")
    supabase = None
    supabase_storage = None

# Roboflow fall detection: only active if ROBOFLOW_API_KEY is set
if ROBOFLOW_API_KEY:
    print("✅ Roboflow fall detection configured (using direct API calls)")
else:
    print("⚠️ ROBOFLOW_API_KEY not set; fall detection will be skipped. Set env var to enable.")

# ---------------------------
# Load Models
# ---------------------------
print("🔄 Loading YOLO models...")
person_model = YOLO('yolov8n.pt')  # YOLOv8 nano for person detection
print("✅ Person detection model loaded")

print("🔄 Loading PPE detection model (best.pt)...")
ppe_model = YOLO('best.pt')  # Custom trained model for PPE detection
print("✅ PPE detection model loaded")
print(f"📋 PPE Model class names: {list(ppe_model.names.values())}")
print(f"📋 PPE Model class IDs: {dict(ppe_model.names)}")

# ---------------------------
# Load Reference Faces
# ---------------------------
def load_reference_faces():
    """Load all reference faces from Members folder"""
    reference_faces = {}
    
    if not os.path.exists(MEMBERS_FOLDER):
        print(f"⚠️ Warning: {MEMBERS_FOLDER} folder not found")
        return reference_faces
    
    print(f"📂 Loading reference faces from {MEMBERS_FOLDER}...")
    
    for filename in os.listdir(MEMBERS_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(MEMBERS_FOLDER, filename)
            name = os.path.splitext(filename)[0]
            reference_faces[name] = filepath
            print(f"  ✅ Loaded: {name}")
    
    print(f"✅ Loaded {len(reference_faces)} reference faces")
    return reference_faces

REFERENCE_FACES = load_reference_faces()

# ---------------------------
# Supabase Storage Helper
# ---------------------------
def upload_image_to_storage(image_array, folder_name="detections", file_prefix="img"):
    """
    Upload image to Supabase Storage bucket 'images'
    Uses service role key to bypass RLS policies
    Returns: public URL of uploaded image or None if failed
    """
    # Use storage client (with service role key) if available, otherwise fallback to regular client
    storage_client = supabase_storage if supabase_storage is not None else supabase
    
    if storage_client is None:
        print("⚠️ Supabase Storage not configured - cannot upload image to storage")
        print("   Set SUPABASE_SERVICE_ROLE_KEY environment variable for Storage operations")
        return None
    
    try:
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{folder_name}/{file_prefix}_{timestamp}.jpg"
        
        # Convert image array to bytes
        if isinstance(image_array, np.ndarray):
            # Ensure RGB format
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:
                    # Check if BGR and convert to RGB
                    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image_array
            else:
                image_rgb = image_array
            
            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', image_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_bytes = buffer.tobytes()
        else:
            image_bytes = image_array
        
        # Upload to Supabase Storage
        bucket_name = "images"
        try:
            upload_result = storage_client.storage.from_(bucket_name).upload(
                path=filename,
                file=image_bytes,
                file_options={"content-type": "image/jpeg", "upsert": "true"}
            )
            
            # Check for upload errors
            if hasattr(upload_result, 'error') and upload_result.error:
                error_msg = upload_result.error
                print(f"⚠️ Upload failed: {error_msg}")
                if "row-level security" in str(error_msg).lower() or "unauthorized" in str(error_msg).lower():
                    print("   💡 Tip: Make sure SUPABASE_SERVICE_ROLE_KEY is set and bucket 'images' exists and is public")
                return None
        except Exception as upload_error:
            error_str = str(upload_error)
            print(f"⚠️ Upload error: {error_str}")
            if "403" in error_str or "unauthorized" in error_str.lower() or "row-level security" in error_str.lower():
                print("   💡 This is an RLS/permissions issue.")
                print("   💡 SOLUTION: Add SUPABASE_SERVICE_ROLE_KEY as a SECRET in Hugging Face Space")
                print("   💡 Steps:")
                print("      1. Go to your Space → Settings → Variables and secrets")
                print("      2. Click 'New secret' (NOT 'New variable')")
                print("      3. Name: SUPABASE_SERVICE_ROLE_KEY")
                print("      4. Value: Get from Supabase Dashboard → Settings → API → service_role key")
                print("      5. Save and restart your Space")
                print("   💡 Alternative: Create RLS policies for Storage bucket (more complex)")
            return None
        
        # Get public URL
        try:
            public_url_response = storage_client.storage.from_(bucket_name).get_public_url(filename)
            
            # Handle different response formats
            if isinstance(public_url_response, dict):
                public_url = public_url_response.get('publicUrl') or public_url_response.get('public_url')
            elif isinstance(public_url_response, str):
                public_url = public_url_response
            else:
                # Fallback: construct URL manually
                if SUPABASE_URL:
                    # SUPABASE_URL already has trailing slash removed at initialization
                    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{filename}"
                else:
                    public_url = None
            
            print(f"✅ Image uploaded to Supabase Storage: {filename}")
            print(f"   Public URL: {public_url}")
            return public_url
            
        except Exception as url_error:
            print(f"⚠️ Error getting public URL: {str(url_error)}")
            # Fallback: construct URL manually
            if SUPABASE_URL:
                base_url = SUPABASE_URL.rstrip('/')
                public_url = f"{base_url}/storage/v1/object/public/{bucket_name}/{filename}"
                print(f"   Using constructed URL: {public_url}")
                return public_url
            return None
        
    except Exception as e:
        print(f"❌ Error uploading image to Supabase Storage: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ---------------------------
# Fall Detection
# ---------------------------
def detect_falling(image):
    """
    Detect falling using Roboflow API via direct HTTP requests.
    Returns: (is_falling: bool, confidence: float, detections: list)
    Skips API call if ROBOFLOW_API_KEY is not set.
    """
    try:
        if not ROBOFLOW_API_KEY:
            print("⚠️ Fall detection skipped (ROBOFLOW_API_KEY not set)")
            return False, 0.0, []
        print("🔄 Checking for falling detection...")
        
        # Convert image to format suitable for API
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Encode image to JPEG
            _, buffer = cv2.imencode('.jpg', image_rgb)
            image_bytes = buffer.tobytes()
        else:
            image_bytes = image
        
        # Prepare API request - Roboflow serverless inference endpoint
        # Format: https://serverless.roboflow.com/infer/{model_id}?api_key={api_key}
        api_url = f"https://serverless.roboflow.com/infer/{ROBOFLOW_MODEL_ID}"
        params = {
            "api_key": ROBOFLOW_API_KEY
        }
        
        # Make API request
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(api_url, params=params, files=files, timeout=30)
        
        if response.status_code != 200:
            print(f"⚠️ Roboflow API error: {response.status_code} - {response.text}")
            return False, 0.0, []
        
        # Parse response
        result = response.json()
        
        # Parse result - handle different response formats
        is_falling = False
        max_confidence = 0.0
        detections = []
        
        # Handle different response formats from Roboflow API
        predictions = []
        if isinstance(result, dict):
            if 'predictions' in result:
                predictions = result['predictions']
            elif 'results' in result:
                predictions = result['results']
            elif 'detections' in result:
                predictions = result['detections']
            elif 'inferences' in result:
                predictions = result['inferences']
        elif isinstance(result, list):
            predictions = result
        
        for pred in predictions:
            # Handle different prediction formats
            if isinstance(pred, dict):
                class_name = str(pred.get('class', pred.get('name', pred.get('label', '')))).lower()
                confidence = float(pred.get('confidence', pred.get('score', pred.get('confidence_score', 0.0))))
                
                detection_data = {
                    "class": pred.get('class', pred.get('name', pred.get('label', ''))),
                    "confidence": confidence
                }
                
                # Add bbox if available
                if 'x' in pred and 'y' in pred:
                    detection_data["bbox"] = {
                        "x": pred.get('x', 0),
                        "y": pred.get('y', 0),
                        "width": pred.get('width', 0),
                        "height": pred.get('height', 0)
                    }
                elif 'bbox' in pred:
                    detection_data["bbox"] = pred['bbox']
                
                detections.append(detection_data)
                
                # Check if falling detected (check various class name patterns)
                if any(term in class_name for term in ['fall', 'falling', 'fallen', 'down']):
                    is_falling = True
                    max_confidence = max(max_confidence, confidence)
        
        if is_falling:
            print(f"⚠️ FALLING DETECTED! Confidence: {max_confidence:.2f}")
            print(f"   Detections: {len(detections)}")
        else:
            print(f"✅ No falling detected (checked {len(predictions)} predictions)")
        
        return is_falling, max_confidence, detections
        
    except Exception as e:
        print(f"⚠️ Error in fall detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0.0, []

# ---------------------------
# Face Recognition
# ---------------------------
def recognize_face(face_img):
    """Recognize face using DeepFace"""
    if len(REFERENCE_FACES) == 0:
        return "Unknown", 0.0
    
    try:
        # Save temporary face image
        temp_face_path = "temp_face.jpg"
        cv2.imwrite(temp_face_path, face_img)
        
        best_match = None
        best_distance = float('inf')
        best_name = "Unknown"
        
        # Compare with all reference faces
        for name, ref_path in REFERENCE_FACES.items():
            try:
                result = DeepFace.verify(
                    temp_face_path,
                    ref_path,
                    model_name='VGG-Face',
                    enforce_detection=False
                )
                
                distance = result['distance']
                if distance < best_distance:
                    best_distance = distance
                    best_name = name
                    
            except Exception as e:
                continue
        
        # Clean up
        if os.path.exists(temp_face_path):
            os.remove(temp_face_path)
        
        # Convert distance to confidence (lower distance = higher confidence)
        # VGG-Face threshold is typically around 0.4
        confidence = max(0, 1 - (best_distance / 0.4))
        
        # Only return match if confidence is above threshold
        if confidence > 0.6:
            return best_name, confidence
        else:
            return "Unknown", confidence
            
    except Exception as e:
        print(f"Error in face recognition: {str(e)}")
        return "Unknown", 0.0

# ---------------------------
# PPE Detection
# ---------------------------
def detect_ppe(img, person_box):
    """
    Detect PPE items in person bounding box using custom best.pt model
    Returns: (helmet_status, vest_status, mask_status, detected_items)
    Status values:
    - None: No detection (model didn't detect this PPE item at all)
    - False: Negative detection (model detected NO-Hardhat, NO-Vest, NO-Mask)
    - True: Positive detection (model detected Hardhat, Safety Vest, Mask)
    """
    x1, y1, x2, y2 = map(int, person_box)
    
    # Ensure valid bounding box
    if x2 <= x1 or y2 <= y1:
        return None, None, None, []
    
    person_img = img[y1:y2, x1:x2]
    
    # Skip if person region is too small
    if person_img.size == 0 or person_img.shape[0] < 10 or person_img.shape[1] < 10:
        return None, None, None, []
    
    # Run custom PPE model on person region
    try:
        results = ppe_model(person_img, conf=0.25, verbose=False)
    except Exception as e:
        print(f"⚠️ Error running PPE model: {str(e)}")
        return None, None, None, []
    
    detected_items = []
    helmet = None  # None = not detected, False = NO-Hardhat, True = Hardhat
    vest = None    # None = not detected, False = NO-Safety Vest, True = Safety Vest
    mask = None    # None = not detected, False = NO-Mask, True = Mask
    
    # Get all class names from the model for debugging
    model_class_names = list(ppe_model.names.values())
    print(f"🔍 PPE Model classes: {model_class_names}")
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name_original = ppe_model.names[class_id]  # Keep original case
                class_name = class_name_original.lower()
                detected_items.append(f"{class_name_original}({confidence:.2f})")
                
                print(f"  📦 Detected: {class_name_original} (conf: {confidence:.2f})")
                
                # Handle negative detections (NO-Hardhat, NO-Mask, NO-Safety Vest)
                if class_name.startswith("no-") or class_name_original.startswith("NO-"):
                    if "hardhat" in class_name or "helmet" in class_name:
                        helmet = False
                        print(f"    ❌ NO-Helmet detected (person not wearing helmet)")
                    elif "vest" in class_name or "safety vest" in class_name:
                        vest = False
                        print(f"    ❌ NO-Safety Vest detected (person not wearing vest)")
                    elif "mask" in class_name:
                        mask = False
                        print(f"    ❌ NO-Mask detected (person not wearing mask)")
                    continue
                
                # Skip non-PPE classes
                if class_name in ["person", "safety cone", "machinery", "vehicle"]:
                    continue
                
                # Match positive PPE classes
                # Check for helmet/hardhat (positive detection)
                if class_name in ["hardhat", "helmet", "hard-hat", "safety-helmet"]:
                    helmet = True
                    print(f"    ✅ Helmet detected!")
                
                # Check for safety vest (positive detection)
                elif class_name in ["safety vest", "vest", "safety-vest", "reflective-vest", "high-visibility-vest"]:
                    vest = True
                    print(f"    ✅ Safety Vest detected!")
                
                # Check for mask (positive detection)
                elif class_name in ["mask", "face-mask", "surgical-mask", "n95"]:
                    mask = True
                    print(f"    ✅ Mask detected!")
    
    return helmet, vest, mask, detected_items

# ---------------------------
# Main Detection Function
# ---------------------------
def detect_person_and_ppe(image):
    """
    Main function to detect falling first, then persons and their PPE if no falling.
    
    Flow:
    1. Check for fall detection FIRST
    2. If fall detected:
       - Save to Supabase 'fall_alerts' table
       - Skip PPE and face recognition models
       - Return early with fall detection result
    3. If NO fall detected:
       - Continue with face recognition and PPE detection
       - Save to Supabase 'detections_log' table (NOT fall_alerts)
       - Return normal detection results
    """
    try:
        # Handle None or invalid image
        if image is None:
            return None, json.dumps({"error": "No image provided"}, indent=2)
        
        # Convert PIL Image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Handle empty or invalid image
        if image.size == 0 or len(image.shape) < 2:
            return None, json.dumps({"error": "Invalid image format"}, indent=2)
        
        # Ensure image is in RGB format (Gradio provides RGB)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 1:  # Single channel
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV operations (cv2 expects BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w = image.shape[:2]
        
        # ============================================================
        # STEP 1: Check for falling FIRST (Priority: Fall Detection)
        # ============================================================
        print("🔍 STEP 1: Checking for fall detection...")
        is_falling, fall_confidence, fall_detections = detect_falling(image_bgr)
        
        if is_falling:
            print("🚨 FALLING DETECTED!")
            print("   → Skipping PPE and face recognition models")
            print("   → Saving to Supabase 'fall_alerts' table")
            
            # Annotate image with fall detection results
            annotated_image = image_bgr.copy()
            
            # Add main fall detection message
            cv2.putText(
                annotated_image,
                f"FALLING DETECTED! Confidence: {fall_confidence:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),  # Red
                3
            )
            
            # Add detection details
            y_pos = 80
            cv2.putText(
                annotated_image,
                f"Detections: {len(fall_detections)}",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            # Draw bounding boxes for fall detections if available
            for idx, det in enumerate(fall_detections[:5]):  # Limit to first 5 for display
                if 'bbox' in det and isinstance(det['bbox'], dict):
                    bbox = det['bbox']
                    x = int(bbox.get('x', 0))
                    y = int(bbox.get('y', 0))
                    w = int(bbox.get('width', 0))
                    h = int(bbox.get('height', 0))
                    if w > 0 and h > 0:
                        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        class_name = det.get('class', 'Fall')
                        conf = det.get('confidence', 0.0)
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(annotated_image, label, (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Upload annotated image to Supabase Storage
            image_url = None
            if supabase is not None:
                print("📤 Uploading annotated fall detection image to Supabase Storage...")
                image_url = upload_image_to_storage(
                    annotated_image, 
                    folder_name="fall_detections",
                    file_prefix="fall"
                )
            
            # Save to Supabase fall_alerts table with image URL
            # NOTE: This is the ONLY place we save to fall_alerts table
            if supabase is not None:
                try:
                    fall_data = {
                        "is_falling": True,
                        "confidence": float(fall_confidence),
                        "raw_fall_detections": json.dumps(fall_detections) if fall_detections else None,
                        "image_base64": None,  # Not storing base64 anymore, using Storage instead
                        "image_url": image_url,  # Public URL from Supabase Storage
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    result = supabase.table("fall_alerts").insert(fall_data).execute()
                    print(f"✅ Fall alert saved to Supabase 'fall_alerts' table")
                    print(f"   Confidence: {fall_confidence:.2f}, Detections: {len(fall_detections)}")
                    if image_url:
                        print(f"   Image URL: {image_url[:80]}...")
                except Exception as e:
                    print(f"❌ Error saving fall alert to Supabase: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Supabase not configured - fall alert NOT saved to database")
                print("   Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables")
            
            # Convert BGR to RGB for return
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Return fall detection result
            fall_result = {
                "falling_detected": True,
                "confidence": float(fall_confidence),
                "detections": fall_detections,
                "message": "FALLING DETECTED - PPE detection skipped"
            }
            return annotated_image, json.dumps(fall_result, indent=2)
        
        # ============================================================
        # STEP 2: No falling detected - proceed with normal detection
        # ============================================================
        print("✅ No falling detected")
        print("   → Proceeding with face recognition and PPE detection")
        print("   → Will save to Supabase 'detections_log' table (NOT fall_alerts)")
        
        # Detect persons (YOLO works with both RGB and BGR, but BGR is standard)
        print("🔍 Detecting persons...")
        results = person_model(image_bgr, classes=[0], verbose=False)  # class 0 = person
        
        detections = []
        annotated_image = image_bgr.copy()  # Work with BGR for cv2 operations
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                print(f"✅ Found {len(result.boxes)} person(s)")
                
                for idx, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    # Extract face region (upper 30% of person box)
                    face_y1 = int(y1)
                    face_y2 = int(y1 + (y2 - y1) * 0.3)
                    face_x1 = int(x1)
                    face_x2 = int(x2)
                    
                    face_img = image_bgr[face_y1:face_y2, face_x1:face_x2]
                    
                    # Recognize face
                    print(f"👤 Recognizing person {idx + 1}...")
                    name, face_confidence = recognize_face(face_img)
                    
                    # Set person_id and name based on recognition result
                    if name == "Unknown":
                        person_id = "N/A"
                        name = "unknown person"
                    else:
                        person_id = f"person_{idx + 1}"
                    
                    # Detect PPE
                    print(f"🦺 Detecting PPE for {name}...")
                    helmet, vest, mask, raw_detections = detect_ppe(image_bgr, [x1, y1, x2, y2])
                    
                    # Determine overall compliance
                    # Only compliant if all three are True (wearing all PPE)
                    # None = not detected, False = detected as not wearing, True = detected as wearing
                    is_compliant = (helmet is True) and (vest is True) and (mask is True)
                    
                    # Choose color based on compliance
                    if is_compliant:
                        color = (0, 255, 0)  # Green
                        status = "COMPLIANT"
                    elif name == "unknown person":
                        color = (128, 128, 128)  # Gray
                        status = "UNKNOWN"
                    else:
                        color = (0, 0, 255)  # Red
                        status = "VIOLATION"
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    
                    # Draw label background
                    label_text = f"{name} - {status}"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(
                        annotated_image,
                        (int(x1), int(y1) - label_size[1] - 10),
                        (int(x1) + label_size[0], int(y1)),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_image,
                        label_text,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    
                    # Draw PPE status
                    # Display: ✓ = True (wearing), ✗ = False (not wearing), ? = None (not detected)
                    def ppe_symbol(value):
                        if value is True:
                            return '✓'
                        elif value is False:
                            return '✗'
                        else:  # None
                            return '?'
                    
                    y_offset = int(y2) + 25
                    ppe_status = f"H:{ppe_symbol(helmet)} V:{ppe_symbol(vest)} M:{ppe_symbol(mask)}"
                    cv2.putText(
                        annotated_image,
                        ppe_status,
                        (int(x1), y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )
                    
                    # Store detection data
                    detection_data = {
                        "person_id": person_id,
                        "name": name,
                        "helmet": helmet,
                        "vest": vest,
                        "mask": mask,
                        "confidence": float(face_confidence),
                        "raw_ppe_detections": raw_detections,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "status": status
                    }
                    detections.append(detection_data)
                    
                    # Format for display
                    helmet_str = "NULL" if helmet is None else str(helmet)
                    vest_str = "NULL" if vest is None else str(vest)
                    mask_str = "NULL" if mask is None else str(mask)
                    print(f"✅ {name}: Helmet={helmet_str}, Vest={vest_str}, Mask={mask_str}")
        
        # Upload annotated image to Supabase Storage (after all annotations are done)
        annotated_image_url = None
        if len(detections) > 0 and supabase is not None:
            print("📤 Uploading annotated PPE detection image to Supabase Storage...")
            annotated_image_url = upload_image_to_storage(
                annotated_image,
                folder_name="ppe_detections",
                file_prefix="ppe"
            )
        
        # Save to Supabase detections_log table (NOT fall_alerts)
        # NOTE: fall_alerts table is ONLY used when fall is detected (see STEP 1 above)
        if supabase is not None and len(detections) > 0:
            try:
                print("💾 Saving detections to Supabase 'detections_log' table...")
                saved_count = 0
                for detection in detections:
                    try:
                        # Get PPE values (None = not detected, False = NO-PPE detected, True = PPE detected)
                        helmet_val = detection.get("helmet")
                        vest_val = detection.get("vest")
                        mask_val = detection.get("mask")
                        
                        # Format for display
                        helmet_display = "NULL" if helmet_val is None else str(helmet_val)
                        vest_display = "NULL" if vest_val is None else str(vest_val)
                        mask_display = "NULL" if mask_val is None else str(mask_val)
                        
                        # Get person_id and name from detection
                        person_id_val = detection.get("person_id", "N/A")
                        name_val = detection.get("name", "unknown person")
                        
                        data = {
                            "person_id": person_id_val,
                            "name": name_val,
                            "helmet": helmet_val,  # None, False, or True - Supabase will store as NULL, false, or true
                            "vest": vest_val,      # None, False, or True
                            "mask": mask_val,      # None, False, or True
                            "confidence": detection.get("confidence"),
                            "raw_ppe_detections": detection.get("raw_ppe_detections", []),
                            "image_url": annotated_image_url,  # Public URL from Supabase Storage
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        result = supabase.table("detections_log").insert(data).execute()
                        saved_count += 1
                        print(f"  ✅ Saved to Supabase 'detections_log': {data['name']} - Helmet:{helmet_display}, Vest:{vest_display}, Mask:{mask_display}")
                    except Exception as e:
                        print(f"  ❌ Error saving detection to Supabase: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                print(f"💾 Saved {saved_count}/{len(detections)} detections to Supabase 'detections_log' table")
            except Exception as e:
                print(f"❌ Error saving to Supabase: {str(e)}")
                import traceback
                traceback.print_exc()
        elif len(detections) > 0:
            print("⚠️ Supabase not configured - detections NOT saved to database")
            print("   Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables")
        
        if len(detections) == 0:
            print("ℹ️ No persons detected in image")
            # Add watermark
            cv2.putText(
                annotated_image,
                "No persons detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
        
        # Ensure annotated image is properly formatted for Gradio
        # Convert BGR to RGB (OpenCV uses BGR, but Gradio expects RGB)
        if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Ensure uint8 dtype
        if annotated_image.dtype != np.uint8:
            if annotated_image.max() <= 1.0:
                annotated_image = (annotated_image * 255).astype(np.uint8)
            else:
                annotated_image = annotated_image.astype(np.uint8)
        
        # Return annotated image and JSON results
        # For API calls, Gradio will return both outputs
        # The API endpoint will receive: [annotated_image_base64, json_string]
        return annotated_image, json.dumps(detections, indent=2)
    
    except Exception as e:
        print(f"❌ Error in detection: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return original image if available, otherwise return None
        try:
            if image is not None and isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    error_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                else:
                    error_image = image.copy()
                if error_image.dtype != np.uint8:
                    error_image = error_image.astype(np.uint8)
                return error_image, json.dumps({"error": str(e)}, indent=2)
            else:
                return None, json.dumps({"error": str(e)}, indent=2)
        except:
            return None, json.dumps({"error": str(e)}, indent=2)

# ---------------------------
# API Endpoint (for programmatic access)
# ---------------------------
def api_detect(image):
    """API endpoint that processes image and saves to Supabase"""
    try:
        # Process image (this will also save to Supabase if configured)
        _, json_result = detect_person_and_ppe(image)
        detections = json.loads(json_result)
        
        # Return success message with detection count
        # Note: detect_person_and_ppe already saved to Supabase if configured
        if isinstance(detections, list):
            return {
                "success": True,
                "detections": detections,
                "count": len(detections),
                "saved_to_supabase": supabase is not None and len(detections) > 0
            }
        elif isinstance(detections, dict) and "error" in detections:
            return detections
        else:
            return {
                "success": False,
                "detections": [],
                "count": 0,
                "saved_to_supabase": False,
                "message": "No detections or unexpected format"
            }
    except Exception as e:
        return {"error": str(e), "success": False, "saved_to_supabase": False}

# ---------------------------
# Gradio Interface
# ---------------------------
with gr.Blocks(title="PPE Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🦺 PPE Detection & Face Recognition System
    
    This system detects:
    - 👤 **Persons** in the image
    - 🪖 **Safety Helmet**
    - 🦺 **Safety Vest**
    - 😷 **Face Mask**
    - 🔍 **Face Recognition** (identifies known team members)
    
    Upload an image to get started!
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", type="numpy")
            detect_btn = gr.Button("🔍 Detect PPE & Recognize Faces", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(label="Annotated Result")
            output_json = gr.JSON(label="Detection Results (JSON)")
    
    gr.Markdown("""
    ### 📊 Result Legend:
    - 🟢 **Green Box**: Fully compliant (all PPE present)
    - 🔴 **Red Box**: Violation (missing PPE)
    - ⚫ **Gray Box**: Unknown person
    
    ### 🎯 PPE Status:
    - **H**: Helmet
    - **V**: Vest
    - **M**: Mask
    - ✓ = Present, ✗ = Missing
    """)
    
    # Examples
    gr.Markdown("### 📸 Examples:")
    gr.Examples(
        examples=[],
        inputs=input_image,
    )
    
    # Connect button
    detect_btn.click(
        fn=detect_person_and_ppe,
        inputs=input_image,
        outputs=[output_image, output_json]
    )
    
    # API endpoint - Gradio automatically creates /api/predict
    # This uses the first function in the interface
    # We'll use detect_person_and_ppe but handle API format differently

# Launch with API enabled
if __name__ == "__main__":
    print("🚀 Starting PPE Detection System...")
    print(f"📂 Members folder: {MEMBERS_FOLDER}")
    print(f"👥 Reference faces loaded: {len(REFERENCE_FACES)}")
    print("="*60)
    # Enable API and set share=False for public access
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

