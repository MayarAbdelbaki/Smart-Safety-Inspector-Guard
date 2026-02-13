---
title: PPE Detection System
emoji: 🦺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# 🦺 PPE Detection & Face Recognition System

This Hugging Face Space detects Personal Protective Equipment (PPE), recognizes faces against a reference gallery, and optionally runs fall detection. **All secrets (Supabase, Roboflow) are configured via Space Variables/Secrets—no keys in code.**

## Features

- 👤 **Person Detection** using YOLOv8
- 🔍 **Face Recognition** using DeepFace
- 🪖 **Helmet Detection**
- 🦺 **Safety Vest Detection**
- 😷 **Face Mask Detection**
- 📊 **Compliance Status** (Compliant/Violation/Unknown)

## Setup Instructions

### 1. Create this Space on Hugging Face

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - **SDK**: Gradio
   - **Space name**: your-ppe-detection (or any name)
   - **Visibility**: Public or Private

### 2. Upload Files

Upload these files to your Space:
- `app.py` - Main application
- `requirements.txt` - Python dependencies
- `best.pt` - Custom PPE YOLO model (train or obtain separately)
- `README.md` - This file
- `Members/` folder - Create this folder and add reference photos (filename = person name)

### 3. Add Reference Photos

Create a `Members` folder in your Space and add photos of team members:
```
Members/
  ├── Amr.jpg
  ├── Ashraf.jpg
  ├── Ayman.jpg
  ├── Badrawi.jpg
  ├── Mariam.jpg
  └── Mayar.jpg
```

**Photo Requirements:**
- Clear, front-facing photos
- Good lighting
- One person per photo
- Filename = Person's name (without spaces)

### 4. Get Your Space URL

After the Space builds successfully, you'll get a URL like:
```
https://YOUR-USERNAME-YOUR-SPACE.hf.space
```

Use this URL in your PC camera script:
```
https://YOUR-USERNAME-YOUR-SPACE.hf.space/api/predict
```

## API Usage

### Programmatic Access

Send POST request with image:

```python
import requests

url = "https://YOUR-USERNAME-YOUR-SPACE.hf.space/api/predict"
files = {"data": open("image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()
print(result)
```

### Response Format

```json
{
  "data": [
    {
      "person_id": "person_1",
      "name": "Mayar",
      "helmet": true,
      "vest": true,
      "mask": false,
      "confidence": 0.87,
      "raw_ppe_detections": ["helmet", "vest"],
      "bbox": [100, 50, 300, 400],
      "status": "VIOLATION"
    }
  ]
}
```

## Local Testing

To test locally before deploying:

```bash
pip install -r requirements.txt
python app.py
```

## Supabase Configuration

### Environment Variables

Set these in your Hugging Face Space settings (Settings → Variables):

1. **SUPABASE_URL** - Your Supabase project URL
   - Format: `https://xxxxx.supabase.co`

2. **SUPABASE_KEY** - Your Supabase anon/public key
   - Found in: Supabase Dashboard → Settings → API → anon/public key

3. **SUPABASE_SERVICE_ROLE_KEY** - Your Supabase service role key (required for Storage)
   - Found in: Supabase Dashboard → Settings → API → service_role key
   - ⚠️ **Important**: This is required for uploading images to Storage
   - ⚠️ **Security**: Never expose this key in client-side code

### Storage Bucket Setup

1. **Create Storage Bucket**:
   - Go to Supabase Dashboard → Storage
   - Click "New bucket"
   - Name: `images`
   - Make it **Public** (toggle "Public bucket" ON)

2. **Bucket Permissions** (if using RLS):
   - Go to Storage → Policies
   - Create policy for `images` bucket:
     - Policy name: `Allow authenticated uploads`
     - Allowed operation: `INSERT`
     - Policy definition: `true` (or use service role key to bypass RLS)

### Troubleshooting Storage Upload Errors

If you get `403 Unauthorized` or `row-level security policy` errors:

1. ✅ **Set SUPABASE_SERVICE_ROLE_KEY** environment variable (recommended solution)
2. ✅ Make sure the `images` bucket exists and is **public**
3. ✅ Check that your service role key is correct
4. Alternative: Disable RLS for the `images` bucket (less secure)

### RLS (Row-Level Security) Policies

**Recommended Approach: Use Service Role Key** ✅
- The code automatically uses `SUPABASE_SERVICE_ROLE_KEY` for both database and storage operations
- This **bypasses RLS** entirely, which is perfect for server-side applications
- **No RLS policies needed** if you use the service role key

**Alternative: If Using Anon Key Only** ⚠️
If you only want to use the anon key, you'll need to create RLS policies:

#### For `fall_alerts` table:
```sql
-- Allow INSERT for authenticated users (or anonymous if needed)
CREATE POLICY "Allow insert fall alerts"
ON public.fall_alerts
FOR INSERT
TO authenticated, anon
WITH CHECK (true);
```

#### For `detections_log` table:
```sql
-- Allow INSERT for authenticated users (or anonymous if needed)
CREATE POLICY "Allow insert detections log"
ON public.detections_log
FOR INSERT
TO authenticated, anon
WITH CHECK (true);
```

#### For Storage `images` bucket:
- Go to Storage → Policies → `images` bucket
- Create policy:
  - **Policy name**: `Allow public uploads`
  - **Allowed operation**: `INSERT`
  - **Policy definition**: `true`
  - **Target roles**: `authenticated, anon`

**Note**: Using service role key is simpler and more secure for server-side operations.

## Notes

- First run will download YOLO and DeepFace models (one-time)
- Processing time: 2-5 seconds per image
- Works with images containing multiple persons
- Unknown persons are still checked for PPE compliance
- Images are uploaded to Supabase Storage and URLs are saved in database

## Support

For issues or questions, open an issue on the Space's Community tab.