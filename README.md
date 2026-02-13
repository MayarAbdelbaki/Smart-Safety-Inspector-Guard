# SSIG — Smart Safety & Identification System

**Graduation project:** A distributed system for on-site **person identification**, **PPE (Personal Protective Equipment) compliance**, and **fall detection**, with optional text-to-speech announcements on edge devices (autocar / PC client).

---

## Overview

SSIG combines:

- **Face recognition** against a reference gallery (e.g. team members).
- **PPE detection** (helmet, safety vest, face mask) using a custom YOLO model.
- **Fall detection** via Roboflow (optional).
- **Cloud/back-end:** Gradio app (Hugging Face) + optional Supabase for logs and storage.
- **Admin dashboard:** React web app that reads from Supabase and shows PPE detections and fall alerts in real time (with optional login).
- **Edge:** Autocar or PC client with camera → sends images to a relay PC → receives identification and compliance results, with optional TTS.

All sensitive configuration (API keys, IPs, Supabase credentials) is driven by **environment variables** so the repository stays safe to share and reuse.

---

## Features

| Component | Description |
|-----------|-------------|
| **Person detection** | YOLOv8 for bounding boxes; face region extracted for recognition. |
| **Face recognition** | DeepFace (VGG-Face) vs. reference photos in `huggingface/Members/`. |
| **PPE detection** | Custom YOLO (`best.pt`) for helmet, vest, mask (present/absent). |
| **Fall detection** | Roboflow API (optional; requires `ROBOFLOW_API_KEY`). |
| **Persistence** | Optional Supabase: `detections_log`, `fall_alerts`, and image storage. |
| **Admin dashboard** | React app that displays PPE detections and fall alerts from Supabase (real-time, optional auth). |
| **Text-to-speech** | Optional pyttsx3 on autocar/PC client to announce person or “unknown person”. |

---

## Architecture

```
┌─────────────────┐     HTTP (image)      ┌─────────────────┐     Gradio API      ┌─────────────────────────┐
│  Autocar /      │ ───────────────────►  │  PC Server      │ ───────────────────►  │  Hugging Face Space     │
│  PC Client      │   :5000/webhook       │  (pc_server.py)  │   (HF Space URL)     │  (PPE + Face + Fall)     │
│  (camera+TTS)   │                       │                 │                       │  Optional: Supabase      │
└─────────────────┘                       └────────┬────────┘                       └───────────┬─────────────┘
        ▲                                          │                                              │
        │ HTTP (JSON result)                        │                                              │ INSERT
        └──────────────────────────────────────────┘                                              ▼
                    :5001/message (autocar) or :5002 (PC client)              ┌─────────────────────────┐
                                                                               │  Supabase               │
                                                                               │  detections_log,         │
                                                                               │  fall_alerts, Storage    │
                                                                               └───────────┬─────────────┘
                                                                                           │
                                                                               real-time   │  read / subscribe
                                                                                           ▼
                                                                               ┌─────────────────────────┐
                                                                               │  Admin Dashboard (React) │
                                                                               │  PPE + Fall views, auth  │
                                                                               └─────────────────────────┘
```

- **Autocar / PC client:** Capture frames, detect faces (e.g. OpenCV Haar), send image to PC server, receive result, display and optionally speak it (TTS).
- **PC server:** Receives images, calls Hugging Face Space API, parses response, forwards result to autocar or PC client.
- **Hugging Face app:** Runs full pipeline: fall check → if no fall: person + PPE + face recognition; optionally uploads images and writes to Supabase.
- **Admin dashboard:** React app that connects to the same Supabase project, shows **PPE Detection** (from `detections_log`) and **Fall Detection** (from `fall_alerts`) with real-time updates and optional Supabase Auth login.

---

## Repository Structure

```
SSIG-main/
├── README.md                 # This file
├── .env.example              # Example environment variables (copy to .env)
├── .gitignore
├── CODE_EXPLANATION.md       # Detailed flow and component description
├── autocar_main.py           # Autocar: camera, face capture, send to PC, receive result, TTS
├── pc_server.py              # PC: Flask webhook, HF client, forward results to autocar/PC client
├── pc_client.py              # Optional: PC with webcam acting like autocar
├── requirements_autocar.txt  # Autocar dependencies
├── requirements_pc.txt       # PC server dependencies
├── requirements_pc_client.txt
├── huggingface/
│   ├── app.py                # Gradio app: PPE + face + fall detection, optional Supabase
│   ├── README.md             # Hugging Face Space setup and Supabase notes
│   ├── requirements.txt
│   ├── best.pt               # Custom PPE YOLO model (if included or linked)
│   └── Members/              # Reference face images (filename = person name)
├── webhookPC/                # Scripts/setup for PC webhook
├── AdminDashboard-main/      # React admin dashboard (Supabase detections + fall alerts)
│   ├── src/
│   │   ├── pages/            # Dashboard, FallDetection, Guard, FinancialPenalty, Login
│   │   └── styles/
│   └── package.json
└── TEST_CONNECTION.bat / .sh # Simple connection tests
```

---

## Setup

### 1. Clone and configure environment

```bash
git clone https://github.com/MayarAbdelbaki/Smart-Safety-Inspector-Guard
cd Smart-Safety-Inspector-Guard
cp .env.example .env
# Edit .env with your IPs, Supabase keys (optional), Roboflow key (optional), HF Space URL.
# Do not commit .env (it is in .gitignore).
```

Do **not** commit `.env`; it is listed in `.gitignore`.  
Optional: `pip install python-dotenv` so that `autocar_main.py`, `pc_server.py`, and `pc_client.py` load variables from `.env` automatically.

### 2. Hugging Face Space (PPE + Face + Fall)

- Create a **Gradio** Space and upload the contents of the `huggingface/` folder (e.g. `app.py`, `requirements.txt`, `Members/` with reference photos, and `best.pt` if you use it).
- In the Space **Settings → Variables and secrets**, set (if you use them):
  - `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
  - `ROBOFLOW_API_KEY` (and optionally `ROBOFLOW_MODEL_ID`) for fall detection
- Note the Space URL (e.g. `https://YOUR-USERNAME-YOUR-SPACE.hf.space`) and set `HUGGINGFACE_SPACE_URL` in `.env` on the PC that runs `pc_server.py`.

See `huggingface/README.md` for Supabase storage and RLS notes.

### 3. PC server (relay between autocar and Hugging Face)

```bash
pip install -r requirements_pc.txt
# Set in .env: AUTOCAR_IP, HUGGINGFACE_SPACE_URL, (optional) PC_CLIENT_IP
python pc_server.py
```

Server listens on port 5000 for images and sends results to the autocar (and optionally to a PC client).

### 4. Autocar

```bash
pip install -r requirements_autocar.txt
# Set in .env (or in code): PC_IP = IP of the machine running pc_server.py
python autocar_main.py
```

Optional: install `pyttsx3` for text-to-speech (see `requirements_autocar.txt`).

### 5. Optional: PC client (webcam instead of autocar)

```bash
pip install -r requirements_pc_client.txt
# Set MAIN_PC_IP in .env to the PC server IP
python pc_client.py
```

### 6. Admin Dashboard (Supabase data in a web UI)

The **Admin Dashboard** is a React app that reads from the same Supabase project used by the Hugging Face app. It shows PPE detections and fall alerts in real time (with optional Supabase Auth login).

**Prerequisites:** The Hugging Face app must be writing to Supabase (`detections_log`, `fall_alerts`, and optionally Storage). Enable Supabase in your Space and ensure the dashboard uses the same project.

1. Go to `AdminDashboard-main/` and install dependencies:

   ```bash
   cd AdminDashboard-main
   npm install
   ```

2. Create a `.env` file (or `.env.local`) in `AdminDashboard-main/` with your Supabase credentials:

   ```
   REACT_APP_SUPABASE_URL=https://your-project.supabase.co
   REACT_APP_SUPABASE_ANON_KEY=your-anon-key
   ```

   Use the same Supabase project as the Hugging Face Space. The anon key is safe for browser use; restrict access with Supabase RLS if needed.

3. Start the app:

   ```bash
   npm start
   ```

   The app runs at `http://localhost:3000` by default. You get:

   - **Dashboard:** Recent PPE detections from `detections_log` (person, helmet/vest/mask, status, confidence) with real-time inserts.
   - **Fall Detection:** Alerts from `fall_alerts` with real-time updates.
   - **Guard** and **Financial Penalty** views as implemented in the app.
   - **Login:** Optional Supabase Auth (configure Auth in Supabase Dashboard if you use it).

---

## Configuration (environment variables)

| Variable | Where | Purpose |
|----------|--------|---------|
| `PC_IP` | Autocar | IP of the PC running `pc_server.py`. |
| `AUTOCAR_IP` | PC server | IP of the autocar (to send results back). |
| `HUGGINGFACE_SPACE_URL` | PC server | Full URL of your Gradio Space (e.g. `https://user-space.hf.space`). |
| `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_ROLE_KEY` | Hugging Face app | Optional; for saving detections and images. |
| `ROBOFLOW_API_KEY`, `ROBOFLOW_MODEL_ID` | Hugging Face app | Optional; for fall detection. |
| `MAIN_PC_IP` | PC client | IP of the PC running `pc_server.py`. |
| `PC_CLIENT_IP`, `PC_CLIENT_MESSAGE_PORT` | PC server | Optional; for a second client (e.g. PC with webcam). |
| `REACT_APP_SUPABASE_URL`, `REACT_APP_SUPABASE_ANON_KEY` | Admin Dashboard | Supabase project URL and anon key (in `AdminDashboard-main/.env`). |

See `.env.example` for a full template. For the dashboard, use `AdminDashboard-main/.env` (do not commit it).

---

## Usage

1. Start the **PC server** (`python pc_server.py`).
2. Start the **Hugging Face Space** (or use an already deployed Space).
3. Start the **autocar** or **PC client** (`python autocar_main.py` or `python pc_client.py`).
4. When a face is detected, the image is sent to the PC → HF → results (person name, PPE status, fall if enabled) are returned and shown (and optionally spoken) on the device.

If Supabase is enabled, the **Admin Dashboard** (`AdminDashboard-main/`, `npm start`) shows live PPE detections and fall alerts from the same database.

You can also use the Gradio UI or the Space’s API directly (e.g. `/api/predict`) with an image to test the pipeline without the autocar.

---

## License

This project is provided as-is for educational and graduation project use.  
If you use or adapt it, please credit the authors and the repository.

---

## Acknowledgments

- **YOLOv8** (Ultralytics) for person and custom PPE detection  
- **DeepFace** for face recognition  
- **Roboflow** for optional fall detection  
- **Gradio** and **Hugging Face** for deployment  
- **Supabase** for optional backend and storage  

---

For a concise flow description and component roles, see **CODE_EXPLANATION.md**.
