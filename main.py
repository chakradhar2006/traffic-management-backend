import os
import io
import cv2
import time
import asyncio
import secrets
import boto3
from botocore.client import Config
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from database import (
    get_recent_rule_breakers, log_vehicle_count, log_rule_breaker,
    log_emergency, DB_DIR, init_db, create_user, get_user_by_id, hash_password
)
from yolo_detector import YOLODetector
from traffic_controller import TrafficController
from zebra_violation import ZebraCrossingMonitor

# Initialize FastAPI
app = FastAPI(title="AI Smart Traffic Signal Management System")


@app.on_event("startup")
async def on_startup():
    print("Application starting up...")
    # Initialize database
    init_db()
    
    # Pre-warm controller (fast)
    get_controller()
    
    # Ensure default admin exists
    admin = get_user_by_id("admin")
    if not admin:
        print("Creating default admin account...")
        create_user(
            first_name="System",
            last_name="Admin",
            user_id="admin",
            region="Global",
            phone="0000000000",
            experience="10",
            password="admin",
            role="admin"
        )
    print("Startup checks complete. Application ready.")




# ── CORS ──────────────────────────────────────────────────────────────────────
# allow_origins=["*"] cannot be combined with allow_credentials=True (browser spec).
# Use explicit origins instead.
FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",   # VS Code Live Server
    "http://127.0.0.1:5500",
    "null",                   # file:// protocol (dev only)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── S3 client (violation images) ──────────────────────────────────────────────
_BUCKET_NAME     = os.environ.get("BUCKET_NAME", "")
_BUCKET_REGION   = os.environ.get("BUCKET_REGION", "")
_BUCKET_ENDPOINT = os.environ.get("BUCKET_ENDPOINT", "")
_BUCKET_ACCESS_KEY = os.environ.get("BUCKET_ACCESS_KEY", "")
_BUCKET_SECRET_KEY = os.environ.get("BUCKET_SECRET_KEY", "")

def _get_s3():
    return boto3.client(
        "s3",
        region_name=_BUCKET_REGION,
        endpoint_url=_BUCKET_ENDPOINT,
        aws_access_key_id=_BUCKET_ACCESS_KEY,
        aws_secret_access_key=_BUCKET_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global instances (Lazy Loaded)
_detector = None
_controller = None
_zebra_monitors = None

def get_detector():
    global _detector
    if _detector is None:
        print("Lazy-loading YOLODetector...")
        _detector = YOLODetector()
    return _detector

def get_controller():
    global _controller
    if _controller is None:
        _controller = TrafficController()
    return _controller

def get_zebra_monitors():
    global _zebra_monitors
    if _zebra_monitors is None:
        _zebra_monitors = {i: ZebraCrossingMonitor() for i in range(1, 5)}
    return _zebra_monitors

# State
video_sources = {1: None, 2: None, 3: None, 4: None}
system_running = False
latest_frames = {1: None, 2: None, 3: None, 4: None}


def process_lane_video(lane_id, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    global system_running
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)
    
    last_db_log_time = 0
    
    while system_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        start_time = time.time()
        
        # Keep feed alive even if AI processing fails for any frame.
        frame = cv2.resize(frame, (640, 480))
        processed_frame = frame
        v_count = 0
        emergency = False
        bboxes = []
        try:
            # Lazy load on first use within the thread
            det = get_detector()
            ctrl = get_controller()
            zb_mons = get_zebra_monitors()

            # YOLO Detection
            processed_frame, v_count, emergency, bboxes = det.process_frame(frame)
            
            # Update Controller
            ctrl.update_density(lane_id, v_count, emergency)
            
            # Zebra Crossing Violation
            signal_state = ctrl.lanes[lane_id]["state"]
            violation_img = zb_mons[lane_id].check_violation(processed_frame, bboxes, signal_state, lane_id)

            
            if violation_img:
                filename, image_bytes = violation_img
                log_rule_breaker(lane_id, filename, image_bytes)
                
            if emergency:
                log_emergency(lane_id, "EMERGENCY_VEHICLE")
        except Exception as e:
            print(f"Lane {lane_id} processing error: {e}")
            controller.update_density(lane_id, 0, False)
            
        # Log density periodically (every 5 seconds)
        current_time = time.time()
        if current_time - last_db_log_time > 5:
            log_vehicle_count(lane_id, v_count)
            last_db_log_time = current_time

        # Draw generic informative overlays
        cv2.putText(processed_frame, f"LANE {lane_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(processed_frame, f"Density: {v_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Signal: {signal_state}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if signal_state=="GREEN" else ((0,255,255) if signal_state=="YELLOW" else (0,0,255)), 2)
        
        # Calculate and show FPS
        processing_time = time.time() - start_time
        calc_fps = 1.0 / processing_time if processing_time > 0 else 0
        cv2.putText(processed_frame, f"FPS: {int(calc_fps)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Store latest frame for MJPEG streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if ret:
            global latest_frames
            latest_frames[lane_id] = buffer.tobytes()
            
        sleep_time = max(0, (delay / 1000.0) - processing_time)
        time.sleep(sleep_time)
        
    cap.release()


async def gen_frames(lane_id):
    try:
        while True:
            frame_data = latest_frames.get(lane_id)
            if frame_data:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            await asyncio.sleep(0.05)  # Slightly slower to reduce CPU load (20fps)
    except asyncio.CancelledError:
        print(f"Disconnected from Lane {lane_id} stream.")



@app.post("/upload-videos")
async def upload_videos(lane1: UploadFile = File(None), 
                        lane2: UploadFile = File(None), 
                        lane3: UploadFile = File(None), 
                        lane4: UploadFile = File(None)):
    files = {1: lane1, 2: lane2, 3: lane3, 4: lane4}
    saved_files = {}
    
    for lane_id, file in files.items():
        if file:
            filepath = os.path.join(UPLOAD_DIR, f"lane_{lane_id}_{file.filename}")
            with open(filepath, "wb") as buffer:
                buffer.write(await file.read())
            video_sources[lane_id] = filepath
            saved_files[lane_id] = filepath
            
    return {"message": "Files uploaded successfully", "files": saved_files}


@app.post("/start-system")
async def start_system():
    global system_running
    if system_running:
        return {"message": "System is already running"}
        
    system_running = True
    get_controller().start()
    
    for lane_id, source in video_sources.items():

        if source:
            import threading
            threading.Thread(target=process_lane_video, args=(lane_id, source), daemon=True).start()
        else:
            # Fallback for empty slots: Provide a blank frame or ignore
            print(f"Warning: No video source for Lane {lane_id}")
            
    return {"message": "System started successfully"}


@app.post("/stop-system")
async def stop_system():
    global system_running
    system_running = False
    get_controller().stop()
    return {"message": "System stopped"}



@app.get("/video-feed/{lane_id}")
async def video_feed(lane_id: int):
    return StreamingResponse(
        gen_frames(lane_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/lane-status")
async def lane_status():
    ctrl = get_controller()
    status = ctrl.get_status()
    densities = {l: ctrl.lanes[l]["density"] for l in ctrl.lanes}
    emergencies = {l: ctrl.lanes[l]["emergency"] for l in ctrl.lanes}
    return {"signals": status, "densities": densities, "emergencies": emergencies}



@app.get("/rule-breakers")
async def rule_breakers():
    return get_recent_rule_breakers(20)


@app.get("/violations/{filename}")
async def get_violation_image(filename: str):
    """Fetch a violation image from S3 and stream it to the client."""
    s3_key = f"violations/{filename}"
    try:
        s3 = _get_s3()
        obj = s3.get_object(Bucket=_BUCKET_NAME, Key=s3_key)
        image_bytes = obj["Body"].read()
        return Response(content=image_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Violation image not found: {e}")


@app.post("/simulate-emergency/{lane_id}")
async def sim_emergency(lane_id: int):
    get_controller().simulate_emergency(lane_id)
    return {"message": f"Emergency simulated in lane {lane_id}"}



# ── Auth routes ───────────────────────────────────────────────────────────────
# Simple in-memory token store  {token: user_id}
# Replace with Redis / DB in production.
_active_tokens: dict[str, str] = {}



@app.post("/api/signup")
async def api_signup(request: Request):
    t0 = time.time()
    form = await request.form()
    first_name = form.get("first_name", "").strip()
    last_name  = form.get("last_name",  "").strip()
    user_id    = form.get("user_id",    "").strip().lower()
    region     = form.get("region",     "").strip()
    phone      = form.get("phone",      "").strip()
    experience = form.get("experience", "0").strip()
    password   = form.get("password",   "")

    if not all([first_name, last_name, user_id, region, phone, password]):
        raise HTTPException(status_code=422, detail="All fields are required.")

    print(f"Signup attempt: {user_id}")
    success = create_user(first_name, last_name, user_id, region, phone, experience, password)
    
    print(f"Signup processed in {time.time() - t0:.4f}s")
    if not success:
        print(f"Signup failed: User {user_id} already exists.")
        raise HTTPException(status_code=409, detail="User ID already taken. Please choose another.")

    print(f"Signup successful: {user_id}")
    return {"message": "Account created successfully."}



@app.post("/api/login")
async def api_login(request: Request):
    t0 = time.time()
    form = await request.form()
    user_id  = form.get("user_id",  "").strip().lower()
    password = form.get("password", "")

    if not user_id or not password:
        raise HTTPException(status_code=422, detail="User ID and password are required.")

    print(f"Login attempt: {user_id}")
    user = get_user_by_id(user_id)
    
    if not user:
        print(f"Login failed: User {user_id} not found.")
        raise HTTPException(status_code=401, detail="Invalid user ID or password.")
        
    if user["password_hash"] != hash_password(password):
        print(f"Login failed: Incorrect password for {user_id}.")
        raise HTTPException(status_code=401, detail="Invalid user ID or password.")

    token = secrets.token_hex(32)
    _active_tokens[token] = user_id

    role = user.get("role", "user")
    print(f"Login successful for {user_id} in {time.time() - t0:.4f}s")
    return {"token": token, "user_id": user_id, "role": role}



@app.get("/api/profile")
async def api_profile(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token.")

    token = authorization.split(" ", 1)[1]
    user_id = _active_tokens.get(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Token expired or invalid. Please log in again.")

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    return {
        "first_name": user["first_name"],
        "last_name":  user["last_name"],
        "user_id":    user["user_id"],
        "region":     user["region"],
        "phone":      user["phone"],
        "experience": user["experience"],
        "role":       user.get("role", "user"),
    }



@app.post("/api/logout")
async def api_logout(authorization: str = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        _active_tokens.pop(token, None)
    return {"message": "Logged out."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
