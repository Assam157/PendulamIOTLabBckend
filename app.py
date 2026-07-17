import os
import cv2
import json
import time
import logging
import numpy as np
from collections import deque

from ultralytics import YOLO
from flask import Flask
from flask_sock import Sock

# ------------------------------------------------------------------
# LOGGING – only show warnings and errors for speed
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("PendulumBackend")
logger.setLevel(logging.INFO)   # keep info for connection messages

# ------------------------------------------------------------------
# FLASK & SOCK
# ------------------------------------------------------------------
app = Flask(__name__)
sock = Sock(app)

# ------------------------------------------------------------------
# CONFIGURATION – optimised for speed
# ------------------------------------------------------------------
MODEL_PATH = "./LaterModelMadeWith120Epochs.pt"

# YOLO inference every N frames (higher = faster, but rely on flow)
YOLO_EVERY_N_FRAMES = 10          # was 4

# Pivot point (pixels) – unchanged
PIVOT_X = 320
PIVOT_Y = 50

SMOOTH_N = 4                      # moving average window for position
PIXELS_PER_METRE = 320.0
CONFIDENCE_THRESHOLD = 0.20

# Optical flow – smaller window & fewer levels = faster
lk_params = dict(
    winSize=(15, 15),             # was (31,31)
    maxLevel=3,                   # was 4
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              15, 0.03)
)

# Downscale factor for optical flow (2 = half size)
FLOW_SCALE = 2

# ------------------------------------------------------------------
# LOAD MODEL – try GPU first
# ------------------------------------------------------------------
logger.info("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Use GPU if available, otherwise CPU
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
model.to(device)
logger.info(f"YOLO loaded on {device}")

# ------------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------------
@app.route("/")
def health():
    return "OK", 200

# ------------------------------------------------------------------
# WEBSOCKET – optimised loop
# ------------------------------------------------------------------
@sock.route("/ws")
def websocket(ws):
    logger.info("Client connected")

    # Buffers for smoothing bob position
    cx_buf = deque(maxlen=SMOOTH_N)
    cy_buf = deque(maxlen=SMOOTH_N)

    # State for angular velocity
    prev_theta = None
    prev_time = None
    prev_omega = 0.0

    # Optical flow tracking
    prev_gray_small = None        # downscaled gray
    prev_point_small = None       # point in downscaled coordinates
    frame_counter = 0

    try:
        while True:
            message = ws.receive()
            if message is None:
                break

            # Decode frame
            np_arr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame = cv2.flip(frame, 1)          # mirror
            h, w = frame.shape[:2]

            # Convert to grayscale (full resolution)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Downscaled version for optical flow (faster)
            gray_small = cv2.resize(gray, (w // FLOW_SCALE, h // FLOW_SCALE))

            found_bob = False
            cx = cy = 0.0

            # -------------------------------------------------------
            # YOLO detection (only every N frames)
            # -------------------------------------------------------
            frame_counter += 1
            if (frame_counter % YOLO_EVERY_N_FRAMES == 0) or (prev_point_small is None):
                # YOLO input is already small (320x240)
                small_rgb = cv2.resize(frame, (320, 240))
                results = model(small_rgb, imgsz=320, verbose=False)

                # Scale factors to map back to full resolution
                sx = w / 320.0
                sy = h / 240.0

                for result in results:
                    for box in result.boxes:
                        if float(box.conf[0]) < CONFIDENCE_THRESHOLD:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = ((x1 + x2) / 2.0) * sx
                        cy = ((y1 + y2) / 2.0) * sy
                        found_bob = True
                        break
                    if found_bob:
                        break

                if found_bob:
                    # Store point in downscaled coordinates for flow
                    prev_point_small = np.array([[cx / FLOW_SCALE, cy / FLOW_SCALE]],
                                                dtype=np.float32)

            # -------------------------------------------------------
            # Optical flow (if YOLO didn't detect)
            # -------------------------------------------------------
            if not found_bob and prev_point_small is not None and prev_gray_small is not None:
                next_point, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray_small,
                    gray_small,
                    prev_point_small,
                    None,
                    **lk_params
                )
                if status[0][0] == 1:
                    # Scale back to full resolution
                    cx = next_point[0][0] * FLOW_SCALE
                    cy = next_point[0][1] * FLOW_SCALE
                    prev_point_small = next_point
                    found_bob = True

            # -------------------------------------------------------
            # Compute pendulum state
            # -------------------------------------------------------
            if found_bob:
                cx_buf.append(cx)
                cy_buf.append(cy)
                cx_s = float(np.mean(cx_buf))
                cy_s = float(np.mean(cy_buf))

                dx = cx_s - PIVOT_X
                dy = cy_s - PIVOT_Y
                theta = float(np.arctan2(dx, dy))
                length_px = float(np.hypot(dx, dy))
                length_m = max(length_px / PIXELS_PER_METRE, 0.05)

                now = time.perf_counter()
                omega = prev_omega
                if prev_theta is not None and prev_time is not None:
                    dt = now - prev_time
                    if 0.0 < dt < 0.15:
                        raw_omega = (theta - prev_theta) / dt
                        omega = 0.6 * raw_omega + 0.4 * prev_omega

                prev_theta = theta
                prev_time = now
                prev_omega = omega

                state = {
                    "theta": theta,
                    "omega": omega,
                    "length": length_m,
                    "detected": True
                }
            else:
                state = {
                    "theta": 0.0,
                    "omega": 0.0,
                    "length": 1.0,
                    "detected": False
                }
                cx_buf.clear()
                cy_buf.clear()
                prev_theta = None
                prev_time = None
                prev_omega = 0.0
                # Clear flow history to avoid stale points
                prev_point_small = None

            # -------------------------------------------------------
            # Send JSON (only state, no logging per frame)
            # -------------------------------------------------------
            ws.send(json.dumps(state))

            # Store downscaled gray for next flow
            prev_gray_small = gray_small.copy()

    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        logger.info("Client disconnected")

# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
