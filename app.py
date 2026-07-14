import os
import cv2
import numpy as np
import asyncio
import websockets
import json
import time
from ultralytics import YOLO
from collections import deque

# =========================
# CONFIG
# =========================
MODEL_PATH = r"./LaterModelMadeWith120Epochs.pt"

PIVOT_X, PIVOT_Y = 320, 50
SMOOTH_N = 4
PIXELS_PER_METRE = 320.0

# =========================
# TRACKING PARAMETERS
# =========================
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
)

# Load YOLO model once
model = YOLO(MODEL_PATH)

# =========================
# PER‑CLIENT PROCESSING
# =========================
async def handler(ws):
    """Receives JPEG frames, runs YOLO + optical flow, returns pendulum state."""
    cx_buf = deque(maxlen=SMOOTH_N)
    cy_buf = deque(maxlen=SMOOTH_N)
    prev_theta = None
    prev_time = None
    prev_omega = 0.0
    prev_gray = None
    prev_point = None

    print(f"Client connected: {ws.remote_address}")

    try:
        async for message in ws:
            if not isinstance(message, bytes):
                continue

            # Decode JPEG frame
            np_arr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            found_bob = False
            tracked = False

            # 1. YOLO detection
            results = model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    if float(box.conf[0]) < 0.2:
                        continue
                    coords = box.xyxy[0].tolist()
                    cx = (coords[0] + coords[2]) / 2
                    cy = (coords[1] + coords[3]) / 2
                    prev_point = np.array([[cx, cy]], dtype=np.float32)
                    found_bob = True
                    break
                if found_bob:
                    break

            # 2. Optical flow fallback
            if not found_bob and prev_point is not None and prev_gray is not None:
                next_point, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_point, None, **lk_params
                )
                if status[0][0] == 1:
                    cx, cy = next_point[0]
                    prev_point = next_point
                    tracked = True
                    found_bob = True

            prev_gray = gray.copy()

            # 3. Process motion
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
                    if 0 < dt < 0.15:
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

            await ws.send(json.dumps(state))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {ws.remote_address}")
    except Exception as e:
        print(f"Error: {e}")

# =========================
# START SERVER
# =========================
async def main():
    port = int(os.environ.get("PORT", 8765))
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"WebSocket server running on 0.0.0.0:{port}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
