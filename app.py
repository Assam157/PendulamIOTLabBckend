import cv2
import numpy as np
import asyncio
import websockets
import json
import time
from ultralytics import YOLO
from collections import deque

try:
    import nest_asyncio
    nest_asyncio.apply()
except:
    pass

# =========================
# CONFIG
# =========================
# Download the Kaggle model .pt file and place it next to this script.
MODEL_PATH = "./best.pt"   # <-- Updated to Kaggle model path

PIVOT_X, PIVOT_Y = 320, 50
SMOOTH_N = 4
PIXELS_PER_METRE = 320.0

# =========================
# TRACKING PARAMETERS (unchanged)
# =========================
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
)

# Global YOLO model – loaded once when server starts
model = YOLO(MODEL_PATH)

# =========================
# PER‑CLIENT PROCESSING
# =========================
async def handler(ws):
    """
    Each connected client gets its own state.
    The client sends JPEG frames as binary WebSocket messages.
    The server replies with a JSON string containing the pendulum state.
    """
    # Per‑client state (independent for every connection)
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
            # Only binary messages are processed (JPEG frames)
            if not isinstance(message, bytes):
                continue

            # Decode the received JPEG frame
            np_arr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # --- Optional: mirror the frame (same as original flip) ---
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            found_bob = False
            tracked = False

            # =========================
            # 1. YOLO DETECTION
            # =========================
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

            # =========================
            # 2. OPTICAL FLOW (FALLBACK)
            # =========================
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

            # =========================
            # 3. PROCESS MOTION
            # =========================
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
                # No bob found – reset smoothing and send empty state
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

            # =========================
            # SEND STATE BACK TO THIS CLIENT
            # =========================
            await ws.send(json.dumps(state))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {ws.remote_address}")
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        # Cleanup not strictly necessary for local variables
        pass

# =========================
# START SERVER
# =========================
async def main():
    # Bind to all interfaces on port 8765 (use localhost only for local dev)
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("WebSocket server running on ws://0.0.0.0:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
