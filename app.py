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
MODEL_PATH = r"./LaterModelMadeWith120Epochs.pt"

PIVOT_X, PIVOT_Y = 320, 50
SMOOTH_N = 4
PIXELS_PER_METRE = 320.0

# =========================
# STATE
# =========================
state = {
    "theta": 0.0,
    "omega": 0.0,
    "length": 1.0,
    "detected": False
}

cx_buf = deque(maxlen=SMOOTH_N)
cy_buf = deque(maxlen=SMOOTH_N)

prev_theta = None
prev_time = None
prev_omega = 0.0

# --- TRACKING ---
prev_gray = None
prev_point = None

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
)

model = YOLO(MODEL_PATH)
clients = set()

# =========================
# WEBSOCKET
# =========================
async def handler(ws):
    clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        clients.remove(ws)

# =========================
# MAIN LOOP
# =========================
async def main_loop():
    global prev_theta, prev_time, prev_omega
    global prev_gray, prev_point

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

            state["theta"] = theta
            state["omega"] = omega
            state["length"] = length_m
            state["detected"] = True

            # =========================
            # DRAW
            # =========================
            color = (0, 255, 0) if not tracked else (255, 0, 0)

            cv2.circle(frame, (PIVOT_X, PIVOT_Y), 8, (0, 255, 255), -1)
            cv2.line(frame, (PIVOT_X, PIVOT_Y), (int(cx_s), int(cy_s)), color, 2)
            cv2.circle(frame, (int(cx_s), int(cy_s)), 12, (0, 0, 255), -1)

            angle_deg = np.degrees(theta)
            mode = "TRACKING" if tracked else "DETECTION"

            cv2.putText(frame,
                f"{mode} | theta={angle_deg:+.1f} omega={omega:+.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        else:
            state["detected"] = False
            cx_buf.clear()
            cy_buf.clear()
            prev_theta = None
            prev_time = None

        # =========================
        # SEND DATA
        # =========================
        if clients:
            msg = json.dumps(state)
            await asyncio.gather(
                *(ws.send(msg) for ws in clients),
                return_exceptions=True
            )

        cv2.imshow("Pendulum Tracker (ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        await asyncio.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

# =========================
# START SERVER
# =========================
async def start():
    async with websockets.serve(handler, "127.0.0.1", 8765):
        print("WebSocket on ws://127.0.0.1:8765")
        await main_loop()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(start())
