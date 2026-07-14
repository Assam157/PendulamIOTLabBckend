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

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger("PendulumBackend")

# ==========================================================
# FLASK
# ==========================================================

app = Flask(__name__)
sock = Sock(app)

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = "./LaterModelMadeWith120Epochs.pt"

PIVOT_X = 320
PIVOT_Y = 50

SMOOTH_N = 4

PIXELS_PER_METRE = 320.0

CONFIDENCE_THRESHOLD = 0.20

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        15,
        0.03
    )
)

# ==========================================================
# LOAD MODEL
# ==========================================================

logger.info("Loading YOLO model...")

model = YOLO(MODEL_PATH)

logger.info("YOLO loaded successfully.")

# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.route("/")
def health():
    return "OK", 200

# ==========================================================
# WEBSOCKET
# ==========================================================

@sock.route("/ws")
def websocket(ws):

    logger.info("Client connected")

    cx_buf = deque(maxlen=SMOOTH_N)
    cy_buf = deque(maxlen=SMOOTH_N)

    prev_theta = None
    prev_time = None
    prev_omega = 0.0

    prev_gray = None
    prev_point = None

    try:

        while True:

            message = ws.receive()

            if message is None:
                break

            logger.info(
                "Received frame: %d bytes",
                len(message)
            )

            np_arr = np.frombuffer(
                message,
                np.uint8
            )

            frame = cv2.imdecode(
                np_arr,
                cv2.IMREAD_COLOR
            )

            if frame is None:
                logger.warning(
                    "Frame decode failed"
                )
                continue

            logger.info(
                "Frame shape: %s",
                frame.shape
            )

            frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            found_bob = False
            tracked = False

            # -------------------------------------------------
            # YOLO
            # -------------------------------------------------

            results = model(
                frame,
                verbose=False
            )

            for result in results:

                for box in result.boxes:

                    confidence = float(
                        box.conf[0]
                    )

                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    prev_point = np.array(
                        [[cx, cy]],
                        dtype=np.float32
                    )

                    found_bob = True

                    logger.info(
                        "Detection %.2f",
                        confidence
                    )

                    break

                if found_bob:
                    break

            # -------------------------------------------------
            # Optical Flow
            # -------------------------------------------------

            if (
                not found_bob
                and prev_point is not None
                and prev_gray is not None
            ):

                next_point, status, _ = (
                    cv2.calcOpticalFlowPyrLK(
                        prev_gray,
                        gray,
                        prev_point,
                        None,
                        **lk_params
                    )
                )

                if status[0][0] == 1:

                    cx, cy = next_point[0]

                    prev_point = next_point

                    tracked = True
                    found_bob = True

            prev_gray = gray.copy()
            # -------------------------------------------------
            # Compute Pendulum State
            # -------------------------------------------------

            if found_bob:

                cx_buf.append(cx)
                cy_buf.append(cy)

                cx_s = float(np.mean(cx_buf))
                cy_s = float(np.mean(cy_buf))

                dx = cx_s - PIVOT_X
                dy = cy_s - PIVOT_Y

                theta = float(np.arctan2(dx, dy))

                length_px = float(np.hypot(dx, dy))

                length_m = max(
                    length_px / PIXELS_PER_METRE,
                    0.05
                )

                now = time.perf_counter()

                omega = prev_omega

                if (
                    prev_theta is not None
                    and prev_time is not None
                ):

                    dt = now - prev_time

                    if 0 < dt < 0.15:

                        raw_omega = (
                            theta - prev_theta
                        ) / dt

                        omega = (
                            0.6 * raw_omega
                            + 0.4 * prev_omega
                        )

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

            logger.info(state)

            # -------------------------------------------------
            # Send JSON
            # -------------------------------------------------

            ws.send(
                json.dumps(state)
            )

    except Exception as e:

        logger.exception(
            "WebSocket error: %s",
            e
        )

    finally:

        logger.info(
            "Client disconnected"
        )


# ==========================================================
# LOCAL RUN
# ==========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    logger.info(
        "Starting Flask server on port %d",
        port
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
