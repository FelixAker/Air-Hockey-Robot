import cv2
import numpy as np
import serial
import time
import configparser
import glob
import sys
from collections import deque
from typing import Optional, Tuple

"""
Air Hockey – DEFENSE LINE + HARD BOUNDARIES + AUTO ATTACK

Overview
--------
This script controls a CoreXY air-hockey robot with vision. It tracks the puck, the
opponent mallet, and the robot, predicts likely shots, and chooses commands to
defend on a crease-aligned vertical line or (optionally) auto-attack when the puck
lingers in our half.

Key ideas
---------
- Two modes:
  * "line": hold position along a vertical defense line in front of our goal, move to
    predicted intercepts of opponent shots.
  * "puck": keep X on the defense line and track the puck vertically (simpler fallback).
- Hard safety clamps: every target is clamped into our legal half including arena
  margins and a no-go "crease" band to prevent self-goals or rail crashes.
- Auto-attack: if the puck stays in our half longer than ATTACK_TRIGGER_SEC,
  approach then push it back toward the opponent side with controlled bursts.
- CoreXY mapping: pixel targets are mapped into step-space using the CoreXY mix
  (t1 = Y−X, t2 = Y+X), with a smooth tanh burst vs. distance and a deadband to
  avoid chatter.

Hotkeys
-------
- p: toggle "line" ↔ "puck"
- o: flip sides (which half is the opponent)
- f: flip only the displayed camera image (logic remains the same)
- b: toggle border ignore (useful for visual debugging only — disables safety clamps!)
"""


# Config ----------------

config = configparser.ConfigParser()
config.read('config.txt')

# Video and control pacing
FPS = int(config.get('PARAMS', 'FPS', fallback=60))  # Camera FPS request (may be capped by the camera/driver)
CONTROL_MODE = "line"  # "line" (defensive) or "puck" (simple tracking)
SWAP_STEPS = True      # If your Arduino expects (step2, step1), keep True
PIXEL_DEADBAND = 6     # Minimum pixel error (radius) to bother moving (prevents jitter)
SEND_MIN_INTERVAL = 0.010  # Throttle serial commands (s) to reduce spam/overrun
GOAL_MARGIN_PIX = 8    # Stop shot prediction when near goal line by this margin (px)
IGNORE_BORDERS = False # DEBUG ONLY: disable clamps + visuals. Keep False in real play!

# Auto-attack behavior (engages when puck stays in our half too long)
ATTACK_ENABLE = True
ATTACK_TRIGGER_SEC = 2.0  # Puck dwell time in our half before attack
APPROACH_DIST_PIX = 40    # First approach puck if we’re farther than this (px)
HIT_PUSH_PIX = 80         # When close, push puck this many pixels toward opponent
ATTACK_COOLDOWN_SEC = 0.8 # Min spacing between push targets (reduces oscillation)

# Step-space mapping bounds (calibration to your mechanics)
X_STEPS_MAX = 8000        # Steps corresponding to image x = w−1
Y_STEPS_MAX = 8000        # Steps corresponding to image y = h−1

# Table orientation and camera presentation
OPPONENT_SIDE = 'right'   # 'left' if opponent appears on the LEFT half of the image
MIDDLE_LINE_X = 980       # Pixel x of the table midline in the camera frame
ROBOT_SIDE_MARGIN = 20    # Keep this many pixels away from the midline on our side
CAMERA_MIRROR_X = False   # Flip camera image horizontally (display/detection only)

# Physical safety margins and "crease" (no-go strip in front of our goal)
ARENA_MARGIN_X = 0        # Reserved pixels near left/right walls (x)
ARENA_MARGIN_Y = 100      # Reserved pixels near top/bottom rails (y)
GOAL_CREASE_DEPTH = 100   # Depth of the no-go band from our goal line inward

# HSV thresholds: tune to your lighting/camera
# Opponent mallet (green) and our robot (orange). Puck color handled in detect_puck().
player_lower = np.array([40, 80, 50], np.uint8)
player_upper = np.array([85, 255, 255], np.uint8)
robot_lower  = np.array([7, 180, 180], np.uint8)
robot_upper  = np.array([13, 255, 255], np.uint8)



# Serial helpers ------------------------

def autodetect_serial_port():
    """
    Try common device patterns to find a connected Arduino on macOS/Linux/Windows.
    Returns the first matching port string or None if none found.
    """
    patterns = []
    if sys.platform.startswith("darwin"):
        patterns = ["/dev/cu.usbmodem*", "/dev/tty.usbmodem*", "/dev/cu.usbserial*", "/dev/tty.usbserial*"]
    elif sys.platform.startswith("linux"):
        patterns = ["/dev/ttyACM*", "/dev/ttyUSB*", "/dev/serial/by-id/*"]
    elif sys.platform.startswith("win"):
        patterns = [f"COM{i}" for i in range(1, 40)]
    for pat in patterns:
        for p in glob.glob(pat):
            return p
    return None


def open_serial(port=None, baud=9600, timeout=0.01):
    """
    Open the serial port to the Arduino (non-fatal if absent — code still runs visuals).
    - baud: must match your Arduino sketch
    - timeout: short so readlines don't block the main loop
    """
    if port is None:
        port = autodetect_serial_port()
    if port is None:
        print("[WARN] No serial port found. Running without Arduino.")
        return None
    try:
        s = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)  # Give Arduino time to reset
        print(f"[INFO] Serial connected: {port}")
        return s
    except Exception as e:
        print(f"[WARN] Could not open serial '{port}': {e}. Running without Arduino.")
        return None


ser = open_serial()


def send_data_to_arduino(step1, step2):
    """
    Send one CoreXY burst to the Arduino as: "step1,step2\n".
    A small scaling (÷2.5) is applied here to soften bursts at the firmware interface.
    Adjust/remove this if your firmware expects raw steps.

    NOTE: If SWAP_STEPS=True the caller already swapped the order (compatibility mode).
    """
    if ser is None:
        return
    try:
        ser.write(f"{step1 / 2.5},{step2 / 2.5}\n".encode())
        print(f"{step1} {step2}")  # Console trace of unscaled steps
        resp = ser.readline().decode(errors='ignore').strip()
        if resp:
            print(f"[Arduino] {resp}")
    except serial.SerialException as e:
        print(f"[Serial Error] {e}")



# Camera ---------------------

def open_camera(prefer=(0, 1, 2, 3)):
    """
    Open the first camera index that yields a valid frame and configure 1080p@FPS.
    Returns cv2.VideoCapture or raises RuntimeError if none work.
    """
    for idx in prefer:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"[INFO] Camera opened on index {idx}, frame {frame.shape[1]}x{frame.shape[0]}")
            return cap
        cap.release()
    raise RuntimeError("No camera produced frames. Check connections and indices.")



# Zone / boundary helpers (HARD ENFORCEMENT) ---------------------

def robot_zone_is_right_raw() -> bool:
    """
    True if our robot occupies the RIGHT half of the frame (i.e., opponent is left).
    This is defined by OPPONENT_SIDE only, independent of any visual mirroring.
    """
    return OPPONENT_SIDE.lower() == 'left'


def midline_x(w: int) -> int:
    """
    Pixel x of the table's midline. If your camera view shifts, adjust MIDDLE_LINE_X.
    """
    return MIDDLE_LINE_X


def robot_zone_is_right_eff(w: int) -> bool:
    """
    Effective side computation point. Currently identical to 'raw'; kept separate
    to make it easy to add mirror-aware decisions in the future.
    """
    return robot_zone_is_right_raw()


def defense_x_for_frame(w: int) -> int:
    """
    X position of our vertical DEFENSE LINE (inner edge of our crease).
    - On our side we may not cross this line toward our own goal ("crease" no-go).
    - When IGNORE_BORDERS is True, this simplifies to crease at edge of frame.
    """
    if IGNORE_BORDERS:
        return (w - GOAL_CREASE_DEPTH) if robot_zone_is_right_eff(w) else GOAL_CREASE_DEPTH

    mid = midline_x(w)
    x = GOAL_CREASE_DEPTH  # Distance from our goal line into the field
    min_x = ARENA_MARGIN_X
    max_x = mid - ROBOT_SIDE_MARGIN
    return max(min_x, min(x, max_x))


def clamp_to_robot_zone(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    """
    Clamp any (x,y) to our legal region:
      - Our half only (respect midline + ROBOT_SIDE_MARGIN)
      - Keep off top/bottom rails by ARENA_MARGIN_Y
      - Do not enter the goal crease (stay at or inside defense_x)
    This function is the final safety barrier before computing any steps.
    """
    if IGNORE_BORDERS:
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        return x, y

    mid = midline_x(w)
    if robot_zone_is_right_eff(w):
        # Our side = right half; do not cross left of midline & margin, and
        # also do not enter the crease near the right goal line.
        min_x = mid + ROBOT_SIDE_MARGIN
        max_x = w - 1 - ARENA_MARGIN_X
        max_x = min(max_x, defense_x_for_frame(w))
    else:
        # Our side = left half; symmetric constraints.
        min_x = ARENA_MARGIN_X
        max_x = mid - ROBOT_SIDE_MARGIN
        min_x = max(min_x, defense_x_for_frame(w))

    x = max(min_x, min(x, max_x))

    # Keep Y away from the rails (top/bottom).
    min_y = ARENA_MARGIN_Y
    max_y = min(h - 1 - ARENA_MARGIN_Y, h - 1)
    y = max(min_y, min(y, max_y))
    return x, y



# Prediction (enemy shot) ------------------------
def predict_enemy_shot_until_boundary(px, py, ex, ey, frame_w, frame_h, goal_margin=GOAL_MARGIN_PIX, max_bounces=2):
    """
    Predict a straight-line shot from the puck toward the side it's currently moving,
    reflecting off horizontal rails (top/bottom) up to 'max_bounces' times, and stop
    when reaching the near goal line (±goal_margin).

    Returns:
      segments: list of ((x0, y0), (x1, y1)) line segments showing the predicted path
      impact:   final (x, y) where the shot hits the stop condition (goal line or last wall)
    """
    dx, dy = (px - ex), (py - ey)  # Vector from opponent mallet toward puck (shot direction proxy)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return [], (px, py)

    p0 = np.array([float(px), float(py)], dtype=np.float32)
    v  = np.array([float(dx), float(dy)], dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return [], (px, py)
    v /= n  # Unit vector

    heading_left = v[0] < 0
    goal_x = float(goal_margin if heading_left else frame_w - 1 - goal_margin)

    segments = []
    eps = 1e-3  # Small offset after reflection to avoid re-hitting the same wall

    for _ in range(max_bounces + 1):
        t_goal   = (goal_x - p0[0]) / v[0] if abs(v[0]) > 1e-9 else np.inf
        t_top    = (0.0 - p0[1]) / v[1] if v[1] < 0 else np.inf
        t_bottom = ((frame_h - 1.0) - p0[1]) / v[1] if v[1] > 0 else np.inf
        t_wall   = min(t_top, t_bottom)

        # Reaches goal x before any rail → stop there
        if t_goal > 0 and t_goal <= t_wall:
            p1 = p0 + t_goal * v
            segments.append((tuple(p0.astype(int)), tuple(p1.astype(int))))
            return segments, (int(p1[0]), int(p1[1]))

        # Otherwise bounce on top/bottom rail (if any positive intersection)
        if not np.isfinite(t_wall) or t_wall <= 0:
            break

        p1 = p0 + t_wall * v
        segments.append((tuple(p0.astype(int)), tuple(p1.astype(int))))
        v[1] = -v[1]  # Reflect vertical component
        p0 = p1 + eps * v

    last_pt = segments[-1][1] if segments else (px, py)
    return segments, last_pt



# Detection

def detect_puck(frame):
    """
    Locate the (yellow) puck via HSV thresholding + median blur + largest contour.
    Returns (x, y, radius) in pixels, or None if not found.
    NOTE: Tune 'lower/upper' to your lighting; radius>5 filters noise specks.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Yellow puck HSV range (tuned for typical LED lighting; adjust as needed)
    lower = np.array([18, 120, 80], np.uint8)
    upper = np.array([32, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        if radius > 5:
            return int(x), int(y), int(radius)
    return None


def detect_blob(frame, lower, upper, min_radius=24, max_radius=120, min_area=70, circ_thresh=0.20):
    """
    Generic circular blob detector via HSV mask + morphology + circularity filter.
    Use for opponent mallet and robot detection.

    Returns (x, y, radius) or None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_r = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circularity = 4 * np.pi * area / (perim ** 2)  # 1.0 is perfect circle
        if circularity < circ_thresh:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if min_radius <= radius <= max_radius and radius > best_r:
            best = (int(x), int(y), int(radius))
            best_r = radius
    return best


def draw_circles_on_frame(frame, detection, color=(255, 0, 0)):
    """
    Convenience: draw a circle + center dot for any (x,y,radius) detection.
    """
    if detection is not None:
        x, y, radius = detection
        cv2.circle(frame, (x, y), radius, color, 2)
        cv2.circle(frame, (x, y), 3, color, -1)



# Motion mapping (with HARD CLAMP) -----------

def pixel_to_steps(target_x, target_y, robot_x, robot_y, w, h, attack=False):
    """
    Convert pixel error between (robot_x,robot_y) and (target_x,target_y) to a CoreXY
    step burst (s1,s2):

    1) Deadband: Ignore tiny errors to prevent chattering.
    2) Burst magnitude: Smooth tanh(d/scale) curve saturating at MAX_BURST_*.
    3) Map pixels→step-space using calibrated linear maps for X/Y.
    4) CoreXY mix:
         t1 = Y - X
         t2 = Y + X
       Compute error in (t1,t2), normalize to unit vector, then scale by 'burst'.

    Returns:
      (s1, s2) integers (can be negative). Might be (0,0) inside deadband.
    """
    DEAD_PIX = max(3, int(PIXEL_DEADBAND))
    PX_FULL_SPEED = 120
    MIN_BURST_STEPS = 300
    MAX_BURST_NORMAL = 3000
    MAX_BURST_ATTACK = 4000

    # Keep inputs in-bounds to prevent numeric surprises
    target_x = int(np.clip(target_x, 0, w - 1))
    target_y = int(np.clip(target_y, 0, h - 1))
    robot_x  = int(np.clip(robot_x,  0, w - 1))
    robot_y  = int(np.clip(robot_y,  0, h - 1))

    # Pixel error and magnitude
    dx_px = float(target_x - robot_x)
    dy_px = float(target_y - robot_y)
    d_px  = float(np.hypot(dx_px, dy_px))

    if d_px <= DEAD_PIX:
        return 0, 0

    # Smooth burst vs. distance (bounded & monotonic)
    burst_max = MAX_BURST_ATTACK if attack else MAX_BURST_NORMAL
    gain  = np.tanh(d_px / max(1.0, PX_FULL_SPEED))  # 0..~1
    burst = int(MIN_BURST_STEPS + (burst_max - MIN_BURST_STEPS) * gain)

    # Pixel → step-space (linear calibration)
    txs = np.interp(target_x, [0, w - 1], [0, X_STEPS_MAX])
    tys = np.interp(target_y, [0, h - 1], [0, Y_STEPS_MAX])
    rxs = np.interp(robot_x,  [0, w - 1], [0, X_STEPS_MAX])
    rys = np.interp(robot_y,  [0, h - 1], [0, Y_STEPS_MAX])

    # CoreXY mix in step-space
    t1, t2 = (tys - txs), (tys + txs)
    r1, r2 = (rys - rxs), (rys + rxs)
    e1, e2 = (t1 - r1), (t2 - r2)
    mag = float(np.hypot(e1, e2))
    if mag < 1e-6:
        return 0, 0

    # Unit direction * burst => discrete steps
    u1, u2 = (e1 / mag), (e2 / mag)
    s1 = int(np.round(u1 * burst))
    s2 = int(np.round(u2 * burst))

    # Ensure at least one step when error exists
    if s1 == 0:
        s1 = 1 if e1 > 0 else -1
    if s2 == 0:
        s2 = 1 if e2 > 0 else -1

    return s1, s2



# Prediction helpers ---------------
PUCK_TRACE: deque = deque(maxlen=8)             # Recent puck positions (for velocity estimate)
PREDICTED_POINT: Optional[Tuple[int, int]] = None  # Cached intercept on defense line


def extract_xy(det) -> Optional[Tuple[int, int]]:
    """
    Robustly extract (x,y) from various detection shapes: tuple/list, dict-like, or objects.
    Returns None if nothing can be extracted.
    """
    if det is None:
        return None
    if isinstance(det, (tuple, list)) and len(det) >= 2:
        try:
            return int(det[0]), int(det[1])
        except Exception:
            pass
    if hasattr(det, "get"):
        for kx, ky in (("cx", "cy"), ("x", "y"), ("center_x", "center_y")):
            try:
                return int(det.get(kx)), int(det.get(ky))
            except Exception:
                pass
    for kx, ky in (("cx", "cy"), ("x", "y"), ("center_x", "center_y")):
        try:
            return int(getattr(det, kx)), int(getattr(det, ky))
        except Exception:
            pass
    return None


def to_display_xy(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
    """
    Apply camera mirroring to (x,y) for display if enabled. Logic is not affected.
    """
    try:
        if CAMERA_MIRROR_X:
            y = (h - 1) - y
    except NameError:
        pass
    return int(x), int(y)


def _reflect_y(y: float, ymin: int, ymax: int) -> float:
    """
    Reflect a y coordinate between [ymin, ymax] as if bouncing off rails.
    Useful to approximate where a straight-line path would be after wall bounces.
    """
    L = float(max(1, ymax - ymin))
    m = (y - ymin) % (2.0 * L)
    if m > L:
        m = 2.0 * L - m
    return ymin + m


def predict_intercept_on_defense(p0: Tuple[int, int], p1: Tuple[int, int], w: int, h: int) -> Optional[Tuple[int, int]]:
    """
    Simple linear prediction from two recent puck points to where it will cross our
    defense line (includes virtual reflections on top/bottom rails).
    Returns (dx, y_at_cross) or None if the puck is moving away from our defense line.
    """
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])

    # If camera is mirrored for display, un-mirror Y for the prediction math
    try:
        if CAMERA_MIRROR_X:
            y0 = (h - 1) - y0
            y1 = (h - 1) - y1
    except NameError:
        pass

    vx = x1 - x0
    vy = y1 - y0
    if vx == 0:
        return None  # Vertical movement only → no X crossing prediction

    dx = defense_x_for_frame(w)
    going_right = vx > 0
    right_side = robot_zone_is_right_eff(w)
    # Only useful if puck is moving toward our defense line
    if (right_side and not going_right) or ((not right_side) and going_right):
        return None

    t = (dx - x1) / float(vx)
    if t < 0:
        return None

    y_at = y1 + vy * t

    if not IGNORE_BORDERS:
        ymin = ARENA_MARGIN_Y
        ymax = h - 1 - ARENA_MARGIN_Y
        y_at = _reflect_y(y_at, ymin, ymax)

    # Re-apply mirror for display coordinates
    try:
        if CAMERA_MIRROR_X:
            y_at = (h - 1) - y_at
    except NameError:
        pass

    y_px = max(0, min(int(round(y_at)), h - 1))
    return int(dx), y_px



# UI helpers -----------------------

def draw_middle_line(frame, x):
    """
    Draw the midline and tick marks; label as the robot boundary (our half limit).
    """
    h = frame.shape[0]
    cv2.line(frame, (x, 0), (x, h), (255, 100, 0), 3)
    for y in range(0, h, 100):
        cv2.line(frame, (x - 10, y), (x + 10, y), (255, 100, 0), 2)
    cv2.putText(frame, "ROBOT BOUNDARY", (x - 110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)


def shade_rect(frame, x1, y1, x2, y2, color=(60, 60, 60), alpha=0.18):
    """
    Overlay a translucent rectangle (used to visualize margins and crease areas).
    """
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_margins_and_crease(frame):
    """
    Visualize arena safety margins and the defense line/crease on our side only.
    Disabled when IGNORE_BORDERS=True.
    """
    if IGNORE_BORDERS:
        return
    h, w = frame.shape[:2]
    # Side & top/bottom safety
    shade_rect(frame, 0, 0, ARENA_MARGIN_X, h, (60, 60, 200), 0.18)
    shade_rect(frame, w - ARENA_MARGIN_X, 0, w, h, (60, 60, 200), 0.18)
    shade_rect(frame, 0, 0, w, ARENA_MARGIN_Y, (60, 200, 60), 0.18)
    shade_rect(frame, 0, h - ARENA_MARGIN_Y, w, h, (60, 200, 60), 0.18)

    dx = defense_x_for_frame(w)
    if robot_zone_is_right_eff(w):
        # Our goal is on the right → shade right crease
        shade_rect(frame, w - GOAL_CREASE_DEPTH, 0, w, h, (0, 140, 255), 0.12)
        cv2.line(frame, (dx, 0), (dx, h), (0, 140, 255), 2)
        cv2.putText(frame, "DEFENSE LINE", (max(10, dx - 120), 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    else:
        # Our goal is on the left → shade left crease
        shade_rect(frame, 0, 0, GOAL_CREASE_DEPTH, h, (0, 140, 255), 0.12)
        cv2.line(frame, (dx, 0), (dx, h), (0, 140, 255), 2)
        cv2.putText(frame, "DEFENSE LINE", (max(10, dx - 120), 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)



# Main loop --------------------

cap = open_camera()

# Latest known detections (persist between frames to avoid NaNs when momentarily lost)
puck_x = puck_y = 0
player_x = player_y = 0
robot_x = robot_y = 0
last_robot_xy = None         # If robot blob is lost, reuse last known position
last_send = 0.0              # For serial rate limiting

# Attack state machine
puck_half_since: Optional[float] = None
last_attack_push_time: float = 0.0
attack_active: bool = False

print("[INFO] Running. Press ESC or 'q' to quit. 'p' = toggle mode, 'o' = flip sides, 'f' = camera flip only, 'b' = toggle border ignore.")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[ERROR] Camera read failed.")
        break

    h, w = frame.shape[:2]

    # Optional display mirror (helps operators; logic/prediction use the unmirrored frame)
    if CAMERA_MIRROR_X:
        frame = cv2.flip(frame, 1)

    defense_x = defense_x_for_frame(w)
    midx = midline_x(w)

    # --- Detect all targets ----------------------------------------------------
    puck_det   = detect_puck(frame)
    player_det = detect_blob(frame, player_lower, player_upper, min_radius=26, max_radius=110, min_area=80,  circ_thresh=0.15)
    robot_det  = detect_blob(frame, robot_lower,  robot_upper,  min_radius=24, max_radius=100, min_area=70,  circ_thresh=0.25)

    if puck_det is not None:
        puck_x, puck_y = puck_det[0], puck_det[1]

    if player_det is not None:
        player_x, player_y = player_det[0], player_det[1]

    if robot_det is not None:
        robot_x, robot_y = robot_det[0], robot_det[1]
        last_robot_xy = (robot_x, robot_y)
    elif last_robot_xy is not None:
        # If robot blob is lost temporarily, hold last good position
        robot_x, robot_y = last_robot_xy
    else:
        # First frames before robot is seen → center fallback
        robot_x, robot_y = w // 2, h // 2

    # --- Visual annotations ----------------------------------------------------
    draw_circles_on_frame(frame, puck_det,   color=(0, 0, 0))     # puck (black)
    draw_circles_on_frame(frame, player_det, color=(0, 255, 0))   # opponent (green)
    draw_circles_on_frame(frame, robot_det,  color=(0, 0, 255))   # robot (red/blue)

    if not IGNORE_BORDERS:
        draw_middle_line(frame, midx)
        draw_margins_and_crease(frame)

    sent_this_frame = False
    now = time.time()

    # --- Attack trigger tracking ----------------------------------------------
    # If puck remains in our half long enough, arm the attack routine.
    if ATTACK_ENABLE and puck_det is not None:
        in_our_half = (puck_x >= midx) if robot_zone_is_right_eff(w) else (puck_x <= midx)
        if in_our_half:
            if puck_half_since is None:
                puck_half_since = now
            if (now - puck_half_since) >= ATTACK_TRIGGER_SEC:
                attack_active = True
        else:
            puck_half_since = None
            attack_active = False
    else:
        puck_half_since = None
        attack_active = False

    # --- ATTACK routine (takes priority) --------------------------------------
    if attack_active and puck_det is not None:
        cv2.putText(frame, "ATTACK!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # If far from puck, first go to it; otherwise push it toward the opponent
        dx = puck_x - robot_x
        dy = puck_y - robot_y
        dist2 = dx * dx + dy * dy

        if dist2 > (APPROACH_DIST_PIX * APPROACH_DIST_PIX):
            tx, ty = clamp_to_robot_zone(puck_x, puck_y, w, h)
        else:
            dir_push = -1 if robot_zone_is_right_eff(w) else 1  # Right-side robot pushes left (−x), else +x
            target_push_x = puck_x + dir_push * HIT_PUSH_PIX
            tx, ty = clamp_to_robot_zone(target_push_x, puck_y, w, h)

            # Rate-limit push retargeting to avoid oscillatory commands
            if now - last_attack_push_time < ATTACK_COOLDOWN_SEC:
                tx, ty = clamp_to_robot_zone(tx, ty, w, h)
            else:
                last_attack_push_time = now

        # Draw the chosen attack target
        cv2.circle(frame, (tx, ty), 9, (0, 0, 255), 2)
        cv2.line(frame, (robot_x, robot_y), (tx, ty), (0, 0, 255), 2)

        # Move if outside pixel deadband, send at most once per SEND_MIN_INTERVAL
        d2 = (robot_x - tx) ** 2 + (robot_y - ty) ** 2
        if d2 > (PIXEL_DEADBAND * PIXEL_DEADBAND):
            s1, s2 = pixel_to_steps(tx, ty, robot_x, robot_y, w, h, True)
            out_a, out_b = (s2, s1) if SWAP_STEPS else (s1, s2)
            if now - last_send >= SEND_MIN_INTERVAL:
                send_data_to_arduino(out_a, out_b)
                last_send = now
                sent_this_frame = True

    # --- Defensive logic ("line" mode) ----------------------------------------
    if not sent_this_frame and CONTROL_MODE == "line" and (puck_det is not None and player_det is not None):
        # Show projected opponent shot (vector from opponent mallet to puck)
        cv2.line(frame, (player_x, player_y), (puck_x, puck_y), (0, 165, 255), 2)

        segs, impact = predict_enemy_shot_until_boundary(
            puck_x, puck_y, player_x, player_y,
            frame_w=w, frame_h=h,
            goal_margin=GOAL_MARGIN_PIX,
            max_bounces=2
        )

        # Visualize predicted bounces and impact
        for (a, b) in segs:
            cv2.line(frame, a, b, (0, 255, 255), 2)
        cv2.circle(frame, impact, 6, (0, 255, 255), -1)

        # Find where the predicted path crosses our defense line x = defense_x
        intercept = None
        for (a, b) in segs:
            x0, y0 = a
            x1, y1 = b
            if (x0 - defense_x) == 0 and (x1 - defense_x) == 0:
                intercept = (defense_x, int((y0 + y1) / 2)); break
            if (x0 <= defense_x <= x1) or (x1 <= defense_x <= x0):
                if x1 != x0:
                    t = (defense_x - x0) / (x1 - x0)
                    y = y0 + t * (y1 - y0)
                    intercept = (defense_x, int(round(y))); break

        # If shot isn't heading to our line, patrol by following the puck's Y
        if intercept is None:
            intercept = (defense_x, puck_y)

        # Safety-clamp the target (last line of defense)
        tx, ty = clamp_to_robot_zone(intercept[0], intercept[1], w, h)

        # Draw chosen defensive target
        cv2.circle(frame, (tx, ty), 7, (255, 0, 255), -1)
        cv2.line(frame, (robot_x, robot_y), (tx, ty), (255, 0, 255), 2)

        # Command movement with rate limit
        d2 = (robot_x - tx) ** 2 + (robot_y - ty) ** 2
        if d2 > (PIXEL_DEADBAND * PIXEL_DEADBAND):
            s1, s2 = pixel_to_steps(tx, ty, robot_x, robot_y, w, h, False)
            out_a, out_b = (s2, s1) if SWAP_STEPS else (s1, s2)
            if now - last_send >= SEND_MIN_INTERVAL:
                send_data_to_arduino(out_a, out_b)
                last_send = now
                sent_this_frame = True

    # --- "puck" mode fallback --------------------------------------------------
    if not sent_this_frame and CONTROL_MODE == "puck" and puck_det is not None:
        # Keep X pinned to the defense line; follow puck in Y only (after clamping)
        tx, ty = clamp_to_robot_zone(defense_x, puck_y, w, h)
        s1, s2 = pixel_to_steps(tx, ty, robot_x, robot_y, w, h, False)
        out_a, out_b = (s2, s1) if SWAP_STEPS else (s1, s2)
        if now - last_send >= SEND_MIN_INTERVAL:
            send_data_to_arduino(out_a, out_b)
            last_send = now

    # --- Prediction trace & overlay -------------------------------------------
    status = f"Mode:{CONTROL_MODE}  Opp:{OPPONENT_SIDE}  CamFlip:{CAMERA_MIRROR_X}  Puck:{puck_det is not None} Enemy:{player_det is not None} Robot:{robot_det is not None} Borders:{'OFF' if IGNORE_BORDERS else 'ON'}  Attack:{attack_active}"
    try:
        # Build short puck history to estimate future defense-line intercept
        pt = extract_xy(puck_det)
        if pt is not None:
            PUCK_TRACE.append(pt)
        if len(PUCK_TRACE) >= 2:
            pred = predict_intercept_on_defense(PUCK_TRACE[-2], PUCK_TRACE[-1], w, h)
            if pred is not None:
                PREDICTED_POINT = pred
        if PREDICTED_POINT is not None:
            px, py = to_display_xy(PREDICTED_POINT[0], PREDICTED_POINT[1], w, h)
            cv2.circle(frame, (int(px), int(py)), 6, (0,255,255), 2)
            cv2.line(frame, (int(px)-12, int(py)), (int(px)+12, int(py)), (0,255,255), 1)
            cv2.line(frame, (int(px), int(py)-12), (int(px), int(py)+12), (0,255,255), 1)
    except Exception:
        # Never let overlay math crash the control loop
        pass

    # Status HUD + console trace
    cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)
    print(f"Puck({puck_x:4d},{puck_y:4d})  Robot({robot_x:4d},{robot_y:4d})  Attack:{attack_active}  Sent:{sent_this_frame}")

    # Show the annotated feed and handle hotkeys
    cv2.imshow("Air Hockey – Defense + Attack", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):  # ESC or 'q' to quit
        break
    if key == ord('p'):
        CONTROL_MODE = "puck" if CONTROL_MODE == "line" else "line"
        print(f"[INFO] CONTROL_MODE -> {CONTROL_MODE}")
    elif key == ord('o'):
        # Flips which half belongs to the opponent; also flips our legal half
        OPPONENT_SIDE = 'left' if OPPONENT_SIDE == 'right' else 'right'
        side = 'right' if robot_zone_is_right_raw() else 'left'
        print(f"[INFO] OPPONENT_SIDE -> {OPPONENT_SIDE} (robot zone now {side})")
    elif key == ord('f'):
        # Visual only — logic remains on unmirrored coordinates
        CAMERA_MIRROR_X = not CAMERA_MIRROR_X
        print(f"[INFO] CAMERA_MIRROR_X -> {CAMERA_MIRROR_X} (logic unchanged)")
    elif key == ord('b'):
        # WARNING: Disables clamps; useful only for debug/visualization
        IGNORE_BORDERS = not IGNORE_BORDERS
        print(f"[INFO] IGNORE_BORDERS -> {IGNORE_BORDERS} (border clamps & visuals {'disabled' if IGNORE_BORDERS else 'enabled'})")

# Cleanup on exit
cap.release()
cv2.destroyAllWindows()
