import cv2
import numpy as np
import time
import math
import configparser
# import serial  # Uncomment if you want to ENABLE Arduino serial I/O

# =============================================================
# Config & Parameters
# =============================================================
config = configparser.ConfigParser()
config.read('config.txt')

# Table / camera params (kept from your original config)
CAM_PIX_WIDTH = 1
CAM_PIX_HEIGHT = 1
CAM_PIX_TO_MM = float(config.get('PARAMS', 'CAMPIXTOMM', fallback=1.25))
TABLE_LENGTH  = int(config.get('PARAMS', 'TABLELENGTH', fallback=710))
TABLE_WIDTH   = int(config.get('PARAMS', 'TABLEWIDTH',  fallback=400))
FPS           = int(config.get('PARAMS', 'FPS', fallback=60))
DEFENSE_POSITION = 60 + TABLE_WIDTH // 20

# Which half belongs to the OPPONENT? 'left' or 'right'
# Change this if your camera is flipped.
OPPONENT_SIDE = 'left'  # <-- set to 'right' to swap sides

# Vertical boundary that separates halves (in image pixels)
MIDDLE_LINE_X     = 850   # tune to your table center in the image
ROBOT_SIDE_MARGIN = 50    # how far inside our half the robot is allowed to go

# If you intentionally want the on-screen zone labels flipped (as you asked earlier),
# leave this True. Set to False to show normal labels.
FLIP_ZONE_LABELS = True

# =============================================================
# Color thresholds (OpenCV HSV)
# =============================================================
# Puck: dark blue (rgba(29,61,132,255) ~ H≈110). Single band wide enough for shadows.
PUCK_BANDS = [
    (np.array([100, 120,  60], np.uint8),  # H,S,V lower
     np.array([125, 255, 255], np.uint8))  # H,S,V upper
]

# Player (green placeholder)
player_lower = np.array([40, 80, 50])
player_upper = np.array([85, 255, 255])

# Robot marker (orange)
robot_lower = np.array([7, 180, 180])
robot_upper = np.array([13, 255, 255])

# =============================================================
# Globals (tracked positions)
# =============================================================
puck_x = puck_y = 0
player_x = player_y = 0
robot_x = robot_y = 0

# =============================================================
# Serial (kept but COMMENTED so you can run)
# =============================================================
# To enable Arduino I/O:
# 1) Uncomment the `import serial` at the very top.
# 2) Uncomment the two lines below and set the correct port.
# SERIAL_PORT = '/dev/cu.usbmodem21301'
# ser = serial.Serial(SERIAL_PORT, 9600, timeout=0.01)


def send_data_to_arduino(step1: int, step2: int) -> None:
    """Send motor deltas to Arduino. Currently NO-OP because serial writes are commented.

    Enable by uncommenting serial setup above and the two marked lines below.
    """
    try:
        data = f"{step1},{step2}\n"
        # ser.write(data.encode())            # <-- UNCOMMENT to actually send
        time.sleep(0.01)
        # resp = ser.readline().decode().strip()  # <-- UNCOMMENT to read reply
        # if resp:
        #     print(f"[Arduino] {resp}")
    except Exception as e:
        # If serial is disabled, this will usually never trigger.
        print(f"[Serial] Error: {e}")

# =============================================================
# Geometry / Zones
# =============================================================

def robot_zone_is_right() -> bool:
    """If opponent is on the left, robot is on the right; else robot is on the left."""
    return OPPONENT_SIDE.lower() == 'left'


def clamp_target_to_robot_zone(tx: int, ty: int, frame_w: int, frame_h: int) -> tuple[int, int]:
    """Keep the command target inside the robot's half (with margin)."""
    if robot_zone_is_right():
        # Robot owns RIGHT half -> target must be to the RIGHT of center line
        min_x_allowed = MIDDLE_LINE_X + ROBOT_SIDE_MARGIN
        tx = max(tx, min_x_allowed)
    else:
        # Robot owns LEFT half -> target must be to the LEFT of center line
        max_x_allowed = MIDDLE_LINE_X - ROBOT_SIDE_MARGIN
        tx = min(tx, max_x_allowed)

    # Clamp to frame limits / mechanics (keep your original Y cap)
    tx = max(0, min(tx, frame_w - 1))
    ty = max(0, min(ty, 960))
    return tx, ty


def puck_is_in_opponent_zone(px: int) -> bool:
    """True if puck is inside the opponent half (respecting margin)."""
    if robot_zone_is_right():
        # Opponent is LEFT half
        return px < (MIDDLE_LINE_X - ROBOT_SIDE_MARGIN)
    else:
        # Opponent is RIGHT half
        return px > (MIDDLE_LINE_X + ROBOT_SIDE_MARGIN)

# =============================================================
# Pixel → Steps mapping
# =============================================================

def pixel_to_steps(target_x: int, target_y: int, robot_x_: int, robot_y_: int) -> tuple[int, int]:
    """Map pixel coords to motor deltas using your S1/S2 mixing (Y−X / Y+X).

    NOTE: X interpolation uses 1080 per your original mapping. If your mechanics
    really span the full width, change `[0, 1080]` to `[0, frame_w]` in both maps.
    """
    frame_w, frame_h = 1920, 1080
    target_x, target_y = clamp_target_to_robot_zone(target_x, target_y, frame_w, frame_h)

    # Pixel → step linear maps (kept as you had them)
    tx_steps = int(np.interp(target_x, [0, 1080], [0, 8000]))
    ty_steps = int(np.interp(target_y, [0,  960], [0, 8000]))
    rx_steps = int(np.interp(robot_x_, [0, 1080], [0, 8000]))
    ry_steps = int(np.interp(robot_y_, [0,  960], [0, 8000]))

    # Mix to (step1, step2)
    t_s1, t_s2 = (ty_steps - tx_steps), (ty_steps + tx_steps)
    r_s1, r_s2 = (ry_steps - rx_steps), (ry_steps + rx_steps)

    d1 = t_s1 - r_s1
    d2 = t_s2 - r_s2
    return d1, d2

# =============================================================
# Detection utilities
# =============================================================

def detect_puck(frame: np.ndarray,
                bands=PUCK_BANDS,
                min_radius: int = 20,
                max_radius: int = 80,
                blue_bias: bool = True):
    """Detect the puck by combining multiple HSV bands and (optionally) blue dominance."""
    global puck_x, puck_y
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Combine masks (OR)
    mask = None
    for low, up in bands:
        m = cv2.inRange(hsv, low, up)
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    # Optional: require blue > green & blue > red
    if blue_bias:
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        b, g, r = cv2.split(bgr)
        dom = (b > (g + 10)) & (b > (r + 10))
        dom = (dom.astype(np.uint8)) * 255
        mask = cv2.bitwise_and(mask, dom)

    # Morphology to clean up noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = None
    max_r = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circ = 4 * math.pi * area / (peri * peri)
        if circ < 0.3:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if min_radius <= r <= max_radius and r > max_r:
            largest = (int(x), int(y), int(r))
            max_r = r

    if largest is not None:
        puck_x, puck_y = largest[0], largest[1]
    return largest


def detect_blob(frame: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                min_radius: int, max_radius: int, min_circ: float = 0.3):
    """Generic circular blob detector for player/robot markers."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_r = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circ = 4 * math.pi * area / (peri * peri)
        if circ < min_circ:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if min_radius <= r <= max_radius and r > best_r:
            best = (int(x), int(y), int(r))
            best_r = r
    return best


def detect_player(frame: np.ndarray, low: np.ndarray, up: np.ndarray,
                  min_r: int = 30, max_r: int = 100):
    global player_x, player_y
    res = detect_blob(frame, low, up, min_r, max_r, min_circ=0.3)
    if res is not None:
        player_x, player_y = res[0], res[1]
    return res


def detect_robot(frame: np.ndarray, low: np.ndarray, up: np.ndarray,
                 min_r: int = 30, max_r: int = 70):
    global robot_x, robot_y
    res = detect_blob(frame, low, up, min_r, max_r, min_circ=0.3)
    if res is not None:
        robot_x, robot_y = res[0], res[1]
    return res

# =============================================================
# Visualization
# =============================================================

def draw_circles_on_frame(frame: np.ndarray, detection, color=(255, 0, 0)) -> None:
    if detection is None:
        return
    x, y, radius = detection
    cv2.circle(frame, (x, y), radius, color, 2)
    cv2.circle(frame, (x, y), 3, color, -1)


def draw_middle_line(frame: np.ndarray, x_position: int) -> None:
    """Draw the middle boundary line."""
    h = frame.shape[0]
    cv2.line(frame, (x_position, 0), (x_position, h), (255, 100, 0), 3)
    for y in range(0, h, 100):
        cv2.line(frame, (x_position - 10, y), (x_position + 10, y), (255, 100, 0), 2)
    cv2.putText(frame, "ROBOT BOUNDARY", (x_position - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)


def predict_enemy_shot_to_side(px: int, py: int, ex: int, ey: int,
                               frame_w: int, frame_h: int,
                               side: str = 'left', goal_margin: int = 8,
                               max_bounces: int = 2):
    """Predict puck path toward a side goal, bouncing off horizontal rails only."""
    dx, dy = (px - ex), (py - ey)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return [], (px, py)

    p0 = np.array([float(px), float(py)], dtype=np.float32)
    v  = np.array([float(dx), float(dy)], dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return [], (px, py)
    v /= n

    # Determine goal line & required heading
    if side == 'left':
        goal_x = float(goal_margin)
        if v[0] >= 0:
            return [], (px, py)
    else:
        goal_x = float(frame_w - 1 - goal_margin)
        if v[0] <= 0:
            return [], (px, py)

    segments = []
    eps = 1e-3

    for _ in range(max_bounces + 1):
        t_goal = (goal_x - p0[0]) / v[0] if ((side == 'left' and v[0] < 0) or (side == 'right' and v[0] > 0)) else np.inf
        t_top    = (0.0 - p0[1]) / v[1] if v[1] < 0 else np.inf
        t_bottom = ((frame_h - 1.0) - p0[1]) / v[1] if v[1] > 0 else np.inf
        t_wall = min(t_top, t_bottom)

        if t_goal > 0 and t_goal <= t_wall:
            p1 = p0 + t_goal * v
            segments.append((tuple(p0.astype(int)), tuple(p1.astype(int))))
            return segments, (int(p1[0]), int(p1[1]))

        if not np.isfinite(t_wall) or t_wall <= 0:
            break
        p1 = p0 + t_wall * v
        segments.append((tuple(p0.astype(int)), tuple(p1.astype(int))))
        v[1] = -v[1]  # reflect off horizontal wall
        p0 = p1 + eps * v

    last_pt = segments[-1][1] if segments else (px, py)
    return segments, last_pt


def draw_segments(frame: np.ndarray, segments, color=(0, 255, 255), thickness: int = 2) -> None:
    for (a, b) in segments:
        cv2.line(frame, a, b, color, thickness)


def _closest_point_on_segment(p, a, b):
    p = np.asarray(p, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-9:
        return tuple(a.astype(int)), float(np.dot(p - a, p - a))
    t = float(np.dot(p - a, ab) / ab_len2)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    d2 = float(np.dot(p - q, p - q))
    return (int(q[0]), int(q[1])), d2


def closest_point_on_polyline(segments, p):
    best_q, best_d2, best_i = None, float('inf'), -1
    for i, (a, b) in enumerate(segments):
        q, d2 = _closest_point_on_segment(p, a, b)
        if d2 < best_d2:
            best_q, best_d2, best_i = q, d2, i
    return best_q, best_d2, best_i

# =============================================================
# Main
# =============================================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, FPS)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]

    # --- Detect objects ---
    puck_det   = detect_puck(frame)
    player_det = detect_player(frame, player_lower, player_upper)
    robot_det  = detect_robot(frame, robot_lower, robot_upper)

    # --- Predict shot towards opponent goal and move to closest point on path ---
    if puck_det is not None and player_det is not None:
        cv2.line(frame, (player_x, player_y), (puck_x, puck_y), (0, 165, 255), 2)

        side = 'left' if OPPONENT_SIDE.lower() == 'left' else 'right'
        segs, impact = predict_enemy_shot_to_side(
            puck_x, puck_y, player_x, player_y,
            frame_w=w, frame_h=h,
            side=side, goal_margin=8, max_bounces=2
        )

        draw_segments(frame, segs)
        cv2.circle(frame, impact, 6, (0, 255, 255), -1)

        if segs:
            closest_pt, d2, _ = closest_point_on_polyline(segs, (robot_x, robot_y))
            cx, cy = clamp_target_to_robot_zone(closest_pt[0], closest_pt[1], w, h)
            closest_pt = (cx, cy)
            cv2.circle(frame, closest_pt, 6, (255, 0, 255), -1)
            cv2.line(frame, (robot_x, robot_y), closest_pt, (255, 0, 255), 2)

            PIXEL_DEADBAND = 6
            if d2 > PIXEL_DEADBAND * PIXEL_DEADBAND:
                s1_t, s2_t = pixel_to_steps(closest_pt[0], closest_pt[1], robot_x, robot_y)
                send_data_to_arduino(s1_t, s2_t)

    # --- Regular tracking to puck (also clamped) ---
    bx, by = clamp_target_to_robot_zone(puck_x, puck_y, w, h)
    s1, s2 = pixel_to_steps(bx, by, robot_x, robot_y)
    send_data_to_arduino(s1, s2)

    # --- Visuals ---
    draw_circles_on_frame(frame, puck_det,   color=(0, 0, 0))
    draw_circles_on_frame(frame, player_det, color=(0, 255, 0))
    draw_circles_on_frame(frame, robot_det,  color=(0, 0, 255))

    draw_middle_line(frame, MIDDLE_LINE_X)

    # Zone label (optionally flipped)
    in_opp = puck_is_in_opponent_zone(puck_x)
    if FLIP_ZONE_LABELS:
        in_opp = not in_opp

    if in_opp:
        cv2.putText(frame, "PUCK IN OPPONENT ZONE", (750, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "PUCK IN ROBOT ZONE", (780, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Debug print (kept)
    print(puck_x, puck_y, "---", robot_x, robot_y, "---", -s2, -s1)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
