import cv2
import numpy as np
import socket
import serial
import time
import configparser
import glob
import sys

"""
Air Hockey – DEFENSE LINE + HARD BOUNDARIES
- Defend on a vertical line in front of our goal (inner edge of the goal crease)
- Never command the robot outside its legal zone (midline margin, arena margins, crease)
- Can switch between 'line' (defense line interception) and 'puck' tracking with 'p'

Notes:
- Keep SWAP_STEPS=True if your Arduino expects (step2, step1)
- Set OPPONENT_SIDE = 'left' if opponent is on the left half in the image; 'right' otherwise
- Use 'o' to flip sides (opponent left/right). Use 'f' to flip the camera image ONLY.
- Press 'x' to invert motor X sense, 'y' for motor Y sense (does not change camera).
"""

# =============================================================
# Config
# =============================================================
config = configparser.ConfigParser()
config.read('config.txt')

CAM_PIX_TO_MM = float(config.get('PARAMS', 'CAMPIXTOMM', fallback=1.25))
TABLE_LENGTH  = int(config.get('PARAMS', 'TABLELENGTH',  fallback=710))
TABLE_WIDTH   = int(config.get('PARAMS', 'TABLEWIDTH',   fallback=400))
FPS           = int(config.get('PARAMS', 'FPS',          fallback=60))

CONTROL_MODE        = "line"   # "line" (defend at crease line) or "puck"
SWAP_STEPS          = True     # True keeps your historical (step2, step1) send order
PIXEL_DEADBAND      = 6        # pixels (squared used for comparison)
SEND_MIN_INTERVAL   = 0.010    # serial send rate limit (seconds)
GOAL_MARGIN_PIX     = 8        # for path prediction goal line proximity

# Step-space & motor sense
X_STEPS_MAX        = 8000
Y_STEPS_MAX        = 8000
MOTOR_INVERT_X     = False   # if robot moves opposite along defense line, toggle with 'x'
MOTOR_INVERT_Y     = False   # if vertical sense is reversed, toggle with 'y'

# Sides/orientation
OPPONENT_SIDE       = 'left'   # 'left' if opponent is left in the image, else 'right'
MIDDLE_LINE_X       = 940      # visual midline x (pixels)
ROBOT_SIDE_MARGIN   = 140      # how far away from midline our robot must stay
CAMERA_MIRROR_X     = True     # flip camera image only (no logic flip)

# Safety margins and crease (no-go areas)
ARENA_MARGIN_X      = 120      # keep away from left/right walls (bigger)
ARENA_MARGIN_Y      = 140       # keep away from top/bottom rails (bigger)
GOAL_CREASE_DEPTH   = 260      # depth of the no-go strip in front of a goal (bigger)

# HSV - puck (blue), player (green), robot (orange)
PUCK_BANDS = [
    (np.array([100, 120,  60], np.uint8),
     np.array([125, 255, 255], np.uint8))
]
player_lower = np.array([40,  80,  50], np.uint8)
player_upper = np.array([85, 255, 255], np.uint8)
robot_lower  = np.array([ 7, 180, 180], np.uint8)
robot_upper  = np.array([13, 255, 255], np.uint8)

# Networking (kept for future use)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_addr   = ('192.168.4.1', 2222)

# =============================================================
# Serial helpers
# =============================================================

def autodetect_serial_port():
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
    if port is None:
        port = autodetect_serial_port()
    if port is None:
        print("[WARN] No serial port found. Running without Arduino.")
        return None
    try:
        s = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)
        print(f"[INFO] Serial connected: {port}")
        return s
    except Exception as e:
        print(f"[WARN] Could not open serial '{port}': {e}. Running without Arduino.")
        return None


ser = open_serial()


def send_data_to_arduino(step1, step2):
    """Sends 'step1,step2\n' if serial is available."""
    if ser is None:
        return
    try:
        ser.write(f"{step2},{step1}\n".encode())
        resp = ser.readline().decode(errors='ignore').strip()
        if resp:
            print(f"[Arduino] {resp}")
    except serial.SerialException as e:
        print(f"[Serial Error] {e}")

# =============================================================
# Camera
# =============================================================

def open_camera(prefer=(1, 0, 2, 3)):
    for idx in prefer:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS,          FPS)
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"[INFO] Camera opened on index {idx}, frame {frame.shape[1]}x{frame.shape[0]}")
            return cap
        cap.release()
    raise RuntimeError("No camera produced frames. Check connections and indices.")

# (moved cap = open_camera() to after optional self-tests)

# =============================================================
# Zone / boundary helpers (HARD ENFORCEMENT)
# =============================================================

def robot_zone_is_right_raw() -> bool:
    return OPPONENT_SIDE.lower() == 'left'


def midline_x(w: int) -> int:
    return MIDDLE_LINE_X


def robot_zone_is_right_eff(w: int) -> bool:
    # Effective side after optional horizontal mirror
    return robot_zone_is_right_raw()


def defense_x_for_frame(w: int) -> int:
    """Return the X position of our vertical DEFENSE LINE (inner edge of our crease)."""
    mid = midline_x(w)
    if robot_zone_is_right_eff(w):
        x = w - GOAL_CREASE_DEPTH
        min_x = mid + ROBOT_SIDE_MARGIN
        max_x = w - 1 - ARENA_MARGIN_X
        return max(min_x, min(x, max_x))
    else:
        x = GOAL_CREASE_DEPTH
        min_x = ARENA_MARGIN_X
        max_x = mid - ROBOT_SIDE_MARGIN
        return max(min_x, min(x, max_x))


def clamp_to_robot_zone(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    """Clamp ANY point to legal robot area (half + margins + crease on OUR side)."""
    mid = midline_x(w)
    if robot_zone_is_right_eff(w):
        min_x = mid + ROBOT_SIDE_MARGIN
        max_x = w - 1 - ARENA_MARGIN_X
        # don't enter the crease itself (stay at or inside defense line)
        max_x = min(max_x, defense_x_for_frame(w))
    else:
        min_x = ARENA_MARGIN_X
        max_x = mid - ROBOT_SIDE_MARGIN
        min_x = max(min_x, defense_x_for_frame(w))

    x = max(min_x, min(x, max_x))

    min_y = ARENA_MARGIN_Y
    max_y = min(h - 1 - ARENA_MARGIN_Y, h - 1)
    y = max(min_y, min(y, max_y))
    return x, y

# =============================================================
# Geometry utilities
# =============================================================

def _closest_point_on_segment(p, a, b):
    p = np.asarray(p, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-9:
        q = a
    else:
        t = float(np.dot(p - a, ab) / ab_len2)
        t = max(0.0, min(1.0, t))
        q = a + t * ab
    d2 = float(np.dot(p - q, p - q))
    return (int(q[0]), int(q[1])), d2


def closest_point_on_polyline(segments, p):
    best_q, best_d2, best_i = None, float("inf"), -1
    for i, (a, b) in enumerate(segments):
        q, d2 = _closest_point_on_segment(p, a, b)
        if d2 < best_d2:
            best_q, best_d2, best_i = q, d2, i
    return best_q, best_d2, best_i


def intersect_polyline_with_vertical_x(segments, x: int):
    """Return the FIRST intersection (along path order) between segments and vertical line x.
    If none, return None.
    """
    for (a, b) in segments:
        x0, y0 = a
        x1, y1 = b
        if (x0 - x) == 0 and (x1 - x) == 0:
            return (x, int((y0 + y1) / 2))
        if (x0 <= x <= x1) or (x1 <= x <= x0):
            if x1 != x0:
                t = (x - x0) / (x1 - x0)
                y = y0 + t * (y1 - y0)
                return (x, int(round(y)))
    return None

# =============================================================
# Prediction
# =============================================================

def predict_enemy_shot_until_boundary(px, py, ex, ey, frame_w, frame_h, goal_margin=GOAL_MARGIN_PIX, max_bounces=2):
    """Predict path from puck toward whichever side it's currently heading, with horizontal bounces.
    Stops on near side goal line.
    """
    dx, dy = (px - ex), (py - ey)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return [], (px, py)

    p0 = np.array([float(px), float(py)], dtype=np.float32)
    v  = np.array([float(dx), float(dy)], dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return [], (px, py)
    v /= n

    heading_left = v[0] < 0
    goal_x = float(goal_margin if heading_left else frame_w - 1 - goal_margin)

    segments = []
    eps = 1e-3

    for _ in range(max_bounces + 1):
        t_goal   = (goal_x - p0[0]) / v[0] if abs(v[0]) > 1e-9 else np.inf
        t_top    = (0.0 - p0[1]) / v[1] if v[1] < 0 else np.inf
        t_bottom = ((frame_h - 1.0) - p0[1]) / v[1] if v[1] > 0 else np.inf
        t_wall   = min(t_top, t_bottom)

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

# =============================================================
# Detection
# =============================================================

def detect_puck(frame, bands=PUCK_BANDS, min_radius=14, max_radius=90, use_blue_bias=True):
    global puck_x, puck_y
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = None
    for low, up in bands:
        m = cv2.inRange(hsv, low, up)
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    if use_blue_bias:
        b, g, r = cv2.split(frame)
        dom = (b.astype(np.int16) > (g.astype(np.int16) + 15)) & (b.astype(np.int16) > (r.astype(np.int16) + 15))
        dom = dom.astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, dom)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest = None
    max_radius_found = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circularity = 4 * np.pi * area / (perim ** 2)
        if circularity < 0.35:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if min_radius <= radius <= max_radius and radius > max_radius_found:
            largest = (int(x), int(y), int(radius))
            max_radius_found = radius

    if largest is not None:
        puck_x, puck_y = largest[0], largest[1]

    return largest


def detect_blob(frame, lower, upper, min_radius=24, max_radius=120, min_area=70, circ_thresh=0.20):
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
        circularity = 4 * np.pi * area / (perim ** 2)
        if circularity < circ_thresh:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if min_radius <= radius <= max_radius and radius > best_r:
            best = (int(x), int(y), int(radius))
            best_r = radius
    return best


def draw_circles_on_frame(frame, detection, color=(255, 0, 0)):
    if detection is not None:
        x, y, radius = detection
        cv2.circle(frame, (x, y), radius, color, 2)
        cv2.circle(frame, (x, y), 3, color, -1)

# =============================================================
# Motion mapping (with HARD CLAMP)
# =============================================================

def pixel_to_steps(target_x, target_y, robot_x, robot_y, w, h):
    """CoreXY mapping using actual frame size (w,h).
    BOTH target and measured robot are clamped to the legal zone first.
    Returns (delta_step1, delta_step2).
    """
    tx, ty = clamp_to_robot_zone(int(target_x), int(target_y), w, h)
    rx, ry = clamp_to_robot_zone(int(robot_x),  int(robot_y),  w, h)

    # map pixels -> nominal step coordinates
    tx_steps = int(np.interp(tx, [0, w - 1], [0, X_STEPS_MAX]))
    ty_steps = int(np.interp(ty, [0, h - 1], [0, Y_STEPS_MAX]))
    rx_steps = int(np.interp(rx, [0, w - 1], [0, X_STEPS_MAX]))
    ry_steps = int(np.interp(ry, [0, h - 1], [0, Y_STEPS_MAX]))

    # optional motor-axis inversions (fixes direction without changing camera flip)
    if MOTOR_INVERT_X:
        tx_steps = X_STEPS_MAX - tx_steps
        rx_steps = X_STEPS_MAX - rx_steps
    if MOTOR_INVERT_Y:
        ty_steps = Y_STEPS_MAX - ty_steps
        ry_steps = Y_STEPS_MAX - ry_steps

    target_step1 = ty_steps - tx_steps
    target_step2 = ty_steps + tx_steps
    robot_step1  = ry_steps - rx_steps
    robot_step2  = ry_steps + rx_steps

    return target_step1 - robot_step1, target_step2 - robot_step2

# =============================================================
# UI helpers
# =============================================================

def draw_middle_line(frame, x):
    h = frame.shape[0]
    cv2.line(frame, (x, 0), (x, h), (255, 100, 0), 3)
    for y in range(0, h, 100):
        cv2.line(frame, (x - 10, y), (x + 10, y), (255, 100, 0), 2)
    cv2.putText(frame, "ROBOT BOUNDARY", (x - 110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)


def shade_rect(frame, x1, y1, x2, y2, color=(60, 60, 60), alpha=0.18):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_margins_and_crease(frame):
    h, w = frame.shape[:2]
    # Side & top/bottom safety
    shade_rect(frame, 0, 0, ARENA_MARGIN_X, h, (60, 60, 200), 0.18)
    shade_rect(frame, w - ARENA_MARGIN_X, 0, w, h, (60, 60, 200), 0.18)
    shade_rect(frame, 0, 0, w, ARENA_MARGIN_Y, (60, 200, 60), 0.18)
    shade_rect(frame, 0, h - ARENA_MARGIN_Y, w, h, (60, 200, 60), 0.18)

    # Goal creases (shade only our crease for clarity)
    dx = defense_x_for_frame(w)
    if robot_zone_is_right_eff(w):
        shade_rect(frame, w - GOAL_CREASE_DEPTH, 0, w, h, (0, 140, 255), 0.12)
        cv2.line(frame, (dx, 0), (dx, h), (0, 140, 255), 2)
        cv2.putText(frame, "DEFENSE LINE", (max(10, dx - 120), 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    else:
        shade_rect(frame, 0, 0, GOAL_CREASE_DEPTH, h, (0, 140, 255), 0.12)
        cv2.line(frame, (dx, 0), (dx, h), (0, 140, 255), 2)
        cv2.putText(frame, "DEFENSE LINE", (max(10, dx - 120), 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

# =============================================================
# Optional self-tests (run: python thisfile.py --selftest)
# =============================================================

def _run_selftests():
    print("[TEST] Running self-tests...")
    # 1) intersect with vertical line
    segs = [((10, 10), (110, 60))]
    inter = intersect_polyline_with_vertical_x(segs, 60)
    assert inter == (60, 35), f"intersect expected (60,35), got {inter}"

    # 2) clamp_to_robot_zone keeps inside legal X for each side
    global OPPONENT_SIDE
    orig_side = OPPONENT_SIDE
    w, h = 1920, 1080

    OPPONENT_SIDE = 'left'  # robot on right half
    x, y = clamp_to_robot_zone(1910, 10, w, h)
    assert x <= defense_x_for_frame(w) and x >= midline_x(w) + ROBOT_SIDE_MARGIN
    assert ARENA_MARGIN_Y <= y <= h-1-ARENA_MARGIN_Y

    OPPONENT_SIDE = 'right'  # robot on left half
    x, y = clamp_to_robot_zone(10, 10, w, h)
    assert x >= defense_x_for_frame(w) and x <= midline_x(w) - ROBOT_SIDE_MARGIN

    # 3) pixel_to_steps yields zero deltas when target == robot
    d1, d2 = pixel_to_steps(500, 400, 500, 400, w, h)
    assert d1 == 0 and d2 == 0

    OPPONENT_SIDE = orig_side
    print("[TEST] All tests passed.")

if '--selftest' in sys.argv:
    _run_selftests()
    sys.exit(0)

# =============================================================
# Main loop
# =============================================================

# open camera only for runtime mode
cap = open_camera()

puck_x = puck_y = 0
player_x = player_y = 0
robot_x = robot_y = 0
last_robot_xy = None
last_send = 0.0

print("[INFO] Running. Press ESC or 'q' to quit. Press 'p' to toggle mode. 'o' = flip sides, 'f' = camera flip only.")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[ERROR] Camera read failed.")
        break

    h, w = frame.shape[:2]

    # Camera mirror if enabled (display & detection only; logic remains same)
    if CAMERA_MIRROR_X:
        frame = cv2.flip(frame, 1)

    defense_x = defense_x_for_frame(w)

    # --- Detect ---
    puck_det   = detect_puck(frame)
    player_det = detect_blob(frame, player_lower, player_upper, min_radius=26, max_radius=110, min_area=80, circ_thresh=0.15)
    robot_det  = detect_blob(frame, robot_lower,  robot_upper,  min_radius=24, max_radius=100, min_area=70, circ_thresh=0.25)

    if player_det is not None:
        player_x, player_y = player_det[0], player_det[1]
    if robot_det is not None:
        robot_x, robot_y = robot_det[0], robot_det[1]
        last_robot_xy = (robot_x, robot_y)
    elif last_robot_xy is not None:
        robot_x, robot_y = last_robot_xy
    else:
        robot_x, robot_y = w // 2, h // 2

    # --- Visuals ---
    draw_circles_on_frame(frame, puck_det,   color=(0, 0, 0))     # puck (black)
    draw_circles_on_frame(frame, player_det, color=(0, 255, 0))   # opponent (green)
    draw_circles_on_frame(frame, robot_det,  color=(0, 0, 255))   # robot (red/blue)

    draw_middle_line(frame, midline_x(w))
    draw_margins_and_crease(frame)

    sent_this_frame = False

    # --- Defense logic (CONTROL_MODE == 'line') ---
    if CONTROL_MODE == "line" and (puck_det is not None and player_det is not None):
        # Draw enemy->puck vector
        cv2.line(frame, (player_x, player_y), (puck_x, puck_y), (0, 165, 255), 2)

        segs, impact = predict_enemy_shot_until_boundary(
            puck_x, puck_y, player_x, player_y,
            frame_w=w, frame_h=h,
            goal_margin=GOAL_MARGIN_PIX,
            max_bounces=2
        )

        # Visualize predicted path
        for (a, b) in segs:
            cv2.line(frame, a, b, (0, 255, 255), 2)
        cv2.circle(frame, impact, 6, (0, 255, 255), -1)

        # INTERSECT with our DEFENSE LINE
        intercept = intersect_polyline_with_vertical_x(segs, defense_x)

        if intercept is None:
            # Not heading toward us; patrol along defense line by y = puck_y
            intercept = (defense_x, puck_y)

        # Clamp target strictly to legal zone (so we never command outside)
        tx, ty = clamp_to_robot_zone(intercept[0], intercept[1], w, h)

        # Draw chosen target
        cv2.circle(frame, (tx, ty), 7, (255, 0, 255), -1)
        cv2.line(frame, (robot_x, robot_y), (tx, ty), (255, 0, 255), 2)

        # Only move if far enough
        d2 = (robot_x - tx) ** 2 + (robot_y - ty) ** 2
        if d2 > (PIXEL_DEADBAND * PIXEL_DEADBAND):
            s1, s2 = pixel_to_steps(tx, ty, robot_x, robot_y, w, h)
            out_a, out_b = (s2, s1) if SWAP_STEPS else (s1, s2)
            now = time.time()
            if now - last_send >= SEND_MIN_INTERVAL:
                send_data_to_arduino(out_a, out_b)
                last_send = now
                sent_this_frame = True

    # --- Fallback: track puck but X locked to defense line ---
    if CONTROL_MODE == "puck" and puck_det is not None and not sent_this_frame:
        tx, ty = clamp_to_robot_zone(defense_x, puck_y, w, h)  # x stays on defense line
        s1, s2 = pixel_to_steps(tx, ty, robot_x, robot_y, w, h)
        out_a, out_b = (s2, s1) if SWAP_STEPS else (s1, s2)
        now = time.time()
        if now - last_send >= SEND_MIN_INTERVAL:
            send_data_to_arduino(out_a, out_b)
            last_send = now

    status = f"Mode:{CONTROL_MODE}  Opp:{OPPONENT_SIDE}  CamFlip:{CAMERA_MIRROR_X}  InvX:{MOTOR_INVERT_X} InvY:{MOTOR_INVERT_Y}  Puck:{puck_det is not None} Enemy:{player_det is not None} Robot:{robot_det is not None}"
    cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)
    print(f"Puck({puck_x:4d},{puck_y:4d})  Robot({robot_x:4d},{robot_y:4d})  Sent:{sent_this_frame}")

    cv2.imshow("Air Hockey – Defense Line", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):  # ESC or q
        break
    if key == ord('p'):
        CONTROL_MODE = "puck" if CONTROL_MODE == "line" else "line"
        print(f"[INFO] CONTROL_MODE -> {CONTROL_MODE}")
    elif key == ord('o'):
        OPPONENT_SIDE = 'left' if OPPONENT_SIDE == 'right' else 'right'
        side = 'right' if robot_zone_is_right_raw() else 'left'
        print(f"[INFO] OPPONENT_SIDE -> {OPPONENT_SIDE} (robot zone now {side})")
    elif key == ord('f'):
        CAMERA_MIRROR_X = not CAMERA_MIRROR_X
        print(f"[INFO] CAMERA_MIRROR_X -> {CAMERA_MIRROR_X} (logic unchanged)")
    elif key == ord('x'):
        MOTOR_INVERT_X = not MOTOR_INVERT_X
        print(f"[INFO] MOTOR_INVERT_X -> {MOTOR_INVERT_X}")
    elif key == ord('y'):
        MOTOR_INVERT_Y = not MOTOR_INVERT_Y
        print(f"[INFO] MOTOR_INVERT_Y -> {MOTOR_INVERT_Y}")

cap.release()
cv2.destroyAllWindows()
