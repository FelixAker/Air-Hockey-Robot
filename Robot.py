import cv2
import numpy as np
import socket
import serial
import time
import configparser
import glob
import sys

# =============================================================
# Config & knobs
# =============================================================
config = configparser.ConfigParser()
config.read('config.txt')

CAM_PIX_TO_MM = float(config.get('PARAMS', 'CAMPIXTOMM', fallback=1.25))
TABLE_LENGTH  = int(config.get('PARAMS', 'TABLELENGTH',  fallback=710))
TABLE_WIDTH   = int(config.get('PARAMS', 'TABLEWIDTH',   fallback=400))
FPS           = int(config.get('PARAMS', 'FPS',          fallback=60))

# Which half is OURS? Change to "left" if your robot plays left half.
ROBOT_SIDE = 'left'

# Control & motion
CONTROL_MODE        = "line"   # "line" (defense using line prediction) or "puck" (follow puck)
SWAP_STEPS          = True     # True keeps your historical (step2, step1) send order
PIXEL_DEADBAND      = 6        # pixels (squared used for comparison)
SEND_MIN_INTERVAL   = 0.010    # serial send rate limit (seconds)
GOAL_MARGIN_PIX     = 8        # distance from side wall to treat as "goal line"

# --- ATTACK tunables ---
REQUIRE_STILLNESS_BEFORE_ATTACK = True  # set False to attack immediately on our side
PUCK_STILLNESS_SECS     = 2.0
PUCK_MOVE_EPS_PIX       = 5.0
ATTACK_BEHIND_OFFSET    = 60
ATTACK_STRIKE_OVERSHOOT = 180
ATTACK_CLOSE_ENOUGH     = 18
ATTACK_STRIKE_TIMEOUT   = 0.6
ATTACK_RECOVER_TIMEOUT  = 1.0

# --- HARD BORDERS (our half + arena margins + optional crease) ---
IGNORE_BORDERS     = False   # toggle with 'b' at runtime (visuals + clamps)
ARENA_MARGIN_X     = 80      # keep away from left/right rails
ARENA_MARGIN_Y     = 80      # keep away from top/bottom rails
ROBOT_SIDE_MARGIN  = 100     # do not cross near the midline into opponent side
GOAL_CREASE_DEPTH  = 0       # 0 = allow full half for attack; set >0 to forbid entering crease
MIDDLE_LINE_OFFSET = 0       # if your table center is offset in the image, set pixels here

# HSV thresholds
PUCK_BANDS = [
    (np.array([100, 120,  60], np.uint8),
     np.array([125, 255, 255], np.uint8))
]
player_lower = np.array([40,  80,  50], np.uint8)
player_upper = np.array([85, 255, 255], np.uint8)
robot_lower  = np.array([ 7, 180, 180], np.uint8)
robot_upper  = np.array([13, 255, 255], np.uint8)

# =============================================================
# IO: Serial + Camera
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
        import glob as _glob
        for p in _glob.glob(pat):
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
    if ser is None:
        return
    try:
        ser.write(f"{step1},{step2}\n".encode())
        resp = ser.readline().decode(errors='ignore').strip()
        if resp:
            print(f"[Arduino] {resp}")
    except serial.SerialException as e:
        print(f"[Serial Error] {e}")

def open_camera(prefer=(1,0,2,3)):
    for idx in prefer:
        cap = cv2.VideoCapture(1)
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

cap = open_camera()

# =============================================================
# Border helpers (HARD ENFORCEMENT)
# =============================================================
def midline_x(w: int) -> int:
    return (w // 2) + int(MIDDLE_LINE_OFFSET)

def robot_is_right() -> bool:
    return ROBOT_SIDE == 'right'

def defense_line_x(w: int) -> int:
    # The inner edge of our crease (optional). If GOAL_CREASE_DEPTH==0 we won't use it to clamp.
    return (w - GOAL_CREASE_DEPTH) if robot_is_right() else GOAL_CREASE_DEPTH

def clamp_to_robot_zone(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    if IGNORE_BORDERS:
        x = max(0, min(int(x), w-1))
        y = max(0, min(int(y), h-1))
        return x, y

    mid = midline_x(w)

    if robot_is_right():
        min_x = mid + ROBOT_SIDE_MARGIN
        max_x = w - 1 - ARENA_MARGIN_X
        if GOAL_CREASE_DEPTH > 0:
            # keep OUT of our crease (stay at or left of defense_line_x)
            max_x = min(max_x, defense_line_x(w))
    else:
        min_x = ARENA_MARGIN_X
        max_x = mid - ROBOT_SIDE_MARGIN
        if GOAL_CREASE_DEPTH > 0:
            # keep OUT of our crease (stay at or right of defense_line_x)
            min_x = max(min_x, defense_line_x(w))

    x = max(min_x, min(int(x), max_x))

    min_y = ARENA_MARGIN_Y
    max_y = h - 1 - ARENA_MARGIN_Y
    y = max(min_y, min(int(y), max_y))
    return x, y

def is_puck_on_robot_side(px: int, w: int) -> bool:
    mid = midline_x(w)
    return (px >= mid) if robot_is_right() else (px <= mid)

# Visual helpers
def draw_middle_line(frame, x):
    h = frame.shape[0]
    cv2.line(frame, (x, 0), (x, h), (255, 100, 0), 3)
    for y in range(0, h, 100):
        cv2.line(frame, (x-10, y), (x+10, y), (255, 100, 0), 2)
    cv2.putText(frame, "MIDLINE", (max(10, x-70), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

def shade_rect(frame, x1, y1, x2, y2, color=(60, 60, 60), alpha=0.18):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_margins_and_zone(frame):
    if IGNORE_BORDERS:
        return
    h, w = frame.shape[:2]
    # arena safety
    shade_rect(frame, 0, 0, ARENA_MARGIN_X, h, (60, 60, 200), 0.18)
    shade_rect(frame, w-ARENA_MARGIN_X, 0, w, h, (60, 60, 200), 0.18)
    shade_rect(frame, 0, 0, w, ARENA_MARGIN_Y, (60, 200, 60), 0.18)
    shade_rect(frame, 0, h-ARENA_MARGIN_Y, w, h, (60, 200, 60), 0.18)

    # midline & our half highlight
    mid = midline_x(w)
    draw_middle_line(frame, mid)
    if robot_is_right():
        shade_rect(frame, 0, 0, mid + ROBOT_SIDE_MARGIN, h, (40, 40, 40), 0.10)  # shaded opponent zone + margin
        if GOAL_CREASE_DEPTH > 0:
            cv2.line(frame, (defense_line_x(w), 0), (defense_line_x(w), h), (0, 140, 255), 2)
    else:
        shade_rect(frame, mid - ROBOT_SIDE_MARGIN, 0, w, h, (40, 40, 40), 0.10)
        if GOAL_CREASE_DEPTH > 0:
            cv2.line(frame, (defense_line_x(w), 0), (defense_line_x(w), h), (0, 140, 255), 2)

# =============================================================
# Geometry & prediction
# =============================================================
def draw_segments(frame, segments, color=(0, 255, 255), thickness=2):
    for (a, b) in segments:
        cv2.line(frame, a, b, color, thickness)

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

def predict_enemy_shot_until_boundary(px, py, ex, ey, frame_w, frame_h, goal_margin=GOAL_MARGIN_PIX, max_bounces=2):
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
        v[1] = -v[1]
        p0 = p1 + eps * v

    last_pt = segments[-1][1] if segments else (px, py)
    return segments, last_pt

# =============================================================
# Detection
# =============================================================
puck_x = puck_y = 0
player_x = player_y = 0
robot_x = robot_y = 0
last_robot_xy = None

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

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest = None
    best_r = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circ = 4 * np.pi * area / (perim ** 2)
        if circ < 0.35:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if min_radius <= r <= max_radius and r > best_r:
            largest = (int(x), int(y), int(r))
            best_r = r

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
        circ = 4 * np.pi * area / (perim ** 2)
        if circ < circ_thresh:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if min_radius <= r <= max_radius and r > best_r:
            best = (int(x), int(y), int(r))
            best_r = r
    return best

def draw_circles_on_frame(frame, detection, color=(255, 0, 0)):
    if detection is None:
        return
    x, y, r = detection
    cv2.circle(frame, (x, y), r, color, 2)
    cv2.circle(frame, (x, y), 3, color, -1)

# =============================================================
# Mapping (clamps INSIDE)
# =============================================================
def pixel_to_steps(target_x, target_y, robot_x_, robot_y_, w, h):
    # Final safety clamp to legal zone
    tx, ty = clamp_to_robot_zone(target_x, target_y, w, h)
    rx, ry = clamp_to_robot_zone(robot_x_,  robot_y_,  w, h)

    # Map pixels to step-space
    tx_steps = int(np.interp(tx, [0, w - 1], [0, 8000]))
    ty_steps = int(np.interp(ty, [0, h - 1], [0, 8000]))
    rx_steps = int(np.interp(rx, [0, w - 1], [0, 8000]))
    ry_steps = int(np.interp(ry, [0, h - 1], [0, 8000]))

    t_s1, t_s2 = (ty_steps - tx_steps), (ty_steps + tx_steps)
    r_s1, r_s2 = (ry_steps - rx_steps), (ry_steps + rx_steps)

    d1 = t_s1 - r_s1
    d2 = t_s2 - r_s2
    return d1, d2

# =============================================================
# ATTACK state machine
# =============================================================
mode = 'DEFENSE'   # 'DEFENSE' | 'ATTACK_APPROACH' | 'ATTACK_STRIKE' | 'ATTACK_RECOVER'
mode_ts = time.time()
last_puck_pos = None
last_puck_move_time = time.time()

def puck_is_still(now_ts):
    return (now_ts - last_puck_move_time) >= PUCK_STILLNESS_SECS

def compute_attack_targets(px, py, w, h):
    # Stage behind the puck on our side, then strike toward opponent side
    if robot_is_right():
        behind_raw = (px + ATTACK_BEHIND_OFFSET, py)
        strike_raw = (px - ATTACK_STRIKE_OVERSHOOT, py)
    else:
        behind_raw = (px - ATTACK_BEHIND_OFFSET, py)
        strike_raw = (px + ATTACK_STRIKE_OVERSHOOT, py)
    # Hard clamp to our legal zone (prevents stepping into opponent half or walls)
    bx, by = clamp_to_robot_zone(behind_raw[0], behind_raw[1], w, h)
    sx, sy = clamp_to_robot_zone(strike_raw[0], strike_raw[1], w, h)
    return (bx, by), (sx, sy)

# =============================================================
# Main loop
# =============================================================
last_send = 0.0
print("[INFO] Running. Press ESC or 'q' to quit. Press 'p' to toggle mode. 'b' toggles borders.")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[ERROR] Camera read failed.")
        break

    h, w = frame.shape[:2]
    now = time.time()

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
    draw_circles_on_frame(frame, puck_det,   color=(0, 0, 0))
    draw_circles_on_frame(frame, player_det, color=(0, 255, 0))
    draw_circles_on_frame(frame, robot_det,  color=(0, 0, 255))
    draw_margins_and_zone(frame)

    # --- Track puck stillness ---
    if puck_det is not None:
        if last_puck_pos is None:
            last_puck_pos = (puck_x, puck_y)
            last_puck_move_time = now
        else:
            dx = float(puck_x - last_puck_pos[0])
            dy = float(puck_y - last_puck_pos[1])
            if (dx*dx + dy*dy) > (PUCK_MOVE_EPS_PIX * PUCK_MOVE_EPS_PIX):
                last_puck_move_time = now
            last_puck_pos = (puck_x, puck_y)

    # --- State selection strictly by side ---
    sent_this_frame = False
    control_target = None

    if puck_det is None:
        mode = 'DEFENSE'
    else:
        on_robot_side = is_puck_on_robot_side(puck_x, w)
        if on_robot_side:
            if mode == 'DEFENSE':
                if (not REQUIRE_STILLNESS_BEFORE_ATTACK) or puck_is_still(now):
                    mode = 'ATTACK_APPROACH'
                    mode_ts = now
        else:
            mode = 'DEFENSE'

    # Abort ATTACK if puck leaves our side or is lost
    if mode.startswith('ATTACK'):
        if (puck_det is None) or (not is_puck_on_robot_side(puck_x, w)):
            mode = 'DEFENSE'
            mode_ts = now

    # --- ATTACK ---
    if mode.startswith('ATTACK') and puck_det is not None:
        (bx, by), (sx, sy) = compute_attack_targets(puck_x, puck_y, w, h)

        if mode == 'ATTACK_APPROACH':
            control_target = (bx, by)
            d2 = (robot_x - bx)**2 + (robot_y - by)**2
            if d2 <= (ATTACK_CLOSE_ENOUGH**2):
                mode = 'ATTACK_STRIKE'
                mode_ts = now

        elif mode == 'ATTACK_STRIKE':
            control_target = (sx, sy)
            d2 = (robot_x - sx)**2 + (robot_y - sy)**2
            if d2 <= (ATTACK_CLOSE_ENOUGH**2) or (now - mode_ts) >= ATTACK_STRIKE_TIMEOUT:
                mode = 'ATTACK_RECOVER'
                mode_ts = now

        elif mode == 'ATTACK_RECOVER':
            # go to a neutral point near center of our half
            mid = midline_x(w)
            if robot_is_right():
                home_x = (mid + ROBOT_SIDE_MARGIN + (w - ARENA_MARGIN_X)) // 2
            else:
                home_x = (ARENA_MARGIN_X + (mid - ROBOT_SIDE_MARGIN)) // 2
            home_y = h // 2
            home_x, home_y = clamp_to_robot_zone(home_x, home_y, w, h)
            control_target = (home_x, home_y)
            d2 = (robot_x - home_x)**2 + (robot_y - home_y)**2
            if d2 <= (ATTACK_CLOSE_ENOUGH**2) or (now - mode_ts) >= ATTACK_RECOVER_TIMEOUT:
                mode = 'DEFENSE'
                mode_ts = now

        if control_target is not None:
            # Draw and send (with final clamp in pixel_to_steps too)
            cx, cy = clamp_to_robot_zone(control_target[0], control_target[1], w, h)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), 2)
            cv2.line(frame, (robot_x, robot_y), (cx, cy), (0, 0, 255), 2)

            s1, s2 = pixel_to_steps(cx, cy, robot_x, robot_y, w, h)
            out_a, out_b = (s2, s1) if SWAP_STEPS else (s1, s2)
            now2 = time.time()
            if now2 - last_send >= SEND_MIN_INTERVAL:
                send_data_to_arduino(out_a, out_b)
                last_send = now2
                sent_this_frame = True

    # --- DEFENSE (line prediction) ---
    if mode == 'DEFENSE':
        if CONTROL_MODE == "line" and (puck_det is not None and player_det is not None):
            cv2.line(frame, (player_x, player_y), (puck_x, puck_y), (0, 165, 255), 2)

            segs, impact = predict_enemy_shot_until_boundary(
                puck_x, puck_y, player_x, player_y,
                frame_w=w, frame_h=h,
                goal_margin=GOAL_MARGIN_PIX, max_bounces=2
            )

            draw_segments(frame, segs, color=(0, 255, 255), thickness=2)
            cv2.circle(frame, impact, 6, (0, 255, 255), -1)

            if segs:
                closest_pt, d2, _ = closest_point_on_polyline(segs, (robot_x, robot_y))
                tx, ty = clamp_to_robot_zone(closest_pt[0], closest_pt[1], w, h)
                cv2.circle(frame, (tx, ty), 6, (255, 0, 255), -1)
                cv2.line(frame, (robot_x, robot_y), (tx, ty), (255, 0, 255), 2)

                if d2 > (PIXEL_DEADBAND * PIXEL_DEADBAND):
                    s1, s2 = pixel_to_steps(tx, ty, robot_x, robot_y, w, h)
                    out_a, out_b = (s2, s1) if SWAP_STEPS else (s1, s2)
                    now2 = time.time()
                    if now2 - last_send >= SEND_MIN_INTERVAL:
                        send_data_to_arduino(out_a, out_b)
                        last_send = now2
                        sent_this_frame = True
            else:
                # fallback: track puck directly (clamped)
                tx, ty = clamp_to_robot_zone(puck_x, puck_y, w, h)
                s1, s2 = pixel_to_steps(tx, ty, robot_x, robot_y, w, h)
                out_a, out_b = (s2, s1) if SWAP_STEPS else (s1, s2)
                now2 = time.time()
                if now2 - last_send >= SEND_MIN_INTERVAL:
                    send_data_to_arduino(out_a, out_b)
                    last_send = now2
                    sent_this_frame = True

        if CONTROL_MODE == "puck" and puck_det is not None and not sent_this_frame:
            tx, ty = clamp_to_robot_zone(puck_x, puck_y, w, h)
            s1, s2 = pixel_to_steps(tx, ty, robot_x, robot_y, w, h)
            out_a, out_b = (s2, s1) if SWAP_STEPS else (s1, s2)
            now2 = time.time()
            if now2 - last_send >= SEND_MIN_INTERVAL:
                send_data_to_arduino(out_a, out_b)
                last_send = now2
                sent_this_frame = True

    # --- HUD ---
    side_txt = "ROBOT SIDE" if (puck_det and is_puck_on_robot_side(puck_x, w)) else "OPPONENT SIDE"
    status = f"Mode:{CONTROL_MODE}  State:{mode}  Side:{side_txt}  Borders:{'OFF' if IGNORE_BORDERS else 'ON'}"
    cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)
    print(f"Puck({puck_x:4d},{puck_y:4d})  Robot({robot_x:4d},{robot_y:4d})  Sent:{sent_this_frame}  State:{mode}  {side_txt}")

    cv2.imshow("Air Hockey – Defense + Attack (Hard Borders)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    if key == ord('p'):
        CONTROL_MODE = "puck" if CONTROL_MODE == "line" else "line"
        print(f"[INFO] CONTROL_MODE -> {CONTROL_MODE}")
    if key == ord('b'):
        IGNORE_BORDERS = not IGNORE_BORDERS
        print(f"[INFO] IGNORE_BORDERS -> {IGNORE_BORDERS}")

cap.release()
cv2.destroyAllWindows()
