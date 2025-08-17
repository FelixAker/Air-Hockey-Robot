import cv2
import numpy as np
import socket
import serial
import time
import configparser
import glob
import sys
import os

config = configparser.ConfigParser()
config.read('config.txt')

CAM_PIX_TO_MM = float(config.get('PARAMS', 'CAMPIXTOMM', fallback=1.25))
TABLE_LENGTH  = int(config.get('PARAMS', 'TABLELENGTH',  fallback=710))
TABLE_WIDTH   = int(config.get('PARAMS', 'TABLEWIDTH',   fallback=400))
FPS           = int(config.get('PARAMS', 'FPS',          fallback=60))

CONTROL_MODE        = "line"   # "line" or "puck"
SWAP_STEPS          = True     # True keeps your historical (step2, step1) send order
PIXEL_DEADBAND      = 6        # pixels (squared used for comparison)
SEND_MIN_INTERVAL   = 0.010    # serial send rate limit (seconds)
GOAL_MARGIN_PIX     = 8        # distance from side wall to treat as "goal line"

PUCK_BANDS = [
    (np.array([100, 120,  60], np.uint8),
     np.array([125, 255, 255], np.uint8))
]
player_lower = np.array([40,  80,  50], np.uint8)
player_upper = np.array([85, 255, 255], np.uint8)
robot_lower  = np.array([ 7, 180, 180], np.uint8)
robot_upper  = np.array([13, 255, 255], np.uint8)

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_addr   = ('192.168.4.1', 2222)

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
    """Sends 'step1,step2\\n' if serial is available."""
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

def pixel_to_steps(target_x, target_y, robot_x, robot_y, w, h):
    """
    CoreXY mapping using actual frame size (w,h).
    Returns (delta_step1, delta_step2).
    """
    target_x = int(np.clip(target_x, 0, w - 1))
    target_y = int(np.clip(target_y, 0, h - 1))
    robot_x  = int(np.clip(robot_x,  0, w - 1))
    robot_y  = int(np.clip(robot_y,  0, h - 1))

    target_x_steps = int(np.interp(target_x, [0, w - 1], [0, 8000]))
    target_y_steps = int(np.interp(target_y, [0, h - 1], [0, 8000]))
    robot_x_steps  = int(np.interp(robot_x,  [0, w - 1], [0, 8000]))
    robot_y_steps  = int(np.interp(robot_y,  [0, h - 1], [0, 8000]))

    target_step1 = target_y_steps - target_x_steps
    target_step2 = target_y_steps + target_x_steps
    robot_step1  = robot_y_steps  - robot_x_steps
    robot_step2  = robot_y_steps  + robot_x_steps

    return target_step1 - robot_step1, target_step2 - robot_step2

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
    """
    Start at (px,py), travel along direction (puck - enemy).
    Reflect ONLY on horizontal rails (y=0 and y=frame_h-1).
    Stop when reaching the near side based on direction:
      - if heading left (v.x<0): stop at x <= goal_margin
      - if heading right(v.x>0): stop at x >= frame_w-1-goal_margin
    Returns: (segments, impact_point)
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
    if heading_left:
        goal_x = float(goal_margin)
    else:
        goal_x = float(frame_w - 1 - goal_margin)

    segments = []
    eps = 1e-3

    for _ in range(max_bounces + 1):
        t_goal = (goal_x - p0[0]) / v[0] if abs(v[0]) > 1e-9 else np.inf
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

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest = None
    max_radius_found = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:  # reject tiny noise
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circularity = 4 * np.pi * area / (perim ** 2)
        if circularity < 0.35:  # slightly loose; adjust if false positives
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
        cv2.circle(frame, (x, y), radius, color, 2)  # outer circle
        cv2.circle(frame, (x, y), 3, color, -1)      # center dot

last_send = 0.0
print("[INFO] Running. Press ESC or 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[ERROR] Camera read failed.")
        break

    h, w = frame.shape[:2]

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

    draw_circles_on_frame(frame, puck_det,   color=(0, 0, 0))     # puck (black outline)
    draw_circles_on_frame(frame, player_det, color=(0, 255, 0))   # enemy (green)
    draw_circles_on_frame(frame, robot_det,  color=(0, 0, 255))   # robot (red/blue)

    sent_this_frame = False

    if CONTROL_MODE == "line" and (puck_det is not None and player_det is not None):
        cv2.line(frame, (player_x, player_y), (puck_x, puck_y), (0, 165, 255), 2)

        segs, impact = predict_enemy_shot_until_boundary(
            puck_x, puck_y, player_x, player_y,
            frame_w=w, frame_h=h,
            goal_margin=GOAL_MARGIN_PIX,
            max_bounces=2
        )

        draw_segments(frame, segs, color=(0, 255, 255), thickness=2)
        cv2.circle(frame, impact, 6, (0, 255, 255), -1)

        if segs:
            closest_pt, d2, _ = closest_point_on_polyline(segs, (robot_x, robot_y))
            cv2.circle(frame, closest_pt, 6, (255, 0, 255), -1)
            cv2.line(frame, (robot_x, robot_y), closest_pt, (255, 0, 255), 2)

            if d2 > (PIXEL_DEADBAND * PIXEL_DEADBAND):
                s1, s2 = pixel_to_steps(closest_pt[0], closest_pt[1], robot_x, robot_y, w, h)
                out_a, out_b = (s2, s1) if SWAP_STEPS else (s1, s2)
                now = time.time()
                if now - last_send >= SEND_MIN_INTERVAL:
                    send_data_to_arduino(out_a, out_b)
                    last_send = now
                    sent_this_frame = True
        else:
            if puck_det is not None:
                s1, s2 = pixel_to_steps(puck_x, puck_y, robot_x, robot_y, w, h)
                out_a, out_b = (s2, s1) if SWAP_STEPS else (s1, s2)
                now = time.time()
                if now - last_send >= SEND_MIN_INTERVAL:
                    send_data_to_arduino(out_a, out_b)
                    last_send = now
                    sent_this_frame = True

    if CONTROL_MODE == "puck" and puck_det is not None and not sent_this_frame:
        s1, s2 = pixel_to_steps(puck_x, puck_y, robot_x, robot_y, w, h)
        out_a, out_b = (s2, s1) if SWAP_STEPS else (s1, s2)
        now = time.time()
        if now - last_send >= SEND_MIN_INTERVAL:
            send_data_to_arduino(out_a, out_b)
            last_send = now

    status = f"Mode:{CONTROL_MODE}  Puck:{puck_det is not None} Enemy:{player_det is not None} Robot:{robot_det is not None}"
    cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)
    print(f"Puck({puck_x:4d},{puck_y:4d})  Robot({robot_x:4d},{robot_y:4d})  Sent:{sent_this_frame}")

    cv2.imshow("Air Hockey - Detection & Prediction", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):  # ESC or q
        break
    if key == ord('p'):
        CONTROL_MODE = "puck" if CONTROL_MODE == "line" else "line"
        print(f"[INFO] CONTROL_MODE -> {CONTROL_MODE}")

cap.release()
cv2.destroyAllWindows()