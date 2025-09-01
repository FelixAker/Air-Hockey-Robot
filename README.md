# Air Hockey – Vision + CoreXY Defense & Auto-Attack

Python vision controller for a CoreXY air-hockey robot. Tracks the **puck**, **opponent**, and **robot**, predicts shots, and commands the robot to defend or auto-attack.

![Demo](docs/demo.gif)

---

## Features

- **Two modes**
  - **line**: defend on a crease-aligned vertical line, intercept predicted shots
  - **puck**: track the puck vertically with X fixed to defense line
- **Auto-attack:** when the puck stays in our half too long, approach and push it back
- **Safety clamps:** keep the robot in its half, away from rails and crease
- **CoreXY mapping:** pixels → step space with smooth bursts
- **HSV vision:** puck, opponent, and robot detected via simple color ranges
- **Hotkeys** for live control

---

## Hardware

- **Robot:** CoreXY gantry (X/Y) driven by Arduino
- **Camera:** phone camera or USB webcam
- **Computer:** Python 3.9+ on macOS/Linux/Windows

---

## Camera Notes

The script tries to open a camera at index `0` by default.  
On some systems (e.g., when using a phone as webcam), the camera may show up at index `1` or higher.  

If you see `[ERROR] Camera read failed`, try changing the index in `open_camera(prefer=(0,1,2,3))` inside the script.

---

## Run

1. Plug in Arduino and camera
2. Enter the Arduino port number for your device 
3. (Optional) create `config.txt`:
   ```ini
   [PARAMS]
   FPS=60
   ```
4. Run:
   ```bash
   python air_hockey.py
   ```

---

## Controls

| Key | Action |
| --- | --- |
| **p** | Toggle line ↔ puck mode |
| **o** | Flip opponent side |
| **f** | Flip camera image (visual only) |
| **b** | Toggle border ignore (debug only) |
| **Esc / q** | Quit |

---

## Tune

- `MIDDLE_LINE_X` → set to table midline in pixels
- `OPPONENT_SIDE` → 'left' or 'right'
- `ARENA_MARGIN_Y`, `GOAL_CREASE_DEPTH`, `ROBOT_SIDE_MARGIN` → keep robot safe
- `HSV ranges` → adjust for lighting
- `X_STEPS_MAX`, `Y_STEPS_MAX` → calibrate travel span
- `PIXEL_DEADBAND`, `SEND_MIN_INTERVAL`, `MAX_BURST` → adjust smoothness
- `ATTACK_* constants` → change aggressiveness

---

## Structure

```
air_hockey.py
config.txt
README.md
docs/demo.gif
```
---

## Controllers

- **Robot.py** — Main controller. Holds an aligned defense line (or “puck” mode that tracks vertically), triggers auto-attack if the puck lingers in robot's half, applies CoreXY mapping and safety clamps and streams step bursts to Arduino over serial.

- **RobotVersion2.py** — Proactive controller. Predicts the puck path and repositions off the line to intercept earlier; uses the same serial protocol and tuning knobs.

- **DefenseOnlyVersion2.py** — Defense only feature of RobotVersion2.py.


