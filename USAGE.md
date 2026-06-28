# Usage Guide

---

## CLI

```bash
# Default — opens a camera window and writes info.md every frame
python INDEX.py

# No window — just writes the md file (good for background use)
python INDEX.py --headless

# Window but no file output
python INDEX.py --no-md

# Write to a different file
python INDEX.py --output C:\Users\you\Desktop\hands.md

# Use a different camera (if 0 doesn't work, try 1)
python INDEX.py --camera 1

# Only track one hand
python INDEX.py --max-hands 1

# Combine flags
python INDEX.py --headless --output mydata.md --max-hands 2
```

Double-clicking `run_debug_tracker.bat` does the same as `python INDEX.py`.

---

## Library — basic use

```python
from INDEX import HandTracker

with HandTracker(write_md=False) as tracker:
    for frame in tracker.frames():
        for hand in frame.hands:
            print(hand.label)       # "Left" or "Right"
            print(hand.gesture)     # "FIST", "PEACE", "PINCH", ...
            print(hand.is_pinching) # bool
```

`tracker.frames()` is a generator that yields one `FrameData` per webcam frame.
It blocks until ESC is pressed or you break out of the loop yourself.

---

## Library — the data you get

### FrameData

```python
frame.hands      # List[Hand]  -- detected hands this frame (0, 1, or 2)
frame.events     # List[MotionEvent]  -- motion gestures fired this frame
frame.frame_n    # int  -- frame counter since start
frame.elapsed    # float  -- seconds since tracker opened
frame.fps        # float  -- rolling fps
frame.width      # int  -- frame width in pixels
frame.height     # int  -- frame height in pixels
frame.image      # np.ndarray  -- annotated BGR frame (ready for cv2.imshow)
```

### Hand

```python
hand.label             # "Left" or "Right"
hand.confidence        # float 0-1
hand.gesture           # str  e.g. "PINCH"
hand.is_pinching       # bool  (thumb + index touching)
hand.landmarks         # list of 21 NormalizedLandmark  (x, y, z each 0-1)
hand.finger_extension  # dict  {"THUMB": True, "INDEX": False, ...}
hand.pinch_distances   # dict  {"INDEX": 0.04, "MIDDLE": 0.18, ...}  (3D normalised)
hand.palm_centre_norm  # (x, y)  palm centre in normalised coords
hand.bounding_box_norm # (x0, y0, x1, y1)  hand bounding box, normalised
hand.palm_normal       # (nx, ny, nz)  rough normal vector of the palm plane

# Helpers
hand.wrist             # NormalizedLandmark for landmark 0
hand.tip("INDEX")      # NormalizedLandmark for the named fingertip
hand.landmark(8)       # NormalizedLandmark by index number
```

Landmark indices follow the standard MediaPipe layout:

```
0   WRIST
1-4   THUMB  (CMC, MCP, IP, TIP)
5-8   INDEX  (MCP, PIP, DIP, TIP)
9-12  MIDDLE
13-16 RING
17-20 PINKY
```

### MotionEvent

```python
ev.name        # str  e.g. "CLAP", "WAVE", "GRAB"
ev.confidence  # float 0-1
ev.hand        # "Left", "Right", or None for two-hand events
```

---

## Library — common patterns

**Get the right hand's index fingertip position in pixels:**
```python
with HandTracker(write_md=False) as t:
    for frame in t.frames():
        right = next((h for h in frame.hands if h.label == "Right"), None)
        if right:
            tip = right.tip("INDEX")
            x_px = int(tip.x * frame.width)
            y_px = int(tip.y * frame.height)
            print(x_px, y_px)
```

**React to motion events:**
```python
with HandTracker(write_md=False) as t:
    for frame in t.frames():
        for ev in frame.events:
            if ev.name == "CLAP":
                print("clapped!")
            elif ev.name == "WAVE" and ev.hand == "Right":
                print("right hand wave")
```

**Stop after 10 seconds:**
```python
import time
start = time.time()
with HandTracker(write_md=False) as t:
    for frame in t.frames():
        if time.time() - start > 10:
            break
        # do stuff
```

**Show the annotated frame yourself:**
```python
import cv2
with HandTracker(write_md=False) as t:
    for frame in t.frames():
        cv2.imshow("my window", frame.image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cv2.destroyAllWindows()
```

**Using the callback style instead of a generator:**
```python
def on_frame(frame):
    if frame.hands:
        print(frame.hands[0].gesture)
    # return False to stop the loop

HandTracker(write_md=False).run(on_frame=on_frame)
```

**Write your own md file alongside the tracker's:**
```python
with HandTracker() as t:       # still writes info.md
    for frame in t.frames():
        # your own output
        with open("my_log.txt", "a") as f:
            for hand in frame.hands:
                f.write(f"{frame.elapsed:.2f}  {hand.label}  {hand.gesture}\n")
```

---

## The info.md file

While INDEX is running, `info.md` is rewritten every frame. Open it in VS Code
(or any markdown preview) and it updates live.

It contains:

- Frame number, elapsed time, FPS, resolution
- Motion events fired this frame
- A rolling log of the last 10 motion events with timestamps
- For each hand:
  - Detected gesture + confidence
  - Palm centre and bounding box (normalised and pixel)
  - Palm normal vector
  - Finger extension table
  - All 21 landmark positions (normalised x/y/z and pixel x/y)
  - Pinch distances from thumb to each fingertip
  - All 10 fingertip-to-fingertip distances with touch detection
  - Per-finger curl measurement (MCP-to-tip distance)
- Inter-hand distances when both hands are visible

---

## Volume control

Separate script (`volume_control.py`), launched via `run.bat`.

1. Show both hands to the camera
2. Pinch both hands (thumb tip touches index tip on each hand) — the line between them turns green
3. While pinched, **spread hands apart** to raise volume, **bring them together** to lower it
4. Release either pinch to lock the volume where it is
5. Re-pinching picks up from the current volume — it won't snap back

**Keys while it's running:**

| Key | Action |
|-----|--------|
| T | Toggle real volume control on/off (the display still works) |
| ESC | Quit |

The green bar on the right shows the current system volume. If you adjust the volume manually (keyboard, taskbar, etc.) while the app is open, the bar updates automatically.

---

## Troubleshooting

**Camera doesn't open**
Try `--camera 1` or `--camera 2`. Some setups have virtual cameras at index 0 that block the real one.

**Hands detected but info.md says "no hands"**
This was a known bug caused by the MediaPipe C++ binding being consumed twice. Fixed — update INDEX.py if you have an old version.

**Gestures feel slow to trigger**
Motion events need ~0.5–1.5 seconds of history to fire. Static gestures are instant.

**CLENCHED FIST never triggers**
The threshold requires tips to be clearly below MCPs (at least 4% of frame height). Make sure your fist is tight and facing the camera.

**Wave not detecting**
It needs 3+ direction reversals in ~25 frames. Slow, small movements won't trigger it — make the wave deliberate.
