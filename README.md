# INDEX
### Immersive Navigation Dynamic Embedding X-Ray

Webcam hand tracker that reads your gestures and turns them into data.
Built on MediaPipe. Runs local, no cloud, no API key.

```
  WRIST ──► THUMB_CMC ──► THUMB_MCP ──► THUMB_TIP
    │
    ├──► INDEX_MCP ──► INDEX_PIP ──► INDEX_DIP ──► INDEX_TIP
    │
    ├──► MIDDLE_MCP ──► ...
    │
    ├──► RING_MCP ──► ...
    │
    └──► PINKY_MCP ──► ...
```

---

## What it does

- Tracks both hands at once, 21 joints per hand
- Detects static gestures (fist, pinch, peace, ok, pointing, etc.)
- Detects motion events across frames — clap, fist bump, wave, grab, throw, push
- Writes everything to `info.md` live, every single frame
- Works as a standalone script or an importable Python library

---

## Requirements

```
pip install opencv-python mediapipe pycaw comtypes numpy
```

Python 3.9+. Windows only for the volume controller (`pycaw`).
INDEX.py itself runs on any OS.

The hand landmark model (~27 MB) downloads automatically on first run.

---

## Files

| File | What it is |
|------|------------|
| `INDEX.py` | Hand tracker — library + CLI |
| `volume_control.py` | Pinch-gesture volume control |
| `run_debug_tracker.bat` | Launch INDEX |
| `run.bat` | Launch volume control |
| `install_deps.bat` | Install everything |
| `info.md` | Live output — updated every frame while INDEX runs |

---

## Gestures

**Static** — detected from a single frame:

| Name | Description |
|------|-------------|
| OPEN HAND | All fingers extended |
| FIST | All fingers closed |
| CLENCHED FIST | Tight fist, tips well below MCPs |
| PINCH | Thumb + index touching |
| POINTING | Index only |
| PEACE | Index + middle |
| THUMBS UP | Thumb only, rest closed |
| ROCK ON | Index + pinky |
| GUN | Thumb + index |
| OK | Thumb + index circle, others extended |
| PINKY OUT | Pinky only |
| CUSTOM | Anything else |

**Motion** — detected across ~1 second of frames:

| Name | Trigger |
|------|---------|
| CLAP | Both open hands come together fast |
| FIST BUMP | Both fists come together fast |
| WAVE | Hand sweeps left-right 3+ times |
| GRAB | Open hand closes into a fist |
| THROW | Fist opens suddenly |
| PUSH | Wrist moves toward the camera |

---

## Volume control

Separate script. Pinch both hands to activate, then spread or close them to adjust volume. Re-pinching always picks up from wherever the volume currently is — it doesn't snap back.

Press **T** to toggle whether it actually controls your system volume (useful if you want to test the gesture without touching your audio).

---

See [USAGE.md](USAGE.md) for the full library API and CLI reference.
