"""
INDEX.py  --  Hand tracking library + CLI

  CLI:
    python INDEX.py                        window + info.md (default)
    python INDEX.py --headless             no window, just info.md
    python INDEX.py --no-md               window only, no file
    python INDEX.py --output myfile.md    custom md path
    python INDEX.py --camera 1            different camera
    python INDEX.py --max-hands 1         track one hand only

  Library:
    from INDEX import HandTracker

    with HandTracker(write_md=False) as t:
        for frame in t.frames():
            for hand in frame.hands:
                print(hand.label, hand.gesture, hand.is_pinching)
            for ev in frame.events:           # motion events
                print(ev.name, ev.hand)

    # Callback style:
    HandTracker().run(on_frame=lambda f: print(f.hands[0].gesture) if f.hands else None)
"""

from __future__ import annotations
import argparse, dataclasses, math, os, time, urllib.request
from collections import deque
from typing import Callable, Deque, Dict, Iterator, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Model ─────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "hand_landmarker.task")
DEFAULT_MD = os.path.join(_DIR, "info.md")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~27 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

# ── Landmark index tables ─────────────────────────────────────────────────────
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",
]
# (name, mcp, pip, dip, tip)
FINGERS = [
    ("INDEX",  5,  6,  7,  8),
    ("MIDDLE", 9,  10, 11, 12),
    ("RING",   13, 14, 15, 16),
    ("PINKY",  17, 18, 19, 20),
]
TIPS       = {4:"THUMB", 8:"INDEX", 12:"MIDDLE", 16:"RING", 20:"PINKY"}
_TIP_IDX   = {v:k for k,v in TIPS.items()}
TOUCH_NORM = 0.05
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),(0,17),
]
_COLORS = {"Left":(255,120,0), "Right":(0,200,255)}

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class MotionEvent:
    """A gesture event that spans multiple frames."""
    name:       str           # e.g. "CLAP", "WAVE", "GRAB"
    confidence: float         # 0.0 – 1.0
    hand:       Optional[str] # "Left", "Right", or None (two-hand event)

@dataclasses.dataclass
class Hand:
    """All data for one detected hand in a single frame."""
    label:             str
    confidence:        float
    gesture:           str            # static gesture name
    landmarks:         list           # 21 NormalizedLandmark (x,y,z in 0-1)
    finger_extension:  Dict[str,bool]
    is_pinching:       bool
    pinch_distances:   Dict[str,float]# finger→3d dist to thumb tip
    palm_centre_norm:  Tuple[float,float]
    bounding_box_norm: Tuple[float,float,float,float]
    palm_normal:       Tuple[float,float,float]

    def tip(self, finger:str):
        """NormalizedLandmark for the named tip: THUMB|INDEX|MIDDLE|RING|PINKY"""
        return self.landmarks[_TIP_IDX[finger.upper()]]

    @property
    def wrist(self): return self.landmarks[0]
    def landmark(self, i:int): return self.landmarks[i]

@dataclasses.dataclass
class FrameData:
    """Everything produced for one webcam frame."""
    hands:   List[Hand]
    events:  List[MotionEvent]   # motion events fired this frame
    frame_n: int
    elapsed: float
    fps:     float
    width:   int
    height:  int
    image:   object              # np.ndarray annotated BGR frame

# ── Geometry helpers ──────────────────────────────────────────────────────────

def _d3(a,b)->float:
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)
def _d2px(a,b,w,h)->float:
    return math.hypot((a.x-b.x)*w,(a.y-b.y)*h)
def _touching(a,b)->bool:
    return _d3(a,b)<TOUCH_NORM
def _mean(lst): return sum(lst)/len(lst) if lst else 0.0

# ── Static gesture detection ──────────────────────────────────────────────────

def _finger_ext(lms,mcp,tip)->bool:
    return lms[tip].y < lms[mcp].y

def _thumb_ext(lms,label)->bool:
    return lms[4].x < lms[3].x if label=="Right" else lms[4].x > lms[3].x

def _ext_map(lms,label)->Dict[str,bool]:
    return {
        "THUMB":  _thumb_ext(lms,label),
        "INDEX":  _finger_ext(lms,5,8),
        "MIDDLE": _finger_ext(lms,9,12),
        "RING":   _finger_ext(lms,13,16),
        "PINKY":  _finger_ext(lms,17,20),
    }

def _is_clenched(lms)->bool:
    """Tighter than FIST: all tips are well below their MCPs."""
    return all(lms[tip].y > lms[mcp].y + 0.04
               for _,mcp,_,_,tip in FINGERS)

def _detect_static_gesture(lms, label)->Tuple[str,Dict[str,bool]]:
    ext = _ext_map(lms, label)
    i,m,r,p = ext["INDEX"],ext["MIDDLE"],ext["RING"],ext["PINKY"]
    t = ext["THUMB"]
    pinching = _touching(lms[4],lms[8])

    if not any(ext.values()):
        return ("CLENCHED FIST" if _is_clenched(lms) else "FIST"), ext
    if all(ext.values()):              return "OPEN HAND", ext
    if pinching:                       return "PINCH", ext
    if i and not m and not r and not p: return "POINTING", ext
    if i and m and not r and not p:    return "PEACE", ext
    if t and not i and not m and not r and not p: return "THUMBS UP", ext
    if i and p and not m and not r:    return "ROCK ON", ext
    if t and i and not m and not r and not p:     return "GUN", ext
    if not i and not m and not r and p:           return "PINKY OUT", ext
    # OK sign: index tip close to thumb, others extended
    if _d3(lms[4],lms[8])<TOUCH_NORM and m and r and p: return "OK", ext
    return "CUSTOM", ext

# ── Motion tracking ───────────────────────────────────────────────────────────

@dataclasses.dataclass
class _HSnap:
    """Lightweight per-hand snapshot for the motion buffer."""
    label:    str
    gesture:  str
    wx: float; wy: float; wz: float   # wrist position
    is_fist:  bool
    is_open:  bool

@dataclasses.dataclass
class _FSnap:
    """Lightweight per-frame snapshot."""
    elapsed: float
    hands:   Dict[str,"_HSnap"]  # label -> snap
    iwd:     Optional[float]     # inter-wrist distance (3D norm)

def _make_fsnap(fd:FrameData) -> _FSnap:
    hs = {}
    for h in fd.hands:
        hs[h.label] = _HSnap(
            label=h.label, gesture=h.gesture,
            wx=h.wrist.x, wy=h.wrist.y, wz=h.wrist.z,
            is_fist="FIST" in h.gesture,
            is_open=h.gesture=="OPEN HAND",
        )
    iwd = None
    if "Left" in hs and "Right" in hs:
        l,r = hs["Left"],hs["Right"]
        iwd = math.sqrt((l.wx-r.wx)**2+(l.wy-r.wy)**2+(l.wz-r.wz)**2)
    return _FSnap(elapsed=fd.elapsed, hands=hs, iwd=iwd)

# ── Individual motion detectors ───────────────────────────────────────────────

def _two_hand_iwd(buf, n_recent, n_older):
    """Returns (recent_avg_iwd, older_avg_iwd) or (None,None)."""
    recent = list(buf)[-n_recent:]
    older  = list(buf)[-(n_recent+n_older):-n_recent]
    r = [s.iwd for s in recent if s.iwd is not None
         and "Left" in s.hands and "Right" in s.hands]
    o = [s.iwd for s in older  if s.iwd is not None
         and "Left" in s.hands and "Right" in s.hands]
    if not r or not o: return None, None
    return _mean(r), _mean(o)

def _det_clap(buf)->Optional[MotionEvent]:
    if len(buf)<14: return None
    ri, oi = _two_hand_iwd(buf, 5, 9)
    if ri is None: return None
    recent = list(buf)[-5:]
    # hands must NOT be fists (clap = open/semi-open hands coming together)
    if any(s.hands.get("Left",_HSnap("","",0,0,0,False,False)).is_fist or
           s.hands.get("Right",_HSnap("","",0,0,0,False,False)).is_fist
           for s in recent if "Left" in s.hands and "Right" in s.hands):
        return None
    drop = oi - ri
    if ri < 0.18 and drop > 0.10:
        return MotionEvent("CLAP", min(1.0, drop/0.20), None)
    return None

def _det_fist_bump(buf)->Optional[MotionEvent]:
    if len(buf)<14: return None
    ri, oi = _two_hand_iwd(buf, 5, 9)
    if ri is None: return None
    recent = list(buf)[-5:]
    both_fists = all(
        s.hands.get("Left",_HSnap("","",0,0,0,False,False)).is_fist and
        s.hands.get("Right",_HSnap("","",0,0,0,False,False)).is_fist
        for s in recent if "Left" in s.hands and "Right" in s.hands
    )
    if not both_fists: return None
    drop = oi - ri
    if ri < 0.20 and drop > 0.09:
        return MotionEvent("FIST BUMP", min(1.0, drop/0.18), None)
    return None

def _det_wave(buf, label)->Optional[MotionEvent]:
    if len(buf)<20: return None
    snaps = [s for s in list(buf)[-28:] if label in s.hands]
    if len(snaps)<14: return None
    xs = [s.hands[label].wx for s in snaps]
    # count direction reversals in smoothed signal
    dirs = []
    for i in range(1,len(xs)):
        d = xs[i]-xs[i-1]
        if abs(d)>0.012: dirs.append(1 if d>0 else -1)
    revs = sum(1 for i in range(1,len(dirs)) if dirs[i]!=dirs[i-1])
    if revs>=3:
        return MotionEvent("WAVE", min(1.0,revs/5), label)
    return None

def _det_grab(buf, label)->Optional[MotionEvent]:
    if len(buf)<10: return None
    recent = [s for s in list(buf)[-5:]  if label in s.hands]
    older  = [s for s in list(buf)[-10:-5] if label in s.hands]
    if not recent or not older: return None
    was_open = any(s.hands[label].is_open for s in older)
    now_fist = all(s.hands[label].is_fist for s in recent)
    if was_open and now_fist:
        return MotionEvent("GRAB", 0.85, label)
    return None

def _det_throw(buf, label)->Optional[MotionEvent]:
    if len(buf)<10: return None
    recent = [s for s in list(buf)[-5:]  if label in s.hands]
    older  = [s for s in list(buf)[-10:-5] if label in s.hands]
    if not recent or not older: return None
    was_fist = any(s.hands[label].is_fist for s in older)
    now_open = any(s.hands[label].is_open for s in recent)
    if was_fist and now_open:
        return MotionEvent("THROW", 0.85, label)
    return None

def _det_push(buf, label)->Optional[MotionEvent]:
    """Hand moves rapidly toward the camera (z drops)."""
    if len(buf)<12: return None
    recent = [s for s in list(buf)[-5:]  if label in s.hands]
    older  = [s for s in list(buf)[-12:-5] if label in s.hands]
    if not recent or not older: return None
    rz = _mean([s.hands[label].wz for s in recent])
    oz = _mean([s.hands[label].wz for s in older])
    if oz - rz > 0.06:  # z is negative closer → drops as hand approaches
        return MotionEvent("PUSH", min(1.0,(oz-rz)/0.10), label)
    return None

_TWO_HAND_DETECTORS = [_det_clap, _det_fist_bump]
_ONE_HAND_DETECTORS = [_det_wave, _det_grab, _det_throw, _det_push]

class MotionTracker:
    """
    Keeps a rolling buffer of frame snapshots and fires MotionEvents
    when motion patterns are detected.

    Usage:
        mt = MotionTracker()
        for frame in tracker.frames():
            events = mt.update(frame)
    """
    HISTORY  = 50    # frames (~1.5 s at 30 fps)
    COOLDOWN = 0.9   # seconds before the same event can fire again

    def __init__(self):
        self._buf: Deque[_FSnap] = deque(maxlen=self.HISTORY)
        self._last: Dict[str,float] = {}

    def update(self, fd:FrameData) -> List[MotionEvent]:
        self._buf.append(_make_fsnap(fd))
        now = fd.elapsed
        out = []
        # Two-hand events
        for det in _TWO_HAND_DETECTORS:
            ev = det(self._buf)
            if ev and now - self._last.get(ev.name, -999) > self.COOLDOWN:
                self._last[ev.name] = now
                out.append(ev)
        # Per-hand events
        for label in ("Left","Right"):
            for det in _ONE_HAND_DETECTORS:
                ev = det(self._buf, label)
                key = f"{ev.name}_{label}" if ev else ""
                if ev and now - self._last.get(key,-999) > self.COOLDOWN:
                    self._last[key] = now
                    out.append(ev)
        return out

# ── Build Hand object ─────────────────────────────────────────────────────────

def _build_hand(lms:list, hedness:list) -> Hand:
    label   = hedness[0].category_name
    conf    = hedness[0].score
    gesture, ext = _detect_static_gesture(lms, label)
    thumb   = lms[4]
    pdists  = {fname: _d3(thumb, lms[tip]) for fname,_,_,_,tip in FINGERS}
    cx      = sum(l.x for l in lms)/21
    cy      = sum(l.y for l in lms)/21
    xs      = [l.x for l in lms]; ys = [l.y for l in lms]
    wi  = (lms[5].x-lms[0].x, lms[5].y-lms[0].y, lms[5].z-lms[0].z)
    wp  = (lms[17].x-lms[0].x,lms[17].y-lms[0].y,lms[17].z-lms[0].z)
    nx  = wi[1]*wp[2]-wi[2]*wp[1]
    ny  = wi[2]*wp[0]-wi[0]*wp[2]
    nz  = wi[0]*wp[1]-wi[1]*wp[0]
    mag = math.sqrt(nx**2+ny**2+nz**2) or 1e-9
    return Hand(
        label=label, confidence=conf, gesture=gesture, landmarks=lms,
        finger_extension=ext,
        is_pinching=_touching(thumb,lms[8]),
        pinch_distances=pdists,
        palm_centre_norm=(cx,cy),
        bounding_box_norm=(min(xs),min(ys),max(xs),max(ys)),
        palm_normal=(nx/mag,ny/mag,nz/mag),
    )

# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_hand(img, lms, w, h, color):
    pts = [(int(l.x*w),int(l.y*h)) for l in lms]
    for a,b in HAND_CONNECTIONS:
        cv2.line(img,pts[a],pts[b],color,2)
    for i,pt in enumerate(pts):
        cv2.circle(img,pt,4,color,-1)
        if i in TIPS or i==0:
            cv2.putText(img,TIPS.get(i,"WRIST"),(pt[0]+5,pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.36,color,1)
        if i in TIPS and i!=4 and _touching(lms[4],lms[i]):
            cv2.circle(img,pt,13,(0,255,120),2)

def _draw_events(img, recent_events):
    """Show last N motion events as fading labels."""
    now = time.time()
    y   = 110
    for ev, t in recent_events:
        age   = now - t
        if age > 2.0: continue
        alpha = max(0, 1.0 - age/2.0)
        bright = int(255*alpha)
        label = f"[{ev.hand or 'BOTH'}]  {ev.name}  {ev.confidence:.0%}"
        cv2.putText(img, label, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, bright, bright//2), 2)
        y += 32

# ── Markdown builder ──────────────────────────────────────────────────────────

def build_md(fd:FrameData, session_start:float, recent_events=None)->str:
    L = []
    L.append("# Hand Tracking - Live Info\n")
    L.append("| Field | Value |")
    L.append("|-------|-------|")
    L.append(f"| Frame | {fd.frame_n} |")
    L.append(f"| Time  | {fd.elapsed:.3f} s |")
    L.append(f"| FPS   | {fd.fps:.1f} |")
    L.append(f"| Resolution | {fd.width} x {fd.height} px |")
    L.append(f"| Session started | {time.strftime('%H:%M:%S',time.localtime(session_start))} |")
    L.append(f"| Hands detected | {len(fd.hands)} |")
    L.append("")

    # Motion events this frame
    if fd.events:
        L.append("### Motion Events (this frame)\n")
        L.append("| Event | Hand | Confidence |")
        L.append("|-------|------|------------|")
        for ev in fd.events:
            L.append(f"| **{ev.name}** | {ev.hand or 'BOTH'} | {ev.confidence:.2f} |")
        L.append("")

    # Recent event log
    if recent_events:
        L.append("### Recent Events Log\n")
        L.append("| Event | Hand | Confidence | Age |")
        L.append("|-------|------|------------|-----|")
        now = time.time()
        for ev, t in list(recent_events)[-10:]:
            age = now - t
            L.append(f"| {ev.name} | {ev.hand or 'BOTH'} | {ev.confidence:.2f} | {age:.1f}s ago |")
        L.append("")

    if not fd.hands:
        L.append("---\n\n_No hands in frame._\n")
        L.append(f"\n---\n_Last updated: {time.strftime('%H:%M:%S')}_")
        return "\n".join(L)

    w,h = fd.width,fd.height

    for hand in fd.hands:
        lms = hand.landmarks
        cx,cy = hand.palm_centre_norm
        x0,y0,x1,y1 = hand.bounding_box_norm
        nx,ny,nz = hand.palm_normal

        L.append("---\n")
        L.append(f"## {hand.label} Hand\n")

        L.append("### Overview\n")
        L.append("| | |")
        L.append("|---|---|")
        L.append(f"| Confidence | {hand.confidence:.3f} |")
        L.append(f"| Gesture | **{hand.gesture}** |")
        L.append(f"| Pinching | {'YES' if hand.is_pinching else 'no'} |")
        L.append(f"| Palm centre (norm) | x={cx:.4f}  y={cy:.4f} |")
        L.append(f"| Palm centre (px) | x={int(cx*w)}  y={int(cy*h)} |")
        L.append(f"| Bounding box (norm) | ({x0:.3f},{y0:.3f}) -> ({x1:.3f},{y1:.3f}) |")
        L.append(f"| Bounding box (px) | ({int(x0*w)},{int(y0*h)}) -> ({int(x1*w)},{int(y1*h)}) |")
        L.append(f"| Palm normal | nx={nx:.3f}  ny={ny:.3f}  nz={nz:.3f} |")
        L.append("")

        L.append("### Finger Extension\n")
        L.append("| Finger | Extended |")
        L.append("|--------|----------|")
        for fn,st in hand.finger_extension.items():
            L.append(f"| {fn} | {'YES' if st else 'no'} |")
        L.append("")

        L.append("### All 21 Landmarks\n")
        L.append("| # | Name | x (norm) | y (norm) | z (norm) | px x | px y |")
        L.append("|---|------|----------|----------|----------|------|------|")
        for i,lm in enumerate(lms):
            L.append(f"| {i:2d} | `{LANDMARK_NAMES[i]:<14}` "
                     f"| {lm.x:.5f} | {lm.y:.5f} | {lm.z:.5f} "
                     f"| {int(lm.x*w):4d} | {int(lm.y*h):4d} |")
        L.append("")

        L.append("### Pinch Distances (tip -> thumb)\n")
        L.append("| Finger | 3D norm | px | Touching |")
        L.append("|--------|---------|----|----------|")
        for fname,_,_,_,tip in FINGERS:
            t2 = lms[tip]
            mark = "**TOUCHING**" if _touching(lms[4],t2) else "no"
            L.append(f"| {fname} | {_d3(lms[4],t2):.5f} | {_d2px(lms[4],t2,w,h):.1f}px | {mark} |")
        L.append("")

        tip_list = list(TIPS.items())
        L.append("### Fingertip-to-Fingertip\n")
        L.append("| Pair | 3D norm | px | Touching |")
        L.append("|------|---------|----|----------|")
        for i in range(len(tip_list)):
            for j in range(i+1,len(tip_list)):
                ai,an=tip_list[i]; bi,bn=tip_list[j]
                mark = "**TOUCHING**" if _touching(lms[ai],lms[bi]) else "no"
                L.append(f"| {an}-{bn} | {_d3(lms[ai],lms[bi]):.5f} | {_d2px(lms[ai],lms[bi],w,h):.1f}px | {mark} |")
        L.append("")

        L.append("### Per-finger Curl\n")
        L.append("| Finger | MCP->TIP norm | MCP->TIP px | Curled? |")
        L.append("|--------|---------------|-------------|---------|")
        for fn,mcp,_,_,tip in FINGERS:
            dn=_d3(lms[mcp],lms[tip]); dp=_d2px(lms[mcp],lms[tip],w,h)
            L.append(f"| {fn} | {dn:.5f} | {dp:.1f}px | {'**yes**' if dn<0.12 else 'no'} |")
        L.append("")

    if len(fd.hands)==2:
        la,lb = fd.hands[0],fd.hands[1]
        L.append("---\n")
        L.append(f"## Inter-hand ({la.label} <-> {lb.label})\n")
        L.append("| Landmark | 3D norm | px | Touching |")
        L.append("|----------|---------|----|----------|")
        for ai,bi,pl in [(0,0,"Wrist"),(4,4,"Thumb tip"),(8,8,"Index tip"),
                         (12,12,"Middle tip"),(16,16,"Ring tip"),(20,20,"Pinky tip")]:
            a,b=la.landmarks[ai],lb.landmarks[bi]
            mark="**TOUCHING**" if _touching(a,b) else "no"
            L.append(f"| {pl} | {_d3(a,b):.5f} | {_d2px(a,b,w,h):.1f}px | {mark} |")
        wp=_d2px(la.landmarks[0],lb.landmarks[0],w,h)
        ip=_d2px(la.landmarks[8],lb.landmarks[8],w,h)
        L.append(f"\n**Wrist separation:** {wp:.0f} px  \n**Index separation:** {ip:.0f} px\n")

    L.append("\n---")
    L.append(f"_Last updated: {time.strftime('%H:%M:%S')}_")
    return "\n".join(L)

# ── HandTracker ───────────────────────────────────────────────────────────────

class HandTracker:
    """
    Webcam-based hand tracker with static + motion gesture detection.

    Quick start (library use):
        with HandTracker(write_md=False) as t:
            for frame in t.frames():
                for hand in frame.hands:
                    print(hand.label, hand.gesture)
                for ev in frame.events:
                    print(ev.name, ev.hand, ev.confidence)
    """

    def __init__(
        self,
        camera:               int   = 0,
        headless:             bool  = False,
        write_md:             bool  = True,
        md_path:              Optional[str] = None,
        max_hands:            int   = 2,
        detection_confidence: float = 0.75,
        tracking_confidence:  float = 0.75,
    ):
        self.camera               = camera
        self.headless             = headless
        self.write_md             = write_md
        self.md_path              = md_path or DEFAULT_MD
        self.max_hands            = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence  = tracking_confidence
        self._cap = self._detector = None
        self._start = self._session_start = None
        self._frame_n = 0; self._fps = 0.0; self._fps_t = None

    def open(self)->"HandTracker":
        _ensure_model()
        self._cap = cv2.VideoCapture(self.camera)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera}")
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_confidence,
            min_hand_presence_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(opts)
        self._start = self._session_start = time.time()
        self._fps_t = self._start; self._frame_n = 0
        return self

    def close(self):
        if self._cap:     self._cap.release();     self._cap=None
        if self._detector: self._detector.close(); self._detector=None

    def __enter__(self): return self.open()
    def __exit__(self,*_): self.close(); cv2.destroyAllWindows()

    def frames(self)->Iterator[FrameData]:
        """Generator — yields FrameData for every webcam frame."""
        if self._cap is None: self.open()
        mt = MotionTracker()
        while True:
            ok,raw = self._cap.read()
            if not ok: continue
            frame = cv2.flip(raw,1)
            h,w   = frame.shape[:2]
            now   = time.time(); elapsed = now-self._start
            ts_ms = int(elapsed*1000)
            if self._frame_n%30==0 and self._frame_n>0:
                self._fps = 30/max(now-self._fps_t,1e-9); self._fps_t=now
            rgb    = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result = self._detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb), ts_ms)

            hands:List[Hand] = []
            if result.hand_landmarks:
                for lms,hed in zip(result.hand_landmarks,result.handedness):
                    hands.append(_build_hand(list(lms),list(hed)))

            annotated = frame.copy()
            for hand in hands:
                draw_hand(annotated,hand.landmarks,w,h,_COLORS.get(hand.label,(200,200,200)))

            fd = FrameData(hands=hands,events=[],frame_n=self._frame_n,
                           elapsed=elapsed,fps=self._fps,width=w,height=h,image=annotated)
            fd.events = mt.update(fd)
            self._frame_n += 1
            yield fd

    def run(self, on_frame:Optional[Callable[[FrameData],None]]=None):
        """
        Blocking run loop.
        on_frame(fd) is called every frame; return False from it to stop.
        """
        session_start  = self._session_start or time.time()
        recent_events: Deque[Tuple[MotionEvent,float]] = deque(maxlen=30)

        for fd in self.frames():
            for ev in fd.events:
                recent_events.append((ev, time.time()))

            if self.write_md:
                with open(self.md_path,"w",encoding="utf-8") as f:
                    f.write(build_md(fd, session_start, recent_events))

            if on_frame is not None:
                if on_frame(fd) is False: break

            if not self.headless:
                img = fd.image.copy()
                cv2.putText(img,
                    f"FPS {fd.fps:.0f}  Frame {fd.frame_n}  Hands {len(fd.hands)}",
                    (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
                # static gesture labels per hand
                for hand in fd.hands:
                    cx,cy = hand.palm_centre_norm
                    px,py = int(cx*fd.width),int(cy*fd.height)-20
                    cv2.putText(img,hand.gesture,(px-30,py),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,_COLORS.get(hand.label,(200,200,200)),2)
                _draw_events(img, recent_events)
                lbl = f"Writing: {os.path.basename(self.md_path)}  |  ESC=quit" if self.write_md else "ESC=quit"
                cv2.putText(img,lbl,(10,fd.height-12),cv2.FONT_HERSHEY_SIMPLEX,0.45,(100,100,100),1)
                cv2.imshow("INDEX - Hand Tracker",img)
                if cv2.waitKey(1)&0xFF==27: break
            else:
                if fd.frame_n%30==0:
                    evs = ",".join(e.name for e,_ in list(recent_events)[-3:]) or "-"
                    print(f"\rFrame {fd.frame_n:6d}  FPS {fd.fps:5.1f}  "
                          f"Hands {len(fd.hands)}  Events: {evs}     ",end="",flush=True)

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="INDEX - live hand tracker with motion gesture detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Static gestures:  FIST, CLENCHED FIST, OPEN HAND, PINCH, POINTING,
                  PEACE, THUMBS UP, ROCK ON, GUN, PINKY OUT, OK, CUSTOM

Motion events:    CLAP, FIST BUMP, WAVE, GRAB, THROW, PUSH
        """,
    )
    p.add_argument("--headless",   action="store_true", help="no camera window")
    p.add_argument("--no-md",      action="store_true", help="do not write info.md")
    p.add_argument("--camera",     type=int,   default=0,    metavar="N")
    p.add_argument("--output",     default=None,             metavar="PATH")
    p.add_argument("--max-hands",  type=int,   default=2,    metavar="N")
    args = p.parse_args()

    print("Headless:" if args.headless else "Press ESC in the camera window to quit.")
    tracker = HandTracker(
        camera=args.camera, headless=args.headless,
        write_md=not args.no_md, md_path=args.output,
        max_hands=args.max_hands,
    )
    with tracker:
        try:
            tracker.run()
        except KeyboardInterrupt:
            pass
    print("\nStopped.")

if __name__=="__main__":
    main()
