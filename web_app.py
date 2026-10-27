"""
Deep-Live-Cam × Lynkio 1.3.1 — WebSocket binary streaming server.
Memory-Optimized for 2GB RAM (Railway/Render)
"""
import os, sys, types, base64, re, json, asyncio, time
from pathlib import Path

# ===========================================================================
# Model persistence — before any insightface import
# ===========================================================================
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("INSIGHTFACE_HOME", str(ROOT / "models" / "insightface"))
os.environ.setdefault("HOME", str(ROOT / "models"))
os.environ.setdefault("OMP_NUM_THREADS", "1") # Prevent ONNX from spawning too many threads
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "faces").mkdir(exist_ok=True)

# ===========================================================================
# Headless stub for modules.ui
# ===========================================================================
def _install_ui_stub():
    stub = types.ModuleType("modules.ui")
    def _noop(*_a, **_k): return None
    for n in ("update_status","init","destroy","close","create_analyser",
              "register_analyser","register_frame_processor","set_frame_processor",
              "start","start_preview","stop_preview","init_preview",
              "update_preview","update_tk_image","update_frame"):
        setattr(stub, n, _noop)
    stub.__dict__["__getattr__"] = lambda _n: _noop
    stub.UI_AVAILABLE = False
    sys.modules["modules.ui"] = stub
_install_ui_stub()

sys.path.insert(0, str(ROOT))
os.environ["PATH"] = str(ROOT) + os.pathsep + os.environ.get("PATH", "")

import cv2, numpy as np

# Monkeypatch InsightFace to use a smaller detection size (Saves ~300MB RAM)
import insightface
_orig_init = insightface.app.FaceAnalysis.__init__
def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.prepare(ctx_id=-1, det_size=(320, 320))
insightface.app.FaceAnalysis.__init__ = _patched_init

from download import download_models
download_models()

import modules.globals
_p = os.environ.get("EXECUTION_PROVIDER", "cpu").lower()
modules.globals.execution_providers = {
    "cuda":   ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}.get(_p, ["CPUExecutionProvider"])

from modules.face_analyser import get_one_face
from modules.processors.frame.core import get_frame_processors_modules

# ===========================================================================
# Preflight
# ===========================================================================
try:
    # ONLY load face_swapper to save RAM
    _processors = get_frame_processors_modules(["face_swapper"])
    print(f"[DLC] preflight OK — {len(_processors)} processor(s)", flush=True)
except BaseException as e:
    import traceback; traceback.print_exc()
    print(f"[DLC] FATAL: {e!r}", flush=True)
    sys.exit(1)

# ===========================================================================
# Tunables
# ===========================================================================
TARGET_FPS   = float(os.environ.get("TARGET_FPS", "5"))
MAX_W        = int(os.environ.get("MAX_WIDTH",  "384"))
MAX_H        = int(os.environ.get("MAX_HEIGHT", "288"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "70")) # Lower quality = less RAM

# ===========================================================================
# State
# ===========================================================================
_active = {"face": None, "name": None}      
_lock   = asyncio.Lock()                    
_last   = {"jpeg": None}                    
_last_warn = 0

def _resize(frame):
    h, w = frame.shape[:2]
    if w <= MAX_W and h <= MAX_H: return frame
    s = min(MAX_W / w, MAX_H / h)
    return cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def _encode_jpeg(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None

def _decode_jpeg(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 _-]", "", name or "").strip()[:32] or "unnamed"

def _list_faces():
    return sorted([p.stem for p in (ROOT / "faces").glob("*.jpg")])

def _load_face(name: str):
    path = ROOT / "faces" / f"{name}.jpg"
    if not path.exists(): return False
    bgr = cv2.imread(str(path))
    if bgr is None: return False
    
    h, w = bgr.shape[:2]
    if w > 720:
        s = 720 / w
        bgr = cv2.resize(bgr, (720, int(h * s)), interpolation=cv2.INTER_AREA)
        
    face = get_one_face(bgr)
    if face is None: 
        print(f"[DLC] WARN: No face detected in {name}.jpg", flush=True)
        return False
    
    _active["face"] = face
    _active["name"] = name
    modules.globals.source_path = str(path)
    modules.globals.frame_processors = ["face_swapper"] # Override config to force only swapper
    _last["jpeg"] = None
    print(f"[DLC] SUCCESS: Loaded source face '{name}'", flush=True)
    return True

def _process_frame_sync(bgr, face):
    for p in _processors:
        bgr = p.process_frame(face, bgr)
    return bgr

# ===========================================================================
# Lynkio app
# ===========================================================================
from lynkio import Lynk, render_template

app = Lynk(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", "7860")),
    protocol="AUTO",
    serve_client=True,
    rate_limit=120,
    max_payload_size=20 * 1024 * 1024,
    debug=False,
)

app.static("/faces", str(ROOT / "faces"))

@app.get("/")
async def index(req):
    return render_template("index.html", {"title": "Deep-Live-Cam Studio"})

# ===========================================================================
# WebSocket Events
# ===========================================================================
@app.on_internal("connect")
async def on_connect(client, _):
    print(f"[DLC] client connected: {client.id}", flush=True)
    payload = json.dumps({
        "event": "ready",
        "data": {"faces": _list_faces(), "active": _active["name"]},
    })
    await client.send(payload)

@app.on("ping")
async def ws_ping(client, data):
    await client.send(json.dumps({"event": "pong", "data": time.time()}))

@app.on("save_face")
async def ws_save_face(client, data):
    name = _safe_name(data.get("name", ""))
    b64  = data.get("image", "")
    if not name or not b64:
        return await client.send(json.dumps({"event": "error", "data": "Name and image required"}))
    
    try:
        raw = base64.b64decode(b64.split(",")[-1])
        bgr = _decode_jpeg(raw)
        if bgr is None:
            return await client.send(json.dumps({"event": "error", "data": "Invalid image format"}))
        
        cv2.imwrite(str(ROOT / "faces" / f"{name}.jpg"), bgr)
        
        if not _load_face(name):
            return await client.send(json.dumps({"event": "error", "data": "Saved, but no face detected."}))
        
        await client.send(json.dumps({
            "event": "face_saved",
            "data": {"name": name, "faces": _list_faces()}
        }))
    except Exception as e:
        await client.send(json.dumps({"event": "error", "data": str(e)}))

@app.on("select_face")
async def ws_select_face(client, data):
    name = data.get("name", "")
    if _load_face(name):
        await client.send(json.dumps({"event": "face_changed", "data": {"name": name}}))
    else:
        await client.send(json.dumps({"event": "error", "data": "Failed to load face"}))

@app.on("delete_face")
async def ws_delete_face(client, data):
    name = _safe_name(data.get("name", ""))
    path = ROOT / "faces" / f"{name}.jpg"
    if path.exists():
        path.unlink()
        if _active["name"] == name:
            _active["face"] = None
            _active["name"] = None
    await client.send(json.dumps({
        "event": "face_deleted", 
        "data": {"faces": _list_faces(), "active": _active["name"]}
    }))

@app.on_binary_event("frame")
async def ws_frame(client, payload: bytes):
    global _last_warn
    
    if _active["face"] is None:
        now = time.time()
        if now - _last_warn > 5:
            print("[DLC] WARN: Received frame but no face is selected", flush=True)
            _last_warn = now
        await client.send_binary_event("out", payload)
        return

    if _lock.locked():
        if _last["jpeg"]: await client.send_binary_event("out", _last["jpeg"])
        return

    async with _lock:
        try:
            bgr = _decode_jpeg(payload)
            if bgr is None:
                await client.send_binary_event("out", payload)
                return
            
            bgr = _resize(bgr)
            out_bgr = await asyncio.to_thread(_process_frame_sync, bgr, _active["face"])
                
            out = _encode_jpeg(out_bgr)
            if out:
                _last["jpeg"] = out
                await client.send_binary_event("out", out)
            else:
                await client.send_binary_event("out", payload)
        except BaseException as e:
            print(f"[DLC] frame error: {e!r}", flush=True)
            await client.send_binary_event("out", payload)

if __name__ == "__main__":
    print(f"[DLC] Lynkio listening on 0.0.0.0:{os.environ.get('PORT', '7860')}", flush=True)
    app.run()
