"""
Deep-Live-Cam × Lynkio 1.3.1 — WebSocket binary streaming server.
"""
import os, sys, types, base64, re, json, asyncio, time
from pathlib import Path

# ===========================================================================
# Model persistence — before any insightface import
# ===========================================================================
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("INSIGHTFACE_HOME", str(ROOT / "models" / "insightface"))
os.environ.setdefault("HOME", str(ROOT / "models"))
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "faces").mkdir(exist_ok=True)

# ===========================================================================
# Headless stub for modules.ui — prevents PySide6 / libEGL import
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
MAX_W        = int(os.environ.get("MAX_WIDTH",  "480"))
MAX_H        = int(os.environ.get("MAX_HEIGHT", "360"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "75"))

# ===========================================================================
# State
# ===========================================================================
_active = {"face": None, "name": None}      # current source face
_lock   = asyncio.Lock()                    # serialises frame processing
_last   = {"jpeg": None}                    # last processed frame


def _resize(frame):
    h, w = frame.shape[:2]
    if w <= MAX_W and h <= MAX_H:
        return frame
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
    if not path.exists():
        return None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    face = get_one_face(_resize(bgr))
    if face is None:
        return None
    _active["face"] = face
    _active["name"] = name
    modules.globals.source_path = str(path)
    modules.globals.frame_processors = ["face_swapper"]
    _last["jpeg"] = None
    return face


# ===========================================================================
# Lynkio app
# ===========================================================================
from lynkio import Lynk, render_template, json_response, abort

app = Lynk(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", "7860")),
    protocol="AUTO",
    serve_client=True,          # serves /lynkio/client.js
    rate_limit=60,              # 60 WS messages/sec per client
    max_payload_size=1024 * 1024,
    debug=False,
)

# ===========================================================================
# Static / page routes
# ===========================================================================
app.static("/static", str(ROOT / "static"))


@app.get("/")
async def index(req):
    return render_template("index.html", {"title": "Deep-Live-Cam Studio"})


# ===========================================================================
# Face library HTTP API
# ===========================================================================
@app.get("/api/faces")
async def api_list_faces(req):
    return json_response({
        "faces": _list_faces(),
        "active": _active["name"],
    })


@app.post("/api/faces")
async def api_save_face(req):
    data = await req.json()
    name = _safe_name(data.get("name", ""))
    b64  = data.get("image", "")
    if not name or not b64:
        abort(400, "name and image required")
    try:
        raw = base64.b64decode(b64.split(",")[-1])
        bgr = _decode_jpeg(raw)
        if bgr is None:
            abort(400, "invalid image")
        if get_one_face(_resize(bgr)) is None:
            abort(400, "no face detected")
        cv2.imwrite(str(ROOT / "faces" / f"{name}.jpg"), _resize(bgr))
        return json_response({"ok": True, "name": name})
    except Exception as e:
        abort(400, str(e))


@app.post("/api/faces/select")
async def api_select_face(req):
    data = await req.json()
    name = data.get("name", "")
    if _load_face(name) is None:
        abort(404, "face not found or unreadable")
    return json_response({"ok": True, "active": name})


@app.delete("/api/faces/{name}")
async def api_delete_face(req, name: str):
    path = ROOT / "faces" / f"{name}.jpg"
    if not path.exists():
        abort(404, "not found")
    path.unlink()
    if _active["name"] == name:
        _active["face"] = None
        _active["name"] = None
    return json_response({"ok": True})


# ===========================================================================
# WebSocket events
# ===========================================================================
@app.on("ping")
async def ws_ping(client, data):
    await client.send({"event": "pong", "data": time.time()})


@app.on("set_face")
async def ws_set_face(client, data):
    name = data.get("name", "")
    if _load_face(name):
        await client.send({"event": "face_changed", "data": {"name": name}})
    else:
        await client.send({"event": "error", "data": "face not found"})


@app.on_binary_event("frame")
async def ws_frame(client, payload: bytes):
    """Receive JPEG → process → send JPEG back as named binary event."""
    if _active["face"] is None:
        await client.send_binary_event("out", payload)
        return

    if _lock.locked():
        # Drop stale frame — send last result immediately, keep latency bounded
        if _last["jpeg"]:
            await client.send_binary_event("out", _last["jpeg"])
        return

    async with _lock:
        try:
            bgr = _decode_jpeg(payload)
            if bgr is None:
                await client.send_binary_event("out", payload)
                return
            bgr = _resize(bgr)
            for p in _processors:
                bgr = p.process_frame(_active["face"], bgr)
            out = _encode_jpeg(bgr)
            if out:
                _last["jpeg"] = out
                await client.send_binary_event("out", out)
            else:
                await client.send_binary_event("out", payload)
        except BaseException as e:
            print(f"[DLC] frame error: {e!r}", flush=True)
            await client.send_binary_event("out", payload)


@app.on_internal("connect")
async def on_connect(client, _):
    print(f"[DLC] client connected: {client.id}", flush=True)
    await client.send({
        "event": "ready",
        "data": {"faces": _list_faces(), "active": _active["name"]},
    })


# ===========================================================================
# Run
# ===========================================================================
if __name__ == "__main__":
    print(f"[DLC] Lynkio listening on 0.0.0.0:{os.environ.get('PORT', '7860')}", flush=True)
    app.run()
