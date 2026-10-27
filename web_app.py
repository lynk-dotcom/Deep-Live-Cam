"""
Deep-Live-Cam — WhatsApp-style Web Studio (mobile + PC)
"""
import os, sys, types

# ===========================================================================
# 1. Model persistence — MUST be before any insightface import
# ===========================================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ["PATH"] = ROOT + os.pathsep + os.environ.get("PATH", "")

_MODELS = os.path.join(ROOT, "models")
os.makedirs(_MODELS, exist_ok=True)
os.makedirs(os.path.join(_MODELS, "insightface"), exist_ok=True)
os.environ.setdefault("INSIGHTFACE_HOME", os.path.join(_MODELS, "insightface"))
os.environ.setdefault("HOME", _MODELS)

# ===========================================================================
# 2. Headless stub for modules.ui — MUST be before any modules.* import
# ===========================================================================
def _install_ui_stub():
    stub = types.ModuleType("modules.ui")
    def _noop(*_a, **_k): return None
    for _n in ("update_status", "init", "destroy", "close",
               "create_analyser", "register_analyser", "register_frame_processor",
               "set_frame_processor", "start", "start_preview", "stop_preview",
               "init_preview", "update_preview", "update_tk_image", "update_frame"):
        setattr(stub, _n, _noop)
    stub.__dict__["__getattr__"] = lambda _n: _noop
    stub.UI_AVAILABLE = False
    sys.modules["modules.ui"] = stub

_install_ui_stub()

# ===========================================================================
# 3. Imports + model bootstrap
# ===========================================================================
import threading, re, json, time
import cv2
import numpy as np

from download import download_models
download_models()

import modules.globals
_p = os.environ.get("EXECUTION_PROVIDER", "cpu").lower()
modules.globals.execution_providers = {
    "cuda":   ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}.get(_p, ["CPUExecutionProvider"])

TARGET_FPS        = float(os.environ.get("TARGET_FPS",         "5"))
STREAM_EVERY      = 1.0 / TARGET_FPS
MAX_W             = int(os.environ.get("MAX_WIDTH",            "480"))
MAX_H             = int(os.environ.get("MAX_HEIGHT",           "360"))
STREAM_TIME_LIMIT = int(os.environ.get("STREAM_TIME_LIMIT",    "120"))

FACES_DIR = os.path.join(ROOT, "faces")
os.makedirs(FACES_DIR, exist_ok=True)

import gradio as gr
from modules.face_analyser import get_one_face
from modules.processors.frame.core import get_frame_processors_modules

try:
    _preload = get_frame_processors_modules(["face_swapper"])
    print(f"[DLC] preflight OK — {len(_preload)} processor(s) loaded", flush=True)
except BaseException as e:
    import traceback; traceback.print_exc()
    print(f"[DLC] FATAL: {e!r}", flush=True)
    sys.exit(1)

# ===========================================================================
# 4. State + helpers
# ===========================================================================
_state     = {"source_face": None, "processors": _preload, "source_name": None}
_proc_lock = threading.Lock()
_last_out  = {"frame": None}


def _resize(frame):
    if frame is None: return frame
    h, w = frame.shape[:2]
    if w <= MAX_W and h <= MAX_H: return frame
    s = min(MAX_W/w, MAX_H/h)
    return cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)


def _status(kind, text):
    return f'<div class="wa-status {kind}">{text}</div>'


# ===========================================================================
# 5. Face library
# ===========================================================================
def _safe_name(name):
    return re.sub(r'[^a-zA-Z0-9 _-]', '', name or '').strip()[:32] or "unnamed"


def _list_faces():
    items = []
    if not os.path.isdir(FACES_DIR): return items
    for f in sorted(os.listdir(FACES_DIR)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            items.append((os.path.splitext(f)[0], os.path.join(FACES_DIR, f)))
    return items


def _gallery_items():
    return [(p, n) for n, p in _list_faces()]


def _save_face(name, img_rgb):
    if img_rgb is None:
        return _status("err", "No image to save"), _gallery_items()
    name = _safe_name(name)
    try:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        bgr = _resize(bgr)
        if get_one_face(bgr) is None:
            return _status("err", "No face detected in image"), _gallery_items()
        cv2.imwrite(os.path.join(FACES_DIR, f"{name}.jpg"), bgr)
        return _status("ok", f"Saved '{name}' ✓"), _gallery_items()
    except BaseException as e:
        import traceback; traceback.print_exc()
        return _status("err", f"{type(e).__name__}: {e}"), _gallery_items()


def _load_face_by_name(name):
    for n, path in _list_faces():
        if n == name:
            try:
                bgr = cv2.imread(path)
                if bgr is None: return _status("err", "Cannot read file")
                bgr = _resize(bgr)
                face = get_one_face(bgr)
                if face is None: return _status("err", "No face in image")
                _state["source_face"] = face
                _state["source_name"] = name
                modules.globals.source_path = path
                modules.globals.frame_processors = ["face_swapper"]
                _last_out["frame"] = None
                return _status("ok", f"Active face: {name}")
            except BaseException as e:
                return _status("err", f"{type(e).__name__}: {e}")
    return _status("err", "Face not found")


def _delete_face(name):
    for n, path in _list_faces():
        if n == name:
            try:
                os.remove(path)
                if _state.get("source_name") == name:
                    _state["source_face"] = None
                    _state["source_name"] = None
                return _status("ok", f"Deleted '{name}'"), _gallery_items()
            except Exception as e:
                return _status("err", str(e)), _gallery_items()
    return _status("err", "Not found"), _gallery_items()


def _auto_select_first():
    faces = _list_faces()
    if faces:
        return _load_face_by_name(faces[0][0])
    return _status("idle", "Tap 🎭 to pick or add a face")


def _on_face_selected(evt: gr.SelectData):
    faces = _list_faces()
    if 0 <= evt.index < len(faces):
        return _load_face_by_name(faces[evt.index][0])
    return _status("err", "Invalid selection")


# ===========================================================================
# 6. Frame processing
# ===========================================================================
def _process_frame(frame_rgb):
    if frame_rgb is None: return None
    if _state["source_face"] is None: return frame_rgb
    if not _proc_lock.acquire(blocking=False):
        prev = _last_out["frame"]
        return prev if prev is not None else frame_rgb
    try:
        frame_rgb = _resize(frame_rgb)
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for p in _state["processors"]:
            bgr = p.process_frame(_state["source_face"], bgr)
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        _last_out["frame"] = out
        return out
    except BaseException as e:
        print(f"[DLC] FRAME ERROR: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return _last_out["frame"] if _last_out["frame"] is not None else frame_rgb
    finally:
        _proc_lock.release()


def _end_stream():
    _last_out["frame"] = None
    return None


# ===========================================================================
# 7. CSS — WhatsApp video-call look
# ===========================================================================
WA_CSS = """
:root {
    --wa-bg: #0b141a;
    --wa-panel: #111b21;
    --wa-accent: #00a884;
    --wa-text: #e9edef;
    --wa-dim: #8696a0;
}

html, body {
    margin: 0 !important; padding: 0 !important;
    height: 100vh !important; overflow: hidden !important;
    background: var(--wa-bg) !important;
}
.gradio-container {
    background: var(--wa-bg) !important;
    max-width: 100% !important; margin: 0 !important; padding: 0 !important;
    height: 100vh !important; max-height: 100vh !important;
    overflow: hidden !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.gradio-container > .main, .gradio-container > .wrap {
    padding: 0 !important; margin: 0 !important;
    height: 100vh !important; max-height: 100vh !important;
}
footer, .built-with { display: none !important; }
.block, .form, .gr-box {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ===== STAGE (output video, fullscreen) ===== */
#wa-stage {
    position: fixed !important;
    inset: 0;
    z-index: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: radial-gradient(circle at center, #1a1f2e 0%, #050810 100%) !important;
}
#wa-stage .gr-image, #wa-stage .image-container, #wa-stage .image-frame {
    width: 100% !important; height: 100% !important;
    padding: 0 !important; margin: 0 !important;
    border: none !important; border-radius: 0 !important;
    background: transparent !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
}
#wa-stage img {
    max-width: 100% !important; max-height: 100% !important;
    width: auto !important; height: auto !important;
    object-fit: contain !important;
    border-radius: 0 !important;
}
#wa-stage .empty, #wa-stage .placeholder { display: none !important; }

/* ===== PIP (selfie webcam) ===== */
#wa-pip {
    position: fixed !important;
    bottom: 140px; right: 20px;
    width: 130px; height: 175px;
    z-index: 15 !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 2px solid rgba(255,255,255,.22) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,.6) !important;
    background: #000 !important;
}
#wa-pip .gr-image, #wa-pip .image-container, #wa-pip .image-frame {
    width: 100% !important; height: 100% !important;
    padding: 0 !important; margin: 0 !important;
    border: none !important; border-radius: 0 !important;
    background: #000 !important;
}
#wa-pip video, #wa-pip img {
    width: 100% !important; height: 100% !important;
    object-fit: cover !important;
    border-radius: 0 !important;
}
#wa-pip .label-wrap, #wa-pip .icon-buttons,
#wa-pip .progress-text, #wa-pip .image-controls { display: none !important; }

/* ===== TOP BAR ===== */
#wa-topbar {
    position: fixed !important;
    top: 0; left: 0; right: 0;
    z-index: 20 !important;
    padding: 16px 24px !important;
    background: linear-gradient(180deg, rgba(0,0,0,.8) 0%, transparent 100%) !important;
    pointer-events: none;
}
#wa-topbar * { pointer-events: auto; }
.wa-caller { display: flex; align-items: center; gap: 12px; }
.wa-avatar {
    width: 42px; height: 42px; border-radius: 50%;
    background: linear-gradient(135deg, #00a884, #128c7e);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 700; color: #fff;
    box-shadow: 0 4px 12px rgba(0,168,132,.4);
}
.wa-caller-name { font-size: 16px; font-weight: 600; color: #fff; }
.wa-caller-status {
    font-size: 12px; color: rgba(255,255,255,.72);
    display: flex; align-items: center; gap: 6px; margin-top: 2px;
}
.wa-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #25d366; box-shadow: 0 0 8px #25d366;
    animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: .35 } }
.wa-timer {
    font-family: 'SF Mono', 'JetBrains Mono', monospace;
    font-size: 13px; color: rgba(255,255,255,.9);
    background: rgba(0,0,0,.45);
    padding: 6px 12px; border-radius: 20px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,.12);
}
.wa-topbar-inner {
    display: flex; align-items: center; justify-content: space-between;
    max-width: 1600px; margin: 0 auto;
}

/* ===== CONTROL BAR ===== */
#wa-controls {
    position: fixed !important;
    bottom: 0; left: 0; right: 0;
    z-index: 20 !important;
    padding: 20px 20px 32px !important;
    background: linear-gradient(0deg, rgba(0,0,0,.92) 0%, transparent 100%) !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 22px !important;
}
#wa-controls > div, #wa-controls > .block { background: transparent !important; border: none !important; padding: 0 !important; }
#wa-controls button {
    width: 62px !important; height: 62px !important;
    min-width: 62px !important; min-height: 62px !important;
    border-radius: 50% !important;
    background: rgba(255,255,255,.15) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255,255,255,.15) !important;
    color: #fff !important;
    font-size: 26px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all .15s !important;
    cursor: pointer !important;
}
#wa-controls button:hover {
    background: rgba(255,255,255,.25) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,.5) !important;
}
#wa-controls button.primary { background: var(--wa-accent) !important; }
#wa-controls button.danger  { background: #ff3b30 !important; }
#wa-controls button.danger:hover {
    background: #ff1744 !important;
    box-shadow: 0 0 24px rgba(255,59,48,.6) !important;
}

/* ===== FACE PICKER PANEL ===== */
#wa-face-panel {
    position: fixed !important;
    inset: 0 !important;
    z-index: 100 !important;
    background: rgba(0,0,0,.85) !important;
    backdrop-filter: blur(12px) !important;
    display: none !important;
    align-items: flex-end !important;
    justify-content: center !important;
}
#wa-face-panel.open { display: flex !important; }
#wa-face-panel > div, #wa-face-panel > .block {
    background: #1a1f2e !important;
    border-radius: 24px 24px 0 0 !important;
    padding: 20px !important;
    width: 100% !important;
    max-width: 640px !important;
    max-height: 82vh !important;
    overflow-y: auto !important;
    border: none !important;
    animation: waSlideUp .3s ease;
}
@keyframes waSlideUp { from { transform: translateY(100%) } to { transform: translateY(0) } }
.wa-panel-title {
    font-size: 20px; font-weight: 700; color: #fff;
    margin-bottom: 16px;
    display: flex; justify-content: space-between; align-items: center;
}
.wa-close-btn {
    background: rgba(255,255,255,.1) !important;
    border: none !important; color: #fff !important;
    width: 36px !important; height: 36px !important;
    border-radius: 50% !important;
    font-size: 20px !important; cursor: pointer !important;
    min-width: 36px !important; min-height: 36px !important;
}

/* Gallery */
#wa-face-gallery .grid-wrap, #wa-face-gallery .grid { gap: 10px !important; }
#wa-face-gallery .thumbnail-item {
    border-radius: 12px !important;
    border: 2px solid transparent !important;
    transition: all .15s !important;
    cursor: pointer !important;
    background: #12161d !important;
    overflow: hidden;
}
#wa-face-gallery .thumbnail-item:hover { border-color: var(--wa-accent) !important; }
#wa-face-gallery .caption {
    font-size: 11px !important;
    color: #b0b8c0 !important;
    text-align: center !important;
    padding: 4px !important;
}

/* Status */
.wa-status {
    padding: 10px 14px; border-radius: 8px;
    font-size: 13px; font-weight: 500; margin-top: 10px;
}
.wa-status.idle { background: rgba(250,204,21,.1); color: #facc15; border: 1px solid rgba(250,204,21,.3); }
.wa-status.ok   { background: rgba(37,211,102,.1); color: #25d366; border: 1px solid rgba(37,211,102,.3); }
.wa-status.err  { background: rgba(248,113,113,.1); color: #f87171; border: 1px solid rgba(248,113,113,.3); }

/* Camera panel (floating) */
#wa-cam-panel {
    position: fixed !important;
    top: 76px; right: 20px;
    background: #1a1f2e;
    border: 1px solid rgba(255,255,255,.15);
    border-radius: 12px;
    padding: 14px;
    z-index: 30;
    display: none;
    box-shadow: 0 8px 32px rgba(0,0,0,.6);
    min-width: 240px;
}
#wa-cam-panel.open { display: block; }
#wa-cam-panel label { display: block; font-size: 11px; color: var(--wa-dim); margin-bottom: 8px; letter-spacing: 1px; text-transform: uppercase; }
#wa-cam-panel select {
    width: 100%;
    background: #0b141a;
    color: var(--wa-text);
    border: 1px solid #2a3942;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    cursor: pointer;
    outline: none;
}

/* Gradio overrides inside panel */
#wa-face-panel .block, #wa-face-panel .form, #wa-face-panel .gr-box {
    background: #12161d !important;
    border: 1px solid #2a3942 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    box-shadow: none !important;
}
#wa-face-panel label {
    color: var(--wa-dim) !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
#wa-face-panel input, #wa-face-panel textarea {
    background: #0b141a !important;
    color: var(--wa-text) !important;
    border: 1px solid #2a3942 !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    font-size: 14px !important;
}
#wa-face-panel button.primary {
    background: var(--wa-accent) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
#wa-face-panel .label-wrap { color: var(--wa-dim) !important; }

/* Mobile */
@media (max-width: 768px) {
    #wa-pip { width: 105px; height: 140px; bottom: 120px; right: 12px; border-radius: 12px !important; }
    #wa-topbar { padding: 12px 16px !important; }
    .wa-avatar { width: 36px; height: 36px; font-size: 17px; }
    .wa-caller-name { font-size: 14px; }
    .wa-timer { font-size: 11px; padding: 4px 10px; }
    #wa-controls { gap: 16px !important; padding: 14px 14px 26px !important; }
    #wa-controls button {
        width: 56px !important; height: 56px !important;
        min-width: 56px !important; min-height: 56px !important;
        font-size: 22px !important;
    }
    #wa-cam-panel { top: 68px; right: 12px; left: 12px; min-width: auto; }
}
"""

# ===========================================================================
# 8. JavaScript — camera panel, timer, panel toggle
# ===========================================================================
SCRIPTS = r"""
<script>
(function () {
    // ---- Timer (only counts while webcam is actually streaming) ----
    let timerStart = null;
    function tickTimer() {
        const v = document.querySelector('#wa-pip video');
        const active = v && v.srcObject && v.readyState >= 2;
        const el = document.getElementById('wa-timer');
        if (!el) return;
        if (active) {
            if (!timerStart) timerStart = Date.now();
            const s = Math.floor((Date.now() - timerStart) / 1000);
            el.textContent = String(Math.floor(s/60)).padStart(2,'0') + ':' + String(s%60).padStart(2,'0');
        } else {
            timerStart = null;
            if (el.textContent !== '00:00') el.textContent = '00:00';
        }
    }
    setInterval(tickTimer, 500);

    // ---- Camera panel ----
    function buildCamPanel() {
        if (document.getElementById('wa-cam-panel')) return;
        const panel = document.createElement('div');
        panel.id = 'wa-cam-panel';
        panel.innerHTML = '<label>Camera source</label><select id="wa-cam-select"></select>';
        document.body.appendChild(panel);
        document.getElementById('wa-cam-select').addEventListener('change', switchCamera);
    }

    async function refreshCameras() {
        const sel = document.getElementById('wa-cam-select');
        if (!sel) return;
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const cams = devices.filter(d => d.kind === 'videoinput');
            const pip = document.querySelector('#wa-pip video');
            const cur = pip && pip.srcObject && pip.srcObject.getVideoTracks()[0]
                ? pip.srcObject.getVideoTracks()[0].getSettings().deviceId : null;
            sel.innerHTML = '';
            cams.forEach((c, i) => {
                const o = document.createElement('option');
                o.value = c.deviceId;
                o.textContent = c.label || ('Camera ' + (i + 1));
                if (c.deviceId === cur) o.selected = true;
                sel.appendChild(o);
            });
        } catch {}
    }

    async function switchCamera(e) {
        const deviceId = e.target.value;
        try {
            const ns = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: { exact: deviceId } }
            });
            document.querySelectorAll('#wa-pip video').forEach(v => {
                if (v.srcObject) v.srcObject.getTracks().forEach(t => t.stop());
                v.srcObject = ns;
                v.play().catch(() => {});
            });
            timerStart = null;
        } catch (err) {
            alert('Camera switch failed: ' + err.message);
        }
    }

    // Toggle camera panel via the 🎥 button
    document.addEventListener('click', async (e) => {
        if (e.target.closest('#wa-cam-btn')) {
            buildCamPanel();
            await refreshCameras();
            document.getElementById('wa-cam-panel').classList.toggle('open');
        }
        // Close cam panel on outside click
        if (!e.target.closest('#wa-cam-panel') && !e.target.closest('#wa-cam-btn')) {
            const p = document.getElementById('wa-cam-panel');
            if (p) p.classList.remove('open');
        }
        // Open face panel
        if (e.target.closest('#wa-face-btn')) {
            const p = document.getElementById('wa-face-panel');
            if (p) p.classList.add('open');
        }
    });

    // Watch device changes
    if (navigator.mediaDevices) {
        navigator.mediaDevices.addEventListener('devicechange', () => {
            refreshCameras();
        });
    }

    // Reset timer on stream restart
    new MutationObserver(() => {
        const v = document.querySelector('#wa-pip video');
        if (v && !v.srcObject) timerStart = null;
    }).observe(document.body, { childList: true, subtree: true });
})();
</script>
"""

TOPBAR_HTML = """
<div class="wa-topbar-inner">
    <div class="wa-caller">
        <div class="wa-avatar">🎭</div>
        <div>
            <div class="wa-caller-name">Deep-Live-Cam</div>
            <div class="wa-caller-status">
                <span class="wa-dot"></span>
                <span>Live · Encrypted</span>
            </div>
        </div>
    </div>
    <div class="wa-timer" id="wa-timer">00:00</div>
</div>
"""


# ===========================================================================
# 9. UI
# ===========================================================================
with gr.Blocks(title="Deep-Live-Cam · Live") as demo:

    # --- Stage (fullscreen output) ---
    output = gr.Image(
        label="", streaming=True, show_label=False,
        elem_id="wa-stage", interactive=False,
    )

    # --- PiP (selfie webcam) ---
    webcam = gr.Image(
        label="", sources=["webcam"], type="numpy",
        streaming=True, show_label=False,
        elem_id="wa-pip",
    )

    # --- Top bar ---
    gr.HTML(TOPBAR_HTML, elem_id="wa-topbar")

    # --- Control bar ---
    with gr.Row(elem_id="wa-controls"):
        face_btn = gr.Button("🎭", elem_id="wa-face-btn")
        cam_btn  = gr.Button("🎥", elem_id="wa-cam-btn")
        end_btn  = gr.Button("📴", elem_id="wa-end-btn", elem_classes="danger")

    # --- Face picker panel ---
    with gr.Column(elem_id="wa-face-panel"):
        gr.HTML("""
            <div class="wa-panel-title">
                <span>Choose a face</span>
                <button type="button" class="wa-close-btn"
                        onclick="document.getElementById('wa-face-panel').classList.remove('open')">✕</button>
            </div>
        """)
        face_gallery = gr.Gallery(
            value=_gallery_items(),
            show_label=False,
            columns=4, rows=2, height=280,
            allow_preview=False, preview=False,
            elem_id="wa-face-gallery",
        )
        status = gr.HTML(_status("idle", "Tap a face to select it"))

        with gr.Accordion("➕ Add new face", open=False):
            new_name = gr.Textbox(label="Name", placeholder="e.g. Alice", max_lines=1)
            new_img  = gr.Image(label="Photo", type="numpy", sources=["upload"], height=180)
            save_btn = gr.Button("Save to library", variant="primary", elem_classes="primary")

    # ===========================================================================
    # Event wiring
    # ===========================================================================
    demo.load(_auto_select_first, outputs=[status])

    face_gallery.select(_on_face_selected, outputs=[status])
    save_btn.click(_save_face, inputs=[new_name, new_img], outputs=[status, face_gallery])

    # Stream event — serialized via concurrency_id, does not block other events
    stream_event = None
    try:
        stream_event = webcam.stream(
            fn=_process_frame,
            inputs=[webcam],
            outputs=[output],
            time_limit=STREAM_TIME_LIMIT,
            stream_every=STREAM_EVERY,
            concurrency_limit=1,
            concurrency_id="video",
            api_name=False,
            show_progress="hidden",
        )
    except TypeError:
        # Older Gradio signature
        stream_event = webcam.stream(
            fn=_process_frame,
            inputs=[webcam],
            outputs=[output],
        )
    except BaseException as e:
        print(f"[DLC] stream setup failed: {e}", flush=True)
        import traceback; traceback.print_exc()

    if stream_event is not None:
        end_btn.click(_end_stream, cancels=[stream_event])


# ===========================================================================
# 10. Launch
# ===========================================================================
if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=8,   # allows face save / picker / cam switch to run
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=WA_CSS,
        head=SCRIPTS,
        show_error=True,
        favicon_path=None,
    )
