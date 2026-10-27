"""
Deep-Live-Cam — Web (mobile + PC)

Env vars:
  EXECUTION_PROVIDER   cpu | cuda | coreml            (default: cpu)
  TARGET_FPS           outgoing frames/sec            (default: 12)
  MAX_WIDTH            inference resolution cap, px   (default: 640)
  MAX_HEIGHT           inference resolution cap, px   (default: 480)
"""

import os
import sys
import threading
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path + ffmpeg setup (mirrors run.py) — must run before processor imports
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ["PATH"] = ROOT + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# Execution provider must be set BEFORE onnxruntime loads
# ---------------------------------------------------------------------------
import modules.globals

_p = os.environ.get("EXECUTION_PROVIDER", "cpu").lower()
modules.globals.execution_providers = {
    "cuda":   ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}.get(_p, ["CPUExecutionProvider"])

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
TARGET_FPS   = float(os.environ.get("TARGET_FPS",   "12"))
STREAM_EVERY = 1.0 / TARGET_FPS
MAX_W        = int(os.environ.get("MAX_WIDTH",   "640"))
MAX_H        = int(os.environ.get("MAX_HEIGHT",  "480"))

# ---------------------------------------------------------------------------
# Deferred imports (after provider is set)
# ---------------------------------------------------------------------------
import gradio as gr
from modules.face_analyser import get_one_face
from modules.processors.frame.core import get_frame_processors_modules

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_state = {
    "source_face": None,
    "processors":  None,
}

# Bounded-latency frame pipeline:
#   - one processing slot (lock)
#   - if busy, return last completed frame immediately (drop stale input)
_proc_lock = threading.Lock()
_last_out  = {"frame": None}


def _resize(frame):
    """Downscale to fit within MAX_W x MAX_H, preserving aspect."""
    h, w = frame.shape[:2]
    if w <= MAX_W and h <= MAX_H:
        return frame
    s = min(MAX_W / w, MAX_H / h)
    return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Source face loader
# ---------------------------------------------------------------------------
def _load_source(source_img):
    if source_img is None:
        return '<div class="dlc-status idle">● Awaiting source image</div>'
    try:
        bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
        bgr = _resize(bgr)

        path = os.path.join(ROOT, "web_source_face.jpg")
        cv2.imwrite(path, bgr)
        modules.globals.source_path     = path
        modules.globals.frame_processors = ["face_swapper"]

        face = get_one_face(bgr)
        if face is None:
            return '<div class="dlc-status err">● No face detected in source</div>'

        _state["source_face"] = face
        _state["processors"]  = get_frame_processors_modules(["face_swapper"])
        _last_out["frame"]    = None  # reset so stale output isn't reused
        return '<div class="dlc-status ok">● Source ready — start your camera</div>'
    except Exception as e:
        return f'<div class="dlc-status err">● {e}</div>'


# ---------------------------------------------------------------------------
# Frame processor (with drop-stale-frame for A/V sync)
# ---------------------------------------------------------------------------
def _process_frame(frame_rgb):
    if frame_rgb is None or _state["source_face"] is None:
        return frame_rgb

    # Previous frame still in flight → drop this one, reuse last result
    if not _proc_lock.acquire(blocking=False):
        return _last_out["frame"] if _last_out["frame"] is not None else frame_rgb

    try:
        frame_rgb = _resize(frame_rgb)
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for p in _state["processors"]:
            bgr = p.process_frame(_state["source_face"], bgr)
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        _last_out["frame"] = out
        return out
    except Exception as e:
        print("frame error:", e)
        return _last_out["frame"] if _last_out["frame"] is not None else frame_rgb
    finally:
        _proc_lock.release()


# ---------------------------------------------------------------------------
# Zoom-style dark CSS
# ---------------------------------------------------------------------------
ZOOM_CSS = """
:root, body, .gradio-container {
    color-scheme: dark !important;
    --body-background-fill: #0d1117 !important;
    --background-fill-primary: #0d1117 !important;
    --background-fill-secondary: #161b22 !important;
    --body-text-color: #e6edf3 !important;
    --block-background-fill: #161b22 !important;
    --block-border-color: #21262d !important;
    --block-label-text-color: #8b949e !important;
    --block-title-text-color: #e6edf3 !important;
    --border-color-primary: #21262d !important;
    --input-background-fill: #0d1117 !important;
    --color-accent: #2d8cff !important;
    --color-accent-soft: rgba(45,140,255,.15) !important;
}

.gradio-container {
    background: radial-gradient(circle at 20% 0%, #101827 0%, #0d1117 60%) !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Header */
.dlc-header {
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(135deg, #1e3a8a 0%, #0e4fa8 100%);
    padding: 20px 28px; border-radius: 16px; margin-bottom: 22px;
    box-shadow: 0 8px 32px rgba(30,58,138,.35), inset 0 1px 0 rgba(255,255,255,.08);
}
.dlc-brand { display:flex; align-items:center; gap:14px; }
.dlc-logo {
    width:52px; height:52px; font-size:28px;
    background: rgba(255,255,255,.15); border-radius:14px;
    display:flex; align-items:center; justify-content:center;
    backdrop-filter: blur(10px);
}
.dlc-title { color:#fff; font-size:20px; font-weight:700; letter-spacing:.2px; }
.dlc-sub   { color:rgba(255,255,255,.75); font-size:13px; margin-top:2px; }
.dlc-badge {
    background: rgba(255,255,255,.2); color:#fff;
    padding:6px 14px; border-radius:20px;
    font-size:11px; font-weight:600; letter-spacing:.8px; text-transform:uppercase;
    display:flex; align-items:center; gap:6px;
    backdrop-filter: blur(10px);
}
.dlc-badge::before {
    content:''; width:7px; height:7px; border-radius:50%;
    background:#4ade80; box-shadow:0 0 8px #4ade80;
    animation: pulse 1.8s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

/* Section titles */
.dlc-section {
    color:#8b949e; font-size:11px; font-weight:700;
    text-transform:uppercase; letter-spacing:1.2px;
    margin: 4px 0 10px 4px;
}

/* Cards */
.block, .form, .gr-box {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,.3) !important;
    padding: 12px !important;
}

/* Video / image rounding */
video, .dlc-webcam video, .dlc-output img, .dlc-source img, .gr-image img {
    border-radius: 10px !important;
    background: #000 !important;
}

/* Status pills */
.dlc-status {
    padding: 10px 16px; border-radius: 10px;
    font-size: 13px; font-weight: 500; margin-top: 10px;
    border: 1px solid transparent; transition: all .25s;
}
.dlc-status.idle { background:rgba(250,204,21,.08); color:#facc15; border-color:rgba(250,204,21,.25); }
.dlc-status.ok   { background:rgba(74,222,128,.08); color:#4ade80; border-color:rgba(74,222,128,.25); }
.dlc-status.err  { background:rgba(248,113,113,.08); color:#f87171; border-color:rgba(248,113,113,.25); }

/* Camera selector injected by JS */
.dlc-cam-selector {
    display:flex; align-items:center; gap:8px;
    background: rgba(45,140,255,.08);
    border: 1px solid rgba(45,140,255,.25);
    padding: 8px 12px; border-radius: 10px; margin-bottom: 10px;
}
.dlc-cam-select {
    flex:1; background:#0d1117; color:#e6edf3;
    border:1px solid #21262d; border-radius:6px;
    padding:6px 10px; font-size:13px; cursor:pointer; outline:none;
}
.dlc-cam-select:focus { border-color:#2d8cff; }

/* Buttons */
button.primary, button.lg {
    background: linear-gradient(135deg,#2d8cff,#1a6fe0) !important;
    border: none !important; color:#fff !important;
    border-radius: 10px !important; font-weight: 600 !important;
    transition: transform .15s, box-shadow .15s !important;
}
button.primary:hover, button.lg:hover {
    transform: translateY(-1px); box-shadow: 0 4px 14px rgba(45,140,255,.4);
}

/* Footer */
.dlc-footer { text-align:center; color:#484f58; font-size:12px; padding:24px 0 8px; }
.dlc-footer a { color:#2d8cff; text-decoration:none; }

/* Mobile */
@media (max-width: 900px) {
    .gradio-container { padding: 10px !important; }
    .dlc-header { flex-direction:column; align-items:flex-start; gap:12px; padding:16px 20px; }
    .dlc-title { font-size:17px; }
    .dlc-logo { width:44px; height:44px; font-size:22px; }
    .gradio-row { flex-direction: column !important; }
    .dlc-webcam video, .dlc-output img { max-height: 48vh !important; }
    button, input, select, textarea { font-size:16px !important; }
    .dlc-badge { display:none; }
}
"""

# ---------------------------------------------------------------------------
# Client-side camera selector (PC only — hidden when single device)
# ---------------------------------------------------------------------------
SCRIPTS = r"""
<script>
(function () {
  const SEL_ID = 'dlc-camera-selector';

  async function inject(video) {
    if (!video || !video.srcObject) return;
    const tracks = video.srcObject.getVideoTracks();
    if (!tracks.length) return;
    if (video.parentElement.querySelector('#' + SEL_ID)) return;

    let devices;
    try { devices = await navigator.mediaDevices.enumerateDevices(); }
    catch { return; }
    const cams = devices.filter(d => d.kind === 'videoinput');
    if (cams.length <= 1) return;

    const s = tracks[0].getSettings ? tracks[0].getSettings() : {};
    const curId = s.deviceId;

    const wrap = document.createElement('div');
    wrap.id = SEL_ID;
    wrap.className = 'dlc-cam-selector';

    const ico = document.createElement('span');
    ico.textContent = '🎥';
    ico.style.cssText = 'font-size:16px;';

    const sel = document.createElement('select');
    sel.className = 'dlc-cam-select';
    cams.forEach((c, i) => {
      const o = document.createElement('option');
      o.value = c.deviceId;
      o.textContent = c.label || ('Camera ' + (i + 1));
      if (c.deviceId === curId) o.selected = true;
      sel.appendChild(o);
    });

    sel.addEventListener('change', async () => {
      try {
        const ns = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: sel.value } }
        });
        if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = ns;
        await video.play().catch(() => {});
      } catch (e) {
        alert('Camera switch failed: ' + e.message);
      }
    });

    wrap.appendChild(ico);
    wrap.appendChild(sel);
    video.parentElement.insertBefore(wrap, video);
  }

  function scan() {
    document.querySelectorAll('video').forEach(v => {
      if (v.dataset.dlcInit === '1') return;
      v.dataset.dlcInit = '1';
      let n = 0;
      const iv = setInterval(async () => {
        if (v.srcObject && v.srcObject.getVideoTracks().length > 0) {
          clearInterval(iv);
          await new Promise(r => setTimeout(r, 900));  // wait for labels
          await inject(v);
        }
        if (++n > 150) clearInterval(iv);
      }, 200);
    });
  }

  window.addEventListener('load', () => {
    setTimeout(scan, 1000);
    new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
    if (navigator.mediaDevices) {
      navigator.mediaDevices.addEventListener('devicechange', () => {
        document.querySelectorAll('#' + SEL_ID).forEach(el => el.remove());
        document.querySelectorAll('video').forEach(v => { v.dataset.dlcInit = '0'; });
        scan();
      });
    }
  });
})();
</script>
"""

HEADER_HTML = """
<div class="dlc-header">
  <div class="dlc-brand">
    <div class="dlc-logo">🎭</div>
    <div>
      <div class="dlc-title">Deep-Live-Cam Studio</div>
      <div class="dlc-sub">Real-time face transformation · mobile &amp; desktop</div>
    </div>
  </div>
  <div class="dlc-badge">Live</div>
</div>
"""

FOOTER_HTML = """
<div class="dlc-footer">
  Tip: on desktop, use the <b>camera selector</b> above the video to choose
  FaceTime, OBS Virtual Camera, or any connected device.<br>
  <a href="https://github.com/lynk-dotcom/Deep-Live-Cam" target="_blank">View on GitHub</a>
</div>
"""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Deep-Live-Cam Studio") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(elem_classes="dlc-main-row"):
        with gr.Column(scale=1, min_width=280):
            gr.HTML('<div class="dlc-section">Source Face</div>')
            source_upload = gr.Image(
                label="", type="numpy",
                sources=["upload"], height=250,
                elem_classes="dlc-source",
            )
            status = gr.HTML(
                '<div class="dlc-status idle">● Awaiting source image</div>'
            )

        with gr.Column(scale=3, min_width=380):
            gr.HTML('<div class="dlc-section">Live Camera</div>')
            webcam = gr.Image(
                label="", sources=["webcam"], type="numpy",
                streaming=True, height=400,
                elem_classes="dlc-webcam",
            )

    gr.HTML('<div class="dlc-section">Processed Output</div>')
    output = gr.Image(
        label="", streaming=True, height=520,
        elem_classes="dlc-output",
    )

    gr.HTML(FOOTER_HTML)

    source_upload.change(_load_source, inputs=[source_upload], outputs=[status])

    # stream_every / time_limit valid in Gradio 5 & 6; fall back if signature differs
    try:
        webcam.stream(
            fn=_process_frame,
            inputs=[webcam],
            outputs=[output],
            time_limit=3600,
            stream_every=STREAM_EVERY,
        )
    except TypeError:
        webcam.stream(fn=_process_frame, inputs=[webcam], outputs=[output])

# ---------------------------------------------------------------------------
# Launch (Gradio 6: css/head go here, not in Blocks())
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=ZOOM_CSS,
        head=SCRIPTS,
        show_error=True,
    )
