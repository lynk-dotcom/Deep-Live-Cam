"""
Deep-Live-Cam — Web Studio (streaming platform UI)

Env: EXECUTION_PROVIDER, TARGET_FPS, MAX_WIDTH, MAX_HEIGHT
"""
import os, sys, threading, time
import cv2
import numpy as np

from download import download_models
download_models()


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ["PATH"] = ROOT + os.pathsep + os.environ.get("PATH", "")

import modules.globals
_p = os.environ.get("EXECUTION_PROVIDER", "cpu").lower()
modules.globals.execution_providers = {
    "cuda":   ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}.get(_p, ["CPUExecutionProvider"])

TARGET_FPS   = float(os.environ.get("TARGET_FPS",  "12"))
STREAM_EVERY = 1.0 / TARGET_FPS
MAX_W        = int(os.environ.get("MAX_WIDTH",  "640"))
MAX_H        = int(os.environ.get("MAX_HEIGHT", "480"))

import gradio as gr
from modules.face_analyser import get_one_face
from modules.processors.frame.core import get_frame_processors_modules

# --- preflight: fail fast in docker logs, not mid-stream --------------------
try:
    _preload = get_frame_processors_modules(["face_swapper"])
    print(f"[DLC] preflight OK — {len(_preload)} processor(s) loaded")
except BaseException as e:
    import traceback; traceback.print_exc()
    print(f"[DLC] FATAL: face_swapper failed to load: {e!r}")
    sys.exit(1)

_state     = {"source_face": None, "processors": _preload}
_proc_lock = threading.Lock()
_last_out  = {"frame": None}


def _resize(frame):
    h, w = frame.shape[:2]
    if w <= MAX_W and h <= MAX_H:
        return frame
    s = min(MAX_W / w, MAX_H / h)
    return cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)


def _status(kind, text):
    return f'<div class="dlc-status {kind}">● {text}</div>'


def _load_source(source_img):
    if source_img is None:
        return _status("idle", "Awaiting source image")
    try:
        bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
        bgr = _resize(bgr)
        path = os.path.join(ROOT, "web_source_face.jpg")
        cv2.imwrite(path, bgr)
        modules.globals.source_path      = path
        modules.globals.frame_processors = ["face_swapper"]

        face = get_one_face(bgr)
        if face is None:
            return _status("err", "No face detected in source")

        _state["source_face"] = face
        _last_out["frame"]    = None
        return _status("ok", "Source ready — start your camera")
    except BaseException as e:
        import traceback; traceback.print_exc()
        return _status("err", f"{type(e).__name__}: {e}")


def _process_frame(frame_rgb):
    if frame_rgb is None or _state["source_face"] is None:
        return frame_rgb
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
    except BaseException as e:
        print("frame error:", e)
        return _last_out["frame"] if _last_out["frame"] is not None else frame_rgb
    finally:
        _proc_lock.release()


# ===========================================================================
# STREAMING STUDIO THEME
# ===========================================================================
ZOOM_CSS = """
:root, body, .gradio-container {
    color-scheme: dark !important;
    --body-background-fill: #0a0e14 !important;
    --background-fill-primary: #0a0e14 !important;
    --background-fill-secondary: #12161d !important;
    --body-text-color: #e6edf3 !important;
    --block-background-fill: #12161d !important;
    --block-border-color: #1e242e !important;
    --block-label-text-color: #7d8590 !important;
    --block-title-text-color: #e6edf3 !important;
    --border-color-primary: #1e242e !important;
    --input-background-fill: #0a0e14 !important;
    --color-accent: #ff2d55 !important;
    --color-accent-soft: rgba(255,45,85,.15) !important;
}

.gradio-container {
    background: #0a0e14 !important;
    max-width: 1600px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* === TOP BAR === */
.stream-topbar {
    display:flex; align-items:center; justify-content:space-between;
    padding: 14px 28px;
    background: linear-gradient(180deg,#12161d 0%,#0a0e14 100%);
    border-bottom: 1px solid #1e242e;
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(20px) saturate(180%);
}
.tb-brand { display:flex; align-items:center; gap:12px; }
.tb-logo {
    width:38px; height:38px;
    background: linear-gradient(135deg,#ff2d55,#a020f0);
    border-radius:10px; display:flex; align-items:center; justify-content:center;
    font-size:20px; box-shadow: 0 0 24px rgba(255,45,85,.5);
}
.tb-name { font-size:15px; font-weight:700; color:#fff; letter-spacing:-.2px; }
.tb-sub  { font-size:11px; color:#7d8590; margin-top:1px; letter-spacing:.3px; }
.tb-center { display:flex; align-items:center; gap:16px; }
.live-pill {
    display:flex; align-items:center; gap:8px;
    background: linear-gradient(135deg,#ff2d55,#d91e44);
    color:#fff; padding:6px 14px; border-radius:6px;
    font-size:11px; font-weight:800; letter-spacing:1.8px; text-transform:uppercase;
    box-shadow: 0 0 24px rgba(255,45,85,.5), inset 0 1px 0 rgba(255,255,255,.2);
}
.live-pill::before {
    content:''; width:8px; height:8px; background:#fff; border-radius:50%;
    animation: blink 1.2s infinite; box-shadow: 0 0 8px #fff;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.tb-stats { display:flex; gap:22px; font-family:'JetBrains Mono','SF Mono',monospace; }
.tb-stat  { display:flex; flex-direction:column; align-items:flex-end; }
.tb-stat-val { font-size:14px; font-weight:700; color:#e6edf3; }
.tb-stat-lbl { font-size:10px; color:#7d8590; letter-spacing:.8px; text-transform:uppercase; }

/* === MAIN === */
.stream-main { padding: 20px 24px 24px 24px !important; gap: 20px !important; background: #0a0e14; }
.program-col { min-width: 0 !important; }

.program-stage {
    position: relative;
    background: #000;
    border-radius: 14px;
    border: 1px solid #1e242e;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.03);
}
.program-stage::before {
    content:'PROGRAM';
    position:absolute; top:14px; left:14px;
    background:rgba(0,0,0,.75); color:#ff2d55;
    padding:5px 12px; border-radius:6px;
    font-size:10px; font-weight:800; letter-spacing:2px; z-index:10;
    backdrop-filter:blur(10px); border:1px solid rgba(255,45,85,.3);
}
.program-stage::after {
    content:'●';
    position:absolute; top:12px; right:14px;
    color:#ff2d55; font-size:18px; z-index:10;
    animation: blink 1.5s infinite; text-shadow: 0 0 12px #ff2d55;
}
.program-stage .gr-image {
    border:none !important; border-radius:0 !important;
    background:#000 !important; padding:0 !important;
}
.program-stage video, .program-stage img { border-radius:0 !important; }

.sources-rail {
    background:#0d1117; border:1px solid #1e242e; border-radius:14px;
    padding: 16px !important; display:flex; flex-direction:column; gap:12px;
}
.rail-title {
    font-size:11px; font-weight:800; color:#7d8590;
    letter-spacing:1.5px; text-transform:uppercase; padding-left:2px;
}
.source-card {
    background:#12161d !important; border:1px solid #1e242e !important;
    border-radius:10px !important; padding:12px !important; position:relative;
}
.source-card::before {
    content:''; position:absolute; top:0; left:0; width:100%; height:2px;
    background: linear-gradient(90deg,#2d8cff,#a020f0);
    border-radius:10px 10px 0 0; opacity:.6;
}
.card-label {
    font-size:10px; font-weight:700; color:#7d8590;
    letter-spacing:1.2px; text-transform:uppercase; margin-bottom:8px;
}
.source-card .gr-image {
    border:none !important; background:#000 !important; border-radius:8px !important;
}

.dlc-status {
    padding:10px 14px; border-radius:8px; font-size:12px; font-weight:600;
    display:flex; align-items:center; gap:8px;
}
.dlc-status.idle { background:rgba(250,204,21,.08); color:#facc15; border:1px solid rgba(250,204,21,.25); }
.dlc-status.ok   { background:rgba(74,222,128,.08); color:#4ade80; border:1px solid rgba(74,222,128,.25); }
.dlc-status.err  { background:rgba(248,113,113,.08); color:#f87171; border:1px solid rgba(248,113,113,.25); }

.dlc-cam-selector {
    display:flex; align-items:center; gap:8px;
    background:rgba(255,45,85,.06); border:1px solid rgba(255,45,85,.25);
    padding:8px 10px; border-radius:8px; margin-top:8px;
}
.dlc-cam-select {
    flex:1; background:#0a0e14; color:#e6edf3;
    border:1px solid #1e242e; border-radius:6px;
    padding:6px 10px; font-size:12px; cursor:pointer; outline:none;
}
.dlc-cam-select:focus { border-color:#ff2d55; box-shadow: 0 0 0 3px rgba(255,45,85,.15); }

.block, .form, .gr-box {
    background:transparent !important; border:none !important;
    box-shadow:none !important; padding:0 !important;
}

.dlc-footer {
    text-align:center; color:#484f58; font-size:12px; padding:8px 0 24px;
}
.dlc-footer a { color:#ff2d55; text-decoration:none; }

@media (max-width: 900px) {
    .stream-main { padding:12px !important; gap:12px !important; }
    .gradio-row { flex-direction: column !important; }
    .stream-topbar { padding:10px 16px; flex-wrap:wrap; gap:8px; }
    .tb-center { order:3; width:100%; justify-content:center; margin-top:4px; }
    .tb-stats { gap:12px; }
    .program-stage video, .program-stage img { max-height: 52vh !important; }
    button, input, select, textarea { font-size:16px !important; }
}
"""


SCRIPTS = r"""
<script>
(function () {
  /* ---------- camera selector (PC only) ---------- */
  const SEL_ID = 'dlc-camera-selector';
  async function inject(video) {
    if (!video || !video.srcObject) return;
    const tracks = video.srcObject.getVideoTracks();
    if (!tracks.length) return;
    if (video.parentElement.querySelector('#' + SEL_ID)) return;
    let devices;
    try { devices = await navigator.mediaDevices.enumerateDevices(); } catch { return; }
    const cams = devices.filter(d => d.kind === 'videoinput');
    if (cams.length <= 1) return;
    const s = tracks[0].getSettings ? tracks[0].getSettings() : {};
    const curId = s.deviceId;

    const wrap = document.createElement('div');
    wrap.id = SEL_ID; wrap.className = 'dlc-cam-selector';
    const ico = document.createElement('span');
    ico.textContent = '🎥'; ico.style.fontSize = '14px';
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
        const ns = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: sel.value } } });
        if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = ns;
        await video.play().catch(() => {});
      } catch (e) { alert('Camera switch failed: ' + e.message); }
    });
    wrap.appendChild(ico); wrap.appendChild(sel);
    video.parentElement.insertBefore(wrap, video);
  }
  function scanCams() {
    document.querySelectorAll('video').forEach(v => {
      if (v.dataset.dlcInit === '1') return;
      v.dataset.dlcInit = '1';
      let n = 0;
      const iv = setInterval(async () => {
        if (v.srcObject && v.srcObject.getVideoTracks().length > 0) {
          clearInterval(iv);
          await new Promise(r => setTimeout(r, 900));
          await inject(v);
        }
        if (++n > 150) clearInterval(iv);
      }, 200);
    });
  }
  window.addEventListener('load', () => {
    setTimeout(scanCams, 1000);
    new MutationObserver(scanCams).observe(document.body, { childList: true, subtree: true });
    if (navigator.mediaDevices) {
      navigator.mediaDevices.addEventListener('devicechange', () => {
        document.querySelectorAll('#' + SEL_ID).forEach(el => el.remove());
        document.querySelectorAll('video').forEach(v => v.dataset.dlcInit = '0');
        scanCams();
      });
    }
  });

  /* ---------- FPS + session timer ---------- */
  function waitFor(id, cb) {
    const t0 = Date.now();
    const iv = setInterval(() => {
      const el = document.getElementById(id);
      if (el) { clearInterval(iv); cb(el); }
      else if (Date.now() - t0 > 15000) clearInterval(iv);
    }, 150);
  }
  waitFor('dlc-timer', el => {
    const start = Date.now();
    setInterval(() => {
      const s = Math.floor((Date.now() - start) / 1000);
      const mm = String(Math.floor(s/60)).padStart(2,'0');
      const ss = String(s%60).padStart(2,'0');
      el.textContent = mm + ':' + ss;
    }, 1000);
  });
  waitFor('dlc-fps', el => {
    let count = 0, last = performance.now();
    const attached = new WeakSet();
    function attach(img) {
      if (attached.has(img)) return;
      attached.add(img);
      new MutationObserver(() => { count++; })
        .observe(img, { attributes: true, attributeFilter: ['src'] });
    }
    setInterval(() => {
      document.querySelectorAll('.program-stage img, .program-stage canvas').forEach(attach);
      const now = performance.now(), dt = (now - last) / 1000;
      if (dt >= 1) {
        el.textContent = (count / dt).toFixed(1);
        count = 0; last = now;
      }
    }, 500);
  });
})();
</script>
"""


TOPBAR_HTML = """
<div class="stream-topbar">
  <div class="tb-brand">
    <div class="tb-logo">🎭</div>
    <div>
      <div class="tb-name">Deep-Live-Cam</div>
      <div class="tb-sub">Live Studio</div>
    </div>
  </div>
  <div class="tb-center">
    <div class="live-pill">Live</div>
  </div>
  <div class="tb-stats">
    <div class="tb-stat">
      <div class="tb-stat-val" id="dlc-fps">0.0</div>
      <div class="tb-stat-lbl">FPS</div>
    </div>
    <div class="tb-stat">
      <div class="tb-stat-val" id="dlc-timer">00:00</div>
      <div class="tb-stat-lbl">Session</div>
    </div>
  </div>
</div>
"""

FOOTER_HTML = """
<div class="dlc-footer">
  On desktop, use the <b>camera selector</b> above the video to switch between
  FaceTime, OBS Virtual Camera, or any connected device.
  &nbsp;·&nbsp;
  <a href="https://github.com/lynk-dotcom/Deep-Live-Cam" target="_blank">GitHub</a>
</div>
"""


with gr.Blocks(title="Deep-Live-Cam Studio") as demo:
    gr.HTML(TOPBAR_HTML)

    with gr.Row(elem_classes="stream-main"):
        with gr.Column(scale=4, elem_classes="program-col"):
            with gr.Group(elem_classes="program-stage"):
                output = gr.Image(
                    label="", streaming=True,
                    height=620, show_label=False,
                    elem_classes="program-video",
                )

        with gr.Column(scale=2, min_width=320, elem_classes="sources-rail"):
            gr.HTML('<div class="rail-title">Sources</div>')

            with gr.Group(elem_classes="source-card"):
                gr.HTML('<div class="card-label">Source Face</div>')
                source_upload = gr.Image(
                    label="", type="numpy",
                    sources=["upload"], height=200,
                    show_label=False,
                )
                status = gr.HTML(_status("idle", "Awaiting source image"))

            with gr.Group(elem_classes="source-card"):
                gr.HTML('<div class="card-label">Camera Input</div>')
                webcam = gr.Image(
                    label="", sources=["webcam"], type="numpy",
                    streaming=True, height=200,
                    show_label=False,
                )

    gr.HTML(FOOTER_HTML)

    source_upload.change(_load_source, inputs=[source_upload], outputs=[status])

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


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=ZOOM_CSS,
        head=SCRIPTS,
        show_error=True,
    )
