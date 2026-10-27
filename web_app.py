import os
import sys
import cv2
import numpy as np

# ---- path + ffmpeg setup (mirrors run.py) ---------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ["PATH"] = ROOT + os.pathsep + os.environ.get("PATH", "")

# ---- execution provider must be set BEFORE importing processors -----------
import modules.globals
_provider = os.environ.get("EXECUTION_PROVIDER", "cpu").lower()
if _provider == "cuda":
    modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
elif _provider == "coreml":
    modules.globals.execution_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
else:
    modules.globals.execution_providers = ["CPUExecutionProvider"]

import gradio as gr
from modules.face_analyser import get_one_face
from modules.processors.frame.core import get_frame_processors_modules

# ---- state ----------------------------------------------------------------
_state = {"source_face": None, "processors": None}

def _load_source(source_img):
    if source_img is None:
        return "⚠️ Upload a source face image first."
    try:
        bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
        path = os.path.join(ROOT, "web_source_face.jpg")
        cv2.imwrite(path, bgr)
        modules.globals.source_path = path
        modules.globals.frame_processors = ["face_swapper"]

        face = get_one_face(bgr)
        if face is None:
            return "❌ No face detected in source image."
        _state["source_face"] = face
        _state["processors"] = get_frame_processors_modules(["face_swapper"])
        return "✅ Source face loaded. Click the camera to start."
    except Exception as e:
        return f"❌ {e}"

def _process_frame(frame_rgb):
    if frame_rgb is None or _state["source_face"] is None:
        return frame_rgb
    try:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for p in _state["processors"]:
            if p.NAME == "DLC.FACE-ENHANCER":
                bgr = p.process_frame(bgr)
            else:
                bgr = p.process_frame(_state["source_face"], bgr)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("frame error:", e)
        return frame_rgb

# ---- mobile CSS -----------------------------------------------------------
MOBILE_CSS = """
@media (max-width: 768px) {
    .gradio-container { max-width: 100% !important; padding: 8px !important; }
    [style*="grid-template-columns"] { grid-template-columns: 1fr !important; }
    .gr-video, .gr-image { max-height: 45vh !important; }
    button, input, select, textarea { font-size: 16px !important; }
}
"""

# ---- PC camera selector (enumerateDevices + swap srcObject) ---------------
CAMERA_SELECTOR_JS = r"""
<script>
(function() {
  const ID = 'dlc-camera-selector';

  async function build(video) {
    if (!video.srcObject) return;
    const tracks = video.srcObject.getVideoTracks();
    if (!tracks.length) return;
    if (video.parentElement && video.parentElement.querySelector('#' + ID)) return;

    let devices;
    try { devices = await navigator.mediaDevices.enumerateDevices(); }
    catch (e) { return; }

    const cams = devices.filter(d => d.kind === 'videoinput');
    if (cams.length <= 1) return;   // hide on phones with a single lens

    const cur = tracks[0].getSettings ? tracks[0].getSettings() : {};
    const wrap = document.createElement('div');
    wrap.id = ID;
    wrap.style.cssText = 'margin:8px 0;display:flex;align-items:center;gap:8px;';

    const label = document.createElement('span');
    label.textContent = '🎥 Camera:';
    label.style.cssText = 'font-size:14px;font-weight:600;color:#222;';

    const sel = document.createElement('select');
    sel.style.cssText = 'flex:1;padding:6px 10px;font-size:14px;border-radius:6px;border:1px solid #ccc;';

    cams.forEach((c, i) => {
      const o = document.createElement('option');
      o.value = c.deviceId;
      o.textContent = c.label || ('Camera ' + (i + 1));
      if (c.deviceId === cur.deviceId) o.selected = true;
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
      } catch (e) { alert('Camera switch failed: ' + e.message); }
    });

    wrap.appendChild(label);
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
          await new Promise(r => setTimeout(r, 600));  // let labels populate
          await build(v);
        }
        if (++n > 100) clearInterval(iv);
      }, 200);
    });
  }

  window.addEventListener('load', () => {
    setTimeout(scan, 800);
    new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
  });
})();
</script>
"""

# ---- UI -------------------------------------------------------------------
with gr.Blocks(css=MOBILE_CSS, head=CAMERA_SELECTOR_JS,
               title="Deep-Live-Cam Web") as demo:
    gr.Markdown("# 🎭 Deep-Live-Cam — Web (Mobile + PC)")
    gr.Markdown(
        "1. Upload a source face  \n"
        "2. Click the camera box to start your webcam  \n"
        "3. **On PC:** use the dropdown above the video to pick FaceTime / OBS / any camera"
    )

    source_upload = gr.Image(label="Source Face", type="numpy", sources=["upload"])
    status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        webcam = gr.Image(sources=["webcam"], type="numpy",
                          streaming=True, label="Your Camera")
        output = gr.Image(label="Result", streaming=True)

    source_upload.change(_load_source, inputs=[source_upload], outputs=[status])
    webcam.stream(_process_frame, inputs=[webcam], outputs=[output],
                  time_limit=120, stream_every=0.2)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",      # expose to LAN / internet
        server_port=7860,
        show_error=True,
    )
