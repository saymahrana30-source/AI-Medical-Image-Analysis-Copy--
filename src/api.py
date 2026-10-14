"""Flask REST API + tiny upload page.

    python main.py serve            ->  http://127.0.0.1:5000
    POST /predict  (multipart field "image")   [?gradcam=true]
    GET  /health
"""

import base64
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

from . import config
from .gradcam import make_gradcam_heatmap, overlay_heatmap
from .predict import interpret, load_trained_model, model_img_size, predict_batch
from .preprocessing import preprocess_bytes

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Medical Image AI demo</title>
<style>body{font-family:system-ui;max-width:640px;margin:40px auto;padding:0 16px}
.warn{background:#fff3cd;border:1px solid #ffe69c;padding:10px;border-radius:6px}
pre{background:#f4f4f4;padding:12px;border-radius:6px;white-space:pre-wrap;word-break:break-word}</style></head>
<body><h2>AI chest X-ray pneumonia demo</h2>
<p class="warn">__DISCLAIMER__</p>
<input type="file" id="f" accept="image/*"><label><input type="checkbox" id="g"> Grad-CAM</label>
<button onclick="go()">Predict</button><pre id="o">Choose an image.</pre><img id="i">
<script>
async function go(){const f=document.getElementById('f').files[0];if(!f){return}
const fd=new FormData();fd.append('image',f);
const r=await fetch('/predict'+(document.getElementById('g').checked?'?gradcam=true':''),{method:'POST',body:fd});
const j=await r.json();const png=j.gradcam_png_base64;delete j.gradcam_png_base64;
document.getElementById('o').textContent=JSON.stringify(j,null,2);
document.getElementById('i').src=png?('data:image/png;base64,'+png):''}
</script></body></html>""".replace("__DISCLAIMER__", config.DISCLAIMER)


def create_app(model_path=config.MODEL_PATH, threshold: float = config.DECISION_THRESHOLD) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    model = load_trained_model(model_path)  # fail fast if there is no model
    size = model_img_size(model)
    lock = threading.Lock()  # one prediction at a time keeps TensorFlow happy

    @app.get("/")
    def index():
        return PAGE

    @app.get("/health")
    def health():
        return jsonify(status="ok", model=Path(model_path).name, input_size=list(size))

    @app.post("/predict")
    def predict():
        file = request.files.get("image")
        if file is None or file.filename == "":
            return jsonify(error="Send an image in the multipart form field 'image'."), 400
        if Path(file.filename).suffix.lower() not in config.ALLOWED_EXTENSIONS:
            return jsonify(error=f"Allowed types: {sorted(config.ALLOWED_EXTENSIONS)}"), 400
        try:
            batch = preprocess_bytes(file.read(), size)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

        want_cam = request.args.get("gradcam", "").lower() in ("1", "true", "yes")
        with lock:
            if want_cam:
                heat, prob = make_gradcam_heatmap(model, batch, threshold=threshold)
            else:
                prob = float(predict_batch(model, batch)[0])
        result = interpret(prob, threshold)
        result["disclaimer"] = config.DISCLAIMER
        if want_cam:
            base = np.clip(batch[0], 0, 255).astype("uint8")
            overlay = overlay_heatmap(base, heat)
            ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if ok:
                result["gradcam_png_base64"] = base64.b64encode(png.tobytes()).decode()
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f"File too large (max {config.MAX_UPLOAD_MB} MB)."), 413

    return app
