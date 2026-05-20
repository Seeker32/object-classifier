from __future__ import annotations

import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .runtime import build_pipeline


def create_app(
    *,
    storage_root: str | Path = "data/object-classifier",
    cache_dir: str | Path | None = None,
    backend: str = "pytorch",
    provider: str = "statistics",
    model_id: str | None = None,
    device: str = "cpu",
    repo_dir: str | Path | None = None,
    weights_dir: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Object Classifier")
    pipeline = build_pipeline(
        storage_root=storage_root,
        cache_dir=cache_dir,
        backend=backend,
        provider=provider,
        model_id=model_id,
        device=device,
        repo_dir=repo_dir,
        weights_dir=weights_dir,
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _html_shell()

    @app.post("/api/register")
    async def register(
        sku_name: str | None = Form(default=None),
        images: list[UploadFile] | None = File(default=None),
    ) -> dict:
        if not sku_name or not sku_name.strip():
            raise HTTPException(status_code=400, detail="sku_name is required")
        if not images:
            raise HTTPException(status_code=400, detail="images are required")

        try:
            with tempfile.TemporaryDirectory(prefix="object-classifier-register-") as temp_dir:
                saved_paths = await _save_uploads(images, Path(temp_dir))
                result = pipeline.register(sku_name.strip(), saved_paths)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive API boundary
            raise HTTPException(status_code=500, detail="service unavailable") from exc

        return {
            "decision": result.decision,
            "sku_id": result.sku.sku_id if result.sku else None,
            "sku_name": result.sku.sku_name if result.sku else None,
            "sample_ids": [sample.sample_id for sample in result.samples],
            "warnings": result.warnings,
            "review_id": result.review_id,
            "reasons": result.reasons,
            "candidates": [_serialize_candidate(candidate, pipeline) for candidate in result.candidates],
        }

    @app.post("/api/identify")
    async def identify(image: UploadFile | None = File(default=None)) -> dict:
        if image is None:
            raise HTTPException(status_code=400, detail="image is required")

        try:
            with tempfile.TemporaryDirectory(prefix="object-classifier-identify-") as temp_dir:
                saved_path = await _save_upload(image, Path(temp_dir))
                result = pipeline.identify(saved_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive API boundary
            raise HTTPException(status_code=500, detail="service unavailable") from exc

        return {
            "decision": result.decision,
            "status": result.status,
            "sku_id": result.sku_id,
            "review_id": result.review_id,
            "reasons": result.reasons,
            "top_candidate": _serialize_candidate(result.top_candidate, pipeline) if result.top_candidate else None,
            "candidates": [_serialize_candidate(candidate, pipeline) for candidate in result.candidates],
            "metadata": _to_jsonable(result.metadata),
        }

    return app


async def _save_uploads(files: list[UploadFile], directory: Path) -> list[Path]:
    saved = []
    for index, upload in enumerate(files, start=1):
        saved.append(await _save_upload(upload, directory, filename=f"capture-{index}{_suffix_for(upload.filename)}"))
    return saved


async def _save_upload(upload: UploadFile, directory: Path, *, filename: str | None = None) -> Path:
    if not upload.filename and filename is None:
        raise HTTPException(status_code=400, detail="uploaded file must have a filename")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (filename or upload.filename or "capture.png")
    payload = await upload.read()
    if not payload:
        raise HTTPException(status_code=400, detail="uploaded image is empty")
    target.write_bytes(payload)
    return target


def _serialize_candidate(candidate, pipeline) -> dict:
    sku = pipeline.repository.get_sku(candidate.sku_id)
    return {
        "sample_id": candidate.sample_id,
        "sku_id": candidate.sku_id,
        "sku_name": sku.sku_name if sku else None,
        "score": candidate.rerank_score if candidate.rerank_score is not None else candidate.global_score,
        "global_score": candidate.global_score,
        "rerank_score": candidate.rerank_score,
        "best_sample_id": candidate.best_sample_id,
        "hit_count": candidate.hit_count,
    }


def _suffix_for(filename: str | None) -> str:
    if not filename:
        return ".png"
    suffix = Path(filename).suffix
    return suffix or ".png"


def _to_jsonable(value):
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _html_shell() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Object Classifier</title>
  <style>
    :root {
      --bg: #f3efe6;
      --panel: #fffaf2;
      --ink: #1f2933;
      --muted: #58616d;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --line: #d9d2c4;
      --warn: #b45309;
      --error: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 32%),
        linear-gradient(180deg, #f7f4ec 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .page {
      min-height: 100vh;
      padding: 24px;
      display: grid;
      gap: 24px;
      grid-template-columns: minmax(320px, 1.2fr) minmax(320px, 1fr);
    }
    .panel {
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 18px 40px rgba(31, 41, 51, 0.08);
    }
    h1, h2 { margin: 0 0 12px; }
    p { margin: 0 0 12px; color: var(--muted); }
    video, canvas.preview {
      width: 100%;
      border-radius: 16px;
      background: #0f172a;
      aspect-ratio: 4 / 3;
      object-fit: cover;
    }
    canvas { display: none; }
    .actions, .stack, .sample-list, .candidate-list { display: grid; gap: 12px; }
    .actions { grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 16px; }
    button {
      border: 0;
      border-radius: 12px;
      padding: 12px 16px;
      font: inherit;
      cursor: pointer;
      color: white;
      background: var(--accent);
    }
    button.secondary { background: #64748b; }
    button.ghost { background: transparent; color: var(--accent-strong); border: 1px solid var(--line); }
    input[type="text"] {
      width: 100%;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      font: inherit;
      background: white;
    }
    .sample-item, .candidate-item, .result-box {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.75);
    }
    .sample-item, .candidate-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .meta { font-size: 14px; color: var(--muted); }
    .status { min-height: 24px; color: var(--muted); }
    .status.error { color: var(--error); }
    .status.warn { color: var(--warn); }
    .thumbnail {
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--line);
      display: none;
    }
    @media (max-width: 960px) {
      .page { grid-template-columns: 1fr; padding: 16px; }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="panel">
      <h1>Object Classifier</h1>
      <p>Use the live camera feed to capture a frame, then register a product or run identification.</p>
      <video id="camera" autoplay playsinline muted></video>
      <canvas id="captureCanvas" class="preview"></canvas>
      <img id="thumbnail" class="thumbnail" alt="Current capture preview">
      <div class="actions">
        <button id="captureButton" type="button">Capture Frame</button>
        <button id="resetButton" class="secondary" type="button">Reset Capture</button>
      </div>
      <p id="cameraStatus" class="status"></p>
    </section>

    <section class="stack">
      <div class="panel">
        <h2>Register Item</h2>
        <p>Capture as many samples as needed, then submit them with the SKU name.</p>
        <input id="skuNameInput" type="text" placeholder="SKU name">
        <div id="sampleList" class="sample-list"></div>
        <div class="actions">
          <button id="addSampleButton" type="button">Add Capture To Batch</button>
          <button id="registerButton" type="button">Register Item</button>
        </div>
      </div>

      <div class="panel">
        <h2>Identify Item</h2>
        <p>Use the current capture to ask the Python service for the best matching candidates.</p>
        <div class="actions">
          <button id="identifyButton" type="button">Identify Capture</button>
        </div>
      </div>

      <div class="panel">
        <h2>Current Result</h2>
        <p id="resultStatus" class="status"></p>
        <div id="resultBox" class="result-box">No result yet.</div>
      </div>
    </section>
  </main>

  <script>
    const camera = document.getElementById("camera");
    const canvas = document.getElementById("captureCanvas");
    const thumbnail = document.getElementById("thumbnail");
    const cameraStatus = document.getElementById("cameraStatus");
    const resultStatus = document.getElementById("resultStatus");
    const resultBox = document.getElementById("resultBox");
    const sampleList = document.getElementById("sampleList");
    const skuNameInput = document.getElementById("skuNameInput");
    const state = { captureBlob: null, sampleBlobs: [] };

    async function bootCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
        camera.srcObject = stream;
        cameraStatus.textContent = "Camera ready.";
      } catch (error) {
        cameraStatus.textContent = `Camera unavailable: ${error.message}`;
        cameraStatus.className = "status error";
      }
    }

    function renderSamples() {
      sampleList.innerHTML = "";
      if (!state.sampleBlobs.length) {
        sampleList.innerHTML = '<div class="meta">No registration samples queued.</div>';
        return;
      }
      state.sampleBlobs.forEach((blob, index) => {
        const row = document.createElement("div");
        row.className = "sample-item";
        row.innerHTML = `<span>Sample ${index + 1}</span>`;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ghost";
        button.textContent = "Remove";
        button.addEventListener("click", () => {
          state.sampleBlobs.splice(index, 1);
          renderSamples();
        });
        row.appendChild(button);
        sampleList.appendChild(row);
      });
    }

    async function captureFrame() {
      if (!camera.videoWidth || !camera.videoHeight) {
        cameraStatus.textContent = "Camera stream is not ready yet.";
        cameraStatus.className = "status warn";
        return;
      }
      canvas.width = camera.videoWidth;
      canvas.height = camera.videoHeight;
      const context = canvas.getContext("2d");
      context.drawImage(camera, 0, 0, canvas.width, canvas.height);
      state.captureBlob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      thumbnail.src = URL.createObjectURL(state.captureBlob);
      thumbnail.style.display = "block";
      cameraStatus.textContent = "Frame captured.";
      cameraStatus.className = "status";
    }

    function resetCapture() {
      state.captureBlob = null;
      thumbnail.removeAttribute("src");
      thumbnail.style.display = "none";
      cameraStatus.textContent = "Capture cleared.";
      cameraStatus.className = "status";
    }

    function ensureCapture() {
      if (!state.captureBlob) {
        throw new Error("Capture a frame first.");
      }
    }

    function showResult(payload) {
      resultBox.textContent = "";
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(payload, null, 2);
      resultBox.appendChild(pre);
      resultStatus.textContent = payload.review_id ? `Manual review: ${payload.review_id}` : "Request completed.";
      resultStatus.className = payload.review_id ? "status warn" : "status";
    }

    async function registerSamples() {
      try {
        if (!skuNameInput.value.trim()) {
          throw new Error("SKU name is required.");
        }
        if (!state.sampleBlobs.length) {
          throw new Error("Add at least one captured sample.");
        }
        const formData = new FormData();
        formData.set("sku_name", skuNameInput.value.trim());
        state.sampleBlobs.forEach((blob, index) => formData.append("images", blob, `capture-${index + 1}.png`));
        const response = await fetch("/api/register", { method: "POST", body: formData });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Register failed.");
        showResult(payload);
      } catch (error) {
        resultStatus.textContent = error.message;
        resultStatus.className = "status error";
      }
    }

    async function identifyCapture() {
      try {
        ensureCapture();
        const formData = new FormData();
        formData.set("image", state.captureBlob, "identify.png");
        const response = await fetch("/api/identify", { method: "POST", body: formData });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Identify failed.");
        showResult(payload);
      } catch (error) {
        resultStatus.textContent = error.message;
        resultStatus.className = "status error";
      }
    }

    document.getElementById("captureButton").addEventListener("click", captureFrame);
    document.getElementById("resetButton").addEventListener("click", resetCapture);
    document.getElementById("addSampleButton").addEventListener("click", () => {
      try {
        ensureCapture();
        state.sampleBlobs.push(state.captureBlob);
        renderSamples();
        cameraStatus.textContent = `Queued ${state.sampleBlobs.length} sample(s).`;
        cameraStatus.className = "status";
      } catch (error) {
        cameraStatus.textContent = error.message;
        cameraStatus.className = "status warn";
      }
    });
    document.getElementById("registerButton").addEventListener("click", registerSamples);
    document.getElementById("identifyButton").addEventListener("click", identifyCapture);

    renderSamples();
    bootCamera();
  </script>
</body>
</html>
""".strip()
