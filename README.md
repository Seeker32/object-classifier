# object-classifier

## Web UI

Run the FastAPI service and built-in camera page:

```bash
uv run python main.py serve
```

Open `http://127.0.0.1:8000` in a browser that can access the camera.

The page supports:

- camera preview and manual frame capture
- multi-image SKU registration
- single-image identification with ranked candidate results
- HTTP API endpoints at `/api/health`, `/api/register`, and `/api/identify`

## Export

Split model export into two explicit steps:

```bash
python tools/export_onnx.py --output-dir data/object-classifier/export
python tools/export_rknn.py --output-dir data/object-classifier/export
```

`tools/export_onnx.py` exports `embedding.onnx` and `patch_tokens.onnx`.
`tools/export_rknn.py` consumes existing ONNX files and writes `rknn_report.json`.
