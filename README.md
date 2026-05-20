# object-classifier

## Export

Split model export into two explicit steps:

```bash
python tools/export_onnx.py --output-dir data/object-classifier/export
python tools/export_rknn.py --output-dir data/object-classifier/export
```

`tools/export_onnx.py` exports `embedding.onnx` and `patch_tokens.onnx`.
`tools/export_rknn.py` consumes existing ONNX files and writes `rknn_report.json`.
