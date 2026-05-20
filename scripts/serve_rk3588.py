#!/usr/bin/env python3
"""RK3588 板端 Web 服务启动脚本。

用法:
    python scripts/serve_rk3588.py

环境变量覆盖:
    RKNN_MODEL_DIR  — .rknn 模型文件目录（默认 data/object-classifier/export）
    STORAGE_ROOT    — 数据持久化根目录（默认 data/object-classifier）
    HOST            — 监听地址（默认 0.0.0.0）
    PORT            — 监听端口（默认 8000）
"""

import os
from pathlib import Path

import uvicorn
from object_classifier.web import create_app

RKNN_MODEL_DIR = Path(os.getenv("RKNN_MODEL_DIR", "data/object-classifier/export"))
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "data/object-classifier"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

app = create_app(
    storage_root=STORAGE_ROOT,
    backend="rknn",
    weights_dir=RKNN_MODEL_DIR,
)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
