from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from object_classifier.cli import main


def write_pattern_image(path: Path, rgb: tuple[int, int, int]) -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[..., 0] = rgb[0]
    image[..., 1] = rgb[1]
    image[..., 2] = rgb[2]
    image[::2, ::2] = np.clip(image[::2, ::2] + 20, 0, 255)
    Image.fromarray(image).save(path)


def test_cli_register_and_identify(tmp_path, capsys) -> None:
    storage_root = tmp_path / "store"
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    query = tmp_path / "query-red.png"
    write_pattern_image(red, (220, 40, 30))
    write_pattern_image(blue, (30, 40, 220))
    write_pattern_image(query, (210, 50, 40))

    assert main(["--storage-root", str(storage_root), "register", "Red Widget", str(red)]) == 0
    register_payload = json.loads(capsys.readouterr().out)
    assert register_payload["sku_id"].startswith("sku-")

    assert main(["--storage-root", str(storage_root), "register", "Blue Widget", str(blue)]) == 0
    _ = capsys.readouterr()

    assert main(["--storage-root", str(storage_root), "identify", str(query)]) == 0
    identify_payload = json.loads(capsys.readouterr().out)
    assert identify_payload["decision"] == "auto_accept"
    assert identify_payload["sku_id"] == register_payload["sku_id"]


def test_cli_export_writes_report(tmp_path, capsys) -> None:
    output_dir = tmp_path / "export"

    assert main(["export", "--output-dir", str(output_dir)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert Path(payload["report_path"]).exists()
    assert payload["status"] in {"ready", "blocked"}


def test_cli_supports_serve_command() -> None:
    from object_classifier.cli import build_parser

    args = build_parser().parse_args(["serve", "--host", "127.0.0.1", "--port", "9000"])

    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 9000
