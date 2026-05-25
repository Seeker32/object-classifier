from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from object_classifier.cli import main


def write_pattern_image(path: Path, rgb: tuple[int, int, int]) -> None:
    image = np.zeros((512, 512, 3), dtype=np.uint8)
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
    assert "review_id" not in register_payload

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


def test_cli_reset_requires_yes_flag(tmp_path, capsys) -> None:
    storage_root = tmp_path / "store"
    storage_root.mkdir(parents=True)
    marker = storage_root / "metadata.sqlite3"
    marker.write_text("db", encoding="utf-8")

    assert main(["--storage-root", str(storage_root), "reset"]) == 1

    assert marker.exists()
    assert "--yes is required" in capsys.readouterr().err


def test_cli_reset_removes_runtime_state_and_preserves_export(tmp_path, capsys) -> None:
    storage_root = tmp_path / "store"
    export_dir = storage_root / "export"
    export_dir.mkdir(parents=True)
    export_file = export_dir / "artifact.json"
    export_file.write_text('{"ok": true}', encoding="utf-8")

    (storage_root / "metadata.sqlite3").write_text("db", encoding="utf-8")
    (storage_root / "index.faiss").write_text("index", encoding="utf-8")
    (storage_root / "index_mapping.json").write_text("mapping", encoding="utf-8")
    for dirname in ("metadata", "features", "patch_tokens", "cache"):
        target = storage_root / dirname
        target.mkdir()
        (target / "stale.txt").write_text(dirname, encoding="utf-8")

    assert main(["--storage-root", str(storage_root), "reset", "--yes"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "ok"
    assert str(export_dir) in payload["preserved_paths"]
    assert str(storage_root / "metadata.sqlite3") in payload["removed_paths"]
    assert str(storage_root / "features") in payload["removed_paths"]
    assert export_file.exists()
    assert not (storage_root / "metadata.sqlite3").exists()
    assert not (storage_root / "index.faiss").exists()
    assert not (storage_root / "index_mapping.json").exists()
    assert not (storage_root / "metadata").exists()
    assert not (storage_root / "features").exists()
    assert not (storage_root / "patch_tokens").exists()
    assert not (storage_root / "cache").exists()


def test_cli_reset_is_noop_for_missing_storage_root(tmp_path, capsys) -> None:
    storage_root = tmp_path / "missing-store"

    assert main(["--storage-root", str(storage_root), "reset", "--yes"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "noop"
    assert payload["removed_paths"] == []


def test_cli_supports_serve_command() -> None:
    from object_classifier.cli import build_parser

    args = build_parser().parse_args(["serve", "--host", "127.0.0.1", "--port", "9000"])

    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 9000


def test_cli_parser_does_not_expose_review_confirm() -> None:
    from object_classifier.cli import build_parser

    parser = build_parser()

    try:
        parser.parse_args(["review-confirm", "review-000001", "bind_existing_sku"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("review-confirm command should not be available")


def test_cli_parser_supports_reset_command() -> None:
    from object_classifier.cli import build_parser

    args = build_parser().parse_args(["reset", "--yes"])

    assert args.command == "reset"
    assert args.yes is True
