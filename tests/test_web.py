from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from object_classifier.web import create_app


def write_pattern_image(path: Path, rgb: tuple[int, int, int]) -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[..., 0] = rgb[0]
    image[..., 1] = rgb[1]
    image[..., 2] = rgb[2]
    image[::2, ::2] = np.clip(image[::2, ::2] + 20, 0, 255)
    Image.fromarray(image).save(path)


def encode_image_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def test_web_health_endpoint_returns_ok(tmp_path) -> None:
    from fastapi.testclient import TestClient

    app = create_app(storage_root=tmp_path / "store")
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_web_serves_single_page_shell(tmp_path) -> None:
    from fastapi.testclient import TestClient

    app = create_app(storage_root=tmp_path / "store")
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Object Classifier" in response.text
    assert "navigator.mediaDevices.getUserMedia" in response.text


def test_web_register_endpoint_accepts_multiple_images(tmp_path) -> None:
    from fastapi.testclient import TestClient

    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    write_pattern_image(red, (220, 40, 30))
    write_pattern_image(blue, (30, 40, 220))

    app = create_app(storage_root=tmp_path / "store")
    client = TestClient(app)

    response = client.post(
        "/api/register",
        data={"sku_name": "Camera Widget"},
        files=[
            ("images", ("red.png", encode_image_bytes(red), "image/png")),
            ("images", ("blue.png", encode_image_bytes(blue), "image/png")),
        ],
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["decision"] in {"safe_create", "possible_duplicate", "manual_block"}
    assert payload["sample_ids"] or payload["review_id"]
    assert payload["sku_name"] in {"Camera Widget", None}


def test_web_identify_returns_ranked_candidates(tmp_path) -> None:
    from fastapi.testclient import TestClient

    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    query = tmp_path / "query-red.png"
    write_pattern_image(red, (220, 40, 30))
    write_pattern_image(blue, (30, 40, 220))
    write_pattern_image(query, (210, 50, 40))

    app = create_app(storage_root=tmp_path / "store")
    client = TestClient(app)

    client.post(
        "/api/register",
        data={"sku_name": "Red Widget"},
        files=[("images", ("red.png", encode_image_bytes(red), "image/png"))],
    )
    client.post(
        "/api/register",
        data={"sku_name": "Blue Widget"},
        files=[("images", ("blue.png", encode_image_bytes(blue), "image/png"))],
    )

    response = client.post(
        "/api/identify",
        files={"image": ("query-red.png", encode_image_bytes(query), "image/png")},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["candidates"]
    assert payload["candidates"][0]["sku_id"].startswith("sku-")
    assert payload["candidates"][0]["sku_name"] == "Red Widget"
    assert payload["candidates"][0]["score"] >= payload["candidates"][-1]["score"]


def test_web_register_requires_sku_name(tmp_path) -> None:
    from fastapi.testclient import TestClient

    image_path = tmp_path / "sample.png"
    write_pattern_image(image_path, (220, 40, 30))

    app = create_app(storage_root=tmp_path / "store")
    client = TestClient(app)

    response = client.post(
        "/api/register",
        data={"sku_name": ""},
        files=[("images", ("sample.png", encode_image_bytes(image_path), "image/png"))],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "sku_name is required"


def test_web_identify_requires_image(tmp_path) -> None:
    from fastapi.testclient import TestClient

    app = create_app(storage_root=tmp_path / "store")
    client = TestClient(app)

    response = client.post("/api/identify")

    assert response.status_code == 400
    assert response.json()["detail"] == "image is required"
