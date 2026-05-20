#!/usr/bin/env python3
"""Interactive ROI marker — select 4 points on a camera feed to define an ROI quadrilateral."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


class ROIMarker:
    def __init__(self, camera_id: int = 0, width: int | None = None, height: int | None = None):
        self.camera_id = camera_id
        self.req_width = width
        self.req_height = height
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.points: list[tuple[int, int]] = []
        self.drag_index: int | None = None
        self.window = "ROI Marker"

    def run(self) -> list[tuple[int, int]]:
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            sys.exit(f"Cannot open camera {self.camera_id}")

        if self.req_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        if self.req_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)

        # read one frame to get the actual resolution
        ret, frame = cap.read()
        if not ret:
            sys.exit("Cannot read from camera")
        self.actual_height, self.actual_width = frame.shape[:2]
        print(f"Camera native resolution: {self.actual_width}x{self.actual_height}")

        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._on_mouse)

        print("Click 4 points to define the ROI quadrilateral.")
        print(" ENTER — confirm    r — reset    d — undo last    q — quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = self._render(frame)
            cv2.imshow(self.window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == ord("r"):
                self.points.clear()
                self.drag_index = None
            elif key == ord("d"):
                if self.points:
                    self.points.pop()
                self.drag_index = None
            elif key in (13, 10) and len(self.points) == 4:
                # ENTER — confirm
                break

            # handle window close via X button
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()
        return self.points

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drag_index is not None:
                return  # in drag mode, handled by mouse move / release
            if len(self.points) >= 4:
                # check if clicking near an existing point → start drag
                self.drag_index = self._nearest_point(x, y, threshold_sq=100)
                if self.drag_index is None:
                    return  # ignore click beyond 4
            if self.drag_index is None and len(self.points) < 4:
                self.points.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_index = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_index is not None:
            self.points[self.drag_index] = (x, y)

    def _nearest_point(self, x: int, y: int, threshold_sq: int = 100) -> int | None:
        best_idx, best_d2 = None, threshold_sq + 1
        for i, (px, py) in enumerate(self.points):
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        return best_idx

    def _render(self, frame):
        display = frame.copy()
        h, w = display.shape[:2]

        # draw points
        for i, pt in enumerate(self.points):
            cv2.circle(display, pt, 6, (0, 255, 0), -1)
            cv2.putText(display, f"P{i}", (pt[0] + 10, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # draw edges
        pts = self.points
        n = len(pts)
        if n >= 2:
            for i in range(n - 1):
                cv2.line(display, pts[i], pts[i + 1], (0, 255, 0), 2)
        if n == 4:
            cv2.line(display, pts[3], pts[0], (0, 255, 0), 2)
            # filled overlay
            overlay = display.copy()
            cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], (0, 255, 0))
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

        # HUD
        lines = [
            f"Frame: {w}x{h}",
            *[f"P{i}: ({p[0]}, {p[1]})" for i, p in enumerate(pts)],
        ]
        y0 = h - 15 * len(lines) - 10
        for i, line in enumerate(lines):
            cv2.putText(display, line, (12, y0 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if len(pts) < 4:
            hint = f"Click point {len(pts) + 1} of 4"
        else:
            hint = "ROI locked — drag points to adjust | ENTER to confirm, r to reset"
        cv2.putText(display, hint, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return display


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark ROI region on camera feed")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=None, help="Capture width (default: camera native)")
    parser.add_argument("--height", type=int, default=None, help="Capture height (default: camera native)")
    parser.add_argument("--output", type=Path, default=None, help="Save coordinates to JSON file")
    args = parser.parse_args()

    marker = ROIMarker(camera_id=args.camera, width=args.width, height=args.height)
    points = marker.run()

    if len(points) != 4:
        print("ROI not fully defined (need 4 points).")
        sys.exit(1)

    result = [{"x": x, "y": y} for x, y in points]
    print("\nROI coordinates:")
    print(json.dumps(result, indent=2))

    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
