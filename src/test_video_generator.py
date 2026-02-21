"""
합성 교통 영상 생성기 (테스트용)

실제 CCTV가 없는 환경에서 분석 파이프라인 검증용으로 사용.
가상의 도로 위에 움직이는 차량(사각형)을 그려 mp4 파일 생성.
"""

import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import VIDEO_FPS, ROAD_WIDTH_METERS


def make_test_video(
    output_path: str = "test_traffic.mp4",
    duration_sec: int = 30,
    fps: int = 15,
    width: int = 1280,
    height: int = 720,
    seed: int = 42,
):
    """
    도로 위를 달리는 차량을 시뮬레이션한 합성 영상 생성

    레이아웃:
      왼쪽 절반: 상행선 (차량이 오른쪽→왼쪽 이동)
      오른쪽 절반: 하행선 (차량이 왼쪽→오른쪽 이동)
    """
    rng = random.Random(seed)
    total_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ─── 차량 정의 ─────────────────────────────────────────────
    # COCO 클래스 ID 기준 차량 크기/색상 설정
    VEHICLE_TYPES = [
        # (class_name, w, h, color_bgr, speed_range, is_large)
        ("car",        70, 35, (50, 130, 220),  (3, 7),  False),
        ("car",        65, 32, (80, 80, 200),   (4, 8),  False),
        ("truck",     110, 45, (30, 30, 160),   (2, 5),  True),
        ("bus",       120, 50, (20, 100, 200),  (2, 4),  True),
        ("motorcycle", 35, 20, (150, 50, 150),  (5, 10), False),
    ]

    mid_x = width // 2  # 중앙선

    class Vehicle:
        _id_counter = 0

        def __init__(self, direction: str):
            Vehicle._id_counter += 1
            self.id = Vehicle._id_counter
            vtype = rng.choice(VEHICLE_TYPES)
            self.name, self.w, self.h = vtype[0], vtype[1], vtype[2]
            self.color = vtype[3]
            self.speed = rng.uniform(*vtype[4])
            self.is_large = vtype[5]
            self.direction = direction  # "up" or "down"

            # 차선 위치 (상행: 좌측, 하행: 우측)
            if direction == "up":
                lane_x_min, lane_x_max = 20, mid_x - 20
                self.x = float(rng.randint(lane_x_max - 20, lane_x_max))
                self.dx = -self.speed  # 오른쪽→왼쪽
            else:
                lane_x_min, lane_x_max = mid_x + 20, width - 20
                self.x = float(rng.randint(lane_x_min, lane_x_min + 20))
                self.dx = self.speed   # 왼쪽→오른쪽

            # y 위치: 도로 내 랜덤 차선
            lane_count = 2
            lane_h = (height - 40) // lane_count
            lane = rng.randint(0, lane_count - 1)
            base_y = 20 + lane * lane_h + lane_h // 2
            self.y = float(base_y + rng.randint(-10, 10))

        @property
        def cx(self): return int(self.x)
        @property
        def cy(self): return int(self.y)
        @property
        def x1(self): return int(self.x - self.w // 2)
        @property
        def y1(self): return int(self.y - self.h // 2)
        @property
        def x2(self): return int(self.x + self.w // 2)
        @property
        def y2(self): return int(self.y + self.h // 2)

        def move(self):
            self.x += self.dx

        def is_alive(self) -> bool:
            return -self.w < self.x < width + self.w

        def draw(self, frame: np.ndarray):
            # 차량 본체
            cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2),
                          self.color, -1)
            # 윤곽선
            cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2),
                          (200, 200, 200), 1)
            # 앞유리(밝은 영역)
            win_w = self.w // 3
            win_h = self.h // 2
            if self.dx < 0:  # 좌측 이동 → 왼쪽이 앞
                wx1 = self.x1 + 4
            else:
                wx1 = self.x2 - win_w - 4
            wy1 = self.y1 + self.h // 4
            cv2.rectangle(frame,
                          (wx1, wy1),
                          (wx1 + win_w, wy1 + win_h),
                          (200, 230, 255), -1)
            # 대형차 표시 (지붕 라인)
            if self.is_large:
                cv2.rectangle(frame,
                              (self.x1 + 2, self.y1 - 8),
                              (self.x2 - 2, self.y1 + 2),
                              self.color, -1)

    # ─── 차량 스폰 스케줄 ──────────────────────────────────────
    # 상행: 오른쪽에서 스폰, 하행: 왼쪽에서 스폰
    vehicles = []
    spawn_intervals_up = []
    spawn_intervals_down = []

    # 랜덤 스폰 시점 생성 (30초 × fps 프레임 기준)
    avg_spawn_gap = fps * 2  # 평균 2초마다 1대
    t = 0
    while t < total_frames:
        gap = int(rng.gauss(avg_spawn_gap, avg_spawn_gap * 0.4))
        gap = max(fps // 2, gap)
        t += gap
        if t < total_frames:
            spawn_intervals_up.append(t)

    t = int(fps * 0.5)  # 하행은 0.5초 후부터
    while t < total_frames:
        gap = int(rng.gauss(avg_spawn_gap * 0.9, avg_spawn_gap * 0.3))
        gap = max(fps // 2, gap)
        t += gap
        if t < total_frames:
            spawn_intervals_down.append(t)

    # ─── 프레임 렌더링 ─────────────────────────────────────────
    for frame_idx in range(total_frames):
        # 배경: 도로 (짙은 회색)
        frame = np.full((height, width, 3), 45, dtype=np.uint8)

        # 도로 표시 (약간 밝은 아스팔트)
        cv2.rectangle(frame, (0, 0), (width, height), (55, 55, 55), -1)

        # 중앙 분리선
        dash_len = 30
        dash_gap = 20
        for y in range(0, height, dash_len + dash_gap):
            cv2.line(frame, (mid_x, y), (mid_x, y + dash_len),
                     (255, 255, 0), 3)

        # 차선 구분선 (점선, 상행/하행 각 2차선)
        for side_x in [mid_x // 2, mid_x + mid_x // 2]:
            for y in range(0, height, dash_len + dash_gap):
                cv2.line(frame, (side_x, y), (side_x, y + dash_len),
                         (180, 180, 180), 1, cv2.LINE_AA)

        # 도로 경계선
        cv2.line(frame, (5, 0), (5, height), (255, 255, 255), 2)
        cv2.line(frame, (width - 5, 0), (width - 5, height), (255, 255, 255), 2)

        # 방향 라벨
        cv2.putText(frame, f"[{chr(8592)} 상행선]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.putText(frame, f"[하행선 {chr(8594)}]", (mid_x + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        # 프레임 번호
        t_sec = frame_idx / fps
        cv2.putText(frame, f"t={t_sec:.1f}s  frame={frame_idx}",
                    (width - 250, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # 차량 스폰
        if frame_idx in spawn_intervals_up:
            vehicles.append(Vehicle("up"))
        if frame_idx in spawn_intervals_down:
            vehicles.append(Vehicle("down"))

        # 차량 이동 + 그리기
        alive = []
        for v in vehicles:
            v.move()
            if v.is_alive():
                v.draw(frame)
                alive.append(v)
        vehicles = alive

        out.write(frame)

    out.release()
    n_up = sum(1 for t in spawn_intervals_up)
    n_down = sum(1 for t in spawn_intervals_down)
    print(f"테스트 영상 생성 완료: {output_path}")
    print(f"  해상도: {width}x{height}  FPS: {fps}  시간: {duration_sec}초")
    print(f"  상행선 차량: {n_up}대 스폰 예정")
    print(f"  하행선 차량: {n_down}대 스폰 예정")
    return output_path


if __name__ == "__main__":
    make_test_video("test_traffic.mp4", duration_sec=30, fps=15)
