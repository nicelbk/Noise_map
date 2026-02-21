"""
차량 속도 추정 모듈

알고리즘:
  1. 원근 변환 (Perspective Transform / Homography)
     - 카메라 사각 영상 → 버드아이뷰(Bird's Eye View) 변환
     - 실제 도로 좌표로 매핑하여 픽셀 → 미터 변환

  2. 프레임 간 이동 거리 측정
     - 버드아이뷰 상에서 연속 프레임의 중심점 이동 거리 계산
     - Δdistance (m) / Δtime (s) = 속도 (m/s) → × 3.6 = km/h

  3. 이동 평균 필터로 노이즈 제거

  4. 방향별 평균 속도 계산

한계 및 가정:
  - 카메라의 정확한 설치 위치/각도를 모르므로 원근 기준점은 추정값
  - ROAD_SEGMENT_METERS 설정으로 스케일을 조정하면 더 정확해짐
  - 동일 track_id가 최소 5프레임 이상 추적되어야 신뢰 가능한 속도 산출
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    VIDEO_FPS,
    ROAD_WIDTH_METERS,
    ROAD_SEGMENT_METERS,
    SPEED_SMOOTHING_WINDOW,
    PERSPECTIVE_SRC_POINTS,
    PERSPECTIVE_DST_POINTS,
    UPBOUND_LABEL,
    DOWNBOUND_LABEL,
)
from yolo_tracker import TrackHistory

logger = logging.getLogger(__name__)


# ─── 원근 변환 관리 ──────────────────────────────────────────────

class PerspectiveTransform:
    """
    카메라 뷰 → 버드아이뷰(탑뷰) 원근 변환

    실제 도로 위 좌표계:
      x축: 도로 폭 방향 (0 ~ ROAD_WIDTH_METERS)
      y축: 도로 진행 방향 (0 ~ ROAD_SEGMENT_METERS)
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        src_points: Optional[List[List[float]]] = None,
        road_width_m: float = ROAD_WIDTH_METERS,
        road_length_m: float = ROAD_SEGMENT_METERS,
    ):
        self.w = frame_width
        self.h = frame_height
        self.road_width = road_width_m
        self.road_length = road_length_m

        # 픽셀 기준점 (원근 변환 소스)
        if src_points:
            src = np.array(
                [[p[0] * frame_width, p[1] * frame_height] for p in src_points],
                dtype=np.float32,
            )
        else:
            src = np.array(
                [
                    [p[0] * frame_width, p[1] * frame_height]
                    for p in PERSPECTIVE_SRC_POINTS
                ],
                dtype=np.float32,
            )

        # 실제 거리 기준점 (버드아이뷰 목적지)
        # 출력 좌표: (m 단위) — 나중에 픽셀로 표시할 때 스케일 적용
        dst = np.array(
            [
                [p[0] * road_width_m, p[1] * road_length_m]
                for p in PERSPECTIVE_DST_POINTS
            ],
            dtype=np.float32,
        )

        # 변환 행렬 계산
        try:
            self.M = cv2.getPerspectiveTransform(src, dst)
            self.M_inv = cv2.getPerspectiveTransform(dst, src)
            self._valid = True
        except cv2.error as e:
            logger.warning(f"원근 변환 행렬 계산 실패: {e}. 픽셀 비율 모드 사용")
            self._valid = False
            # 폴백: 단순 선형 스케일
            self._px_per_m_x = frame_width / road_width_m
            self._px_per_m_y = frame_height / road_length_m

    def pixel_to_world(
        self, pixel_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        픽셀 좌표 → 실제 세계 좌표 (미터)
        Returns: (x_meters, y_meters)
        """
        if not self._valid:
            return (
                pixel_point[0] / self._px_per_m_x,
                pixel_point[1] / self._px_per_m_y,
            )

        pt = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
        world_pt = cv2.perspectiveTransform(pt, self.M)
        return float(world_pt[0][0][0]), float(world_pt[0][0][1])

    def world_to_pixel(
        self, world_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """실제 세계 좌표 (미터) → 픽셀 좌표"""
        if not self._valid:
            return (
                world_point[0] * self._px_per_m_x,
                world_point[1] * self._px_per_m_y,
            )

        pt = np.array([[[world_point[0], world_point[1]]]], dtype=np.float32)
        px_pt = cv2.perspectiveTransform(pt, self.M_inv)
        return float(px_pt[0][0][0]), float(px_pt[0][0][1])

    def distance_meters(
        self,
        pixel_pt1: Tuple[float, float],
        pixel_pt2: Tuple[float, float],
    ) -> float:
        """두 픽셀 좌표 사이의 실제 거리 (미터)"""
        w1 = self.pixel_to_world(pixel_pt1)
        w2 = self.pixel_to_world(pixel_pt2)
        dx = w2[0] - w1[0]
        dy = w2[1] - w1[1]
        return float(np.sqrt(dx**2 + dy**2))


# ─── 속도 추정기 ─────────────────────────────────────────────────

class SpeedEstimator:
    """
    차량 추적 이력을 바탕으로 프레임 간 속도를 계산
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fps: float = VIDEO_FPS,
    ):
        self.fps = fps
        self.dt = 1.0 / fps  # 프레임 간격 (초)

        # 원근 변환 초기화
        self.transform = PerspectiveTransform(frame_width, frame_height)

        # 트랙별 속도 이력 (이동 평균용)
        self._speed_buffer: Dict[int, deque] = {}

        # 최종 속도 저장 {track_id: speed_kmh}
        self.speeds: Dict[int, float] = {}

    def update(self, track_histories: Dict[int, TrackHistory]):
        """
        전체 추적 이력으로 속도 계산 (배치 처리)
        """
        for tid, hist in track_histories.items():
            if len(hist.centroids) < 2:
                continue
            speed = self._compute_track_speed(hist)
            if speed is not None:
                hist.speed_kmh = speed
                self.speeds[tid] = speed

    def update_single_frame(
        self,
        track_id: int,
        prev_centroid: Tuple[float, float],
        curr_centroid: Tuple[float, float],
    ) -> Optional[float]:
        """
        단일 프레임 업데이트용 (실시간 처리)
        Returns: 현재 속도 (km/h) 또는 None
        """
        dist_m = self.transform.distance_meters(prev_centroid, curr_centroid)
        speed_ms = dist_m / self.dt
        speed_kmh = speed_ms * 3.6

        # 비현실적인 값 필터링 (0~200 km/h)
        if not (0 < speed_kmh < 200):
            return None

        # 이동 평균 필터
        if track_id not in self._speed_buffer:
            self._speed_buffer[track_id] = deque(maxlen=SPEED_SMOOTHING_WINDOW)
        self._speed_buffer[track_id].append(speed_kmh)

        smoothed = float(np.mean(self._speed_buffer[track_id]))
        self.speeds[track_id] = smoothed
        return smoothed

    def _compute_track_speed(self, hist: TrackHistory) -> Optional[float]:
        """
        추적 이력 전체에서 평균 속도 계산
        연속 프레임 간 속도를 계산하고 이동 평균 적용
        """
        centroids = hist.centroids
        frame_indices = hist.frame_indices

        if len(centroids) < 2:
            return None

        speeds = []
        for i in range(1, len(centroids)):
            prev = centroids[i - 1]
            curr = centroids[i]

            # 프레임 차이 (1 이상이면 dt 보정)
            frame_gap = frame_indices[i] - frame_indices[i - 1]
            if frame_gap <= 0:
                continue
            actual_dt = frame_gap * self.dt

            dist_m = self.transform.distance_meters(prev, curr)
            speed_ms = dist_m / actual_dt
            speed_kmh = speed_ms * 3.6

            # 비현실적 값 제거
            if 0 < speed_kmh < 200:
                speeds.append(speed_kmh)

        if not speeds:
            return None

        # 이상치 제거 (±2σ 범위)
        speeds_arr = np.array(speeds)
        mean = np.mean(speeds_arr)
        std = np.std(speeds_arr)
        filtered = speeds_arr[np.abs(speeds_arr - mean) < 2 * std]

        return float(np.mean(filtered)) if len(filtered) > 0 else float(mean)

    def get_speed_label(self, track_id: int) -> str:
        """트랙 ID의 속도 레이블 문자열 반환"""
        speed = self.speeds.get(track_id)
        if speed is None:
            return ""
        return f"{speed:.0f}km/h"

    def get_speed_labels_dict(self) -> Dict[int, str]:
        """모든 트랙의 속도 레이블 딕셔너리 반환"""
        return {tid: self.get_speed_label(tid) for tid in self.speeds}

    # ─────────────────────────────────────────────────────────────
    # 방향별 속도 통계
    # ─────────────────────────────────────────────────────────────

    def compute_direction_stats(
        self, track_histories: Dict[int, TrackHistory]
    ) -> Dict[str, Dict]:
        """
        방향별 속도 통계 계산
        Returns:
          {
            "상행선": {"mean": 60.5, "median": 62.0, "std": 5.2,
                       "min": 45.0, "max": 80.0, "count": 12},
            "하행선": {...}
          }
        """
        direction_speeds: Dict[str, List[float]] = {
            UPBOUND_LABEL: [],
            DOWNBOUND_LABEL: [],
        }

        for tid, hist in track_histories.items():
            if hist.direction and hist.speed_kmh is not None:
                direction_speeds[hist.direction].append(hist.speed_kmh)

        result = {}
        for direction, speeds in direction_speeds.items():
            if not speeds:
                result[direction] = {
                    "mean": 0.0, "median": 0.0, "std": 0.0,
                    "min": 0.0, "max": 0.0, "count": 0,
                }
                continue

            arr = np.array(speeds)
            result[direction] = {
                "mean": round(float(np.mean(arr)), 1),
                "median": round(float(np.median(arr)), 1),
                "std": round(float(np.std(arr)), 1),
                "min": round(float(np.min(arr)), 1),
                "max": round(float(np.max(arr)), 1),
                "count": len(speeds),
            }

        return result

    # ─────────────────────────────────────────────────────────────
    # 교정 도구
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def calibrate_from_known_distance(
        pixel_distance: float,
        real_distance_m: float,
        fps: float,
        frame_gap: int = 1,
    ) -> float:
        """
        알려진 두 점 사이 거리로 픽셀-미터 스케일 계산
        (캘리브레이션 도구)

        pixel_distance: 두 기준점 사이 픽셀 거리
        real_distance_m: 실제 거리 (m)
        Returns: 초당 1픽셀 이동 시 속도 (m/s)
        """
        px_per_m = pixel_distance / real_distance_m
        speed_per_px_per_frame = (1.0 / px_per_m) * fps / frame_gap
        logger.info(
            f"캘리브레이션 결과: {px_per_m:.2f} px/m, "
            f"속도 계수: {speed_per_px_per_frame:.4f} m/s per px/frame"
        )
        return speed_per_px_per_frame

    def draw_bev_debug(
        self,
        frame: np.ndarray,
        track_histories: Dict[int, TrackHistory],
    ) -> np.ndarray:
        """
        버드아이뷰 변환 결과 디버그 시각화
        실제 좌표로 변환된 이동 경로를 표시
        """
        h, w = frame.shape[:2]
        bev_scale = 10  # 1m = 10픽셀
        bev_w = int(self.transform.road_width * bev_scale) + 40
        bev_h = int(self.transform.road_length * bev_scale) + 40
        bev = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

        # 도로 경계 그리기
        cv2.rectangle(bev, (20, 20), (bev_w - 20, bev_h - 20), (40, 40, 40), -1)
        cv2.rectangle(bev, (20, 20), (bev_w - 20, bev_h - 20), (100, 100, 100), 1)

        for tid, hist in track_histories.items():
            if len(hist.centroids) < 2:
                continue
            world_pts = [
                self.transform.pixel_to_world(c) for c in hist.centroids[-20:]
            ]
            bev_pts = [
                (int(p[0] * bev_scale) + 20, int(p[1] * bev_scale) + 20)
                for p in world_pts
            ]

            color = (0, 200, 0) if hist.direction == UPBOUND_LABEL else (0, 0, 200)
            for i in range(1, len(bev_pts)):
                cv2.line(bev, bev_pts[i - 1], bev_pts[i], color, 2)

        return bev
