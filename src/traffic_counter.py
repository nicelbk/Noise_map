"""
방향별 차량 계수 모듈 (상행선 / 하행선)

동작 원리:
  - 화면에 두 개의 가상 계수선(Counting Line)을 그음
  - 차량 중심점이 계수선을 교차할 때 이동 방향 판별
  - 상행/하행 각각 대수 및 대형차 수 집계

방향 판별 기준:
  LANE_SPLIT_AXIS = "horizontal" (기본값)
    → 화면 좌측 통과 = 상행선, 우측 통과 = 하행선
    → 좌→우 이동: 하행, 우→좌 이동: 상행

  LANE_SPLIT_AXIS = "vertical" (드론뷰/정상뷰)
    → 위쪽 통과 = 상행선, 아래쪽 통과 = 하행선
    → 위→아래 이동: 하행, 아래→위 이동: 상행

계수 알고리즘:
  1. 각 track_id의 이동 경로 마지막 N 포인트 확인
  2. 계수선(가상 직선)과 교차 여부 판별
  3. 교차 시 이동 방향 벡터로 상행/하행 결정
  4. 한 ID는 한 방향으로 한 번만 계수 (중복 방지)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    COUNTING_LINE_RATIO,
    DIRECTION_SEPARATION_RATIO,
    LANE_SPLIT_AXIS,
    UPBOUND_LABEL,
    DOWNBOUND_LABEL,
    LARGE_VEHICLE_CLASS_IDS,
)
from yolo_tracker import Detection, TrackHistory

logger = logging.getLogger(__name__)


# ─── 결과 데이터 구조 ────────────────────────────────────────────

@dataclass
class DirectionCount:
    """방향별 계수 결과"""
    label: str              # "상행선" or "하행선"
    total: int = 0          # 전체 차량 수
    large: int = 0          # 대형차 수
    small: int = 0          # 소형차 수
    by_class: Dict[str, int] = field(default_factory=dict)  # 차종별

    @property
    def large_ratio(self) -> float:
        """대형차 비율 (0.0~1.0)"""
        return self.large / self.total if self.total > 0 else 0.0

    def __repr__(self):
        return (
            f"{self.label}: 총 {self.total}대 "
            f"(대형 {self.large}대, {self.large_ratio*100:.1f}%)"
        )


@dataclass
class CountingResult:
    """30초 1회 분석 결과"""
    upbound: DirectionCount
    downbound: DirectionCount
    duration_sec: float
    total_frames: int
    fps: float

    @property
    def total_vehicles(self) -> int:
        return self.upbound.total + self.downbound.total

    def summary_dict(self) -> dict:
        return {
            "duration_sec": round(self.duration_sec, 1),
            "fps": round(self.fps, 1),
            "total_frames": self.total_frames,
            f"{UPBOUND_LABEL}_count": self.upbound.total,
            f"{UPBOUND_LABEL}_large": self.upbound.large,
            f"{UPBOUND_LABEL}_large_ratio": round(self.upbound.large_ratio, 3),
            f"{DOWNBOUND_LABEL}_count": self.downbound.total,
            f"{DOWNBOUND_LABEL}_large": self.downbound.large,
            f"{DOWNBOUND_LABEL}_large_ratio": round(self.downbound.large_ratio, 3),
            "total_vehicles": self.total_vehicles,
        }


# ─── 계수선 정의 ─────────────────────────────────────────────────

class CountingLine:
    """
    가상 계수선 (Virtual Counting Line)

    차량 중심점이 이 선을 통과할 때 이동 방향을 판별.
    """

    def __init__(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        positive_direction: str = "right",  # "right"/"left"/"up"/"down"
        label: str = "",
    ):
        """
        pt1, pt2: 선의 두 끝점 (픽셀 좌표)
        positive_direction: 이 방향으로 통과하면 "하행"으로 판별
        """
        self.pt1 = np.array(pt1, dtype=float)
        self.pt2 = np.array(pt2, dtype=float)
        self.label = label
        self.positive_direction = positive_direction

        # 선 벡터
        self._line_vec = self.pt2 - self.pt1
        # 법선 벡터 (오른쪽이 양수)
        self._normal = np.array(
            [self._line_vec[1], -self._line_vec[0]], dtype=float
        )
        norm = np.linalg.norm(self._normal)
        if norm > 0:
            self._normal /= norm

    def signed_distance(self, point: Tuple[float, float]) -> float:
        """점에서 선까지의 부호 있는 거리 (법선 방향 기준)"""
        p = np.array(point, dtype=float)
        return float(np.dot(p - self.pt1, self._normal))

    def crossed(
        self,
        prev_point: Tuple[float, float],
        curr_point: Tuple[float, float],
    ) -> Optional[str]:
        """
        두 연속 점이 선을 교차했는지 확인
        Returns: "upbound" | "downbound" | None
        """
        d_prev = self.signed_distance(prev_point)
        d_curr = self.signed_distance(curr_point)

        # 부호가 바뀌면 교차
        if d_prev * d_curr >= 0:
            return None  # 교차 없음

        # 이동 방향: d_prev < 0 → d_curr > 0 이면 '양수 방향'으로 통과
        moved_positive = d_curr > 0

        if self.positive_direction in ("right", "down"):
            return DOWNBOUND_LABEL if moved_positive else UPBOUND_LABEL
        else:
            return UPBOUND_LABEL if moved_positive else DOWNBOUND_LABEL

    def draw(self, frame: np.ndarray, color=(0, 255, 255), thickness=2) -> np.ndarray:
        """계수선을 프레임에 그리기"""
        p1 = tuple(self.pt1.astype(int))
        p2 = tuple(self.pt2.astype(int))
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
        if self.label:
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
            cv2.putText(
                frame, self.label, mid,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA
            )
        return frame


# ─── 차량 계수기 ─────────────────────────────────────────────────

class TrafficCounter:
    """
    추적 이력을 분석하여 방향별 차량 수와 대형차 비율을 산출
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.w = frame_width
        self.h = frame_height

        # 계수선 생성
        self.counting_lines = self._build_counting_lines()

        # 이미 계수된 track_id 집합
        self._counted_ids: Dict[str, set] = {
            UPBOUND_LABEL: set(),
            DOWNBOUND_LABEL: set(),
        }

        self.upbound = DirectionCount(label=UPBOUND_LABEL)
        self.downbound = DirectionCount(label=DOWNBOUND_LABEL)

    def _build_counting_lines(self) -> List[CountingLine]:
        """
        카메라 방향 설정에 따라 계수선 생성

        horizontal: 화면 중앙에 수직선 2개 (좌측 영역 = 상행, 우측 = 하행)
        vertical:   화면 중앙에 수평선 2개 (상단 영역 = 상행, 하단 = 하행)
        """
        lines = []

        if LANE_SPLIT_AXIS == "horizontal":
            # 수직 계수선 (화면 좌/우로 나눔)
            mid_x = int(self.w * COUNTING_LINE_RATIO)

            # 상행선 계수선: 화면 좌측 1/4 지점
            x_up = int(self.w * 0.25)
            lines.append(CountingLine(
                pt1=(x_up, 0),
                pt2=(x_up, self.h),
                positive_direction="right",
                label=f"계수선(상행)",
            ))

            # 하행선 계수선: 화면 우측 3/4 지점
            x_down = int(self.w * 0.75)
            lines.append(CountingLine(
                pt1=(x_down, 0),
                pt2=(x_down, self.h),
                positive_direction="right",
                label=f"계수선(하행)",
            ))

        else:  # "vertical"
            # 수평 계수선 (화면 상/하로 나눔)
            # 상행선 계수선: 화면 상단 1/4 지점
            y_up = int(self.h * 0.25)
            lines.append(CountingLine(
                pt1=(0, y_up),
                pt2=(self.w, y_up),
                positive_direction="down",
                label=f"계수선(상행)",
            ))

            # 하행선 계수선: 화면 하단 3/4 지점
            y_down = int(self.h * 0.75)
            lines.append(CountingLine(
                pt1=(0, y_down),
                pt2=(self.w, y_down),
                positive_direction="down",
                label=f"계수선(하행)",
            ))

        return lines

    def update(self, track_histories: Dict[int, TrackHistory]):
        """
        전체 추적 이력을 분석하여 계수 갱신

        각 트랙의 최근 2개 위치로 계수선 교차 여부 판별
        """
        for tid, hist in track_histories.items():
            if hist.counted:
                continue
            if len(hist.centroids) < 2:
                continue

            prev = hist.centroids[-2]
            curr = hist.centroids[-1]

            for line in self.counting_lines:
                direction = line.crossed(prev, curr)
                if direction is None:
                    continue

                # 이미 같은 방향으로 계수된 ID는 스킵
                if tid in self._counted_ids[direction]:
                    continue

                self._counted_ids[direction].add(tid)

                # 계수 증가
                target = (
                    self.upbound if direction == UPBOUND_LABEL else self.downbound
                )
                target.total += 1
                if hist.is_large:
                    target.large += 1
                else:
                    target.small += 1

                # 차종별 집계
                name = hist.class_name
                target.by_class[name] = target.by_class.get(name, 0) + 1

                # 트랙에 방향 기록
                hist.direction = direction
                hist.counted = True

                logger.debug(
                    f"[계수] ID#{tid} {hist.class_name} → {direction} "
                    f"({'대형' if hist.is_large else '소형'})"
                )
                break  # 한 라인에서만 계수

    def update_from_detections(
        self,
        current_detections: List[Detection],
        track_histories: Dict[int, TrackHistory],
    ):
        """프레임별 실시간 업데이트용 (update의 alias)"""
        self.update(track_histories)

    def get_result(
        self, duration_sec: float, total_frames: int, fps: float
    ) -> CountingResult:
        return CountingResult(
            upbound=self.upbound,
            downbound=self.downbound,
            duration_sec=duration_sec,
            total_frames=total_frames,
            fps=fps,
        )

    def reset(self):
        """새 분석 세션을 위한 초기화"""
        self._counted_ids = {
            UPBOUND_LABEL: set(),
            DOWNBOUND_LABEL: set(),
        }
        self.upbound = DirectionCount(label=UPBOUND_LABEL)
        self.downbound = DirectionCount(label=DOWNBOUND_LABEL)

    # ─────────────────────────────────────────────────────────────
    # 시각화
    # ─────────────────────────────────────────────────────────────

    def draw_counting_lines(self, frame: np.ndarray) -> np.ndarray:
        """계수선을 프레임에 오버레이"""
        for line in self.counting_lines:
            frame = line.draw(frame)
        return frame

    def draw_stats_overlay(self, frame: np.ndarray) -> np.ndarray:
        """화면 상단에 통계 정보 오버레이"""
        overlay = frame.copy()

        stats = [
            f"{UPBOUND_LABEL}: {self.upbound.total}대 "
            f"(대형 {self.upbound.large_ratio*100:.0f}%)",
            f"{DOWNBOUND_LABEL}: {self.downbound.total}대 "
            f"(대형 {self.downbound.large_ratio*100:.0f}%)",
        ]

        bg_h = 30 * len(stats) + 10
        cv2.rectangle(overlay, (0, 0), (350, bg_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, text in enumerate(stats):
            cv2.putText(
                frame, text, (8, 28 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA
            )
        return frame


# ─────────────────────────────────────────────────────────────────
# 구간 기반 방향 감지 (계수선 대신 사용 가능한 대안)
# ─────────────────────────────────────────────────────────────────

class ZoneBasedDirectionDetector:
    """
    계수선 대신 화면 구역(Zone)을 기반으로 방향 판별

    차량이 처음 나타난 위치와 현재 위치를 비교하여
    전체 이동 방향을 결정하는 방식 (단방향 도로에 적합)
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.w = frame_width
        self.h = frame_height
        self.split = DIRECTION_SEPARATION_RATIO

    def classify_direction(self, history: TrackHistory) -> Optional[str]:
        """
        이동 이력 전체를 분석하여 주행 방향 분류
        최소 3개 포인트 필요
        """
        if len(history.centroids) < 3:
            return None

        pts = np.array(history.centroids)

        if LANE_SPLIT_AXIS == "horizontal":
            # 평균 x 위치로 레인 결정
            avg_x = float(np.mean(pts[:, 0]))
            # 이동 방향: dx 부호
            dx = float(pts[-1][0] - pts[0][0])
            if avg_x < self.w * self.split:
                # 화면 좌측 → 상행선 방향으로 이동 (오른쪽이 앞쪽)
                return UPBOUND_LABEL if dx < 0 else DOWNBOUND_LABEL
            else:
                # 화면 우측 → 하행선 방향으로 이동 (왼쪽이 앞쪽)
                return DOWNBOUND_LABEL if dx > 0 else UPBOUND_LABEL

        else:  # "vertical"
            avg_y = float(np.mean(pts[:, 1]))
            dy = float(pts[-1][1] - pts[0][1])
            if avg_y < self.h * self.split:
                return UPBOUND_LABEL if dy < 0 else DOWNBOUND_LABEL
            else:
                return DOWNBOUND_LABEL if dy > 0 else UPBOUND_LABEL

    def analyze_all(
        self, track_histories: Dict[int, TrackHistory]
    ) -> Tuple[DirectionCount, DirectionCount]:
        """
        전체 추적 이력에서 방향별 카운트 계산
        Returns: (상행선 카운트, 하행선 카운트)
        """
        upbound = DirectionCount(label=UPBOUND_LABEL)
        downbound = DirectionCount(label=DOWNBOUND_LABEL)

        for tid, hist in track_histories.items():
            direction = self.classify_direction(hist)
            if direction is None:
                continue

            hist.direction = direction
            target = upbound if direction == UPBOUND_LABEL else downbound
            target.total += 1
            if hist.is_large:
                target.large += 1
            else:
                target.small += 1
            target.by_class[hist.class_name] = (
                target.by_class.get(hist.class_name, 0) + 1
            )

        return upbound, downbound
