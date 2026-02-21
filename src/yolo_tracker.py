"""
YOLOv8 + ByteTrack 기반 차량 탐지 및 추적 모듈

기능:
  - YOLOv8로 차량(승용차/버스/트럭/오토바이) 탐지
  - ByteTrack으로 프레임 간 차량 추적 (ID 유지)
  - 차종 분류 (대형/소형)
  - 탐지 결과를 구조화된 데이터로 반환
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("ultralytics 미설치. pip install ultralytics 실행 필요")

from config import (
    YOLO_MODEL, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_IMG_SIZE,
    VEHICLE_CLASS_IDS, LARGE_VEHICLE_CLASS_IDS,
    SHOW_TRACKING_ID, SHOW_SPEED, VIDEO_FPS,
)

logger = logging.getLogger(__name__)


# ─── 데이터 구조 ─────────────────────────────────────────────────

@dataclass
class Detection:
    """단일 프레임 내 하나의 차량 탐지 결과"""
    track_id: int               # 추적 고유 ID
    class_id: int               # COCO 클래스 ID
    class_name: str             # 클래스명 (car/bus/truck/motorcycle)
    is_large: bool              # 대형차 여부
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float           # 탐지 신뢰도
    centroid: Tuple[float, float]    # 중심점 (x, y)
    frame_idx: int              # 프레임 번호


@dataclass
class TrackHistory:
    """특정 추적 ID의 전체 이동 이력"""
    track_id: int
    class_id: int
    class_name: str
    is_large: bool
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    first_seen: int = 0
    last_seen: int = 0
    counted: bool = False       # 계수선 통과 여부 (카운팅용)
    direction: Optional[str] = None  # "상행선" or "하행선"
    speed_kmh: Optional[float] = None  # 추정 속도

    def add_detection(self, det: "Detection"):
        self.centroids.append(det.centroid)
        self.frame_indices.append(det.frame_idx)
        self.bboxes.append(det.bbox)
        self.last_seen = det.frame_idx
        if not self.frame_indices or len(self.frame_indices) == 1:
            self.first_seen = det.frame_idx


# ─── YOLO 탐지기 ─────────────────────────────────────────────────

class VehicleDetector:
    """
    YOLOv8 + ByteTrack 기반 차량 탐지·추적기
    """

    def __init__(self, model_path: str = YOLO_MODEL):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics 패키지가 필요합니다: pip install ultralytics")

        logger.info(f"YOLO 모델 로딩: {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()  # 추론 속도 향상

        # 추적 이력 저장소
        self.track_histories: Dict[int, TrackHistory] = {}
        self.frame_idx: int = 0

        # 지원 차량 클래스명 맵
        self.class_names = VEHICLE_CLASS_IDS
        self.large_classes = LARGE_VEHICLE_CLASS_IDS

        logger.info("VehicleDetector 초기화 완료")

    def reset(self):
        """새 분석 세션 시작 시 이력 초기화"""
        self.track_histories.clear()
        self.frame_idx = 0

    # ─────────────────────────────────────────────────────────────
    # 프레임 처리
    # ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        단일 프레임에 YOLOv8 + ByteTrack 적용
        Returns: 이 프레임에서 탐지된 Detection 리스트
        """
        results = self.model.track(
            frame,
            persist=True,                       # 트랙 ID 유지
            classes=list(VEHICLE_CLASS_IDS.keys()),
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            imgsz=YOLO_IMG_SIZE,
            tracker="bytetrack.yaml",           # ByteTrack 사용
            verbose=False,
        )

        detections: List[Detection] = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                try:
                    # track_id가 없는 경우 스킵
                    if box.id is None:
                        continue

                    track_id = int(box.id.item())
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())

                    if class_id not in VEHICLE_CLASS_IDS:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    det = Detection(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=VEHICLE_CLASS_IDS[class_id],
                        is_large=(class_id in LARGE_VEHICLE_CLASS_IDS),
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        centroid=(cx, cy),
                        frame_idx=self.frame_idx,
                    )
                    detections.append(det)

                    # 이력 업데이트
                    self._update_history(det)

                except Exception as e:
                    logger.debug(f"탐지 결과 파싱 오류: {e}")
                    continue

        self.frame_idx += 1
        return detections

    def process_frames(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        프레임 리스트 전체 처리
        Returns: 프레임별 Detection 리스트의 리스트
        """
        all_detections: List[List[Detection]] = []
        total = len(frames)

        logger.info(f"총 {total}프레임 처리 시작...")
        for i, frame in enumerate(frames):
            dets = self.process_frame(frame)
            all_detections.append(dets)
            if VERBOSE_PROGRESS and (i + 1) % 30 == 0:
                logger.debug(f"  처리 진행: {i+1}/{total}")

        logger.info(f"프레임 처리 완료. 추적된 차량 수: {len(self.track_histories)}")
        return all_detections

    # ─────────────────────────────────────────────────────────────
    # 추적 이력 관리
    # ─────────────────────────────────────────────────────────────

    def _update_history(self, det: Detection):
        """탐지 결과로 추적 이력 갱신"""
        tid = det.track_id
        if tid not in self.track_histories:
            self.track_histories[tid] = TrackHistory(
                track_id=tid,
                class_id=det.class_id,
                class_name=det.class_name,
                is_large=det.is_large,
                first_seen=det.frame_idx,
            )
        self.track_histories[tid].add_detection(det)

    def get_active_tracks(self, max_age_frames: int = 10) -> Dict[int, TrackHistory]:
        """최근 max_age_frames 이내 활성 트랙만 반환"""
        cutoff = self.frame_idx - max_age_frames
        return {
            tid: hist
            for tid, hist in self.track_histories.items()
            if hist.last_seen >= cutoff
        }

    def get_all_histories(self) -> Dict[int, TrackHistory]:
        return self.track_histories

    # ─────────────────────────────────────────────────────────────
    # 시각화
    # ─────────────────────────────────────────────────────────────

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        extra_info: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """
        프레임에 탐지 결과, 추적 ID, 속도 등을 오버레이
        extra_info: {track_id: "표시할 문자열"} (예: 속도)
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            tid = det.track_id

            # 차종별 색상
            color = self._get_color(det.class_id)

            # 바운딩 박스
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 레이블 구성
            label_parts = [det.class_name]
            if SHOW_TRACKING_ID:
                label_parts.append(f"#{tid}")
            if SHOW_SPEED and extra_info and tid in extra_info:
                label_parts.append(extra_info[tid])

            label = " ".join(label_parts)

            # 레이블 배경
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - lh - 6),
                (x1 + lw + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # 중심점
            cx, cy = int(det.centroid[0]), int(det.centroid[1])
            cv2.circle(annotated, (cx, cy), 3, color, -1)

        return annotated

    def draw_track_trails(
        self,
        frame: np.ndarray,
        trail_length: int = 20,
    ) -> np.ndarray:
        """활성 트랙의 이동 궤적을 프레임에 그리기"""
        annotated = frame.copy()
        active = self.get_active_tracks(max_age_frames=trail_length)

        for tid, hist in active.items():
            if len(hist.centroids) < 2:
                continue
            color = self._get_color(hist.class_id)
            pts = hist.centroids[-trail_length:]

            for i in range(1, len(pts)):
                alpha = i / len(pts)  # 최근일수록 진하게
                thickness = max(1, int(2 * alpha))
                pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(annotated, pt1, pt2, color, thickness, cv2.LINE_AA)

        return annotated

    @staticmethod
    def _get_color(class_id: int) -> Tuple[int, int, int]:
        """차종별 고정 색상 (BGR)"""
        color_map = {
            2: (0, 200, 0),     # car: 초록
            3: (200, 0, 200),   # motorcycle: 마젠타
            5: (0, 100, 255),   # bus: 주황
            7: (0, 0, 255),     # truck: 빨강
        }
        return color_map.get(class_id, (128, 128, 128))


# verbose progress flag (config에서 가져오되 순환 import 방지)
try:
    from config import VERBOSE as VERBOSE_PROGRESS
except ImportError:
    VERBOSE_PROGRESS = True
