"""
단일 세션(30초) 분석 오케스트레이터

프레임 리스트를 받아:
  1. YOLO + ByteTrack 으로 차량 탐지/추적
  2. 방향별 차량 계수 (계수선 + 구역 기반 보완)
  3. 속도 추정
  4. 분석 영상 저장 (선택)
  5. 결과 구조체 반환
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    OUTPUT_DIR,
    SAVE_ANNOTATED_VIDEO,
    VIDEO_FPS,
    UPBOUND_LABEL,
    DOWNBOUND_LABEL,
    DEBUG_MODE,
)
from yolo_tracker import VehicleDetector, Detection, TrackHistory
from traffic_counter import TrafficCounter, ZoneBasedDirectionDetector, CountingResult
from speed_estimator import SpeedEstimator

logger = logging.getLogger(__name__)


class SessionAnalyzer:
    """
    30초 단위 단일 분석 세션
    """

    def __init__(self, session_id: int = 0):
        self.session_id = session_id
        self.detector: Optional[VehicleDetector] = None
        self.counter: Optional[TrafficCounter] = None
        self.speed_est: Optional[SpeedEstimator] = None
        self.zone_detector: Optional[ZoneBasedDirectionDetector] = None

        # 분석 결과
        self.counting_result: Optional[CountingResult] = None
        self.speed_stats: Optional[Dict] = None
        self.track_histories: Dict[int, TrackHistory] = {}

        # 메타데이터
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.actual_fps: float = VIDEO_FPS
        self.total_frames: int = 0

    def analyze(
        self,
        frames: List[np.ndarray],
        fps: float = VIDEO_FPS,
        output_video_path: Optional[str] = None,
    ) -> Dict:
        """
        프레임 리스트 분석 실행
        Returns: 분석 결과 딕셔너리
        """
        if not frames:
            logger.error("분석할 프레임이 없습니다")
            return {}

        self.total_frames = len(frames)
        self.actual_fps = fps
        h, w = frames[0].shape[:2]
        self.frame_width = w
        self.frame_height = h
        duration_sec = self.total_frames / fps

        logger.info(
            f"\n[세션 {self.session_id}] 분석 시작 "
            f"({w}x{h}, {self.total_frames}프레임, {duration_sec:.1f}초)"
        )

        # 컴포넌트 초기화
        self.detector = VehicleDetector()
        self.counter = TrafficCounter(w, h)
        self.speed_est = SpeedEstimator(w, h, fps=fps)
        self.zone_detector = ZoneBasedDirectionDetector(w, h)

        # 출력 비디오 설정
        out_writer = None
        if output_video_path and SAVE_ANNOTATED_VIDEO:
            os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, (w, h)
            )
            logger.info(f"분석 영상 저장 경로: {output_video_path}")

        # ─── 프레임별 처리 ────────────────────────────────────────
        t_start = time.time()
        prev_centroids: Dict[int, Tuple[float, float]] = {}

        for frame_idx, frame in enumerate(frames):
            # YOLO 탐지 + ByteTrack 추적
            detections = self.detector.process_frame(frame)

            # 실시간 속도 업데이트
            for det in detections:
                tid = det.track_id
                if tid in prev_centroids:
                    self.speed_est.update_single_frame(
                        tid, prev_centroids[tid], det.centroid
                    )
                prev_centroids[tid] = det.centroid

            # 계수선 기반 카운팅
            self.counter.update(self.detector.track_histories)

            # 시각화
            if out_writer or DEBUG_MODE:
                speed_labels = self.speed_est.get_speed_labels_dict()
                annotated = self.detector.annotate_frame(
                    frame, detections, extra_info=speed_labels
                )
                annotated = self.detector.draw_track_trails(annotated)
                annotated = self.counter.draw_counting_lines(annotated)
                annotated = self.counter.draw_stats_overlay(annotated)

                # 프레임 번호 표시
                cv2.putText(
                    annotated,
                    f"Frame: {frame_idx+1}/{self.total_frames}  "
                    f"Session: {self.session_id}",
                    (w - 340, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                if out_writer:
                    out_writer.write(annotated)

                if DEBUG_MODE:
                    cv2.imshow(f"Session {self.session_id}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        if out_writer:
            out_writer.release()
        if DEBUG_MODE:
            cv2.destroyAllWindows()

        t_elapsed = time.time() - t_start
        logger.info(
            f"처리 완료: {self.total_frames}프레임 / {t_elapsed:.1f}초 "
            f"(처리 FPS: {self.total_frames/t_elapsed:.1f})"
        )

        # ─── 사후 분석 ────────────────────────────────────────────
        self.track_histories = self.detector.get_all_histories()

        # 배치 속도 계산 (트랙 전체 이력 기반)
        self.speed_est.update(self.track_histories)

        # 구역 기반 방향 재분류 (계수선 방식 보완)
        self._post_classify_directions()

        # 최종 속도 통계
        self.speed_stats = self.speed_est.compute_direction_stats(
            self.track_histories
        )

        # 카운팅 결과 생성
        self.counting_result = self.counter.get_result(
            duration_sec=duration_sec,
            total_frames=self.total_frames,
            fps=self.actual_fps,
        )

        return self._build_result_dict()

    def _post_classify_directions(self):
        """
        계수선 미통과 차량(계수되지 않은 차량)에 구역 기반 방향 적용
        영상 가장자리에서 잘린 차량을 구제
        """
        for tid, hist in self.track_histories.items():
            if hist.direction is None and len(hist.centroids) >= 3:
                direction = self.zone_detector.classify_direction(hist)
                if direction:
                    hist.direction = direction

    def _build_result_dict(self) -> Dict:
        """결과를 직렬화 가능한 딕셔너리로 변환"""
        cr = self.counting_result
        ss = self.speed_stats or {}

        result = {
            "session_id": self.session_id,
            "duration_sec": round(cr.duration_sec, 1) if cr else 0,
            "total_frames": self.total_frames,
            "fps": round(self.actual_fps, 1),
            "frame_size": f"{self.frame_width}x{self.frame_height}",
            "vehicles": {
                "total": cr.total_vehicles if cr else 0,
                UPBOUND_LABEL: {
                    "count": cr.upbound.total if cr else 0,
                    "large": cr.upbound.large if cr else 0,
                    "large_ratio": round(cr.upbound.large_ratio * 100, 1) if cr else 0,
                    "by_class": cr.upbound.by_class if cr else {},
                },
                DOWNBOUND_LABEL: {
                    "count": cr.downbound.total if cr else 0,
                    "large": cr.downbound.large if cr else 0,
                    "large_ratio": round(cr.downbound.large_ratio * 100, 1) if cr else 0,
                    "by_class": cr.downbound.by_class if cr else {},
                },
            },
            "speed_kmh": {
                UPBOUND_LABEL: ss.get(UPBOUND_LABEL, {}),
                DOWNBOUND_LABEL: ss.get(DOWNBOUND_LABEL, {}),
            },
            "tracked_vehicle_ids": len(self.track_histories),
        }
        return result

    def print_summary(self):
        """콘솔에 분석 결과 요약 출력"""
        r = self._build_result_dict()
        v = r["vehicles"]
        s = r["speed_kmh"]

        print(f"\n{'='*55}")
        print(f" 세션 {r['session_id']} 분석 결과")
        print(f"{'='*55}")
        print(f" 분석 시간: {r['duration_sec']}초  |  프레임: {r['total_frames']}")
        print(f" 추적 차량 수: {r['tracked_vehicle_ids']}대 (고유 ID 기준)")
        print(f"{'-'*55}")

        for dir_label in [UPBOUND_LABEL, DOWNBOUND_LABEL]:
            vc = v[dir_label]
            sc = s.get(dir_label, {})
            print(f" {dir_label}:")
            print(f"   차량 수: {vc['count']}대")
            print(f"   대형차: {vc['large']}대 ({vc['large_ratio']:.1f}%)")
            if vc.get("by_class"):
                cls_str = ", ".join(
                    f"{k}:{n}" for k, n in vc["by_class"].items()
                )
                print(f"   차종별: {cls_str}")
            if sc.get("count", 0) > 0:
                print(
                    f"   평균속도: {sc['mean']:.1f} km/h "
                    f"(중앙값 {sc['median']:.1f}, "
                    f"범위 {sc['min']:.0f}~{sc['max']:.0f})"
                )
            else:
                print("   속도 정보 없음 (추적 데이터 부족)")

        print(f"{'='*55}\n")
