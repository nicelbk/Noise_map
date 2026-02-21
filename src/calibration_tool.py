"""
원근 변환 캘리브레이션 도구 (Interactive)

사용법:
  python src/calibration_tool.py --video <video.mp4>
  python src/calibration_tool.py --image <frame.jpg>

조작:
  - 마우스 클릭: 원근 변환 기준점 4개 선택
    순서: 상단좌 → 상단우 → 하단우 → 하단좌
  - 's': 결과를 config.py에 반영할 수 있는 좌표 출력
  - 'r': 초기화
  - 'q': 종료

기준점 선택 요령:
  - 도로 차선이 직선으로 보이는 4개 모서리 선택
  - 상단: 원근으로 수렴하는 지점 (소실점 방향)
  - 하단: 카메라에 가까운 도로 부분
  - 실제 거리를 알면 정확한 속도 계산 가능
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PERSPECTIVE_SRC_POINTS,
    ROAD_WIDTH_METERS,
    ROAD_SEGMENT_METERS,
)


class CalibrationTool:
    def __init__(self, frame: np.ndarray):
        self.original = frame.copy()
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.points = []
        self.max_points = 4
        self.labels = ["상단좌", "상단우", "하단우", "하단좌"]
        self.colors = [
            (0, 255, 0),
            (0, 255, 255),
            (0, 165, 255),
            (255, 0, 0),
        ]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < self.max_points:
                self.points.append((x, y))
                self._redraw()

    def _redraw(self):
        self.frame = self.original.copy()

        # 기준점 그리기
        for i, pt in enumerate(self.points):
            cv2.circle(self.frame, pt, 6, self.colors[i], -1)
            cv2.circle(self.frame, pt, 8, (255, 255, 255), 1)
            cv2.putText(
                self.frame, self.labels[i],
                (pt[0] + 10, pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                self.colors[i], 2, cv2.LINE_AA,
            )

        # 4점 모두 선택 시: 사다리꼴 그리기 + BEV 미리보기
        if len(self.points) == 4:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(
                self.frame, [pts], True, (0, 255, 255), 2, cv2.LINE_AA
            )
            self._draw_bev_preview()

        # 안내 텍스트
        remaining = self.max_points - len(self.points)
        if remaining > 0:
            next_label = self.labels[len(self.points)]
            msg = f"클릭: {next_label} ({remaining}개 남음)"
        else:
            msg = "완료! 's'=저장  'r'=초기화  'q'=종료"
        cv2.putText(
            self.frame, msg, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            self.frame,
            f"도로폭: {ROAD_WIDTH_METERS}m  구간: {ROAD_SEGMENT_METERS}m",
            (10, self.h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

    def _draw_bev_preview(self):
        """버드아이뷰 변환 미리보기 (우측 상단에 소형 표시)"""
        if len(self.points) != 4:
            return

        src = np.array(self.points, dtype=np.float32)
        bev_w, bev_h = 200, 300
        margin = 20
        dst = np.array([
            [margin, margin],
            [bev_w - margin, margin],
            [bev_w - margin, bev_h - margin],
            [margin, bev_h - margin],
        ], dtype=np.float32)

        try:
            M = cv2.getPerspectiveTransform(src, dst)
            bev = cv2.warpPerspective(self.original, M, (bev_w, bev_h))

            # 우측 상단에 배치
            x_off = self.w - bev_w - 10
            y_off = 10
            # 반투명 배경
            overlay = self.frame.copy()
            cv2.rectangle(
                overlay,
                (x_off - 5, y_off - 5),
                (x_off + bev_w + 5, y_off + bev_h + 5),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.5, self.frame, 0.5, 0, self.frame)
            self.frame[y_off:y_off+bev_h, x_off:x_off+bev_w] = bev
            cv2.putText(
                self.frame, "BEV Preview",
                (x_off, y_off + bev_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )
        except cv2.error:
            pass

    def print_config(self):
        """config.py에 넣을 수 있는 형식으로 좌표 출력"""
        if len(self.points) != 4:
            print("기준점 4개를 먼저 선택하세요")
            return

        print("\n" + "=" * 50)
        print("config.py 에 아래 값을 복사하세요:")
        print("=" * 50)
        print("PERSPECTIVE_SRC_POINTS = [")
        for pt in self.points:
            rx = round(pt[0] / self.w, 3)
            ry = round(pt[1] / self.h, 3)
            print(f"    [{rx}, {ry}],   # ({pt[0]}px, {pt[1]}px)")
        print("]")
        print()

        # 실제 도로 거리 (픽셀 기준)
        bottom_left = np.array(self.points[3])
        bottom_right = np.array(self.points[2])
        road_width_px = float(np.linalg.norm(bottom_right - bottom_left))
        px_per_m = road_width_px / ROAD_WIDTH_METERS
        print(f"# 하단 도로 폭: {road_width_px:.0f}px = {ROAD_WIDTH_METERS}m")
        print(f"# 1m ≈ {px_per_m:.1f}px (하단 기준)")
        print("=" * 50)

    def run(self, window_name: str = "Calibration Tool"):
        cv2.namedWindow(window_name, cv2.WINDOW_RESIZABLE)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        self._redraw()

        print("\n[캘리브레이션 도구]")
        print("  마우스 클릭: 기준점 선택 (상단좌→상단우→하단우→하단좌)")
        print("  's': 좌표 출력  |  'r': 초기화  |  'q': 종료\n")

        while True:
            cv2.imshow(window_name, self.frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                self.points = []
                self._redraw()
            elif key == ord("s"):
                self.print_config()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="원근 변환 캘리브레이션 도구")
    parser.add_argument("--video", help="비디오 파일 경로")
    parser.add_argument("--image", help="이미지 파일 경로")
    parser.add_argument(
        "--frame", type=int, default=30,
        help="비디오에서 추출할 프레임 번호 (기본: 30)",
    )
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"이미지 로드 실패: {args.image}")
            sys.exit(1)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"비디오 프레임 추출 실패: {args.video}")
            sys.exit(1)
    else:
        # 기본값: 현재 설정값으로 빈 캔버스 표시
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "--video 또는 --image 옵션으로 영상 지정 필요",
            (100, 360),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        )

    tool = CalibrationTool(frame)
    tool.run()


if __name__ == "__main__":
    main()
