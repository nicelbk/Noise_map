"""
서대전나들목삼거리 CCTV 교통량 분석 시스템
==============================================

실행 방법:
  python main.py                     # 기본 (CCTV 자동 캡처 + 10회 분석)
  python main.py --test <video.mp4>  # 로컬 영상 파일로 테스트
  python main.py --iterations 5      # 반복 횟수 조정
  python main.py --no-save           # 분석 영상 미저장
  python main.py --debug             # 실시간 시각화

분석 흐름:
  1. Selenium으로 CCTV 사이트 접속 → 서대전나들목삼거리 클릭
  2. HLS/RTSP 스트림 URL 자동 추출
  3. 30초 영상 캡처 → YOLO + ByteTrack 분석
  4. 방향별 차량 계수 + 속도 추정
  5. 위 과정을 10회 반복
  6. 10회 평균 결과 도출 및 리포트 저장
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    CAPTURE_DURATION_SEC,
    NUM_ITERATIONS,
    ITERATION_INTERVAL_SEC,
    OUTPUT_DIR,
    REPORT_FILENAME,
    SUMMARY_FILENAME,
    VIDEO_FPS,
    UPBOUND_LABEL,
    DOWNBOUND_LABEL,
    SAVE_ANNOTATED_VIDEO,
    TARGET_LOCATION,
)
from cctv_capture import CCTVCapture, load_frames_from_file
from traffic_analyzer import SessionAnalyzer

# ─── 로깅 설정 ──────────────────────────────────────────────────

def setup_logging(verbose: bool = True):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                Path(OUTPUT_DIR) / "analysis.log", encoding="utf-8"
            ),
        ],
    )

logger = logging.getLogger(__name__)


# ─── 평균 결과 계산 ──────────────────────────────────────────────

def average_results(results: List[Dict]) -> Dict:
    """
    여러 세션 결과의 평균 계산
    """
    if not results:
        return {}

    def avg_field(field_path: List[str]) -> float:
        """중첩 딕셔너리에서 필드 값을 추출하여 평균"""
        values = []
        for r in results:
            val = r
            try:
                for key in field_path:
                    val = val[key]
                if isinstance(val, (int, float)) and val > 0:
                    values.append(float(val))
            except (KeyError, TypeError):
                continue
        return round(float(np.mean(values)), 2) if values else 0.0

    def avg_count(direction: str) -> int:
        return int(round(avg_field(["vehicles", direction, "count"])))

    def avg_large_ratio(direction: str) -> float:
        return avg_field(["vehicles", direction, "large_ratio"])

    def avg_speed(direction: str, metric: str) -> float:
        return avg_field(["speed_kmh", direction, metric])

    summary = {
        "location": TARGET_LOCATION,
        "analysis_datetime": datetime.now().isoformat(),
        "num_sessions": len(results),
        "capture_duration_sec": CAPTURE_DURATION_SEC,
        "average": {
            "vehicles": {
                "total": int(round(avg_field(["vehicles", "total"]))),
                UPBOUND_LABEL: {
                    "count_per_30s": avg_count(UPBOUND_LABEL),
                    "large_ratio_pct": avg_large_ratio(UPBOUND_LABEL),
                    "speed_mean_kmh": avg_speed(UPBOUND_LABEL, "mean"),
                    "speed_median_kmh": avg_speed(UPBOUND_LABEL, "median"),
                },
                DOWNBOUND_LABEL: {
                    "count_per_30s": avg_count(DOWNBOUND_LABEL),
                    "large_ratio_pct": avg_large_ratio(DOWNBOUND_LABEL),
                    "speed_mean_kmh": avg_speed(DOWNBOUND_LABEL, "mean"),
                    "speed_median_kmh": avg_speed(DOWNBOUND_LABEL, "median"),
                },
            },
        },
        "sessions": results,
    }
    return summary


def print_final_summary(summary: Dict):
    """최종 평균 결과 콘솔 출력"""
    avg = summary.get("average", {})
    v = avg.get("vehicles", {})

    print("\n" + "=" * 60)
    print(f"  {TARGET_LOCATION} 교통량 분석 최종 결과")
    print(f"  ({summary['num_sessions']}회 × {summary['capture_duration_sec']}초 평균)")
    print("=" * 60)

    for dir_label in [UPBOUND_LABEL, DOWNBOUND_LABEL]:
        dc = v.get(dir_label, {})
        print(f"\n  [{dir_label}]")
        print(f"    30초당 차량 수  : {dc.get('count_per_30s', 0):4d} 대")
        print(f"    시간당 차량 수  : {dc.get('count_per_30s', 0)*120:5d} 대 (추정)")
        print(f"    대형차 비율    : {dc.get('large_ratio_pct', 0):5.1f} %")
        spd = dc.get('speed_mean_kmh', 0)
        spd_med = dc.get('speed_median_kmh', 0)
        if spd > 0:
            print(f"    평균 속도      : {spd:5.1f} km/h (중앙값 {spd_med:.1f})")
        else:
            print(f"    평균 속도      : 데이터 부족")

    print(f"\n  총 차량 (30초평균): {v.get('total', {})}")
    print("=" * 60 + "\n")


def save_report(summary: Dict, output_dir: str):
    """분석 결과를 JSON 및 CSV로 저장"""
    os.makedirs(output_dir, exist_ok=True)

    # JSON 저장
    json_path = Path(output_dir) / SUMMARY_FILENAME
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 리포트 저장: {json_path}")

    # CSV 저장 (세션별 상세 데이터)
    rows = []
    for r in summary.get("sessions", []):
        row = {"session_id": r["session_id"], "duration_sec": r["duration_sec"]}
        for dir_label in [UPBOUND_LABEL, DOWNBOUND_LABEL]:
            vc = r.get("vehicles", {}).get(dir_label, {})
            sc = r.get("speed_kmh", {}).get(dir_label, {})
            row[f"{dir_label}_count"] = vc.get("count", 0)
            row[f"{dir_label}_large"] = vc.get("large", 0)
            row[f"{dir_label}_large_ratio"] = vc.get("large_ratio", 0)
            row[f"{dir_label}_speed_mean"] = sc.get("mean", 0)
            row[f"{dir_label}_speed_median"] = sc.get("median", 0)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = Path(output_dir) / REPORT_FILENAME
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"CSV 리포트 저장: {csv_path}")

        # 통계 요약 출력
        print("\n[세션별 데이터]")
        print(df.to_string(index=False))


# ─── 메인 분석 루프 ──────────────────────────────────────────────

def run_cctv_analysis(args):
    """
    실제 CCTV 스트림 캡처 + 분석 메인 루프
    """
    results = []
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    with CCTVCapture() as capture:
        # 초기 세션 시작 (브라우저 열기 + 스트림 추출)
        logger.info("CCTV 캡처 세션 시작...")
        if not capture.start_session():
            logger.error(
                "CCTV 접속 실패. 브라우저와 ChromeDriver가 설치되어 있는지 확인하세요."
            )
            logger.info("테스트: python main.py --test <video.mp4>")
            return None

        for i in range(args.iterations):
            logger.info(f"\n{'─'*40}")
            logger.info(f"  반복 {i+1}/{args.iterations} 시작")
            logger.info(f"{'─'*40}")

            # 2회 이후: 재클릭으로 새 30초 세그먼트 시작
            if i > 0:
                time.sleep(ITERATION_INTERVAL_SEC)
                capture.click_for_next_iteration()

            # 30초 프레임 캡처
            video_path = None
            if SAVE_ANNOTATED_VIDEO or args.save_raw:
                video_path = str(output_dir / f"raw_session_{i+1:02d}.mp4")

            frames = capture.capture_frames(
                duration=CAPTURE_DURATION_SEC,
                output_path=video_path if args.save_raw else None,
            )

            if not frames:
                logger.warning(f"세션 {i+1}: 프레임 없음, 건너뜀")
                continue

            # 분석 실행
            annotated_path = None
            if SAVE_ANNOTATED_VIDEO:
                annotated_path = str(
                    output_dir / f"analyzed_session_{i+1:02d}.mp4"
                )

            analyzer = SessionAnalyzer(session_id=i + 1)
            result = analyzer.analyze(
                frames,
                fps=capture.actual_fps,
                output_video_path=annotated_path,
            )
            analyzer.print_summary()
            results.append(result)

            logger.info(f"세션 {i+1} 완료")

    return results


def run_file_analysis(args):
    """
    로컬 비디오 파일로 테스트 분석
    """
    video_path = args.test
    if not os.path.exists(video_path):
        logger.error(f"파일 없음: {video_path}")
        return None

    logger.info(f"파일 분석 모드: {video_path}")
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    frames, fps = load_frames_from_file(
        video_path, duration=CAPTURE_DURATION_SEC
    )

    if not frames:
        logger.error("프레임 로드 실패")
        return None

    results = []
    iterations = min(args.iterations, 3)  # 파일 모드는 최대 3회 (동일 파일 반복)

    for i in range(iterations):
        logger.info(f"\n반복 {i+1}/{iterations}")
        annotated_path = str(output_dir / f"analyzed_session_{i+1:02d}.mp4")

        analyzer = SessionAnalyzer(session_id=i + 1)
        result = analyzer.analyze(
            frames,
            fps=fps,
            output_video_path=annotated_path if SAVE_ANNOTATED_VIDEO else None,
        )
        analyzer.print_summary()
        results.append(result)

    return results


# ─── 진입점 ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="서대전나들목삼거리 CCTV 교통량 YOLO 분석 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test", metavar="VIDEO_PATH",
        help="로컬 비디오 파일로 테스트 (실제 CCTV 대신)",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=NUM_ITERATIONS,
        help=f"분석 반복 횟수 (기본값: {NUM_ITERATIONS})",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="분석 영상 저장 안 함",
    )
    parser.add_argument(
        "--save-raw", action="store_true",
        help="원본 캡처 영상도 저장",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="실시간 시각화 (X11 디스플레이 필요)",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"출력 디렉토리 (기본값: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model", default=None,
        help="YOLO 모델 경로 (기본값: config.py 설정)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 설정 덮어쓰기
    if args.no_save:
        import src.config as cfg
        cfg.SAVE_ANNOTATED_VIDEO = False
    if args.debug:
        import src.config as cfg
        cfg.DEBUG_MODE = True
    if args.model:
        import src.config as cfg
        cfg.YOLO_MODEL = args.model

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging()

    logger.info(f"{'='*55}")
    logger.info(f" 서대전나들목삼거리 CCTV 교통량 분석 시스템")
    logger.info(f"{'='*55}")
    logger.info(f" 대상: {TARGET_LOCATION}")
    logger.info(f" 반복: {args.iterations}회 × {CAPTURE_DURATION_SEC}초")
    logger.info(f" 출력: {OUTPUT_DIR}/")
    logger.info(f"{'='*55}\n")

    # 분석 실행
    if args.test:
        results = run_file_analysis(args)
    else:
        results = run_cctv_analysis(args)

    if not results:
        logger.error("분석 결과 없음")
        sys.exit(1)

    # 평균 결과 계산 및 출력
    summary = average_results(results)
    print_final_summary(summary)
    save_report(summary, OUTPUT_DIR)

    logger.info("분석 완료!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
