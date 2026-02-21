"""
테스트 실행 스크립트
합성 교통 영상으로 분석 파이프라인 전체 검증

사용: python run_test.py [--iterations N]
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

import src.config as cfg
# 테스트용 설정 재정의
cfg.SAVE_ANNOTATED_VIDEO = True
cfg.DEBUG_MODE = False
cfg.VERBOSE = True

from src.test_video_generator import make_test_video
from src.cctv_capture import load_frames_from_file
from src.traffic_analyzer import SessionAnalyzer
from main import average_results, print_final_summary, save_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. 합성 영상 생성 ─────────────────────────────────────────
iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 3

print("\n" + "=" * 60)
print("  서대전나들목삼거리 CCTV 분석 파이프라인 테스트")
print(f"  (합성 영상 {iterations}회 반복 분석)")
print("=" * 60)

video_files = []
for i in range(iterations):
    vpath = f"{OUTPUT_DIR}/test_input_{i+1:02d}.mp4"
    # 매번 다른 시드로 약간씩 다른 교통 패턴 생성
    make_test_video(vpath, duration_sec=30, fps=15, seed=42 + i * 7)
    video_files.append(vpath)

# ─── 2. 각 세션 분석 ─────────────────────────────────────────
results = []

for i, vpath in enumerate(video_files):
    print(f"\n{'─'*50}")
    print(f" 세션 {i+1}/{iterations}: {vpath}")
    print(f"{'─'*50}")

    frames, fps = load_frames_from_file(vpath, duration=30)
    logger.info(f"프레임 로드: {len(frames)}개, FPS: {fps:.1f}")

    annotated_path = f"{OUTPUT_DIR}/analyzed_{i+1:02d}.mp4"

    analyzer = SessionAnalyzer(session_id=i + 1)
    result = analyzer.analyze(
        frames,
        fps=fps,
        output_video_path=annotated_path,
    )
    analyzer.print_summary()
    results.append(result)

# ─── 3. 평균화 + 최종 리포트 ─────────────────────────────────
summary = average_results(results)
print_final_summary(summary)
save_report(summary, OUTPUT_DIR)

# ─── 4. 결과 파일 확인 ───────────────────────────────────────
print("\n[생성된 파일]")
for f in sorted(Path(OUTPUT_DIR).iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:40s} {size_kb:8.1f} KB")

print("\n테스트 완료!")
