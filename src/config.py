"""
Configuration for 대전 교통 CCTV 분석 시스템
서대전나들목삼거리 CCTV 차량 분석 설정
"""

# ─── CCTV 웹사이트 설정 ───────────────────────────────────────
CCTV_BASE_URL = "https://traffic.daejeon.go.kr/traffic/cctv"
TARGET_LOCATION = "서대전나들목삼거리"

# ─── 분석 파라미터 ───────────────────────────────────────────
CAPTURE_DURATION_SEC = 30       # 1회 캡처 시간 (초)
NUM_ITERATIONS = 10             # 반복 횟수 (평균화용)
ITERATION_INTERVAL_SEC = 5      # 반복 간 대기 시간 (초)

# ─── YOLO 모델 설정 ──────────────────────────────────────────
# yolov8n.pt(빠름/정확도낮음) ~ yolov8x.pt(느림/정확도높음)
YOLO_MODEL = "yolov8n.pt"   # nano(빠름). 더 정확하려면 yolov8s.pt or yolov8m.pt
YOLO_CONF_THRESHOLD = 0.45      # 탐지 신뢰도 임계값
YOLO_IOU_THRESHOLD = 0.45       # NMS IOU 임계값
YOLO_IMG_SIZE = 640

# ─── COCO 차량 클래스 ID ─────────────────────────────────────
# COCO dataset class IDs
VEHICLE_CLASS_IDS = {
    2: "car",           # 승용차
    3: "motorcycle",    # 오토바이
    5: "bus",           # 버스
    7: "truck",         # 트럭
}
LARGE_VEHICLE_CLASS_IDS = {5, 7}   # 대형차 (버스 + 트럭)

# ─── 방향 감지 (가상 계수선) ──────────────────────────────────
# 화면을 두 구역으로 분리하여 상행/하행 판별
# 값은 0.0~1.0 (비율), 실제 픽셀은 영상 크기에 곱해서 사용
# 카메라 방향에 따라 조정 필요

# 계수선 설정: 두 개의 수평선으로 차량 이동 방향 판별
COUNTING_LINE_RATIO = 0.5        # 화면 중앙 계수선 (상대 y 위치)
DIRECTION_SEPARATION_RATIO = 0.5 # 좌우 분리선 (상대 x 위치, 수직 카메라용)

# 방향 판별 기준
# "horizontal": 화면 좌우가 상/하행 (카메라가 도로 측면을 바라봄)
# "vertical":   화면 위아래가 상/하행 (카메라가 도로를 내려다봄)
LANE_SPLIT_AXIS = "horizontal"  # or "vertical"

# 상행선: 화면 기준 좌측 (x < 중앙) 또는 위쪽 (y < 중앙)
# 하행선: 화면 기준 우측 (x > 중앙) 또는 아래쪽 (y > 중앙)
UPBOUND_LABEL = "상행선"
DOWNBOUND_LABEL = "하행선"

# ─── 속도 추정 설정 ──────────────────────────────────────────
VIDEO_FPS = 15                   # CCTV 영상 FPS (실제 확인 후 업데이트)

# 원근 보정 기준점 (픽셀 → 실제 거리 변환)
# 실제 도로 폭/길이를 알면 더 정확한 계산 가능
# 기본값: 전형적인 왕복 4차로 도로 기준
ROAD_WIDTH_METERS = 14.0         # 편도 2차로 기준 (3.5m × 2차로 × 2)
ROAD_SEGMENT_METERS = 30.0       # 영상에 보이는 도로 구간 길이(m) 추정값

# 속도 평활화 (급격한 변동 제거)
SPEED_SMOOTHING_WINDOW = 5       # 이동 평균 프레임 수

# 원근 변환 행렬용 기준점 (영상 비율, 0.0~1.0)
# 도로가 사다리꼴 형태로 보이는 경우 설정
# [상단좌, 상단우, 하단우, 하단좌] 순서
PERSPECTIVE_SRC_POINTS = [
    [0.3, 0.2],   # 상단 좌측 (도로 소실점 근처)
    [0.7, 0.2],   # 상단 우측
    [1.0, 0.9],   # 하단 우측
    [0.0, 0.9],   # 하단 좌측
]
# 변환 후 직사각형 좌표 (버드아이뷰)
PERSPECTIVE_DST_POINTS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
]

# ─── 추적기 설정 (ByteTrack) ─────────────────────────────────
TRACKER_TYPE = "bytetrack"       # ultralytics에 내장된 ByteTrack
TRACK_MAX_AGE = 30               # 트랙 유지 최대 프레임 수
TRACK_MIN_HITS = 3               # 트랙 확정 최소 탐지 횟수
TRACK_IOU_THRESH = 0.3

# ─── 출력/저장 설정 ──────────────────────────────────────────
OUTPUT_DIR = "output"
SAVE_ANNOTATED_VIDEO = True      # 분석된 영상 저장 여부
SAVE_FRAME_SNAPSHOTS = True      # 대표 프레임 저장 여부
REPORT_FILENAME = "traffic_report.csv"
SUMMARY_FILENAME = "traffic_summary.json"

# ─── 셀레니움 설정 ──────────────────────────────────────────
SELENIUM_TIMEOUT = 30            # 페이지 로딩 대기 (초)
SELENIUM_HEADLESS = False        # True: 백그라운드 실행, False: 브라우저 표시
CHROME_WINDOW_SIZE = "1280,720"

# 스트림 감지 대기 시간
STREAM_DETECT_WAIT_SEC = 8       # 클릭 후 스트림 URL 감지 대기

# ─── 디버그 설정 ─────────────────────────────────────────────
DEBUG_MODE = False               # True: 중간 결과 시각화
VERBOSE = True                   # 상세 로그 출력
SHOW_TRACKING_ID = True          # 영상에 추적 ID 표시
SHOW_SPEED = True                # 영상에 속도 표시
