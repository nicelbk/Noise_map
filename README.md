# 서대전나들목삼거리 CCTV 교통량 분석 시스템

대전광역시 교통정보시스템 실시간 CCTV를 활용한 **YOLOv8 + ByteTrack** 기반 교통량 자동 분석 도구

## 분석 내용

- **방향별 차량 계수**: 상행선 / 하행선 30초당 통과 차량 수
- **대형차 비율**: 버스 + 트럭 비율 (%)
- **차종별 분류**: 승용차 / 오토바이 / 버스 / 트럭
- **평균 속도 추정**: 원근 변환 + 트랙킹으로 km/h 역추적
- **10회 반복 평균**: 30초 세션을 10회 반복하여 통계적으로 안정된 결과 도출

## 시스템 구성

```
Noise_map/
├── main.py                    # 메인 실행 진입점
├── requirements.txt           # 의존성 패키지
└── src/
    ├── config.py              # 전체 설정 (CCTV URL, YOLO 파라미터 등)
    ├── cctv_capture.py        # Selenium 기반 CCTV 스트림 캡처
    ├── yolo_tracker.py        # YOLOv8 + ByteTrack 탐지/추적
    ├── traffic_counter.py     # 방향별 가상 계수선 카운팅
    ├── speed_estimator.py     # 원근 변환 기반 속도 추정
    ├── traffic_analyzer.py    # 세션 오케스트레이터
    └── calibration_tool.py    # 원근 변환 캘리브레이션 GUI
```

## 분석 흐름

```
[Selenium] 사이트 접속 + 서대전나들목삼거리 클릭
      ↓
[스트림 추출] HLS(m3u8) / RTSP URL 자동 감지
      ↓
[영상 캡처] 30초 프레임 수집 (OpenCV / ffmpeg)
      ↓
[YOLOv8] 차량 탐지 (승용차/버스/트럭/오토바이)
      ↓
[ByteTrack] 프레임 간 ID 유지 추적
      ↓
[계수선] 가상선 교차로 상행/하행 판별 + 계수
      ↓
[속도 추정] 원근 변환 → 픽셀 이동 → km/h
      ↓
      × 10회 반복
      ↓
[평균화] 통계 집계 + CSV/JSON 리포트 저장
```

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# YOLOv8 모델 자동 다운로드 (첫 실행 시)
# yolov8s.pt (~22MB) 자동 다운로드됨

# Chrome/Chromium 설치 필요 (CCTV 스트림 접속용)
# Ubuntu: sudo apt install chromium-browser
# or: https://www.google.com/chrome/
```

## 실행

```bash
# 기본 실행 (실제 CCTV, 10회 분석)
python main.py

# 반복 횟수 지정
python main.py --iterations 5

# 로컬 비디오 파일로 테스트
python main.py --test sample_traffic.mp4

# 실시간 시각화 (X11/디스플레이 필요)
python main.py --debug

# 분석 영상 저장 안 함
python main.py --no-save

# 출력 디렉토리 지정
python main.py --output-dir ./results
```

## 캘리브레이션 (속도 정확도 향상)

원근 변환 기준점을 CCTV 영상에 맞게 조정하면 속도 추정 정확도가 향상됩니다.

```bash
# 비디오로 캘리브레이션
python src/calibration_tool.py --video output/raw_session_01.mp4

# 이미지로 캘리브레이션
python src/calibration_tool.py --image frame.jpg
```

조작:
- 마우스 클릭: 도로 모서리 4점 선택 (상단좌→상단우→하단우→하단좌)
- `s`: 좌표 출력 (config.py에 복사)
- `r`: 초기화
- `q`: 종료

## 설정 조정 (`src/config.py`)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `YOLO_MODEL` | `yolov8s.pt` | n(빠름)~x(정확) |
| `CAPTURE_DURATION_SEC` | `30` | 1회 캡처 시간 |
| `NUM_ITERATIONS` | `10` | 반복 횟수 |
| `LANE_SPLIT_AXIS` | `"horizontal"` | 방향 분리 기준 |
| `ROAD_WIDTH_METERS` | `14.0` | 편도 2차로 폭 |
| `ROAD_SEGMENT_METERS` | `30.0` | 영상 내 도로 구간 |

## 출력 결과

```
output/
├── analysis.log               # 분석 로그
├── traffic_summary.json       # 전체 평균 결과 (JSON)
├── traffic_report.csv         # 세션별 상세 데이터 (CSV)
├── analyzed_session_01.mp4    # 분석 영상 (바운딩박스+추적선)
└── ...
```

## 알고리즘 세부사항

### 방향 판별

- **계수선 방식**: 화면에 가상의 수직선 2개를 그어 차량이 통과할 때 이동 방향 벡터로 상/하행 판별
- **구역 방식 (보완)**: 트랙 전체 이동 경로의 평균 방향으로 보완 분류

### 속도 추정

- 연속 프레임에서 동일 ID 차량의 중심점 이동 거리 계산
- Homography(원근 변환)로 픽셀 → 실제 미터 변환
- `속도(km/h) = 이동거리(m) / 시간(s) × 3.6`
- 이동 평균 필터(5프레임)로 노이즈 제거
- 비현실적 값(0~200 km/h 외) 자동 제거

### 차량 분류

| YOLO 클래스 | 분류 |
|-------------|------|
| car (2) | 소형차 |
| motorcycle (3) | 소형차 |
| bus (5) | **대형차** |
| truck (7) | **대형차** |

## 한계 및 주의사항

- CCTV 스트림 URL은 사이트 업데이트 시 추출 방식이 변경될 수 있음
- 속도 추정은 도로 실제 치수(`ROAD_SEGMENT_METERS`) 설정의 정확도에 의존
- 야간/악천후/카메라 각도에 따라 탐지 정확도 변동
- 화면에 부분적으로 잘린 차량은 계수에서 누락될 수 있음
