"""
CCTV 스트림 캡처 모듈
대전 교통정보 시스템에서 서대전나들목삼거리 CCTV 스트림을 추출하고 캡처

접근 전략:
  1순위: Selenium 네트워크 인터셉트로 HLS(m3u8) 스트림 URL 추출
  2순위: <video> 태그 src 직접 파싱
  3순위: mss 화면 캡처 (Fallback)
"""

import time
import re
import os
import subprocess
import tempfile
import threading
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import requests

try:
    import undetected_chromedriver as uc
    UC_AVAILABLE = True
except ImportError:
    UC_AVAILABLE = False

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

from config import (
    CCTV_BASE_URL, TARGET_LOCATION, CAPTURE_DURATION_SEC,
    SELENIUM_TIMEOUT, SELENIUM_HEADLESS, CHROME_WINDOW_SIZE,
    STREAM_DETECT_WAIT_SEC, VIDEO_FPS, OUTPUT_DIR, VERBOSE
)

logger = logging.getLogger(__name__)


class StreamURLInterceptor:
    """Selenium 네트워크 요청을 모니터링하여 HLS 스트림 URL 추출"""

    def __init__(self):
        self.stream_urls: List[str] = []
        self._lock = threading.Lock()

    def add_url(self, url: str):
        with self._lock:
            if url not in self.stream_urls:
                self.stream_urls.append(url)
                logger.debug(f"[StreamURL 감지] {url}")

    def get_best_url(self) -> Optional[str]:
        """m3u8 > mp4 > 기타 순으로 최적 URL 반환"""
        with self._lock:
            for url in self.stream_urls:
                if ".m3u8" in url:
                    return url
            for url in self.stream_urls:
                if any(ext in url for ext in [".mp4", ".ts", ".flv", ".rtmp"]):
                    return url
            return self.stream_urls[0] if self.stream_urls else None


class CCTVCapture:
    """
    대전 교통 CCTV 웹사이트에서 스트림을 추출하고 프레임을 캡처하는 클래스
    """

    def __init__(self):
        self.driver: Optional[webdriver.Chrome] = None
        self.interceptor = StreamURLInterceptor()
        self.stream_url: Optional[str] = None
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.actual_fps: float = VIDEO_FPS
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._frame_buffer: List[np.ndarray] = []
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_capture = threading.Event()

    # ─────────────────────────────────────────────────────────────
    # 드라이버 초기화
    # ─────────────────────────────────────────────────────────────

    def _build_chrome_options(self) -> Options:
        options = Options()
        if SELENIUM_HEADLESS:
            options.add_argument("--headless=new")
        options.add_argument(f"--window-size={CHROME_WINDOW_SIZE}")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--autoplay-policy=no-user-gesture-required")
        options.add_argument("--disable-blink-features=AutomationControlled")

        # 네트워크 로깅 활성화 (스트림 URL 감지용)
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option(
            "excludeSwitches", ["enable-automation"]
        )
        return options

    def _init_driver(self):
        """Chrome WebDriver 초기화"""
        logger.info("Chrome WebDriver 초기화 중...")
        options = self._build_chrome_options()

        if UC_AVAILABLE:
            # undetected-chromedriver 사용 (봇 탐지 우회)
            self.driver = uc.Chrome(options=options, version_main=None)
        else:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)

        self.driver.implicitly_wait(SELENIUM_TIMEOUT)
        logger.info("WebDriver 초기화 완료")

    # ─────────────────────────────────────────────────────────────
    # 페이지 탐색 및 메뉴 클릭
    # ─────────────────────────────────────────────────────────────

    def navigate_and_click(self) -> bool:
        """
        CCTV 페이지로 이동하여 '서대전나들목삼거리' 메뉴 클릭
        Returns: 성공 여부
        """
        logger.info(f"CCTV 페이지 접속 중: {CCTV_BASE_URL}")
        self.driver.get(CCTV_BASE_URL)
        time.sleep(3)  # 초기 렌더링 대기

        wait = WebDriverWait(self.driver, SELENIUM_TIMEOUT)

        # 전략 1: 텍스트로 직접 요소 찾기
        target_found = self._click_by_text(wait)
        if not target_found:
            # 전략 2: 링크 태그 탐색
            target_found = self._click_by_link(wait)
        if not target_found:
            # 전략 3: JavaScript로 텍스트 검색
            target_found = self._click_by_js()

        if target_found:
            logger.info(f"'{TARGET_LOCATION}' 클릭 성공")
            time.sleep(STREAM_DETECT_WAIT_SEC)  # 영상 로딩 대기
        else:
            logger.error(f"'{TARGET_LOCATION}' 메뉴 항목을 찾지 못했습니다.")

        return target_found

    def _click_by_text(self, wait: WebDriverWait) -> bool:
        """XPath 텍스트로 요소 클릭"""
        try:
            selectors = [
                f"//*[contains(text(), '{TARGET_LOCATION}')]",
                f"//li[contains(text(), '{TARGET_LOCATION}')]",
                f"//a[contains(text(), '{TARGET_LOCATION}')]",
                f"//span[contains(text(), '{TARGET_LOCATION}')]",
                f"//div[contains(text(), '{TARGET_LOCATION}')]",
            ]
            for xpath in selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, xpath)
                    if elements:
                        elements[0].click()
                        logger.debug(f"클릭 성공 (XPath): {xpath}")
                        return True
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"텍스트 클릭 실패: {e}")
        return False

    def _click_by_link(self, wait: WebDriverWait) -> bool:
        """링크 텍스트로 요소 클릭"""
        try:
            element = wait.until(
                EC.element_to_be_clickable(
                    (By.PARTIAL_LINK_TEXT, TARGET_LOCATION)
                )
            )
            element.click()
            return True
        except Exception as e:
            logger.debug(f"링크 클릭 실패: {e}")
            return False

    def _click_by_js(self) -> bool:
        """JavaScript로 텍스트 포함 요소 검색 후 클릭"""
        try:
            script = f"""
            var elements = document.querySelectorAll('*');
            for (var el of elements) {{
                if (el.childNodes.length === 1 &&
                    el.textContent.trim().includes('{TARGET_LOCATION}')) {{
                    el.click();
                    return true;
                }}
            }}
            return false;
            """
            result = self.driver.execute_script(script)
            return bool(result)
        except Exception as e:
            logger.debug(f"JS 클릭 실패: {e}")
            return False

    # ─────────────────────────────────────────────────────────────
    # 스트림 URL 추출
    # ─────────────────────────────────────────────────────────────

    def extract_stream_url(self) -> Optional[str]:
        """
        네트워크 로그 분석 + <video> 태그로 스트림 URL 추출
        """
        logger.info("스트림 URL 추출 시도 중...")

        # 방법 1: 성능 로그에서 네트워크 요청 분석
        url = self._extract_from_performance_logs()
        if url:
            self.stream_url = url
            logger.info(f"[성능로그] 스트림 URL 발견: {url}")
            return url

        # 방법 2: video 태그 src 직접 읽기
        url = self._extract_from_video_tag()
        if url:
            self.stream_url = url
            logger.info(f"[video태그] 스트림 URL 발견: {url}")
            return url

        # 방법 3: iframe 내부 탐색
        url = self._extract_from_iframe()
        if url:
            self.stream_url = url
            logger.info(f"[iframe] 스트림 URL 발견: {url}")
            return url

        # 방법 4: 페이지 소스에서 정규식으로 URL 파싱
        url = self._extract_from_page_source()
        if url:
            self.stream_url = url
            logger.info(f"[페이지소스] 스트림 URL 발견: {url}")
            return url

        logger.warning("스트림 URL 자동 추출 실패 → 화면 캡처 모드로 전환")
        return None

    def _extract_from_performance_logs(self) -> Optional[str]:
        """Chrome 성능 로그에서 미디어 요청 URL 추출"""
        try:
            import json
            logs = self.driver.get_log("performance")
            stream_patterns = [
                r'https?://[^\'">\s]+\.m3u8[^\'">\s]*',
                r'https?://[^\'">\s]+\.mp4[^\'">\s]*',
                r'rtmp://[^\'">\s]+',
                r'rtsp://[^\'">\s]+',
                r'https?://[^\'">\s]+/stream[^\'">\s]*',
                r'https?://[^\'">\s]+/live[^\'">\s]*',
                r'https?://[^\'">\s]+\.ts[^\'">\s]*',
            ]
            for entry in logs:
                try:
                    message = json.loads(entry["message"])["message"]
                    if message.get("method") in [
                        "Network.requestWillBeSent",
                        "Network.responseReceived",
                    ]:
                        url = (
                            message.get("params", {})
                            .get("request", {})
                            .get("url", "")
                            or message.get("params", {})
                            .get("response", {})
                            .get("url", "")
                        )
                        for pattern in stream_patterns:
                            match = re.search(pattern, url, re.IGNORECASE)
                            if match:
                                return match.group(0)
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"성능 로그 분석 실패: {e}")
        return None

    def _extract_from_video_tag(self) -> Optional[str]:
        """<video> 태그의 src 또는 <source> 태그에서 URL 추출"""
        try:
            scripts = [
                # video 태그 직접 접근
                "return document.querySelector('video') ? document.querySelector('video').src : null;",
                # source 태그
                "return document.querySelector('video source') ? document.querySelector('video source').src : null;",
                # currentSrc (실제 재생 중인 URL)
                "return document.querySelector('video') ? document.querySelector('video').currentSrc : null;",
            ]
            for script in scripts:
                result = self.driver.execute_script(script)
                if result and result.startswith("http"):
                    return result

            # 모든 video/source 태그 수집
            result = self.driver.execute_script("""
                var urls = [];
                document.querySelectorAll('video, video source').forEach(function(el) {
                    if (el.src) urls.push(el.src);
                    if (el.currentSrc) urls.push(el.currentSrc);
                });
                return urls;
            """)
            if result:
                for url in result:
                    if url and url.startswith("http"):
                        return url
        except Exception as e:
            logger.debug(f"video 태그 추출 실패: {e}")
        return None

    def _extract_from_iframe(self) -> Optional[str]:
        """iframe 내부에서 스트림 URL 탐색"""
        try:
            iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
            for iframe in iframes:
                try:
                    self.driver.switch_to.frame(iframe)
                    url = self._extract_from_video_tag()
                    self.driver.switch_to.default_content()
                    if url:
                        return url
                except Exception:
                    self.driver.switch_to.default_content()
                    continue
        except Exception as e:
            logger.debug(f"iframe 탐색 실패: {e}")
        return None

    def _extract_from_page_source(self) -> Optional[str]:
        """페이지 HTML 소스에서 정규식으로 미디어 URL 파싱"""
        try:
            source = self.driver.page_source
            patterns = [
                r'(?:src|url|href)\s*[=:]\s*[\'"]([^\'"]+\.m3u8[^\'"]*)[\'"]',
                r'(?:src|url|href)\s*[=:]\s*[\'"]([^\'"]+/stream[^\'"]*)[\'"]',
                r'(?:src|url|href)\s*[=:]\s*[\'"]([^\'"]+/live[^\'"]*)[\'"]',
                r'(?:src|url|href)\s*[=:]\s*[\'"]([^\'"]+\.mp4[^\'"]*)[\'"]',
                r'"(rtmp://[^"]+)"',
                r'"(rtsp://[^"]+)"',
                r"'(rtmp://[^']+)'",
                r"'(rtsp://[^']+)'",
            ]
            for pattern in patterns:
                matches = re.findall(pattern, source, re.IGNORECASE)
                if matches:
                    for m in matches:
                        if m.startswith(("http", "rtmp", "rtsp")):
                            return m
                        elif m.startswith("/"):
                            # 상대 경로 → 절대 경로
                            from urllib.parse import urljoin
                            return urljoin(CCTV_BASE_URL, m)
        except Exception as e:
            logger.debug(f"페이지 소스 파싱 실패: {e}")
        return None

    # ─────────────────────────────────────────────────────────────
    # 프레임 캡처
    # ─────────────────────────────────────────────────────────────

    def capture_frames(
        self,
        duration: int = CAPTURE_DURATION_SEC,
        output_path: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        스트림에서 지정 시간 동안 프레임 캡처
        Returns: 캡처된 numpy 프레임 리스트
        """
        if self.stream_url:
            return self._capture_from_stream(duration, output_path)
        else:
            logger.warning("스트림 URL 없음 → 화면 직접 캡처")
            return self._capture_from_screen(duration, output_path)

    def _capture_from_stream(
        self, duration: int, output_path: Optional[str]
    ) -> List[np.ndarray]:
        """OpenCV로 HLS/RTSP/MP4 스트림 캡처"""
        logger.info(f"스트림 캡처 시작: {self.stream_url}")

        # ffmpeg 옵션 (HLS 지연 최소화)
        cap = cv2.VideoCapture(self.stream_url)

        # OpenCV가 직접 열지 못하는 경우 ffmpeg 파이프 사용
        if not cap.isOpened():
            logger.info("OpenCV 직접 연결 실패 → ffmpeg 파이프 시도")
            return self._capture_via_ffmpeg(duration, output_path)

        self.actual_fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
        if self.actual_fps <= 0 or self.actual_fps > 60:
            self.actual_fps = VIDEO_FPS

        logger.info(f"FPS: {self.actual_fps:.1f}")

        frames = []
        start_time = time.time()
        frame_count = 0

        # 출력 비디오 설정
        out_writer = None
        if output_path:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(
                output_path, fourcc, self.actual_fps, (w, h)
            )

        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                logger.warning("프레임 읽기 실패, 재연결 시도...")
                time.sleep(0.1)
                continue

            frames.append(frame.copy())
            if out_writer:
                out_writer.write(frame)
            frame_count += 1

        cap.release()
        if out_writer:
            out_writer.release()

        elapsed = time.time() - start_time
        logger.info(
            f"캡처 완료: {frame_count}프레임 / {elapsed:.1f}초 "
            f"(실효 FPS: {frame_count/elapsed:.1f})"
        )
        return frames

    def _capture_via_ffmpeg(
        self, duration: int, output_path: Optional[str]
    ) -> List[np.ndarray]:
        """ffmpeg 파이프를 통한 프레임 추출"""
        logger.info("ffmpeg 파이프 캡처 모드")

        # ffmpeg로 raw 프레임 stdout 출력
        cmd = [
            "ffmpeg", "-i", self.stream_url,
            "-t", str(duration),
            "-vf", f"fps={VIDEO_FPS}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ]

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )

            # 영상 크기 먼저 파악 (ffprobe)
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                self.stream_url,
            ]
            try:
                probe_out = subprocess.check_output(
                    probe_cmd, stderr=subprocess.DEVNULL
                ).decode().strip()
                w, h = map(int, probe_out.split("x"))
            except Exception:
                w, h = 1280, 720  # 기본값

            frame_size = w * h * 3
            frames = []

            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                frames.append(frame.copy())

            proc.terminate()
            logger.info(f"ffmpeg 캡처 완료: {len(frames)}프레임")
            return frames

        except FileNotFoundError:
            logger.error("ffmpeg를 찾을 수 없습니다. pip install ffmpeg-python 또는 시스템에 ffmpeg 설치 필요")
            return []
        except Exception as e:
            logger.error(f"ffmpeg 파이프 캡처 오류: {e}")
            return []

    def _capture_from_screen(
        self, duration: int, output_path: Optional[str]
    ) -> List[np.ndarray]:
        """
        화면 직접 캡처 (스트림 URL 추출 실패 시 fallback)
        브라우저에서 재생 중인 영상 영역만 캡처
        """
        if not MSS_AVAILABLE:
            logger.error("mss 패키지가 없습니다. pip install mss")
            return []

        logger.info("화면 캡처 모드 (mss)")

        # 비디오 요소 위치 파악
        video_rect = self._get_video_element_rect()

        with mss.mss() as sct:
            monitor = {
                "top": video_rect["y"],
                "left": video_rect["x"],
                "width": video_rect["width"],
                "height": video_rect["height"],
            }

            frames = []
            start_time = time.time()
            frame_interval = 1.0 / VIDEO_FPS

            out_writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    VIDEO_FPS,
                    (video_rect["width"], video_rect["height"]),
                )

            while (time.time() - start_time) < duration:
                t0 = time.time()
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)[:, :, :3]  # BGRA → BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frames.append(frame.copy())
                if out_writer:
                    out_writer.write(frame)

                # FPS 조절
                elapsed = time.time() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if out_writer:
                out_writer.release()

        logger.info(f"화면 캡처 완료: {len(frames)}프레임")
        return frames

    def _get_video_element_rect(self) -> dict:
        """브라우저 내 video 요소의 화면 좌표 반환"""
        try:
            rect = self.driver.execute_script("""
                var el = document.querySelector('video');
                if (!el) return null;
                var r = el.getBoundingClientRect();
                return {x: r.left, y: r.top,
                        width: r.width, height: r.height};
            """)
            if rect and rect.get("width", 0) > 0:
                # 브라우저 위치 오프셋 추가
                browser_pos = self.driver.get_window_position()
                # 브라우저 내 컨텐츠 영역 보정 (대략적인 툴바 높이)
                toolbar_height = 80
                return {
                    "x": int(browser_pos["x"] + rect["x"]),
                    "y": int(browser_pos["y"] + toolbar_height + rect["y"]),
                    "width": int(rect["width"]),
                    "height": int(rect["height"]),
                }
        except Exception as e:
            logger.debug(f"video 요소 위치 파악 실패: {e}")

        # 기본값: 전체 화면 우측 절반
        return {"x": 640, "y": 0, "width": 640, "height": 720}

    # ─────────────────────────────────────────────────────────────
    # 세션 관리
    # ─────────────────────────────────────────────────────────────

    def start_session(self) -> bool:
        """전체 캡처 세션 시작 (드라이버 초기화 + 페이지 이동 + 스트림 추출)"""
        self._init_driver()
        if not self.navigate_and_click():
            return False
        self.extract_stream_url()
        return True

    def click_for_next_iteration(self) -> bool:
        """
        다음 30초 분석을 위해 메뉴 항목 재클릭 또는 재생 버튼 클릭
        일부 사이트는 클릭할 때마다 새 스트림 세그먼트가 시작됨
        """
        try:
            # 재생 버튼 클릭 시도
            play_clicked = self.driver.execute_script("""
                var video = document.querySelector('video');
                if (video) {
                    video.pause();
                    video.currentTime = 0;
                    video.play();
                    return true;
                }
                return false;
            """)
            if not play_clicked:
                # 메뉴 항목 재클릭
                self._click_by_text(WebDriverWait(self.driver, 10))
                time.sleep(STREAM_DETECT_WAIT_SEC)

            # 스트림 URL 갱신
            new_url = self.extract_stream_url()
            if new_url:
                self.stream_url = new_url
            return True
        except Exception as e:
            logger.error(f"반복 클릭 실패: {e}")
            return False

    def close(self):
        """세션 종료 및 리소스 해제"""
        if self.video_capture:
            self.video_capture.release()
        if self._ffmpeg_proc:
            self._ffmpeg_proc.terminate()
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
        logger.info("CCTV 캡처 세션 종료")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────────────────────────
# 로컬 비디오 파일 캡처 (테스트/오프라인 분석용)
# ─────────────────────────────────────────────────────────────────

def load_frames_from_file(
    video_path: str,
    duration: Optional[int] = None,
    fps_limit: Optional[int] = None,
) -> Tuple[List[np.ndarray], float]:
    """
    로컬 비디오 파일에서 프레임 로드
    Returns: (frames, actual_fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"비디오 파일 열기 실패: {video_path}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(duration * actual_fps) if duration else total_frames

    frames = []
    frame_skip = max(1, int(actual_fps / fps_limit)) if fps_limit else 1
    frame_idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            frames.append(frame.copy())
        frame_idx += 1

    cap.release()
    logger.info(
        f"파일 로드 완료: {len(frames)}프레임 "
        f"(FPS: {actual_fps:.1f}, 파일: {video_path})"
    )
    return frames, actual_fps
