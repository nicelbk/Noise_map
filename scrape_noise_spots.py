"""
noiseinfo.or.kr 소음 측정지점 GIS 데이터 수집기
https://www.noiseinfo.or.kr/noise/spot.do

수집 데이터: 측정지점의 WGS84 좌표 (위경도)
출력 형식: CSV, GeoJSON
"""

import requests
import json
import csv
import time
import os
from pathlib import Path

# API 기본 설정
BASE_URL = "https://www.noiseinfo.or.kr"
SPOT_API = "/api/noise/getSpot.dojson"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.noiseinfo.or.kr/noise/spot.do",
}

# 지역 코드 (사이트에서 추출)
LOCAL_CODES = {
    "WJ": "강원원주시",
    "CC": "강원춘천시",
    "SW": "경기수원시",
    "CW": "경남창원시",
    "PH": "경북포항시",
    "GJ": "광주광역시",
    "DG": "대구광역시",
    "DJ": "대전광역시",
    "BS": "부산광역시",
    "SU": "서울특별시",
    "SJ": "세종특별자치시",
    "WS": "울산광역시",
    "KH": "인천강화",
    "IC": "인천광역시",
    "SC": "전남순천시",
    "YS": "전남여수시",
    "JJ": "전북전주시",
    "JE": "제주제주시",
    "CA": "충남천안시",
    "CJ": "충북청주시",
}

# 소음 유형
NOISE_TYPES = {
    "EA": "환경기준(자동)",
    "EM": "환경기준(수동)",
    "A":  "항공기소음",
    "R":  "도로교통소음",
    "RV": "도로교통소음(이동)",
}


def fetch_spots(noise_type: str, local_code: str, page: int = 1) -> dict:
    """지정된 소음유형·지역·페이지의 측정지점 데이터를 가져온다."""
    params = {
        "noiseType": noise_type,
        "mangCode": "",
        "localCode": local_code,
        "currentPage": page,
    }
    resp = requests.get(
        BASE_URL + SPOT_API,
        params=params,
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def collect_all_spots(
    local_codes: dict = None,
    noise_types: dict = None,
    delay: float = 0.3,
) -> list[dict]:
    """
    모든 지역·소음유형에 걸쳐 측정지점을 수집한다.

    Parameters
    ----------
    local_codes : 수집할 지역 코드 딕셔너리 (None이면 전체)
    noise_types : 수집할 소음유형 딕셔너리 (None이면 전체)
    delay       : 요청 간격 (초)
    """
    if local_codes is None:
        local_codes = LOCAL_CODES
    if noise_types is None:
        noise_types = NOISE_TYPES

    all_records = []
    seen_spots = set()          # (noiseType, spotCode) 중복 제거용

    for lc, lname in local_codes.items():
        for nt, nname in noise_types.items():
            page = 1
            while True:
                try:
                    data = fetch_spots(nt, lc, page)
                except requests.RequestException as e:
                    print(f"  [오류] {lname} / {nname} p{page}: {e}")
                    break

                rows = data.get("output", [])
                if not rows:
                    break

                for item in rows:
                    key = (nt, item.get("spotCode", ""))
                    if key in seen_spots:
                        continue
                    seen_spots.add(key)

                    lat = item.get("wgs84Lat")
                    lon = item.get("wgs84Lon")
                    if not (lat and lon):
                        continue

                    all_records.append({
                        "noise_type":     nt,
                        "noise_type_name": nname,
                        "local_code":     item.get("localCode", lc),
                        "local_name":     item.get("localName", lname),
                        "spot_code":      item.get("spotCode", ""),
                        "spot_name":      item.get("spotName", ""),
                        "div_name":       item.get("divName", ""),
                        "legl_sect_code": item.get("leglSectCode", ""),
                        "use_sect_name":  item.get("useSectName", ""),
                        "tm_loc_x":       item.get("tmLocX"),
                        "tm_loc_y":       item.get("tmLocY"),
                        "latitude":       lat,
                        "longitude":      lon,
                    })

                total = data.get("totalCnt", 0)
                fetched_so_far = (page - 1) * len(rows) + len(rows)
                print(
                    f"  {lname} / {nname}: "
                    f"p{page} ({fetched_so_far}/{total}건)"
                )

                if fetched_so_far >= total:
                    break
                page += 1
                time.sleep(delay)

            time.sleep(delay)

    return all_records


def save_csv(records: list[dict], path: str) -> None:
    """측정지점 데이터를 CSV로 저장한다."""
    if not records:
        print("[경고] 저장할 데이터가 없습니다.")
        return

    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV 저장 완료: {path}  ({len(records)}건)")


def save_geojson(records: list[dict], path: str) -> None:
    """측정지점 데이터를 GeoJSON (Point)으로 저장한다."""
    if not records:
        print("[경고] 저장할 데이터가 없습니다.")
        return

    features = []
    for r in records:
        props = {k: v for k, v in r.items() if k not in ("latitude", "longitude")}
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r["longitude"], r["latitude"]],   # GeoJSON은 [lon, lat]
            },
            "properties": props,
        })

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"GeoJSON 저장 완료: {path}  ({len(features)}건)")


def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("noiseinfo.or.kr 측정지점 GIS 데이터 수집")
    print("=" * 60)
    print(f"대상 지역 수: {len(LOCAL_CODES)}")
    print(f"소음 유형 수: {len(NOISE_TYPES)}")
    print()

    records = collect_all_spots()

    print()
    print(f"총 수집 건수: {len(records)}")
    print()

    save_csv(records, str(output_dir / "noise_spots.csv"))
    save_geojson(records, str(output_dir / "noise_spots.geojson"))

    # 소음유형별 GeoJSON 분리 저장
    by_type: dict[str, list] = {}
    for r in records:
        by_type.setdefault(r["noise_type"], []).append(r)

    for nt, rows in by_type.items():
        fname = f"noise_spots_{nt.lower()}.geojson"
        save_geojson(rows, str(output_dir / fname))

    print()
    print("완료.")


if __name__ == "__main__":
    main()
