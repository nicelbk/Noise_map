# Stage 0: 현황 분석 (Site Analyzer)
# 대상지 기본 현황을 분석하고 유사 사례를 매칭한다.

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import shape, mapping
from shapely.ops import transform
import pyproj

# 프로젝션 변환기 (WGS84 → EPSG:5179 한국중부원점)
_wgs84 = pyproj.CRS("EPSG:4326")
_korea = pyproj.CRS("EPSG:5179")
_proj_to_korea = pyproj.Transformer.from_crs(_wgs84, _korea, always_xy=True).transform
_proj_to_wgs84 = pyproj.Transformer.from_crs(_korea, _wgs84, always_xy=True).transform


def geojson_to_korea(geom_wgs84):
    """WGS84 GeoJSON 지오메트리를 EPSG:5179로 변환"""
    return transform(_proj_to_korea, geom_wgs84)


def korea_to_geojson(geom_korea):
    """EPSG:5179 지오메트리를 WGS84 GeoJSON으로 변환"""
    return transform(_proj_to_wgs84, geom_korea)


def load_case_studies():
    """유사 사례 DB 로드"""
    db_path = Path(__file__).parent.parent / "reference" / "case_studies.json"
    with open(db_path, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


class SiteAnalyzer:
    """Stage 0: 대상지 현황 분석"""

    def __init__(self):
        self.cases = load_case_studies()

    def analyze(self, boundary_geojson: dict, design_params: dict = None) -> dict:
        """
        대상지 분석 수행

        Args:
            boundary_geojson: Leaflet.draw로 그린 GeoJSON 폴리곤
            design_params: 설계 입력 파라미터 (선택)

        Returns:
            분석 결과 딕셔너리
        """
        # GeoJSON → Shapely 지오메트리
        if boundary_geojson.get("type") == "FeatureCollection":
            geom_wgs84 = shape(boundary_geojson["features"][0]["geometry"])
        elif boundary_geojson.get("type") == "Feature":
            geom_wgs84 = shape(boundary_geojson["geometry"])
        else:
            geom_wgs84 = shape(boundary_geojson)

        # 투영좌표계로 변환 (면적 계산을 위해)
        geom_korea = geojson_to_korea(geom_wgs84)

        # 기본 면적 / 둘레 계산
        area_m2 = geom_korea.area
        area_ha = area_m2 / 10000
        perimeter_m = geom_korea.length

        # 형상 지수 (원형 = 1.0, 복잡할수록 작아짐)
        # Polsby–Popper 지수
        shape_index = (4 * math.pi * area_m2) / (perimeter_m ** 2)

        # 면적 유효성 검사
        constraints = []
        if area_m2 < 1000:
            constraints.append({
                "level": "error",
                "code": "TOO_SMALL",
                "message": f"대상지 면적({area_m2:.0f}m²)이 최소 기준(1,000m²)보다 작습니다."
            })
        if area_m2 > 1_000_000:  # 100ha
            constraints.append({
                "level": "warning",
                "code": "TOO_LARGE",
                "message": f"대상지 면적({area_ha:.1f}ha)이 100ha를 초과합니다. 단계적 계획을 권장합니다."
            })

        # 재개발 가능 여부 (면적 5,000m² 이상)
        if design_params and design_params.get("development_type") == "재개발":
            if area_m2 < 5000:
                constraints.append({
                    "level": "error",
                    "code": "REDEVEL_MIN_AREA",
                    "message": "재개발 최소 면적(5,000m²)에 미달합니다."
                })

        # 지형 유형 추정 (slope_data가 없으면 평지로 기본 처리)
        slope_data = design_params.get("slope_data") if design_params else None
        terrain_type = self._estimate_terrain(slope_data)

        # 블록 크기 권장
        recommended_block_size = self._recommend_block_size(area_ha)

        # 유사 사례 매칭
        dev_type = design_params.get("development_type", "신규개발") if design_params else "신규개발"
        program = design_params.get("program", "혼합용도") if design_params else "혼합용도"
        similar_cases = self._match_cases(area_ha, dev_type, program)

        # 유사 사례의 평균 용도 비율 계산 (초기 계획안 제안용)
        suggested_zoning = self._calc_suggested_zoning(similar_cases)

        # 대상지 중심점 (WGS84)
        centroid = korea_to_geojson(geom_korea.centroid)
        centroid_coords = [centroid.x, centroid.y]  # [lng, lat]

        return {
            "area_m2": round(area_m2, 1),
            "area_ha": round(area_ha, 3),
            "perimeter_m": round(perimeter_m, 1),
            "shape_index": round(shape_index, 3),
            "terrain_type": terrain_type,
            "similar_cases": similar_cases[:3],
            "recommended_block_size": recommended_block_size,
            "suggested_zoning_mix": suggested_zoning,
            "constraints": constraints,
            "centroid": centroid_coords,
            "boundary_korea": mapping(geom_korea),  # 내부 처리용 투영좌표
        }

    def _estimate_terrain(self, slope_data) -> str:
        """경사도 데이터 기반 지형 유형 추정"""
        if slope_data is None:
            return "평지"  # 기본값
        avg_slope = np.mean(slope_data)
        if avg_slope < 5:
            return "평지"
        elif avg_slope < 15:
            return "완경사"
        else:
            return "급경사"

    def _recommend_block_size(self, area_ha: float) -> str:
        """대상지 면적에 따른 권장 블록 크기"""
        if area_ha < 5:
            return "소블록"   # 2,000~5,000m²
        elif area_ha < 30:
            return "중블록"   # 5,000~15,000m²
        else:
            return "대블록"   # 15,000~30,000m²

    def _match_cases(self, area_ha: float, dev_type: str, program: str) -> list:
        """
        유사 사례 매칭 알고리즘
        - 면적 유사도 (40%)
        - 개발유형 일치 (30%)
        - 프로그램 유사도 (20%)
        - 입지특성 유사도 (10%)
        """
        # 프로그램 → 특성 매핑
        program_char_map = {
            "주거중심": ["주거", "저밀"],
            "상업중심": ["상업", "고밀"],
            "혼합용도": ["혼합", "자족"],
            "자족도시": ["자족", "고용"],
            "산업단지": ["산업", "공업"],
        }
        target_chars = program_char_map.get(program, [])

        # 개발유형 매핑
        type_map = {
            "신규개발": ["신도시", "행정중심복합도시", "경제자유구역"],
            "재개발": ["재개발"],
            "재건축": ["재건축", "재개발"],
            "도시재생": ["도시재생", "재개발"],
        }
        target_types = type_map.get(dev_type, [dev_type])

        scored = []
        for case in self.cases:
            score = 0.0

            # 1. 면적 유사도 (40%)
            case_ha = case["area_ha"]
            if case_ha > 0:
                ratio = min(area_ha, case_ha) / max(area_ha, case_ha)
                score += ratio * 0.40

            # 2. 개발유형 일치 (30%)
            if case["type"] in target_types:
                score += 0.30
            elif any(t in case["type"] for t in target_types):
                score += 0.15

            # 3. 프로그램/특성 유사도 (20%)
            case_chars = case.get("characteristics", [])
            char_match = sum(1 for c in target_chars if any(c in cc for cc in case_chars))
            if target_chars:
                score += (char_match / len(target_chars)) * 0.20

            # 4. 기본 점수 (항상 일부 포함)
            score += 0.10

            scored.append({
                "id": case["id"],
                "name": case["name"],
                "type": case["type"],
                "area_ha": case["area_ha"],
                "population": case["population"],
                "similarity_score": round(score, 3),
                "zoning_mix": case["zoning_mix"],
                "far_avg": case["far_avg"],
                "avg_block_size_m2": case["avg_block_size_m2"],
                "road_density_km_per_km2": case["road_density_km_per_km2"],
                "lessons": case["lessons"],
                "road_pattern": case.get("road_pattern", "격자형"),
            })

        # 유사도 기준 정렬
        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored

    def _calc_suggested_zoning(self, similar_cases: list) -> dict:
        """유사 사례 상위 3개의 용도 비율 평균 계산"""
        top3 = similar_cases[:3]
        if not top3:
            return {
                "residential": 0.45,
                "commercial": 0.08,
                "industrial": 0.0,
                "green": 0.22,
                "public": 0.10,
                "road": 0.15,
            }

        keys = ["residential", "commercial", "industrial", "green", "public", "road"]
        avg = {}
        for k in keys:
            vals = [c["zoning_mix"].get(k, 0) for c in top3]
            avg[k] = round(sum(vals) / len(vals), 3)

        # 합계가 1.0이 되도록 정규화
        total = sum(avg.values())
        if total > 0:
            avg = {k: round(v / total, 3) for k, v in avg.items()}

        return avg
