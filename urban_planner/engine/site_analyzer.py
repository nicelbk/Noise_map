"""
대상지 현황 분석 모듈
"""

import json
import math
import os
import geopandas as gpd
from shapely.geometry import shape, Polygon
from pyproj import Transformer

EPSG_WGS84 = 4326
EPSG_5179 = 5179

transformer_to_5179 = Transformer.from_crs(EPSG_WGS84, EPSG_5179, always_xy=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_STUDIES_PATH = os.path.join(BASE_DIR, 'reference', 'case_studies.json')


class SiteAnalyzer:

    def __init__(self):
        self._case_studies = self._load_case_studies()

    def _load_case_studies(self) -> list:
        try:
            with open(CASE_STUDIES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('cases', [])
        except Exception:
            return []

    def analyze(
        self,
        boundary_geojson: dict,
        context: dict
    ) -> dict:
        """
        반환:
        {
          'area_m2': float,
          'area_ha': float,
          'perimeter_m': float,
          'shape_index': float,
          'terrain_type': str,
          'similar_cases': list[dict],
          'recommended_block_size': str,
          'constraints': list[str],
          'entry_point_count': int
        }
        """
        try:
            geom = shape(boundary_geojson)
            coords = list(geom.exterior.coords)
            transformed = [transformer_to_5179.transform(lon, lat) for lon, lat in coords]
            poly_5179 = Polygon(transformed)

            area_m2 = poly_5179.area
            area_ha = area_m2 / 10000.0
            perimeter_m = poly_5179.length

            # Shape index: compactness ratio (1.0 = perfect circle)
            # Shape index = perimeter / (2 * sqrt(pi * area))
            shape_index = perimeter_m / (2 * math.sqrt(math.pi * area_m2)) if area_m2 > 0 else 1.0

            # Terrain type based on context
            terrain_type = self._estimate_terrain_type(context)

            # Entry point count from context
            entry_points = context.get('entry_points', [])
            entry_point_count = len(entry_points)

            # Get design params if provided
            dev_type = context.get('development_type', '신규개발')
            program = context.get('program', '혼합용도')

            similar_cases = self._match_similar_cases(area_ha, dev_type, program)

            # Recommended block size
            recommended_block_size = self._recommend_block_size(area_ha, program)

            # Constraints
            constraints = self._identify_constraints(
                area_ha, entry_point_count, context
            )

            return {
                'area_m2': round(area_m2, 1),
                'area_ha': round(area_ha, 2),
                'perimeter_m': round(perimeter_m, 1),
                'shape_index': round(shape_index, 3),
                'terrain_type': terrain_type,
                'similar_cases': similar_cases,
                'recommended_block_size': recommended_block_size,
                'constraints': constraints,
                'entry_point_count': entry_point_count,
            }
        except Exception as e:
            return {
                'area_m2': 0.0,
                'area_ha': 0.0,
                'perimeter_m': 0.0,
                'shape_index': 1.0,
                'terrain_type': '평지',
                'similar_cases': [],
                'recommended_block_size': '5,000~10,000㎡',
                'constraints': [],
                'entry_point_count': 0,
            }

    def _estimate_terrain_type(self, context: dict) -> str:
        """지형 유형 추정"""
        ctx_analysis = context.get('context', {})
        recommendations = ctx_analysis.get('recommendations', [])
        for rec in recommendations:
            if '구릉' in rec or '경사' in rec:
                return '구릉지'
            if '수변' in rec or '하천' in rec:
                return '수변'
        return '평지'

    def _match_similar_cases(
        self,
        area_ha: float,
        dev_type: str,
        program: str
    ) -> list:
        """
        reference/case_studies.json에서
        면적/유형/프로그램 유사도로 상위 3개 반환
        각 케이스에 similarity_score(0~1) 포함
        """
        if not self._case_studies:
            return []

        scored = []
        for case in self._case_studies:
            score = 0.0

            # Area similarity (log scale)
            case_area = case.get('area_ha', 1)
            if case_area > 0 and area_ha > 0:
                ratio = min(area_ha, case_area) / max(area_ha, case_area)
                score += ratio * 0.4  # 40% weight

            # Development type match
            if case.get('type', '') == dev_type:
                score += 0.3  # 30% weight

            # Program match
            case_prog = case.get('program', '')
            prog_map = {
                '주거중심': ['주거중심', '주거'],
                '혼합용도': ['혼합용도', '복합', '자족도시'],
                '상업중심': ['상업중심', '상업'],
                '자족도시': ['자족도시', '혼합용도'],
            }
            target_progs = prog_map.get(program, [program])
            if case_prog in target_progs or program in case_prog:
                score += 0.3  # 30% weight
            elif any(p in case_prog for p in target_progs):
                score += 0.15

            scored.append({**case, 'similarity_score': round(score, 3)})

        scored.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored[:3]

    def _recommend_block_size(self, area_ha: float, program: str) -> str:
        """블록 크기 권장"""
        if program == '상업중심':
            if area_ha < 10:
                return '3,000~6,000㎡'
            elif area_ha < 50:
                return '5,000~10,000㎡'
            else:
                return '8,000~15,000㎡'
        elif program == '주거중심':
            if area_ha < 10:
                return '3,000~5,000㎡'
            elif area_ha < 50:
                return '5,000~8,000㎡'
            else:
                return '6,000~10,000㎡'
        else:  # 혼합용도, 자족도시
            if area_ha < 10:
                return '3,000~6,000㎡'
            elif area_ha < 50:
                return '5,000~10,000㎡'
            else:
                return '6,000~12,000㎡'

    def _identify_constraints(
        self,
        area_ha: float,
        entry_point_count: int,
        context: dict
    ) -> list:
        """계획 제약 조건 식별"""
        constraints = []

        if entry_point_count == 0:
            constraints.append("진입도로 없음: 신규 진입로 계획 필요")
        elif entry_point_count == 1:
            constraints.append("단일 진입점: 비상 접근성 확보 필요")

        if area_ha < 1:
            constraints.append("소규모 부지(1ha 미만): 공공시설 설치 면제 가능")
        elif area_ha < 3:
            constraints.append("소규모 부지(3ha 미만): 학교용지 기부채납 면제")

        ctx_analysis = context.get('context', {})
        if ctx_analysis.get('is_tod_zone'):
            constraints.append("역세권: 주차장 설치 완화 가능, 용적률 상향 검토")

        nearby_subway = ctx_analysis.get('nearest_subway_m', 9999)
        if nearby_subway > 1000:
            constraints.append("대중교통 취약지역: 버스 노선 확충 또는 공유 모빌리티 계획 필요")

        return constraints
