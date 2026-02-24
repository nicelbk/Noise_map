# Stage 4: 공공시설 배치 (Facility Placer)
# 블록 배치 결과를 바탕으로 공공시설을 자동 배치한다.

import yaml
import math
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import (
    shape, mapping, Polygon, Point, MultiPolygon, box
)
from shapely.ops import transform, unary_union
import pyproj

_wgs84 = pyproj.CRS("EPSG:4326")
_korea = pyproj.CRS("EPSG:5179")
_to_korea = pyproj.Transformer.from_crs(_wgs84, _korea, always_xy=True).transform
_to_wgs84 = pyproj.Transformer.from_crs(_korea, _wgs84, always_xy=True).transform


def _load_facility_rules():
    path = Path(__file__).parent.parent / "rules" / "facility_rules.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


FACILITY_RULES = _load_facility_rules()


class FacilityPlacer:
    """Stage 4: 공공시설 자동 배치"""

    def place(self, blocks_geojson: dict, design_params: dict, site_analysis: dict) -> dict:
        """
        Args:
            blocks_geojson: Stage 3 블록 GeoJSON
            design_params: 설계 파라미터
            site_analysis: Stage 0 분석 결과

        Returns:
            {facilities GeoJSON, coverage_analysis}
        """
        # 블록 파싱
        blocks = []
        residential_blocks = []
        total_area = 0
        res_area = 0

        for f in blocks_geojson.get("features", []):
            geom_wgs84 = shape(f["geometry"])
            geom_korea = transform(_to_korea, geom_wgs84)
            cat = f["properties"].get("category", "")
            area = geom_korea.area
            total_area += area

            block_info = {
                "geometry": geom_korea,
                "properties": f["properties"],
                "area_m2": area,
                "centroid": geom_korea.centroid,
            }
            blocks.append(block_info)

            if cat == "주거":
                residential_blocks.append(block_info)
                res_area += area

        # 인구/세대 추정
        pop_rules = FACILITY_RULES.get("population_estimation", {})
        avg_floor_area = pop_rules.get("평균_세대면적_m2", 84)
        persons_per_hh = pop_rules.get("공동주택_세대당_인원", 2.5)
        res_ratio = pop_rules.get("주거용도_거주비율", 0.85)

        # 평균 용적률로 연면적 추정
        avg_far = 180  # 기본값
        if blocks:
            far_vals = [b["properties"].get("far_max", 180) for b in blocks
                        if b["properties"].get("category") == "주거"]
            if far_vals:
                avg_far = sum(far_vals) / len(far_vals)

        total_floor_area = res_area * (avg_far / 100) * res_ratio
        est_households = int(total_floor_area / avg_floor_area) if avg_floor_area > 0 else 0
        est_population = int(est_households * persons_per_hh)

        # 시설 배치
        facilities = []

        # 공원 배치
        parks = self._place_parks(blocks, residential_blocks, total_area)
        facilities.extend(parks)

        # 학교 배치
        schools = self._place_schools(blocks, residential_blocks, est_households)
        facilities.extend(schools)

        # 주민센터
        community = self._place_community_facilities(blocks, est_households)
        facilities.extend(community)

        # WGS84로 변환
        features = []
        for fac in facilities:
            geom_out = transform(_to_wgs84, fac["geometry"])
            features.append({
                "type": "Feature",
                "geometry": mapping(geom_out),
                "properties": fac["properties"],
            })

        # 커버리지 분석
        coverage = self._analyze_coverage(facilities, residential_blocks)

        return {
            "facilities": {
                "type": "FeatureCollection",
                "features": features,
            },
            "coverage_analysis": coverage,
            "estimates": {
                "households": est_households,
                "population": est_population,
                "residential_area_m2": round(res_area, 1),
            }
        }

    # ─────────────────────────────────────────
    # 시설 배치 내부 메서드
    # ─────────────────────────────────────────

    def _place_parks(self, blocks: list, residential_blocks: list, total_area: float) -> list:
        """공원 배치"""
        parks = []
        park_rules = FACILITY_RULES.get("parks", {})

        if not residential_blocks:
            return parks

        # 근린공원 (1개소 이상)
        np_rule = park_rules.get("근린공원", {})
        np_min_area = np_rule.get("min_area_m2", 10000)
        np_target_area = max(np_min_area, total_area * 0.05)

        # 대상지 내 가장 큰 주거 블록 인접 위치에 배치
        sorted_res = sorted(residential_blocks, key=lambda b: b["area_m2"], reverse=True)
        if sorted_res:
            ref_block = sorted_res[0]
            centroid = ref_block["centroid"]
            park_geom = centroid.buffer(math.sqrt(np_target_area / math.pi))
            parks.append({
                "geometry": park_geom,
                "properties": {
                    "facility_type": "근린공원",
                    "code": "NP",
                    "area_m2": round(park_geom.area, 1),
                    "service_distance_m": 500,
                    "color": "#32CD32",
                    "label": "근린공원",
                }
            })

        # 어린이공원 (250m 유치거리 기준으로 배치)
        cp_rule = park_rules.get("어린이공원", {})
        cp_min_area = cp_rule.get("min_area_m2", 1500)
        cp_service_dist = cp_rule.get("service_distance_m", 250)

        # 주거 블록들을 250m 격자로 나눠 어린이공원 배치
        if residential_blocks:
            all_res_union = unary_union([b["geometry"] for b in residential_blocks])
            res_bounds = all_res_union.bounds
            minx, miny, maxx, maxy = res_bounds

            placed_parks = [p["geometry"].centroid for p in parks]
            grid_spacing = cp_service_dist * 1.5  # 약 375m 간격으로 배치

            y = miny + grid_spacing / 2
            count = 0
            while y < maxy and count < 6:
                x = minx + grid_spacing / 2
                while x < maxx and count < 6:
                    pt = Point(x, y)
                    # 주거지역 내 위치인지 확인
                    if all_res_union.contains(pt) or all_res_union.distance(pt) < 50:
                        # 기존 공원과 250m 이상 떨어져 있는지 확인
                        too_close = any(
                            pt.distance(ep) < cp_service_dist * 0.8
                            for ep in placed_parks
                        )
                        if not too_close:
                            cp_geom = pt.buffer(math.sqrt(cp_min_area / math.pi))
                            parks.append({
                                "geometry": cp_geom,
                                "properties": {
                                    "facility_type": "어린이공원",
                                    "code": "CP",
                                    "area_m2": round(cp_geom.area, 1),
                                    "service_distance_m": 250,
                                    "color": "#90EE90",
                                    "label": "어린이공원",
                                }
                            })
                            placed_parks.append(pt)
                            count += 1
                    x += grid_spacing
                y += grid_spacing

        return parks

    def _place_schools(self, blocks: list, residential_blocks: list, households: int) -> list:
        """학교 배치"""
        schools = []
        school_rules = FACILITY_RULES.get("schools", {})

        if not residential_blocks or households == 0:
            return schools

        # 초등학교 배치
        es_rule = school_rules.get("초등학교", {})
        es_per_hh = es_rule.get("households_per_unit", 500)
        es_classes = es_rule.get("typical_classes", 18)
        es_area_per_class = es_rule.get("min_area_m2_per_class", 660)
        es_min_area = es_classes * es_area_per_class

        num_es = max(1, int(households / es_per_hh))
        num_es = min(num_es, 5)  # 최대 5개소

        # 주거 블록들을 균등 분배하여 배치
        res_centroids = [b["centroid"] for b in residential_blocks]
        if res_centroids:
            # 균등 배분: 전체 주거 영역을 num_es 구역으로 분할
            all_x = [c.x for c in res_centroids]
            all_y = [c.y for c in res_centroids]

            for i in range(num_es):
                frac = (i + 0.5) / num_es
                # 주거 중심 영역을 따라 배치
                idx = int(frac * len(res_centroids))
                if idx < len(res_centroids):
                    pt = res_centroids[idx]
                    school_geom = pt.buffer(math.sqrt(es_min_area / math.pi))
                    schools.append({
                        "geometry": school_geom,
                        "properties": {
                            "facility_type": "초등학교",
                            "code": "ES",
                            "area_m2": round(school_geom.area, 1),
                            "classes": es_classes,
                            "service_distance_m": 500,
                            "color": "#4169E1",
                            "label": f"초등학교 {i+1}",
                        }
                    })

        # 중학교 배치
        ms_rule = school_rules.get("중학교", {})
        ms_per_hh = ms_rule.get("households_per_unit", 2000)
        ms_classes = ms_rule.get("typical_classes", 24)
        ms_area_per_class = ms_rule.get("min_area_m2_per_class", 900)
        ms_min_area = ms_classes * ms_area_per_class

        num_ms = max(0, int(households / ms_per_hh))
        num_ms = min(num_ms, 3)

        if num_ms > 0 and res_centroids:
            mid_idx = len(res_centroids) // 2
            pt = res_centroids[mid_idx]
            # 초등학교와 약간 오프셋
            pt_offset = Point(pt.x + 100, pt.y + 100)
            ms_geom = pt_offset.buffer(math.sqrt(ms_min_area / math.pi))
            schools.append({
                "geometry": ms_geom,
                "properties": {
                    "facility_type": "중학교",
                    "code": "MS",
                    "area_m2": round(ms_geom.area, 1),
                    "classes": ms_classes,
                    "service_distance_m": 1000,
                    "color": "#1E90FF",
                    "label": "중학교",
                }
            })

        return schools

    def _place_community_facilities(self, blocks: list, households: int) -> list:
        """주민센터 등 커뮤니티 시설 배치"""
        facilities = []
        cf_rules = FACILITY_RULES.get("community_facilities", {})

        if not blocks or households == 0:
            return facilities

        # 주민센터
        cc_rule = cf_rules.get("주민센터", {})
        cc_per_hh = cc_rule.get("households_per_unit", 3000)
        cc_min_area = cc_rule.get("min_area_m2", 1000)

        num_cc = max(1, int(households / cc_per_hh))
        num_cc = min(num_cc, 3)

        # 중심부 블록에 배치
        if blocks:
            all_centroids = [b["centroid"] for b in blocks]
            # 전체 중심점 계산
            cx = np.mean([c.x for c in all_centroids])
            cy = np.mean([c.y for c in all_centroids])
            center = Point(cx, cy)

            # 가장 중심에 가까운 블록 선택
            blocks_sorted = sorted(blocks, key=lambda b: b["centroid"].distance(center))
            pt = blocks_sorted[0]["centroid"]
            cc_geom = pt.buffer(math.sqrt(cc_min_area / math.pi))
            facilities.append({
                "geometry": cc_geom,
                "properties": {
                    "facility_type": "주민센터",
                    "code": "CC",
                    "area_m2": round(cc_geom.area, 1),
                    "service_distance_m": None,
                    "color": "#9370DB",
                    "label": "주민센터",
                }
            })

        return facilities

    def _analyze_coverage(self, facilities: list, residential_blocks: list) -> dict:
        """공원/학교 유치권 커버리지 분석"""
        if not residential_blocks:
            return {
                "park_coverage_ratio": 0.0,
                "school_coverage_ratio": 0.0,
                "uncovered_areas": {"type": "FeatureCollection", "features": []},
            }

        all_res = unary_union([b["geometry"] for b in residential_blocks])
        res_area = all_res.area

        # 어린이공원 유치권
        park_buffers = []
        for fac in facilities:
            if fac["properties"].get("code") in ["CP", "NP"]:
                dist = fac["properties"].get("service_distance_m", 250)
                if dist:
                    park_buffers.append(fac["geometry"].centroid.buffer(dist))

        if park_buffers:
            park_union = unary_union(park_buffers)
            park_covered = all_res.intersection(park_union)
            park_coverage = park_covered.area / res_area if res_area > 0 else 0
        else:
            park_coverage = 0

        # 학교 유치권
        school_buffers = []
        for fac in facilities:
            if fac["properties"].get("code") in ["ES", "MS"]:
                dist = fac["properties"].get("service_distance_m", 500)
                if dist:
                    school_buffers.append(fac["geometry"].centroid.buffer(dist))

        if school_buffers:
            school_union = unary_union(school_buffers)
            school_covered = all_res.intersection(school_union)
            school_coverage = school_covered.area / res_area if res_area > 0 else 0
        else:
            school_coverage = 0

        # 미커버 지역
        if park_buffers:
            uncovered_geom = all_res.difference(park_union)
        else:
            uncovered_geom = all_res

        uncovered_features = []
        if not uncovered_geom.is_empty and uncovered_geom.area > 100:
            geom_wgs84 = transform(_to_wgs84, uncovered_geom)
            uncovered_features.append({
                "type": "Feature",
                "geometry": mapping(geom_wgs84),
                "properties": {"type": "공원_미커버", "area_m2": round(uncovered_geom.area, 1)},
            })

        return {
            "park_coverage_ratio": round(park_coverage, 3),
            "school_coverage_ratio": round(school_coverage, 3),
            "uncovered_areas": {
                "type": "FeatureCollection",
                "features": uncovered_features,
            },
        }
