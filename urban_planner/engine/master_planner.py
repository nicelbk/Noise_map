# Stage 1: 토지이용 마스터플랜 (Master Planner)
# 용도지역 배분 및 공간 배치 계획을 수립한다.

import yaml
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from shapely.geometry import (
    shape, mapping, Polygon, MultiPolygon, Point, LineString, box
)
from shapely.ops import transform, unary_union, split
import pyproj

_wgs84 = pyproj.CRS("EPSG:4326")
_korea = pyproj.CRS("EPSG:5179")
_to_korea = pyproj.Transformer.from_crs(_wgs84, _korea, always_xy=True).transform
_to_wgs84 = pyproj.Transformer.from_crs(_korea, _wgs84, always_xy=True).transform


def _load_zoning_rules():
    path = Path(__file__).parent.parent / "rules" / "zoning_rules.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


ZONING_RULES = _load_zoning_rules()


class MasterPlanner:
    """Stage 1: 용도지역 배분 및 공간 배치 계획"""

    def plan(self, boundary_geojson: dict, site_analysis: dict, design_params: dict) -> dict:
        """
        Args:
            boundary_geojson: 대상지 경계 GeoJSON
            site_analysis: Stage 0 분석 결과
            design_params: 설계 파라미터

        Returns:
            GeoJSON FeatureCollection (용도지역별 폴리곤)
        """
        # 대상지 경계 → 투영좌표
        if boundary_geojson.get("type") == "FeatureCollection":
            geom_wgs84 = shape(boundary_geojson["features"][0]["geometry"])
        elif boundary_geojson.get("type") == "Feature":
            geom_wgs84 = shape(boundary_geojson["geometry"])
        else:
            geom_wgs84 = shape(boundary_geojson)

        boundary_korea = transform(_to_korea, geom_wgs84)

        # 설계 파라미터 추출
        program = design_params.get("program", "혼합용도")
        region = design_params.get("region", "기타")
        transit = design_params.get("transit_access", "없음")
        priority = design_params.get("priority", {})

        # 1. 초기 용도 비율 결정 (유사사례 기반 or 프로그램 기본값)
        base_ratios = self._get_base_ratios(site_analysis, program, design_params)

        # 2. 우선순위 가중치 적용
        adjusted_ratios = self._apply_priority_weights(base_ratios, priority)

        # 3. 법규 최소기준 검증 및 보정
        valid_ratios = self._enforce_legal_minimums(adjusted_ratios, design_params)

        # 4. 공간 배치 (용도지역 폴리곤 생성)
        zone_polygons = self._allocate_zones(
            boundary_korea, valid_ratios, design_params, site_analysis
        )

        # 5. WGS84로 변환하여 GeoJSON FeatureCollection 생성
        features = []
        for zone in zone_polygons:
            geom_wgs84_out = transform(_to_wgs84, zone["geometry"])
            zoning_code = zone["zoning"]
            rule = ZONING_RULES["zoning_types"].get(zoning_code, {})

            features.append({
                "type": "Feature",
                "geometry": mapping(geom_wgs84_out),
                "properties": {
                    "zoning": zoning_code,
                    "category": rule.get("category", "기타"),
                    "color": rule.get("color", "#CCCCCC"),
                    "far_max": self._get_far(zoning_code, region),
                    "bcr_max": rule.get("bcr", 60),
                    "area_m2": round(zone["geometry"].area, 1),
                    "label": zoning_code,
                }
            })

        # 통계 계산
        total_area = boundary_korea.area
        stats = self._calc_stats(features, total_area)

        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "ratios": valid_ratios,
                "stats": stats,
                "total_area_m2": round(total_area, 1),
            }
        }

    # ─────────────────────────────────────────
    # 내부 메서드
    # ─────────────────────────────────────────

    def _get_base_ratios(self, site_analysis: dict, program: str, design_params: dict) -> dict:
        """초기 용도 비율 결정"""
        # 유사 사례 평균값 우선
        suggested = site_analysis.get("suggested_zoning_mix")
        if suggested:
            return dict(suggested)

        # 프로그램 기본값 사용
        program_ratios = ZONING_RULES.get("program_zoning_ratios", {})
        base = program_ratios.get(program, program_ratios.get("혼합용도"))
        return dict(base)

    def _apply_priority_weights(self, ratios: dict, priority: dict) -> dict:
        """우선순위 가중치 적용 (1~5점)"""
        if not priority:
            return ratios

        r = dict(ratios)

        # 밀도 우선순위: 높을수록 주거/상업 비율 ↑, 녹지 ↓
        density = priority.get("density", 3)
        green_prio = priority.get("green", 3)
        walk_prio = priority.get("walkability", 3)
        public_prio = priority.get("publicfacility", 3)

        # 밀도 조정 (±5% per point from center)
        density_delta = (density - 3) * 0.02
        r["residential"] = max(0.1, r.get("residential", 0.4) + density_delta)

        # 녹지 조정
        green_delta = (green_prio - 3) * 0.015
        r["green"] = max(0.10, r.get("green", 0.20) + green_delta)

        # 공공시설 조정
        public_delta = (public_prio - 3) * 0.01
        r["public"] = max(0.05, r.get("public", 0.10) + public_delta)

        # 합계 정규화
        total = sum(r.values())
        if total > 0:
            r = {k: round(v / total, 3) for k, v in r.items()}

        return r

    def _enforce_legal_minimums(self, ratios: dict, design_params: dict) -> dict:
        """법규 최소기준 적용"""
        r = dict(ratios)

        # 최소 녹지율 10%
        green_min = design_params.get("green_ratio_min", 0.10)
        if r.get("green", 0) < green_min:
            diff = green_min - r.get("green", 0)
            r["green"] = green_min
            # 주거에서 차감
            r["residential"] = max(0.1, r.get("residential", 0.4) - diff)

        # 최소 도로율 20%
        road_target = design_params.get("road_ratio_target", 0.25)
        if r.get("road", 0) < 0.20:
            r["road"] = max(0.20, road_target)

        # 합계 정규화
        total = sum(r.values())
        if total > 0:
            r = {k: round(v / total, 3) for k, v in r.items()}

        return r

    def _allocate_zones(
        self, boundary: Polygon, ratios: dict, design_params: dict, site_analysis: dict
    ) -> list:
        """
        용도지역 공간 배치:
        - 외곽 도로변 → 상업/준주거
        - 중심부 → 공공시설
        - 북측 → 공원/녹지
        - 내부 → 주거 (위치별 차등)
        """
        bbox = boundary.bounds  # (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = bbox
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2
        width = maxx - minx
        height = maxy - miny

        transit = design_params.get("transit_access", "없음")
        program = design_params.get("program", "혼합용도")
        region = design_params.get("region", "기타")

        zones = []
        remaining = boundary

        # 도로 면적 (나중에 road_generator가 처리하므로 여기선 예비 할당만)
        road_ratio = ratios.get("road", 0.25)

        # 상업/준주거 배치: 외곽 버퍼 (남측/동측)
        commercial_ratio = ratios.get("commercial", 0.08)
        if commercial_ratio > 0.02:
            commercial_depth = min(width * 0.15, height * 0.15, 100)
            # 남측 외곽 띠
            south_strip = boundary.intersection(
                box(minx, miny, maxx, miny + commercial_depth)
            )
            if not south_strip.is_empty and south_strip.area > 500:
                south_strip = south_strip.intersection(remaining)
                if not south_strip.is_empty:
                    zones.append({"geometry": south_strip, "zoning": "근린상업"})
                    remaining = remaining.difference(south_strip)

            # 역세권이면 중심부도 상업
            if transit == "역세권" and commercial_ratio > 0.10:
                center_commercial = Point(cx, cy).buffer(min(width, height) * 0.12)
                center_commercial = center_commercial.intersection(remaining)
                if not center_commercial.is_empty and center_commercial.area > 1000:
                    zones.append({"geometry": center_commercial, "zoning": "일반상업"})
                    remaining = remaining.difference(center_commercial)

        # 공원/녹지 배치: 북측
        green_ratio = ratios.get("green", 0.20)
        if green_ratio > 0.05:
            green_depth = height * (green_ratio * 0.6)
            north_green = boundary.intersection(
                box(minx, maxy - green_depth, maxx, maxy)
            )
            north_green = north_green.intersection(remaining)
            if not north_green.is_empty and north_green.area > 500:
                zones.append({"geometry": north_green, "zoning": "공원"})
                remaining = remaining.difference(north_green)

            # 추가 녹지 (서측 or 분산)
            extra_green_ratio = green_ratio * 0.4
            if extra_green_ratio > 0.03:
                extra_depth = width * extra_green_ratio
                west_green = boundary.intersection(
                    box(minx, miny, minx + extra_depth, maxy)
                )
                west_green = west_green.intersection(remaining)
                if not west_green.is_empty and west_green.area > 500:
                    zones.append({"geometry": west_green, "zoning": "자연녹지"})
                    remaining = remaining.difference(west_green)

        # 공공시설 배치: 중심부
        public_ratio = ratios.get("public", 0.10)
        if public_ratio > 0.03:
            public_r = min(width, height) * 0.10
            public_area = Point(cx, cy).buffer(public_r)
            public_area = public_area.intersection(remaining)
            if not public_area.is_empty and public_area.area > 500:
                zones.append({"geometry": public_area, "zoning": "공공시설"})
                remaining = remaining.difference(public_area)

        # 주거 배치: 나머지 면적 (위치별 차등)
        if not remaining.is_empty and remaining.area > 0:
            res_ratio = ratios.get("residential", 0.45)
            industrial_ratio = ratios.get("industrial", 0.0)

            if industrial_ratio > 0.02:
                # 공업 (동측)
                ind_depth = width * industrial_ratio * 1.5
                ind_area = boundary.intersection(
                    box(maxx - ind_depth, miny, maxx, maxy)
                )
                ind_area = ind_area.intersection(remaining)
                if not ind_area.is_empty and ind_area.area > 500:
                    zones.append({"geometry": ind_area, "zoning": "준공업"})
                    remaining = remaining.difference(ind_area)

            if not remaining.is_empty:
                # 주거: 내부 영역을 남/북으로 분할해 용도지역 차등
                mid_y = miny + (maxy - miny) * 0.5
                # 남측(대로변 쪽) → 제3종 또는 준주거
                south_res = remaining.intersection(box(minx, miny, maxx, mid_y))
                north_res = remaining.intersection(box(minx, mid_y, maxx, maxy))

                if program in ["상업중심", "혼합용도"] and transit == "역세권":
                    south_zone = "준주거"
                    north_zone = "제2종일반주거"
                elif program == "주거중심":
                    south_zone = "제2종일반주거"
                    north_zone = "제1종일반주거"
                else:
                    south_zone = "제3종일반주거"
                    north_zone = "제2종일반주거"

                if not south_res.is_empty and south_res.area > 500:
                    zones.append({"geometry": south_res, "zoning": south_zone})
                if not north_res.is_empty and north_res.area > 500:
                    zones.append({"geometry": north_res, "zoning": north_zone})

        return zones

    def _get_far(self, zoning_code: str, region: str) -> float:
        """용도지역 및 지역에 따른 용적률 반환"""
        rule = ZONING_RULES["zoning_types"].get(zoning_code, {})
        if region == "서울":
            return rule.get("far_seoul", rule.get("far_default", 200))
        return rule.get("far_default", 200)

    def _calc_stats(self, features: list, total_area: float) -> dict:
        """용도별 면적 통계 계산"""
        category_areas = {}
        for f in features:
            cat = f["properties"].get("category", "기타")
            area = f["properties"].get("area_m2", 0)
            category_areas[cat] = category_areas.get(cat, 0) + area

        return {
            k: {
                "area_m2": round(v, 1),
                "ratio": round(v / total_area, 3) if total_area > 0 else 0
            }
            for k, v in category_areas.items()
        }

    def generate_alternatives(
        self, boundary_geojson: dict, site_analysis: dict, base_params: dict
    ) -> dict:
        """
        대안 3개 생성:
        A안 (균형안): 입력 우선순위 그대로
        B안 (고밀안): 밀도 +2, 녹지 -1
        C안 (친환경안): 녹지 +2, 밀도 -1
        """
        params_a = dict(base_params)
        params_a["priority"] = dict(base_params.get("priority", {}))

        params_b = dict(base_params)
        priority_b = dict(base_params.get("priority", {}))
        priority_b["density"] = min(5, priority_b.get("density", 3) + 2)
        priority_b["green"] = max(1, priority_b.get("green", 3) - 1)
        params_b["priority"] = priority_b

        params_c = dict(base_params)
        priority_c = dict(base_params.get("priority", {}))
        priority_c["green"] = min(5, priority_c.get("green", 3) + 2)
        priority_c["density"] = max(1, priority_c.get("density", 3) - 1)
        params_c["priority"] = priority_c

        plan_a = self.plan(boundary_geojson, site_analysis, params_a)
        plan_b = self.plan(boundary_geojson, site_analysis, params_b)
        plan_c = self.plan(boundary_geojson, site_analysis, params_c)

        return {
            "A": {"label": "A안 (균형)", "plan": plan_a},
            "B": {"label": "B안 (고밀)", "plan": plan_b},
            "C": {"label": "C안 (친환경)", "plan": plan_c},
        }
