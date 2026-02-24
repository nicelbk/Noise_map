# Stage 3: 블록 분할 (Block Divider)
# 도로로 대상지를 분할하여 개발 블록을 생성한다.

from typing import Optional
import numpy as np

import yaml
from pathlib import Path

from shapely.geometry import (
    shape, mapping, Polygon, MultiPolygon, Point, LineString, box
)
from shapely.ops import transform, unary_union, polygonize
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


class BlockDivider:
    """Stage 3: 도로망으로 대상지를 블록으로 분할"""

    def divide(
        self,
        boundary_geojson: dict,
        road_polygons: dict,
        master_plan: dict,
        design_params: dict,
    ) -> dict:
        """
        Args:
            boundary_geojson: 대상지 경계 GeoJSON
            road_polygons: Stage 2 도로 폴리곤 GeoJSON
            master_plan: Stage 1 마스터플랜 GeoJSON
            design_params: 설계 파라미터

        Returns:
            GeoJSON FeatureCollection (블록 폴리곤 + 속성)
        """
        # 대상지 경계
        if boundary_geojson.get("type") == "FeatureCollection":
            boundary_wgs84 = shape(boundary_geojson["features"][0]["geometry"])
        elif boundary_geojson.get("type") == "Feature":
            boundary_wgs84 = shape(boundary_geojson["geometry"])
        else:
            boundary_wgs84 = shape(boundary_geojson)
        boundary_korea = transform(_to_korea, boundary_wgs84)

        # 도로 폴리곤 합집합
        road_geoms = []
        for f in road_polygons.get("features", []):
            geom = shape(f["geometry"])
            geom_korea = transform(_to_korea, geom)
            road_geoms.append(geom_korea)

        if road_geoms:
            road_union = unary_union(road_geoms)
        else:
            road_union = Polygon()  # 빈 폴리곤

        # 용도지역 폴리곤 (마스터플랜)
        zone_geoms = []
        for f in master_plan.get("features", []):
            geom = shape(f["geometry"])
            geom_korea = transform(_to_korea, geom)
            zone_geoms.append({
                "geometry": geom_korea,
                "properties": f.get("properties", {}),
            })

        # 대상지에서 도로 제외 → 블록 영역
        if not road_union.is_empty:
            block_area = boundary_korea.difference(road_union)
        else:
            block_area = boundary_korea

        # 블록 폴리곤 추출
        raw_blocks = self._extract_blocks(block_area)

        # 각 블록에 용도지역 속성 부여
        region = design_params.get("region", "기타")
        blocks_with_props = []
        for i, block_geom in enumerate(raw_blocks):
            if block_geom.area < 500:  # 너무 작은 파편 제거
                continue

            # 가장 많이 겹치는 용도지역 찾기
            zoning = self._assign_zoning(block_geom, zone_geoms)

            # 접하는 최대 도로폭 계산
            adj_road_width = self._get_adjacent_road_width(
                block_geom, road_geoms, road_polygons
            )

            # 법규 적용
            rule = ZONING_RULES["zoning_types"].get(zoning, {})
            far_max = self._get_far(zoning, region)
            bcr_max = rule.get("bcr", 60)

            # 최고 높이 추정 (용적률 / 건폐율 기반)
            if bcr_max > 0 and far_max > 0:
                floors = far_max / bcr_max
                max_height = int(floors * 3.0)  # 층당 3m 기준
            else:
                max_height = None

            # 공공기여 필요 여부 (용적률 인센티브 시)
            public_contribution = far_max > 200

            block_geom_wgs84 = transform(_to_wgs84, block_geom)

            blocks_with_props.append({
                "type": "Feature",
                "geometry": mapping(block_geom_wgs84),
                "properties": {
                    "block_id": f"BLK-{i+1:04d}",
                    "area_m2": round(block_geom.area, 1),
                    "zoning": zoning,
                    "category": rule.get("category", "기타"),
                    "color": rule.get("color", "#CCCCCC"),
                    "far_max": far_max,
                    "bcr_max": bcr_max,
                    "max_height_m": max_height,
                    "adjacent_road_width_m": adj_road_width,
                    "public_contribution": public_contribution,
                    "has_road_access": adj_road_width > 0,
                }
            })

        # 법규 검증: 2,000m² 미만 블록 경고
        issues = []
        for b in blocks_with_props:
            if b["properties"]["area_m2"] < 2000:
                issues.append({
                    "block_id": b["properties"]["block_id"],
                    "issue": "SMALL_BLOCK",
                    "message": f"블록 면적 {b['properties']['area_m2']:.0f}m²가 최소기준(2,000m²)에 미달"
                })
            if not b["properties"]["has_road_access"]:
                issues.append({
                    "block_id": b["properties"]["block_id"],
                    "issue": "NO_ROAD_ACCESS",
                    "message": "도로에 접하지 않는 맹지 발생"
                })

        return {
            "type": "FeatureCollection",
            "features": blocks_with_props,
            "properties": {
                "total_blocks": len(blocks_with_props),
                "issues": issues,
            }
        }

    def _extract_blocks(self, block_area) -> list:
        """블록 영역에서 개별 폴리곤 추출"""
        blocks = []
        if block_area.is_empty:
            return blocks

        if isinstance(block_area, Polygon):
            if block_area.is_valid and not block_area.is_empty:
                blocks.append(block_area)
        elif isinstance(block_area, MultiPolygon):
            for geom in block_area.geoms:
                if geom.is_valid and not geom.is_empty:
                    blocks.append(geom)
        return blocks

    def _assign_zoning(self, block_geom, zone_geoms: list) -> str:
        """블록 폴리곤에 가장 많이 겹치는 용도지역 할당"""
        best_zone = "제2종일반주거"
        best_area = 0

        for z in zone_geoms:
            try:
                intersection = block_geom.intersection(z["geometry"])
                if not intersection.is_empty:
                    area = intersection.area
                    if area > best_area:
                        best_area = area
                        best_zone = z["properties"].get("zoning", "제2종일반주거")
            except Exception:
                pass

        return best_zone

    def _get_adjacent_road_width(
        self, block_geom, road_geoms: list, road_polygons_geojson: dict
    ) -> float:
        """블록에 인접한 최대 도로 폭원 반환"""
        max_width = 0
        block_boundary = block_geom.boundary

        for i, road_poly in enumerate(road_geoms):
            try:
                if block_boundary.distance(road_poly) < 1.0:
                    # 해당 도로의 폭원 정보 가져오기
                    if i < len(road_polygons_geojson.get("features", [])):
                        w = road_polygons_geojson["features"][i]["properties"].get(
                            "width_m", 8
                        )
                        max_width = max(max_width, w)
            except Exception:
                pass

        # 최소 접면 8m
        return max_width if max_width > 0 else 8

    def _get_far(self, zoning_code: str, region: str) -> float:
        """용도지역 및 지역에 따른 용적률 반환"""
        rule = ZONING_RULES["zoning_types"].get(zoning_code, {})
        if region == "서울":
            return rule.get("far_seoul", rule.get("far_default", 200))
        return rule.get("far_default", 200)
