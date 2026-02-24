# Stage 2: 도로망 생성 (Road Generator)
# 대상지 경계와 마스터플랜을 바탕으로 도로망을 생성한다.

import yaml
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx
from shapely.geometry import (
    shape, mapping, Polygon, MultiPolygon, Point,
    LineString, MultiLineString, box
)
from shapely.ops import transform, unary_union, split, linemerge
import pyproj

_wgs84 = pyproj.CRS("EPSG:4326")
_korea = pyproj.CRS("EPSG:5179")
_to_korea = pyproj.Transformer.from_crs(_wgs84, _korea, always_xy=True).transform
_to_wgs84 = pyproj.Transformer.from_crs(_korea, _wgs84, always_xy=True).transform


def _load_road_rules():
    path = Path(__file__).parent.parent / "rules" / "road_rules.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


ROAD_RULES = _load_road_rules()


class RoadGenerator:
    """Stage 2: 도로망 생성"""

    def generate(self, boundary_geojson: dict, master_plan: dict, design_params: dict) -> dict:
        """
        Args:
            boundary_geojson: 대상지 경계 GeoJSON
            master_plan: Stage 1 마스터플랜 GeoJSON
            design_params: 설계 파라미터

        Returns:
            {road_network, road_polygons, road_stats}
        """
        # 대상지 경계 → 투영좌표
        if boundary_geojson.get("type") == "FeatureCollection":
            geom_wgs84 = shape(boundary_geojson["features"][0]["geometry"])
        elif boundary_geojson.get("type") == "Feature":
            geom_wgs84 = shape(boundary_geojson["geometry"])
        else:
            geom_wgs84 = shape(boundary_geojson)

        boundary_korea = transform(_to_korea, geom_wgs84)
        bbox = boundary_korea.bounds
        minx, miny, maxx, maxy = bbox
        width = maxx - minx
        height = maxy - miny
        area_m2 = boundary_korea.area

        # 기존 도로 처리
        existing_roads = design_params.get("existing_roads", [])

        # 도로율 목표
        road_ratio_target = design_params.get("road_ratio_target", 0.25)

        # 1. 주간선도로 생성 (대상지 관통)
        arterial_lines = self._generate_arterial_roads(
            boundary_korea, width, height, minx, miny, maxx, maxy, design_params
        )

        # 2. 보조간선도로 생성
        collector_lines = self._generate_collector_roads(
            boundary_korea, arterial_lines, width, height, minx, miny, maxx, maxy
        )

        # 3. 국지도로 생성
        local_lines = self._generate_local_roads(
            boundary_korea, arterial_lines, collector_lines,
            width, height, minx, miny, maxx, maxy
        )

        # 4. 모든 도로선 합병
        all_lines = arterial_lines + collector_lines + local_lines

        # 5. 도로 폴리곤 생성 (선형 → 면적, 폭 적용)
        road_polygons_korea = self._lines_to_polygons(all_lines, boundary_korea)

        # 6. 도로율 검증 및 조정
        road_area = sum(p["geometry"].area for p in road_polygons_korea)
        road_ratio = road_area / area_m2

        # 도율율이 기준 미달이면 국지도로 추가
        if road_ratio < 0.20:
            extra_lines = self._add_extra_roads(
                boundary_korea, all_lines, width, height, minx, miny, maxx, maxy
            )
            all_lines += extra_lines
            road_polygons_korea = self._lines_to_polygons(all_lines, boundary_korea)

        # 7. WGS84로 변환
        network_features = []
        for road in all_lines:
            geom_out = transform(_to_wgs84, road["geometry"])
            network_features.append({
                "type": "Feature",
                "geometry": mapping(geom_out),
                "properties": {
                    "road_type": road["road_type"],
                    "width_m": road["width_m"],
                    "hierarchy": road["hierarchy"],
                    "color": road.get("color", "#808080"),
                }
            })

        polygon_features = []
        for rpoly in road_polygons_korea:
            geom_out = transform(_to_wgs84, rpoly["geometry"])
            polygon_features.append({
                "type": "Feature",
                "geometry": mapping(geom_out),
                "properties": {
                    "road_type": rpoly["road_type"],
                    "width_m": rpoly["width_m"],
                    "area_m2": round(rpoly["geometry"].area, 1),
                    "color": "#AAAAAA",
                }
            })

        # 통계 계산
        road_area_final = sum(p["geometry"].area for p in road_polygons_korea)
        road_ratio_final = road_area_final / area_m2

        hierarchy_stats = {}
        for road in all_lines:
            h = road["hierarchy"]
            hierarchy_stats[h] = hierarchy_stats.get(h, 0) + road["geometry"].length

        stats = {
            "total_length_m": round(sum(r["geometry"].length for r in all_lines), 1),
            "road_area_m2": round(road_area_final, 1),
            "road_ratio": round(road_ratio_final, 3),
            "hierarchy_breakdown": {k: round(v, 1) for k, v in hierarchy_stats.items()},
        }

        return {
            "road_network": {
                "type": "FeatureCollection",
                "features": network_features,
            },
            "road_polygons": {
                "type": "FeatureCollection",
                "features": polygon_features,
            },
            "road_stats": stats,
        }

    # ─────────────────────────────────────────
    # 도로 생성 내부 메서드
    # ─────────────────────────────────────────

    def _generate_arterial_roads(
        self, boundary, width, height, minx, miny, maxx, maxy, design_params
    ) -> list:
        """주간선도로 생성 (20~35m 폭)"""
        roads = []
        # 대상지를 동서로 가로지르는 주간선 (약 1/3 지점, 2/3 지점)
        offsets = [0.33, 0.67] if height > 300 else [0.50]
        for frac in offsets:
            y = miny + height * frac
            line = LineString([(minx, y), (maxx, y)])
            line = line.intersection(boundary)
            if line.is_empty:
                continue
            if hasattr(line, 'geoms'):
                line = max(line.geoms, key=lambda g: g.length)
            roads.append({
                "geometry": line,
                "road_type": "주간선",
                "hierarchy": "대로3류",
                "width_m": 25,
                "color": "#666666",
            })

        # 남북 방향 주간선 (너비가 충분하면)
        offsets_x = [0.33, 0.67] if width > 300 else [0.50]
        for frac in offsets_x:
            x = minx + width * frac
            line = LineString([(x, miny), (x, maxy)])
            line = line.intersection(boundary)
            if line.is_empty:
                continue
            if hasattr(line, 'geoms'):
                line = max(line.geoms, key=lambda g: g.length)
            roads.append({
                "geometry": line,
                "road_type": "주간선",
                "hierarchy": "대로3류",
                "width_m": 25,
                "color": "#666666",
            })

        return roads

    def _generate_collector_roads(
        self, boundary, arterial_lines, width, height, minx, miny, maxx, maxy
    ) -> list:
        """보조간선도로 생성 (12~20m 폭)"""
        roads = []
        # 주간선 사이 200~400m 간격으로 보조간선 배치
        spacing = min(max(200, height / 4), 400)
        y = miny + spacing
        while y < maxy:
            line = LineString([(minx, y), (maxx, y)])
            line = line.intersection(boundary)
            if not line.is_empty:
                if hasattr(line, 'geoms'):
                    line = max(line.geoms, key=lambda g: g.length)
                roads.append({
                    "geometry": line,
                    "road_type": "보조간선",
                    "hierarchy": "중로2류",
                    "width_m": 15,
                    "color": "#888888",
                })
            y += spacing

        spacing_x = min(max(200, width / 4), 400)
        x = minx + spacing_x
        while x < maxx:
            line = LineString([(x, miny), (x, maxy)])
            line = line.intersection(boundary)
            if not line.is_empty:
                if hasattr(line, 'geoms'):
                    line = max(line.geoms, key=lambda g: g.length)
                roads.append({
                    "geometry": line,
                    "road_type": "보조간선",
                    "hierarchy": "중로2류",
                    "width_m": 15,
                    "color": "#888888",
                })
            x += spacing_x

        return roads

    def _generate_local_roads(
        self, boundary, arterial_lines, collector_lines, width, height, minx, miny, maxx, maxy
    ) -> list:
        """국지도로 생성 (6~12m 폭)"""
        roads = []
        # 블록 내부 접근을 위한 국지도로 (100~200m 간격)
        spacing = min(max(80, height / 6), 150)
        y = miny + spacing / 2
        while y < maxy:
            line = LineString([(minx, y), (maxx, y)])
            line = line.intersection(boundary)
            if not line.is_empty:
                if hasattr(line, 'geoms'):
                    for seg in line.geoms:
                        if seg.length > 20:
                            roads.append({
                                "geometry": seg,
                                "road_type": "국지",
                                "hierarchy": "소로2류",
                                "width_m": 8,
                                "color": "#AAAAAA",
                            })
                elif line.length > 20:
                    roads.append({
                        "geometry": line,
                        "road_type": "국지",
                        "hierarchy": "소로2류",
                        "width_m": 8,
                        "color": "#AAAAAA",
                    })
            y += spacing

        spacing_x = min(max(80, width / 6), 150)
        x = minx + spacing_x / 2
        while x < maxx:
            line = LineString([(x, miny), (x, maxy)])
            line = line.intersection(boundary)
            if not line.is_empty:
                if hasattr(line, 'geoms'):
                    for seg in line.geoms:
                        if seg.length > 20:
                            roads.append({
                                "geometry": seg,
                                "road_type": "국지",
                                "hierarchy": "소로2류",
                                "width_m": 8,
                                "color": "#AAAAAA",
                            })
                elif line.length > 20:
                    roads.append({
                        "geometry": line,
                        "road_type": "국지",
                        "hierarchy": "소로2류",
                        "width_m": 8,
                        "color": "#AAAAAA",
                    })
            x += spacing_x

        return roads

    def _add_extra_roads(
        self, boundary, existing_lines, width, height, minx, miny, maxx, maxy
    ) -> list:
        """도로율 미달 시 추가 국지도로 생성"""
        extra = []
        spacing = 60
        y = miny + 30
        count = 0
        while y < maxy and count < 5:
            line = LineString([(minx, y), (maxx, y)])
            line = line.intersection(boundary)
            if not line.is_empty and not hasattr(line, 'geoms') and line.length > 20:
                extra.append({
                    "geometry": line,
                    "road_type": "국지",
                    "hierarchy": "소로3류",
                    "width_m": 6,
                    "color": "#BBBBBB",
                })
                count += 1
            y += spacing
        return extra

    def _lines_to_polygons(self, road_lines: list, boundary: Polygon) -> list:
        """도로 선형에 폭을 적용하여 면 폴리곤 생성"""
        polygons = []
        for road in road_lines:
            half_w = road["width_m"] / 2
            try:
                road_poly = road["geometry"].buffer(half_w, cap_style=2)
                road_poly = road_poly.intersection(boundary)
                if not road_poly.is_empty and road_poly.area > 1:
                    polygons.append({
                        "geometry": road_poly,
                        "road_type": road["road_type"],
                        "width_m": road["width_m"],
                    })
            except Exception:
                pass
        return polygons
